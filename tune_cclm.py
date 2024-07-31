# region preamble
from lightning import LightningDataModule, LightningModule

from cc_models.cc_lm import LoraLightningModule, CodeConditionedLM
from cc_models.cc_olmo import CodeConditionedOLMoForCausalLM
from tune_planner import init_planner_components
from muzero import MuZero
from pretorch_util import assign_visible_gpus; assign_visible_gpus()
from time import time
from typing import Mapping, Any, Optional, Union
from peft import LoraModel, LoraConfig
from args import get_args
from util import get_logger, get_ckpt_callbacks, Namespace, get_relevant_ckpt_path
from cc_models.cc_gpt2 import CodeConditionedGPT2LMHeadModel
from lightning.pytorch.trainer.states import RunningStage
from lightning_fabric.utilities.optimizer import _optimizers_to_device, _optimizer_to_device
from transformers import BitsAndBytesConfig, GPT2LMHeadModel
import inspect
import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import SingleDeviceStrategy
from lightning.pytorch.strategies.strategy import log
from dataset import WikiDataModule, WikiDataset
from cc_models.cc_llama import CodeConditionedLlamaForCausalLM
from socket import gethostname
import os
import warnings
from hf_olmo import OLMoForCausalLM
warnings.filterwarnings("ignore", category=UserWarning, message=".*pydantic*.")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# endregion

def mypath(rel_path):
    if gethostname() == 'theoden':
        # Stored in local folder for speedup
        return os.path.join('/export/home1/NoCsBack/hci/nathan/hub', rel_path)
    else:
        return rel_path

NAME2CLASS = {
    "meta-llama/Llama-2-7b-hf": CodeConditionedLlamaForCausalLM,
    "gpt2": CodeConditionedGPT2LMHeadModel,
    "gpt2-xl": CodeConditionedGPT2LMHeadModel,
    "allenai/OLMo-1B": CodeConditionedOLMoForCausalLM,
}
def get_quantization_config(args, model_name=None):
    if model_name is None:
        model_name = args.base_model_name
    if "llama" in model_name:
        bnb_cfg_dict = {
            'llm_int8_skip_modules': ['lm_head', 'gate', 'code_v_proj'],
            f'load_in_{4 if args.use_4bit else 8}bit': True,
        }
        quantization_config = BitsAndBytesConfig(**bnb_cfg_dict)
    else:
        quantization_config = None
    return quantization_config
def main():
    args = get_args()
    get_tuned_cclm(args)


def filter_wikidm_args(args):
    # Needed when reloading from a checkpoint. If when reloading you changed some args (like passing an lm_ckpt_wid), but still continue training, lightning will complain datamodule hyperparams and model hyperparams don't match
    # Since we don't need some of the args to be hyperparams anyway, we just remove them from the wikidm_args
    args_to_filter = ['lm_ckpt_wid', 'pcodes_dir', 'logger_wid', 'skip_pretraining_validation']
    wikidm_args = {k: v for k, v in vars(args).items() if k not in args_to_filter}
    wikidm_args = Namespace(**wikidm_args)
    return wikidm_args


def load(model, initial_model):
    for k, v in initial_model.state_dict().items():
        assert k in model.state_dict(), f"Key {k} not in model"
        model.state_dict()[k].copy_(v)


def get_tuned_cclm(args, dataset: WikiDataset=None, logger=None, planner=None, initial_model=None, joint_planner_lm=False):
    lora = args.og_params_finetune_type == 'lora'
    use_planner_codes = (planner is not None) and not joint_planner_lm

    skip_fit = (not use_planner_codes and args.skip_lm_train) or (use_planner_codes and args.skip_plm_train)
    skip_val = args.skip_pretraining_validation or (skip_fit and not args.validate_even_if_skip_train)

    if use_planner_codes:
        with torch.no_grad():
            if not skip_fit:
                dataset.add_planner_codes_or_scores(planner, args, logger, splits=['train', 'esval'])
            elif not skip_val:
                dataset.add_planner_codes_or_scores(planner, args, logger, splits=['esval'])

    quantization_config = get_quantization_config(args)
    LMClass: LightningModule = NAME2CLASS[args.base_model_name]
    codebook = None if dataset is None else dataset.codebook

    ckpt_path = get_current_ckpt_path(args, joint_planner_lm, use_planner_codes)

    if lora:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=32,
            lora_dropout=0.01,
        )
        if 'OLMo' in args.base_model_name:
            lora_config.target_modules = ['att_proj']
    else:
        lora_config = None

    if ckpt_path is None:
        loading_info, model = load_model_nockpt(LMClass, args, codebook, quantization_config)
        # Initialize code-conditioning parameters
        model.init_code_conditioning(loading_info['missing_keys'], args.cc_param_seed, codebook, planner if joint_planner_lm else None)
        if lora:
            inner_model = model
            lora_args = {'model': inner_model, 'config': lora_config, 'adapter_name': 'default'}
            model = LoraLightningModule(lora_args)
        if initial_model is not None:
            load(model, initial_model)

        maybe_set_lm_planner(args, dataset, joint_planner_lm, model, planner)
    else:
        model = load_model_from_ckpt_path(ckpt_path, LMClass, args, codebook, dataset, joint_planner_lm, lora,
                                          lora_config, planner, quantization_config)

    if lora:
        model.unfreeze_nonog_params()  # Lora wrapper internally freezes all non-lora parameters, including my added nonog_params


    if (not skip_val) or (not skip_fit):
        if use_planner_codes:
            dataset.use_planner_codes = True
        wikidm_args = filter_wikidm_args(args)
        wiki_dm = WikiDataModule(**wikidm_args, dataset=dataset)
        trainer_args = prep_trainer_args(args, quantization_config, logger, use_planner_codes)
        trainer = pl.Trainer(**trainer_args)

        if not skip_val:
            if not args.skip_pretraining_validation:
                trainer.validate(model, wiki_dm.val_dataloader()) #, ckpt_path=args.lm_ckpt_path)  # Get validation performance before finetuning
            maybe_nocc_separate_validate(args, model, trainer, wiki_dm)

        if not skip_fit:
            if joint_planner_lm:
                assert not (args.jpl_freeze_planner and args.jpl_freeze_lm), "Training jpl but no parameters are trainable, don't set both freeze flags to True"
                if args.jpl_freeze_planner:
                    for param in model.planner.parameters():
                        param.requires_grad = False
                if args.jpl_freeze_lm:
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.planner.parameters():
                        param.requires_grad = True

            trainer.fit(model, wiki_dm) #, ckpt_path=args.lm_ckpt_path)
            # set model equal to best or last model, depending on args
            ckpt_path = get_relevant_ckpt_path(args, trainer)
            model = load_model_from_ckpt_path(ckpt_path, LMClass, args, codebook, dataset, joint_planner_lm, lora, lora_config, planner, quantization_config)

    if not args.cc_type == 'none':
        model.disable_no_cluster() # No cluster only during LMFT phase
    dataset.use_planner_codes = False
    return model


def load_model_from_ckpt_path(ckpt_path, LMClass, args, codebook, dataset, joint_planner_lm, lora, lora_config, planner,
                              quantization_config):
    print(f"Loading model from {ckpt_path}")
    start = time()
    extra_init_args = {"trust_remode_code": True}
    if joint_planner_lm and not args.uniform_mix:
        model_f, model_g, model_h = init_planner_components(args, dataset)
        extra_init_args['planner'] = MuZero(model_h, model_g, model_f, None, policy_only=args.only_policy_head, **args)
    if lora:
        _, inner_model = load_model_nockpt(LMClass, args, codebook, quantization_config)
        lora_args = {'model': inner_model, 'config': lora_config, 'adapter_name': 'default'}
        model = LoraLightningModule.load_from_checkpoint(ckpt_path, lora_args=lora_args, **extra_init_args)
    else:
        loaded_model = LMClass.load_from_checkpoint(ckpt_path,
                                                    **extra_init_args)  # We don't just use loaded model in case we want to eg initialize a target model with settings A with the ckpt from a model with settings B
        _, model = load_model_nockpt(LMClass, args, codebook, quantization_config)
        maybe_set_lm_planner(args, dataset, joint_planner_lm, model, planner)
        load(model, loaded_model)
    print(f"Loaded model in {time() - start} seconds")

    # There's probably a smarter way than explicitly setting it here when loading a ckpt that doesn't have it
    if args.log_nll_per_relative_tkn_idx:
        model.hparams['log_nll_per_relative_tkn_idx'] = args.log_nll_per_relative_tkn_idx
    return model


def maybe_set_lm_planner(args, dataset, joint_planner_lm, model, planner):
    if joint_planner_lm and not args.uniform_mix:
        model.hparams['joint_planner_lm'] = joint_planner_lm
        if not args.uniform_mix:
            if planner is not None:
                model.planner = planner
            else:
                model_f, model_g, model_h = init_planner_components(args, dataset)
                model.planner = MuZero(model_h, model_g, model_f, None, policy_only=args.only_policy_head, **args)


def get_current_ckpt_path(args, joint_planner_lm, use_planner_codes):
    if use_planner_codes:
        ckpt_path = args.plm_ckpt_path
    elif joint_planner_lm:
        ckpt_path = args.jplm_ckpt_path
    else:
        ckpt_path = args.lm_ckpt_path
    return ckpt_path


def maybe_nocc_separate_validate(args, model, trainer, wiki_dm):
    if args.cc_type == 'insert' and not args.plantoken:
        old = model.only_nocc
        model.only_nocc = True
        nocc_dataloader = wiki_dm.split_dataloader('esval-nocc')
        print("Validating on nocc data")
        trainer.validate(model, nocc_dataloader, ckpt_path=args.lm_ckpt_path)
        model.only_nocc = old


def load_model_nockpt(LMClass, args, codebook, quantization_config):
    print("Loading model from online")
    s = time()
    model, loading_info = LMClass.from_pretrained(args.base_model_name,
                                                  quantization_config=quantization_config,
                                                  output_loading_info=True, codebook=codebook,
                                                  device_map='cpu',**args)
    print(f"Loaded model in {time() - s} seconds")
    return loading_info, model


def prep_trainer_args(args, quantization_config=None, logger=None, use_planner_codes=False):
    trainer_args = {k: v for k, v in vars(args).items() if k in inspect.signature(pl.Trainer.__init__).parameters}
    logger = get_logger(args) if logger is None else logger

    if args.joint_planner_lm:
        pref = "jp"
    elif use_planner_codes:
        pref = "p"
    else:
        pref = ""

    checkpoint_callbacks = get_ckpt_callbacks(args, logger, monitor=f'{pref}cclm/val_loss', prefix=f'{pref}lm-')
    trainer_args |= {
        'logger': logger,
        'max_epochs': args.lm_epochs,
        'accelerator': 'gpu',
        'callbacks': checkpoint_callbacks,
        'fast_dev_run': args.fast_dev_run,
        'accumulate_grad_batches': args.accumulate_lm_grad_batches,
    }
    if torch.cuda.device_count() > 1:
        trainer_args["strategy"] = "ddp_find_unused_parameters_true"
    if quantization_config is not None:
        trainer_args['strategy'] = MySingleDeviceStrategy(device='cuda:0')
    return trainer_args


class MySingleDeviceStrategy(SingleDeviceStrategy):

    def model_to_device(self) -> None:
        pass

    def teardown(self) -> None:
        """This method is called to teardown the training process.

        It is the right place to release memory and free other resources.

        """
        _optimizers_to_device(self.optimizers, torch.device("cpu"))
        my_extra_condition = self.lightning_module.trainer.state.stage != RunningStage.VALIDATING # Needed to work with bitsandbytes quantization.
        if self.lightning_module is not None and my_extra_condition:
            log.debug(f"{self.__class__.__name__}: moving model to CPU")
            self.lightning_module.cpu()
        self.precision_plugin.teardown()
        assert self.accelerator is not None
        self.accelerator.teardown()
        self.checkpoint_io.teardown()

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        optimizer_states = checkpoint["optimizer_states"]
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(opt_state)
            _optimizer_to_device(optimizer, self.root_device)

    # def setup_environment(self) -> None:
    #     pass
if __name__ == '__main__':
    main()