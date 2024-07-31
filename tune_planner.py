import inspect
from time import time

import torch
from lightning import pytorch as pl

from constants import SF_CACHE_DIR
from dynamics_functions import SentenceBasedModelG, ModelG

from muzero import MuZero, is_embedding_space_prediction
from prediction_functions import SentenceBasedModelF, ModelF
from representation_functions import SentenceBasedModelH, ModelH
from util import get_logger, get_ckpt_callbacks, get_relevant_ckpt_path
from dataset import WikiDataModule


def get_oracle_pretrained_planner(args, dataset, language_model, logger=None):
    model_f, model_g, model_h = init_planner_components(args, dataset)

    if args.mz_ckpt_path is None:
        planner = MuZero(model_h, model_g, model_f, language_model, policy_only=args.only_policy_head, **args)
    else:
        planner = get_planner_from_ckpt_path(args.mz_ckpt_path, args, language_model, model_f, model_g, model_h)

    if args.soft_planner_targets:
        planner.dataset = dataset

    if args.skip_mz_train:
        planner = planner.to('cuda')
    else:
        # iterable_dataset = IterableWrapper(dataset)
        # wiki_dm = WikiDataModule(**args, dataset=iterable_dataset, phase='mz')
        wiki_dm = WikiDataModule(**args, dataset=dataset, phase='mz')
        logger = logger if logger is not None else get_logger(args)
        trainer_args = {k: v for k, v in vars(args).items() if k in inspect.signature(pl.Trainer.__init__).parameters}
        trainer_args |= {
            "callbacks": get_ckpt_callbacks(args, logger, monitor="mzpt/val_loss", prefix="mz-"),
            "logger": logger,
            "max_epochs": args.mz_epochs,
            "fast_dev_run" : args.fast_dev_run,
            "accumulate_grad_batches": args.accumulate_mz_grad_batches,
        }
        if torch.cuda.device_count() > 1:
            trainer_args["strategy"] = "ddp_find_unused_parameters_true"
        trainer = pl.Trainer(**trainer_args)
        if not args.skip_pretraining_validation:
            trainer.validate(planner, wiki_dm.val_dataloader(), ckpt_path=args.mz_ckpt_path)  # Get validation performance before training
        trainer.fit(planner, wiki_dm, ckpt_path=args.mz_ckpt_path)

        # set model equal to best or last model, depending on args
        ckpt_path = get_relevant_ckpt_path(args, trainer)
        planner = get_planner_from_ckpt_path(ckpt_path, args, language_model, model_f, model_g, model_h)
    return planner.to('cuda')


def get_planner_from_ckpt_path(ckpt_path, args, language_model, model_f, model_g, model_h):
    print(f"Loading planner model from {ckpt_path}")
    s = time()
    planner = MuZero.load_from_checkpoint(ckpt_path, model_h=model_h, model_g=model_g,
                                          model_f=model_f, language_model=language_model,
                                          policy_only=args.only_policy_head, strict=False,
                                          **args)  # strict False because model.lm should not be loaded from state dict
    print(f"Loaded planner model in {time() - s:.2f}s")
    return planner


def init_planner_components(args, dataset):
    # Define the input and output sizes and the learning rate
    # input_size, output_size = [args.mz_hidden_size]*2
    embedding_size = args.mz_hidden_size
    # Initialize the models
    if args.only_policy_head and args.mz_max_timesteps == 1:
        model_g = None
    else:
        no_reward_head = args.no_reward_head or args.only_policy_head
        if args.mz_multi_vector_representation:
            model_g = SentenceBasedModelG(embedding_size, embedding_size, num_actions=dataset.codebook.shape[0],
                                          num_layers=args.mz_g_num_layers, dropout_p=args.mz_g_dropout_p,
                                          no_reward_head=no_reward_head)
        else:
            model_g = ModelG(embedding_size, embedding_size, num_actions=dataset.codebook.shape[0],
                             num_layers=args.mz_g_num_layers, dropout_p=args.mz_g_dropout_p,
                             skip_and_norm=args.mz_g_skip_and_norm, no_reward_head=no_reward_head)
        if args.init_action_embedder_with_codebook:
            model_g.action_embedder.weight.data = torch.tensor(dataset.codebook).to(
                model_g.action_embedder.weight.device)
    if args.mz_multi_vector_representation:
        h_codebook = torch.tensor(dataset.codebook).to('cuda') if args.mz_h_sent_model_cluster else None
        model_h = SentenceBasedModelH(model_name=args.mz_h_sent_model, output_size=embedding_size,
                                      cache_dir=SF_CACHE_DIR, codebook=h_codebook, tokenizer_name=args.base_model_name)
        model_f = SentenceBasedModelF(embedding_size, hidden_size=args.mz_f_hidden_size,
                                      action_space_size=len(dataset.codebook), num_layers=args.mz_f_num_layers,
                                      dropout_p=args.mz_f_dropout_p, only_policy_head=args.only_policy_head)
    else:
        h_model_name = args.mz_h_model_name if args.mz_h_model_name is not None else args.base_model_name
        if args.base_model_name != 'gpt2':
            raise NotImplementedError("TODO")
        model_h = ModelH(model_name=h_model_name, output_size=embedding_size, freeze=args.freeze_mz_h,
                         cache_dir=SF_CACHE_DIR, extra_transformer_layers=args.mz_h_extra_transformer_layers)
        model_f = ModelF(embedding_size, hidden_size=args.mz_f_hidden_size, action_space_size=len(dataset.codebook),
                         num_layers=args.mz_f_num_layers,
                         dropout_p=args.mz_f_dropout_p, skip_and_norm=args.mz_f_skip_and_norm,
                         only_policy_head=args.only_policy_head,
                         regression_policy_head=is_embedding_space_prediction(args.mz_policy_loss_fn),
                         action_embedding_size=dataset.codebook.shape[1])
    if args.init_policy_head_with_codebook and not is_embedding_space_prediction(args.mz_policy_loss_fn):
        model_f.policy_head.weight.data = torch.tensor(dataset.codebook).to(model_f.policy_head.weight.device)
    return model_f, model_g, model_h
