import re
from typing import Dict, Any
import torch
from peft import LoraModel
from torch import nn
from time import time
from tqdm import tqdm as std_tqdm
from functools import partial

tqdm = partial(std_tqdm, dynamic_ncols=True)

from torch.nn import functional as F, CrossEntropyLoss
from transformers import AutoModelForCausalLM

from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from util import get_codebook, tn


def _code_condition(module, unconditioned, codes, sent_embs=None, planner_scores=None):
    # codes = codes.to(self.codebook.device) # Work on CPU b/c unclustered codebook is too large to fit on GPU
    device = unconditioned.device
    code_v = get_code_repr(module, codes, sent_embs, device, unconditioned, planner_scores)
    return unconditioned + (module.gate if not module.no_gate else 1) * code_v


def sample_gumbel(shape, device, scale=1.0, eps=1e-20):
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps) * scale


def gumbel_softmax_sample(logits, temperature, scale=1.0):
    y = logits + sample_gumbel(logits.size(), logits.device, scale)
    return F.softmax(y / temperature, dim=-1)


def get_code_embs(module, planner_scores):
    temperature = module.hparams.jpl_temperature

    if module.hparams.jpl_gumbel and module.training:
        planner_probs = gumbel_softmax_sample(planner_scores, temperature, scale=module.hparams.gumbel_scale)
    else:
        planner_probs = F.softmax(planner_scores / temperature, dim=-1)

    if not module.hparams.straight_through:
        return torch.matmul(planner_probs, module.codebook)
    else:
        shape = planner_probs.size()
        _, ind = planner_probs.max(dim=-1)
        planner_probs_hard = torch.zeros_like(planner_probs).view(-1, shape[-1])
        planner_probs_hard.scatter_(1, ind.view(-1, 1), 1)
        planner_probs_hard = planner_probs_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        planner_probs_hard = (planner_probs_hard - planner_probs).detach() + planner_probs
        return torch.matmul(planner_probs_hard, module.codebook)
    # planner_probs = F.softmax(planner_scores, dim=-1)
    # code_embs = torch.matmul(planner_probs, module.codebook)
    # return code_embs



def get_code_repr(module, codes, sent_embs, device, unconditioned, planner_scores=None):
    if module.no_cluster:
        code_embs = sent_embs.to(device)
    else:
        if planner_scores is None:
            assert codes is not None
            code_embs = module.codebook[codes]
        else:
            code_embs = get_code_embs(module, planner_scores)
    insert_no_proj = hasattr(module, 'hparams') and module.hparams.insert_no_proj # hasattr(self, 'hparams') for backwards compatibility
    if not insert_no_proj:
        adapter_extra_context = hasattr(module, 'hparams') and module.hparams.adapter_extra_context is not None # hasattr(self, 'hparams') for backwards compatibility
        # code_k = self.code_k_proj(code).view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        if adapter_extra_context:
            ctx_type = module.hparams.adapter_extra_context

            # Average unconditioned embeddings, ensuring causal masking
            cumulative_sum = torch.cumsum(unconditioned, dim=1)

            divisor = torch.arange(1, unconditioned.size(1) + 1, device=unconditioned.device).view(1, -1, 1).expand_as(cumulative_sum)
            unconditioned_pooled = cumulative_sum / divisor

            if ctx_type == "concat":
                code_v = module.code_v_proj(torch.cat([code_embs, unconditioned_pooled], dim=-1))
            elif ctx_type == "gate":
                code_v = module.code_v_proj(code_embs)
                code_v = code_v * module.ctx_gate(torch.cat([code_embs, unconditioned_pooled], dim=-1))
        else:
            code_v = module.code_v_proj(code_embs)
        # code_attn_weights = torch.matmul(query_states, code_k.transpose(2, 3)) / math.sqrt(self.head_dim)
        # code_attn_weights = self.gate * nn.functional.softmax(code_attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # code_attn_output = torch.matmul(code_attn_weights, code_v)
        return code_v
    else:
        return code_embs


def register_codebook(module, codebook, config, unfreeze_codebook_after_step, name='codebook'):
    codebook = module.maybe_quantize(codebook, config)
    if 'randinit_lm_code_embedder' in module.hparams and module.hparams.randinit_lm_code_embedder:
        codebook = torch.rand_like(codebook)
    if unfreeze_codebook_after_step < 0:
        module.register_buffer(name, codebook)
    else:
        module.__setattr__(name, nn.Parameter(codebook.clone()))
        module.register_buffer(f'og_{name}', codebook.clone())


class CodeConditionedLMAttention(nn.Module):

    def code_condition(self, *args, **kwargs):
        return _code_condition(self, *args, **kwargs)

    def prep_code_conditioning_params(self, config, codebook, no_cluster=False, no_gate=False, unfreeze_codebook_after_step=-1, hparams=None, **_):
        code_dim = codebook.shape[1]
        try:
            hidden_size = config.hidden_size
        except AttributeError:
            hidden_size = config.d_model
        if hparams.adapter_extra_context == "concat":

            self.code_v_proj = nn.Sequential(nn.Linear(code_dim + hidden_size, hidden_size // 8),
                                             nn.GELU(), 
                                             nn.Linear(hidden_size // 8, hidden_size))
        else:
            self.code_v_proj = nn.Linear(code_dim, hidden_size)

        if hparams.adapter_extra_context == "gate":
            self.ctx_gate = nn.Sequential(nn.Linear(code_dim + hidden_size, hidden_size // 8),
                                      nn.GELU(), 
                                      nn.Linear(hidden_size // 8, 1),
                                      nn.Sigmoid())
        self.no_cluster = no_cluster
        self.no_gate = no_gate
        if not no_gate:
                self.gate = nn.Parameter(torch.zeros(1,hidden_size))
        self.hparams = hparams
        register_codebook(self, codebook, config, unfreeze_codebook_after_step)


    def maybe_quantize(self, codebook, config):
        raise NotImplementedError('Implement in subclass')


class CodeConditionedLMBaseModel(nn.Module):

    def init_cc_layer(self, _, config, codebook, no_gate, ucas, LayerModule, AttentionModule, attn_key):
        new_layer = LayerModule(config)
        new_layer.__setattr__(attn_key,
                              AttentionModule(config, codebook=codebook, no_cluster=self.hparams.no_cluster, no_gate=no_gate,
                                              unfreeze_codebook_after_step=ucas,
                                              hparams=self.hparams))  # works for gpt-2 and Llama
        return new_layer

    def prep_code_conditioning_params(self, config, codebook, no_cluster=False, no_gate=False, cc_type='adapter', unfreeze_codebook_after_step=-1, hparams=None, **_):
        self.cc_type = cc_type
        self.hparams = hparams
        if cc_type in ['insert', 'add_to_embedding']:
            code_dim = codebook.shape[1]
            register_codebook(self, codebook, config, unfreeze_codebook_after_step)
            if not hparams.insert_no_proj:
                self.code_v_proj = nn.Linear(code_dim, config.hidden_size)
            if cc_type == 'add_to_embedding':
                self.no_cluster = no_cluster
                self.no_gate = no_gate
                self.gate = nn.Parameter(torch.zeros(1,config.hidden_size))

    def maybe_quantize(self, codebook, config):
        raise NotImplementedError('Implement in subclass')

    def maybe_addition_style_code_condition_embedding(self, inputs_embeds, codes, sent_embs, code_mask=None):
        if self.cc_type == 'add_to_embedding' and codes is not None:
            inputs_embeds = _code_condition(self, inputs_embeds, codes, sent_embs)
        return inputs_embeds

    def get_embeds(self, code_mask, input_and_code_ids):
        msk = code_mask[:,-input_and_code_ids.shape[1]:] # In case input_and_code_ids is shorter than code_mask
        for_input_emb = torch.where(msk, torch.zeros_like(input_and_code_ids),
                                    input_and_code_ids)
        for_code_emb = torch.where(msk, input_and_code_ids,
                                   torch.zeros_like(input_and_code_ids))
        input_embs = self.wte(for_input_emb)
        code_embs = self.codebook[for_code_emb]
        if self.cc_type == 'insert' and not self.hparams.insert_no_proj:
            code_embs = self.code_v_proj(code_embs)
        input_and_code_embs = torch.where(msk.unsqueeze(-1), code_embs, input_embs)
        return input_and_code_embs

    def get_maybe_cc_embeds(self, code_mask, input_ids, inputs_embeds):
        if inputs_embeds is None:
            # plantoken = 'plantoken' in self.hparams and self.hparams.plantoken
            # if plantoken: # NOT SURE WHY LET THIS DEPEND ON PLANTOKEN??
            if code_mask is not None:
                inputs_embeds = self.get_embeds(code_mask, input_ids)
            else:
                inputs_embeds = self.wte(input_ids)
        return inputs_embeds

    @property
    def layers_module(self):
        raise NotImplementedError('Implement in subclass')

    def get_layermodule_kwargs(self, i, config):
        raise NotImplementedError('Implement in subclass')


class CodeConditionedLM(LightningModule):
    '''
    Base class for code-conditioned language models
    '''
    def _common_init(self, *args, **kwargs):
        self.save_hyperparameters(ignore=['dataset','max_articles','planner'])
        if 'cc_type' not in self.hparams:
            self.hparams.cc_type = 'adapter' # Backwards compatibility
        elif self.hparams['cc_type'] == 'extra_token':
            self.hparams.cc_type = 'insert' # Backwards compatibility
        self.only_nocc = (self.hparams.cc_type == 'none')
        if 'planner' in kwargs and kwargs['planner'] is not None:
            self.planner = kwargs['planner']

    def wte(self, *args, **kwargs):
        return self.base_model.wte(*args, **kwargs)

    def get_embeds(self, *args, **kwargs):
        return self.base_model.get_embeds(*args, **kwargs)

    def prep_config_for_base_model(self, config):
        return config

    def prep_code_conditioning_params(self, AttentionModule, BaseModel, LayerModule, attn_key, config,
                                      data_pkl_path, num_layers_to_condition=None, no_cluster=False, unclustered_pkl_path=None, codebook=None, **_):
        if self.hparams.cc_type != 'none':
            self.no_cluster = no_cluster
            if codebook is not None:
                codebook = torch.tensor(codebook)
            else:
                print('Getting codebook as tensor'); s = time()
                codebook = torch.tensor(get_codebook(data_pkl_path if not no_cluster else unclustered_pkl_path))
                print(f'Got codebook in {time() - s} seconds')
            base_model_key = self.base_model_prefix

            # Backwards compatibility
            for key, default_value in (('no_gate', False),
                                       ('unfreeze_codebook_after_step', -1),
                                       ('lm_as_planner', False),
                                       ('cc_type', 'adapter'),
                                       ('adapter_extra_context', None),
                                       ('og_params_finetune_type', 'freeze')):
                self.hparams[key] = self.hparams[key] if key in self.hparams else default_value


            no_gate = self.hparams.no_gate

            # Code-conditioning in base_model
            ucas = self.hparams.unfreeze_codebook_after_step
            if ucas is None:
                ucas = -1 # Backwards compatibility
            base_config = self.prep_config_for_base_model(config) # Needed for Olmo to not initialize things on meta device when loading run from ckpt
            self.__setattr__(base_model_key, BaseModel(base_config, codebook=codebook, no_cluster=self.no_cluster, no_gate=no_gate, cc_type=self.hparams.cc_type, unfreeze_codebook_after_step=ucas, hparams=self.hparams))
            if self.hparams.lm_as_planner and self.hparams.cc_type != 'insert':
                register_codebook(self, codebook, base_config, ucas, name='output_codebook') # else: we use base_model.codebook also output_codebook
                raise NotImplementedError("TODO still need to see how to do teacher forcing when # input elements is smaller than # output elements")
            # Code conditioning in layers
            if self.hparams.cc_type == 'adapter':
                num_existing_layers = len(self.base_model.layers_module)
                if num_layers_to_condition is None:
                    num_layers_to_condition = num_existing_layers - 2  # Taking default from https://arxiv.org/pdf/2303.16199.pdf
                num_normal_layers = num_existing_layers - num_layers_to_condition
                for i in list(range(num_existing_layers))[num_normal_layers:]:
                    # layermodule_kwargs = self.base_model.get_layermodule_kwargs(i, config)
                    # new_layer = LayerModule(**layermodule_kwargs)
                    # new_layer.__setattr__(attn_key, AttentionModule(config, codebook=codebook, no_cluster=self.no_cluster, no_gate=no_gate, unfreeze_codebook_after_step=ucas, hparams=self.hparams)) # works for gpt-2 and Llama
                    new_layer = self.base_model.init_cc_layer(i, base_config, codebook, no_gate, ucas, LayerModule, AttentionModule, attn_key)
                    self.base_model.layers_module[i] = new_layer

        match self.hparams.og_params_finetune_type:
            case 'freeze' | 'lora':
                self.freeze_og_params()
            case 'full_finetune':
                pass
            case _:
                raise ValueError(f'Unexpected value for og_params_finetune_type: {self.hparams.og_params_finetune_type}')


    def freeze_og_params(self):
        for name, param in self.named_parameters():
            if self.is_original_key(name):
                param.requires_grad = False

    def unfreeze_nonog_params(self):
        for name, param in self.named_parameters():
            if not self.is_original_key(name):
                param.requires_grad = True


    def is_original_key(self, name):
        return not any([el in name for el in ['code_k_proj', 'code_v_proj', 'gate', 'codebook', 'lora_', 'planner']])


    def _split_step(self, batch, batch_idx, split):
        planner_based = any([k in batch for k in ['planner_codes', 'input_ids_and_planner_codes']])
        jpl = hasattr(self.hparams, "joint_planner_lm") and self.hparams.joint_planner_lm
        forward_args = {}
        if not self.only_nocc and (self.base_model.cc_type == 'insert'):
            codes_between_tokens = True
            code_mask = batch['code_mask']
            input_ids_and_codes = batch[f'input_ids_and_{"planner_" if planner_based else ""}codes']
            input_and_code_embs = self.get_embeds(code_mask, input_ids_and_codes)
            forward_args |= {'inputs_embeds': input_and_code_embs, 'code_mask': code_mask}

            nonlabel_mask = torch.logical_or(code_mask, ~batch['label_mask']) if not self.hparams.plantoken else ~batch['label_mask']
            labels = torch.masked_fill(input_ids_and_codes, nonlabel_mask, -100)
        else:
            input_ids = batch['input_ids']
            forward_args['input_ids']= input_ids
            labels = torch.masked_fill(input_ids, ~batch['label_mask'], -100)
            codes_between_tokens = False
            if not self.only_nocc:
                codes = batch[('codes' if not planner_based else 'planner_codes')]
                forward_args['codes'] = codes
                if self.no_cluster:
                    forward_args |= {'sent_embs': batch['sent_embs']}
        forward_args |= {
            'labels': labels,
            'attention_mask': batch['attention_mask']
        }
        if jpl:
            assert self.base_model.cc_type != 'insert', "joint_planner_lm not implemented yet with insert cc-type"
            forward_args['planner_scores'] = self.get_planner_scores_for_train_batch(batch)
        if not self.only_nocc or self.hparams.cc_type == 'none':
            # outputs = self(input_ids, codes, sent_embs, labels=labels, attention_mask=batch['attention_mask'], inputs_embeds=input_and_code_embs, code_mask=code_mask)
            outputs = self(**forward_args)
            loss = outputs.loss

            prefix = CodeConditionedLM.get_cclm_prefix(jpl, planner_based)
            if split != 'train' and self.hparams.log_nll_per_relative_tkn_idx:
                self.log_nll_per_relative_tkn_idx(outputs.logits, labels, cclm_prefix=prefix)
            self.log(f'{prefix}cclm/{split}_loss', loss, prog_bar=True)

        # if split=='train':
            # mean_gate_size = torch.mean(torch.tensor([l.attn_module.gate.abs().mean() for l in self.base_model.layers_module if ("CodeConditioned" in l._get_name()) and hasattr(l.attn_module,'gate')]))
            # self.log(f'{prefix}cclm/mean_gate_size', mean_gate_size)
        if split != 'train':
            if not self.hparams.cc_type == 'none' and ((not codes_between_tokens) or self.only_nocc):    # In this case, we cannot use the same chunks to calculate cc-loss as equivalent no-cc loss. Instead, we do it in a separate loop in which only_nocc is set to True
                nocc_outputs = self(input_ids, None, labels=labels, attention_mask=batch['attention_mask'])
                nocc_loss = nocc_outputs.loss
                self.log(f'cclm/{split}_nocc_loss', nocc_loss, prog_bar=True)
                if self.only_nocc:
                    loss = nocc_loss

                if split != 'train' and self.hparams.log_nll_per_relative_tkn_idx:
                    self.log_nll_per_relative_tkn_idx(nocc_outputs.logits, labels, 'nocc')
        return loss

    @staticmethod
    def get_cclm_prefix(jpl, planner_based):
        if jpl:
            return "jp"
        elif planner_based:
            return "p"
        else:
            return ""

    def get_planner_scores_for_train_batch(self, batch):
        snt_idxs = batch['snt_idxs']
        B = snt_idxs.shape[0]
        # The first sentence with tokens in this chunk could have started much earlier than in the previous chunk even. We've loaded its context separately
        first_snt_ctxs, first_snt_attn_masks = batch['first_sent_ctx_input_ids'], batch['first_sent_ctx_attention_mask']

        # For subsequent sentences, we can get the context from a combination of the current and previous chunk
        prev_and_input_ids = torch.cat([batch['prev_input_ids'], batch['input_ids']],-1)
        prev_and_input_mask = torch.cat([batch['prev_attention_mask'], batch['attention_mask']],-1)
        N = batch['input_ids'].shape[-1]
        nonfirst_snt_starts = [torch.nonzero(r[1:] != r[:-1], as_tuple=True)[0] + 1 for r in snt_idxs]
        nonfirst_snt_ctxs, nonfirst_snt_attn_masks = [[[el[row,:N+col+1][-N:] for col in s] for row, s in enumerate(nonfirst_snt_starts)]
                                              for el in [prev_and_input_ids,prev_and_input_mask]]

        # Combine the first and non-first sentence contexts (and masks)
        snt_ctxs = [[first] + nonfirst for first, nonfirst in zip(first_snt_ctxs, nonfirst_snt_ctxs)]
        snt_attn_masks = [[first] + nonfirst for first, nonfirst in zip(first_snt_attn_masks, nonfirst_snt_attn_masks)]

        # Flatten so planner can calculate in parallel, then reshape back to being per-row
        flat_snt_ctx, flat_snt_attn_mask = [torch.stack([e for row in el for e in row])
                                            for el in [snt_ctxs, snt_attn_masks]]
        flat2nested = [(row,col) for row in range(len(snt_ctxs)) for col in range(len(snt_ctxs[row]))]
        nested2flat = [[flat2nested.index((row,col)) for col in range(len(snt_ctxs[row]))] for row in range(B)]
        if self.hparams.uniform_mix:
            planner_scores = torch.zeros(flat_snt_ctx.shape[0], self.hparams.cluster_count).to(flat_snt_ctx.device)
        else:
            planner_scores = self.planner(flat_snt_ctx, flat_snt_attn_mask, num_timesteps=self.hparams.mz_max_timesteps)
        pscores_per_row = [planner_scores[r[0]:r[-1]+1] for r in nested2flat]

        # Repeat planner scores for each token in the sentence
        rel_snt_idxs_per_row = [r - r.min() for r in snt_idxs]
        assert(all([r.max() == pscores.shape[0] - 1 for r, pscores in zip(rel_snt_idxs_per_row, pscores_per_row)]))
        expanded_planner_scores = torch.stack([torch.index_select(pscores_per_row[row],0, rel_snt_idxs_per_row[row]) for row in range(B)])
        return expanded_planner_scores

    def log_nll_per_relative_tkn_idx(self, logits, labels, prefix='', cclm_prefix = ''):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        nll_per_token = CrossEntropyLoss(reduction='none')(shift_logits.view(-1, shift_logits.shape[-1]),
                                                           shift_labels.view(-1)).view(shift_labels.shape)
        nll_mask = (shift_labels != -100).float()

        sum_legit_nlls = torch.sum(nll_per_token * nll_mask, dim=0)
        count_legit_nlls = torch.sum(nll_mask, dim=0)

        NUM_TOKENS_TO_LOG = count_legit_nlls.shape[0]
        for i in range(NUM_TOKENS_TO_LOG):
            if (N:=count_legit_nlls[i]) > 0:
                self.log(f'per_tkn_{cclm_prefix}cclm/{prefix}{i}_nll', sum_legit_nlls[i] / N)
    @property
    def base_model(self):
        return self.__getattr__(self.base_model_prefix)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self._configure_optimizers(self)

    @staticmethod # Static: to avoid code-duplication for LoraLightningModule wrapper
    def _configure_optimizers(self) -> OptimizerLRScheduler:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, *args, **kwargs):
        return self._training_step(self, *args, **kwargs)

    @staticmethod # Static: to avoid code-duplication for LoraLightningModule wrapper
    def _training_step(self, batch, batch_idx):
        self.maybe_unfreeze_codebook(batch, batch_idx)
        self.maybe_unfreeze_planner(batch_idx)
        return self._split_step(batch, batch_idx,'train')


    def maybe_unfreeze_codebook(self, batch, batch_idx):
        if self.hparams.unfreeze_codebook_after_step > 0:
            for l in self.base_model.h:
                if hasattr(l.attn, 'codebook'):
                    self.log(f'{"p" if "planner_codes" in batch else ""}cclm/codebook_change',
                             (l.attn.codebook - l.attn.og_codebook).abs().mean())
                    if batch_idx == self.hparams.unfreeze_codebook_after_step:  # This will trigger every epoch, but that's fine because in subsequent epochs it just sets requires_grad from True to True
                        l.attn.codebook.requires_grad = True
            if hasattr(self.base_model, 'codebook'):
                self.log(f'{"p" if "planner_codes" in batch else ""}cclm/codebook_change',
                         (self.base_model.codebook - self.base_model.og_codebook).abs().mean())
                if batch_idx == self.hparams.unfreeze_codebook_after_step:
                    self.base_model.codebook.requires_grad = True

    def get_unfreeze_step(self):
        s = self.hparams.jpl_unfreeze_planner_after_step_or_frac
        TOTAL_N_STEPS = self.trainer.estimated_stepping_batches
        # Check if the string is an integer
        try:
            int_value = int(s)
            return int_value
        except ValueError:
            pass

        # Check if the string is a float
        try:
            float_value = float(s)
            if 0 < float_value < 1:
                return int(float_value * TOTAL_N_STEPS)
            else:
                raise ValueError("Float value must be between 0 and 1 non-inclusive.")
        except ValueError:
            pass

        # If neither, raise an error
        raise ValueError(f"String '{s}' is neither an integer nor a valid float between 0 and 1.")

    def maybe_unfreeze_planner(self, batch_idx):
        unfreeze_step = self.get_unfreeze_step()
        if hasattr(self, 'planner') and unfreeze_step > 0:
            if batch_idx == unfreeze_step:
                for name, param in self.planner.named_parameters():
                    if not (self.hparams.jpl_freeze_planner_sentence_transformer and self.planner.is_sentence_transformer_param(name)):
                        param.requires_grad = True
    def on_train_start(self):
        self.maybe_freeze_codebook()
        self.maybe_freeze_planner()

    def maybe_freeze_codebook(self):
        if self.hparams.unfreeze_codebook_after_step > 0:
            for l in self.base_model.h:
                if hasattr(l.attn, 'codebook'):
                    l.attn.codebook.requires_grad = False
            if hasattr(self.base_model, 'codebook'):
                self.base_model.codebook.requires_grad = False


    def maybe_freeze_planner(self):
        if hasattr(self, 'planner'):
            if self.hparams.jpl_freeze_planner_sentence_transformer:
                for name, param in self.planner.named_parameters():
                    if self.planner.is_sentence_transformer_param(name):
                        param.requires_grad = False
            if self.get_unfreeze_step() != 0:
                for param in self.planner.parameters():
                    param.requires_grad = False


    def validation_step(self, *args, **kwargs):
        return self._validation_step(self, *args, **kwargs)
    @staticmethod # Static: to avoid code-duplication for LoraLightningModule wrapper
    def _validation_step(self, batch, batch_idx):
        return self._split_step(batch, batch_idx,'val')

    def test_step(self, *args, **kwargs):
        return self._test_step(self, *args, **kwargs)

    @staticmethod # Static: to avoid code-duplication for LoraLightningModule wrapper
    def _test_step(self, batch, batch_idx):
        return self._split_step(batch, batch_idx,'test')

    def init_code_conditioning(self, keys_to_init, seed, codebook, planner=None):
        for key in keys_to_init:
            key = self._maybe_fix_key(key)
            if 'code_v_proj' in key:
                if key.endswith('weight'):
                    # set seed
                    torch.manual_seed(seed)
                    nn.init.xavier_normal_(self.state_dict()[key]) # Following https://ai.stackexchange.com/questions/30491/is-there-a-proper-initialization-technique-for-the-weight-matrices-in-multi-head
                elif key.endswith('bias'):
                    nn.init.zeros_(self.state_dict()[key])
            elif bool(re.search(r"(attn|transformer\.blocks\.\d+)\.gate", key)):
                # if not self.hparams.no_zero_init_gate:
                #     nn.init.zeros_(self.state_dict()[key]) # Following https://github.com/OpenGVLab/LLaMA-Adapter/blob/4c68b3407f52563b403d833f25daa77689be10ab/llama/model.py#L112
                # else:
                #     nn.init.ones_(self.state_dict()[key])
                if self.hparams.init_gate == 'zero':
                    nn.init.zeros_(self.state_dict()[key]) # Following https://github.com/OpenGVLab/LLaMA-Adapter/blob/4c68b3407f52563b403d833f25daa77689be10ab/llama/model.py#L112
                elif self.hparams.init_gate == 'one':
                    nn.init.ones_(self.state_dict()[key])
                elif self.hparams.init_gate == 'xavier_normal':
                    torch.manual_seed(seed)
                    nn.init.xavier_normal_(self.state_dict()[key])
                else:
                    raise NotImplementedError(f'Unknown init_gate: {self.hparams.init_gate}')


            # elif any(s in key for s in ['attn.codebook',    'transformer.codebook',    'model.codebook',
            #                             'attn.og_codebook', 'transformer.og_codebook', 'model.og_codebook']):
            elif 'codebook' in key: # just hoping nothing existing has 'codebook' in its name 🤞
                if 'randinit_lm_code_embedder' in self.hparams and self.hparams.randinit_lm_code_embedder:
                    codebook = torch.rand(*codebook.shape)
                self.state_dict()[key].data.copy_(torch.tensor(codebook)) # just doing self.state_dict()[key] = torch.tensor(codebook) doesn't work because not in-place
                assert torch.equal(self.state_dict()[key], torch.tensor(codebook))
            elif 'ff_out' in key:
                assert self.config.weight_tying
                # If weight tying, ff_out isn't used anyway (but is still created in de existing olmo code for some reason)
                continue
            else:
                raise ValueError(f'Unexpected key: {key}')

        if planner is not None:
            self.planner = planner

        # if self.hparams.plantoken:
        #     self.add_codebook_to_embeddings()

    # def add_codebook_to_embeddings(self):
    #     # add n_codes elements to self.lm_head.weight
    #     extended_head = torch.cat([self.lm_head.weight.data, self.base_model.codebook],
    #                               dim=0)  # intialized with codebook here, but could also be randomly initialized
    #     self.lm_head = nn.Linear(self.lm_head.weight.shape[1],
    #                              self.lm_head.weight.shape[0] + self.base_model.codebook.shape[0], bias=False)
    #     self.lm_head.weight.data = extended_head
    #     self.base_model.wte = nn.Embedding(self.base_model.wte.num_embeddings + self.base_model.codebook.shape[0],
    #                                        self.base_model.wte.embedding_dim)
    #     self.base_model.wte.weight = self.lm_head.weight  # Weight tying

    def disable_no_cluster(self):
        if hasattr(self.base_model, 'layers_module'):
            for m in self.base_model.layers_module:
                if hasattr(m, 'attn_module') and m.attn_module.no_cluster:
                    m.attn_module.no_cluster = False

    def _maybe_fix_key(self, key):
        '''
        For Llama 2 and GPT2 whether the base_model_prefix is included differs
        '''
        if key not in self.state_dict():
            key = self.base_model_prefix + '.' + key
            if key not in self.state_dict():
                raise ValueError(f'Unexpected key: {key}')
        return key

    def on_save_checkpoint(self, *args, **kwargs):
        return self._on_save_checkpoint(self, *args, **kwargs)

    @staticmethod # Static: to avoid code-duplication for LoraLightningModule wrapper
    def _on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        match self.hparams.og_params_finetune_type:
            case 'freeze' | 'lora':
                '''
                Because LLMs can be quite large, and we are only finetuning added parameters in this case, we only save the added parameters
                '''
                for key in list(checkpoint['state_dict'].keys()):
                    if self.is_original_key(key):
                        del checkpoint['state_dict'][key]
            case 'full_finetune':
                pass
            case _:
                raise ValueError(f'Unexpected value for og_params_finetune_type: {self.hparams.og_params_finetune_type}')

    def on_load_checkpoint(self, *args, **kwargs):
        return self._on_load_checkpoint(self, *args, **kwargs)

    @staticmethod # Static: to avoid code-duplication for LoraLightningModule wrapper
    def _on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if 'og_params_finetune_type' not in self.hparams:
            self.hparams.og_params_finetune_type = 'freeze'
        if 'plantoken' not in self.hparams:
            self.hparams.plantoken = False
        match self.hparams.og_params_finetune_type:
            case 'freeze' | 'lora':
                base_model = AutoModelForCausalLM.from_pretrained(checkpoint['hyper_parameters']['base_model_name'],trust_remote_code=True)
                lora = self.hparams.og_params_finetune_type == 'lora'
                checkpoint['state_dict'].update(base_model.state_dict() if not lora else {'model.' + k: v for k, v in base_model.state_dict().items()})
                if lora:
                    for k, _ in self.named_parameters():
                        if k not in checkpoint['state_dict'] and 'base_layer' in k:
                            k_wo_baselayer = k.replace('base_layer.','')
                            assert k_wo_baselayer in checkpoint['state_dict']
                            checkpoint['state_dict'][k] = checkpoint['state_dict'].pop(k_wo_baselayer)

            case 'full_finetune':
                pass
            case _:
                raise ValueError(f'Unexpected value for og_params_finetune_type: {self.hparams.og_params_finetune_type}')

         # backwards compatibility: renamed 'extra_token' to 'insert' for cc_type
        if checkpoint['hyper_parameters']['cc_type'] == 'extra_token':
            checkpoint['hyper_parameters']['cc_type'] = 'insert'
        if checkpoint['datamodule_hyper_parameters']['cc_type'] == 'extra_token':
            checkpoint['datamodule_hyper_parameters']['cc_type'] = 'insert'

        # if self.hparams.plantoken:
        #     assert checkpoint['state_dict']['transformer.wte.weight'].shape[0] == checkpoint['state_dict']['lm_head.weight'].shape[0] == self.base_model.wte.weight.shape[0] + self.base_model.codebook.shape[0]
        #     self.add_codebook_to_embeddings()

        # backward compatibility
        # if 'og_params_finetune_type' not in checkpoint['hyper_parameters']:
        #     checkpoint['hyper_parameters']['og_params_finetune_type'] = 'freeze'
        #
        # # backward compatibility
        # if 'lm_as_planner' not in checkpoint['hyper_parameters']:
        #     checkpoint['hyper_parameters']['lm_as_planner'] = checkpoint['hyper_parameters']['plantoken']
        d = {
            'og_params_finetune_type': 'freeze',
            'jpl_temperature': 1.0,
            'jpl_gumbel': False,
            'straight_through': False,
             'lm_as_planner': checkpoint['hyper_parameters']['plantoken'] if 'lm_as_planner' not in checkpoint['hyper_parameters'] else ...,
             }
        for k, v in d.items():
            if k not in checkpoint['hyper_parameters']:
                self.hparams.__setattr__(k, v)

        # if Olmo: this is tied anyway
        ff_out_keys = [k for k in checkpoint['state_dict'] if 'transformer.ff_out.weight' in k]
        if len(ff_out_keys) > 0:
            assert len(ff_out_keys) == 1
            ff_out_key = ff_out_keys[0]
            wte_key = ff_out_key.replace('ff_out', 'wte')
            assert torch.equal(checkpoint['state_dict'][ff_out_key], checkpoint['state_dict'][wte_key])
            del checkpoint['state_dict'][ff_out_key]

        # Backwards compatibility when we still stored gate parameters in checkpoints that didn't use them
        if self.hparams.no_gate:
            for k in list(checkpoint['state_dict'].keys()):
                if k.endswith('gate'):
                    del checkpoint['state_dict'][k]

    def update_code_mask_and_input_ids(self, code_mask, input_ids):
        if self.hparams.plantoken:
            if (code_mask[:, -1] == -1).all():
                # In this case: deduce code_mask from input_ids: this requires that code ids are strictly bigger than vocab size. Used during lm.generate()
                # code_mask_last_part = input_ids[:,-1] >= self.lm_head.weight.shape[0]
                code_mask[:, -1] = input_ids[:, -1] >= self.vocab_size()
                code_mask = code_mask.bool()
        if code_mask is not None:
            if self.vocab_size() < self.base_model.codebook.shape[0]:
                raise ValueError(
                    "I was lazy: the following code only works if have less codes than tokens. If want more, rewrite this check")
            # if any input_id at position where code_mask is True is bigger than vocab size, it presumably means we are at generation-time. Subtract vocab size from those input_ids
            if input_ids is not None:
                msk = code_mask[:, -input_ids.shape[1]:]
                if (input_ids[msk] >= self.vocab_size()).any():  # -input_ids.shape[1]: is for scenario with key-value caching
                    assert (input_ids[msk] >= self.vocab_size()).all()
                    # input_ids[msk] -= self.lm_head.weight.shape[0]
                    # non-in-place variant
                    input_ids = input_ids - (msk * self.vocab_size())
        return code_mask, input_ids

    def vocab_size(self):
        return self.lm_head.weight.shape[0]

    def maybe_shift_code_labels(self, code_mask, labels):
        if 'plantoken' in self.hparams and self.hparams.plantoken:
            labels = torch.where(code_mask, labels + self.vocab_size(), labels)
        return labels

    def maybe_append_code_logits_to_lm_logits(self, hidden_states, lm_logits):
        assert (self.hparams.cc_type == 'insert') == hasattr(self.base_model, 'codebook')
        use_input_codebook_as_output_codebook = (self.hparams.cc_type == 'insert')
        if ('lm_as_planner' in self.hparams and self.hparams.lm_as_planner) or self.hparams.plantoken:
            code_logits = F.linear(hidden_states,
                                   self.output_codebook if not use_input_codebook_as_output_codebook else self.base_model.codebook)
            lm_logits = torch.cat([lm_logits, code_logits], dim=-1)
        return lm_logits

    def compute_codes_or_scores_for_articles(self, articles, bos_id, scores_iso_codes=False):
        from dataset import WikiDataset
        from eval import get_last_nonpad_idx_per_row
        if scores_iso_codes:
            raise NotImplementedError("TODO finish implementing option to condition not on single code, but distribution of codes")
        all_context_ids, all_context_masks, all_code_masks = [], [], []
        artidx2bound = {}
        prev_bound = 0
        if self.hparams.cc_type != 'insert':
            raise NotImplementedError("This function is only implemented for cc_type='insert'")
        for artidx, article in enumerate(tqdm(articles, desc="Gathering context_ids and context_masks for articles")):
            tokenized_sentences = article['tokenized_sentences']
            # art_context_ids = torch.tensor([42] * (self.hparams.max_seq_len - 1) + [bos_id]).unsqueeze(
            #     0)  # This is left padding, but in this case no problem because we convert to text anyway and ignore the padding
            # art_context_mask = torch.tensor([0] * (self.hparams.max_seq_len - 1) + [1]).unsqueeze(0)
            # for s in tokenized_sentences[:-1]:
            #     s = torch.tensor(s)
            #     new_context_ids = torch.cat([art_context_ids[-1][None], s[None]], dim=1)[:, -self.hparams.max_seq_len:]
            #     new_context_mask = torch.cat([art_context_mask[-1][None], torch.ones_like(s)[None]], dim=1)[:,
            #                        -self.hparams.max_seq_len:]
            #     art_context_ids = torch.cat([art_context_ids, new_context_ids], dim=0)
            #     art_context_mask = torch.cat([art_context_mask, new_context_mask], dim=0)
            # all_context_ids.append(art_context_ids)
            # all_context_masks.append(art_context_mask)
            artidx2bound[artidx] = (prev_bound, prev_bound + len(tokenized_sentences))
            prev_bound += len(tokenized_sentences)
            unpacked_art = WikiDataset.unpack_article(article, bos_id, self.hparams.cc_type)
            end = 1 # because of bos token
            for s in tokenized_sentences:
                assert unpacked_art.art_code_mask[end] == 1
                start = max(0, end - self.hparams.max_seq_len)
                padded_start = start
                padded_end = start + self.hparams.max_seq_len
                chunk = WikiDataset.get_chunk(unpacked_art,
                                              start=start,
                                              end=end,
                                              padded_start=padded_start,
                                              padded_end=padded_end,
                                      max_seq_len=self.hparams.max_seq_len, article_idx=artidx, bos_id=bos_id, cc_type=self.hparams.cc_type, get_eval_code=False)
                all_context_ids.append(chunk['input_ids_and_codes'])
                all_context_masks.append(chunk['attention_mask'])
                all_code_masks.append(chunk['code_mask'])
                end += len(s) + 1
        # stack list into tensor
        all_context_ids = torch.stack(all_context_ids)
        all_context_masks = torch.stack(all_context_masks)
        all_code_masks = torch.stack(all_code_masks)

        codes = []
        BS = 32
        if self.device.type == 'cpu':
            self.to('cuda')
        for i in tqdm(range(0, all_context_ids.shape[0], BS), desc="Computing planning codes for articles"):
            # representation = self.model_h(all_context_ids[i:i + BS].to('cuda'), all_context_masks[i:i + BS].to('cuda'))
            # if self.hparams.prob_cc:
            #     raise NotImplementedError(
            #         "TODO finish implementing option to condition not on single code, but distribution of codes")
            # else:
            #     code = self._get_planner_code(representation)
            #     codes.extend(code)
            inputs, att_mask, code_mask = [el[i:i + BS].to('cuda') for el in [all_context_ids, all_context_masks, all_code_masks]]
            all_logits = self(inputs,
                              attention_mask=att_mask,
                              code_mask = code_mask).logits
            last_nonpad_idx = get_last_nonpad_idx_per_row(att_mask)
            logits = all_logits[torch.arange(all_logits.shape[0]), last_nonpad_idx]
            code_logits = logits[:, -self.base_model.codebook.shape[0]:]
            codes.extend(code_logits.argmax(dim=-1).tolist())

        result = []
        for artidx, article in tqdm(enumerate(articles), desc="Adding planning codes to articles"):
            planner_codes = codes[artidx2bound[artidx][0]:artidx2bound[artidx][1]]
            assert len(planner_codes) == len(article['codes'])
            # article['planner_codes'] = planner_codes
            result.append(planner_codes)
        return result


class LoraLightningModule(LightningModule, LoraModel):

    def __init__(self, lora_args): #, lightningmodule_args):
        # filter args matching lightningmodule signature
        # lightningmodule_args = {k: v for k, v in lora_args.items() if k in inspect.signature(LightningModule.__init__).parameters}
        # LightningModule.__init__(self, **lora_args)
        LoraModel.__init__(self, **lora_args)


    def training_step(self, *args, **kwargs):
        return self._training_step(self, *args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self._validation_step(self, *args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self._test_step(self, *args, **kwargs)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self._configure_optimizers(self)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._on_save_checkpoint(self, checkpoint)

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self._on_load_checkpoint(self, checkpoint)