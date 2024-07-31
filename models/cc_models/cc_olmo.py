from typing import Optional, Sequence, Tuple, List, Union

import math
import torch
from hf_olmo.modeling_olmo import create_model_config_from_pretrained_config
from olmo.model import get_causal_attention_bias, OlmoSequentialBlock
from olmo.torch_util import ensure_finite_

from cc_models.cc_lm import CodeConditionedLM, CodeConditionedLMBaseModel, CodeConditionedLMAttention
from hf_olmo import OLMoForCausalLM
from olmo import Olmo, OlmoOutput, ActivationCheckpointingStrategy
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import functional as F

class CodeConditionedOLMoForCausalLM(OLMoForCausalLM, CodeConditionedLM):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._common_init(config, **kwargs)
        self.prep_code_conditioning_params(None, CodeConditionedOlmo,
                                           CodeConditionedOlmoSequentialBlock, 'self_attn', config, **kwargs)

    def prep_config_for_base_model(self, config):
        model_config = create_model_config_from_pretrained_config(config)
        # Initialize model (always on CPU to start with so we don't run out of GPU memory).
        model_config.init_device = "cpu"
        return model_config

    def forward(
        # region Same as OLMoForCausalLM
        self,
        input_ids: torch.LongTensor = None,
        # endregion
        codes: int = None, # ADDED ARGUMENT
        sent_embs: torch.Tensor = None, # ADDED ARGUMENT
        code_mask: torch.Tensor = None, # ADDED ARGUMENT
        planner_scores: torch.Tensor = None, # ADDED ARGUMENT
        # region Same as OLMoForCausalLM
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # endregion
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # region Same as OLMoForCausalLM
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OLMo")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        #endregion
        code_mask, input_ids = self.update_code_mask_and_input_ids(code_mask, input_ids)
        extra_args = {'codes': codes, 'sent_embs': sent_embs, 'code_mask': code_mask, 'planner_scores': planner_scores}
        if all([v is None for v in extra_args.values()]):
            extra_args = {}
        # region Same as OLMoForCausalLM
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # endregion
        outputs = self.model.forward(
            # region Same as OLMoForCausalLM
            input_ids=input_ids,
            # endregion
            **extra_args,
            # region Same as OLMoForCausalLM
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            # endregion
        )
        # region Same as OLMoForCausalLM
        logits = outputs.logits
        hidden_states = outputs.hidden_states
        # endregion
        logits = self.maybe_append_code_logits_to_lm_logits(hidden_states, logits)
        # region Same as OLMoForCausalLM
        loss = None
        # endregion
        if labels is not None:
            labels = self.maybe_shift_code_labels(code_mask, labels)
            # region Same as OLMoForCausalLM
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.embedding_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # endregion
        # region Same as OLMoForCausalLM
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )
        # endregion

    def vocab_size(self):
        return self.model.transformer.wte.weight.shape[0]

    def wte(self, *args, **kwargs):
        return self.model.transformer.wte(*args, **kwargs)

class CodeConditionedOlmo(Olmo, CodeConditionedLMBaseModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.__cache = self._Olmo__cache # Needed cuz name mangling
        self.__num_fwd_flops = 0 # Needed cuz name mangling
        self.prep_code_conditioning_params(config, **kwargs)

    def init_cc_layer(self, layer_id, config, codebook, no_gate, ucas, LayerModule, *_):
        return LayerModule(layer_id, config, self.__cache, codebook=codebook, no_cluster=self.hparams.no_cluster,
                           no_gate=no_gate,
                           unfreeze_codebook_after_step=ucas,
                           hparams=self.hparams)

    def maybe_quantize(self, codebook, config):
        # if config.quantization_config.load_in_8bit:
        #     codebook = codebook.to(torch.float16)
        return codebook

    @property
    def layers_module(self):
        return self.transformer.blocks

    def forward(
            # region Same as OLMo
            self,
            input_ids: torch.LongTensor,
            # endregion
            codes: int = None, # ADDED ARGUMENT
            sent_embs: torch.Tensor = None, # ADDED ARGUMENT
            code_mask: torch.Tensor = None, # ADDED ARGUMENT
            planner_scores: torch.Tensor = None, # ADDED ARGUMENT
            # region Same as OLMo
            input_embeddings: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            attention_bias: Optional[torch.Tensor] = None,
            past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            last_logits_only: bool = False,
            output_hidden_states: Optional[bool] = None,
            # endregion
    ) -> OlmoOutput:
        # region Same as OLMo
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        # endregion
        # x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore
        x =  self.get_maybe_cc_embeds(code_mask, input_ids, input_embeddings)
        # region Same as OLMo
        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x
        # endregion
        x = self.maybe_addition_style_code_condition_embedding(x, codes, sent_embs, code_mask)
        # region Same as OLMo
        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
                attention_bias is not None
                or attention_mask is not None
                or self.config.alibi
                # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
                # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
                # scores correctly.
                or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        # endregion
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                # region Same as OLMo
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                # endregion
                code_conditioned = type(block) == CodeConditionedOlmoSequentialBlock # ADDED
                block_kwargs = {'x': x, 'attention_bias': attention_bias, 'layer_past': layer_past, 'use_cache': use_cache} # ADDED
                if code_conditioned: # ADDED
                    block_kwargs |= {'codes': codes, 'sent_embs': sent_embs, 'planner_scores': planner_scores} # ADDED
                if (
                    # region  Same as OLMo
                        (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                        or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                )
                        or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                )
                        or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                )
                    # endregion
                ):
                    # region Same as OLMo
                    # shape: (batch_size, seq_len, d_model)
                    # endregion
                    # x, cache = self._activation_checkpoint_fn(
                    #     block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                    # )
                    x, cache = self._activation_checkpoint_fn(block, **block_kwargs) # ADDED
                else:
                    # region Same as OLMo
                    # shape: (batch_size, seq_len, d_model)
                    # endregion
                    # x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                    x, cache = block(**block_kwargs) # ADDED
                # region Same as OLMo
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
                # endregion
        else:
            raise NotImplementedError("block_group_size > 1 not yet supported")
            # region Same as OLMo
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                         group_idx * self.config.block_group_size: (group_idx + 1) * self.config.block_group_size
                         ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)
            # endregion

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OlmoOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(
            all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]
        # endregion

    def wte(self, *args, **kwargs):
        return self.transformer.wte(*args, **kwargs)

class CodeConditionedOlmoSequentialBlock(OlmoSequentialBlock, CodeConditionedLMAttention):

    def __init__(self, layer_id, config, cache, **kwargs):
        super().__init__(layer_id, config, cache)
        self.prep_code_conditioning_params(config, **kwargs)


    def forward(
        # region Same as OlmoSequentialBlock
        self,
        x: torch.Tensor,
        # endregion
        codes: torch.Tensor = None, # ADDED ARGUMENT
        sent_embs: torch.Tensor = None, # ADDED ARGUMENT
        planner_scores: torch.Tensor = None, # ADDED ARGUMENT
        # region Same as OlmoSequentialBlock
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        # endregion
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # region Same as OlmoSequentialBlock
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        if self._activation_checkpoint_fn is not None:
            qkv = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x))
        else:
            qkv = self.att_proj(self.attn_norm(x))

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        # endregion
        # region Code conditioning
        if codes is not None or planner_scores is not None:
            x = self.code_condition(x, codes, sent_embs, planner_scores)
        # endregion
        # region Same as OlmoSequentialBlock
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache
        # endregion

    def maybe_quantize(self, codebook, config):
        return codebook

    @property
    def attn_module(self):
        return self