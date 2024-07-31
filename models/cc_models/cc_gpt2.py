# region preamble
import math
from typing import Optional, Tuple, List, Union, Dict, Any
import bitsandbytes as bnb
import torch

from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaModel, GPT2LMHeadModel, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, \
    CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Attention
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv, LlamaDecoderLayer, \
    logger
from transformers.utils import ModelOutput

from cc_models.cc_lm import CodeConditionedLM, CodeConditionedLMAttention, CodeConditionedLMBaseModel
from lightning import LightningModule
from util import get_codebook
# endregion

class CodeConditionedGPT2Attention(GPT2Attention, CodeConditionedLMAttention):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.prep_code_conditioning_params(config, **kwargs)

    def forward(
        # region Same as GPT2Attention
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        # endregion
        codes: torch.Tensor = None, # ADDED ARGUMENT
        sent_embs: torch.Tensor = None, # ADDED ARGUMENT
        planner_scores: torch.Tensor = None, # ADDED ARGUMENT
        # region Same as GPT2Attention
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        # endregion
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        # region Same as GPT2Attention
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        # endregion

        # region Code conditioning
        if codes is not None or planner_scores is not None:
            attn_output = self.code_condition(attn_output, codes, sent_embs, planner_scores)
        # endregion

        # region Same as GPT2Attention
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
        # endregion

    def maybe_quantize(self, codebook, config):
        return codebook


class CodeConditionedGPT2LMHeadModel(GPT2LMHeadModel, CodeConditionedLM):


    # def __init__(self, config, num_layers_to_condition: int = None, code_dim=768, data_pkl_path=None, **_):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self._common_init(config, **kwargs)
        self.prep_code_conditioning_params(CodeConditionedGPT2Attention, CodeConditionedGPT2Model,
                                           CodeConditionedGPT2Block, 'attn', config, **kwargs)

    def forward(
            # region Same as GPT2LMHeadModel
            self,
            input_ids: Optional[torch.LongTensor] = None,
            # endregion
            codes: int = None, # ADDED ARGUMENT
            sent_embs: torch.Tensor = None, # ADDED ARGUMENT
            code_mask: torch.Tensor = None, # ADDED ARGUMENT
            planner_scores: torch.Tensor = None, # ADDED ARGUMENT
            # region Same as GPT2LMHeadModel
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # endregion
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        # region Same as GPT2LMHeadModel
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # endregion

        code_mask, input_ids = self.update_code_mask_and_input_ids(code_mask, input_ids)

        extra_args = [ codes, sent_embs, code_mask, planner_scores ] if self.hparams.cc_type != 'none' else []

        transformer_outputs = self.transformer(
            # region Same as GPT2LMHeadModel
            input_ids,
            # endregion
            *extra_args, # ADDED ARGUMENTS
            # region Same as GPT2LMHeadModel
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # endregion
        )
        # region Same as GPT2LMHeadModel
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)
        # endregion
        lm_logits = self.maybe_append_code_logits_to_lm_logits(hidden_states, lm_logits)
        # region Same as GPT2LMHeadModel
        loss = None
        #endregion
        if labels is not None:
            labels = self.maybe_shift_code_labels(code_mask, labels)
            # region Same as GPT2LMHeadModel
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # endregion
        # region Same as GPT2LMHeadModel
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        # endregion


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        # region Same as GPT2Model
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        # endregion
        if 'code_mask' in kwargs:
            model_inputs['code_mask'] = kwargs['code_mask']

        return model_inputs

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: Dict[str, Any],
        is_encoder_decoder: bool = False,
        standardize_cache_format: bool = False,
    ) -> Dict[str, Any]:
        # region Same as GenerationMixin
        # update past_key_values
        model_kwargs["past_key_values"] = self._extract_past_from_model_output(
            outputs, standardize_cache_format=standardize_cache_format
        )
        if getattr(outputs, "state", None) is not None:
            model_kwargs["state"] = outputs.state

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = torch.cat([token_type_ids, token_type_ids[:, -1].unsqueeze(-1)], dim=-1)

        if not is_encoder_decoder:
            # update attention mask
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )
        else:
            # update decoder attention mask
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                model_kwargs["decoder_attention_mask"] = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask.new_ones((decoder_attention_mask.shape[0], 1))],
                    dim=-1,
                )
        # endregion
        if 'code_mask' in model_kwargs:
            code_mask = model_kwargs['code_mask']
            model_kwargs['code_mask'] = torch.cat(
                [
                    code_mask,
                    -1* code_mask.new_ones((code_mask.shape[0], 1)) # correct boolean value to be derived from input_ids later
                ], dim=-1)
        return model_kwargs

class CodeConditionedGPT2Block(GPT2Block):

    def forward(
        # region Same as GPT2Block
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        # endregion
        codes: torch.Tensor, # ADDED ARGUMENT
        sent_embs: torch.Tensor, # ADDED ARGUMENT
        planner_scores: torch.Tensor, # ADDED ARGUMENT
        # region Same as GPT2Block
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        # endregion
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # region Same as GPT2Block
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        # endregion
        attn_args = ({"hidden_states": hidden_states, "layer_past": layer_past, "attention_mask": attention_mask, "head_mask": head_mask, "use_cache": use_cache, "output_attentions": output_attentions} |
                     ({"codes":codes, "sent_embs": sent_embs, "planner_scores": planner_scores} if type(self.attn) == CodeConditionedGPT2Attention else {}))
        attn_outputs = self.attn(**attn_args)
        # region Same as GPT2Block
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)
        # endregion

    @property
    def attn_module(self):
        return self.attn


class CodeConditionedGPT2Model(GPT2Model, CodeConditionedLMBaseModel):

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.prep_code_conditioning_params(config, **kwargs)

    def maybe_quantize(self, codebook, config):
        return codebook

    def forward(
        # region Same as GPT2Model
        self,
        input_ids: Optional[torch.LongTensor] = None,
        # endregion
        codes: int = None, # ADDED ARGUMENT
        sent_embs: torch.Tensor = None, # ADDED ARGUMENT
        code_mask: torch.Tensor = None, # ADDED ARGUMENT
        planner_scores: torch.Tensor = None, # ADDED ARGUMENT
        # region Same as GPT2Model
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # endregion
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # region Same as GPT2Model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        # endregion
        inputs_embeds = self.get_maybe_cc_embeds(code_mask, input_ids, inputs_embeds)
        # region Same as GPT2Model
        position_embeds = self.wpe(position_ids)
        # endregion
        inputs_embeds = self.maybe_addition_style_code_condition_embedding(inputs_embeds, codes, sent_embs, code_mask)
        # region Same as GPT2Model
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        # endregion
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # region Same as GPT2Model
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            # endregion
            code_conditioned = type(block) == CodeConditionedGPT2Block # ADDED
            if self.gradient_checkpointing and self.training:
                # region Same as GPT2Model
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward
                # endregion
                args = [hidden_states, None, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask] + ([codes, sent_embs, planner_scores] if code_conditioned else []) # ADDED
                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    *args
                )
            else:
                kwargs = {'layer_past': layer_past, 'attention_mask': attention_mask, 'head_mask': head_mask[i],
                          'encoder_hidden_states': encoder_hidden_states,
                          'encoder_attention_mask': encoder_attention_mask, 'use_cache': use_cache,
                          'output_attentions': output_attentions} | ({'codes': codes, 'sent_embs': sent_embs, 'planner_scores': planner_scores} if code_conditioned else {}) # ADDED
                outputs = block(
                    hidden_states,**kwargs
                )
            # region Same as GPT2Model
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))
            # endregion
        # region Same as GPT2Model
        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        # endregion


    @property
    def layers_module(self):
        return self.h
