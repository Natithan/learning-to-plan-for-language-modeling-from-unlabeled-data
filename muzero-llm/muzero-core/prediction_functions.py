import torch
from torch import nn
# import equinox as eqx
# import jax.random as jrandom
from nnets import ResidualLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from util import tn

class ModelF(nn.Module):
    def __init__(self, input_size, hidden_size, action_space_size, num_layers: int =1, dropout_p=0.5, skip_and_norm=False, only_policy_head=False, regression_policy_head=False, action_embedding_size=384):
        super(ModelF, self).__init__()
        assert num_layers >= 1, "Must have at least one layer"

        layers = [ResidualLayer(input_size, hidden_size, dropout_p, use_residual=skip_and_norm, use_norm=skip_and_norm)] + \
                 [ResidualLayer(hidden_size, hidden_size, dropout_p, use_residual=skip_and_norm, use_norm=skip_and_norm)] * (num_layers - 1)
        self.model = nn.Sequential(*layers)

        self.policy_head = nn.Linear(hidden_size, action_space_size if not regression_policy_head else action_embedding_size)
        self.only_policy_head = only_policy_head
        if not only_policy_head:
            self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input):
        output = self.model(input)
        policy_logits = self.policy_head(output)
        if not self.only_policy_head:
            value = self.value_head(output).squeeze(-1)
            return policy_logits, value
        else:
            return policy_logits


class SentenceBasedModelF(nn.Module):
    def __init__(self, input_size, hidden_size, action_space_size, num_layers: int =1, dropout_p=0.5, only_policy_head=False):
        super(SentenceBasedModelF, self).__init__()
        assert num_layers >= 1, "Must have at least one layer"

        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=2, dropout=dropout_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers, enable_nested_tensor=False) # to avoid warning "enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)"

        self.policy_head = nn.Linear(hidden_size, action_space_size)
        self.only_policy_head = only_policy_head
        if not only_policy_head:
            self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_tuple):
        input, input_length = input_tuple
        mask = self.generate_mask(input_length, input.size()[1])
        output = self.transformer_encoder(input.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)
        
        # Apply average pooling
        output_masked = output.masked_fill(mask.unsqueeze(-1), 0)
        output_sum = output_masked.sum(dim=1)
        seq_len_unpadded = (~mask).sum(dim=1, keepdim=True)
        if (seq_len_unpadded == 0).any():
            raise ValueError("seq len 0")
        output_avg = output_sum / seq_len_unpadded

        policy_logits = self.policy_head(output_avg)

        if not self.only_policy_head:
            value = self.value_head(output_avg).squeeze(-1)
            return policy_logits, value
        else:
            return policy_logits


    def generate_mask(self, input_length, max_len):
        mask = input_length.unsqueeze(1) <= torch.arange(max_len).expand(input_length.size()[0], -1).to(input_length.device)
        return mask
