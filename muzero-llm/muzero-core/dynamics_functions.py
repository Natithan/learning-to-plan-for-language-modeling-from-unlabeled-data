import torch
from torch import nn
from nnets import ResidualLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class ModelG(nn.Module):
    def __init__(self, input_size, output_size, num_actions, hidden_size=None, num_layers=1, dropout_p=0.5, skip_and_norm=False, no_reward_head=False):
        super(ModelG, self).__init__()
        hidden_size = output_size if hidden_size is None else hidden_size
        self.action_embedder = nn.Embedding(num_actions, input_size)

        # self.linear = nn.Linear(input_size, hidden_size)
        # self.activation = nn.GELU()
        layers = [ResidualLayer(input_size, hidden_size, dropout_p, use_residual=skip_and_norm, use_norm=skip_and_norm)] + \
                 [ResidualLayer(hidden_size, hidden_size, dropout_p, use_residual=skip_and_norm, use_norm=skip_and_norm)] * (num_layers - 1)
        self.common_layers = nn.Sequential(*layers)

        self.state_head = nn.Linear(hidden_size, output_size)
        self.no_reward_head = no_reward_head
        if not no_reward_head:
            self.reward_head = nn.Linear(hidden_size, 1)
        #else:
        #    raise NotImplementedError("If no reward head, haven't yet implemented update Muzero training step and eval step to deal with that scenario.")

    def forward(self, batch):
        prev_state, action = batch
        action_embedding = self.action_embedder(action)
        prev_state = prev_state + action_embedding
        # x = self.linear(prev_state)
        # x = self.activation(x)
        x = self.common_layers(prev_state)
        next_state = self.state_head(x)
        if not self.no_reward_head:
            reward = self.reward_head(x).squeeze(-1)
            return next_state, reward
        else:
            return next_state


class SentenceBasedModelG(nn.Module):
    def __init__(self, input_size, output_size, num_actions, hidden_size=None, num_layers=1, dropout_p=0.5, no_reward_head=False):
        super(SentenceBasedModelG, self).__init__()
        hidden_size = output_size if hidden_size is None else hidden_size
        self.action_embedder = nn.Embedding(num_actions, hidden_size)
        self.new_embedding = nn.Parameter(torch.randn(hidden_size))

        encoder_layers = TransformerEncoderLayer(d_model=input_size, nhead=2, dropout=dropout_p)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.no_reward_head = no_reward_head
        if not no_reward_head:
            self.reward_head = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        prev_state, action = batch
        action_embedding = self.action_embedder(action)
        action_embedding = action_embedding + self.new_embedding

        # Add action to the front of the state tensor
        state, state_length = prev_state
        state = torch.cat([action_embedding.unsqueeze(1), state], dim=1)
        state_length = state_length + 1

        mask = self.generate_mask(state_length, state.size()[1])
        output = self.transformer_encoder(state.transpose(0, 1), src_key_padding_mask=mask).transpose(0, 1)

        if not self.no_reward_head:
            reward = self.reward_head(output[:, 0]).squeeze(-1) 
            return (output, state_length), reward
        else:
            return (output, state_length)

    def generate_mask(self, input_length, max_len):
        mask = input_length.unsqueeze(1) <= torch.arange(max_len).expand(input_length.size()[0], -1).to(input_length.device)
        return mask
