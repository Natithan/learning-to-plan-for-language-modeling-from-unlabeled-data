import torch
from torch import nn
from util import scale_gradient, tn
from representation_functions import ModelH
from dynamics_functions import ModelG
from prediction_functions import ModelF
import random
import string

# Define a new class for the forward pass with multiple timesteps
class MuZeroTree(nn.Module):
    def __init__(self, model_h, model_g, model_f, num_timesteps, multi_vector_representation=True, only_policy_head=False):
        super(MuZeroTree, self).__init__()
        self.model_h = model_h
        self.model_g = model_g
        self.model_f = model_f
        self.num_timesteps = num_timesteps
        self.multi_vector_representation = multi_vector_representation
        self.policy_only = only_policy_head

    def single_prediction():
        f_out = self.model_f(dynamics)
        if self.policy_only:
            policy_logits = f_out
            return policy_logits
        else:
            policy_logits, value = f_out
            return policy_logits, value

    def single_step(self, dynamics, action):

        if self.policy_only:
            dynamics = self.model_g((dynamics, action))
            reward = None
        else:
            dynamics, reward = self.model_g((dynamics, action))

        if self.multi_vector_representation:
            dynamics = (scale_gradient(dynamics[0], 0.5), dynamics[1])
        else:
            dynamics = scale_gradient(dynamics, 0.5)

        return dynamics, reward

    def forward(self, input_ids, attention_mask, actions=None, num_timesteps=None, teacher_forcing=False):

        if num_timesteps is None:
            num_timesteps = self.num_timesteps

        representation = self.model_h(input_ids, attention_mask)

        no_dynamics = (num_timesteps == 1) and self.policy_only
        if no_dynamics:
            return self.model_f(representation)
        else:
            dynamics = representation
            policy_logit_list, value_list, reward_list = [], [], []
            for i in range(num_timesteps):
                assert teacher_forcing == False or actions is not None, "Actions must be provided for multiple timesteps"
                if self.policy_only:
                    policy_logits = self.model_f(dynamics)
                else:
                    policy_logits, value = self.model_f(dynamics)

                if teacher_forcing:
                    a = actions[:, i]
                else:
                    a = torch.argmax(policy_logits, dim=-1)
                dynamics, reward = self.single_step(dynamics, a)
                policy_logit_list.append(policy_logits)

                if not self.policy_only:
                    value_list.append(value)
                    reward_list.append(reward)

            if self.policy_only:
                return torch.stack(policy_logit_list, dim=1)
            else:
                return torch.stack(policy_logit_list, dim=1), torch.stack(value_list, dim=1), torch.stack(reward_list, dim=1)

def main():
    # Define the input and output sizes and the learning rate
    input_size = 768  # The output size of the vector from model_h is 768
    output_size = 768  # For model_g, input and output size need to be the same
    learning_rate = 0.001
    num_epochs = 1000
    batch_size = 32
    embedding_size = 384
    n_time_steps = 5

    # Initialize the models
    model_g = ModelG(input_size, output_size)
    model_h = ModelH(model_name='gpt2')
    model_f = ModelF(input_size, embedding_size, 10)  # The output size for model_f is 10
    model_muzero_tree = MuZeroTree(model_h, model_g, model_f, n_time_steps)

    # Define the optimizer
    optimizer = torch.optim.Adam(list(model_g.parameters()) + list(model_h.parameters()) + list(model_f.parameters()), lr=learning_rate)

    # Define the training loop
    for epoch in range(num_epochs):
        # Generate dummy data
        input_data = [' '.join(random.choices(string.ascii_uppercase + string.ascii_lowercase, k=random.randint(n_time_steps,15))) for _ in range(batch_size)]
        # target_data = [torch.randn((batch_size, 10)) for _ in range(5)]  # different targets per step
        target_data = torch.randn((batch_size, n_time_steps, 10))  # match dimension order of predictions
        # Forward pass through the models with multiple timesteps
        predictions = model_muzero_tree(input_data)

        # Compute the loss
        loss = sum(nn.MSELoss()(prediction, target) for prediction, target in zip(predictions, target_data)) / len(predictions)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print the loss every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

if __name__ == "__main__":
    main()
