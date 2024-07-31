import pandas as pd
import ast
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torch.nn import functional as F


def get_text_target_pairs(filename, max_future_steps):
    df = pd.read_csv(filename, quotechar='|')
    text_target_pairs = []
    for j, row in df.iterrows():
        pairs = ast.literal_eval(row['conversation_sentence_wm_pairs'])
        for i in range(1, len(pairs) - max_future_steps + 1):
            text = ' '.join([pair[0] for pair in pairs[:i]])
            targets = [pair[2] for pair in pairs[i:i+max_future_steps]]
            text_target_pairs.append((j, text, targets))
    return text_target_pairs

class TextTargetDataset(Dataset):
    def __init__(self, text_target_pairs, max_future_steps, unique_targets = None):
        self.text_target_pairs = text_target_pairs
        self.max_future_steps = max_future_steps

        # Create a mapping from unique targets to integers
        targets = [target for _, _, targets in text_target_pairs for target in targets]
        if unique_targets is None:
            unique_targets = list(set(targets))
        
        self.unique_targets = unique_targets
        self.target_to_int = {target: i for i, target in enumerate(unique_targets)}
        self.padding_index = -100  # Index for padding, -1 is ignored by PyTorch's CrossEntropyLoss

        # Convert targets to integer labels and pad to max_future_steps
        self.text_target_pairs = [(j, text, self.pad_targets([self.target_to_int[target] for target in targets])) for j, text, targets in text_target_pairs]

        # Save the set of unique texts which is all the j's in the pairs
        self.unique_texts = set(j for j, _, _ in self.text_target_pairs)

        self.num_classes = len(unique_targets)  # Account for padding index

    def pad_targets(self, targets):
        # If targets length is less than max_future_steps, pad it
        if len(targets) < self.max_future_steps:
            targets += [self.padding_index] * (self.max_future_steps - len(targets))
        return targets

    def __len__(self):
        return len(self.text_target_pairs)

    def __getitem__(self, idx):
        _, text, targets = self.text_target_pairs[idx]
        targets = torch.tensor(targets)
        return text, targets
    
def create_dataloader(filename, batch_size, max_future_steps):
    text_target_pairs = get_text_target_pairs(filename, max_future_steps)
    dataset = TextTargetDataset(text_target_pairs, max_future_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader