import argparse
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from muzero_dataloader import create_dataloader, TextTargetDataset
from muzero_tree import ModelG, ModelH, ModelF, MuZeroTree
from torch.utils.data import SubsetRandomSampler
import random
from collections import Counter
from sklearn.metrics import accuracy_score
from misc_functions import embed_strings


def evaluate_model(args, dataloader, model, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_rank = 0
        for i, batch in enumerate(dataloader):

            input_data, target_data = batch
            target_data = target_data.to(device)  # Move the target data to the GPU if available
            output = model(input_data)
            batch_size = target_data.size(0)
            loss = criterion(output.view(batch_size * output.size(1), -1), target_data.view(batch_size * output.size(1)))
            total_loss += loss.item()
            _, predicted = torch.max(output, -1)
            total_correct += (predicted == target_data).sum().item()
            total_samples += target_data.size(0)
            # Compute the rank of the correct target label for each sample in the batch
            # The previous implementation was buggy and always returned 0 for everything.
            # Here is the corrected version:
            # Sort the output probabilities and get the indices for the entire batch
            indices = torch.argsort(output, dim=-1, descending=True)
            # Create a 2D mask where each row corresponds to a sample and has True at the position of the correct label and False elsewhere
            mask = (indices == target_data.unsqueeze(-1))
            # Get the ranks of the correct labels for all samples
            ranks = (mask.nonzero(as_tuple=True)[-1] + 1).float()
            # Add the ranks to the total rank
            total_rank += ranks.sum().item()
        print(f'Loss: {total_loss / len(dataloader)}')
        print(f'Accuracy: {total_correct / total_samples * 100}%')
        print(f'Average Rank: {total_rank / total_samples}')

def train_model(args, train_dataloader, dev_dataloader, device, pretrained_embeddings=None):
    # Define the models
    #print(args.use_self_attention)
    #sys.exit()
    model_h = ModelH(model_name=args.model_name, use_self_attention=args.use_self_attention, output_size=args.embedding_size, freeze=args.freeze).to(device)
    model_g = ModelG(args.embedding_size, args.embedding_size).to(device)

    # if we want to initialize the output layer with pretrained embeddings
    model_f = ModelF(args.embedding_size, args.embedding_size, args.num_classes, target_embeddings=pretrained_embeddings).to(device)  # The output size for model_f is 10
    model_muzero_tree = MuZeroTree(model_h, model_g, model_f, args.max_future_steps).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(list(model_g.parameters()) + list(model_h.parameters()) + list(model_f.parameters()), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Import tqdm for the progress bar
    from functools import partial
    from tqdm import tqdm as std_tqdm
    tqdm = partial(std_tqdm, dynamic_ncols=True)

    # Define the training loop
    # Initialize a list to keep track of the training losses

    for epoch in range(args.num_epochs):

        model_muzero_tree.train()
        training_losses = []

        # Create a progress bar
        pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            
            #if batch_idx == 10:
            #    break

            input_data, target_data = batch
            target_data = target_data.to(device)
            batch_size = target_data.size(0)

            # Forward pass
            output = model_muzero_tree(input_data)

            # Compute loss
            loss = criterion(output.view(batch_size * output.size(1), -1), target_data.view(batch_size * output.size(1)))

            # Add the loss to the training losses list
            training_losses.append(loss.item())

            loss = loss.mean()
            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % args.accumulation_steps == 0:
                # Optimization step and zero the gradients
                optimizer.step()
                optimizer.zero_grad()

            # Update the progress bar
            pbar.update(1)

        # Close the progress bar
        pbar.close()

        # Print the average training loss for this epoch
        print(f'Average training loss for epoch {epoch+1}: {sum(training_losses)/len(training_losses)}')

        # Evaluate the model every few epochs
        if epoch % 1 == 0:
            evaluate_model(args, dev_dataloader, model_muzero_tree, criterion, device)

    return model_muzero_tree

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default='new_daily-dialog-8000_clean.csv')
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--max_future_steps', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--initialize_outputlayer', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='gpt2-xl')
    parser.add_argument('--use_self_attention', type=bool, default=False)
    parser.add_argument('--embedding_size', type=int, default=384)
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--freeze', type=bool, default=False)
    args = parser.parse_args()

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 

    # Load the data
    filename = args.filename
    batch_size = args.batch_size
    max_future_steps = args.max_future_steps
    
    # Create a TextTargetDataset for all data
    all_data = create_dataloader(filename, batch_size, max_future_steps).dataset

    # Infer the train, dev, test splits by splitting over the all_data.unique_texts list
    unique_texts = range(len(all_data.unique_texts))
    train_data, test_data = train_test_split(unique_texts, test_size=0.2, random_state=42)
    train_data, dev_data = train_test_split(train_data, test_size=0.25, random_state=42)

    # Create SubsetRandomSampler for each split that uses these unique set indices
    train_sampler = SubsetRandomSampler([i for i, text in enumerate(all_data.text_target_pairs) if text[0] in train_data])
    dev_sampler = SubsetRandomSampler([i for i, text in enumerate(all_data.text_target_pairs) if text[0] in dev_data])
    test_sampler = SubsetRandomSampler([i for i, text in enumerate(all_data.text_target_pairs) if text[0] in test_data])

    # Create dataloaders for each split using the same TextTargetDataset but different samplers
    train_dataloader = DataLoader(all_data, batch_size=batch_size, sampler=train_sampler)
    dev_dataloader = DataLoader(all_data, batch_size=batch_size, sampler=dev_sampler)
    test_dataloader = DataLoader(all_data, batch_size=batch_size, sampler=test_sampler)

    compute_baseline = True
    if compute_baseline:
        # Random baseline
        targets = [data[1].tolist() for data in test_dataloader]
        targets = [item for sublist in targets for item in sublist] # Flatten the list
        targets = [item for sublist in targets for item in sublist] # Flatten the list again
        random_baseline = [random.choice(range(all_data.num_classes)) for _ in targets]
        print(f'Random baseline accuracy: {accuracy_score(targets, random_baseline)}')

        # Majority class baseline
        majority_class = Counter(targets).most_common(1)[0][0]
        
        majority_baseline = [majority_class for _ in targets]
        print(f'Majority class baseline accuracy: {accuracy_score(targets, majority_baseline)}')

    # Train the model
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate # Move the model to the GPU if available
    accumulation_steps = args.accumulation_steps
    initialize_outputlayer = args.initialize_outputlayer
    if initialize_outputlayer:
        pretrained_embeddings = embed_strings(all_data.unique_targets)
    else:
        pretrained_embeddings = None

    model = train_model(args, train_dataloader, dev_dataloader, device, pretrained_embeddings=pretrained_embeddings)

    # Evaluate the model on the test set at the end of training
    criterion = torch.nn.CrossEntropyLoss()
    evaluate_model(args, test_dataloader, model, criterion, device)

if __name__ == "__main__":
    main()
