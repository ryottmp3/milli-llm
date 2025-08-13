# A small LLM in python
# Author: H. Ryott Glayzer
# License: GPLv2

# Import Modules
import torch # PyTorch for implementing LLM
from torch import nn # Neural Network module from PyTorch
from torch.nn import functional as F # Neural Network module from PyTorch
import numpy as np
from matplotlib import pyplot as plt
import time
import pandas as pd
import urllib.request
import os

# Create config
MASTER_CONFIG = {
    'context_window': 16,     # Number of characters in each input (x) and target (y) sequence of each batch
    'vocab_size': 65,
    'd_model': 128,
    'epochs': 1000,          # Number of training epochs
    'log_interval': 10,      # Log information every 10 batches during training
    'batch_size': 32,        # Increase batch size to 32
}


### DATA ACQUISITION ###

# Train LLM on shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# where it will be stored locally
file_name = "shakespeare.data"

# Download data
if not os.path.exists(file_name):
    urllib.request.urlretrieve(url, file_name)



### DETERMINE VOCABULARY SIZE ###

lines = open(file_name, "r").read()
vocab = sorted(list(set(lines)))
print(f'First 10 Characters: {vocab[:10]}')
print(f'Vocabulary size: {len(vocab)}')



### INITIALIZE AND CREATE TOKENIZER ###
# Map integers to chars
itos = {i: ch for i, ch in enumerate(vocab)}

# Map chars it ints
stoi = {ch: i for i, ch in enumerate(vocab)}

# Encode: str -> list of ints
def encode(s):
    return [stoi[ch] for ch in s]

# Decode: int -> str
def decode(l):
    return ''.join([itos[i] for i in l])

# Convert data to tensor
dataset = torch.tensor(encode(lines), dtype=torch.int8)

# dataset.shape is 1115394, or ~1 million tokens


### Functions ###

# Function to get batches
def get_batches(data, split, batch_size, context_window, config=MASTER_CONFIG):
    # Split the dataset into training, validation, and test sets
    train = data[:int(.8 * len(data))]
    val = data[int(.8 * len(data)): int(.9 * len(data))]
    test = data[int(.9 * len(data)):]

    # Determine which split to use
    batch_data = train
    if split == 'val':
        batch_data = val
    if split == 'test':
        batch_data = test

    # Pick random starting point in the data
    ix = torch.randint(0, batch_data.size(0) - context_window - 1, (batch_size,))

    # Create input sequences (x) and corresponding target sequences (y)
    x = torch.stack([batch_data[i:i+context_window] for i in ix]).long()
    y = torch.stack([batch_data[i+1:i+context_window] for i in ix]).long()

    return x, y


@torch.no_grad() # Don't compute gradients when evaluating
def evaluate_loss(model, config=MASTER_CONFIG):
    # Placeholder for eval results
    out = {}

    # Set the model to eval mode
    model.eval()

    # iterate through training and validation splits
    for split in ["train", "val"]:
        # Placeholder for individual losses
        losses = []

        # Generate 10 batches for eval
        for _ in range(10):
            xb, yb = get_batches(
                data=dataset,
                split=split,
                batch_size=config["batch_size"],
                context_window=config["context_window"],
            )

            # Perform model inference and calculate loss
            _, loss = model(xb, yb)

            # Append the loss to the list
            losses.append(loss.item())

        # Calculate mean loss and store it in output dict
        out[split] = np.mean(losses)

    # Set the model back to training mode
    model.train()

    return out





### BUILD A BASE NEURAL NETWORK ###

class SimpleBrokenModel(nn.Module):
    def __init__(self, config=MASTER_CONFIG):
        super().__init__()
        self.config = config

        # Embedding layer: char -> vectors
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # Linear layers for modeling relationships between features
        # (to be updated with SwiGLU activation function as in LLaMA)
        self.linear = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.ReLU(),  # Currently using ReLU, will be replaced with SwiGLU as in LLaMA
            nn.Linear(config['d_model'], config['vocab_size']),
        )

        # Print the total number of model parameters
        # print("Model parameters:", sum([m.numel() for m in self.parameters()]))

    # Forward pass function for the base model
    def forward(self, idx, targets=None):
        # Embedding layer converts character indices to vectors
        x = self.embedding(idx)

        # Linear layers for modeling relationships between features
        a = self.linear(x)

        # Apply softmax activation to obtain probability distribution
        logits = F.softmax(a, dim=-1)

        # If targets are provided, calculate and return the cross-entropy loss
        if targets is not None:
            # Reshape logits and targets for cross-entropy calculation
            loss = F.cross_entropy(logits.view(-1, self.config['vocab_size']), targets.view(-1))
            return logits, loss

        # If targets are not provided, return the logits
        else:
            return logits

        # Print the total number of model parameters
        # print("Model parameters:", sum([m.numel() for m in self.parameters()]))

model = SimpleBrokenModel(MASTER_CONFIG)



# Obtain batches for training using the specified batch size and context window
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# Decode the sequences to obtain the corresponding text representations
decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]


logits, loss = model(xs,ys)


optimizer = torch.optim.Adam(
    model.parameters(),      # Pass the model parameters to the optimizer
)




# Function to perform training
def train(model, optimizer, scheduler=None, config=MASTER_CONFIG, print_logs=False):
    # Placeholder for storing losses
    losses = []

    # Start tracking time
    start_time = time.time()

    # Iterate through epochs
    for epoch in range(config['epochs']):
        # Zero out gradients
        optimizer.zero_grad()

        # Obtain batches for training
        xs, ys = get_batches(dataset, 'train', config['batch_size'], config['context_window'])

        # Forward pass through the model to calculate logits and loss
        logits, loss = model(xs, targets=ys)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()

        # If a learning rate scheduler is provided, adjust the learning rate
        if scheduler:
            scheduler.step()

        # Log progress every specified interval
        if epoch % config['log_interval'] == 0:
            # Calculate batch time
            batch_time = time.time() - start_time

            # Evaluate loss on validation set
            x = evaluate_loss(model)

            # Store the validation loss
            losses += [x]

            # Print progress logs if specified
            if print_logs:
                print(f"Epoch {epoch} | val loss {x['val']:.3f} | Time {batch_time:.3f} | ETA in seconds {batch_time * (config['epochs'] - epoch)/config['log_interval'] :.3f}")

            # Reset the timer
            start_time = time.time()

            # Print learning rate if a scheduler is provided
            if scheduler:
                print("lr: ", scheduler.get_lr())

    # Print the final validation loss
    print("Validation loss: ", losses[-1]['val'])

    # Plot the training and validation loss curves
    return pd.DataFrame(losses).plot()

# Execute the training process
train(model, optimizer)






