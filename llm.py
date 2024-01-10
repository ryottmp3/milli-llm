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
    'batch_size': 8,          # Number of batches to be processed at each random split
    'context_window': 16,     # Number of characters in each input (x) and target (y) sequence of each batch
    'vocab_size': 65,
    'd_model': 128,
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

print(f'params: {sum([m.numel() for m in model.parameters()])}')


















# Obtain batches for training using the specified batch size and context window
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# Decode the sequences to obtain the corresponding text representations
decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

# Print the random sample
print(decoded_samples)

logits, loss = model(xs,ys)

print(f'logits: {logits}')
print(f'loss: {loss}')













