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























# Obtain batches for training using the specified batch size and context window
xs, ys = get_batches(dataset, 'train', MASTER_CONFIG['batch_size'], MASTER_CONFIG['context_window'])

# Decode the sequences to obtain the corresponding text representations
decoded_samples = [(decode(xs[i].tolist()), decode(ys[i].tolist())) for i in range(len(xs))]

# Print the random sample
print(decoded_samples)















