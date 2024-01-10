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
    # Will be updated later
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
    return [itos[i] for i in l]

# Convert data to tensor
dataset = torch.tensor(encode(lines), dtype=torch.int8)

print(dataset.shape)












































