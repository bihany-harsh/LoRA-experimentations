import torch
import torch.nn as nn
from torch.nn import functional as F

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset, Dataset

import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple

from tqdm import tqdm
import pickle

## importing from base_model.py
from base_model import ModelHistory, fit, test, MNIST_FFN, create_dataloader, mnist_transform

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Pre-Mult-LoRA-Factarizable ###################

class MNIST_FFN_Pre_Mult_LoRA_Fact(nn.Module):
    def __init__(self, hidden_size, num_layers, classes, lora_rank=4):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.classes = classes
        self.num_classes = len(classes)
        self.min_class = min(classes)
        
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        
        # Linear layers
        self.lin = nn.Linear(channels * width * height, self.hidden_size)
        self.lnorm_in = nn.LayerNorm(self.hidden_size)
        self.layers = nn.ModuleList()
        self.lnorms = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.lnorms.append(nn.LayerNorm(self.hidden_size))
        self.lout = nn.Linear(self.hidden_size, self.num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

        self.softmax = nn.Softmax(dim=1)

        self.pre_mult_lora_rank = lora_rank

        # Full Matrix LoRA Transform Params for each layer
        self.lin_mt_lora_A = nn.Parameter(torch.empty(channels * width * height, self.pre_mult_lora_rank))
        self.lin_mt_lora_B = nn.Parameter(torch.empty(self.pre_mult_lora_rank, channels * width * height))

        self.mt_lora_As = nn.ParameterList()
        self.mt_lora_Bs = nn.ParameterList()
        for i in range(self.num_layers):
            self.mt_lora_As.append(nn.Parameter(torch.empty(self.hidden_size, self.pre_mult_lora_rank)))
            self.mt_lora_Bs.append(nn.Parameter(torch.empty(self.pre_mult_lora_rank, self.hidden_size)))

        self.lout_mt_lora_A = nn.Parameter(torch.empty(self.hidden_size, self.pre_mult_lora_rank))
        self.lout_mt_lora_B = nn.Parameter(torch.empty(self.pre_mult_lora_rank, self.hidden_size))
        
        # Shifting Matrices
        self.shifting_matrices = {}

        self.initialize_lora_parameters()

        # Freeze non-LoRA parameters
        for n, p in self.named_parameters():
            if "mt_lora" not in n:
                p.requires_grad = False

    def initialize_lora_parameters(self):
        
        # Helper function to initialize shift(A@B) to identity
        def init_lora_matrices(A, B):
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            with torch.no_grad():
                A[:,0] = 1
                B[:,:] = 0
                B[0][0] = 1

        # Initialize all LoRA matrices
        init_lora_matrices(self.lin_mt_lora_A, self.lin_mt_lora_B)
        for A, B in zip(self.mt_lora_As, self.mt_lora_Bs):
            init_lora_matrices(A, B)
        init_lora_matrices(self.lout_mt_lora_A, self.lout_mt_lora_B)
    
    def shift(self, M):
        indices = None
        shape = M.shape
        if shape in self.shifting_matrices:
            indices = self.shifting_matrices[shape]
        else:
            n_rows, n_cols = shape
            indices = torch.zeros([n_rows,n_cols],device=device,dtype=torch.int64)
            for i in range(n_rows):
                for j in range(n_cols):
                    indices[i][j] = ((j - i) % n_cols)
            self.shifting_matrices[shape] = indices
        return torch.gather(M, 1, indices)

    def pre_mult_lora_linear(self, x, layer, mt_lora_A, mt_lora_B):
        
        # Apply LoRA transformation: pre-multiplication with full matrix LoRA
        x_transformed = x @ (self.shift(mt_lora_A @ mt_lora_B))  # Full Matrix LoRA transformation
        h = layer(x_transformed)
        return h
    
    def forward(self, x):
        x = torch.flatten(x, 1)

        # Input layer with pre-multiplication LoRA
        x = self.pre_mult_lora_linear(x, self.lin, self.lin_mt_lora_A, self.lin_mt_lora_B)
        x = self.relu(x)
        x = self.lnorm_in(x)
        x = self.dropout(x)

        # Hidden layers with pre-multiplication LoRA
        for i in range(self.num_layers):
            x = self.pre_mult_lora_linear(x, self.layers[i], self.mt_lora_As[i], self.mt_lora_Bs[i])
            x = self.relu(x)
            x = self.lnorms[i](x)
            x = self.dropout(x)

        # Output layer with pre-multiplication LoRA
        x = self.pre_mult_lora_linear(x, self.lout, self.lout_mt_lora_A, self.lout_mt_lora_B)
        x = self.softmax(x)
        return x
    
def pre_mult_lora_experiment(
        hidden_size, 
        num_layers, 
        rank, 
        original_classes, 
        new_classes, 
        train_dataset, 
        validation_dataset, 
        test_dataset,
        criterion
    ):
    # Initialize the model with the specified LoRA rank
    model = MNIST_FFN_Pre_Mult_LoRA_Fact(hidden_size, num_layers, original_classes, lora_rank=rank)
    model.load_state_dict(torch.load(f"./out_files/MNIST_{hidden_size}_{num_layers}_FFN.pt"), strict=False)
    model.to(device)
    model.classes = new_classes
    model.min_class = min(model.classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    print("Pre-Mult-LoRA experiment: lora_rank =", rank)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"For Pre-Mult-LoRA rank: {rank}, #trainable_params: {trainable_params} and #total_params: {total_params}")

    # Training the model
    history = fit(
        model,
        train_dataset,
        validation_dataset,
        epochs=200,
        batch_size=64,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=10,
        print_every=10
    )

    # Testing the model
    history = test(
        model,
        test_dataset,
        batch_size=64,
        history=history,
        criterion=criterion,
        device=device
    )

    # Save training history and model state
    with open(f"./out_files/pre_mult_lora/MNIST_{model.hidden_size}_{model.num_layers}_rank_{rank}_pre_mult_lora_FFN_history.pkl", "wb") as f:
        pickle.dump(history, f)

    torch.save(model.state_dict(), f"./out_files/pre_mult_lora/MNIST_{model.hidden_size}_{model.num_layers}_rank_{rank}_pre_mult_lora_FFN.pt")

    return history

if __name__ == "__main__":
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    # ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ranks = [2**i for i in range(int(math.log2(HIDDEN_SIZE))) + 1]

    histories = {}

    # Load the MNIST dataset with transformations
    mnist_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    train_dataset = MNIST(root="./data", train=True, download=True, transform=mnist_transform)

    # Split the training dataset into training and validation sets
    generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [45000, 15000], generator=generator)

    # Define the original and new classes for the experiment
    original_classes = [i for i in range(5)]
    new_classes = [i for i in range(5, 10)]

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Run experiments for different ranks
    for rank in ranks:
        hist = pre_mult_lora_experiment(
            HIDDEN_SIZE,
            NUM_LAYERS,
            rank,
            original_classes,
            new_classes,
            train_dataset,
            validation_dataset,
            test_dataset,
            criterion
        )

        histories[rank] = hist

    # Save the results of all experiments
    with open(f"./out_files/MNIST_{HIDDEN_SIZE}_{NUM_LAYERS}_pre_mult_lora_experiment.pkl", "wb") as f:
        pickle.dump(histories, f)

    print("Done!")

