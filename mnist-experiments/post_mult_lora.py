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

# Post-Mult-LoRA ######################

class MNIST_FFN_Post_Mult_LoRA(nn.Module):
    def __init__(self, hidden_size, num_layers, classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.classes = classes
        self.num_classes = len(classes)
        self.min_class = min(classes)
        
        self.dims = (1, 28, 28)
        
        channels, width, height = self.dims
        
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

        # Mult-LoRA params
        self.lin_mult_lora_A = nn.Parameter(torch.eye(self.hidden_size))
        self.mult_lora_As = nn.ParameterList()
        for i in range(self.num_layers):
            self.mult_lora_As.append(nn.Parameter(torch.eye(self.hidden_size)))
        
        self.lout_mult_lora_A = nn.Parameter(torch.eye(self.num_classes))

        self.init_parameters()

    def init_parameters(self):
        for n, p in self.named_parameters():
            if "mult_lora" not in n:
                p.requires_grad = False
    
    def mult_lora_linear(self, x, layer, mult_lora_A):
        # Post-multiplication: apply LoRA matrix after the linear layer
        h = (layer(x) @ mult_lora_A)
        return h
    
    def forward(self, x):
        x = torch.flatten(x, 1)

        x = self.mult_lora_linear(x, self.lin, self.lin_mult_lora_A)
        x = self.relu(x)
        x = self.lnorm_in(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.mult_lora_linear(x, self.layers[i], self.mult_lora_As[i])
            x = self.relu(x)
            x = self.lnorms[i](x)
            x = self.dropout(x)

        x = self.mult_lora_linear(x, self.lout, self.lout_mult_lora_A)
        x = self.softmax(x)
        return x
    
def post_mult_lora_experiment(
        hidden_size,
        num_layers,
        original_classes,
        new_classes,
        train_dataset,
        validation_dataset,
        test_dataset,
        criterion
):
    model = MNIST_FFN_Post_Mult_LoRA(hidden_size, num_layers, original_classes)
    model.load_state_dict(torch.load(f"./out_files/MNIST_{hidden_size}_{num_layers}_FFN.pt"), strict=False)
    model.to(device)
    model.classes = new_classes
    model.min_class = min(model.classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    print("Post-Mult-LoRA full rank experiment")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"#trainable_params: {trainable_params} and #total_params: {total_params}")

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

    history = test(
        model,
        test_dataset,
        64,
        history,
        criterion,
        device
    )

    with open(f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_post_mult_lora_FFN_history.pkl", "wb") as f:
        pickle.dump(history, f)

    torch.save(model.state_dict(), f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_post_mult_lora_FFN.pt")

    return history   

if __name__ == "__main__":
    test_dataset = MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    train_dataset = MNIST(root="./data", train=True, download=True, transform=mnist_transform)

    generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [45000, 15000], generator=generator)

    original_classes = [i for i in range(5)]
    new_classes = [i for i in range(5, 10)]

    criterion = nn.CrossEntropyLoss()

    post_mult_lora_experiment(512, 4, original_classes, new_classes, train_dataset, validation_dataset, test_dataset, criterion)

    print("Done!")
