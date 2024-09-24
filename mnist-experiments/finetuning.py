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

if __name__ == "__main__":
    test_dataset = MNIST(root="./data", train=False, download=True, transform=mnist_transform)
    train_dataset = MNIST(root="./data", train=True, download=True, transform=mnist_transform)

    generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [45000, 15000], generator=generator)

    model = MNIST_FFN(
        hidden_size=512,
        num_layers=4,
        classes=[i for i in range(5)]
    )
    model.load_state_dict(torch.load(f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_FFN.pt"))
    model.to(device)

    model.classes = [i for i in range(5, 10)]
    model.num_classes = len(model.classes)
    model.min_class = min(model.classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.CrossEntropyLoss()

    ft_history = fit(
        model,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        epochs=200,
        batch_size=128,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=10,
        print_every=10,
    )

    ft_history = test(
        model,
        test_dataset,
        64,
        ft_history,
        criterion,
        device
    )

    with open(f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_FFN_history_finetuned.pkl", "wb") as f:
        pickle.dump(ft_history, f)

    torch.save(model.state_dict(), f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_FFN_finetuned.pt")

    print("Done!")

