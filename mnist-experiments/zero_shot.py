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
from base_model import ModelHistory, test, MNIST_FFN, create_dataloader, mnist_transform

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

if __name__ == "__main__":

    test_dataset = MNIST(root="./data", train=False, download=True, transform=mnist_transform)

    criterion = nn.CrossEntropyLoss()
    classes = [i for i in range(5)]
    model = MNIST_FFN(
        hidden_size=512,
        num_layers=4,
        classes=classes
    )
    classes = [i for i in range(5, 10)] # because to check for remaining 5 classes

    # load the state_dict
    model.load_state_dict(torch.load(f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}.pt"))
    model.to(device)

    model.classes = classes
    model.num_classes = len(classes)
    model.min_class = min(classes)

    zero_shot_history = test(
        model,
        test_dataset,
        batch_size=64,
        history=ModelHistory(
            trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            total_params=sum(p.numel() for p in model.parameters()),
            results=[],
            test_loss=None,
            test_accuracy=None
        ),
        criterion=criterion,
        device=device
    )

    with open(f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_FFN_history_zero_shot.pkl", "wb") as f:
        pickle.dump(zero_shot_history, f)

    print("Done!")
