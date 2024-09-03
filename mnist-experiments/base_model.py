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

# DEVICE
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


ModelHistory = namedtuple("ModelHistory", ["trainable_params", "total_params", "results", "test_loss", "test_accuracy"])

# DATASET #######################
mnist_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
)

train_dataset = MNIST(root="./data", train=True, download=True, transform=mnist_transform)
test_dataset = MNIST(root="./data", train=False, download=True, transform=mnist_transform)

print(f"len(train_dataset): {len(train_dataset)}, len(test_dataset): {len(test_dataset)}")

CLASSES = {}
for i in range(len(train_dataset.classes)):
    CLASSES[i] = train_dataset.classes[i]

print(f"CLASSES: {CLASSES}")

generator = torch.Generator().manual_seed(42)
train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [45000, 15000], generator=generator)

class FilteredDataset(Dataset):
    def __init__(self, dataset, classes):
        self.dataset = dataset
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.indices = [i for i, (_, label) in enumerate(dataset) if label in self.class_to_idx]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        image, label = self.dataset[actual_idx]
        return image, self.class_to_idx[label]

def create_dataloader(dataset, batch_size, classes, shuffle=True, num_workers=2):
    filtered_dataset = FilteredDataset(dataset, classes)
    return DataLoader(filtered_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

####################################

# BASE MODEL #######################

class MNIST_FFN(nn.Module):
    def __init__(self, hidden_size, num_layers, classes):
        # NOTE! the classes list has to be contiguous
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.classes = classes
        self.num_classes = len(classes)
        self.min_class = min(classes)
        
        self.dims = (1,28,28)
        
        channels, width, height = self.dims
        
        self.lin = nn.Linear(channels*width*height, self.hidden_size)
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
        
    
    def forward(self, x):
        # x is a batch
        x = torch.flatten(x, 1)

        x = self.lin(x)
        x = self.lnorm_in(x)
        x = self.relu(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.layers[i](x)
            x = self.lnorms[i](x)
            x = self.relu(x)
            x = self.dropout(x)

        x = self.lout(x)
        x = self.softmax(x)
        return x

# TRAINING #######################

def fit(model, train_dataset, validation_dataset, epochs, batch_size, optimizer, criterion, device=device, patience=5, print_every=10):
    train_loader = create_dataloader(train_dataset, batch_size, model.classes)
    validation_loader = create_dataloader(validation_dataset, batch_size, model.classes)

    history = ModelHistory(
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
        results=[],
        test_loss=None,
        test_accuracy=None
    )

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = labels - model.min_class

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)

            train_loader_tqdm.set_postfix(loss=running_loss / (total_train / batch_size), accuracy=correct_train / total_train)

        avg_train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        validation_loader_tqdm = tqdm(validation_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in validation_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                # labels = labels - model.min_class

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)

                validation_loader_tqdm.set_postfix(val_loss=val_loss / (total_val / batch_size), val_accuracy=correct_val / total_val)

        avg_val_loss = val_loss / len(validation_loader)
        val_accuracy = correct_val / total_val

        result = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': avg_val_loss,
            'val_accuracy': val_accuracy
        }
        history.results.append(result)

        # Print results after every 'print_every' epochs
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return history

# EVALUATION #######################

def test(model, test_dataset, batch_size, history, criterion, device=device):
    test_loader = create_dataloader(test_dataset, batch_size, model.classes, shuffle=False)
    model.eval()
    model.to(device)

    test_loss = 0.0
    correct_test = 0
    total_test = 0

    test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for inputs, labels in test_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            # labels = labels - model.min_class

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

            test_loader_tqdm.set_postfix(test_loss=test_loss / (total_test / batch_size), test_accuracy=correct_test / total_test)

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_test / total_test

    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    history = history._replace(test_loss=avg_test_loss, test_accuracy=test_accuracy)
    return history

####################################

if __name__ == "__main__":
    HIDDEN_SIZE = 512
    NUM_LAYERS = 4
    criterion = nn.CrossEntropyLoss()
    classes = [i for i in range(5)]

    model = MNIST_FFN(
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        classes=classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    history = fit(
        model,
        train_dataset,
        validation_dataset,
        epochs=100,
        batch_size=128,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=10,
        print_every=10
    )

    history = test(
        model,
        test_dataset,
        batch_size=128,
        history=history,
        criterion=criterion,
        device=device
    )

    with open(f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_FFN_history.pkl", "wb") as f:
        pickle.dump(history, f)

    torch.save(model.state_dict(), f"./out_files/MNIST_{model.hidden_size}_{model.num_layers}_FFN.pt")

    print("Done!")

