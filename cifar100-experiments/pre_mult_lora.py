import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from torchvision.models import resnet50
from base_model import ModelHistory, ExtendedModelHistory, FilteredDataset, create_dataloader, fit, validate, test, cifar_transform
from torchvision.datasets import CIFAR100
import pickle

class PreMult_LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=4):
        super(PreMult_LoRALayer, self).__init__()
        self.base_layer = base_layer
        self.rank = rank

        if isinstance(base_layer, nn.Linear):
            out_features, in_features = base_layer.weight.shape

            self.B = nn.Parameter(
                base_layer.weight.new_zeros((out_features, self.rank))
            )
            self.A = nn.Parameter(
                base_layer.weight.new_zeros((self.rank, out_features))
            )

            self.base_layer.weight.requires_grad = False

        elif isinstance(base_layer, nn.Conv2d):
            in_channels = base_layer.in_channels
            out_channels = base_layer.out_channels
            kernel_size = base_layer.kernel_size[0] if isinstance(base_layer.kernel_size, tuple) else base_layer.kernel_size

            # Adjust B and A dimensions for convolutional layer
            self.B = nn.Parameter(
                base_layer.weight.new_zeros((out_channels, self.rank * kernel_size * kernel_size))
            )
            self.A = nn.Parameter(
                base_layer.weight.new_zeros((self.rank * kernel_size * kernel_size, out_channels))
            )

            self.base_layer.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.B)
        nn.init.zeros_(self.A)
        with torch.no_grad():
            self.A.data.copy_(torch.linalg.pinv(self.B.data))

    def forward(self, x):
        if isinstance(self.base_layer, nn.Linear):
            # Multiply BAW
            adapted_weight = torch.matmul(torch.matmul(self.B, self.A), self.base_layer.weight)
            return F.linear(x, adapted_weight, self.base_layer.bias)
        elif isinstance(self.base_layer, nn.Conv2d):
            # Reshape weights to (out_channels, in_channels * kernel_size * kernel_size)
            weight_reshaped = self.base_layer.weight.view(self.base_layer.weight.size(0), -1)
            
            # Multiply BA with the reshaped weights
            adapted_weight = torch.matmul(torch.matmul(self.B, self.A), weight_reshaped)
            
            # Reshape back to (out_channels, in_channels, kernel_size, kernel_size)
            adapted_weight = adapted_weight.view(self.base_layer.weight.shape)
            
            return self.base_layer._conv_forward(
                x,
                adapted_weight,
                self.base_layer.bias
            )
        else:
            raise ValueError("PreMult_LoRALayer only supports nn.Linear and nn.Conv2d layers")

def apply_pre_mult_lora(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            setattr(model, name, PreMult_LoRALayer(module, rank=rank))
        else:
            apply_pre_mult_lora(module, rank=rank)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # load the cifar100 dataset
    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=cifar_transform)
    test_dataset = CIFAR100(root="./data", train=False, download=True, transform=cifar_transform)

    # Split train dataset into training and validation datasets
    generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000], generator=generator)

    n_classes = 20

    with open("./out_files/resnet50_base_model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
        classes_trained_on = metadata["classes_trained_on"]
    
    print(f"Model trained on: {classes_trained_on}")

    CLASSES = [i for i in range(100)]
    np.random.seed(111)
    classes = np.random.choice(CLASSES, n_classes, replace=False)

    print(f"Model finetuned using modified LoRA on: {classes}")

    ranks = [1, 2, 4, 8, 16, 32, 64]
    histories = {}

    for rank in ranks:
        print(f"Applying modified LoRA for rank = {rank}")

        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

        model.load_state_dict(torch.load("./out_files/resnet50_base_model.pt"))

        apply_pre_mult_lora(model, rank=rank)
        model.to(device)

        # trainable v/s total params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"For Pre Mult LoRA rank: {rank}, #trainable_params: {trainable_params} and #total_params: {total_params}")

        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        history = fit(model, train_dataset, validation_dataset, epochs=100, batch_size=128, optimizer=optimizer, criterion=criterion, device=device, classes=classes, print_every=10)

        history = test(model, test_dataset, criterion, device, classes, history)
        history = ExtendedModelHistory(*history, classes_trained_on=classes_trained_on, classes_inferred_on=classes)

        histories[rank] = history

        torch.save(model.state_dict(), f"./out_files/resnet50_Pre_Mult_LoRA_model_rank_{rank}.pt")
        print(f"Model saved to ./out_files/resnet50_Pre_Mult_LoRA_model_rank_{rank}.pt")

    with open("./out_files/resnet50_Pre_Mult_LoRA_model_histories.pkl", "wb") as f:
        pickle.dump(histories, f)

    print("Histories saved to ./out_files/resnet50_Pre_Mult_LoRA_model_histories.pkl")
