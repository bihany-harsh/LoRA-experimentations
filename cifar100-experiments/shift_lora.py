import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from torchvision.models import resnet50
from base_model import ModelHistory, ExtendedModelHistory, FilteredDataset, create_dataloader, fit, validate, test, cifar_transform
from torchvision.datasets import CIFAR100
import pickle

class ShiftLoRALayer(nn.Module):
    def __init__(self, base_layer, rank=4, fan_in_fan_out=False):
        super(ShiftLoRALayer, self).__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.fan_in_fan_out = fan_in_fan_out

        # computing shift indices
        self.shifting_matrices = {}

        if isinstance(base_layer, nn.Linear):
            if self.fan_in_fan_out:
                in_features, out_features = base_layer.weight.shape
            else:
                out_features, in_features = base_layer.weight.shape
            self.lora_A = nn.Parameter(
                torch.empty(( self.rank, out_features ), dtype=base_layer.weight.dtype)
            )
            self.lora_B = nn.Parameter(
                torch.empty(( in_features, self.rank ), dtype=base_layer.weight.dtype)
            )
            self.base_layer.weight.requires_grad = False
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad = False
        elif isinstance(base_layer, nn.Conv2d):
            in_channels = base_layer.in_channels
            out_channels = base_layer.out_channels

            # Assumption: kernel_size and stride, if tuples, are of the form (a, a)
            kernel_size = base_layer.kernel_size
            if isinstance(kernel_size, tuple):
                kernel_size = kernel_size[0]
            stride = base_layer.stride
            if isinstance(stride, tuple):
                stride = stride[0]

            self.lora_A = nn.Parameter(
                base_layer.weight.new_zeros(( self.rank*kernel_size, in_channels*kernel_size ))
            )
            self.lora_B = nn.Parameter(
                base_layer.weight.new_zeros(( out_channels//base_layer.groups*kernel_size, self.rank*kernel_size ))
            )

            self.base_layer.weight.requires_grad = False

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def shift(self, M):
        shape = M.shape
        if shape not in self.shifting_matrices:
            n_rows, n_cols = shape
            row_indices = torch.arange(n_rows, device=M.device).view(-1, 1)
            col_indices = torch.arange(n_cols, device=M.device).view(1, -1)
            # Compute shift indices using broadcasting and modulo operation
            indices = (col_indices - row_indices) % n_cols
            self.shifting_matrices[shape] = indices
        return torch.gather(M, 1, self.shifting_matrices[shape])

    def forward(self, x):
        if isinstance(self.base_layer, nn.Linear):
            return F.linear(x, 
                            self.base_layer.weight + self.shift(self.lora_B @ self.lora_A).T, 
                            self.base_layer.bias
                        )
        elif isinstance(self.base_layer, nn.Conv2d):
            return self.base_layer._conv_forward(
                x,
                self.base_layer.weight + self.shift(self.lora_B @ self.lora_A).view(self.base_layer.weight.shape),
                self.base_layer.bias
            )
        else:
            raise ValueError("LoRALayer only supports nn.Linear and nn.Conv2d layers")
        
def apply_lora(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            setattr(model, name, ShiftLoRALayer(module, rank=rank))
        else:
            apply_lora(module, rank=rank)

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

    print(f"Model finetuned using LoRA on: {classes}")

    ranks = [1, 2, 4, 8, 16, 32, 64]
    histories = {}

    for rank in ranks:
        print(f"Applying LoRA for rank = {rank}")

        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

        model.load_state_dict(torch.load("./out_files/resnet50_base_model.pt"))

        apply_lora(model, rank=rank)
        model.to(device)

        # trainable v/s total params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"For LoRA rank: {rank}, #trainable_params: {trainable_params} and #total_params: {total_params}")

        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        history = fit(model, train_dataset, validation_dataset, epochs=100, batch_size=128, optimizer=optimizer, criterion=criterion, device=device, classes=classes, print_every=10)

        history = test(model, test_dataset, criterion, device, classes, history)
        history = ExtendedModelHistory(*history, classes_trained_on=classes_trained_on, classes_inferred_on=classes)

        histories[rank] = history

        torch.save(model.state_dict(), f"./out_files/resnet50_Shift_LoRA_model_rank_{rank}.pt")
        print(f"Model saved to ./out_files/resnet50_Shift_LoRA_model_rank_{rank}.pt")

    with open("./out_files/resnet50_Shift_LoRA_model_histories.pkl", "wb") as f:
        pickle.dump(histories, f)

    print("Histories saved to ./out_files/resnet50_Shift_LoRA_model_histories.pkl")
