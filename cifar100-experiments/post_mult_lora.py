import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from torchvision.models import resnet50
from base_model import (
    ModelHistory,
    ExtendedModelHistory,
    FilteredDataset,
    create_dataloader,
    fit,
    validate,
    test,
    cifar_transform
)
from torchvision.datasets import CIFAR100
import pickle

class PostMult_LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=4):
        super(PostMult_LoRALayer, self).__init__()
        self.base_layer = base_layer
        self.rank = rank

        if isinstance(base_layer, nn.Linear):
            out_features, in_features = base_layer.weight.shape

            # Define A and B with correct dimensions for Post-Multiplication LoRA
            self.A = nn.Parameter(
                torch.zeros(in_features, self.rank, device=base_layer.weight.device)
            )
            self.B = nn.Parameter(
                torch.zeros(self.rank, in_features, device=base_layer.weight.device)
            )

            # Freeze the base layer's weights and bias
            base_layer.weight.requires_grad = False
            if base_layer.bias is not None:
                base_layer.bias.requires_grad = False

        elif isinstance(base_layer, nn.Conv2d):
            in_channels = base_layer.in_channels
            out_channels = base_layer.out_channels
            kernel_size = base_layer.kernel_size[0] if isinstance(base_layer.kernel_size, tuple) else base_layer.kernel_size

            # Define A and B with correct dimensions for Post-Multiplication LoRA
            self.A = nn.Parameter(
                torch.zeros(in_channels * kernel_size * kernel_size, self.rank, device=base_layer.weight.device)
            )
            self.B = nn.Parameter(
                torch.zeros(self.rank, in_channels * kernel_size * kernel_size, device=base_layer.weight.device)
            )

            # Freeze the base layer's weights and bias
            base_layer.weight.requires_grad = False
            if base_layer.bias is not None:
                base_layer.bias.requires_grad = False

        else:
            raise ValueError("PostMult_LoRALayer only supports nn.Linear and nn.Conv2d layers")

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.B)
        nn.init.zeros_(self.A)
        with torch.no_grad():
            # Compute the pseudoinverse of B to initialize A
            self.A.data.copy_(torch.linalg.pinv(self.B.data))

    def forward(self, x):
        if isinstance(self.base_layer, nn.Linear):
            # Compute BA
            BA = torch.matmul(self.A, self.B)  # Shape: (in_features, in_features)

            # Adapted weight: W * BA
            adapted_weight = torch.matmul(self.base_layer.weight, BA)  # Shape: (out_features, in_features)

            return F.linear(x, adapted_weight, self.base_layer.bias)

        elif isinstance(self.base_layer, nn.Conv2d):
            # Reshape weights to (out_channels, in_channels * kernel_size * kernel_size)
            W = self.base_layer.weight.view(self.base_layer.out_channels, -1)  # Shape: (out_channels, in_channels * k * k)

            # Compute BA
            BA = torch.matmul(self.A, self.B)  # Shape: (in_channels * k * k, in_channels * k * k)

            # Adapted weight: W * BA
            adapted_weight = torch.matmul(W, BA)  # Shape: (out_channels, in_channels * k * k)

            # Reshape back to (out_channels, in_channels, kernel_size, kernel_size)
            kernel_size = self.base_layer.kernel_size[0] if isinstance(self.base_layer.kernel_size, tuple) else self.base_layer.kernel_size
            adapted_weight = adapted_weight.view_as(self.base_layer.weight)  # Ensure correct shape

            return F.conv2d(
                x,
                adapted_weight,
                self.base_layer.bias,
                stride=self.base_layer.stride,
                padding=self.base_layer.padding,
                dilation=self.base_layer.dilation,
                groups=self.base_layer.groups
            )
        else:
            raise ValueError("PostMult_LoRALayer only supports nn.Linear and nn.Conv2d layers")

def apply_post_mult_lora(model, rank=4):
    """
    Recursively applies PostMult_LoRALayer to all nn.Linear and nn.Conv2d layers in the model.
    """
    for name, module in model.named_children():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            setattr(model, name, PostMult_LoRALayer(module, rank=rank))
        else:
            apply_post_mult_lora(module, rank=rank)

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CIFAR-100 dataset
    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=cifar_transform)
    test_dataset = CIFAR100(root="./data", train=False, download=True, transform=cifar_transform)

    # Split train dataset into training and validation datasets
    generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000], generator=generator)

    n_classes = 20

    # Load metadata
    try:
        with open("./out_files/resnet50_base_model_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            classes_trained_on = metadata.get("classes_trained_on", [])
    except FileNotFoundError:
        classes_trained_on = []
        print("Metadata file not found. Proceeding without metadata.")

    print(f"Model trained on: {classes_trained_on}")

    # Select random classes for fine-tuning
    CLASSES = list(range(100))
    np.random.seed(111)
    classes = np.random.choice(CLASSES, n_classes, replace=False)
    print(f"Model fine-tuned using modified Post-Mult LoRA on classes: {classes}")

    ranks = [1, 2, 4, 8, 16, 32, 64]
    histories = {}

    for rank in ranks:
        print(f"\nApplying modified Post-Multiplication LoRA for rank = {rank}")

        # Initialize the model
        model = resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes)
        )

        # Load pre-trained weights if available
        try:
            model.load_state_dict(torch.load("./out_files/resnet50_base_model.pt", map_location=device))
            print("Loaded base model weights.")
        except FileNotFoundError:
            print("Base model weights not found. Proceeding without loading.")

        # Apply Post-Multiplication LoRA
        apply_post_mult_lora(model, rank=rank)
        model.to(device)

        # Ensure only LoRA parameters are trainable
        for param in model.parameters():
            param.requires_grad = False  # Freeze all parameters initially

        for name, param in model.named_parameters():
            if 'A' in name or 'B' in name:
                param.requires_grad = True  # Unfreeze LoRA parameters

        # Verify parameter counts
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"For Post-Mult LoRA rank {rank}: #trainable_params = {trainable_params}, #total_params = {total_params}")

        # Define optimizer and loss criterion
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        # Train the model
        history = fit(
            model=model,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            epochs=100,
            batch_size=128,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            classes=classes,
            print_every=10
        )

        # Test the model
        history = test(model, test_dataset, criterion, device, classes, history)
        history = ExtendedModelHistory(*history, classes_trained_on=classes_trained_on, classes_inferred_on=classes)

        # Save history
        histories[rank] = history

        # Save the model
        model_save_path = f"./out_files/resnet50_PostMult_LoRA_model_rank_{rank}.pt"
        torch.save(model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")

    # Save all histories
    histories_save_path = "./out_files/resnet50_PostMult_LoRA_model_histories.pkl"
    with open(histories_save_path, "wb") as f:
        pickle.dump(histories, f)
    print(f"Histories saved to {histories_save_path}")
