import torch
import torch.nn as nn
import random
import numpy as np
from torchvision.models import resnet50
from base_model import ModelHistory, ExtendedModelHistory, FilteredDataset, create_dataloader, fit, validate, test, cifar_transform
from torchvision.datasets import CIFAR100
import pickle

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CIFAR100 dataset
    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=cifar_transform)
    test_dataset = CIFAR100(root="./data", train=False, download=True, transform=cifar_transform)

    # Split train dataset into training and validation datasets
    generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000], generator=generator)

    # Load the model
    n_classes = 20
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, n_classes)
    )
    model.load_state_dict(torch.load("./out_files/resnet50_base_model.pt"))
    model.to(device)

    # Load the metadata for classes_trained_on
    with open("./out_files/resnet50_base_model_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
        classes_trained_on = metadata["classes_trained_on"]

    print(f"Model trained on: {classes_trained_on}")

    # A new random set of classes (consistent with the seed)
    CLASSES = [i for i in range(100)]
    np.random.seed(111)
    classes = np.random.choice(CLASSES, n_classes, replace=False)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Initialize history
    history = ModelHistory(
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
        results=[],
        test_loss=None,
        test_accuracy=None
    )

    print(f"New classes: {classes}")
    
    # Test the model on new classes
    history = test(model, test_dataset, criterion, device, classes, history)

    # Update the history with additional metadata
    history = ExtendedModelHistory(
        *history, 
        classes_trained_on=classes_trained_on, 
        classes_inferred_on=classes
    )

    # Save the results
    with open("./out_files/resnet50_zero_shot_history.pkl", "wb") as f:
        pickle.dump(history, f)

    print("Done!")
