import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
import numpy as np
from collections import namedtuple
from tqdm import tqdm
import pickle
import os

# Define the ModelHistory namedtuple
ModelHistory = namedtuple("ModelHistory", ["trainable_params", "total_params", "results", "test_loss", "test_accuracy"])
ExtendedModelHistory = namedtuple("ExtendedModelHistory", ModelHistory._fields + ("classes_trained_on", "classes_inferred_on"))

cifar_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

# Custom Dataset to filter the CIFAR100 dataset for specific classes
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

# Function to create a DataLoader for a filtered dataset
def create_dataloader(dataset, batch_size, classes, shuffle=True, num_workers=2):
    filtered_dataset = FilteredDataset(dataset, classes)
    return DataLoader(filtered_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

# Fit function to train the model with logging every `print_every` epochs
def fit(model, train_dataset, validation_dataset, epochs, batch_size, optimizer, criterion, device, classes, patience=10, print_every=10):
    train_dataloader = create_dataloader(train_dataset, batch_size, classes=classes)
    validation_dataloader = create_dataloader(validation_dataset, batch_size, classes=classes, shuffle=False)

    history = ModelHistory(
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        total_params=sum(p.numel() for p in model.parameters()),
        results=[],
        test_loss=None,
        test_accuracy=None
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Using tqdm for progress tracking
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", leave=False) as pbar:
            for i, (inputs, labels) in enumerate(train_dataloader):
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    "Train Loss": running_loss / (i + 1),
                    "Train Acc": 100 * correct / total
                })
                pbar.update(1)

        # Calculate validation loss and accuracy
        val_loss, val_accuracy = validate(model, validation_dataloader, criterion, device)

        # Log every `print_every` epochs
        if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
            avg_loss = running_loss / len(train_dataloader)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Save the results in history
        history.results.append((epoch + 1, val_loss, val_accuracy))

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()  # Save the best model state
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history

# Validation function to evaluate model on the validation dataset with tqdm
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with tqdm(total=len(dataloader), desc="Validating", unit="batch", leave=False) as pbar:
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.update(1)

    val_loss /= len(dataloader)
    val_accuracy = 100 * correct / total

    return val_loss, val_accuracy

# Test function to evaluate the model on the test dataset with tqdm
def test(model, test_dataset, criterion, device, classes, history):
    test_dataloader = create_dataloader(test_dataset, batch_size=32, classes=classes, shuffle=False)
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with tqdm(total=len(test_dataloader), desc="Testing", unit="batch") as pbar:
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.update(1)

    test_loss /= len(test_dataloader)
    test_accuracy = 100 * correct / total

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Update history
    history = history._replace(test_loss=test_loss, test_accuracy=test_accuracy)

    return history

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load datasets
    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=cifar_transform)
    test_dataset = CIFAR100(root="./data", train=False, download=True, transform=cifar_transform)

    # Split train dataset into training and validation datasets
    generator = torch.Generator().manual_seed(42)
    train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000], generator=generator)

    # Select 20 random classes from CIFAR100
    n_classes = 20
    CLASSES = [i for i in range(100)]
    np.random.seed(42)
    classes = np.random.choice(CLASSES, n_classes, replace=False)

    # Set up the model
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, n_classes)
    )
    model.classes_trained_on = classes     # Add attribute to store classes trained on
    model.to(device)

    # Define optimizer and loss criterion
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    history = fit(model, train_dataset, validation_dataset, epochs=100, batch_size=128, optimizer=optimizer, criterion=criterion, device=device, classes=classes, print_every=10)

    # Test the model
    history = test(model, test_dataset, criterion, device, classes, history)
    # exteneding the ModelHistory tuple to include classes_trained_on and classes_inferred_on
    history = ExtendedModelHistory(*history, classes_trained_on=classes, classes_inferred_on=classes)

    # Display trained classes
    print(f"Model trained on classes: {model.classes_trained_on}")

    # Create output directory if it does not exist
    os.makedirs('./out_files', exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), './out_files/resnet50_base_model.pt')
    print("Model saved to ./out_files/resnet50_base_model.pt")

    with open("./out_files/resnet50_base_model_metadata.pkl", "wb") as f:
        pickle.dump({'classes_trained_on': model.classes_trained_on}, f)

    # Save the model history
    with open('./out_files/resnet50_base_model_hist.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("Model history saved to ./out_files/resnet50_base_model_hist.pkl")
