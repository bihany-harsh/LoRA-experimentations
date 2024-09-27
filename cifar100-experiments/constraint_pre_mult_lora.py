import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision.models import resnet50
from base_model import ModelHistory, ExtendedModelHistory, FilteredDataset, create_dataloader, validate, test, cifar_transform
from torchvision.datasets import CIFAR100
import pickle
from tqdm import tqdm

class Constraint_PreMult_LoRALayer(nn.Module):
    def __init__(self, base_layer, rank=4):
        super(Constraint_PreMult_LoRALayer, self).__init__()
        self.base_layer = base_layer
        self.rank = rank

        self.kernel_size = None

        if isinstance(base_layer, nn.Linear):
            out_features, in_features = self.base_layer.weight.shape

            # (B_1@A_1 + B_2@A_2)@W
            self.B_1 = nn.Parameter(
                self.base_layer.weight.new_zeros(( out_features, self.rank ))
            )
            self.A_1 = nn.Parameter(
                self.base_layer.weight.new_zeros(( self.rank, out_features ))
            )

            self.B_2 = nn.Parameter(
                self.base_layer.weight.new_zeros(( out_features, self.rank ))
            )
            self.A_2 = nn.Parameter(
                self.base_layer.weight.new_zeros(( self.rank, out_features ))
            )
        elif isinstance(base_layer, nn.Conv2d):
            out_channels = base_layer.out_channels
            kernel_size = base_layer.kernel_size[0] if isinstance(base_layer.kernel_size, tuple) else base_layer.kernel_size

            self.kernel_size = kernel_size

            self.B_1 = nn.Parameter(
                base_layer.weight.new_zeros(( out_channels, self.rank * kernel_size * kernel_size ))
            )
            self.A_1 = nn.Parameter(
                base_layer.weight.new_zeros(( self.rank * kernel_size * kernel_size, out_channels ))
            )

            self.B_2 = nn.Parameter(
                base_layer.weight.new_zeros(( out_channels, self.rank * kernel_size * kernel_size ))
            )
            self.A_2 = nn.Parameter(
                base_layer.weight.new_zeros(( self.rank * kernel_size * kernel_size, out_channels ))
            )

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
        
        self.reset_parameters()

    def reset_parameters(self):
        out_features = self.B_1.size(0)
        if self.kernel_size is not None:
            combined_B = nn.init.orthogonal_(torch.empty((out_features, 2 * self.rank * self.kernel_size * self.kernel_size), device=self.B_1.device, dtype=self.B_1.dtype))
        else:
            combined_B = nn.init.orthogonal_(torch.empty((out_features, 2 * self.rank), device=self.B_1.device, dtype=self.B_1.dtype))
            

        with torch.no_grad():
            if self.kernel_size is not None:
                self.B_1.data = combined_B[:, :self.rank*self.kernel_size*self.kernel_size]
                self.B_2.data = combined_B[:, self.rank*self.kernel_size*self.kernel_size:]
            else:    
                self.B_1.data = combined_B[:, :self.rank]
                self.B_2.data = combined_B[:, self.rank:]

        nn.init.zeros_(self.A_1)
        nn.init.zeros_(self.A_2)

        with torch.no_grad():
            self.A_1.data.copy_(torch.linalg.pinv(self.B_1.data))
            self.A_2.data.copy_(torch.linalg.pinv(self.B_2.data))


    def forward(self, x):
        if isinstance(self.base_layer, nn.Linear):
            # B_1 @ (A_1 @ W) + B_2 @ (A_2 @ W)
            adapted_weight = self.B_1 @ (self.A_1 @ self.base_layer.weight) + self.B_2 @ (self.A_2 @ self.base_layer.weight)

            return F.linear(x, adapted_weight, self.base_layer.bias)
        elif isinstance(self.base_layer, nn.Conv2d):
            # Reshape weights to (out_channels, in_channels * kernel_size * kernel_size)
            weight_reshaped = self.base_layer.weight.view(self.base_layer.weight.size(0), -1)
            
            # B_1 @ (A_1 @ W) + B_2 @ (A_2 @ W)
            adapted_weight = self.B_1 @ (self.A_1 @ weight_reshaped) + self.B_2 @ (self.A_2 @ weight_reshaped)            
            # Reshape back to (out_channels, in_channels, kernel_size, kernel_size)
            adapted_weight = adapted_weight.view(self.base_layer.weight.shape)
            
            return self.base_layer._conv_forward(
                x,
                adapted_weight,
                self.base_layer.bias
            )
        else:
            raise ValueError("Constraint_PreMult_LoRALayer only supports nn.Linear and nn.Conv2d layers")
        

    def constraint_loss(self):
        # Calculate ||(B_1^T B_2 - I||_f - frobenius norm
        # B1A1 = torch.matmul(self.B_1, self.A_1)
        # B2A2 = torch.matmul(self.B_2, self.A_2)
        # identity = torch.eye(self.rank, device=B1A1.device)
        
        # product = torch.matmul(B1A1.T, B2A2)
        # frobenius_norm = torch.norm(product - identity, p='fro')
        
        if isinstance(self.base_layer, nn.Linear):
            dim = self.B_1.size(1)
            I = torch.eye(dim, device=self.B_1.device)
            fro_norm = torch.norm((self.B_1.T @ self.B_2) - I, p="fro")
        elif isinstance(self.base_layer, nn.Conv2d):
            dim = self.B_1.size(0)
            I = torch.eye(dim, device=self.B_1.device)
            fro_norm = torch.norm((self.B_1 @ self.B_2.T) - I, p="fro")

        return fro_norm
    
def apply_constraint_pre_mult_lora(model, rank=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            setattr(model, name, Constraint_PreMult_LoRALayer(module, rank=rank))
        else:
            apply_constraint_pre_mult_lora(module, rank=rank)

# Fit function with constrained loss
def constraint_fit(model, train_dataset, validation_dataset, epochs, batch_size, optimizer, criterion, device, classes, patience=10, print_every=10):
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

                # Add constraint loss
                constraint_loss = 0.0
                for module in model.modules():
                    if isinstance(module, Constraint_PreMult_LoRALayer):
                        constraint_loss += module.constraint_loss()

                total_loss = loss + constraint_loss  # Combine the primary loss with constraint loss

                # Backward pass and optimize
                total_loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    "Train Loss": running_loss / (i + 1),
                    "Constraint Loss": constraint_loss.item(),
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

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load the cifar100 dataset
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

        apply_constraint_pre_mult_lora(model, rank=rank)  # Use the new function for constraint LoRA
        model.to(device)

        # Trainable vs total params
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"For Constraint Pre Mult LoRA rank: {rank}, #trainable_params: {trainable_params} and #total_params: {total_params}")

        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()

        history = constraint_fit(model, train_dataset, validation_dataset, epochs=500, batch_size=128, optimizer=optimizer, criterion=criterion, device=device, classes=classes, print_every=10)

        history = test(model, test_dataset, criterion, device, classes, history)
        history = ExtendedModelHistory(*history, classes_trained_on=classes_trained_on, classes_inferred_on=classes)

        histories[rank] = history

        torch.save(model.state_dict(), f"./out_files/resnet50_Constraint_Pre_Mult_LoRA_model_rank_{rank}.pt")
        print(f"Model saved to ./out_files/resnet50_Constraint_Pre_Mult_LoRA_model_rank_{rank}.pt")

    with open("./out_files/resnet50_Constraint_Pre_Mult_LoRA_model_histories.pkl", "wb") as f:
        pickle.dump(histories, f)

    print("Histories saved to ./out_files/resnet50_Constraint_Pre_Mult_LoRA_model_histories.pkl")

