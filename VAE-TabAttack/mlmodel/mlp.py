"""MLP model architecture for tabular data.
"""

# import utils
import os

from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from collections import OrderedDict

import matplotlib.pyplot as plt


# Create a MLP model for the Adult dataset using PyTorch

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [32, 16],
                 output_dim: int = 2,
                 num_categorical: int = 0,
                 embedding_dims: List[Tuple[int, int]] = None,
                 dropout: float = 0.2
                 ) -> None:
        super(MLP, self).__init__()


        self.num_categorical = num_categorical
        self.embedding_dims = embedding_dims

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories, embedding_dim in embedding_dims
        ])

        # Calculate input dimension for concatenated embeddings
        total_embedding_dim = sum([embedding_dim for _, embedding_dim in embedding_dims])
        self.input_dim_after_embedding = total_embedding_dim + (input_dim - num_categorical)


        self.dropout = dropout

        # Create list to hold all layers
        layers = []
        
        # Input layer
        if num_categorical > 0:
            prev_dim = self.input_dim_after_embedding
        else:
            prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Separate categorical and continuous features
        x_cat = x[:, :self.num_categorical].long()
        x_num = x[:, self.num_categorical:]

        # Embed categorical features
        if self.num_categorical > 0:
            x_cat = [embedding_layer(x_cat[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
            x_cat = torch.cat(x_cat, dim=1)
            x = torch.cat([x_cat, x_num], dim=1)
        else:
            x = x_num

        return self.model(x)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x):
        return torch.argmax(self.predict_proba(x), dim=-1)



def train(model, train_data, val_data, criterion, optimizer, config, wandb_run=None):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!

    # Separate train and validation data
    X_train_tensor, y_train_tensor = train_data
    X_val_tensor, y_val_tensor = val_data

    print(f"Training {config['model']} on {config['dataset']}...")

    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    pbar = tqdm(range(config["epochs"]), desc="Training MLP", unit="epoch")

    for epoch in pbar:
        # Training phase
        total_correct_train = 0
        total_loss_train = 0.0
        model.train()  # Set model to training mode

        for i in range(0, X_train_tensor.size(0), config["batch_size"]):
            batch_X = X_train_tensor[i:i+config["batch_size"]].to(config["device"])
            batch_y = y_train_tensor[i:i+config["batch_size"]].to(config["device"])

            loss, correct = train_batch(batch_X, batch_y, model, optimizer, criterion)
            total_loss_train += loss.item()
            total_correct_train += correct

        # Validation phase
        total_correct_val = 0
        total_loss_val = 0.0
        model.eval()  # Set model to evaluation mode

        with torch.no_grad():
            for i in range(0, X_val_tensor.size(0), config["batch_size"]):
                batch_X_val = X_val_tensor[i:i+config["batch_size"]].to(config["device"])
                batch_y_val = y_val_tensor[i:i+config["batch_size"]].to(config["device"])

                val_loss, val_correct = evaluate_batch(batch_X_val, batch_y_val, model, criterion)
                total_loss_val += val_loss.item()
                total_correct_val += val_correct

        # Calculate and log metrics for both training and validation
        train_accuracy = total_correct_train / len(X_train_tensor)
        val_accuracy = total_correct_val / len(X_val_tensor)

        train_loss_avg = total_loss_train / (len(X_train_tensor) / config["batch_size"])
        val_loss_avg = total_loss_val / (len(X_val_tensor) / config["batch_size"])

        metrics = {
            # "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss_avg,
            "val_accuracy": val_accuracy
        }

        # Store metrics
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        pbar.set_description(f"Epoch {epoch}")
        pbar.set_postfix(metrics)

    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
        # print(f"Epoch {epoch}: Train loss: {train_loss_avg}, Train accuracy: {train_accuracy}, Val loss: {val_loss_avg}, Val accuracy: {val_accuracy}")


def train_batch(batch_X, batch_y, model, optimizer, criterion):    
    # Forward pass ➡
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    ## Used for BCEWithLogitsLoss
    # # Calculate the number of correct predictions
    # predicted = (outputs >= 0.5).float()
    # correct = (predicted == batch_y).sum().item()
    
    # Used for CrossEntropyLoss
    # Get the class with the highest probability
    _, predicted_classes = outputs.max(dim=1)  
    correct = (predicted_classes == batch_y).sum().item()

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()

    # Step with optimizer
    optimizer.step()

    return loss, correct

def evaluate_batch(batch_X, batch_y, model, criterion):
    # Forward pass ➡
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    ## Used for BCEWithLogitsLoss
    # # Calculate the number of correct predictions
    # predicted = (outputs >= 0.5).float()
    # correct = (predicted == batch_y).sum().item()

    # Used for CrossEntropyLoss
    # Get the class with the highest probability
    _, predicted_classes = outputs.max(dim=1)  
    correct = (predicted_classes == batch_y).sum().item()

    return loss, correct


def load(model, model_name: str, data_name: str, device, save_dir: str = "models"):
    load_path = os.path.join(
        save_dir,
        "{}_{}.pt".format(model_name, data_name),
    )

    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No model found at {load_path}")
    
    try:
        model.load_state_dict(torch.load(load_path, map_location=device))
    except RuntimeError as e:
        print("Mismatch in state_dict keys. Attempting to remap keys...")
        
        # Load the state_dict
        state_dict = torch.load(load_path, map_location=device)
        
        # Dynamically remap keys
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            # Rename keys based on specific patterns (adjust as needed for your case)
            if key.startswith("fc1."):
                new_key = key.replace("fc1.", "model.0.")
            elif key.startswith("fc2."):
                new_key = key.replace("fc2.", "model.2.")
            elif key.startswith("fc3."):
                new_key = key.replace("fc3.", "model.4.")
            elif key.startswith("fc4."):
                new_key = key.replace("fc4.", "model.6.")
            else:
                new_key = key  # Keep unchanged for other keys
            new_state_dict[new_key] = state_dict[key]
        
        # Try loading the remapped state_dict
        try:
            model.load_state_dict(new_state_dict)
            print("State_dict keys successfully remapped and model loaded.")
        except Exception as final_e:
            print("Failed to load state_dict after remapping keys.")
            raise final_e  # Re-raise the exception if remapping fails
        
    
    print(f"Model loaded from {load_path}")

    return model

def test(model, data, device, return_acc: bool = False):
    X_test_tensor, y_test_tensor = data

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test_tensor.to(device))
        predicted = torch.argmax(test_outputs, dim=1).float()
        accuracy = (predicted == y_test_tensor.to(device)).float().mean()
        # accuracy = (predicted.view(-1) == y_test_tensor.view(-1)).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")
    
    if return_acc:
        return accuracy.item()

def predict(model, data, device):
    X_data = data
    with torch.no_grad():
        model.eval()
        outputs = model(X_data.to(device))
        predicted = torch.argmax(outputs, dim=1)
    
    return predicted

def plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot training and validation metrics over epochs
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        train_accuracies (list): List of training accuracies per epoch
        val_accuracies (list): List of validation accuracies per epoch
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()