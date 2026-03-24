import os

from typing import Union, Tuple, List
from collections import OrderedDict

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

class SoftDecisionTree(nn.Module):
    """Fast implementation of soft decision tree in PyTorch.
    Comes from the paper: "Distilling a Neural Network Into a Soft Decision Tree"
    by Nicholas Frosst, Geoffrey Hinton.

    Implemented by: https://github.com/xuyxu/Soft-Decision-Tree

    Parameters
    ----------
    input_dim : int
      The number of input dimensions.
    output_dim : int
      The number of output dimensions. For example, for a multi-class
      classification problem with `K` classes, it is set to `K`.
    depth : int, default=5
      The depth of the soft decision tree. Since the soft decision tree is
      a full binary tree, setting `depth` to a large value will drastically
      increases the training and evaluating cost.
    lamda : float, default=1e-3
      The coefficient of the regularization term in the training loss. Please
      refer to the paper on the formulation of the regularization term.
    use_cuda : bool, default=False
      When set to `True`, use GPU to fit the model. Training a soft decision
      tree using CPU could be faster considering the inherent data forwarding
      process.

    Attributes
    ----------
    internal_node_num_ : int
      The number of internal nodes in the tree. Given the tree depth `d`, it
      equals to :math:`2^d - 1`.
    leaf_node_num_ : int
      The number of leaf nodes in the tree. Given the tree depth `d`, it equals
      to :math:`2^d`.
    penalty_list : list
      A list storing the layer-wise coefficients of the regularization term.
    inner_nodes : torch.nn.Sequential
      A container that simulates all internal nodes in the soft decision tree.
      The sigmoid activation function is concatenated to simulate the
      probabilistic routing mechanism.
    leaf_nodes : torch.nn.Linear
      A `nn.Linear` module that simulates all leaf nodes in the tree.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            depth: int = 5,
            lamda: float = 1e-3,
            num_categorical: int = 0,
            embedding_dims: List[Tuple[int, int]] = None,
            device: Union[str, torch.device] = "cpu") -> None:
        super(SoftDecisionTree, self).__init__()

        self.num_categorical = num_categorical
        self.embedding_dims = embedding_dims

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories, embedding_dim in embedding_dims
        ])

        # Calculate input dimension for concatenated embeddings
        total_embedding_dim = sum([embedding_dim for _, embedding_dim in embedding_dims])
        self.input_dim_after_embedding = total_embedding_dim + (input_dim - num_categorical)
        
        if num_categorical > 0:
            self.input_dim = self.input_dim_after_embedding
        else:
            self.input_dim = input_dim



        self.output_dim = output_dim

        self.depth = depth
        self.lamda = lamda

        self.device = device

        self._validate_parameters()

        self.internal_node_num_ = 2 ** self.depth - 1
        self.leaf_node_num_ = 2 ** self.depth

        # Different penalty coefficients for nodes in different layers
        self.penalty_list = [
            self.lamda * (2 ** (-depth)) for depth in range(0, self.depth)
        ]

        # Initialize internal nodes and leaf nodes, the input dimension on
        # internal nodes is added by 1, serving as the bias.
        self.inner_nodes = nn.Sequential(
            nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False),
            nn.Sigmoid(),
        )

        self.leaf_nodes = nn.Linear(self.leaf_node_num_,
                                    self.output_dim,
                                    bias=False)

    def forward(self, X, is_training_data=False):
        # Separate categorical and continuous features
        x_cat = X[:, :self.num_categorical].long()
        x_num = X[:, self.num_categorical:]

        # Embed categorical features
        if self.num_categorical > 0:
            x_cat = [embedding_layer(x_cat[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
            x_cat = torch.cat(x_cat, dim=1)
            X = torch.cat([x_cat, x_num], dim=1)
        else:
            X = x_num
        

        _mu, _penalty = self._forward(X)
        y_pred = self.leaf_nodes(_mu)

        # When `X` is the training data, the model also returns the penalty
        # to compute the training loss.
        if is_training_data:
            return y_pred, _penalty
        else:
            return y_pred

    def _forward(self, X):
        """Implementation on the data forwarding process."""

        batch_size = X.size()[0]
        X = self._data_augment(X)

        path_prob = self.inner_nodes(X)
        path_prob = torch.unsqueeze(path_prob, dim=2)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=2)

        _mu = X.data.new(batch_size, 1, 1).fill_(1.0)
        _penalty = torch.tensor(0.0).to(self.device)

        # Iterate through internal odes in each layer to compute the final path
        # probabilities and the regularization term.
        begin_idx = 0
        end_idx = 1

        for layer_idx in range(0, self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            # Extract internal nodes in the current layer to compute the
            # regularization term
            _penalty = _penalty + self._cal_penalty(layer_idx, _mu, _path_prob)
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)

            _mu = _mu * _path_prob  # update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(batch_size, self.leaf_node_num_)

        return mu, _penalty

    def _cal_penalty(self, layer_idx, _mu, _path_prob):
        """
        Compute the regularization term for internal nodes in different layers.
        """

        penalty = torch.tensor(0.0).to(self.device)

        batch_size = _mu.size()[0]
        _mu = _mu.view(batch_size, 2 ** layer_idx)
        _path_prob = _path_prob.view(batch_size, 2 ** (layer_idx + 1))

        for node in range(0, 2 ** (layer_idx + 1)):
            alpha = torch.sum(
                _path_prob[:, node] * _mu[:, node // 2], dim=0
            ) / torch.sum(_mu[:, node // 2], dim=0)

            coeff = self.penalty_list[layer_idx]

            penalty -= 0.5 * coeff * (torch.log(alpha) + torch.log(1 - alpha))

        return penalty

    def _data_augment(self, X):
        """Add a constant input `1` onto the front of each sample."""
        batch_size = X.size()[0]
        X = X.view(batch_size, -1)
        bias = torch.ones(batch_size, 1).to(self.device)
        X = torch.cat((bias, X), 1)

        return X

    def _validate_parameters(self):

        if not self.depth > 0:
            msg = ("The tree depth should be strictly positive, but got {}"
                   "instead.")
            raise ValueError(msg.format(self.depth))

        if not self.lamda >= 0:
            msg = (
                "The coefficient of the regularization term should not be"
                " negative, but got {} instead."
            )
            raise ValueError(msg.format(self.lamda))

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

    pbar = tqdm(range(config["epochs"]), desc="Training Soft Decision Tree", unit="epoch")

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
        # tqdm.write(f"Epoch {epoch}: Train loss: {train_loss_avg}, Train accuracy: {train_accuracy}, Val loss: {val_loss_avg}, Val accuracy: {val_accuracy}")

    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
    

def train_batch(batch_X, batch_y, model, optimizer, criterion):    
    # Forward pass ➡ get both predictions and penalty
    outputs, penalty = model(batch_X, is_training_data=True)
    
    # Calculate classification loss
    classification_loss = criterion(outputs, batch_y)
    # Combine classification loss with penalty
    total_loss = classification_loss + penalty

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
    total_loss.backward()

    # Step with optimizer
    optimizer.step()

    return total_loss, correct

def evaluate_batch(batch_X, batch_y, model, criterion):
    # Forward pass ➡ only get predictions during evaluation
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
        
        state_dict = torch.load(load_path, map_location=device)
        new_state_dict = OrderedDict()
        
        for key in state_dict.keys():
            # Remap decision nodes
            if key.startswith("decision_nodes."):
                new_key = key.replace("decision_nodes.", "decision_nodes.")
            # Remap leaf nodes
            elif key.startswith("leaf_nodes."):
                new_key = key.replace("leaf_nodes.", "leaf_nodes.")
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]
        
        try:
            model.load_state_dict(new_state_dict)
            print("State_dict keys successfully remapped and model loaded.")
        except Exception as final_e:
            print("Failed to load state_dict after remapping keys.")
            raise final_e
        
    
    print(f"Model loaded from {load_path}")

    return model

def test(model, data, device):
    X_test_tensor, y_test_tensor = data

    with torch.no_grad():
        model.eval()
        test_outputs = model(X_test_tensor.to(device))
        predicted = torch.argmax(test_outputs, dim=1).float()
        accuracy = (predicted == y_test_tensor.to(device)).float().mean()
        # accuracy = (predicted.view(-1) == y_test_tensor.view(-1)).float().mean()
        print(f"Accuracy: {accuracy.item() * 100:.2f}%")

def predict(model, data, device):
    X_tensor = data

    with torch.no_grad():
        model.eval()
        outputs = model(X_tensor.to(device))
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