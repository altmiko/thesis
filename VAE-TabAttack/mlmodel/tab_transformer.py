import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from hyper_connections import HyperConnections

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# classes

class PreNorm(Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams = 4
    ):
        super().__init__()
        self.layers = ModuleList([])

        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                init_hyper_conn(dim = dim, branch = PreNorm(dim, FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        x = self.expand_streams(x)

        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = ff(x)

        x = self.reduce_streams(x)

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)
# mlp

class MLP(Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

# main class

class TabTransformer(Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 2,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        use_shared_categ_embed = True,
        shared_categ_dim_divisor = 8.,   # in paper, they reserve dimension / 8 for category shared embedding
        num_residual_streams = 4
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories) if self.num_categories > 0 else 0
        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens

        shared_embed_dim = 0 if not use_shared_categ_embed else int(dim // shared_categ_dim_divisor)

        if self.num_categories > 0:
            self.category_embed = nn.Embedding(total_tokens, dim - shared_embed_dim)

        # take care of shared category embed

        self.use_shared_categ_embed = use_shared_categ_embed

        if use_shared_categ_embed and self.num_categories > 0:
            self.shared_category_embed = nn.Parameter(torch.zeros(self.num_categories, shared_embed_dim))
            nn.init.normal_(self.shared_category_embed, std = 0.02)

        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        if self.num_unique_categories > 0 and self.num_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
            categories_offset = categories_offset.cumsum(dim = -1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

        # continuous

        self.num_continuous = num_continuous

        if self.num_continuous > 0:
            if exists(continuous_mean_std):
                assert continuous_mean_std.shape == (num_continuous, 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
            self.register_buffer('continuous_mean_std', continuous_mean_std)

            self.norm = nn.LayerNorm(num_continuous)

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            num_residual_streams = num_residual_streams
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous

        hidden_dimensions = [input_size * t for t in  mlp_hidden_mults]
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act = mlp_act)

    def forward(self, x, return_attn = False):
            # Split continuous and categorical features
            # Split input into categorical and numerical features
        x_categ = x[:, :self.num_categories].long()
        x_cont = x[:, self.num_categories:]

        xs = []

        assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} values for your categories input'

        if self.num_unique_categories > 0 and self.num_categories > 0:
            x_categ = x_categ + self.categories_offset

            categ_embed = self.category_embed(x_categ)

            if self.use_shared_categ_embed:
                shared_categ_embed = repeat(self.shared_category_embed, 'n d -> b n d', b = categ_embed.shape[0])
                categ_embed = torch.cat((categ_embed, shared_categ_embed), dim = -1)

            x, attns = self.transformer(categ_embed, return_attn = True)

            flat_categ = rearrange(x, 'b ... -> b (...)')
            xs.append(flat_categ)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} values for your continuous input'

        if self.num_continuous > 0:
            if exists(self.continuous_mean_std):
                mean, std = self.continuous_mean_std.unbind(dim = -1)
                x_cont = (x_cont - mean) / std

            normed_cont = self.norm(x_cont)
            xs.append(normed_cont)

        x = torch.cat(xs, dim = -1)
        logits = self.mlp(x)

        if not return_attn:
            return logits

        return logits, attns


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

    pbar = tqdm(range(config["epochs"]), desc=f"Training {config['model']}", unit="epoch")

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
        # print(f"Epoch {epoch}: Train loss: {train_loss_avg}, Train accuracy: {train_accuracy}, Val loss: {val_loss_avg}, Val accuracy: {val_accuracy}")
    
    plot_training_metrics(train_losses, val_losses, train_accuracies, val_accuracies)
  

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
        
        # Dynamically remap keys for FT-Transformer architecture
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            # Handle transformer-specific key patterns
            if key.startswith("transformer."):
                # Keep transformer blocks as is
                new_key = key
            elif key.startswith("embedding."):
                # Handle embedding layers
                new_key = key.replace("embedding.", "embedder.")
            elif key.startswith("fc."):
                # Handle final classification head
                new_key = key.replace("fc.", "mlp_head.")
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