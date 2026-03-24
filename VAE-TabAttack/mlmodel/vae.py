import os
from typing import Dict, List, Optional, Union

import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from tqdm import tqdm

class CategoricalEmbedder(nn.Module):
    def __init__(self, categories, dim, num_special_tokens=2):
        super().__init__()
        # Validate input categories
        if not isinstance(categories, (list, tuple)):
            raise TypeError("categories must be a list or tuple")
        if not all(isinstance(n, int) and n > 0 for n in categories):
            raise ValueError("All categories must be positive integers")
        
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens
        self.dim = dim

        if self.num_unique_categories > 0:
            # Calculate offsets with proper padding
            categories_offset = F.pad(
                torch.tensor(list(categories), dtype=torch.long),
                (1, 0), 
                value=num_special_tokens
            )
            self.register_buffer('categories_offset', categories_offset.cumsum(dim=-1)[:-1])
            self.embeds = nn.Embedding(self.total_tokens, dim)
        else:
            self.embeds = None

    def forward(self, x_categ):
        if self.num_unique_categories == 0:
            return None
            
        # Apply offset and get embeddings
        x_categ = x_categ + self.categories_offset
        return self.embeds(x_categ)


    def reverse(self, embedded_tensor):
        """Convert embedded tensor back to original categorical values"""
        if self.num_unique_categories == 0 or self.embeds is None:
            return None

        with torch.no_grad():
            # Compute distances between input and all embeddings
            distances = torch.cdist(embedded_tensor, self.embeds.weight)
            # Get the indices of the closest embeddings for each feature
            closest_indices = torch.argmin(distances, dim=2)

            # Determine if indices correspond to special tokens
            is_special_token = closest_indices < self.num_special_tokens

            # Map special tokens directly
            x_categ = torch.where(
                is_special_token, 
                closest_indices, 
                torch.zeros_like(closest_indices)
            )

            # Find the category indices for non-special tokens
            expanded_offsets = self.categories_offset.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, num_categories)
            closest_indices_exp = closest_indices.unsqueeze(-1)  # Shape: (batch, features, 1)
            
            # Determine the category indices using broadcasting
            category_indices = (closest_indices_exp >= expanded_offsets).sum(dim=-1) - 1

            # Mask out invalid categories (indices out of range)
            valid_category_mask = (category_indices >= 0) & (category_indices < self.num_categories)
            category_indices = torch.where(valid_category_mask, category_indices, torch.full_like(category_indices, -1))

            # Subtract the offset for valid categories
            valid_offset = self.categories_offset[category_indices]
            x_categ = torch.where(
                valid_category_mask, 
                closest_indices - valid_offset, 
                torch.full_like(x_categ, -1)
            )

            return x_categ, category_indices


class VariationalAutoencoder(nn.Module):
    def __init__(self, data_name: str, 
                 layers: List, 
                 categories: List[int] = None, 
                 embedding_dim: int = 10, 
                 dropout: float = 0.2,
                 device="cpu"):
        """
        Parameters
        ----------
        data_name:
            Name of the dataset, used for the name when saving and loading the model.
        layers:
            List of layer sizes.
        """
        super(VariationalAutoencoder, self).__init__()

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self.device = device
        self._data_name = data_name
        self._input_dim = layers[0]
        latent_dim = layers[-1]
        self.dropout = dropout

        # initialize the categorical embedder
        self.has_categorical = len(categories) > 0 if categories is not None else False
        if self.has_categorical:
            self.categorical_embedder = CategoricalEmbedder(categories, embedding_dim)
            # self._input_dim = self._input_dim + embedding_dim * (len(categories) - 1)
            self.num_categories = len(categories)
            self.num_embedded = embedding_dim * len(categories)
            self.num_continuous = self._input_dim - (embedding_dim * len(categories))
            self.categories = categories
        else:
            self.categorical_embedder = None
            self.num_categories = 0
            self.num_continuous = self._input_dim
            self.categories = None


        # The VAE components
        lst_encoder = []
        for i in range(1, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.ReLU())
            lst_encoder.append(nn.Dropout(self.dropout))
        encoder = nn.Sequential(*lst_encoder)

        self._mu_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))
        self._log_var_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_decoder.append(nn.ReLU())
        decoder = nn.Sequential(*lst_decoder)

        self.mu_dec = nn.Sequential(
            decoder,
            nn.Linear(layers[1], self._input_dim),
        )

        # Output layer for continuous features (with Sigmoid activation)
        self.cont_output = nn.Sequential(
            nn.Linear(self._input_dim, self.num_continuous),
            nn.Sigmoid()
        )

        # Output layer for categorical features (no activation, produces logits)
        if self.has_categorical:
            self.categ_output = nn.Linear(self._input_dim, sum(categories))  # Logits for all categories

        self.to(self.device)

    def encode(self, x_cont, x_categ=None):
        """
        Encode the input data.
        Parameters
        ----------
        x_cont: torch.Tensor
            Continuous features.
        x_categ: torch.Tensor, optional
            Categorical features.
        """
        if self.categorical_embedder and x_categ is not None:
            # Embed categorical features
            x_categ_embed = self.categorical_embedder(x_categ)
            # Concatenate with continuous features
            x = torch.cat([x_categ_embed.view(x_categ_embed.size(0), -1), x_cont], dim=1)
        else:
            x = x_cont

        return self._mu_enc(x), self._log_var_enc(x)

    def decode(self, z):
        # Pass the latent representation through the main decoder
        x_decoded = self.mu_dec(z)  # Shape: (batch_size, self._input_dim)

        # Reconstruct continuous features
        x_cont_recon = self.cont_output(x_decoded)
        # Shape: (batch_size, self.num_continuous)

        # Reconstruct categorical features
        if self.has_categorical:
            # Get logits for categorical features
            x_categ_logits = self.categ_output(x_decoded)  # Shape: (batch_size, sum(categories))
            
            # Split logits into chunks corresponding to each categorical feature
            logits_chunks = torch.split(x_categ_logits, self.categories, dim=1)  # List of tensors

            # Pad each chunk to match max_categories
            max_categories = max(self.categories)
            padded_logits = []
            for chunk in logits_chunks:
                # Calculate padding size
                padding_size = max_categories - chunk.size(1)
                # Pad the chunk with zeros along the last dimension
                padded_chunk = F.pad(chunk, (0, padding_size), value=0)  # Shape: (batch_size, max_categories)
                padded_logits.append(padded_chunk)

            # Stack the padded logits into a 3D tensor
            x_categ_logits = torch.stack(padded_logits, dim=1)  # Shape: (batch_size, num_categories, max_categories)

            # Get embeddings for categorical features
            x_categ_embedded = x_decoded[:, self.num_continuous:self.num_continuous + self.num_embedded]
            x_categ_embedded = x_categ_embedded.view(x_decoded.size(0), self.num_categories, -1)

            # Restore categorical features using the embedder's reverse method
            x_categ_recon, _ = self.categorical_embedder.reverse(x_categ_embedded)  # Shape: (batch_size, num_categories)

            # Concatenate the reconstructed features
            x_recon = torch.cat([x_categ_recon, x_cont_recon], dim=1)  # Shape: (batch_size, self._input_dim)
            return x_recon, x_categ_logits
        else:
            return x_cont_recon, None


    def __reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x):
        if self.has_categorical:
            # Split input into categorical and numerical features
            x_categ = x[:, :self.num_categories].long()
            x_cont = x[:, self.num_categories:]
        else:
            x_cont = x
            x_categ = None

        # Encode the input
        mu_z, log_var_z = self.encode(x_cont, x_categ)
        z = self.__reparametrization_trick(mu_z, log_var_z)
        # Decode the latent representation
        x_recon, x_categ_logits = self.decode(z)
        return x_recon, x_categ_logits, mu_z, log_var_z

    def predict(self, data):
        return self.forward(data)

    def get_latent_representation(self, x):
        if self.has_categorical:
            # Split input into categorical and numerical features
            x_categ = x[:, :self.num_categories].long()
            x_cont = x[:, self.num_categories:]
        else:
            x_cont = x
            x_categ = None

        # Encode the input
        mu_z, log_var_z = self.encode(x_cont, x_categ)
        z = self.__reparametrization_trick(mu_z, log_var_z)

        return z

    def kld(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD


    def reconstruction_loss(self, x, x_recon, x_categ_logits):
        # Split input into categorical and numerical features
        x_categ = x[:, :self.num_categories].long()  # Ensure categorical features are long integers
        x_cont = x[:, self.num_categories:]

        # Split reconstruction into categorical and numerical features
        x_categ_recon = x_recon[:, :self.num_categories]
        x_cont_recon = x_recon[:, self.num_categories:]

        # Continuous loss (MSE)
        cont_loss = F.mse_loss(x_cont_recon, x_cont, reduction='sum')

        # Continuous loss (MAE)
        # cont_loss = F.l1_loss(x_cont_recon, x_cont, reduction='sum')

        # Continuous loss (Huber)
        # cont_loss = F.smooth_l1_loss(x_cont_recon, x_cont, reduction='sum')

        # Categorical loss (cross-entropy)
        if self.has_categorical:
            cat_loss = F.cross_entropy(
                x_categ_logits.view(-1, x_categ_logits.size(-1)),  # Reshape to (batch_size * num_categories, num_classes)
                x_categ.view(-1),  # Reshape to (batch_size * num_categories)
                reduction='sum'
            )
        else:
            cat_loss = 0.0

        # Total loss
        total_loss = self._cat_lambda * cat_loss + (1 - self._cat_lambda) * cont_loss
        return total_loss, cat_loss, cont_loss
        

    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=5,
        lr=1e-3,
        batch_size=32,
        cat_lambda=0.5,
    ):
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values

        train_loader = torch.utils.data.DataLoader(
            xtrain, batch_size=batch_size, shuffle=True
        )

        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=lambda_reg,
        )
        
        self._cat_lambda = cat_lambda
        self._kl_weight = kl_weight
        self._epochs = epochs

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        KLD = np.zeros((epochs, 1))
        RECON = np.zeros((epochs, 1))
        RECON_CAT = np.zeros((epochs, 1))
        RECON_CONT = np.zeros((epochs, 1))
        print("Start training of Variational Autoencoder...")

        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        for epoch in pbar:

            beta = epoch * kl_weight / epochs

            # Initialize the losses
            train_loss = 0
            train_loss_num = 0

            # Train for all the batches
            for data in train_loader:
                data = data.view(data.shape[0], -1)
                data = data.to(self.device).float()

                # forward pass
                x_recon, x_categ_logits, mu, log_var = self(data)

                # recon_loss = criterion(reconstruction, data)
                recon_loss, cat_recon_loss, cont_recon_loss = self.reconstruction_loss(data, x_recon, x_categ_logits)
                kld_loss = self.kld(mu, log_var)
                loss = recon_loss + beta * kld_loss

                # Update the parameters
                optimizer.zero_grad()
                # Compute the loss
                loss.backward()
                # Update the parameters
                optimizer.step()

                # Collect the losses
                train_loss += loss.item()
                train_loss_num += 1

            # Update cat_lambda dynamically every 5 epochs
            
            avg_cat_loss = np.mean(RECON_CAT[:epoch + 1])
            avg_cont_loss = np.mean(RECON_CONT[:epoch + 1])
            if cat_lambda and self.has_categorical and (epoch + 1) % 5 == 0:
                # Adjust _cat_lambda based on the ratio of losses
                if avg_cat_loss > avg_cont_loss:
                    # Increase weight for categorical loss
                    self._cat_lambda = min(self._cat_lambda + 0.1, 0.9)  # Cap at 0.9
                else:
                    # Decrease weight for categorical loss
                    self._cat_lambda = max(self._cat_lambda - 0.1, 0.1)  # Cap at 0.1

            ELBO[epoch] = train_loss / train_loss_num
            KLD[epoch] = kld_loss.detach().cpu().numpy()
            RECON[epoch] = recon_loss.detach().cpu().numpy()
            RECON_CAT[epoch] = cat_recon_loss.detach().cpu().numpy() if self.has_categorical else 0.0
            RECON_CONT[epoch] = cont_recon_loss.detach().cpu().numpy()


            # if wandb is initialized, log the ELBO
            if wandb.run:
                wandb.log({"epoch": epoch, "ELBO": ELBO[epoch, 0]})

            ELBO_train = ELBO[epoch, 0].round(4)
            pbar.set_postfix({"ELBO": ELBO_train})
            # print("[ELBO train: " + str(ELBO_train) + "]")
        
        self.save()
        print("... finished training of Variational Autoencoder.")

        self.eval()

        plot_training_losses(ELBO, KLD, RECON, RECON_CAT, RECON_CONT)

        # return ELBO
        return ELBO, KLD, RECON, RECON_CAT, RECON_CONT

    def save(self, save_dir: str = "models"):
        """
        Save the model to the specified directory.
        Parameters
        ----------
        save_dir: str
            Directory to save the model file. Defaults to 'models'.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(
            save_dir,
            "{}_{}_{}.pt".format(self._data_name, self._kl_weight, self._epochs),
        )
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, input_shape: int, save_dir: str = "models"):
        """
        Load the model from the specified directory.
        Parameters
        ----------
        input_shape: int
            Input dimension size to load the correct model.
        save_dir: str
            Directory from which to load the model file. Defaults to 'models'.
        """
        load_path = os.path.join(
            save_dir,
            "{}_{}.pt".format(self._data_name, input_shape),
        )

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")

        self.load_state_dict(torch.load(load_path, map_location=self.device))
        self.eval()
        print(f"Model loaded from {load_path}")

        return self


    def plot_latent_space(self, data, categories, labels=None, perplexity=30):
        self.eval()
        with torch.no_grad():
            # Encode the data to get the latent means (mu)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            data = data.to(self.device).float()
            z = self.get_latent_representation(data)

        z = z.cpu().numpy()

        # Reduce to 2D using t-SNE
        if z.shape[1] > 2:  # Only apply t-SNE if latent dim > 2
            z = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(z)
        elif z.shape[1] == 2:
            pass  # Already 2D
        else:
            raise ValueError(f"Latent dimension must be >= 2, got {z.shape[1]}")

        # Scatter plot of the latent space
        plt.figure(figsize=(8, 6), dpi=300)
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis', alpha=0.7)
        if labels is not None:
            plt.colorbar(scatter, label="Labels")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2") 
        plt.title("Latent Space Representation (t-SNE)")
        plt.tight_layout()
        plt.show()

def sample_data(data, labels=None, n_samples=1000):
    if data.shape[0] > n_samples:
        np.random.seed(42)
        indices = np.random.choice(data.shape[0], n_samples, replace=False)
        sampled_data = data[indices]
        sampled_labels = labels[indices] if labels is not None else None
    else:
        sampled_data = data
        sampled_labels = labels
    return sampled_data, sampled_labels

def plot_reconstruction_quality(vae, data, labels=None, xlim_combined=None, 
                                ylim_combined=None, xlim_class0=None, 
                                ylim_class0=None, xlim_class1=None, 
                                ylim_class1=None):
    vae.eval()
    with torch.no_grad():
        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data_tensor = torch.tensor(data).float().to(vae.device)
        else:
            data_tensor = data.float().to(vae.device)
        recon_data, _, _, _ = vae(data_tensor)

    # Bring data back to CPU for plotting
    data_np = data_tensor.cpu().numpy()
    recon_np = recon_data.cpu().numpy()

    # Reduce to 2D if data is high-dimensional
    pca = PCA(n_components=2)
    original_2d = pca.fit_transform(data_np)
    reconstructed_2d = pca.transform(recon_np)
    
    # Create a single figure for both classes
    plt.figure(figsize=(12, 6))
    
    # Plot both classes together
    for label, marker, label_text in [(0, 'o', 'Original (Class 0)'), 
                                    (1, 'o', 'Original (Class 1)'),
                                    (0, 'x', 'Reconstructed (Class 0)'),
                                    (1, 'x', 'Reconstructed (Class 1)')]:
        mask = labels == label
        if 'Original' in label_text:
            plt.scatter(original_2d[mask, 0], original_2d[mask, 1], 
                       alpha=0.5, marker=marker, label=label_text)
        else:
            plt.scatter(reconstructed_2d[mask, 0], reconstructed_2d[mask, 1], 
                       alpha=0.5, marker=marker, label=label_text)
    
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    if ylim_combined is not None:
        plt.ylim(ylim_combined)
    if xlim_combined is not None:
        plt.xlim(xlim_combined)
    plt.legend()
    plt.title("Original vs Reconstructed Data (Both Classes)")
    plt.show()


    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot class 0
    mask_0 = labels == 0
    ax1.scatter(original_2d[mask_0, 0], original_2d[mask_0, 1], alpha=0.5, label="Original (Class 0)")
    ax1.scatter(reconstructed_2d[mask_0, 0], reconstructed_2d[mask_0, 1], alpha=0.5, marker="x", label="Reconstructed (Class 0)")
    ax1.set_xlabel("Principal Component 1")
    ax1.set_ylabel("Principal Component 2")
    if ylim_class0 is not None:
        ax1.set_ylim(ylim_class0)
    if xlim_class0 is not None:
        ax1.set_xlim(xlim_class0)
    ax1.legend()
    ax1.set_title("Class 0: Original vs Reconstructed Data")

    # Plot class 1
    mask_1 = labels == 1
    ax2.scatter(original_2d[mask_1, 0], original_2d[mask_1, 1], alpha=0.5, label="Original (Class 1)")
    ax2.scatter(reconstructed_2d[mask_1, 0], reconstructed_2d[mask_1, 1], alpha=0.5, marker="x", label="Reconstructed (Class 1)")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    if ylim_class1 is not None:
        ax2.set_ylim(ylim_class1)
    if xlim_class1 is not None:
        ax2.set_xlim(xlim_class1)
    ax2.legend()
    ax2.set_title("Class 1: Original vs Reconstructed Data")

    plt.tight_layout()
    plt.show()

def plot_training_losses(elbo, kl_loss, recon_loss, cat_recon_loss=None, cont_recon_loss=None):
    # Determine number of subplots based on available losses

    # Check if losses are all zeros and set to None if true
    def _check_zero_loss(loss):
        if loss is not None:
            if isinstance(loss, (int, float)) and loss == 0:
                return None
            elif isinstance(loss, (list, np.ndarray)) and np.all(loss == 0):
                return None
        return loss

    cat_recon_loss = _check_zero_loss(cat_recon_loss)
    cont_recon_loss = _check_zero_loss(cont_recon_loss)
    
    # If either loss is None, set both to None
    if cat_recon_loss is None or cont_recon_loss is None:
        cat_recon_loss = None
        cont_recon_loss = None
    

    num_plots = 3 + (1 if cat_recon_loss is not None else 0) + (1 if cont_recon_loss is not None else 0)
    
    # Use 2 rows if we have 4 or more plots
    if num_plots >= 4:
        fig, axes = plt.subplots(2, (num_plots + 1)//2, figsize=(15, 10))
        axes = axes.flatten()  # Flatten for easier indexing
    else:
        fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    # Plot ELBO
    axes[0].plot(range(len(elbo)), elbo, label="ELBO")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("ELBO over Epochs")
    axes[0].legend()
    
    # Plot KL Divergence Loss
    axes[1].plot(range(len(kl_loss)), kl_loss, label="KL Divergence", color='orange')
    axes[1].set_xlabel("Epochs") 
    axes[1].set_ylabel("Loss")
    axes[1].set_title("KL Divergence Loss over Epochs")
    axes[1].legend()
    
    # Plot Reconstruction Loss
    axes[2].plot(range(len(recon_loss)), recon_loss, label="Reconstruction", color='green')
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Reconstruction Loss over Epochs")
    axes[2].legend()
    
    # Plot Categorical Reconstruction Loss if available
    if cat_recon_loss is not None:
        axes[3].plot(range(len(cat_recon_loss)), cat_recon_loss, label="Categorical Recon", color='purple')
        axes[3].set_xlabel("Epochs")
        axes[3].set_ylabel("Loss")
        axes[3].set_title("Categorical Recon Loss")
        axes[3].legend()
    
    # Plot Continuous Reconstruction Loss if available
    if cont_recon_loss is not None:
        idx = 4 if cat_recon_loss is not None else 3
        axes[idx].plot(range(len(cont_recon_loss)), cont_recon_loss, label="Continuous Recon", color='red')
        axes[idx].set_xlabel("Epochs")
        axes[idx].set_ylabel("Loss")
        axes[idx].set_title("Continuous Recon Loss")
        axes[idx].legend()
    
    # Hide any unused axes
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()
