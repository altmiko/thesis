import os
from typing import Dict, List, Optional, Union, Tuple

import wandb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

from tqdm import tqdm


class SimpleVAE(nn.Module):
    def __init__(self, data_name: str, 
                 layers: List, 
                 num_categorical: int,
                 num_binary: int,
                 num_numerical: int,
                 embedding_dims: List[Tuple[int, int]],
                 dropout: float = 0.2,
                 num_classes: int = 2,
                 device="cpu"):
        """
        Parameters
        ----------
        data_name:
            Name of the dataset, used for the name when saving and loading the model.
        layers:
            List of layer sizes.
        num_classes:
            Number of classes for classification task.
        """
        super(SimpleVAE, self).__init__()

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self.device = device
        self._data_name = data_name
        self._input_dim = layers[0]
        latent_dim = layers[-1]
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_categorical = num_categorical
        self.num_binary = num_binary
        self.num_numerical = num_numerical
        self.embedding_dims = embedding_dims

        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories, embedding_dim in embedding_dims
        ])

        # Calculate input dimension for concatenated embeddings
        total_embedding_dim = sum([embedding_dim for _, embedding_dim in embedding_dims])
        self.input_dim_after_embedding = total_embedding_dim + (layers[0] - num_categorical)


        # The VAE components
        lst_encoder = []
        lst_encoder.append(nn.Linear(self.input_dim_after_embedding, layers[1]))
        lst_encoder.append(nn.BatchNorm1d(layers[1]))
        lst_encoder.append(nn.ReLU())
        lst_encoder.append(nn.Dropout(self.dropout))
        for i in range(2, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            lst_encoder.append(nn.BatchNorm1d(layers[i]))
            lst_encoder.append(nn.ReLU())
            lst_encoder.append(nn.Dropout(self.dropout))
        self.encoder = nn.Sequential(*lst_encoder)

        # self._mu_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))
        # self._log_var_enc = nn.Sequential(encoder, nn.Linear(layers[-2], latent_dim))
        self._mu_enc = nn.Linear(layers[-2], latent_dim)
        self._log_var_enc = nn.Linear(layers[-2], latent_dim)

        lst_decoder = []
        for i in range(len(layers) - 2, 0, -1):
            lst_decoder.append(nn.Linear(layers[i + 1], layers[i]))
            lst_encoder.append(nn.BatchNorm1d(layers[i]))
            lst_decoder.append(nn.ReLU())
        self.shared_decoder = nn.Sequential(*lst_decoder)

        self.mu_dec = nn.Sequential(
            self.shared_decoder,
            nn.Linear(layers[1], self._input_dim),
            nn.Sigmoid()  # Output in [0,1] range
        )

        # Categorical decoder
        self.cat_decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers[1], layers[1]),
                nn.ReLU(),
                nn.Linear(layers[1], num_categories)  # Output logits for the current feature
            ) for num_categories, _ in embedding_dims
        ])

        # Numerical decoder
        self.num_decoder = nn.Sequential(
            nn.Linear(layers[1], layers[1]),
            nn.ReLU(),
            nn.Linear(layers[1], layers[0] - num_categorical)  # Numerical features output
        )


        # Classification layer
        self.classifier = nn.Sequential(
            # nn.Linear(latent_dim, layers[-1]),
            # nn.ReLU(),
            # # nn.Dropout(self.dropout),
            nn.Linear(layers[-1], self.num_classes)
        )

        self.to(self.device)

    def encode(self, x_cat, x_num):
        # Process categorical features through embeddings
        if self.num_categorical > 0:
            embedded_cat = [self.embedding_layers[i](x_cat[:, i]) for i in range(self.num_categorical)]
            embedded_cat = torch.cat(embedded_cat, dim=1)

        else:
            embedded_cat = torch.tensor([], device=self.device)

        if self.num_numerical > 0 or self.num_binary > 0:
            x_combined = torch.cat([embedded_cat, x_num], dim=1)
        else:
            x_combined = embedded_cat

        # Pass through the encoder
        hidden = self.encoder(x_combined)
        mu = self._mu_enc(hidden)
        log_var = self._log_var_enc(hidden)
        return mu, log_var


    def decode(self, z):
        # Pass through shared decoder
        shared_decoded = self.shared_decoder(z)

        if self.num_categorical > 0:
            # Decode categorical features
            x_cat_recon = [decoder(shared_decoded) for decoder in self.cat_decoder_layers]
            x_cat_recon = torch.cat(x_cat_recon, dim=1)  # Concatenate logits for all categorical features

            # Convert logits to probabilities and then to label-encoded integers
            start_idx = 0
            reconstructed_categories = []
            for i, (num_categories, _) in enumerate(self.embedding_dims):
                end_idx = start_idx + num_categories
                logits = x_cat_recon[:, start_idx:end_idx]

                if num_categories > 1:
                    # For multi-class features, apply softmax and take argmax
                    reconstructed_categories.append(torch.argmax(F.softmax(logits, dim=1), dim=1).unsqueeze(1))
                else:
                    # For binary features, apply sigmoid and round
                    reconstructed_categories.append((torch.sigmoid(logits) > 0.5).long())
                start_idx = end_idx

            x_cat_reconstructed = torch.cat(reconstructed_categories, dim=1)  # Combine all reconstructed categorical features
        else:
            x_cat_recon = None
            x_cat_reconstructed = torch.tensor([], device=self.device)

        if self.num_numerical > 0 or self.num_binary > 0:
            # Decode numerical features
            x_num_recon = self.num_decoder(shared_decoded)
        else:
            x_num_recon = torch.tensor([], device=self.device)

        # Apply sigmoid to the first n columns of x_num_recon (binary features)
        if self.num_binary > 0:
            binary_recon = torch.sigmoid(x_num_recon[:, :self.num_binary])
            binary_recon_rounded = (binary_recon > 0.5).long()  # Round to 0 or 1
            numerical_recon = x_num_recon[:, self.num_binary:]  # The rest are numerical features
        else:
            binary_recon_rounded = torch.tensor([], device=self.device)
            numerical_recon = x_num_recon

        # Combine categorical, binary, and numerical features to match x's shape
        x_recon_combined = torch.cat([x_cat_reconstructed, binary_recon_rounded, numerical_recon], dim=1)

        return x_recon_combined, x_cat_recon, x_num_recon

    def classify(self, z):
        return self.classifier(z)

    def __reparametrization_trick(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)  # the Gaussian random noise
        return mu + std * epsilon

    def forward(self, x, is_training=True):
        x_cat = x[:, :self.num_categorical].long()
        x_num = x[:, self.num_categorical:]

        # Encode the input
        mu_z, log_var_z = self.encode(x_cat, x_num)
        z = self.__reparametrization_trick(mu_z, log_var_z) if is_training else mu_z
        # Decode the latent representation
        x_recon_combined, x_cat_recon, x_num_recon = self.decode(z)
        # Classify
        logits = self.classify(z)
        return x_recon_combined, x_cat_recon, x_num_recon, mu_z, log_var_z, logits

    def predict(self, data):
        return self.forward(data)

    def get_latent_representation(self, x):
        # Encode the input
        mu_z, log_var_z = self.encode(x)
        z = self.__reparametrization_trick(mu_z, log_var_z)
        return z

    def kld(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return KLD

    # def reconstruction_loss(self, x, x_recon):
    #     # Reconstruction loss (MSE)
    #     recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    #     return recon_loss

    def classification_loss(self, logits, y):
        # Classification loss (Cross-entropy)
        class_loss = F.cross_entropy(logits, y, reduction='sum')
        return class_loss

    def reconstruction_loss(self, x, x_cat_recon, x_num_recon):
        """
        Compute reconstruction loss for categorical and numerical data.

        Parameters:
        - x: Original input tensor (categorical + numerical combined).
        - x_cat_recon: Reconstructed logits for categorical features.
        - x_num_recon: Reconstructed numerical features.

        Returns:
        - Total reconstruction loss.
        """
        # Split input into categorical and numerical parts
        x_cat = x[:, :self.num_categorical]  # Label-encoded categorical features
        x_num = x[:, self.num_categorical:]  # Numerical features
        if self.num_binary > 0:
            x_bin = x_num[:, :self.num_binary]
            x_num = x_num[:, self.num_binary:]
        else:
            x_bin = torch.tensor([], device=self.device)
        

        # Split reconstructed logits for each categorical feature
        start_idx = 0
        cat_losses = []
        for i, (num_categories, _) in enumerate(self.embedding_dims):
            end_idx = start_idx + num_categories
            logits = x_cat_recon[:, start_idx:end_idx]
            target = x_cat[:, i]  # Label-encoded target for this feature

            if num_categories > 1:
                # Multi-category case: Use cross-entropy loss
                cat_losses.append(F.cross_entropy(logits, target.long(), reduction='sum'))
            else:
                # Single-category case: Use binary cross-entropy loss
                cat_losses.append(F.binary_cross_entropy_with_logits(logits.squeeze(1), target.float(), reduction='sum'))

            start_idx = end_idx

        # Sum up categorical losses
        total_cat_loss = sum(cat_losses)

        # Binary reconstruction loss (Binary cross-entropy)
        if self.num_binary > 0:
            binary_recon = torch.sigmoid(x_num_recon[:, :self.num_binary])  # Apply sigmoid to binary features
            binary_loss = F.binary_cross_entropy(binary_recon, x_bin, reduction='sum')
        else:
            binary_loss = 0.0

        # Numerical reconstruction loss (MSE)
        numerical_recon = x_num_recon[:, self.num_binary:]  # The rest are numerical features
        num_loss = F.mse_loss(numerical_recon, x_num, reduction='sum')

        return total_cat_loss + binary_loss + num_loss 


    def fit(
        self,
        xtrain: Union[pd.DataFrame, np.ndarray],
        ytrain: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        xval: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        yval: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        kl_weight=0.3,
        lambda_reg=1e-6,
        epochs=5,
        lr=1e-3,
        batch_size=32,
        classification_weight=0.5
    ):
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values
        if ytrain is not None and isinstance(ytrain, pd.DataFrame):
            ytrain = ytrain.values

        train_loader = torch.utils.data.DataLoader(
            list(zip(xtrain, ytrain)) if ytrain is not None else xtrain,
            batch_size=batch_size, shuffle=True
        )

        if xval is not None:
            if isinstance(xval, pd.DataFrame):
                xval = xval.values
            if yval is not None and isinstance(yval, pd.DataFrame):
                yval = yval.values

            val_loader = torch.utils.data.DataLoader(
                list(zip(xval, yval)) if yval is not None else xval,
                batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None


        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=lambda_reg,
        )

        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer=optimizer,
        #     mode='min',
        #     factor=0.5,
        #     patience=5
        # )
        
        self._kl_weight = kl_weight
        self._epochs = epochs
        self._classification_weight = classification_weight

        # Train the VAE with the new prior
        ELBO = np.zeros((epochs, 1))
        KLD = np.zeros((epochs, 1))
        RECON = np.zeros((epochs, 1))
        CLASS = np.zeros((epochs, 1))
        print("Start training of Variational Autoencoder...")

        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        for epoch in pbar:
            # Linear warmup over first 10% of training
            warmup_epochs = int(epochs * 0.1)
            if epoch < warmup_epochs:
                # beta = (epoch / warmup_epochs) * kl_weight
                cls_weight = (epoch / warmup_epochs) * self._classification_weight
            else:
                # beta = kl_weight
                cls_weight = self._classification_weight
            beta = (epoch / epochs) * kl_weight

            # Initialize the losses
            train_loss = 0
            train_loss_num = 0

            # Train for all the batches
            for data in train_loader:
                if ytrain is not None:
                    x, y = data
                    y = y.to(self.device).long()
                else:
                    x = data
                    y = None
                
                x = x.view(x.shape[0], -1)
                x = x.to(self.device).float()

                # forward pass
                x_recon, x_cat_recon, x_num_recon, mu, log_var, logits = self(x)

                recon_loss = self.reconstruction_loss(x, x_cat_recon, x_num_recon)
                kld_loss = self.kld(mu, log_var)
                loss = recon_loss + beta * kld_loss

                # Add classification loss if labels are provided
                if y is not None:
                    class_loss = self.classification_loss(logits, y)
                    loss += cls_weight * class_loss

                # Update the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Collect the losses
                train_loss += loss.item()
                train_loss_num += 1

            val_loss = 0
            val_loss_num = 0

            for data in val_loader:
                if yval is not None:
                    x, y = data
                    y = y.to(self.device).long()
                else:
                    x = data
                    y = None
                
                x = x.view(x.shape[0], -1)
                x = x.to(self.device).float()

                # forward pass
                x_recon, x_cat_recon, x_num_recon, mu, log_var, logits = self(x)

                recon_loss = self.reconstruction_loss(x, x_cat_recon, x_num_recon)
                kld_loss = self.kld(mu, log_var)
                loss = recon_loss + beta * kld_loss

                # Add classification loss if labels are provided
                if y is not None:
                    class_loss = self.classification_loss(logits, y)
                    loss += cls_weight * class_loss

                # Collect the losses
                val_loss += loss.item()
                val_loss_num += 1

            # scheduler.step(val_loss)

            ELBO[epoch] = train_loss / train_loss_num
            KLD[epoch] = kld_loss.detach().cpu().numpy()
            RECON[epoch] = recon_loss.detach().cpu().numpy()
            if y is not None:
                CLASS[epoch] = class_loss.detach().cpu().numpy()

            # if wandb is initialized, log the ELBO
            # if wandb.run:
            #     wandb.log({"epoch": epoch, "ELBO": ELBO[epoch, 0]})

            ELBO_train = ELBO[epoch, 0].round(2)
            pbar.set_postfix({"ELBO": ELBO_train})
        
        # self.save()
        print("... finished training of Variational Autoencoder.")

        self.eval()
        

        plot_training_losses(ELBO, KLD, RECON, CLASS)

        return ELBO, KLD, RECON

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
            "{}_{}_{}_{}.pt".format(self._data_name, self._kl_weight, self._epochs, self._classification_weight),
        )
        torch.save(self.state_dict(), save_path)
        print(f"Model saved to {save_path}")

    def load(self, kl, epoch, cls_weight, save_dir: str = "models"):
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
            "{}_{}_{}_{}.pt".format(self._data_name, kl, epoch, cls_weight),
        )

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"No model found at {load_path}")

        self.load_state_dict(torch.load(load_path, map_location=self.device))
        self.eval()
        print(f"Model loaded from {load_path}")

        return self

    def plot_latent_space(self, data, labels=None, perplexity=30):
        self.eval()
        with torch.no_grad():
            # Encode the data to get the latent means (mu)
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            data = data.to(self.device).float()
            x_cat = data[:, :self.num_categorical].long()
            x_num = data[:, self.num_categorical:]
            z, _ = self.encode(x_cat, x_num)

        z = z.cpu().numpy()

        # Reduce to 2D using t-SNE
        if z.shape[1] > 2:  # Only apply t-SNE if latent dim > 2
            z = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(z)
        elif z.shape[1] == 2:
            pass  # Already 2D
        else:
            raise ValueError(f"Latent dimension must be >= 2, got {z.shape[1]}")

        plt.figure(figsize=(6, 5.5), dpi=300)
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis', alpha=0.7)
        # if labels is not None:
        #     plt.colorbar(scatter, label="Labels")
        # plt.xlabel("t-SNE Dimension 1")
        # plt.ylabel("t-SNE Dimension 2") 
        # plt.title("Latent Space Representation (t-SNE)")
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
        recon_data, _, _, _, _, _ = vae(data_tensor)

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

def plot_training_losses(elbo, kl_loss, recon_loss, class_loss=None):
    if class_loss is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()  # Flatten the 2x2 array for easier indexing
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
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
    
    # Plot Classification Loss if provided
    if class_loss is not None:
        axes[3].plot(range(len(class_loss)), class_loss, label="Classification", color='red')
        axes[3].set_xlabel("Epochs")
        axes[3].set_ylabel("Loss")
        axes[3].set_title("Classification Loss over Epochs")
        axes[3].legend()
    
    plt.tight_layout()
    plt.show()


def calculate_reconstruction_quality(original, reconstructed, categorical_num):
    # Ensure the inputs are PyTorch tensors
    if not isinstance(original, torch.Tensor):
        original = torch.tensor(original, dtype=torch.float32)
    if not isinstance(reconstructed, torch.Tensor):
        reconstructed = torch.tensor(reconstructed, dtype=torch.float32)
    
    original = original[:, categorical_num:]
    reconstructed = reconstructed[:, categorical_num:]
    # Flatten the tensors to 1D arrays for compatibility with scipy and sklearn
    original_flat = original.flatten().numpy()
    reconstructed_flat = reconstructed.flatten().numpy()
    
    # Mean Squared Error (MSE)
    mse = mean_squared_error(original_flat, reconstructed_flat)
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(original_flat, reconstructed_flat)
    
    # Root Mean Squared Error (RMSE)
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # R-squared (Coefficient of Determination)
    r2 = r2_score(original_flat, reconstructed_flat)
    
    # Cosine Similarity
    cosine_sim = 1 - cosine(original_flat, reconstructed_flat)
    
    # Pearson Correlation Coefficient
    pearson_corr, _ = pearsonr(original_flat, reconstructed_flat)
    
    # Return the results as a dictionary
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R-squared': r2,
        'Cosine Similarity': cosine_sim,
        'Pearson Correlation': pearson_corr
    }

def compute_categorical_reconstruction_success_rate(x, x_recon, num_categorical):
    """
    Compute the reconstruction success rate for categorical features.

    Parameters:
    - x: Original input tensor (categorical + numerical combined).
    - x_recon: Reconstructed tensor (categorical + numerical combined).
    - num_categorical: Number of categorical features.

    Returns:
    - A dictionary containing accuracy per feature and overall accuracy.
    """
    # Extract categorical parts of x and x_recon
    x_cat = x[:, :num_categorical].long()
    x_cat_recon = x_recon[:, :num_categorical].long()

    accuracies = []
    results = {}

    # Compute accuracy for each categorical feature
    for i in range(num_categorical):
        true_labels = x_cat[:, i]
        reconstructed_labels = x_cat_recon[:, i]

        # Compute accuracy for this feature
        accuracy = (reconstructed_labels == true_labels).float().mean().item()
        accuracies.append(accuracy)

        # Save per-feature accuracy
        results[f"Feature {i}"] = accuracy * 100  # Convert to percentage

    # Compute overall accuracy
    overall_accuracy = sum(accuracies) / len(accuracies)
    results["Overall"] = overall_accuracy * 100

    return results
