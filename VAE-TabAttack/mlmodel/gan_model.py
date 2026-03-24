import os
from typing import Dict, List, Optional, Union, Tuple

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


class TabularGenerator(nn.Module):
    """
    Generator component of the GAN for tabular data.
    Implements an encoder-decoder architecture to transform data between feature space and latent space.
    """
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
        data_name: str
            Name of the dataset, used for the name when saving and loading the model.
        layers: List
            List of layer sizes for the encoder and decoder.
        num_categorical: int
            Number of categorical features.
        num_binary: int
            Number of binary features.
        num_numerical: int
            Number of numerical features.
        embedding_dims: List[Tuple[int, int]]
            List of (num_categories, embedding_dim) tuples for each categorical feature.
        dropout: float
            Dropout rate for regularization.
        num_classes: int
            Number of classes for classification task.
        device: str
            Device to use for computation (cpu or cuda).
        """
        super(TabularGenerator, self).__init__()

        # Print input configuration
        print(f"Initializing TabularGenerator with:")
        print(f"  - layers: {layers}")
        print(f"  - num_categorical: {num_categorical}")
        print(f"  - num_binary: {num_binary}")
        print(f"  - num_numerical: {num_numerical}")
        print(f"  - embedding_dims: {embedding_dims}")

        if len(layers) < 2:
            raise ValueError(
                "Number of layers have to be at least 2 (input and latent space), and number of neurons bigger than 0"
            )

        self.device = device
        self._data_name = data_name
        self._input_dim = layers[0]
        self.latent_dim = layers[-1]
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_categorical = num_categorical
        self.num_binary = num_binary
        self.num_numerical = num_numerical
        self.embedding_dims = embedding_dims

        # Embedding layers for categorical features
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories, embedding_dim in embedding_dims
        ])

        # Calculate input dimension for concatenated embeddings
        total_embedding_dim = sum([embedding_dim for _, embedding_dim in embedding_dims])
        self.input_dim_after_embedding = total_embedding_dim + (layers[0] - num_categorical)
        print(f"  - input_dim_after_embedding: {self.input_dim_after_embedding}")

        # Encoder network
        lst_encoder = []
        lst_encoder.append(nn.Linear(self.input_dim_after_embedding, layers[1]))
        lst_encoder.append(nn.BatchNorm1d(layers[1]))
        lst_encoder.append(nn.ReLU())
        lst_encoder.append(nn.Dropout(self.dropout))

        print(f"  - Layer 1: {self.input_dim_after_embedding} -> {layers[1]}")
        
        for i in range(2, len(layers) - 1):
            lst_encoder.append(nn.Linear(layers[i - 1], layers[i]))
            print(f"  - Layer {i}: {layers[i - 1]} -> {layers[i]}")
            lst_encoder.append(nn.BatchNorm1d(layers[i]))
            lst_encoder.append(nn.ReLU())
            lst_encoder.append(nn.Dropout(self.dropout))
            
        # Final encoder layer to latent space
        lst_encoder.append(nn.Linear(layers[-2], self.latent_dim))
        print(f"  - Output (latent): {layers[-2]} -> {self.latent_dim}")
        self.encoder = nn.Sequential(*lst_encoder)

        # Decoder network - print dimensions at each layer
        print("\nDecoder dimensions:")
        print(f"  - Input (latent): {self.latent_dim}")

        # Decoder network
        lst_decoder = []
        lst_decoder.append(nn.Linear(self.latent_dim, layers[-2]))
        print(f"  - Layer 1: {self.latent_dim} -> {layers[-2]}")
        lst_decoder.append(nn.BatchNorm1d(layers[-2]))
        lst_decoder.append(nn.ReLU())
        lst_decoder.append(nn.Dropout(self.dropout))
        
        for i in range(len(layers) - 2, 1, -1):
            lst_decoder.append(nn.Linear(layers[i], layers[i-1]))
            print(f"  - Layer {len(layers) - i}: {layers[i]} -> {layers[i-1]}")
            lst_decoder.append(nn.BatchNorm1d(layers[i-1]))
            lst_decoder.append(nn.ReLU())
            lst_decoder.append(nn.Dropout(self.dropout))
            
        self.shared_decoder = nn.Sequential(*lst_decoder)

        # Calculate the output size of the shared decoder
        final_decoder_output_size = layers[1]  # This should be the last layer in your decoder
        print(f"\nShared decoder output size: {final_decoder_output_size}")

        # Print categorical decoder dimensions
        print("\nCategorical decoder dimensions:")
        for i, (num_categories, _) in enumerate(embedding_dims):
            print(f"  - Categorical feature {i}: {final_decoder_output_size} -> {final_decoder_output_size // 2} -> {num_categories}")
    
        # Categorical decoder - separate for each categorical feature
        self.cat_decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layers[1], layers[1] // 2),
                nn.ReLU(),
                nn.Linear(layers[1] // 2, num_categories)  # Output logits for each category
            ) for num_categories, _ in embedding_dims
        ])

        # Numerical decoder dimensions
        print(f"\nNumerical decoder dimensions: {final_decoder_output_size} -> {final_decoder_output_size // 2} -> {num_binary + num_numerical}")
        

        # Numerical decoder - for binary and continuous features
        self.num_decoder = nn.Sequential(
            nn.Linear(layers[1], layers[1] // 2),
            nn.ReLU(),
            nn.Linear(layers[1] // 2, num_binary + num_numerical)  # Output for binary and numerical features
        )

        self.to(self.device)

    def encode(self, x_cat, x_num):
        """
        Encode the input data into the latent space.
        
        Parameters:
        ----------
        x_cat: Tensor
            Categorical features.
        x_num: Tensor
            Numerical features (including binary).
            
        Returns:
        -------
        z: Tensor
            Latent representation.
        """
        # Process categorical features through embeddings
        if self.num_categorical > 0:
            embedded_cat = [self.embedding_layers[i](x_cat[:, i]) for i in range(self.num_categorical)]
            embedded_cat = torch.cat(embedded_cat, dim=1)
        else:
            embedded_cat = torch.tensor([], device=self.device)

        # Combine embedded categorical and numerical features
        if self.num_numerical > 0 or self.num_binary > 0:
            x_combined = torch.cat([embedded_cat, x_num], dim=1)
        else:
            x_combined = embedded_cat

        # Pass through the encoder to get latent representation
        z = self.encoder(x_combined)
        return z

    def decode(self, z):
        """
        Decode the latent representation back to feature space.
        
        Parameters:
        ----------
        z: Tensor
            Latent representation.
            
        Returns:
        -------
        x_recon_combined: Tensor
            Reconstructed features (categorical + binary + numerical).
        x_cat_recon: Tensor
            Logits for categorical features.
        x_num_recon: Tensor
            Reconstructed numerical features (binary + continuous).
        """
        # Pass through shared decoder
        shared_decoded = self.shared_decoder(z)

        # Decode categorical features
        if self.num_categorical > 0:
            # Get logits for each categorical feature
            cat_logits = []
            for decoder in self.cat_decoder_layers:
                cat_logits.append(decoder(shared_decoded))
            x_cat_recon = torch.cat(cat_logits, dim=1)  # Concatenate logits

            # Convert logits to predicted categories
            start_idx = 0
            reconstructed_categories = []
            for i, (num_categories, _) in enumerate(self.embedding_dims):
                end_idx = start_idx + num_categories
                logits = x_cat_recon[:, start_idx:end_idx]

                if num_categories > 1:
                    # Multi-class features: take argmax after softmax
                    reconstructed_categories.append(torch.argmax(F.softmax(logits, dim=1), dim=1).unsqueeze(1))
                else:
                    # Binary features: round sigmoid output
                    reconstructed_categories.append((torch.sigmoid(logits) > 0.5).long())
                
                start_idx = end_idx

            x_cat_reconstructed = torch.cat(reconstructed_categories, dim=1)
        else:
            x_cat_recon = None
            x_cat_reconstructed = torch.tensor([], device=self.device)

        # Decode numerical features
        if self.num_numerical > 0 or self.num_binary > 0:
            x_num_recon = self.num_decoder(shared_decoded)
        else:
            x_num_recon = torch.tensor([], device=self.device)

        # Process binary features
        if self.num_binary > 0:
            binary_recon = torch.sigmoid(x_num_recon[:, :self.num_binary])
            binary_recon_rounded = (binary_recon > 0.5).long()
            numerical_recon = x_num_recon[:, self.num_binary:]
        else:
            binary_recon_rounded = torch.tensor([], device=self.device)
            numerical_recon = x_num_recon

        # Combine all reconstructed features
        x_recon_combined = torch.cat([x_cat_reconstructed, binary_recon_rounded, numerical_recon], dim=1)

        return x_recon_combined, x_cat_recon, x_num_recon

    def forward(self, x):
        """
        Forward pass through the generator.
        
        Parameters:
        ----------
        x: Tensor
            Input data.
            
        Returns:
        -------
        x_recon_combined: Tensor
            Reconstructed features.
        x_cat_recon: Tensor
            Logits for categorical features.
        x_num_recon: Tensor
            Reconstructed numerical features.
        z: Tensor
            Latent representation.
        """
        # Split input into categorical and numerical parts
        x_cat = x[:, :self.num_categorical].long()
        x_num = x[:, self.num_categorical:]

        # Encode
        z = self.encode(x_cat, x_num)
        
        # Decode
        x_recon_combined, x_cat_recon, x_num_recon = self.decode(z)
        
        return x_recon_combined, x_cat_recon, x_num_recon, z

    def generate_samples(self, z=None, batch_size=None):
        """
        Generate new samples from the latent space.
        
        Parameters:
        ----------
        z: Tensor, optional
            Latent vectors to decode. If None, random vectors are generated.
        batch_size: int, optional
            Number of samples to generate if z is None.
            
        Returns:
        -------
        x_gen: Tensor
            Generated samples.
        z: Tensor
            Latent vectors used for generation.
        """
        if z is None:
            if batch_size is None:
                raise ValueError("Either z or batch_size must be provided")
            # Generate random latent vectors
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
        
        # Decode to get generated samples
        x_gen, _, _ = self.decode(z)
        
        return x_gen, z


class TabularDiscriminator(nn.Module):
    """
    Discriminator component of the GAN for tabular data.
    Acts as a classifier that distinguishes between real data classes.
    """
    def __init__(self, input_dim: int, 
                 hidden_dims: List[int],
                 num_categorical: int,
                 num_binary: int,
                 num_numerical: int,
                 embedding_dims: List[Tuple[int, int]],
                 num_classes: int = 2,
                 dropout: float = 0.2,
                 device="cpu"):
        """
        Parameters
        ----------
        input_dim: int
            Input dimension size.
        hidden_dims: List[int]
            List of hidden layer dimensions.
        num_categorical: int
            Number of categorical features.
        num_binary: int
            Number of binary features.
        num_numerical: int
            Number of numerical features.
        embedding_dims: List[Tuple[int, int]]
            List of (num_categories, embedding_dim) tuples for each categorical feature.
        num_classes: int
            Number of output classes.
        dropout: float
            Dropout rate for regularization.
        device: str
            Device to use for computation.
        """
        super(TabularDiscriminator, self).__init__()
        
        self.device = device
        self.num_classes = num_classes
        self.num_categorical = num_categorical
        self.num_binary = num_binary
        self.num_numerical = num_numerical
        self.embedding_dims = embedding_dims
        
        # Embedding layers for categorical features
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embedding_dim)
            for num_categories, embedding_dim in embedding_dims
        ])
        
        # Calculate input dimension after embedding
        total_embedding_dim = sum([embedding_dim for _, embedding_dim in embedding_dims])
        self.input_dim_after_embedding = total_embedding_dim + (input_dim - num_categorical)
        
        # Build neural network layers
        layers = []
        input_size = self.input_dim_after_embedding
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_size, num_classes))
        
        self.model = nn.Sequential(*layers)
        self.to(self.device)
    
    def forward(self, x):
        """
        Forward pass through the discriminator.
        
        Parameters:
        ----------
        x: Tensor
            Input data.
            
        Returns:
        -------
        logits: Tensor
            Class logits.
        """
        # Split input into categorical and numerical parts
        x_cat = x[:, :self.num_categorical].long()
        x_num = x[:, self.num_categorical:]
        
        # Process categorical features through embeddings
        if self.num_categorical > 0:
            embedded_cat = [self.embedding_layers[i](x_cat[:, i]) for i in range(self.num_categorical)]
            embedded_cat = torch.cat(embedded_cat, dim=1)
        else:
            embedded_cat = torch.tensor([], device=self.device)
        
        # Combine embedded categorical and numerical features
        if self.num_numerical > 0 or self.num_binary > 0:
            x_combined = torch.cat([embedded_cat, x_num], dim=1)
        else:
            x_combined = embedded_cat
        
        # Forward pass through the model
        logits = self.model(x_combined)
        
        return logits


class TabularGAN:
    """
    Generative Adversarial Network (GAN) for tabular data.
    Combines generator and discriminator networks to model data distributions.
    """
    def __init__(self, data_name: str,
                 generator_layers: List,
                 discriminator_layers: List,
                 num_categorical: int,
                 num_binary: int,
                 num_numerical: int,
                 embedding_dims: List[Tuple[int, int]],
                 num_classes: int = 2,
                 lambda1: float = 1.0,
                 lambda2: float = 1.0,
                 gamma1: float = 1.0,
                 gamma2: float = 1.0,
                 dropout: float = 0.2,
                 device="cpu"):
        """
        Initialize the TabularGAN model.
        
        Parameters
        ----------
        data_name: str
            Name of the dataset.
        generator_layers: List
            List of layer sizes for the generator.
        discriminator_layers: List
            List of layer sizes for the discriminator.
        num_categorical: int
            Number of categorical features.
        num_binary: int
            Number of binary features.
        num_numerical: int
            Number of numerical features.
        embedding_dims: List[Tuple[int, int]]
            List of (num_categories, embedding_dim) tuples for each categorical feature.
        num_classes: int
            Number of classes for classification.
        lambda1, lambda2: float
            Weights for discriminator loss components.
        gamma1, gamma2: float
            Weights for generator loss components (classification vs. reconstruction).
        dropout: float
            Dropout rate for regularization.
        device: str
            Device to use for computation.
        """
        self.device = device
        self.data_name = data_name
        self.num_classes = num_classes
        self.num_categorical = num_categorical
        self.num_binary = num_binary
        self.num_numerical = num_numerical
        self.embedding_dims = embedding_dims
        
        # Loss weights
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        
        # Create generator and discriminator
        self.generator = TabularGenerator(
            data_name=data_name,
            layers=generator_layers,
            num_categorical=num_categorical,
            num_binary=num_binary,
            num_numerical=num_numerical,
            embedding_dims=embedding_dims,
            num_classes=num_classes,
            dropout=dropout,
            device=device
        )
        
        self.discriminator = TabularDiscriminator(
            input_dim=generator_layers[0],
            hidden_dims=discriminator_layers,
            num_categorical=num_categorical,
            num_binary=num_binary,
            num_numerical=num_numerical,
            embedding_dims=embedding_dims,
            num_classes=num_classes,
            dropout=dropout,
            device=device
        )

    def train(self, xtrain, ytrain, epochs=50, batch_size=32, lr_g=1e-4, lr_d=1e-4, 
              weight_decay=1e-5, save_dir="models"):
        """
        Train the GAN on tabular data.
        
        Parameters:
        ----------
        xtrain: Union[np.ndarray, pd.DataFrame]
            Training data features.
        ytrain: Union[np.ndarray, pd.DataFrame]
            Training data labels.
        epochs: int
            Number of training epochs.
        batch_size: int
            Batch size for training.
        lr_g: float
            Learning rate for generator.
        lr_d: float
            Learning rate for discriminator.
        weight_decay: float
            Weight decay for optimizers.
        save_dir: str
            Directory to save models.
        """
        # Convert to tensors if needed
        if isinstance(xtrain, pd.DataFrame):
            xtrain = xtrain.values
        if isinstance(ytrain, pd.DataFrame):
            ytrain = ytrain.values
        
        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(xtrain),
            torch.LongTensor(ytrain)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        # Optimizers
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=lr_g, weight_decay=weight_decay
        )
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=lr_d, weight_decay=weight_decay
        )
        
        # Loss functions
        ce_loss = nn.CrossEntropyLoss(reduction='mean')
        l1_loss = nn.L1Loss(reduction='mean')
        l2_loss = nn.MSELoss(reduction='mean')

        # Store losses for plotting
        GLOSS = np.zeros((epochs, 1))
        DLOSS = np.zeros((epochs, 1))
        
        # Training loop
        pbar = tqdm(range(epochs), desc="Training GAN", unit="epoch")
        for epoch in pbar:
            g_losses = []
            d_losses = []
            
            for x, y in train_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                batch_size = x.size(0)
                
                # Step 1: Generate samples using the current generator
                x_gen, _, _, _ = self.generator(x)
                
                # Step 2: Train Discriminator
                optimizer_d.zero_grad()
                
                # Real data loss - discriminator should correctly classify real data
                real_logits = self.discriminator(x)
                real_loss = ce_loss(real_logits, y)
                
                # Generated data loss - discriminator should correctly classify generated data
                gen_logits = self.discriminator(x_gen.detach())  # Detach to prevent backprop to generator
                # For generated data, the target is the original class
                gen_loss = ce_loss(gen_logits, y)
                
                # Total discriminator loss
                d_loss = self.lambda1 * real_loss + self.lambda2 * gen_loss
                d_loss.backward()
                optimizer_d.step()
                
                # Step 3: Train Generator
                optimizer_g.zero_grad()
                
                # Generate new samples for updated generator
                x_gen, _, _, _ = self.generator(x)
                
                # Classification loss: generator should produce samples that maintain original class
                gen_logits = self.discriminator(x_gen)
                class_loss = ce_loss(gen_logits, y)
                
                # Similarity loss: generated samples should be close to originals
                similarity_loss = l2_loss(x_gen, x)
                
                # Total generator loss
                g_loss = self.gamma1 * class_loss + self.gamma2 * similarity_loss
                g_loss.backward()
                optimizer_g.step()
                
                # Record losses
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())
            
            # save losses
            GLOSS[epoch] = np.mean(g_losses)
            DLOSS[epoch] = np.mean(d_losses)

            # Update progress bar
            avg_g_loss = sum(g_losses) / len(g_losses)
            avg_d_loss = sum(d_losses) / len(d_losses)
            pbar.set_postfix({
                "G Loss": f"{avg_g_loss:.4f}",
                "D Loss": f"{avg_d_loss:.4f}"
            })
        
        plot_training_losses(DLOSS, GLOSS)
        # Save models
        # self.save(save_dir)
        print("... finished training of GAN")
        # return GLOSS, DLOSS
        
    def save(self, save_dir: str = "models"):
        """
        Save the GAN models.
        
        Parameters:
        ----------
        save_dir: str
            Directory to save models.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        generator_path = os.path.join(save_dir, f"{self.data_name}_generator.pt")
        discriminator_path = os.path.join(save_dir, f"{self.data_name}_discriminator.pt")
        
        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)
        
        print(f"Models saved to {save_dir}")
    
    def load(self, save_dir: str = "models"):
        """
        Load the GAN models.
        
        Parameters:
        ----------
        save_dir: str
            Directory to load models from.
        """
        generator_path = os.path.join(save_dir, f"{self.data_name}_generator.pt")
        discriminator_path = os.path.join(save_dir, f"{self.data_name}_discriminator.pt")
        
        if not os.path.exists(generator_path) or not os.path.exists(discriminator_path):
            raise FileNotFoundError(f"Model files not found in {save_dir}")
        
        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))
        
        self.generator.eval()
        self.discriminator.eval()
        
        print(f"Models loaded from {save_dir}")


    def plot_gan_latent_space(self, data, labels=None, perplexity=30):
        """
        Plot the latent space representation of data using t-SNE.
        
        Parameters:
        -----------
        gan : TabularGAN
            The trained GAN model
        data : numpy.ndarray or torch.Tensor
            Input data to encode
        labels : numpy.ndarray, optional
            Labels for coloring the scatter plot
        perplexity : int, default=30
            Perplexity parameter for t-SNE
        """
        self.generator.eval()
        with torch.no_grad():
            # Convert to tensor if needed
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data)
            data = data.to(self.device).float()
            
            # Split data into categorical and numerical parts
            x_cat = data[:, :self.num_categorical].long()
            x_num = data[:, self.num_categorical:]
            
            # Encode the data to get the latent representation
            z = self.generator.encode(x_cat, x_num)

        # Convert to numpy for t-SNE
        z = z.cpu().numpy()

        # Reduce to 2D using t-SNE if needed
        if z.shape[1] > 2:  
            from sklearn.manifold import TSNE
            z = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(z)
        elif z.shape[1] == 2:
            pass  # Already 2D
        else:
            raise ValueError(f"Latent dimension must be >= 2, got {z.shape[1]}")

        # Create the plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5.5), dpi=300)
        scatter = plt.scatter(z[:, 0], z[:, 1], c=labels, cmap='viridis', alpha=0.7)
        
        if labels is not None:
            plt.colorbar(scatter, label="Labels")
        
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2") 
        plt.title("GAN Latent Space Representation (t-SNE)")
        plt.tight_layout()
        plt.show()
        
        return z  # Return the projected latent points if needed for further analysis


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

def plot_reconstruction_quality(gan, data, labels=None, xlim_combined=None, 
                                    ylim_combined=None, xlim_class0=None, 
                                    ylim_class0=None, xlim_class1=None, 
                                    ylim_class1=None):
    """
    Plot reconstruction quality for GAN model.
    
    Parameters:
    -----------
    gan : TabularGAN
        The trained GAN model
    data : numpy.ndarray or torch.Tensor
        Input data to reconstruct
    labels : numpy.ndarray, optional
        Class labels for the data
    xlim_combined, ylim_combined : tuple, optional
        x and y axis limits for the combined classes plot
    xlim_class0, ylim_class0 : tuple, optional
        x and y axis limits for the class 0 plot
    xlim_class1, ylim_class1 : tuple, optional
        x and y axis limits for the class 1 plot
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    
    gan.generator.eval()
    with torch.no_grad():
        # Convert to tensor if not already
        if not isinstance(data, torch.Tensor):
            data_tensor = torch.tensor(data).float().to(gan.device)
        else:
            data_tensor = data.float().to(gan.device)
        
        # Get reconstruction using the generator
        recon_data, _, _, _ = gan.generator(data_tensor)

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
    plt.title("GAN: Original vs Reconstructed Data (Both Classes)")
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
    
    # Return both the original and reconstructed projections for further analysis if needed
    return original_2d, reconstructed_2d

def plot_training_losses(d_loss, g_loss):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot Discriminator Loss
    axes[0].plot(range(len(d_loss)), d_loss, label="Discriminator", )
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("D-Loss over Epochs")
    axes[0].legend()
    
    # Plot Generator Loss
    axes[1].plot(range(len(g_loss)), g_loss, label="KL Divergence", color='orange')
    axes[1].set_xlabel("Epochs") 
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Generator Loss over Epochs")
    axes[1].legend()
    
    
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
