# models.py
"""
Neural network architectures for IoT Intrusion Detection.

This module implements:
- SimpleMLP: basic multilayer perceptron baseline
- CNNOnly: 1D CNN for spatial/statistical pattern learning
- FTTransformer: feature-tokenizer transformer

All models accept input of shape (batch, num_features) and output logits.

Author: Research Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
from src.classifiers.ft_transformer import FTTransformer

def _transform_input(x: torch.Tensor, transform: str | None) -> torch.Tensor:
    """Apply an optional monotonic stabilization to heavy-tailed tabular inputs."""
    if transform is None:
        return x
    if transform == "asinh":
        return torch.asinh(x)
    raise ValueError(f"unsupported input transform: {transform}")



class SimpleMLP(nn.Module):
    """
    Simple Multilayer Perceptron baseline.

    Architecture:
        Input (num_features) → Dense(128) → ReLU → Dropout
        → Dense(64) → ReLU → Dropout → Dense(num_classes)

    Input shape: (batch, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
        input_transform: str | None = None,
    ):
        """
        Initialize the MLP.

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
            input_transform: Optional monotonic input transform (currently ``"asinh"``)
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.input_transform = input_transform
        _transform_input(torch.zeros(1), input_transform)

        layers = []
        in_dim = num_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features)

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"SimpleMLP expects input shape (batch, {self.num_features}), "
                f"got {tuple(x.shape)}"
            )

        h = self.features(_transform_input(x, self.input_transform))
        return self.classifier(h)


class CNNOnly(nn.Module):
    """
    1D CNN for learning spatial/statistical patterns from tabular features.

    Architecture:
        Input (batch, num_features) → Reshape to (batch, 1, num_features)
        → Conv1d(1→32, k=3) → ReLU → Conv1d(32→64, k=3) → ReLU
        → AdaptiveMaxPool(pool_size) → Flatten → Dense(64) → ReLU → Dropout
        → Dense(num_classes)

    Pool sizes above one preserve coarse feature position; ``pool_size=1``
    retains the compact legacy baseline.

    Input shape: (batch, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: Tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        fc_dim: int = 64,
        dropout: float = 0.3,
        pool_size: int = 1,
        input_transform: str | None = None,
    ):
        """
        Initialize the CNN.

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            conv_channels: Tuple of (first_conv_out, second_conv_out) channels
            kernel_size: Convolution kernel size
            fc_dim: Fully connected layer dimension
            dropout: Dropout probability
            pool_size: Number of position-preserving adaptive pooling bins
            input_transform: Optional monotonic input transform (currently ``"asinh"``)
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}")
        self.pool_size = pool_size
        self.input_transform = input_transform
        _transform_input(torch.zeros(1), input_transform)

        # ``same`` preserves feature positions for both odd and even kernels.
        # The old kernel_size // 2 padding silently added one position for an
        # even kernel, changing the effective alignment before pooling.
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels[0],
            kernel_size=kernel_size,
            padding="same"
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0],
            out_channels=conv_channels[1],
            kernel_size=kernel_size,
            padding="same"
        )

        # More than one bin preserves coarse feature position instead of
        # collapsing every learned channel to a single orderless maximum.
        self.pool = nn.AdaptiveMaxPool1d(pool_size)

        self.fc = nn.Sequential(
            nn.Linear(conv_channels[1] * pool_size, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fc_dim, num_classes)

        # Store output dimension for feature extraction
        self.feature_dim = fc_dim

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, num_features)
            return_features: If True, also return intermediate features

        Returns:
            If return_features=False: logits of shape (batch, num_classes)
            If return_features=True: (logits, features) where features is (batch, fc_dim)
        """
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"CNNOnly expects input shape (batch, {self.num_features}), "
                f"got {tuple(x.shape)}"
            )

        # Reshape for 1D conv: (batch, num_features) → (batch, 1, num_features)
        x = _transform_input(x, self.input_transform).unsqueeze(1)

        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Pooling
        x = self.pool(x).flatten(1)

        # FC layers
        features = self.fc(x)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits




def get_model(
    model_type: str,
    num_features: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models by name.

    Args:
        model_type: One of 'mlp', 'cnn', 'ft_transformer'
        num_features: Number of input features
        num_classes: Number of output classes
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_type is not recognized
    """
    models = {
        'mlp': SimpleMLP,
        'cnn': CNNOnly,
        'ft_transformer': FTTransformer,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: {list(models.keys())}")

    return models[model_type](num_features, num_classes, **kwargs)
if __name__ == '__main__':
    # Quick test of all supported models
    batch_size = 16
    num_features = 45
    num_classes = 6

    x = torch.randn(batch_size, num_features)

    print("Testing all supported models...")

    for name in ['mlp', 'cnn', 'ft_transformer']:
        model = get_model(name, num_features, num_classes)
        output = model(x)
        print(f"{name:15} output shape: {output.shape}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:15} parameters: {n_params:,}")
