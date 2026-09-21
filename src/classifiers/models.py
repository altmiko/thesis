# models.py
"""
Neural network architectures for IoT Intrusion Detection.

This module implements:
- SimpleMLP: Basic multilayer perceptron baseline
- CNNOnly: 1D CNN for spatial/statistical pattern learning
- LSTMOnly: LSTM/BiLSTM for temporal pattern learning
- SerialCNNLSTM: Sequential CNN-LSTM architecture

All models accept input of shape (batch, num_features) and output logits.

Author: Research Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

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


class LSTMOnly(nn.Module):
    """
    LSTM/BiLSTM baseline with explicit tabular and temporal modes.

    The legacy mode treats a tabular vector as one recurrent timestep and also
    accepts true ``(batch, sequence, features)`` inputs. Feature-sequence mode
    instead projects each scalar feature, adds a learned feature-identity
    embedding, and recurrently integrates the fixed schema positions.

    Input shape: (batch, num_features), or in legacy mode
    (batch, seq_len, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        fc_dim: int = 64,
        dropout: float = 0.3,
        feature_sequence: bool = False,
        feature_embedding_dim: int = 16,
        input_transform: str | None = None,
    ):
        """
        Initialize the LSTM.

        Args:
            num_features: Number of input features (or feature dim if using sequences)
            num_classes: Number of output classes
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            fc_dim: Fully connected layer dimension
            dropout: Dropout probability
            feature_sequence: Treat scalar schema features as embedded sequence tokens
            feature_embedding_dim: Token width used by feature-sequence mode
            input_transform: Optional monotonic input transform (currently ``"asinh"``)
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.feature_sequence = feature_sequence
        self.input_transform = input_transform
        _transform_input(torch.zeros(1), input_transform)
        if feature_sequence:
            if feature_embedding_dim < 1:
                raise ValueError(
                    f"feature_embedding_dim must be >= 1, got {feature_embedding_dim}"
                )
            self.value_projection = nn.Linear(1, feature_embedding_dim)
            self.feature_embedding = nn.Parameter(
                torch.empty(num_features, feature_embedding_dim)
            )
            nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)
            lstm_input_size = feature_embedding_dim
        else:
            lstm_input_size = num_features

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output dimension depends on bidirectional
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_dim),
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
            x: Input tensor of shape (batch, num_features) or (batch, seq_len, num_features)
            return_features: If True, also return intermediate features

        Returns:
            If return_features=False: logits of shape (batch, num_classes)
            If return_features=True: (logits, features) where features is (batch, fc_dim)
        """
        x = _transform_input(x, self.input_transform)
        if self.feature_sequence:
            if x.dim() != 2 or x.size(1) != self.num_features:
                raise ValueError(
                    "LSTMOnly feature-sequence mode expects input shape "
                    f"(batch, {self.num_features}), got {tuple(x.shape)}"
                )
            x = self.value_projection(x.unsqueeze(-1))
            x = x + self.feature_embedding.unsqueeze(0)
        elif x.dim() == 2:
            if x.size(1) != self.num_features:
                raise ValueError(
                    f"LSTMOnly expects {self.num_features} input features, "
                    f"got {x.size(1)}"
                )
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.size(2) != self.num_features:
                raise ValueError(
                    f"LSTMOnly expects the last dimension to be "
                    f"{self.num_features}, got {x.size(2)}"
                )
        else:
            raise ValueError(
                "LSTMOnly expects input shape (batch, features) or "
                f"(batch, sequence, {self.num_features}), got {tuple(x.shape)}"
            )
        # h_n contains the final state for the last recurrent layer.  For a
        # bidirectional LSTM, the backward state at output[:, -1] has only
        # seen the final timestep; use both final directional states instead.
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            lstm_out = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            lstm_out = h_n[-1]

        features = self.fc(lstm_out)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits


class SerialCNNLSTM(nn.Module):
    """
    Serial CNN→LSTM architecture (baseline for comparison).

    The CNN extracts local features, which are then fed to LSTM.
    This creates an information bottleneck as noted in the research problem.

    Architecture:
        Input → Conv1d layers → LSTM → FC → Classifier

    Input shape: (batch, num_features)
    Output shape: (batch, num_classes) - logits
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        conv_channels: Tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        lstm_hidden: int = 64,
        fc_dim: int = 64,
        dropout: float = 0.3,
        input_transform: str | None = None,
    ):
        """
        Initialize the Serial CNN-LSTM.

        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            conv_channels: Tuple of conv layer output channels
            kernel_size: Convolution kernel size
            lstm_hidden: LSTM hidden dimension
            fc_dim: Fully connected layer dimension
            dropout: Dropout probability
            input_transform: Optional monotonic input transform (currently ``"asinh"``)
        """
        super().__init__()

        self.num_features = num_features
        self.num_classes = num_classes

        self.input_transform = input_transform
        _transform_input(torch.zeros(1), input_transform)
        # CNN layers
        self.conv1 = nn.Conv1d(1, conv_channels[0], kernel_size, padding="same")
        self.conv2 = nn.Conv1d(
            conv_channels[0], conv_channels[1], kernel_size, padding="same"
        )

        # LSTM: treats conv output as sequence
        # Conv output shape: (batch, conv_channels[1], num_features)
        # For LSTM: (batch, seq_len=num_features, input_dim=conv_channels[1])
        self.lstm = nn.LSTM(
            input_size=conv_channels[1],
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_dim = lstm_hidden * 2  # Bidirectional

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fc_dim, num_classes)
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
            logits or (logits, features) tuple
        """
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"SerialCNNLSTM expects input shape (batch, {self.num_features}), "
                f"got {tuple(x.shape)}"
            )

        # CNN forward
        x = _transform_input(x, self.input_transform).unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Transpose for LSTM: (batch, channels, features) →
        # (batch, features, channels)
        x = x.transpose(1, 2)

        # Use the final hidden state from each direction.  The backward state
        # at the last output timestep is not the final backward state.
        _, (h_n, _) = self.lstm(x)
        lstm_out = torch.cat((h_n[-2], h_n[-1]), dim=1)

        features = self.fc(lstm_out)
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
        model_type: One of 'mlp', 'cnn', 'lstm', 'serial'
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
        'lstm': LSTMOnly,
        'serial': SerialCNNLSTM
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

    for name in ['mlp', 'cnn', 'lstm', 'serial']:
        model = get_model(name, num_features, num_classes)
        output = model(x)
        print(f"{name:15} output shape: {output.shape}")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:15} parameters: {n_params:,}")
