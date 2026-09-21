"""Behavioral checks for neural NIDS classifier contracts."""

from __future__ import annotations

import pytest
import torch

from src.classifiers.models import CNNOnly, LSTMOnly, SerialCNNLSTM, get_model


@pytest.mark.parametrize("model_type", ("mlp", "cnn", "lstm", "serial"))
def test_models_return_class_logits_for_tabular_batches(model_type: str):
    model = get_model(model_type, num_features=7, num_classes=4)
    model.eval()

    logits = model(torch.randn(5, 7))

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (5, 4)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_cls", (CNNOnly, SerialCNNLSTM))
def test_same_padding_preserves_length_for_even_kernel(model_cls):
    model = model_cls(num_features=7, num_classes=3, kernel_size=4)
    features = model.conv1(torch.randn(2, 1, 7))

    assert features.shape == (2, model.conv1.out_channels, 7)


def test_lstm_uses_final_state_from_both_directions():
    model = LSTMOnly(
        num_features=3,
        num_classes=2,
        hidden_dim=5,
        fc_dim=4,
        dropout=0.0,
    )
    model.eval()
    x = torch.randn(2, 4, 3)

    _, (h_n, _) = model.lstm(x)
    expected_features = model.fc(torch.cat((h_n[-2], h_n[-1]), dim=1))
    _, actual_features = model(x, return_features=True)

    assert torch.allclose(actual_features, expected_features)


def test_lstm_feature_sequence_mode_accepts_only_tabular_schema_vectors():
    model = LSTMOnly(
        num_features=7,
        num_classes=3,
        hidden_dim=5,
        fc_dim=4,
        dropout=0.0,
        feature_sequence=True,
        feature_embedding_dim=6,
        input_transform="asinh",
    )
    model.eval()

    logits = model(torch.randn(2, 7))

    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    with pytest.raises(ValueError, match="feature-sequence mode"):
        model(torch.randn(2, 4, 7))


def test_cnn_multi_bin_pool_preserves_output_contract():
    model = CNNOnly(
        num_features=11,
        num_classes=3,
        conv_channels=(4, 5),
        fc_dim=7,
        pool_size=4,
        input_transform="asinh",
    )
    model.eval()

    logits, features = model(torch.randn(2, 11), return_features=True)

    assert logits.shape == (2, 3)
    assert features.shape == (2, 7)
    assert model.fc[0].in_features == 20

def test_serial_cnn_lstm_uses_final_state_from_both_directions():
    model = SerialCNNLSTM(
        num_features=7,
        num_classes=2,
        conv_channels=(4, 5),
        lstm_hidden=3,
        fc_dim=4,
        dropout=0.0,
    )
    model.eval()
    x = torch.randn(2, 7)

    conv = torch.relu(model.conv1(x.unsqueeze(1)))
    conv = torch.relu(model.conv2(conv)).transpose(1, 2)
    _, (h_n, _) = model.lstm(conv)
    expected_features = model.fc(torch.cat((h_n[-2], h_n[-1]), dim=1))
    _, actual_features = model(x, return_features=True)

    assert torch.allclose(actual_features, expected_features)
