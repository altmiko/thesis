"""Behavioral checks for neural NIDS classifier contracts."""

from __future__ import annotations

import pytest
import torch

from src.classifiers.models import CNNOnly, get_model


@pytest.mark.parametrize("model_type", ("mlp", "cnn", "ft_transformer"))
def test_models_return_class_logits_for_tabular_batches(model_type: str):
    model = get_model(model_type, num_features=7, num_classes=4)
    model.eval()

    logits = model(torch.randn(5, 7))

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (5, 4)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("model_cls", (CNNOnly,))
def test_same_padding_preserves_length_for_even_kernel(model_cls):
    model = model_cls(num_features=7, num_classes=3, kernel_size=4)
    features = model.conv1(torch.randn(2, 1, 7))

    assert features.shape == (2, model.conv1.out_channels, 7)




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

