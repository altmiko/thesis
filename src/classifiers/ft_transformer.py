# ft_transformer.py
"""Feature-Tokenizer Transformer (FT-Transformer) victim classifier.

Paper-faithful, dependency-free local implementation of

    Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021.

Design (see ``docs/models/FT_TRANSFORMER_AUDIT.md`` §12-14):

* Each CICIDS2017-DistriNet flow is an INDEPENDENT tabular vector ``x`` of shape
  ``[B, F]`` (F = 79 numeric model-space features). There is NO temporal sequence,
  NO positional encoding over the arbitrary CICFlowMeter column order, and NO
  categorical embedding table (the preprocessed tensor is entirely numeric).
* Per-feature ``NumericalFeatureTokenizer`` maps scalar ``x_j`` to a ``d_token`` vector
  ``T_j = b_j + x_j * W_j`` with its OWN learnable ``W_j`` and ``b_j`` (feature identity).
* A learned ``CLS`` token is appended; PreNorm Transformer blocks (MHSA + ReGLU FFN)
  attend across the feature tokens WITHIN one flow; the head reads ONLY the final CLS
  representation and returns RAW logits ``[B, n_classes]``.

All operations remain differentiable with respect to input features (no detach, argmax,
rounding, NumPy conversion, or `no_grad` path), making this architecture suitable for a future
gradient-attack victim roster.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Architecture version tag stored in checkpoint metadata.
FT_TRANSFORMER_ARCH_VERSION = "ft_transformer.v1"


def _apply_input_transform(x: torch.Tensor, transform: str | None) -> torch.Tensor:
    """Optional monotonic, differentiable stabilizer (parity with MLP/CNN)."""
    if transform is None:
        return x
    if transform == "asinh":
        return torch.asinh(x)
    raise ValueError(f"unsupported input transform: {transform!r}")


class NumericalFeatureTokenizer(nn.Module):
    """Map ``[B, F]`` scalars to ``[B, F, d_token]`` tokens: ``T_j = b_j + x_j * W_j``.

    Every feature ``j`` has its own weight vector ``W_j`` (row of ``weight``) and bias
    ``b_j`` (row of ``bias``) — explicit feature identity, not a shared ``Linear(1, d)``.
    """

    def __init__(self, num_features: int, d_token: int) -> None:
        super().__init__()
        if num_features <= 0 or d_token <= 0:
            raise ValueError("num_features and d_token must be positive")
        self.num_features = num_features
        self.d_token = d_token
        self.weight = nn.Parameter(torch.empty(num_features, d_token))
        self.bias = nn.Parameter(torch.empty(num_features, d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # rtdl-style init: uniform in [-1/sqrt(d), 1/sqrt(d)] for both W and b.
        bound = 1.0 / math.sqrt(self.d_token)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"NumericalFeatureTokenizer expects (batch, {self.num_features}), "
                f"got {tuple(x.shape)}"
            )
        # [B, F, 1] * [F, d] -> [B, F, d], then add per-feature bias [F, d].
        return x.unsqueeze(-1) * self.weight + self.bias


class CLSToken(nn.Module):
    """Learned classification token appended to the feature-token sequence."""

    def __init__(self, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        bound = 1.0 / math.sqrt(d_token)
        nn.init.uniform_(self.weight, -bound, bound)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Append CLS at the END of the sequence; head reads x[:, -1].
        batch = tokens.size(0)
        cls = self.weight.expand(batch, 1, -1)
        return torch.cat([tokens, cls], dim=1)


class ReGLU(nn.Module):
    """ReGLU gate: split the last dim in half, ``a * ReLU(b)``.

    The FFN's first projection produces ``2 * d_hidden``; ReGLU halves it back to
    ``d_hidden`` before the output projection.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) % 2 != 0:
            raise ValueError(f"ReGLU needs an even last dim, got {x.size(-1)}")
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)


class FFN(nn.Module):
    """Position-wise ReGLU feed-forward network (FT-Transformer FFN)."""

    def __init__(self, d_token: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        # First linear outputs 2 * d_hidden for the ReGLU gate.
        self.linear_in = nn.Linear(d_token, 2 * d_hidden)
        self.activation = ReGLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_out = nn.Linear(d_hidden, d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear_in(x))
        x = self.dropout(x)
        return self.linear_out(x)


class MultiheadSelfAttention(nn.Module):
    """Thin wrapper over ``nn.MultiheadAttention`` (batch-first) returning tokens."""

    def __init__(self, d_token: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token {d_token} must be divisible by n_heads {n_heads}")
        self.attention = nn.MultiheadAttention(
            embed_dim=d_token, num_heads=n_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self, x: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out, weights = self.attention(
            x, x, x, need_weights=need_weights, average_attn_weights=need_weights
        )
        return out, weights


class FTTransformerBlock(nn.Module):
    """One PreNorm FT-Transformer block: MHSA sublayer + ReGLU-FFN sublayer.

    PreNorm faithfulness (rtdl): with ``prenormalization=True`` and
    ``first_prenormalization=False`` the FIRST block omits the pre-attention
    normalization (its attention norm is ``Identity``). All other norms are LayerNorm.
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        d_hidden: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
        is_first_block: bool,
        first_prenormalization: bool,
    ) -> None:
        super().__init__()
        skip_attn_norm = is_first_block and not first_prenormalization
        self.attn_norm: nn.Module = nn.Identity() if skip_attn_norm else nn.LayerNorm(d_token)
        self.attention = MultiheadSelfAttention(d_token, n_heads, attention_dropout)
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = FFN(d_token, d_hidden, ffn_dropout)
        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(
        self, x: torch.Tensor, *, need_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attention(self.attn_norm(x), need_weights=need_weights)
        x = x + self.residual_dropout(attn_out)
        ffn_out = self.ffn(self.ffn_norm(x))
        x = x + self.residual_dropout(ffn_out)
        return x, weights


class Head(nn.Module):
    """FT-Transformer prediction head on the CLS token: Norm -> activation -> Linear."""

    def __init__(self, d_token: int, n_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_token)
        self.activation = nn.ReLU()
        self.linear = nn.Linear(d_token, n_classes)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(self.norm(cls)))


class FTTransformer(nn.Module):
    """FT-Transformer victim classifier.

    Args:
        num_features: number of numeric model-space features (79 for CICIDS2017-DistriNet).
        num_classes: number of output logits.
        n_blocks: number of Transformer blocks (default 3).
        d_token: token / block width (default 192).
        attention_n_heads: attention heads (default 8).
        attention_dropout: dropout inside MHSA (default 0.2).
        ffn_multiplier: hidden = round(d_token * ffn_multiplier) (default 4/3).
        ffn_dropout: dropout inside the FFN (default 0.1).
        residual_dropout: dropout on sublayer outputs (default 0.0).
        input_transform: optional monotonic input transform ("asinh") for MLP/CNN parity.
        first_prenormalization: keep rtdl default (False -> first block skips attn norm).

    Forward: ``x`` of shape ``[B, num_features]`` -> raw logits ``[B, num_classes]``.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int,
        *,
        n_blocks: int = 3,
        d_token: int = 192,
        attention_n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_multiplier: float = 4.0 / 3.0,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        input_transform: str | None = None,
        first_prenormalization: bool = False,
    ) -> None:
        super().__init__()
        if num_features <= 0 or num_classes <= 0:
            raise ValueError("num_features and num_classes must be positive")
        if n_blocks <= 0:
            raise ValueError("n_blocks must be positive")

        self.num_features = num_features
        self.num_classes = num_classes
        self.n_blocks = n_blocks
        self.d_token = d_token
        self.attention_n_heads = attention_n_heads
        self.attention_dropout = attention_dropout
        self.ffn_multiplier = ffn_multiplier
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.input_transform = input_transform
        self.first_prenormalization = first_prenormalization
        # Validate the transform name eagerly.
        _apply_input_transform(torch.zeros(1), input_transform)

        d_hidden = int(round(d_token * ffn_multiplier))
        self.tokenizer = NumericalFeatureTokenizer(num_features, d_token)
        self.cls_token = CLSToken(d_token)
        self.blocks = nn.ModuleList(
            FTTransformerBlock(
                d_token=d_token,
                n_heads=attention_n_heads,
                d_hidden=d_hidden,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                residual_dropout=residual_dropout,
                is_first_block=(i == 0),
                first_prenormalization=first_prenormalization,
            )
            for i in range(n_blocks)
        )
        self.head = Head(d_token, num_classes)

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if x.dim() != 2 or x.size(1) != self.num_features:
            raise ValueError(
                f"FTTransformer expects input shape (batch, {self.num_features}), "
                f"got {tuple(x.shape)}"
            )
        tokens = self.tokenizer(_apply_input_transform(x, self.input_transform))
        tokens = self.cls_token(tokens)  # [B, F+1, d]; CLS is the last token.
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            tokens, weights = block(tokens, need_weights=return_attention)
            if return_attention and weights is not None:
                attentions.append(weights)
        logits = self.head(tokens[:, -1])  # CLS-only representation.
        if return_attention:
            return logits, attentions
        return logits

    def optimization_param_groups(
        self, weight_decay: float
    ) -> list[dict[str, object]]:
        """AdamW parameter groups: no weight decay for tokenizer/CLS/bias/LayerNorm.

        Matches the FT-Transformer reference convention of protecting embedding,
        normalization, and bias parameters from weight decay while decaying the
        attention/FFN/head projection weights.
        """
        no_decay: list[nn.Parameter] = []
        decay: list[nn.Parameter] = []
        norm_param_ids: set[int] = set()
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                norm_param_ids.update(id(p) for p in module.parameters(recurse=False))
        tokenizer_cls_ids = {
            id(self.tokenizer.weight),
            id(self.tokenizer.bias),
            id(self.cls_token.weight),
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if (
                id(param) in norm_param_ids
                or id(param) in tokenizer_cls_ids
                or name.endswith(".bias")
                or param.ndim == 1
            ):
                no_decay.append(param)
            else:
                decay.append(param)
        return [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def architecture_config(self) -> dict[str, object]:
        """Serializable architecture description for checkpoint metadata."""
        return {
            "architecture_version": FT_TRANSFORMER_ARCH_VERSION,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "n_blocks": self.n_blocks,
            "d_token": self.d_token,
            "attention_n_heads": self.attention_n_heads,
            "attention_dropout": self.attention_dropout,
            "ffn_multiplier": self.ffn_multiplier,
            "ffn_dropout": self.ffn_dropout,
            "residual_dropout": self.residual_dropout,
            "input_transform": self.input_transform,
            "first_prenormalization": self.first_prenormalization,
        }


def default_ft_transformer_kwargs() -> dict[str, object]:
    """Literature-default FT-Transformer architecture kwargs (task §8/§29)."""
    return {
        "n_blocks": 3,
        "d_token": 192,
        "attention_n_heads": 8,
        "attention_dropout": 0.2,
        "ffn_multiplier": 4.0 / 3.0,
        "ffn_dropout": 0.1,
        "residual_dropout": 0.0,
    }
