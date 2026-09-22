"""Masked residual adversarial head.

The adversarial problem is to modify an *existing* malicious flow minimally so a
victim IDS misclassifies it — not to synthesize an arbitrary flow. This module
learns a small residual around a source sample:

    delta   = A(z', E(x), x)                 # residual head (optionally + latent)
    x_cand  = x + M ⊙ delta                  # M = perturbability mask (0 frozen)
    x_adv   = P0(x_cand)                      # Layer-0 hard projection
              + immutable features copied from x
              + exact derived features recomputed from primitives

The base VAE (class manifold) is separate and typically frozen; this head is what
Stage-B trains (Phase F). Everything is differentiable so latent PGD/C&W and
gradient Stage-B training can drive it.

Three variants (config-controlled): ``latent`` (steer z, decode displacement),
``residual`` (head only), ``both``.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from constraints.layer0 import Layer0Projector


class ResidualHead(nn.Module):
    """Maps (latent code, source sample) -> a bounded residual in model space."""

    def __init__(
        self,
        latent_dim: int,
        n_features: int,
        hidden: tuple[int, ...] = (128,),
        max_delta: float = 3.0,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        dims = [latent_dim + n_features, *hidden]
        layers: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        layers.append(nn.Linear(dims[-1], n_features))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, x_scaled: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z, x_scaled], dim=1))
        # bounded residual keeps perturbations small and stable
        return self.max_delta * torch.tanh(delta)


class ResidualAttackGenerator(nn.Module):
    """Compose base VAE + mask + Layer-0 projection into a differentiable generator.

    Produces adversarial samples in model (scaled) space so victim classifiers (which
    consume scaled features) can be applied directly. Compatible with the existing
    latent PGD/C&W loops: call :meth:`generate` in place of the decode+mask+reimpose
    step.
    """

    def __init__(
        self,
        vae: nn.Module,
        projector: Layer0Projector,
        mutable_mask: torch.Tensor,
        *,
        head: ResidualHead | None = None,
        derivations: list[tuple[int, Callable[[torch.Tensor], torch.Tensor]]] | None = None,
        mode: str = "both",
        apply_layer0: bool = True,
    ) -> None:
        super().__init__()
        if mode not in {"latent", "residual", "both"}:
            raise ValueError(f"mode must be latent/residual/both, got {mode!r}")
        self.vae = vae
        self.projector = projector
        self.head = head
        self.mode = mode
        self.apply_layer0 = bool(apply_layer0)
        self.derivations = list(derivations or [])
        mask = mutable_mask.to(torch.bool)
        self.register_buffer("mutable_bool", mask)
        self.register_buffer("mutable_float", mask.to(torch.float32))
        if mode in {"residual", "both"} and head is None:
            raise ValueError("residual/both modes require a ResidualHead")

    def encode_mu(self, x_scaled: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x_scaled)[0]

    def raw_delta(self, x_scaled: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        mu = self.encode_mu(x_scaled)
        z_use = mu if z is None else z
        delta = torch.zeros_like(x_scaled)
        if self.mode in {"residual", "both"}:
            delta = delta + self.head(z_use, x_scaled)
        if self.mode in {"latent", "both"}:
            x_dec = self.vae.decode(z_use)["continuous_mu"]
            x_dec0 = self.vae.decode(mu)["continuous_mu"]
            delta = delta + (x_dec - x_dec0)
        return delta

    def generate(
        self,
        x_scaled: torch.Tensor,
        z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        delta = self.raw_delta(x_scaled, z)
        # M ⊙ delta: frozen features receive zero residual
        x_candidate = x_scaled + self.mutable_float.unsqueeze(0) * delta

        raw_cand = self.vae.continuous_scaled_to_raw(x_candidate)
        raw_src = self.vae.continuous_scaled_to_raw(x_scaled)
        if self.apply_layer0:
            # Layer-0 hard projection + immutable preservation (frozen -> source values)
            raw_adv = self.projector.project(
                raw_cand, x_source=raw_src, mutable_mask=self.mutable_bool
            )
            # Exact derived features recomputed from primitives (never perturbed directly)
            for target_idx, fn in self.derivations:
                col = fn(raw_adv)
                raw_adv = raw_adv.clone()
                raw_adv[:, target_idx] = col
        else:
            # Layer 0 disabled for this run: frozen features are still preserved
            # because M zeroes their residual; no domain clamp / derived recompute.
            raw_adv = raw_cand

        x_adv = self.vae.continuous_raw_to_scaled(raw_adv)
        meta = {
            "delta_scaled": (x_adv - x_scaled).detach(),
            "raw_adv": raw_adv.detach(),
        }
        return x_adv, meta
