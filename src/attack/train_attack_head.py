"""Stage-B: victim-guided training of the masked residual head.

Two-stage methodology:
* Stage A (elsewhere): the per-class β-VAE learns the malicious-class manifold with
  NO victim guidance.
* Stage B (here): the base VAE is FROZEN; only the :class:`ResidualHead` is trained,
  guided by a victim IDS to make ``x_adv`` classified as the target (benign) class
  with minimal, feature-normalized perturbation and constraint satisfaction.

The victim is an injected dependency, so one trained VAE/head design can be evaluated
against many victims (MLP/CNN/CNN-LSTM/…) without retraining the base manifold.

Train-only discipline: pass TRAIN-split class data to :meth:`fit`; the perturbation
scale is a TRAIN-derived robust scale (the FeatureTransform IQR). No Jacobian / JSMA /
tangent terms here (deferred by design).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from attack.residual_head import ResidualAttackGenerator
from datasets.transforms import FeatureTransform

_EPS = 1e-8


@dataclass
class StageBConfig:
    target_class: int = 0            # benign id
    epochs: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    lambda_attack: float = 1.0
    lambda_delta: float = 1.0        # feature-normalized perturbation cost
    lambda_c1: float = 0.1           # Layer-1 soft penalty
    lambda_c2: float = 0.1           # Layer-2 soft penalty
    lambda_manifold: float = 0.0     # optional ||z - mu||^2 (latent modes)
    seed: int = 42


class VictimGuidedTrainer:
    def __init__(
        self,
        generator: ResidualAttackGenerator,
        victim: Callable[[torch.Tensor], torch.Tensor],
        transform: FeatureTransform,
        engine,
        config: StageBConfig,
        *,
        feature_cost_weights: np.ndarray | None = None,
        device: str = "cpu",
    ) -> None:
        self.gen = generator
        self.victim = victim
        self.engine = engine
        self.cfg = config
        self.device = device
        # robust per-feature scale (TRAIN IQR from the fitted transform)
        scale = torch.tensor(np.asarray(transform.scale), dtype=torch.float32, device=device)
        self.inv_scale = 1.0 / (scale + _EPS)
        w = np.ones(transform.manifest.n_features) if feature_cost_weights is None else feature_cost_weights
        self.cost_w = torch.tensor(np.asarray(w), dtype=torch.float32, device=device)

    # -- losses ----------------------------------------------------------------
    def perturbation_cost(self, x_adv: torch.Tensor, x_src: torch.Tensor) -> torch.Tensor:
        raw_adv = self.gen.vae.continuous_scaled_to_raw(x_adv)
        raw_src = self.gen.vae.continuous_scaled_to_raw(x_src)
        rel = (raw_adv - raw_src).abs() * self.inv_scale.unsqueeze(0)
        return (rel * self.cost_w.unsqueeze(0)).mean()

    def _freeze_base(self) -> None:
        for p in self.gen.vae.parameters():
            p.requires_grad_(False)
        if self.gen.head is not None:
            for p in self.gen.head.parameters():
                p.requires_grad_(True)

    # -- training --------------------------------------------------------------
    def fit(self, x_train_scaled: np.ndarray) -> dict:
        torch.manual_seed(self.cfg.seed)
        self._freeze_base()
        if self.gen.head is None:
            raise ValueError("Stage-B trains a ResidualHead; generator.head is None")
        x = torch.tensor(np.asarray(x_train_scaled), dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam(self.gen.head.parameters(), lr=self.cfg.lr)
        target = torch.full((self.cfg.batch_size,), self.cfg.target_class, dtype=torch.long, device=self.device)
        n = x.shape[0]
        history: list[dict] = []
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            nb = 0
            for start in range(0, n - self.cfg.batch_size + 1, self.cfg.batch_size):
                xb = x[perm[start : start + self.cfg.batch_size]]
                opt.zero_grad()
                x_adv, _ = self.gen.generate(xb)
                logits = self.victim(x_adv)
                l_attack = F.cross_entropy(logits, target)
                l_delta = self.perturbation_cost(x_adv, xb)
                raw_adv = self.gen.vae.continuous_scaled_to_raw(x_adv)
                pen = self.engine.penalty(raw_adv, w1=self.cfg.lambda_c1, w2=self.cfg.lambda_c2)
                loss = self.cfg.lambda_attack * l_attack + self.cfg.lambda_delta * l_delta + pen["total"]
                loss.backward()
                opt.step()
                epoch_loss += float(loss)
                nb += 1
            history.append({"epoch": epoch, "loss": epoch_loss / max(nb, 1)})
        return {"history": history}

    # -- evaluation ------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, x_scaled: np.ndarray) -> dict:
        x = torch.tensor(np.asarray(x_scaled), dtype=torch.float32, device=self.device)
        x_adv, meta = self.gen.generate(x)
        pred = self.victim(x_adv).argmax(dim=1)
        target_rate = float((pred == self.cfg.target_class).float().mean())
        l0_rate = float(self.engine.layer0.validate(meta["raw_adv"]).float().mean())
        cost = float(self.perturbation_cost(x_adv, x))
        return {
            "target_class_rate": target_rate,
            "layer0_valid_rate": l0_rate,
            "mean_perturbation_cost": cost,
        }
