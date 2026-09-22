"""VAE latent-space attacks whose decoder movement is NOT collapsed into (p, alpha).

Two variants share one optimizer contract with :mod:`attack.vae_latent_primitive`: the ONLY
optimizer leaf is ``z_adv``; the decoder stays in the classifier-gradient path; feature values
are never optimized directly. They differ only in how the decoded movement is turned into the
final feature vector, giving a nested ladder of decoder expressiveness:

* :class:`LatentRawAttack`   -- ``VAE-Latent-Raw``.  Diagnostic only. The decoder movement
  ``decode(z_adv) - decode(z0)`` is added to the pristine input across ALL 79 features and only
  a minimal Layer-0 domain clamp is applied (no mask, no dependency recompute, no primitive
  realizability).  Measures raw decoder/manifold evasion power before any constraint.
* :class:`LatentMaskedAttack` -- ``VAE-Latent-Masked``.  The decoder movement is applied ONLY
  to features the CICIDS2017 perturbation mask marks PERTURBABLE; FROZEN features are copied
  from the pristine input; DERIVED_EXACT features are recomputed from their parents; a Layer-0
  domain clamp is applied.  ``z_adv`` still is the only variable and arbitrary (within-domain)
  changes to the perturbable features are allowed -- the decoder is not forced through p/alpha.

Both anchor the movement at ``decode(z0)`` so that ``z_adv = z0`` reproduces the pristine input
(no adversarial change) -- required by the sanity checks and matching the primitive variant.
Neither claims packet-level (Level-C) realizability; masked is a *masked feature-space* attack.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from attack.masks.base import ResolvedMask
from attack.vae_latent_primitive import LatentAttackConfig
from constraints.layer0 import Layer0Projector


@dataclass
class DecoderLatentResult:
    z0: torch.Tensor
    z_adv: torch.Tensor
    decoded_base_raw: torch.Tensor
    decoded_adv_raw: torch.Tensor
    x_adv_continuous_raw: torch.Tensor
    x_adv_realized_raw: torch.Tensor
    clean_logits: torch.Tensor
    continuous_logits: torch.Tensor
    realized_logits: torch.Tensor
    latent_l2: torch.Tensor
    feature_cost: torch.Tensor
    latent_distance_sq: torch.Tensor | None
    grad_norms: dict[str, float] | None = None


class _BaseDecoderLatentAttack:
    """Shared z_adv optimizer (restarts, LR schedule, grad logging) for decoder-movement attacks."""

    def __init__(self, projector: Layer0Projector, config: LatentAttackConfig) -> None:
        self.projector = projector
        self.config = config

    # -- subclass hooks --------------------------------------------------------
    def _realize_continuous(self, raw0: torch.Tensor, decoded_adv: torch.Tensor,
                            decoded_base: torch.Tensor) -> torch.Tensor:
        """Differentiable map: decoder movement -> final raw feature vector (optimization)."""
        raise NotImplementedError

    def _finalize(self, raw0: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        """Discrete/quantized realized vector used for the reported metrics (no grad)."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def optimizer_parameters(z_adv: torch.Tensor) -> list[torch.Tensor]:
        return [z_adv]

    @staticmethod
    def _targeted_loss(logits, target, objective, kappa):
        if objective == "ce":
            return torch.nn.functional.cross_entropy(logits, target)
        target_logit = logits.gather(1, target[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, target[:, None], float("-inf"))
        return torch.relu(masked.max(1).values - target_logit + kappa).mean()

    @staticmethod
    def _targeted_margin(logits, target):
        target_logit = logits.gather(1, target[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, target[:, None], float("-inf"))
        return masked.max(1).values - target_logit

    @staticmethod
    def _mahalanobis(z, realism):
        if realism is None:
            return None
        delta = z - realism["mean"].unsqueeze(0)
        return torch.einsum("ni,ij,nj->n", delta, realism["precision"], delta)

    def _grad_norms(self, vae, victim, raw0, center, scale, z0, decoded_base_raw, target):
        cfg = self.config
        z_adv = (z0 + cfg.init_noise * torch.randn_like(z0)).detach().requires_grad_(True)
        decoded_adv = vae.decode(z_adv)["continuous_mu_raw"]
        decoded_adv.retain_grad()
        x_adv = self._realize_continuous(raw0, decoded_adv, decoded_base_raw)
        x_adv.retain_grad()
        logits = victim((x_adv - center) / scale)
        self._targeted_loss(logits, target, cfg.objective, cfg.kappa).backward()

        def norm(t):
            if t is None or t.grad is None:
                return 0.0
            g = t.grad.reshape(t.grad.shape[0], -1)
            return float(g.norm(p=2, dim=1).mean())

        return {"dL_dx_adv": norm(x_adv), "dL_ddecoder_output": norm(decoded_adv),
                "dL_dz_adv": norm(z_adv)}

    def _optimize_once(self, vae, victim, raw0, center, scale, z0, decoded_base_raw,
                       target, realism):
        cfg = self.config
        z_adv = (z0 + cfg.init_noise * torch.randn_like(z0)).detach().requires_grad_(True)
        optimizer = torch.optim.Adam(self.optimizer_parameters(z_adv), lr=cfg.learning_rate)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps)
                     if cfg.lr_schedule == "cosine" else None)

        for _ in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            decoded_adv = vae.decode(z_adv)["continuous_mu_raw"]
            x_adv = self._realize_continuous(raw0, decoded_adv, decoded_base_raw)
            logits = victim((x_adv - center) / scale)

            cls_loss = self._targeted_loss(logits, target, cfg.objective, cfg.kappa)
            latent_loss = (z_adv - z0).square().sum(1).mean()
            cost_loss = ((x_adv - raw0).abs() / scale).mean(1).mean()
            dist_sq = self._mahalanobis(z_adv, realism)
            if dist_sq is None:
                realism_loss = torch.zeros((), device=raw0.device, dtype=raw0.dtype)
            else:
                realism_loss = torch.relu(dist_sq / realism["threshold_sq"].clamp(min=1e-12) - 1.0).mean()

            loss = (cfg.lambda_cls * cls_loss + cfg.lambda_latent * latent_loss
                    + cfg.lambda_cost * cost_loss + cfg.lambda_realism * realism_loss)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if cfg.epsilon_z is not None:
                with torch.no_grad():
                    dz = z_adv - z0
                    n = dz.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    dz.mul_(torch.clamp(cfg.epsilon_z / n, max=1.0))
                    z_adv.copy_(z0 + dz)

        with torch.no_grad():
            decoded_adv = vae.decode(z_adv)["continuous_mu_raw"]
            x_cont = self._realize_continuous(raw0, decoded_adv, decoded_base_raw)
            logits_cont = victim((x_cont - center) / scale)
            x_real = self._finalize(raw0, x_cont)
            logits_real = victim((x_real - center) / scale)
            latent_l2 = (z_adv - z0).norm(p=2, dim=1)
            feature_cost = ((x_real - raw0).abs() / scale).mean(1)
            dist_sq = self._mahalanobis(z_adv, realism)
            margin = self._targeted_margin(logits_real, target)
        return {"z_adv": z_adv.detach(), "decoded_adv_raw": decoded_adv.detach(),
                "x_cont": x_cont.detach(), "x_real": x_real.detach(),
                "logits_cont": logits_cont.detach(), "logits_real": logits_real.detach(),
                "latent_l2": latent_l2.detach(), "feature_cost": feature_cost.detach(),
                "dist_sq": None if dist_sq is None else dist_sq.detach(),
                "margin": margin.detach()}

    @staticmethod
    def _select(best: dict, cand: dict) -> dict:
        take = cand["margin"] < best["margin"]
        out: dict = {}
        for k, v in best.items():
            if v is None:
                out[k] = None
            elif v.ndim == 1:
                out[k] = torch.where(take, cand[k], v)
            else:
                out[k] = torch.where(take[:, None], cand[k], v)
        return out

    def attack(self, vae, victim, raw0, center, scale, *, target_class=0, realism=None):
        cfg = self.config
        vae.eval(); victim.eval()
        for module in (vae, victim):
            for p in module.parameters():
                p.requires_grad_(False)
        x0_scaled = (raw0 - center) / scale
        with torch.no_grad():
            z0, _ = vae.encode(x0_scaled)
            z0 = z0.detach()
            decoded_base_raw = vae.decode(z0)["continuous_mu_raw"].detach()
            clean_logits = victim(x0_scaled).detach()
        target = torch.full((raw0.shape[0],), int(target_class), dtype=torch.long, device=raw0.device)

        grad_norms = None
        if cfg.log_grad_norms:
            grad_norms = self._grad_norms(vae, victim, raw0, center, scale, z0,
                                          decoded_base_raw, target)

        best = None
        for _ in range(cfg.restarts):
            cand = self._optimize_once(vae, victim, raw0, center, scale, z0,
                                       decoded_base_raw, target, realism)
            best = cand if best is None else self._select(best, cand)

        return DecoderLatentResult(
            z0=z0.detach(), z_adv=best["z_adv"], decoded_base_raw=decoded_base_raw,
            decoded_adv_raw=best["decoded_adv_raw"], x_adv_continuous_raw=best["x_cont"],
            x_adv_realized_raw=best["x_real"], clean_logits=clean_logits,
            continuous_logits=best["logits_cont"], realized_logits=best["logits_real"],
            latent_l2=best["latent_l2"], feature_cost=best["feature_cost"],
            latent_distance_sq=best["dist_sq"], grad_norms=grad_norms)


class LatentRawAttack(_BaseDecoderLatentAttack):
    """VAE-Latent-Raw: decoder movement on all 79 features, minimal Layer-0 domain clamp only."""

    def _realize_continuous(self, raw0, decoded_adv, decoded_base):
        x = raw0 + (decoded_adv - decoded_base)
        return self.projector.project(x)

    def _finalize(self, raw0, x_cont):
        # Minimal projection only (already domain-clamped); no rounding / dependency recompute.
        return x_cont


class LatentMaskedAttack(_BaseDecoderLatentAttack):
    """VAE-Latent-Masked: decoder movement on PERTURBABLE features; frozen copied; derived recomputed."""

    def __init__(self, resolved_mask: ResolvedMask, projector: Layer0Projector,
                 config: LatentAttackConfig, *, integer_perturbable_idx=None) -> None:
        super().__init__(projector, config)
        self.mask = resolved_mask
        pmask = resolved_mask.perturbable_mask().float()
        self._pmask = pmask  # [F], 1 at perturbable positions
        self._int_idx = (torch.as_tensor(integer_perturbable_idx, dtype=torch.long)
                         if integer_perturbable_idx is not None else None)

    def _realize_continuous(self, raw0, decoded_adv, decoded_base):
        pmask = self._pmask.to(raw0.device)
        # decoder movement applied ONLY to perturbable features; frozen stay = raw0.
        x = raw0 + pmask.unsqueeze(0) * (decoded_adv - decoded_base)
        # Layer-0 domain clamp (positive/bounded/type); frozen already == raw0 and stay valid.
        x = self.projector.project(x)
        # recompute DERIVED_EXACT from (perturbable + frozen) parents -> extractor-consistent.
        x = self.mask.recompute(x)
        return x

    def _finalize(self, raw0, x_cont):
        if self._int_idx is None or self._int_idx.numel() == 0:
            return self.mask.recompute(x_cont)
        idx = self._int_idx.to(x_cont.device)
        x = x_cont.clone()
        x[:, idx] = torch.round(x[:, idx])
        # frozen preservation is structural (never written); recompute derived after rounding.
        return self.mask.recompute(x)
