"""Generic VAE Latent-Space Primitive-Constrained Attack.

Computational graph (classifier gradient path):

    z_adv (ONLY optimizer variable)
      -> attack VAE decoder
      -> dataset primitive_model.infer_primitives_from_decoded
      -> primitive_model.generate (continuous realizability layer)
      -> affine feature scaling
      -> victim classifier
      -> targeted-Benign loss

The decoder output is never classified directly. It proposes movement that is collapsed into
true controls, then the dataset realizability layer regenerates the final feature vector.
Final success is re-evaluated after discrete primitive projection and dependency recomputation.

Search strength (added for the bottleneck study, all backward compatible; the defaults below
reproduce the original single-start / constant-LR / no-logging behaviour exactly):

* ``restarts``      -- multiple random latent starts per sample; the per-sample best FINAL
  PROJECTED (realized) result is kept (targeted-Benign success first, then smallest CW margin).
* ``lr_schedule``   -- "constant" (default) or "cosine" annealing of the Adam LR.
* ``log_grad_norms``-- capture the mean per-sample L2 norms of the classifier-loss gradient at
  the intermediate tensors (x_adv, p, alpha, decoder output, z_adv) for the diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from attack.realizability.base import DatasetPrimitiveModel


@dataclass(frozen=True)
class LatentAttackConfig:
    steps: int = 60
    learning_rate: float = 0.05
    objective: str = "cw"  # "cw" | "ce"
    kappa: float = 0.0
    lambda_cls: float = 1.0
    lambda_latent: float = 0.01
    lambda_cost: float = 0.01
    lambda_realism: float = 0.001
    lambda_recon: float = 0.0
    epsilon_z: float | None = 5.0
    init_noise: float = 0.3  # random latent start; escapes the relu(0) dead point (PGD-style)
    restarts: int = 1  # random restarts per sample; best final projected result is kept
    lr_schedule: str = "constant"  # "constant" | "cosine"
    log_grad_norms: bool = False  # capture dL/d{x,p,alpha,decoder,z} norms for diagnostics

    def __post_init__(self) -> None:
        if self.objective not in {"cw", "ce"}:
            raise ValueError("objective must be 'cw' or 'ce'")
        if self.steps < 0 or self.learning_rate <= 0:
            raise ValueError("steps must be non-negative and learning_rate must be positive")
        if self.epsilon_z is not None and self.epsilon_z < 0:
            raise ValueError("epsilon_z must be non-negative or None")
        if self.restarts < 1:
            raise ValueError("restarts must be >= 1")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("lr_schedule must be 'constant' or 'cosine'")


@dataclass
class LatentAttackResult:
    z0: torch.Tensor
    z_adv: torch.Tensor
    decoded_base_raw: torch.Tensor
    decoded_adv_raw: torch.Tensor
    controls_continuous: dict[str, torch.Tensor]
    controls_realized: dict[str, torch.Tensor]
    x_adv_continuous_raw: torch.Tensor


    x_adv_realized_raw: torch.Tensor
    clean_logits: torch.Tensor
    continuous_logits: torch.Tensor
    realized_logits: torch.Tensor
    latent_l2: torch.Tensor
    primitive_cost: torch.Tensor
    reconstruction_error: torch.Tensor
    latent_distance_sq: torch.Tensor | None
    timing_active: torch.Tensor
    grad_norms: dict[str, float] | None = None


class LatentPrimitiveAttack:
    """Optimize each source posterior mean in latent space; never optimize p/alpha directly."""

    def __init__(self, primitive_model: DatasetPrimitiveModel, config: LatentAttackConfig) -> None:
        self.primitive_model = primitive_model
        self.config = config

    @staticmethod
    def optimizer_parameters(z_adv: torch.Tensor) -> list[torch.Tensor]:
        """The per-sample attack optimizer MUST contain only z_adv."""
        return [z_adv]

    @staticmethod
    def _targeted_loss(logits: torch.Tensor, target: torch.Tensor, objective: str, kappa: float) -> torch.Tensor:
        """Per-sample targeted loss; callers sum it to keep each row batch-size invariant."""
        if objective == "ce":
            return F.cross_entropy(logits, target, reduction="none")
        target_logit = logits.gather(1, target[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, target[:, None], float("-inf"))
        strongest_other = masked.max(1).values
        return torch.relu(strongest_other - target_logit + kappa)

    @staticmethod
    def _targeted_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample (strongest_other - target) logit gap; <=0 iff the target class wins."""
        target_logit = logits.gather(1, target[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, target[:, None], float("-inf"))
        strongest_other = masked.max(1).values
        return strongest_other - target_logit

    @staticmethod
    def _mahalanobis(z: torch.Tensor, realism: Mapping[str, torch.Tensor] | None) -> torch.Tensor | None:
        if realism is None:
            return None
        delta = z - realism["mean"].unsqueeze(0)
        return torch.einsum("ni,ij,nj->n", delta, realism["precision"], delta)

    def _grad_norms(self, vae, victim, raw0, center, scale, bounds, z0, decoded_base_raw,
                    target) -> dict[str, float]:
        """Mean per-sample L2 norm of the classifier-loss gradient at each stage.

        A separate autograd pass with ``retain_grad`` on the intermediate tensors; the
        z_adv used here is a fresh random start (same distribution as optimization)."""
        cfg = self.config
        z_adv = (z0 + cfg.init_noise * torch.randn_like(z0)).detach().requires_grad_(True)
        decoded_adv_raw = vae.decode(z_adv)["continuous_mu_raw"]
        decoded_adv_raw.retain_grad()
        controls = self.primitive_model.infer_primitives_from_decoded(
            raw0, decoded_adv_raw, decoded_base_raw, bounds)
        for t in controls.values():
            t.retain_grad()
        x_adv_raw = self.primitive_model.generate(raw0, controls, quantize=False)
        x_adv_raw.retain_grad()
        logits = victim((x_adv_raw - center) / scale)
        cls_loss = self._targeted_loss(logits, target, cfg.objective, cfg.kappa)
        cls_loss.sum().backward()

        def norm(t):
            if t is None or t.grad is None:
                return 0.0
            g = t.grad.reshape(t.grad.shape[0], -1)
            return float(g.norm(p=2, dim=1).mean())

        return {
            "dL_dx_adv": norm(x_adv_raw),
            "dL_dp": norm(controls.get("p")),
            "dL_dalpha": norm(controls.get("alpha")),
            "dL_ddecoder_output": norm(decoded_adv_raw),
            "dL_dz_adv": norm(z_adv),
        }

    def _optimize_once(self, vae, victim, raw0, center, scale, bounds, z0, decoded_base_raw,
                       clean_logits, target, realism):
        """One random-start latent optimization; returns final per-sample instrumentation."""
        cfg = self.config
        z_adv = (z0 + cfg.init_noise * torch.randn_like(z0)).detach().requires_grad_(True)
        optimizer = torch.optim.Adam(self.optimizer_parameters(z_adv), lr=cfg.learning_rate)
        scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.steps)
                     if cfg.lr_schedule == "cosine" and cfg.steps > 0 else None)

        for _ in range(cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            decoded_adv_raw = vae.decode(z_adv)["continuous_mu_raw"]
            controls = self.primitive_model.infer_primitives_from_decoded(
                raw0, decoded_adv_raw, decoded_base_raw, bounds)
            x_adv_raw = self.primitive_model.generate(raw0, controls, quantize=False)
            logits = victim((x_adv_raw - center) / scale)

            cls_loss = self._targeted_loss(logits, target, cfg.objective, cfg.kappa)
            delta_z = z_adv - z0
            latent_loss = delta_z.square().sum(1)
            cost_loss = ((x_adv_raw - raw0).abs() / scale).mean(1)
            decoded_move = (decoded_adv_raw - decoded_base_raw) / scale
            recon_loss = decoded_move.square().mean(1)
            dist_sq = self._mahalanobis(z_adv, realism)
            if dist_sq is None:
                realism_loss = torch.zeros(raw0.shape[0], device=raw0.device, dtype=raw0.dtype)
            else:
                threshold = realism["threshold_sq"].clamp(min=1e-12)
                realism_loss = torch.relu(dist_sq / threshold - 1.0)

            loss_per_sample = (
                cfg.lambda_cls * cls_loss
                + cfg.lambda_latent * latent_loss
                + cfg.lambda_cost * cost_loss
                + cfg.lambda_realism * realism_loss
                + cfg.lambda_recon * recon_loss
            )
            loss_per_sample.sum().backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if cfg.epsilon_z is not None:
                with torch.no_grad():
                    dz = z_adv - z0
                    norm = dz.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
                    dz.mul_(torch.clamp(cfg.epsilon_z / norm, max=1.0))
                    z_adv.copy_(z0 + dz)

        # Re-decode the optimized latent, infer effective controls, then realize/discretize.
        with torch.no_grad():
            decoded_adv_raw = vae.decode(z_adv)["continuous_mu_raw"]
            controls_cont = self.primitive_model.infer_primitives_from_decoded(
                raw0, decoded_adv_raw, decoded_base_raw, bounds)
            x_cont = self.primitive_model.generate(raw0, controls_cont, quantize=False)
            logits_cont = victim((x_cont - center) / scale)
            controls_real = self.primitive_model.project_controls(raw0, controls_cont, bounds)
            x_real = self.primitive_model.generate(raw0, controls_real, quantize=True)
            logits_real = victim((x_real - center) / scale)
            dz = z_adv - z0
            latent_l2 = dz.norm(p=2, dim=1)
            primitive_cost = ((x_real - raw0).abs() / scale).mean(1)
            recon_err = (((decoded_adv_raw - raw0) / scale).square().mean(1))
            dist_sq = self._mahalanobis(z_adv, realism)
            margin = self._targeted_margin(logits_real, target)  # <=0 iff realized is Benign
        return {
            "z_adv": z_adv.detach(), "decoded_adv_raw": decoded_adv_raw.detach(),
            "controls_cont": {k: v.detach() for k, v in controls_cont.items()},
            "controls_real": {k: v.detach() for k, v in controls_real.items()},
            "x_cont": x_cont.detach(), "x_real": x_real.detach(),
            "logits_cont": logits_cont.detach(), "logits_real": logits_real.detach(),
            "latent_l2": latent_l2.detach(), "primitive_cost": primitive_cost.detach(),
            "recon_err": recon_err.detach(),
            "dist_sq": None if dist_sq is None else dist_sq.detach(),
            "margin": margin.detach(),
        }

    @staticmethod
    def _select(best: dict, cand: dict) -> dict:
        """Per-sample keep whichever restart has the smaller realized targeted margin
        (a realized Benign win has margin <= 0, so successes always beat failures)."""
        take = cand["margin"] < best["margin"]  # [N] bool
        out: dict = {}
        for k, v in best.items():
            if v is None:
                out[k] = None
            elif isinstance(v, dict):
                out[k] = {kk: torch.where(take, cand[k][kk], v[kk]) for kk in v}
            elif v.ndim == 1:
                out[k] = torch.where(take, cand[k], v)
            else:
                out[k] = torch.where(take[:, None], cand[k], v)
        return out

    def attack(
        self,
        vae,
        victim,
        raw0: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        bounds: Mapping[str, torch.Tensor],
        *,
        target_class: int = 0,
        realism: Mapping[str, torch.Tensor] | None = None,
    ) -> LatentAttackResult:
        """Run targeted latent optimization and return continuous + realized instrumentation.

        VAE/victim parameters are frozen, but decoder operations remain in the autograd graph
        with respect to ``z_adv``. The optimizer parameter list contains exactly ``[z_adv]``.
        With ``restarts > 1`` the per-sample best FINAL projected result is kept.
        """
        cfg = self.config
        vae.eval(); victim.eval()
        for module in (vae, victim):
            for param in module.parameters():
                param.requires_grad_(False)

        x0_scaled = (raw0 - center) / scale
        with torch.no_grad():
            z0, _ = vae.encode(x0_scaled)
            z0 = z0.detach()
            decoded_base_raw = vae.decode(z0)["continuous_mu_raw"].detach()
            clean_logits = victim(x0_scaled).detach()

        target = torch.full((raw0.shape[0],), int(target_class), dtype=torch.long, device=raw0.device)

        grad_norms = None
        if cfg.log_grad_norms:
            grad_norms = self._grad_norms(vae, victim, raw0, center, scale, bounds, z0,
                                          decoded_base_raw, target)

        best = None
        for _ in range(cfg.restarts):
            cand = self._optimize_once(vae, victim, raw0, center, scale, bounds, z0,
                                       decoded_base_raw, clean_logits, target, realism)
            best = cand if best is None else self._select(best, cand)

        return LatentAttackResult(
            z0=z0.detach(), z_adv=best["z_adv"], decoded_base_raw=decoded_base_raw,
            decoded_adv_raw=best["decoded_adv_raw"], controls_continuous=best["controls_cont"],
            controls_realized=best["controls_real"],
            x_adv_continuous_raw=best["x_cont"], x_adv_realized_raw=best["x_real"],
            clean_logits=clean_logits, continuous_logits=best["logits_cont"],
            realized_logits=best["logits_real"],
            latent_l2=best["latent_l2"], primitive_cost=best["primitive_cost"],
            reconstruction_error=best["recon_err"],
            latent_distance_sq=best["dist_sq"],
            timing_active=self.primitive_model.active_mask(raw0, "alpha").detach(),
            grad_norms=grad_norms,
        )
