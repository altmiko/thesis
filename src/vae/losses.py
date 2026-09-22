"""ELBO for continuous CICIoT2023 window aggregates.

Every CSV feature is reconstructed continuously. Bounded indicator aggregates
are sigmoid-constrained by the decoder and participate in the same continuous
likelihood; no BCE, hard-bit target, or protocol cross-entropy is used.
"""
from __future__ import annotations

import math

import torch

_LOG_2PI = math.log(2.0 * math.pi)
_LOG_2 = math.log(2.0)


def _compute_constraint_loss(raw: torch.Tensor, partition: dict) -> dict[str, torch.Tensor]:
    nonnegative = torch.relu(-raw).mean()
    ttl = torch.relu(raw[:, 2] - 255.0).mean()
    minimum, maximum, average = raw[:, 31], raw[:, 32], raw[:, 33]
    ordering = (torch.relu(minimum - maximum) + torch.relu(minimum - average) + torch.relu(average - maximum)).mean()
    expected_variance = raw[:, 34].square()
    variance = (torch.abs(raw[:, 38] - expected_variance) / (expected_variance.abs() + 1.0)).mean()
    packet_positive = torch.relu(-raw[:, 37]).mean()
    zero = raw.new_tensor(0.0)
    total = nonnegative + ttl + ordering + variance + packet_positive
    return {
        "total": total,
        "nonneg": nonnegative,
        "ttl": ttl,
        "ordering": ordering,
        "variance": variance,
        "packet_positive": packet_positive,
        "packet_integer": zero,
    }


def _compute_physics_constraint_loss(raw: torch.Tensor, partition: dict) -> dict[str, torch.Tensor]:
    expected_total = raw[:, 37] * raw[:, 33]
    p2 = (torch.abs(raw[:, 30] - expected_total) / (raw[:, 30].abs() + 1.0)).mean()
    p4 = torch.relu(raw[:, 34] - 0.5 * (raw[:, 32] - raw[:, 31])).mean()
    # Number is an averaged count, so no singleton/integer branch is imposed.
    zero = raw.new_tensor(0.0)
    return {"total": p2 + p4, "p2_total_bytes": p2, "p4_std_bound": p4, "p5_singleton": zero}


def compute_elbo(
    batch_x_39: torch.Tensor,
    model_out: dict,
    partition: dict,
    beta: float,
    target_ind_binary: torch.Tensor | None = None,
    target_protocol_idx: torch.Tensor | None = None,
    protocol_class_weights: torch.Tensor | None = None,
    protocol_loss_weight: float = 1.0,
    constraint_loss_weight: float = 0.0,
    physics_constraint_loss_weight: float = 0.0,
    continuous_feature_weights: torch.Tensor | None = None,
    binary_feature_weights: torch.Tensor | None = None,
    continuous_logvar_floor: float = -4.0,
    continuous_logvar_ceiling: float = 2.0,
    continuous_likelihood: str = "gaussian",
    continuous_nll_per_sample_cap: float | None = None,
    free_bits_lambda: float = 0.0,
    continuous_target_raw: torch.Tensor | None = None,
    raw_relative_continuous_loss_weight: float = 0.0,
    raw_relative_feature_weights: torch.Tensor | None = None,
    raw_relative_epsilon: float = 1.0,
    raw_relative_tail_focus_quantile: float | None = None,
    raw_relative_tail_focus_weight: float = 0.0,
) -> dict:
    mu = model_out["mu"]
    logvar = model_out["logvar"]
    reconstructed = model_out["continuous_mu"]
    raw = model_out.get("continuous_mu_raw")
    effective_logvar = model_out["continuous_logvar"].clamp(continuous_logvar_floor, continuous_logvar_ceiling)
    target = batch_x_39[:, partition["continuous_idx"]]
    if continuous_likelihood == "gaussian":
        per_feature = 0.5 * (effective_logvar + (target - reconstructed).square() / effective_logvar.exp() + _LOG_2PI)
    elif continuous_likelihood == "laplace":
        per_feature = _LOG_2 + effective_logvar + (target - reconstructed).abs() / effective_logvar.exp()
    else:
        raise ValueError("continuous_likelihood must be 'gaussian' or 'laplace'")
    if continuous_feature_weights is not None:
        per_feature = per_feature * continuous_feature_weights.to(per_feature.device).unsqueeze(0)
    per_sample = per_feature.sum(dim=1)
    if continuous_nll_per_sample_cap is not None:
        per_sample = per_sample.clamp(max=float(continuous_nll_per_sample_cap))
    recon = per_sample.mean()

    zero = mu.new_tensor(0.0)
    raw_relative = zero
    raw_relative_tail = zero
    if raw is not None and continuous_target_raw is not None and raw_relative_continuous_loss_weight > 0.0:
        rel = (raw - continuous_target_raw).abs() / (continuous_target_raw.abs() + float(raw_relative_epsilon))
        if raw_relative_feature_weights is not None:
            rel = rel * raw_relative_feature_weights.to(rel.device).unsqueeze(0)
        per_sample_rel = rel.sum(dim=1)
        raw_relative = per_sample_rel.mean()
        if raw_relative_tail_focus_weight > 0.0 and raw_relative_tail_focus_quantile is not None:
            threshold = torch.quantile(per_sample_rel.detach(), float(raw_relative_tail_focus_quantile))
            raw_relative_tail = per_sample_rel[per_sample_rel >= threshold].mean()

    per_dim_kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
    kl = (per_dim_kl.clamp(min=float(free_bits_lambda)) if free_bits_lambda > 0 else per_dim_kl).sum(dim=1).mean()
    constraints = _compute_constraint_loss(raw, partition) if raw is not None and constraint_loss_weight > 0 else {
        "total": zero, "nonneg": zero, "ttl": zero, "ordering": zero,
        "variance": zero, "packet_positive": zero, "packet_integer": zero,
    }
    physics = _compute_physics_constraint_loss(raw, partition) if raw is not None and physics_constraint_loss_weight > 0 else {
        "total": zero, "p2_total_bytes": zero, "p4_std_bound": zero, "p5_singleton": zero,
    }
    loss = recon + beta * kl + constraint_loss_weight * constraints["total"] + physics_constraint_loss_weight * physics["total"] + raw_relative_continuous_loss_weight * raw_relative + raw_relative_tail_focus_weight * raw_relative_tail
    return {
        "recon_continuous": recon,
        "recon_independent_binary": zero,
        "recon_continuous_raw_relative": raw_relative,
        "recon_continuous_raw_relative_tail": raw_relative_tail,
        "recon_pseudo_binary": zero,
        "recon_protocol": zero,
        "kl": kl,
        "per_dim_kl_mean": per_dim_kl.mean(dim=0),
        "constraint_loss": constraints["total"],
        "constraint_nonneg": constraints["nonneg"],
        "constraint_ttl": constraints["ttl"],
        "constraint_ordering": constraints["ordering"],
        "constraint_variance": constraints["variance"],
        "constraint_packet_positive": constraints["packet_positive"],
        "constraint_packet_integer": constraints["packet_integer"],
        "physics_constraint_loss": physics["total"],
        "physics_p2_total_bytes": physics["p2_total_bytes"],
        "physics_p4_std_bound": physics["p4_std_bound"],
        "physics_p5_singleton": physics["p5_singleton"],
        "loss": loss,
    }


class BetaScheduler:
    def __init__(self, beta_target: float, total_steps: int, warmup_frac: float = 0.3, warmup_steps: int | None = None) -> None:
        self.beta_target = beta_target
        self.total_steps = total_steps
        self.warmup_steps = int(warmup_steps) if warmup_steps is not None else int(warmup_frac * total_steps)
        self._step_count = 0
        self._current_beta = 0.0

    def step(self) -> float:
        if self.warmup_steps > 0 and self._step_count < self.warmup_steps:
            self._current_beta = self.beta_target * (self._step_count + 1) / self.warmup_steps
        else:
            self._current_beta = self.beta_target
        self._step_count += 1
        return self._current_beta

    @property
    def current_beta(self) -> float:
        return self._current_beta


def compute_manifold_elbo(
    batch_x_scaled: torch.Tensor,
    model_out: dict,
    engine,
    beta: float,
    *,
    continuous_likelihood: str = "gaussian",
    free_bits_lambda: float = 0.0,
    continuous_logvar_floor: float = -7.0,
    continuous_logvar_ceiling: float = 2.0,
    pre_projection_weight: float = 0.0,
    constraint_l1_weight: float = 1.0,
    constraint_l2_weight: float = 1.0,
    continuous_feature_weights: torch.Tensor | None = None,
) -> dict:
    """Generic ELBO for the typed VAE + layered ConstraintEngine.

    Responsibilities are separated (spec Step 7):
      recon + beta * KL
      + pre_projection_weight * C0(pre-projection raw)   [Layer-0 soft, usually 0
        for the typed decoder whose activations already satisfy Layer 0]
      + C1 (Layer-1) + C2 (Layer-2) soft penalties on the produced raw output.
    The hard Layer-0 projection itself is structural in the decoder, not a penalty.
    """
    mu = model_out["mu"]
    logvar = model_out["logvar"]
    reconstructed = model_out["continuous_mu"]
    raw = model_out["continuous_mu_raw"]
    eff_logvar = model_out["continuous_logvar"].clamp(continuous_logvar_floor, continuous_logvar_ceiling)
    target = batch_x_scaled

    if continuous_likelihood == "gaussian":
        per_feature = 0.5 * (eff_logvar + (target - reconstructed).square() / eff_logvar.exp() + _LOG_2PI)
    elif continuous_likelihood == "laplace":
        per_feature = _LOG_2 + eff_logvar + (target - reconstructed).abs() / eff_logvar.exp()
    else:
        raise ValueError("continuous_likelihood must be 'gaussian' or 'laplace'")
    if continuous_feature_weights is not None:
        per_feature = per_feature * continuous_feature_weights.to(per_feature.device).unsqueeze(0)
    recon = per_feature.sum(dim=1).mean()

    per_dim_kl = -0.5 * (1.0 + logvar - mu.square() - logvar.exp())
    kl = (per_dim_kl.clamp(min=float(free_bits_lambda)) if free_bits_lambda > 0 else per_dim_kl).sum(dim=1).mean()

    zero = mu.new_tensor(0.0)
    l_pre = engine.layer0.soft_penalty(raw) if pre_projection_weight > 0 else zero
    pen = engine.penalty(raw, w1=constraint_l1_weight, w2=constraint_l2_weight)

    loss = recon + beta * kl + pre_projection_weight * l_pre + pen["total"]
    return {
        "loss": loss,
        "recon_continuous": recon,
        "kl": kl,
        "per_dim_kl_mean": per_dim_kl.mean(dim=0),
        "pre_projection": l_pre,
        "constraint_l1": pen["c1"],
        "constraint_l2": pen["c2"],
        "constraint_total": pen["total"],
    }
