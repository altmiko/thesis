"""Quantization-aware search over CICIDS2017 primitive controls.

The optimizer treats the canonical primitive map and hard per-flow bounds as the feasible
set. Integer padding is enumerated exactly. Only rows unresolved by that deterministic
search enter an adaptive projected-gradient refinement of normalized controls. Candidate
selection always uses the final projected, quantized flow seen by the victim.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from attack.realizability.base import PrimitiveCapabilities


CONTROL_NAMES = ("p", "delay", "shape")
CANDIDATE_IDENTITY = 0
CANDIDATE_EXACT_PADDING = 1
CANDIDATE_ADAPTIVE_CLEAN = 2
CANDIDATE_ADAPTIVE_RANDOM = 3
CANDIDATE_NAMES = {
    CANDIDATE_IDENTITY: "identity",
    CANDIDATE_EXACT_PADDING: "exact-padding",
    CANDIDATE_ADAPTIVE_CLEAN: "adaptive-clean",
    CANDIDATE_ADAPTIVE_RANDOM: "adaptive-random",
}
# Candidate rows per batched victim call during exact padding enumeration.
_PADDING_CHUNK_ROWS = 4096


@dataclass(frozen=True)
class PrimitiveOptimizationResult:
    requested: dict[str, torch.Tensor]
    projected: dict[str, torch.Tensor]
    adversarial_raw: torch.Tensor
    logits: torch.Tensor
    target_margin: torch.Tensor
    normalized_cost: torch.Tensor
    candidate_source: torch.Tensor
    forward_evaluations: int
    backward_evaluations: int


def targeted_margin(logits: torch.Tensor, target_class: int = 0) -> torch.Tensor:
    target = logits[:, target_class]
    masked = logits.clone()
    masked[:, target_class] = -torch.inf
    return masked.amax(1) - target


def _subset_capabilities(caps: PrimitiveCapabilities, rows: torch.Tensor) -> PrimitiveCapabilities:
    return PrimitiveCapabilities(
        pad_allowed=caps.pad_allowed[rows],
        timing_allowed=caps.timing_allowed[rows],
        pad_reason=[],
        timing_reason=[],
    )


def _subset_bounds(bounds: dict[str, torch.Tensor], rows: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[rows] for name, value in bounds.items()}


def _normalized_cost(
    controls: dict[str, torch.Tensor], bounds: dict[str, torch.Tensor]
) -> torch.Tensor:
    p = torch.where(bounds["p"] > 0, controls["p"] / bounds["p"].clamp(min=1.0), 0.0)
    delay = torch.where(
        bounds["delay"] > 0,
        controls["delay"] / bounds["delay"].clamp(min=1.0),
        0.0,
    )
    return p + delay


def _candidate_take(
    best_logits: torch.Tensor,
    best_margin: torch.Tensor,
    best_cost: torch.Tensor,
    cand_logits: torch.Tensor,
    cand_margin: torch.Tensor,
    cand_cost: torch.Tensor,
    target_class: int,
) -> torch.Tensor:
    best_success = best_logits.argmax(1) == target_class
    cand_success = cand_logits.argmax(1) == target_class
    success_upgrade = cand_success & ~best_success
    same_status = cand_success == best_success
    lower_failure_margin = ~cand_success & (cand_margin < best_margin)
    lower_success_cost = cand_success & (
        (cand_cost < best_cost - 1e-12)
        | ((cand_cost <= best_cost + 1e-12) & (cand_margin < best_margin))
    )
    return success_upgrade | (same_status & (lower_failure_margin | lower_success_cost))


def _controls_from_q(q: torch.Tensor, bounds: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "p": bounds["p"] * q[:, 0],
        "delay": bounds["delay"] * q[:, 1],
        "shape": bounds["shape"] * q[:, 2],
    }


def _q_from_controls(
    controls: dict[str, torch.Tensor], bounds: dict[str, torch.Tensor]
) -> torch.Tensor:
    p = torch.where(bounds["p"] > 0, controls["p"] / bounds["p"].clamp(min=1.0), 0.0)
    delay = torch.where(
        bounds["delay"] > 0,
        controls["delay"] / bounds["delay"].clamp(min=1.0),
        0.0,
    )
    shape = torch.where(bounds["shape"] > 0, controls["shape"], 0.0)
    return torch.stack((p, delay, shape), 1).clamp(0.0, 1.0)


def optimize_primitive_candidates(
    model,
    victim,
    raw: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    bounds: dict[str, torch.Tensor],
    caps: PrimitiveCapabilities,
    *,
    steps: int,
    learning_rate: float,
    seed: int,
    target_class: int = 0,
    restarts: int = 2,
) -> PrimitiveOptimizationResult:
    """Return the strongest projected candidate per row, with cost as a success tie-breaker."""
    if steps < 0 or learning_rate <= 0 or restarts < 1:
        raise ValueError("invalid primitive optimizer configuration")
    n = raw.shape[0]
    zeros = torch.zeros(n, dtype=raw.dtype, device=raw.device)
    identity = {"p": zeros, "delay": zeros.clone(), "shape": zeros.clone()}
    with torch.no_grad():
        best_projected = model.project_controls(
            raw, identity, bounds, capabilities=caps
        )
        best_adv = model.generate(
            raw, best_projected, quantize=True, capabilities=caps
        )
        best_logits = victim((best_adv - center) / scale)
        best_margin = targeted_margin(best_logits, target_class)
        best_cost = _normalized_cost(best_projected, bounds)
    best_requested = {name: value.clone() for name, value in best_projected.items()}
    best_source = torch.full(
        (n,), CANDIDATE_IDENTITY, dtype=torch.int8, device=raw.device
    )
    forward_evaluations = 1
    backward_evaluations = 0

    def evaluate(rows: torch.Tensor, requested: dict[str, torch.Tensor]):
        nonlocal forward_evaluations
        raw_s = raw[rows]
        bounds_s = _subset_bounds(bounds, rows)
        caps_s = _subset_capabilities(caps, rows)
        projected = model.project_controls(raw_s, requested, bounds_s, capabilities=caps_s)
        adv = model.generate(raw_s, projected, quantize=True, capabilities=caps_s)
        logits = victim((adv - center) / scale)
        forward_evaluations += 1
        return (
            projected, adv, logits, targeted_margin(logits, target_class),
            _normalized_cost(projected, bounds_s),
        )

    def commit(rows, requested, projected, adv, logits, margin, cost, source) -> None:
        take_local = _candidate_take(
            best_logits[rows], best_margin[rows], best_cost[rows],
            logits, margin, cost, target_class,
        )
        take_rows = rows[take_local]
        if take_rows.numel():
            best_adv[take_rows] = adv[take_local]
            best_logits[take_rows] = logits[take_local]
            best_margin[take_rows] = margin[take_local]
            best_cost[take_rows] = cost[take_local]
            best_source[take_rows] = source
            for name in CONTROL_NAMES:
                best_projected[name][take_rows] = projected[name][take_local]
                best_requested[name][take_rows] = requested[name][take_local]

    # Exact discrete padding search in increasing cost order. A row leaves the sweep once it
    # succeeds (larger padding can only cost more) or once its integer cap is exhausted.
    # Consecutive padding values are scored in one batched victim call; per row, the smallest
    # successful value (else the lowest-margin value) of the chunk is committed, which is
    # exactly the outcome of scoring the values one at a time.
    padding_cap = torch.floor(bounds["p"])
    max_padding = int(padding_cap.max().item()) if n else 0
    value = 1
    with torch.no_grad():
        while value <= max_padding:
            pending = (padding_cap >= value) & (best_logits.argmax(1) != target_class)
            rows = torch.nonzero(pending, as_tuple=False).flatten()
            m = rows.numel()
            if not m:
                break  # pending sets only shrink as value grows
            k = max(1, min(max_padding - value + 1, _PADDING_CHUNK_ROWS // m))
            values = torch.arange(value, value + k, dtype=raw.dtype, device=raw.device)
            flat_rows = rows.repeat(k)
            flat_values = values.repeat_interleave(m)
            usable = flat_values <= padding_cap[flat_rows]
            flat_zeros = zeros[flat_rows]
            requested = {
                "p": torch.minimum(flat_values, padding_cap[flat_rows]),
                "delay": flat_zeros, "shape": flat_zeros,
            }
            projected, adv, logits, margin, cost = evaluate(flat_rows, requested)
            success = ((logits.argmax(1) == target_class) & usable).view(k, m)
            ranked_margin = torch.where(usable, margin, torch.inf).view(k, m)
            choice = torch.where(
                success.any(0), success.to(torch.int8).argmax(0), ranked_margin.argmin(0)
            )
            pick = choice * m + torch.arange(m, device=raw.device)
            commit(
                rows,
                {name: value_[pick] for name, value_ in requested.items()},
                {name: value_[pick] for name, value_ in projected.items()},
                adv[pick], logits[pick], margin[pick], cost[pick],
                CANDIDATE_EXACT_PADDING,
            )
            value += k

    unresolved = best_logits.argmax(1) != target_class
    # Rows without integer delay headroom can only pad, and padding was enumerated exactly.
    movable = bounds["delay"] >= 1.0
    active_rows = torch.nonzero(unresolved & movable, as_tuple=False).flatten()
    if steps and active_rows.numel():
        raw_s = raw[active_rows]
        bounds_s = _subset_bounds(bounds, active_rows)
        caps_s = _subset_capabilities(caps, active_rows)
        generator = torch.Generator(device=raw.device).manual_seed(seed)
        base_q = _q_from_controls(
            {name: best_projected[name][active_rows] for name in CONTROL_NAMES},
            bounds_s,
        )
        checkpoint_interval = max(5, steps // 4)

        for restart in range(restarts):
            if restart == 0:
                q = base_q.clone()
                source = CANDIDATE_ADAPTIVE_CLEAN
            else:
                q = torch.rand(
                    base_q.shape,
                    generator=generator,
                    device=raw.device,
                    dtype=raw.dtype,
                )
                q[:, 0] = torch.where(bounds_s["p"] >= 1.0, q[:, 0], 0.0)
                q[:, 1] = torch.where(bounds_s["delay"] >= 1.0, q[:, 1], 0.0)
                q[:, 2] = torch.where(bounds_s["delay"] >= 1.0, q[:, 2], 0.0)
                source = CANDIDATE_ADAPTIVE_RANDOM
            q.requires_grad_(True)
            velocity = torch.zeros_like(q)
            step_size = torch.full(
                (len(active_rows), 1), learning_rate,
                dtype=raw.dtype,
                device=raw.device,
            )
            restart_best_margin = torch.full(
                (len(active_rows),), torch.inf, dtype=raw.dtype, device=raw.device
            )
            restart_best_q = q.detach().clone()
            checkpoint_margin = restart_best_margin.clone()

            for iteration in range(steps):
                controls = _controls_from_q(q, bounds_s)
                transformed = model.generate(
                    raw_s, controls, quantize=False, capabilities=caps_s
                )
                logits = victim((transformed - center) / scale)
                loss = targeted_margin(logits, target_class).sum()
                grad = torch.autograd.grad(loss, q)[0]
                backward_evaluations += 1
                forward_evaluations += 1
                grad_scale = grad.abs().mean(1, keepdim=True).clamp(min=1e-12)
                velocity.mul_(0.75).add_(grad / grad_scale)
                with torch.no_grad():
                    q.add_(-step_size * velocity.sign()).clamp_(0.0, 1.0)
                    q[:, 0] = torch.where(bounds_s["p"] >= 1.0, q[:, 0], 0.0)
                    q[:, 1] = torch.where(bounds_s["delay"] >= 1.0, q[:, 1], 0.0)
                    q[:, 2] = torch.where(bounds_s["delay"] >= 1.0, q[:, 2], 0.0)
                    requested = _controls_from_q(q, bounds_s)
                    projected, adv, realized_logits, realized_margin, candidate_cost = evaluate(
                        active_rows, requested
                    )
                    improved = realized_margin < restart_best_margin
                    restart_best_margin = torch.where(
                        improved, realized_margin, restart_best_margin
                    )
                    restart_best_q = torch.where(improved[:, None], q, restart_best_q)
                    commit(
                        active_rows, requested, projected, adv, realized_logits,
                        realized_margin, candidate_cost, source,
                    )

                    if (iteration + 1) % checkpoint_interval == 0:
                        stalled = restart_best_margin >= checkpoint_margin - 1e-6
                        step_size[stalled] *= 0.5
                        q[stalled] = restart_best_q[stalled]
                        velocity[stalled] = 0.0
                        checkpoint_margin = restart_best_margin.clone()
                q = q.detach().requires_grad_(True)

    return PrimitiveOptimizationResult(
        requested={name: value.detach() for name, value in best_requested.items()},
        projected={name: value.detach() for name, value in best_projected.items()},
        adversarial_raw=best_adv.detach(),
        logits=best_logits.detach(),
        target_margin=best_margin.detach(),
        normalized_cost=best_cost.detach(),
        candidate_source=best_source.detach(),
        forward_evaluations=forward_evaluations,
        backward_evaluations=backward_evaluations,
    )
