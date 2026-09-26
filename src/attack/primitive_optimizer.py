"""Quantization-aware search over CICIDS2017 primitive controls.

Every optimizer runs on one :class:`RealizedSearch`, which owns the attack space and the
success definition so optimizers cannot diverge on either:

* a candidate is a requested ``(p, delay, shape)``; it is projected into the hard per-flow
  box (integer bytes / integer microseconds), realized through the canonical primitive map
  with quantization, and scored by the victim on that realized flow;
* the victim-side objective is an :class:`AttackObjective`: targeted (reach ``class_id``,
  Benign by default) or untargeted (leave the true source class ``class_id``);
* success = the realized flow meets the objective AND the injected validity gate
  (validator_v2 ``hybrid_valid`` in every experiment runner) accepts it;
* each flow keeps one incumbent: a success beats a failure, successes are ranked by
  normalized primitive cost (then margin), failures by the objective margin
  (targeted: ``max(non-target logits) - target logit``; untargeted:
  ``source logit - max(non-source logits)``; negative = objective met);
* per-flow victim evaluations are counted (realized forward passes, surrogate forward passes
  of the continuous relaxation, backward passes) and optionally capped by a shared budget.

Optimizers:

* :func:`optimize_primitive_candidates` (Hybrid, the proposed/default search): exact integer
  padding enumeration in increasing cost order, then adaptive projected sign-momentum
  refinement of normalized controls with restarts, stall-triggered step halving and reset to
  the restart's best point.
* :func:`optimize_primitive_pgd` (Prim-PGD): fixed-step projected sign-momentum descent on the
  objective margin over normalized controls, clean start + uniform random restarts.
* :func:`optimize_primitive_cw` (Prim-C&W): projected Adam on
  ``cost(q) + c * max(margin + kappa, 0)`` with a per-flow binary search over ``c``.

Gradients come from the continuous relaxation (``generate(..., quantize=False)``); incumbent
selection only ever uses realized, quantized flows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from attack.realizability.base import PrimitiveCapabilities


CONTROL_NAMES = ("p", "delay", "shape")
CANDIDATE_IDENTITY = 0
CANDIDATE_EXACT_PADDING = 1
CANDIDATE_ADAPTIVE_CLEAN = 2
CANDIDATE_ADAPTIVE_RANDOM = 3
CANDIDATE_PGD_CLEAN = 4
CANDIDATE_PGD_RANDOM = 5
CANDIDATE_CW = 6
CANDIDATE_NAMES = {
    CANDIDATE_IDENTITY: "identity",
    CANDIDATE_EXACT_PADDING: "exact-padding",
    CANDIDATE_ADAPTIVE_CLEAN: "adaptive-clean",
    CANDIDATE_ADAPTIVE_RANDOM: "adaptive-random",
    CANDIDATE_PGD_CLEAN: "pgd-clean",
    CANDIDATE_PGD_RANDOM: "pgd-random",
    CANDIDATE_CW: "cw",
}
# Candidate rows per batched victim call during exact padding enumeration.
_PADDING_CHUNK_ROWS = 4096
# Flat candidates per victim forward call (memory bound for the FT-Transformer).
_VICTIM_CHUNK_ROWS = 8192
_NO_EVAL = torch.iinfo(torch.int64).max
# The canonical map copies a flow verbatim wherever p == 0 (resp. delay == 0), so the
# relaxation has an exactly-zero gradient at a zero control. Surrogate evaluations therefore
# use max(q, floor) on coordinates with headroom (straight-through gradient to q). Realized
# scoring never sees the floor.
SURROGATE_FLOOR = 1e-3

ValidityGate = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class AttackObjective:
    """Victim-side objective shared by the success predicate, the incumbent and every gradient.

    ``targeted``: the realized flow must be classified as ``class_id``.
    ``untargeted``: the realized flow must NOT be classified as ``class_id`` (the source class).
    ``margin`` is minimized by every optimizer; it is negative when the objective is met.
    """

    kind: str
    class_id: int

    def __post_init__(self) -> None:
        if self.kind not in ("targeted", "untargeted"):
            raise ValueError(f"unknown attack objective {self.kind!r}")
        if self.class_id < 0:
            raise ValueError("objective class_id must be a non-negative class index")

    def margin(self, logits: torch.Tensor) -> torch.Tensor:
        if self.kind == "targeted":
            return targeted_margin(logits, self.class_id)
        return -targeted_margin(logits, self.class_id)

    def hit(self, logits: torch.Tensor) -> torch.Tensor:
        pred = logits.argmax(1)
        return pred == self.class_id if self.kind == "targeted" else pred != self.class_id


TARGET_BENIGN = AttackObjective("targeted", 0)


@dataclass(frozen=True)
class PrimitiveOptimizationResult:
    requested: dict[str, torch.Tensor]
    projected: dict[str, torch.Tensor]
    adversarial_raw: torch.Tensor
    logits: torch.Tensor
    objective_margin: torch.Tensor
    normalized_cost: torch.Tensor
    valid: torch.Tensor
    success: torch.Tensor
    candidate_source: torch.Tensor
    # Per-flow victim evaluation accounting (int64 tensors, one entry per row).
    realized_evaluations: torch.Tensor
    surrogate_evaluations: torch.Tensor
    backward_evaluations: torch.Tensor
    # 1-based index (in realized + surrogate evaluations) of the first successful / first
    # objective-meeting realized candidate (validity ignored); -1 if never reached.
    first_success_evaluation: torch.Tensor
    first_objective_hit_evaluation: torch.Tensor
    # Restart / binary-search stage index of the first success; -1 for the identity and
    # exact-padding phase, -2 if never reached.
    first_success_phase: torch.Tensor
    iterations: int
    restarts: int

    @property
    def total_evaluations(self) -> torch.Tensor:
        return self.realized_evaluations + self.surrogate_evaluations


def targeted_margin(logits: torch.Tensor, target_class: int = 0) -> torch.Tensor:
    """``max(non-target logits) - target logit``; negative iff ``target_class`` is the argmax."""
    target = logits[:, target_class]
    masked = logits.clone()
    masked[:, target_class] = -torch.inf
    return masked.amax(1) - target


def hybrid_valid_gate(dataset: str) -> ValidityGate:
    """validator_v2 ``hybrid_valid`` of realized raw flows, as a device-preserving bool mask."""
    from validation.attack_interface import structural_masks

    def gate(adv_raw: torch.Tensor) -> torch.Tensor:
        valid = structural_masks(adv_raw.detach().cpu().numpy(), dataset=dataset)["hybrid_valid"]
        return torch.as_tensor(valid, dtype=torch.bool, device=adv_raw.device)

    return gate


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
    best_success: torch.Tensor,
    best_margin: torch.Tensor,
    best_cost: torch.Tensor,
    cand_success: torch.Tensor,
    cand_margin: torch.Tensor,
    cand_cost: torch.Tensor,
) -> torch.Tensor:
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


def _mask_q(q: torch.Tensor, bounds: dict[str, torch.Tensor]) -> torch.Tensor:
    """Zero normalized coordinates whose integer headroom is below one unit."""
    pad = (bounds["p"] >= 1.0).to(q.dtype)
    timing = (bounds["delay"] >= 1.0).to(q.dtype)
    return q * torch.stack((pad, timing, timing), 1)


def _position_within_row(rows: torch.Tensor) -> torch.Tensor:
    """0-based order of appearance of each entry among the entries of the same row."""
    order = torch.argsort(rows, stable=True)
    sorted_rows = rows[order]
    arange = torch.arange(rows.numel(), device=rows.device)
    boundary = torch.ones_like(sorted_rows, dtype=torch.bool)
    boundary[1:] = sorted_rows[1:] != sorted_rows[:-1]
    segment_start = torch.cummax(torch.where(boundary, arange, 0), 0).values
    position = torch.empty_like(arange)
    position[order] = arange - segment_start
    return position


class RealizedSearch:
    """Shared attack space, success predicate, incumbent, and query accounting."""

    def __init__(
        self,
        model,
        victim,
        raw: torch.Tensor,
        center: torch.Tensor,
        scale: torch.Tensor,
        bounds: dict[str, torch.Tensor],
        caps: PrimitiveCapabilities,
        *,
        validity_fn: ValidityGate | None,
        objective: AttackObjective = TARGET_BENIGN,
        eval_budget: int | None = None,
    ) -> None:
        if eval_budget is not None and eval_budget < 1:
            raise ValueError("eval_budget must be >= 1 (the identity evaluation)")
        self.model, self.victim = model, victim
        self.raw, self.center, self.scale = raw, center, scale
        self.bounds, self.caps = bounds, caps
        self.validity_fn = validity_fn
        self.objective = objective
        self.eval_budget = eval_budget
        n = raw.shape[0]
        device = raw.device
        self.n = n
        self.realized = torch.zeros(n, dtype=torch.int64, device=device)
        self.surrogate = torch.zeros(n, dtype=torch.int64, device=device)
        self.backward = torch.zeros(n, dtype=torch.int64, device=device)
        self.first_success = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.first_hit = torch.full((n,), -1, dtype=torch.int64, device=device)
        self.first_success_phase = torch.full((n,), -2, dtype=torch.int64, device=device)
        self.phase = -1
        self.iterations = 0
        self.restarts = 0

        rows = torch.arange(n, device=device)
        zeros = torch.zeros(n, dtype=raw.dtype, device=device)
        identity = {"p": zeros, "delay": zeros.clone(), "shape": zeros.clone()}
        with torch.no_grad():
            s = self._score(rows, identity)
        self.best_requested = {name: s["projected"][name].clone() for name in CONTROL_NAMES}
        self.best_projected = {name: s["projected"][name].clone() for name in CONTROL_NAMES}
        self.best_adv = s["adv"].clone()
        self.best_logits = s["logits"].clone()
        self.best_margin = s["margin"].clone()
        self.best_cost = s["cost"].clone()
        self.best_valid = s["valid"].clone()
        self.best_success = s["success"].clone()
        self.best_source = torch.full((n,), CANDIDATE_IDENTITY, dtype=torch.int8, device=device)
        self.realized += 1
        self.first_hit[s["hit"]] = 1
        self.first_success[s["success"]] = 1
        self.first_success_phase[s["success"]] = -1

    # -- accounting ---------------------------------------------------------------------
    @property
    def spent(self) -> torch.Tensor:
        return self.realized + self.surrogate

    def remaining(self, rows: torch.Tensor) -> torch.Tensor:
        if self.eval_budget is None:
            return torch.full(rows.shape, _NO_EVAL, dtype=torch.int64, device=rows.device)
        return self.eval_budget - self.spent[rows]

    # -- realized candidates ------------------------------------------------------------
    def _score(self, rows: torch.Tensor, requested: dict[str, torch.Tensor]) -> dict:
        parts = []
        for start in range(0, rows.numel(), _VICTIM_CHUNK_ROWS):
            r = rows[start:start + _VICTIM_CHUNK_ROWS]
            req = {name: requested[name][start:start + _VICTIM_CHUNK_ROWS] for name in CONTROL_NAMES}
            raw_s = self.raw[r]
            bounds_s = _subset_bounds(self.bounds, r)
            caps_s = _subset_capabilities(self.caps, r)
            projected = self.model.project_controls(raw_s, req, bounds_s, capabilities=caps_s)
            adv = self.model.generate(raw_s, projected, quantize=True, capabilities=caps_s)
            logits = self.victim((adv - self.center) / self.scale)
            hit = self.objective.hit(logits)
            valid = (
                self.validity_fn(adv) if self.validity_fn is not None
                else torch.ones_like(hit)
            )
            parts.append({
                "projected": projected, "adv": adv, "logits": logits,
                "margin": self.objective.margin(logits),
                "cost": _normalized_cost(projected, bounds_s),
                "hit": hit, "valid": valid, "success": hit & valid,
            })
        if len(parts) == 1:
            return parts[0]
        out = {k: torch.cat([p[k] for p in parts]) for k in parts[0] if k != "projected"}
        out["projected"] = {
            name: torch.cat([p["projected"][name] for p in parts]) for name in CONTROL_NAMES
        }
        return out

    @torch.no_grad()
    def evaluate(
        self,
        rows: torch.Tensor,
        requested: dict[str, torch.Tensor],
        source: int,
        *,
        stop_at_first_success: bool = False,
    ) -> dict:
        """Score realized candidates, charge evaluations, and update incumbents.

        ``rows`` may repeat; a row's candidates are its sequential queries in order of
        appearance. With ``stop_at_first_success`` a row's candidates after its first success
        are discarded and not charged (the batched call reproduces sequential early exit).
        """
        s = self._score(rows, requested)
        position = _position_within_row(rows)
        success, hit = s["success"], s["hit"]
        if stop_at_first_success:
            first = torch.full((self.n,), _NO_EVAL, dtype=torch.int64, device=rows.device)
            first.scatter_reduce_(
                0, rows, torch.where(success, position, _NO_EVAL), reduce="amin"
            )
            seen = position <= first[rows]
        else:
            seen = torch.ones_like(success)
        eval_index = self.spent[rows] + position + 1

        for flags, record in ((hit, self.first_hit), (success, self.first_success)):
            first_eval = torch.full((self.n,), _NO_EVAL, dtype=torch.int64, device=rows.device)
            first_eval.scatter_reduce_(
                0, rows, torch.where(flags & seen, eval_index, _NO_EVAL), reduce="amin"
            )
            new = (record < 0) & (first_eval < _NO_EVAL)
            if record is self.first_success:
                self.first_success_phase[new] = self.phase
            record[new] = first_eval[new]
        self.realized.index_add_(0, rows, seen.to(torch.int64))

        idx = torch.nonzero(seen, as_tuple=False).flatten()
        if idx.numel():
            margin, cost, cand_rows = s["margin"], s["cost"], rows
            secondary = torch.where(success, cost, margin)
            idx = idx[torch.argsort(margin[idx], stable=True)]
            idx = idx[torch.argsort(secondary[idx], stable=True)]
            idx = idx[torch.argsort((~success[idx]).to(torch.int8), stable=True)]
            idx = idx[torch.argsort(cand_rows[idx], stable=True)]
            ordered_rows = cand_rows[idx]
            first_of_row = torch.ones_like(ordered_rows, dtype=torch.bool)
            first_of_row[1:] = ordered_rows[1:] != ordered_rows[:-1]
            pick = idx[first_of_row]
            r = cand_rows[pick]
            take = _candidate_take(
                self.best_success[r], self.best_margin[r], self.best_cost[r],
                success[pick], margin[pick], cost[pick],
            )
            pick, r = pick[take], r[take]
            if r.numel():
                self.best_adv[r] = s["adv"][pick]
                self.best_logits[r] = s["logits"][pick]
                self.best_margin[r] = margin[pick]
                self.best_cost[r] = cost[pick]
                self.best_valid[r] = s["valid"][pick]
                self.best_success[r] = success[pick]
                self.best_source[r] = source
                for name in CONTROL_NAMES:
                    self.best_projected[name][r] = s["projected"][name][pick]
                    self.best_requested[name][r] = requested[name][pick]
        s["seen"] = seen
        return s

    # -- continuous relaxation ----------------------------------------------------------
    def surrogate_logits(self, rows: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Victim logits on the unquantized relaxation at normalized controls ``q``.

        The relaxation is evaluated at ``max(q, SURROGATE_FLOOR)`` on coordinates with
        headroom (value floored, gradient passed straight through to ``q``). Coordinates
        without integer headroom are pinned at 0 and are not optimization variables: their
        autograd path is cut, so their gradient is exactly 0 (a non-finite partial derivative
        of the map at a pinned control, e.g. padding on a flow without padding capability,
        would otherwise turn the whole step into NaN). Charges one surrogate forward and one
        backward per row (callers differentiate via :func:`objective_gradient`).
        """
        bounds_s = _subset_bounds(self.bounds, rows)
        floor = _mask_q(torch.full_like(q, SURROGATE_FLOOR), bounds_s)
        q_eval = q + (floor - q).clamp(min=0.0).detach()
        free = _mask_q(torch.ones_like(q), bounds_s) > 0
        q_eval = torch.where(free, q_eval, q_eval.detach())
        transformed = self.model.generate(
            self.raw[rows], _controls_from_q(q_eval, bounds_s), quantize=False,
            capabilities=_subset_capabilities(self.caps, rows),
        )
        self.surrogate.index_add_(0, rows, torch.ones_like(rows))
        self.backward.index_add_(0, rows, torch.ones_like(rows))
        return self.victim((transformed - self.center) / self.scale)

    def gradient_rows(self, rows: torch.Tensor) -> torch.Tensor:
        """Positions in ``rows`` that can still afford one gradient step (2 evaluations)."""
        return torch.nonzero(self.remaining(rows) >= 2, as_tuple=False).flatten()

    def result(self) -> PrimitiveOptimizationResult:
        return PrimitiveOptimizationResult(
            requested={name: v.detach() for name, v in self.best_requested.items()},
            projected={name: v.detach() for name, v in self.best_projected.items()},
            adversarial_raw=self.best_adv.detach(),
            logits=self.best_logits.detach(),
            objective_margin=self.best_margin.detach(),
            normalized_cost=self.best_cost.detach(),
            valid=self.best_valid.detach(),
            success=self.best_success.detach(),
            candidate_source=self.best_source.detach(),
            realized_evaluations=self.realized.clone(),
            surrogate_evaluations=self.surrogate.clone(),
            backward_evaluations=self.backward.clone(),
            first_success_evaluation=self.first_success.clone(),
            first_objective_hit_evaluation=self.first_hit.clone(),
            first_success_phase=self.first_success_phase.clone(),
            iterations=self.iterations,
            restarts=self.restarts,
        )


def objective_gradient(loss: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """``d loss / d q``; fails loudly instead of letting a non-finite step silently freeze or
    corrupt a flow's controls."""
    grad = torch.autograd.grad(loss, q)[0]
    if not bool(torch.isfinite(grad).all()):
        raise FloatingPointError("non-finite gradient of the PrimAttack surrogate objective")
    return grad


def _movable_rows(bounds: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.nonzero(
        (bounds["p"] >= 1.0) | (bounds["delay"] >= 1.0), as_tuple=False
    ).flatten()


def _random_q(shape, bounds, generator, like: torch.Tensor) -> torch.Tensor:
    q = torch.rand(shape, generator=generator, device=like.device, dtype=like.dtype)
    return _mask_q(q, bounds)


# ----------------------------------------------------------------------------------------
# Hybrid Search (proposed / default PrimAttack optimizer)
# ----------------------------------------------------------------------------------------
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
    validity_fn: ValidityGate | None,
    objective: AttackObjective = TARGET_BENIGN,
    restarts: int | None = 2,
    eval_budget: int | None = None,
) -> PrimitiveOptimizationResult:
    """Hybrid Search: exact padding enumeration, then adaptive projected timing/shape search.

    ``restarts=None`` keeps starting refinement restarts until every refined row has spent
    ``eval_budget`` (requires a budget).
    """
    if steps < 0 or learning_rate <= 0 or (restarts is not None and restarts < 1):
        raise ValueError("invalid primitive optimizer configuration")
    if restarts is None and eval_budget is None:
        raise ValueError("restarts=None (fill the budget) requires eval_budget")
    search = RealizedSearch(
        model, victim, raw, center, scale, bounds, caps,
        validity_fn=validity_fn, objective=objective, eval_budget=eval_budget,
    )
    n = raw.shape[0]
    zeros = torch.zeros(n, dtype=raw.dtype, device=raw.device)

    # Exact discrete padding search in increasing cost order. A row leaves the sweep once it
    # succeeds (larger padding can only cost more) or once its integer cap / query budget is
    # exhausted. Consecutive padding values are scored in one batched victim call; per row,
    # values after the first success are discarded and not charged, which is exactly the
    # outcome of scoring the values one at a time.
    padding_cap = torch.floor(bounds["p"])
    max_padding = int(padding_cap.max().item()) if n else 0
    value = 1
    all_rows = torch.arange(n, device=raw.device)
    with torch.no_grad():
        while value <= max_padding:
            remaining = search.remaining(all_rows)
            pending = (padding_cap >= value) & ~search.best_success & (remaining >= 1)
            rows = torch.nonzero(pending, as_tuple=False).flatten()
            m = rows.numel()
            if not m:
                break  # pending sets only shrink as value grows
            k = max(1, min(max_padding - value + 1, _PADDING_CHUNK_ROWS // m))
            values = torch.arange(value, value + k, dtype=raw.dtype, device=raw.device)
            offsets = torch.arange(k, device=raw.device).repeat_interleave(m)
            flat_rows = rows.repeat(k)
            flat_values = values.repeat_interleave(m)
            usable = (flat_values <= padding_cap[flat_rows]) & (offsets < remaining[flat_rows])
            flat_rows, flat_values = flat_rows[usable], flat_values[usable]
            flat_zeros = zeros[flat_rows]
            search.evaluate(
                flat_rows, {"p": flat_values, "delay": flat_zeros, "shape": flat_zeros},
                CANDIDATE_EXACT_PADDING, stop_at_first_success=True,
            )
            value += k

    # Rows without integer delay headroom can only pad, and padding was enumerated exactly.
    active = torch.nonzero(
        ~search.best_success & (bounds["delay"] >= 1.0), as_tuple=False
    ).flatten()
    if steps and active.numel():
        bounds_a = _subset_bounds(bounds, active)
        generator = torch.Generator(device=raw.device).manual_seed(seed)
        base_q = _q_from_controls(
            {name: search.best_projected[name][active] for name in CONTROL_NAMES}, bounds_a
        )
        checkpoint_interval = max(5, steps // 4)
        restart = 0
        while (restarts is None or restart < restarts) and search.gradient_rows(active).numel():
            search.phase = restart
            search.restarts += 1
            if restart == 0:
                q = base_q.clone()
                source = CANDIDATE_ADAPTIVE_CLEAN
            else:
                q = _random_q(base_q.shape, bounds_a, generator, base_q)
                source = CANDIDATE_ADAPTIVE_RANDOM
            velocity = torch.zeros_like(q)
            step_size = torch.full((len(active), 1), learning_rate, dtype=raw.dtype,
                                   device=raw.device)
            restart_best_margin = torch.full((len(active),), torch.inf, dtype=raw.dtype,
                                             device=raw.device)
            restart_best_q = q.clone()
            checkpoint_margin = restart_best_margin.clone()

            for iteration in range(steps):
                alive = search.gradient_rows(active)
                if not alive.numel():
                    break
                rows = active[alive]
                bounds_s = _subset_bounds(bounds_a, alive)
                q_alive = q[alive].requires_grad_(True)
                logits = search.surrogate_logits(rows, q_alive)
                grad = objective_gradient(objective.margin(logits).sum(), q_alive)
                search.iterations += 1
                with torch.no_grad():
                    grad_scale = grad.abs().mean(1, keepdim=True).clamp(min=1e-12)
                    velocity[alive] = 0.75 * velocity[alive] + grad / grad_scale
                    q_new = _mask_q(
                        (q[alive] - step_size[alive] * velocity[alive].sign()).clamp(0.0, 1.0),
                        bounds_s,
                    )
                    q[alive] = q_new
                    out = search.evaluate(rows, _controls_from_q(q_new, bounds_s), source)
                    improved = out["margin"] < restart_best_margin[alive]
                    restart_best_margin[alive] = torch.where(
                        improved, out["margin"], restart_best_margin[alive]
                    )
                    restart_best_q[alive] = torch.where(
                        improved[:, None], q_new, restart_best_q[alive]
                    )
                    if (iteration + 1) % checkpoint_interval == 0:
                        stalled = restart_best_margin >= checkpoint_margin - 1e-6
                        step_size[stalled] *= 0.5
                        q[stalled] = restart_best_q[stalled]
                        velocity[stalled] = 0.0
                        checkpoint_margin = restart_best_margin.clone()
            restart += 1
    return search.result()


# ----------------------------------------------------------------------------------------
# Prim-PGD
# ----------------------------------------------------------------------------------------
def optimize_primitive_pgd(
    model,
    victim,
    raw: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    bounds: dict[str, torch.Tensor],
    caps: PrimitiveCapabilities,
    *,
    steps: int,
    step_size: float,
    restarts: int,
    seed: int,
    validity_fn: ValidityGate | None,
    momentum: float = 0.75,
    objective: AttackObjective = TARGET_BENIGN,
    eval_budget: int | None = None,
) -> PrimitiveOptimizationResult:
    """Fixed-step projected sign-momentum descent on the objective margin.

    Restart 0 starts at the clean flow (q=0); later restarts start uniformly in the box.
    No padding enumeration, no step adaptation, no reset to a restart's best point.
    """
    if steps < 1 or step_size <= 0 or restarts < 1 or not 0.0 <= momentum < 1.0:
        raise ValueError("invalid Prim-PGD configuration")
    search = RealizedSearch(
        model, victim, raw, center, scale, bounds, caps,
        validity_fn=validity_fn, objective=objective, eval_budget=eval_budget,
    )
    active = _movable_rows(bounds)
    if not active.numel():
        return search.result()
    bounds_a = _subset_bounds(bounds, active)
    generator = torch.Generator(device=raw.device).manual_seed(seed)
    for restart in range(restarts):
        if not search.gradient_rows(active).numel():
            break
        search.phase = restart
        search.restarts += 1
        if restart == 0:
            q = torch.zeros((len(active), 3), dtype=raw.dtype, device=raw.device)
            source = CANDIDATE_PGD_CLEAN
        else:
            q = _random_q((len(active), 3), bounds_a, generator, raw)
            source = CANDIDATE_PGD_RANDOM
        velocity = torch.zeros_like(q)
        for _ in range(steps):
            alive = search.gradient_rows(active)
            if not alive.numel():
                break
            rows = active[alive]
            bounds_s = _subset_bounds(bounds_a, alive)
            q_alive = q[alive].requires_grad_(True)
            logits = search.surrogate_logits(rows, q_alive)
            grad = objective_gradient(objective.margin(logits).sum(), q_alive)
            search.iterations += 1
            with torch.no_grad():
                grad_scale = grad.abs().mean(1, keepdim=True).clamp(min=1e-12)
                velocity[alive] = momentum * velocity[alive] + grad / grad_scale
                q_new = _mask_q(
                    (q[alive] - step_size * velocity[alive].sign()).clamp(0.0, 1.0), bounds_s
                )
                q[alive] = q_new
                search.evaluate(rows, _controls_from_q(q_new, bounds_s), source)
    return search.result()


# ----------------------------------------------------------------------------------------
# Prim-C&W
# ----------------------------------------------------------------------------------------
def optimize_primitive_cw(
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
    stages: int,
    validity_fn: ValidityGate | None,
    c_init: float = 1.0,
    kappa: float = 0.0,
    betas: tuple[float, float] = (0.9, 0.999),
    objective: AttackObjective = TARGET_BENIGN,
    eval_budget: int | None = None,
) -> PrimitiveOptimizationResult:
    """Projected Adam on ``cost(q) + c * max(margin + kappa, 0)`` with binary search on ``c``.

    ``cost(q) = q_p + q_delay`` is the incumbent's normalized primitive cost (shape is free).
    Every stage restarts from the clean flow; per flow, ``c`` is halved toward the lower
    bracket after a stage with a realized success and multiplied by 10 (or bisected once an
    upper bracket exists) otherwise, as in Carlini & Wagner.
    """
    if steps < 1 or learning_rate <= 0 or stages < 1 or c_init <= 0 or kappa < 0:
        raise ValueError("invalid Prim-C&W configuration")
    search = RealizedSearch(
        model, victim, raw, center, scale, bounds, caps,
        validity_fn=validity_fn, objective=objective, eval_budget=eval_budget,
    )
    active = _movable_rows(bounds)
    if not active.numel():
        return search.result()
    bounds_a = _subset_bounds(bounds, active)
    pad_on = (bounds_a["p"] >= 1.0).to(raw.dtype)
    timing_on = (bounds_a["delay"] >= 1.0).to(raw.dtype)
    beta1, beta2 = betas
    c = torch.full((len(active),), c_init, dtype=raw.dtype, device=raw.device)
    lower = torch.zeros_like(c)
    upper = torch.full_like(c, 1e10)
    for stage in range(stages):
        if not search.gradient_rows(active).numel():
            break
        search.phase = stage
        search.restarts += 1
        q = torch.zeros((len(active), 3), dtype=raw.dtype, device=raw.device)
        m1 = torch.zeros_like(q)
        m2 = torch.zeros_like(q)
        t = torch.zeros((len(active), 1), dtype=raw.dtype, device=raw.device)
        stage_success = torch.zeros(len(active), dtype=torch.bool, device=raw.device)
        for _ in range(steps):
            alive = search.gradient_rows(active)
            if not alive.numel():
                break
            rows = active[alive]
            bounds_s = _subset_bounds(bounds_a, alive)
            q_alive = q[alive].requires_grad_(True)
            logits = search.surrogate_logits(rows, q_alive)
            margin = objective.margin(logits)
            cost = q_alive[:, 0] * pad_on[alive] + q_alive[:, 1] * timing_on[alive]
            loss = (cost + c[alive] * (margin + kappa).clamp(min=0.0)).sum()
            grad = objective_gradient(loss, q_alive)
            search.iterations += 1
            with torch.no_grad():
                t[alive] += 1.0
                m1[alive] = beta1 * m1[alive] + (1.0 - beta1) * grad
                m2[alive] = beta2 * m2[alive] + (1.0 - beta2) * grad * grad
                m_hat = m1[alive] / (1.0 - beta1 ** t[alive])
                v_hat = m2[alive] / (1.0 - beta2 ** t[alive])
                q_new = _mask_q(
                    (q[alive] - learning_rate * m_hat / (v_hat.sqrt() + 1e-8)).clamp(0.0, 1.0),
                    bounds_s,
                )
                q[alive] = q_new
                out = search.evaluate(rows, _controls_from_q(q_new, bounds_s), CANDIDATE_CW)
                stage_success[alive] |= out["success"]
        with torch.no_grad():
            upper = torch.where(stage_success, torch.minimum(upper, c), upper)
            lower = torch.where(stage_success, lower, torch.maximum(lower, c))
            bracketed = upper < 1e9
            c = torch.where(bracketed, (lower + upper) / 2.0, c * 10.0)
    return search.result()
