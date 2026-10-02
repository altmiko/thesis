"""Configurable Hybrid Search for the PrimAttack ablations.

``optimize_hybrid_ablation`` is a transcription of
``attack.primitive_optimizer.optimize_primitive_candidates`` (the canonical Hybrid Search) in
which each component can be switched off or replaced through :class:`HybridConfig`. With the
default config it performs exactly the canonical sequence of projections, victim calls and
updates; ``ablations/reference`` verifies this flow-for-flow against the FINAL suite's Hybrid
cells. The canonical module is not modified: it is the locked FINAL-suite code.

Every variant keeps the shared :class:`~attack.primitive_optimizer.RealizedSearch` (attack space,
projection, quantized realization, victim scoring, query accounting), so an ablation changes only
the component it names. The one exception is ``realization_aware_search=False`` (ablation P1):
the search scores its candidates on the continuous primitive state (:class:`ContinuousSearch`)
and the canonical realization is applied once, to the returned candidate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F

from attack.primitive_optimizer import (
    _PADDING_CHUNK_ROWS,
    CANDIDATE_ADAPTIVE_CLEAN,
    CANDIDATE_ADAPTIVE_RANDOM,
    CANDIDATE_EXACT_PADDING,
    CONTROL_NAMES,
    SURROGATE_FLOOR,
    TARGET_BENIGN,
    _VICTIM_CHUNK_ROWS,
    AttackObjective,
    PrimitiveOptimizationResult,
    RealizedSearch,
    _controls_from_q,
    _mask_q,
    _normalized_cost,
    _q_from_controls,
    _random_q,
    _subset_bounds,
    _subset_capabilities,
    objective_gradient,
)

LOSSES = ("margin", "ce", "dlr", "mimicry")
SELECTIONS = ("incumbent", "last")


@dataclass(frozen=True)
class HybridConfig:
    """Hybrid Search components. Defaults = the FINAL-suite Hybrid (``PRIM_ARGS``)."""

    steps: int = 40
    learning_rate: float = 0.1
    eval_budget: int = 256
    restarts: int | None = None          # None: start restarts until the budget is spent
    padding_sweep: bool = True           # stage 2: exhaustive integer padding enumeration
    refinement: bool = True              # stage 3: adaptive projected refinement
    adaptive_step: bool = True           # stall-triggered step halving + reset to restart best
    momentum: float = 0.75
    surrogate_floor: float = SURROGATE_FLOOR
    validity_in_search: bool = True      # validator gate inside the search success predicate
    selection: str = "incumbent"         # "incumbent": success-first lowest cost; "last": last iterate
    fixed_shape: float | None = None     # None: shape is optimized; else pinned to this value
    loss: str = "margin"                 # gradient loss of the refinement stage
    # False (ablation P1): candidates are scored and selected on the continuous primitive state;
    # the canonical realization (integer bytes / µs, box, M(x), quantized φ) is applied once, to
    # the returned candidate, and costs one of the ``eval_budget`` evaluations.
    realization_aware_search: bool = True

    def __post_init__(self) -> None:
        if self.steps < 0 or self.learning_rate <= 0 or self.eval_budget < 1:
            raise ValueError("invalid Hybrid configuration")
        if self.restarts is not None and self.restarts < 1:
            raise ValueError("restarts must be None or >= 1")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must lie in [0, 1)")
        if self.surrogate_floor < 0.0:
            raise ValueError("surrogate_floor must be non-negative")
        if self.selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if self.fixed_shape is not None and not 0.0 <= self.fixed_shape <= 1.0:
            raise ValueError("fixed_shape must lie in [0, 1]")
        if not self.realization_aware_search and self.validity_in_search:
            raise ValueError("validator_v2 judges realized flows: a continuous-state search "
                             "(realization_aware_search=False) needs validity_in_search=False")
        if not self.realization_aware_search and self.eval_budget < 2:
            raise ValueError("a continuous-state search needs eval_budget >= 2 (identity + "
                             "final realization)")

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.realization_aware_search:  # omitted when canonical: keeps earlier configs comparable
            del d["realization_aware_search"]
        return d


@dataclass(frozen=True)
class SearchDiagnostics:
    """Per-row gradient diagnostics of the refinement stage."""

    gradient_steps: torch.Tensor        # refinement gradient steps taken by the row
    zero_gradient_steps: torch.Tensor   # steps whose gradient was exactly 0 on every free coordinate
    nonfinite_gradient_steps: torch.Tensor  # steps with a non-finite gradient (zeroed; D6 only)
    # Per-row arrays an arm records in addition (P1: the continuous search state before the
    # final realization). Saved verbatim to the cell npz.
    extra: dict[str, torch.Tensor] = field(default_factory=dict)


class AblationSearch(RealizedSearch):
    """``RealizedSearch`` with a configurable surrogate floor and last-iterate tracking."""

    def __init__(self, *args, surrogate_floor: float, shape_free: bool, track_last: bool,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.surrogate_floor = surrogate_floor
        self.shape_free = shape_free
        self.track_last = track_last
        if track_last:
            self.last_requested = {k: v.clone() for k, v in self.best_requested.items()}
            self.last_projected = {k: v.clone() for k, v in self.best_projected.items()}
            self.last_adv = self.best_adv.clone()
            self.last_logits = self.best_logits.clone()
            self.last_margin = self.best_margin.clone()
            self.last_cost = self.best_cost.clone()
            self.last_valid = self.best_valid.clone()
            self.last_success = self.best_success.clone()
            self.last_source = self.best_source.clone()

    @torch.no_grad()
    def evaluate(self, rows, requested, source, *, stop_at_first_success=False):
        s = super().evaluate(rows, requested, source, stop_at_first_success=stop_at_first_success)
        if not self.track_last:
            return s
        idx = torch.nonzero(s["seen"], as_tuple=False).flatten()
        if idx.numel():
            r = rows[idx]
            order = torch.argsort(r, stable=True)  # keeps each row's query order
            idx, r = idx[order], r[order]
            last_of_row = torch.ones_like(r, dtype=torch.bool)
            last_of_row[:-1] = r[:-1] != r[1:]
            pick, r = idx[last_of_row], r[last_of_row]
            self.last_adv[r] = s["adv"][pick]
            self.last_logits[r] = s["logits"][pick]
            self.last_margin[r] = s["margin"][pick]
            self.last_cost[r] = s["cost"][pick]
            self.last_valid[r] = s["valid"][pick]
            self.last_success[r] = s["success"][pick]
            self.last_source[r] = source
            for name in CONTROL_NAMES:
                self.last_projected[name][r] = s["projected"][name][pick]
                self.last_requested[name][r] = requested[name][pick]
        return s

    def relaxed(self, rows: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Relaxed (unquantized) flow and victim logits at normalized controls ``q``.

        Identical to ``RealizedSearch.surrogate_logits`` except that the floor value is
        configurable and, with a pinned shape, is not applied to the shape coordinate.
        """
        bounds_s = _subset_bounds(self.bounds, rows)
        floor = _mask_q(torch.full_like(q, self.surrogate_floor), bounds_s)
        if not self.shape_free:
            floor[:, 2] = 0.0
        q_eval = q + (floor - q).clamp(min=0.0).detach()
        free = _mask_q(torch.ones_like(q), bounds_s) > 0
        q_eval = torch.where(free, q_eval, q_eval.detach())
        transformed = self.model.generate(
            self.raw[rows], _controls_from_q(q_eval, bounds_s), quantize=False,
            capabilities=_subset_capabilities(self.caps, rows),
        )
        self.surrogate.index_add_(0, rows, torch.ones_like(rows))
        self.backward.index_add_(0, rows, torch.ones_like(rows))
        return transformed, self.victim((transformed - self.center) / self.scale)

    def result(self) -> PrimitiveOptimizationResult:
        res = super().result()
        if not self.track_last:
            return res
        return PrimitiveOptimizationResult(
            requested={k: v.detach() for k, v in self.last_requested.items()},
            projected={k: v.detach() for k, v in self.last_projected.items()},
            adversarial_raw=self.last_adv.detach(),
            logits=self.last_logits.detach(),
            objective_margin=self.last_margin.detach(),
            normalized_cost=self.last_cost.detach(),
            valid=self.last_valid.detach(),
            success=self.last_success.detach(),
            candidate_source=self.last_source.detach(),
            realized_evaluations=res.realized_evaluations,
            surrogate_evaluations=res.surrogate_evaluations,
            backward_evaluations=res.backward_evaluations,
            first_success_evaluation=res.first_success_evaluation,
            first_objective_hit_evaluation=res.first_objective_hit_evaluation,
            first_success_phase=res.first_success_phase,
            iterations=res.iterations,
            restarts=res.restarts,
        )


def continuous_controls(requested: dict[str, torch.Tensor], bounds: dict[str, torch.Tensor],
                        caps) -> dict[str, torch.Tensor]:
    """``project_controls`` without the integer rounding: the continuous per-flow box
    ``[0, bounds]`` (the domain of the relaxation ``bounds * q``) and the capability mask M(x)."""
    p = torch.minimum(requested["p"].clamp(min=0.0), bounds["p"].clamp(min=0.0))
    delay = torch.minimum(requested["delay"].clamp(min=0.0), bounds["delay"].clamp(min=0.0))
    shape = torch.minimum(requested["shape"].clamp(0.0, 1.0), bounds["shape"].clamp(0.0, 1.0))
    p = torch.where(caps.pad_allowed, p, torch.zeros_like(p))
    delay = torch.where(caps.timing_allowed, delay, torch.zeros_like(delay))
    shape = torch.where((delay > 0.0) & caps.timing_allowed, shape, torch.zeros_like(shape))
    return {"p": p, "delay": delay, "shape": shape}


class ContinuousSearch(AblationSearch):
    """Search whose candidates are scored on the continuous primitive state (ablation P1).

    Every candidate (identity, padding enumeration, refinement iterate) is mapped by
    :func:`continuous_controls` and the canonical φ with ``quantize=False``; its victim logits
    give the objective hit (the search success: ``validity_fn`` is None) and the margin, and the
    canonical incumbent rule selects on them. Nothing is rounded during the search. One
    candidate costs one victim evaluation, as in ``RealizedSearch`` (the inherited
    ``realized`` counter therefore counts continuous candidates here).
    """

    def _score(self, rows: torch.Tensor, requested: dict[str, torch.Tensor]) -> dict:
        parts = []
        for start in range(0, rows.numel(), _VICTIM_CHUNK_ROWS):
            r = rows[start:start + _VICTIM_CHUNK_ROWS]
            req = {name: requested[name][start:start + _VICTIM_CHUNK_ROWS] for name in CONTROL_NAMES}
            raw_s = self.raw[r]
            bounds_s = _subset_bounds(self.bounds, r)
            caps_s = _subset_capabilities(self.caps, r)
            controls = continuous_controls(req, bounds_s, caps_s)
            adv = self.model.generate(raw_s, controls, quantize=False, capabilities=caps_s)
            logits = self.victim((adv - self.center) / self.scale)
            hit = self.objective.hit(logits)
            parts.append({
                "projected": controls, "adv": adv, "logits": logits,
                "margin": self.objective.margin(logits),
                "cost": _normalized_cost(controls, bounds_s),
                "hit": hit, "valid": torch.ones_like(hit), "success": hit,
            })
        if len(parts) == 1:
            return parts[0]
        out = {k: torch.cat([p[k] for p in parts]) for k in parts[0] if k != "projected"}
        out["projected"] = {
            name: torch.cat([p["projected"][name] for p in parts]) for name in CONTROL_NAMES
        }
        return out


@torch.no_grad()
def realize_once(search: ContinuousSearch) -> tuple[PrimitiveOptimizationResult,
                                                     dict[str, torch.Tensor]]:
    """Apply the canonical realization to the continuous search's returned candidate.

    ``RealizedSearch._score`` (the FINAL code path, ``validity_fn=None``) projects the
    requested controls (integer bytes and µs, per-flow box, M(x)), maps them through φ with
    ``quantize=True`` and scores the victim: one evaluation per row. The returned result
    describes the realized flow only (success = objective hit on it); the continuous state is
    returned separately as ``continuous_*`` diagnostics.
    """
    cont = search.result()
    rows = torch.arange(search.n, device=search.raw.device)
    s = RealizedSearch._score(search, rows, cont.requested)
    prior = cont.realized_evaluations + cont.surrogate_evaluations
    final_index = prior + 1
    none = torch.full_like(final_index, -1)
    result = PrimitiveOptimizationResult(
        requested=cont.requested,
        projected=s["projected"],
        adversarial_raw=s["adv"],
        logits=s["logits"],
        objective_margin=s["margin"],
        normalized_cost=s["cost"],
        valid=s["valid"],
        success=s["success"],
        candidate_source=cont.candidate_source,
        realized_evaluations=torch.ones_like(prior),
        # Victim passes on continuous states: candidates + gradient forwards.
        surrogate_evaluations=prior,
        backward_evaluations=cont.backward_evaluations,
        first_success_evaluation=torch.where(s["success"], final_index, none),
        first_objective_hit_evaluation=torch.where(s["hit"], final_index, none),
        first_success_phase=torch.where(s["success"], cont.first_success_phase,
                                        torch.full_like(cont.first_success_phase, -2)),
        iterations=cont.iterations,
        restarts=cont.restarts,
    )
    extra = {
        "continuous_success": cont.success,
        "continuous_pred": cont.logits.argmax(1),
        "continuous_margin": cont.objective_margin,
        "continuous_cost": cont.normalized_cost,
        "continuous_adv_raw": cont.adversarial_raw,
        **{f"continuous_{name}": cont.projected[name] for name in CONTROL_NAMES},
        "continuous_candidate_evaluations": cont.realized_evaluations,
        "continuous_gradient_evaluations": cont.surrogate_evaluations,
        "continuous_first_hit_evaluation": cont.first_objective_hit_evaluation,
        "final_realization_evaluations": result.realized_evaluations,
    }
    return result, extra


def refinement_loss(
    kind: str,
    objective: AttackObjective,
    logits: torch.Tensor,
    transformed: torch.Tensor,
    anchor: torch.Tensor | None,
    center: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Per-row loss minimized by the refinement stage (success/incumbent never use it).

    * ``margin``: the objective margin (canonical).
    * ``ce``: untargeted ``log p(source)``; targeted ``-log p(target)``.
    * ``dlr``: the objective margin divided by ``z_pi1 - z_pi3`` (untargeted) or
      ``z_pi1 - (z_pi3 + z_pi4) / 2`` (targeted), Croce & Hein's scale-invariant DLR.
    * ``mimicry``: squared distance in ``asinh((x - center) / scale)`` space to the row's fixed
      Benign train anchor (Traffic Manipulator-style target feature vector).
    """
    if kind == "margin":
        return objective.margin(logits)
    if kind == "ce":
        logp = F.log_softmax(logits, 1)[:, objective.class_id]
        return logp if objective.kind == "untargeted" else -logp
    if kind == "dlr":
        z = logits.sort(1, descending=True).values
        if objective.kind == "untargeted":
            denom = z[:, 0] - z[:, 2]
        else:
            denom = z[:, 0] - 0.5 * (z[:, 2] + z[:, 3])
        return objective.margin(logits) / (denom + 1e-12)
    if kind == "mimicry":
        if anchor is None:
            raise ValueError("the mimicry loss needs a Benign anchor per row")
        return (torch.asinh((transformed - center) / scale) - anchor).pow(2).sum(1)
    raise KeyError(kind)


def optimize_hybrid_ablation(
    model,
    victim,
    raw: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    bounds: dict[str, torch.Tensor],
    caps,
    *,
    config: HybridConfig,
    seed: int,
    validity_fn,
    objective: AttackObjective = TARGET_BENIGN,
    anchor: torch.Tensor | None = None,
    nonfinite_gradients: str = "raise",
) -> tuple[PrimitiveOptimizationResult, SearchDiagnostics]:
    """Hybrid Search with the components selected by ``config``.

    ``anchor`` (``[n, F]``, asinh-scaled) is required by ``loss="mimicry"`` only.

    ``nonfinite_gradients``: ``"raise"`` (canonical, fail loud) or ``"zero"``. The canonical map
    is not differentiable everywhere outside the capability-admissible set (e.g. padding a flow
    whose packet-length variance is 0 gives d sqrt(var)/dp = inf * 0); only the
    capability-ablated arm (D6), whose search space contains such flows, uses ``"zero"``: the
    affected coordinates get no gradient step (the row still moves by its other coordinates and
    by random restarts) and the event is counted per row.
    """
    if nonfinite_gradients not in ("raise", "zero"):
        raise ValueError("nonfinite_gradients must be 'raise' or 'zero'")
    cfg = config
    if cfg.restarts is None and cfg.eval_budget is None:
        raise ValueError("restarts=None (fill the budget) requires eval_budget")
    # A continuous-state search keeps one evaluation per row for the final realization.
    search_cls = AblationSearch if cfg.realization_aware_search else ContinuousSearch
    search = search_cls(
        model, victim, raw, center, scale, bounds, caps,
        validity_fn=validity_fn if cfg.validity_in_search else None,
        objective=objective,
        eval_budget=cfg.eval_budget - (0 if cfg.realization_aware_search else 1),
        surrogate_floor=cfg.surrogate_floor, shape_free=cfg.fixed_shape is None,
        track_last=cfg.selection == "last",
    )
    n = raw.shape[0]
    device = raw.device
    zeros = torch.zeros(n, dtype=raw.dtype, device=device)
    grad_steps = torch.zeros(n, dtype=torch.int64, device=device)
    zero_grad_steps = torch.zeros(n, dtype=torch.int64, device=device)
    nonfinite_steps = torch.zeros(n, dtype=torch.int64, device=device)

    if cfg.padding_sweep:
        padding_cap = torch.floor(bounds["p"])
        max_padding = int(padding_cap.max().item()) if n else 0
        value = 1
        all_rows = torch.arange(n, device=device)
        with torch.no_grad():
            while value <= max_padding:
                remaining = search.remaining(all_rows)
                pending = (padding_cap >= value) & ~search.best_success & (remaining >= 1)
                rows = torch.nonzero(pending, as_tuple=False).flatten()
                m = rows.numel()
                if not m:
                    break
                k = max(1, min(max_padding - value + 1, _PADDING_CHUNK_ROWS // m))
                values = torch.arange(value, value + k, dtype=raw.dtype, device=device)
                offsets = torch.arange(k, device=device).repeat_interleave(m)
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
        refinable = bounds["delay"] >= 1.0
    else:
        # Without the enumeration, padding-only rows must be refined as well.
        refinable = (bounds["p"] >= 1.0) | (bounds["delay"] >= 1.0)

    active = torch.nonzero(~search.best_success & refinable, as_tuple=False).flatten()
    if cfg.refinement and cfg.steps and active.numel():
        bounds_a = _subset_bounds(bounds, active)
        generator = torch.Generator(device=device).manual_seed(seed)
        base_q = _q_from_controls(
            {name: search.best_projected[name][active] for name in CONTROL_NAMES}, bounds_a
        )
        pinned_shape = None
        if cfg.fixed_shape is not None:
            pinned_shape = cfg.fixed_shape * (bounds_a["delay"] >= 1.0).to(raw.dtype)
            base_q[:, 2] = pinned_shape
        anchor_a = anchor[active] if anchor is not None else None
        checkpoint_interval = max(5, cfg.steps // 4)
        restart = 0
        while (cfg.restarts is None or restart < cfg.restarts) and search.gradient_rows(active).numel():
            search.phase = restart
            search.restarts += 1
            if restart == 0:
                q = base_q.clone()
                source = CANDIDATE_ADAPTIVE_CLEAN
            else:
                q = _random_q(base_q.shape, bounds_a, generator, base_q)
                if pinned_shape is not None:
                    q[:, 2] = pinned_shape
                source = CANDIDATE_ADAPTIVE_RANDOM
            velocity = torch.zeros_like(q)
            step_size = torch.full((len(active), 1), cfg.learning_rate, dtype=raw.dtype,
                                   device=device)
            restart_best_margin = torch.full((len(active),), torch.inf, dtype=raw.dtype,
                                             device=device)
            restart_best_q = q.clone()
            checkpoint_margin = restart_best_margin.clone()

            for iteration in range(cfg.steps):
                alive = search.gradient_rows(active)
                if not alive.numel():
                    break
                rows = active[alive]
                bounds_s = _subset_bounds(bounds_a, alive)
                q_alive = q[alive].requires_grad_(True)
                transformed, logits = search.relaxed(rows, q_alive)
                loss = refinement_loss(
                    cfg.loss, objective, logits, transformed,
                    anchor_a[alive] if anchor_a is not None else None, center, scale,
                )
                if nonfinite_gradients == "raise":
                    grad = objective_gradient(loss.sum(), q_alive)
                else:
                    (grad,) = torch.autograd.grad(loss.sum(), q_alive)
                    bad = ~torch.isfinite(grad)
                    nonfinite_steps[rows] += bad.any(1).to(torch.int64)
                    grad = torch.where(bad, torch.zeros_like(grad), grad)
                search.iterations += 1
                with torch.no_grad():
                    free = _mask_q(torch.ones_like(grad), bounds_s)
                    if pinned_shape is not None:
                        grad[:, 2] = 0.0
                        free[:, 2] = 0.0
                    grad_steps[rows] += 1
                    zero_grad_steps[rows] += ((grad.abs() * free).sum(1) == 0).to(torch.int64)
                    grad_scale = ((grad.abs() * free).sum(1, keepdim=True)
                                  / free.sum(1, keepdim=True).clamp(min=1.0)).clamp(min=1e-12)
                    velocity[alive] = cfg.momentum * velocity[alive] + grad / grad_scale
                    q_new = _mask_q(
                        (q[alive] - step_size[alive] * velocity[alive].sign()).clamp(0.0, 1.0),
                        bounds_s,
                    )
                    if pinned_shape is not None:
                        q_new[:, 2] = pinned_shape[alive]
                    q[alive] = q_new
                    out = search.evaluate(rows, _controls_from_q(q_new, bounds_s), source)
                    improved = out["margin"] < restart_best_margin[alive]
                    restart_best_margin[alive] = torch.where(
                        improved, out["margin"], restart_best_margin[alive]
                    )
                    restart_best_q[alive] = torch.where(
                        improved[:, None], q_new, restart_best_q[alive]
                    )
                    if cfg.adaptive_step and (iteration + 1) % checkpoint_interval == 0:
                        stalled = restart_best_margin >= checkpoint_margin - 1e-6
                        step_size[stalled] *= 0.5
                        q[stalled] = restart_best_q[stalled]
                        velocity[stalled] = 0.0
                        checkpoint_margin = restart_best_margin.clone()
            restart += 1
    if not cfg.realization_aware_search:
        res, extra = realize_once(search)
        return res, SearchDiagnostics(grad_steps, zero_grad_steps, nonfinite_steps, extra)
    return search.result(), SearchDiagnostics(grad_steps, zero_grad_steps, nonfinite_steps)
