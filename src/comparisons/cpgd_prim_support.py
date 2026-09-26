"""C-PGD on PrimAttack's downstream feature support.

This reproduces the constrained-PGD formulation of Simonetto et al., "A Unified
Framework for Adversarial Attack and Defense in Constrained Feature Space"
(IJCAI 2022, DOI 10.24963/ijcai.2022/183), Eq. (2): maximize classification
loss minus differentiable constraint violation inside an Lp ball. The public
reference implementation is ``serval-uni-lu/constrained-attacks``.

The attack directly optimizes supported feature coordinates. It does not optimize
PrimAttack primitives and therefore does not inherit PrimAttack's directions,
budgets, quantization, or deterministic feature coupling. Validator-v2 constraints
are intentionally post-attack checks; only the explicitly encoded differentiable
relations in the shared CAPGD resources contribute gradients.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from comparisons.capgd_cicids2017 import CAPGDResources, CAPGD_PRIM_SUPPORT

CPGD_METHOD_ID = "cpgd_prim_support"
CPGD_PAPER = (
    "Simonetto et al., A Unified Framework for Adversarial Attack and Defense "
    "in Constrained Feature Space, IJCAI 2022, DOI 10.24963/ijcai.2022/183"
)
CPGD_PUBLIC_IMPLEMENTATION = "https://github.com/serval-uni-lu/constrained-attacks"


@dataclass(frozen=True)
class CPGDConfig:
    epsilon: float = 0.5
    norm: str = "L2"
    step_size: float = 0.05
    iterations: int = 40
    constraint_penalty_weight: float = 1.0
    loss: str = "ce"
    random_start: bool = True

    def __post_init__(self) -> None:
        if self.epsilon < 0.0:
            raise ValueError("C-PGD epsilon must be non-negative")
        if self.step_size <= 0.0:
            raise ValueError("C-PGD step size must be positive")
        if self.iterations <= 0:
            raise ValueError("C-PGD iterations must be positive")
        if self.constraint_penalty_weight < 0.0:
            raise ValueError("C-PGD constraint penalty weight must be non-negative")
        if self.norm not in {"L2", "Linf"}:
            raise ValueError("C-PGD norm must be 'L2' or 'Linf'")
        if self.loss != "ce":
            raise ValueError("C-PGD currently supports the published cross-entropy objective only")


@dataclass(frozen=True)
class CPGDResult:
    adversarial_raw: torch.Tensor
    iterations: torch.Tensor
    model_evaluations: torch.Tensor
    final_constraint_violation: torch.Tensor


class CPGDPrimSupportAttack:
    """Published C-PGD objective with the canonical PrimAttack feature-support mask."""

    def __init__(
        self,
        resources: CAPGDResources,
        raw_victim: torch.nn.Module,
        *,
        config: CPGDConfig,
        seed: int,
        device: str | torch.device,
    ) -> None:
        if resources.configuration != CAPGD_PRIM_SUPPORT:
            raise ValueError(
                f"C-PGD requires {CAPGD_PRIM_SUPPORT!r} resources, got "
                f"{resources.configuration!r}"
            )
        self.resources = resources
        self.raw_victim = raw_victim
        self.config = config
        self.seed = int(seed)
        self.device = torch.device(device)
        resources.scaler.output_device = self.device
        self.mutable_mask = torch.as_tensor(
            resources.support_mask, dtype=torch.bool, device=self.device
        )
        self.constraint_executor = resources.api.ConstraintsExecutor(
            resources.api.AndConstraint(resources.constraints.relation_constraints),
            resources.api.PytorchBackend(),
            feature_names=resources.constraints.feature_names,
        )

    def constraint_violation(self, raw: torch.Tensor) -> torch.Tensor:
        """Per-sample differentiable sum of the explicitly encoded relation penalties."""

        violation = self.constraint_executor.execute(raw)
        if violation.ndim != 1 or violation.shape[0] != raw.shape[0]:
            raise RuntimeError(
                f"constraint executor returned {tuple(violation.shape)}, expected ({raw.shape[0]},)"
            )
        return violation

    def objective(
        self, normalized: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ascent objective, untargeted attack loss, and constraint penalty.

        Maximizing ``CE - lambda * violation`` is equivalent to minimizing
        ``attack_loss + lambda * violation`` when ``attack_loss = -CE``.
        """

        raw = self.resources.scaler.inverse_transform(normalized)
        logits = self.raw_victim(raw)
        attack_loss = F.cross_entropy(logits, labels)
        constraint_loss = self.constraint_violation(raw).mean()
        score = attack_loss - self.config.constraint_penalty_weight * constraint_loss
        return score, attack_loss, constraint_loss

    def _project(self, candidate: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
        mask = self.mutable_mask.to(candidate.dtype).unsqueeze(0)
        delta = (candidate - clean) * mask
        if self.config.norm == "L2":
            norms = delta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            factor = torch.minimum(
                torch.ones_like(norms),
                torch.full_like(norms, self.config.epsilon) / norms,
            )
            delta = delta * factor
        else:
            delta = delta.clamp(-self.config.epsilon, self.config.epsilon)

        projected = clean + delta
        # The fitted train box is [0,1]^d. If a held-out clean value lies outside
        # that box, include the clean endpoint rather than force an epsilon-breaking
        # move merely to enter the box; attacks cannot move farther out of range.
        lower = torch.minimum(torch.zeros_like(clean), clean)
        upper = torch.maximum(torch.ones_like(clean), clean)
        bounded = torch.maximum(torch.minimum(projected, upper), lower)
        return torch.where(self.mutable_mask.unsqueeze(0), bounded, clean)

    def _random_start(self, clean: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if not self.config.random_start or self.config.epsilon == 0.0:
            return clean.clone()
        mask = self.mutable_mask.to(clean.dtype).unsqueeze(0)
        if self.config.norm == "L2":
            delta = torch.randn(
                clean.shape, generator=generator, device=clean.device, dtype=clean.dtype
            ) * mask
            norm = delta.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            radius = torch.rand(
                (clean.shape[0], 1),
                generator=generator,
                device=clean.device,
                dtype=clean.dtype,
            )
            delta = delta / norm * radius * self.config.epsilon
        else:
            delta = (
                torch.rand(
                    clean.shape,
                    generator=generator,
                    device=clean.device,
                    dtype=clean.dtype,
                )
                * 2.0
                - 1.0
            ) * self.config.epsilon * mask
        return self._project(clean + delta, clean)

    def _typed_final(self, clean_raw: torch.Tensor, candidate_raw: torch.Tensor) -> torch.Tensor:
        typed = self.resources.api.fix_types(
            clean_raw, candidate_raw, self.resources.feature_types
        )
        frozen = ~self.mutable_mask
        typed[:, frozen] = clean_raw[:, frozen]

        clean_normalized = self.resources.scaler.transform(clean_raw)
        typed_normalized = self.resources.scaler.transform(typed)
        delta = (typed_normalized - clean_normalized) * self.mutable_mask
        if self.config.norm == "L2":
            distance = delta.norm(p=2, dim=1)
        else:
            distance = delta.abs().amax(dim=1)
        bad = distance > self.config.epsilon + 1e-6

        # Integer repair can cross the continuous Lp boundary. For those rows, keep
        # integer coordinates at their clean values and project only continuous ones;
        # this preserves both the declared type and the epsilon invariant.
        if bool(bad.any()):
            integer = torch.as_tensor(
                self.resources.feature_types == "int",
                dtype=torch.bool,
                device=self.device,
            ) & self.mutable_mask
            bad_rows = torch.nonzero(bad, as_tuple=False).flatten()
            integer_cols = torch.nonzero(integer, as_tuple=False).flatten()
            if integer_cols.numel():
                typed[bad_rows[:, None], integer_cols] = clean_raw[
                    bad_rows[:, None], integer_cols
                ]
            typed_normalized = self.resources.scaler.transform(typed)
            typed_normalized[bad_rows] = self._project(
                typed_normalized[bad_rows], clean_normalized[bad_rows]
            )
            typed = self.resources.scaler.inverse_transform(typed_normalized)
            if integer_cols.numel():
                typed[bad_rows[:, None], integer_cols] = clean_raw[
                    bad_rows[:, None], integer_cols
                ]
            typed[:, frozen] = clean_raw[:, frozen]

        final_normalized = self.resources.scaler.transform(typed)
        final_delta = (final_normalized - clean_normalized) * self.mutable_mask
        final_distance = (
            final_delta.norm(p=2, dim=1)
            if self.config.norm == "L2"
            else final_delta.abs().amax(dim=1)
        )
        if bool((final_distance > self.config.epsilon + 1e-5).any()):
            raise AssertionError("C-PGD final type repair violated epsilon projection")
        if not torch.equal(typed[:, frozen], clean_raw[:, frozen]):
            raise AssertionError("C-PGD changed a feature outside PrimAttack support")
        return typed

    def run(self, clean_raw: torch.Tensor, labels: torch.Tensor) -> CPGDResult:
        clean_raw = clean_raw.to(self.device)
        labels = labels.to(self.device)
        clean_normalized = self.resources.scaler.transform(clean_raw).detach()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        adversarial = self._random_start(clean_normalized, generator).detach()

        for _ in range(self.config.iterations):
            adversarial.requires_grad_(True)
            score, _, _ = self.objective(adversarial, labels)
            (gradient,) = torch.autograd.grad(score, adversarial)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError("C-PGD objective produced a non-finite gradient")
            gradient = gradient * self.mutable_mask
            if self.config.norm == "L2":
                gradient = gradient / gradient.norm(p=2, dim=1, keepdim=True).clamp(min=1e-12)
            else:
                gradient = gradient.sign()
            with torch.no_grad():
                adversarial = self._project(
                    adversarial + self.config.step_size * gradient,
                    clean_normalized,
                )
            adversarial = adversarial.detach()

        candidate_raw = self.resources.scaler.inverse_transform(adversarial)
        final_raw = self._typed_final(clean_raw, candidate_raw).detach()
        with torch.no_grad():
            final_violation = self.constraint_violation(final_raw).detach()
        n = clean_raw.shape[0]
        return CPGDResult(
            adversarial_raw=final_raw,
            iterations=torch.full(
                (n,), self.config.iterations, dtype=torch.int64, device=self.device
            ),
            model_evaluations=torch.full(
                (n,), self.config.iterations, dtype=torch.int64, device=self.device
            ),
            final_constraint_violation=final_violation,
        )
