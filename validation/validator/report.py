"""Human-readable rendering of validation results.

Renders a :class:`SampleResult` in the explainable format required by the thesis
(§11): per-provenance pass/fail counts and, for every failed rule, the observed
value, expected value, absolute/relative error, allowed tolerance and a plain
explanation. Also renders a batch summary.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from validation.validator.result import BatchResult, SampleResult


def _fmt(x: float) -> str:
    if x != x:  # nan
        return "nan"
    if abs(x) >= 1e6 or (x != 0 and abs(x) < 1e-4):
        return f"{x:.6g}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def render_sample_result(s: "SampleResult") -> str:
    lines: list[str] = ["VALIDATION RESULT", "=" * 17, ""]
    lines.append(f"structurally_valid:    {str(s.structurally_valid).upper()}")
    lines.append(f"hard_structural_valid: {str(s.hard_structural_valid).upper()}")
    lines.append(f"hybrid_valid:          {str(s.hybrid_valid).upper()}")
    lines.append(f"in_distribution:       {str(s.in_distribution).upper()}"
                 f"  (plausibility score {s.plausibility_score:.3f})")
    lines.append("")
    counts = s.counts_by_source()
    for src in ("SCHEMA", "MINED", "EXTRACTOR", "PROTOCOL"):
        c = counts[src]
        lines += [src, "-" * len(src), f"passed: {c['passed']}", f"failed: {c['failed']}", ""]

    failed = s.failed_rules
    lines.append("FAILED RULES")
    lines.append("=" * 12)
    if not failed:
        lines.append("(none)")
    for fr in failed:
        lines.append("")
        lines.append(f"[{fr['id']}]  ({fr['source_type']})")
        lines.append(fr["expression"])
        if "observed" in fr:
            lines.append("")
            lines.append(f"Observed:        {_fmt(fr['observed'])}")
            lines.append(f"Expected:        {_fmt(fr['expected'])}")
            lines.append(f"Absolute error:  {_fmt(fr['absolute_error'])}")
            lines.append(f"Relative error:  {fr['relative_error'] * 100:.4f}%")
        tol = fr["tolerance"]
        lines.append(f"Allowed tolerance: abs={_fmt(tol['absolute'])} rel={tol['relative'] * 100:.4f}%")
        if fr["features"] and "observed" not in fr:
            lines.append("Features: " + ", ".join(f"{k}={_fmt(v)}" for k, v in fr["features"].items()))
        if fr["description"]:
            lines.append("Explanation:")
            lines.append(fr["description"])
    return "\n".join(lines)


def render_batch_summary(b: "BatchResult") -> str:
    rates = b.rates()
    lines = [f"Batch validation summary  (N={b.n})", "=" * 40]
    for k in ("schema_valid", "protocol_valid", "extractor_valid", "mined_valid",
              "hard_structural_valid", "hybrid_valid", "in_distribution"):
        lines.append(f"  {k:24s} {rates[k]:.4%}")
    rejecting = b.rules_rejecting_any()
    if rejecting:
        lines.append("")
        lines.append("Rules rejecting samples (count):")
        by_id = {r.id: r for r in b.rules}
        for rid, cnt in sorted(rejecting.items(), key=lambda kv: -kv[1]):
            r = by_id[rid]
            lines.append(f"  [{rid}] {r.source_type:9s} {r.expression()}  -> {cnt}")
    return "\n".join(lines)
