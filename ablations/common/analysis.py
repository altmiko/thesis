"""Shared analysis of the PrimAttack ablations.

Unit of analysis: one (dataset, victim, budget, condition). Per attack seed the four classes are
pooled (3,200 frozen flows); ASR is reported as the mean over attack seeds with the seed range.
Each condition is compared with the shared reference arm (``ablations/reference``) on the SAME
flows by a paired McNemar test at attack seed 42 with a Newcombe 95% CI for the paired
difference; Holm corrects within one experiment's family of comparisons. Victims and datasets
are never pooled (pooling victims is pseudoreplication).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ablations.common.runner import CLASSES, REFERENCE
from evaluation.paired_validity_gap import holm_adjust, mcnemar_test, newcombe_paired_ci

REF_SEED = 42
BUDGET_ORDER = {"p75": 0, "unb": 1}


@dataclass(frozen=True)
class Source:
    """Where a condition's per-row artifacts live."""

    results_dir: Path
    condition: str


def load_cells(results_dir: Path) -> pd.DataFrame:
    frames = [pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
              for p in sorted(results_dir.glob("*/cells.json"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def pooled_rows(src: Source, dataset: str, victim: str, budget_label: str, seed: int,
                fields: tuple[str, ...]) -> dict[str, np.ndarray] | None:
    """Per-row fields of one condition cell, classes concatenated in ``CLASSES`` order."""
    art = src.results_dir / dataset / "artifacts"
    parts = []
    for cname in CLASSES:
        path = art / f"{victim}__{cname}__{budget_label}__{src.condition}__seed{seed}.npz"
        if not path.exists():
            return None
        with np.load(path, allow_pickle=True) as z:
            parts.append({f: z[f] for f in ("sample_id",) + tuple(fields)})
    return {f: np.concatenate([p[f] for p in parts]) for f in parts[0]}


def _groups(results_dir: Path, reference_dir: Path, conditions: list[str]):
    cells = load_cells(results_dir)
    ref_cells = load_cells(reference_dir)
    if cells.empty or ref_cells.empty:
        raise FileNotFoundError(f"no cells under {results_dir} or {reference_dir}")
    keys = (cells[cells["condition"].isin(conditions)][["dataset", "victim", "budget_label"]]
            .drop_duplicates().itertuples(index=False))
    return sorted(keys, key=lambda k: (k.dataset, k.victim, BUDGET_ORDER.get(k.budget_label, 9)))


def compare_to_reference(
    results_dir: Path,
    reference_dir: Path,
    conditions: list[str],
    *,
    outcome: str = "valid_success",
    secondary: tuple[str, ...] = ("raw_success",),
    reference_dir_for: dict[str, Path] | None = None,
    baseline: str = REFERENCE.name,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary (one row per group x condition, baseline included) and McNemar tests.

    ``baseline`` is the comparison arm, read from ``reference_dir`` (default: the shared
    reference arm).
    """
    summary, tests = [], []
    for g in _groups(results_dir, reference_dir, conditions):
        arms = [Source(reference_dir, baseline)] + [
            Source((reference_dir_for or {}).get(c, results_dir), c) for c in conditions]
        per_arm = {}
        for src in arms:
            per_seed = {}
            for seed in sorted(_seeds(src.results_dir, g, src.condition)):
                rows = pooled_rows(src, g.dataset, g.victim, g.budget_label, seed,
                                   (outcome,) + secondary)
                if rows is not None:
                    per_seed[seed] = rows
            per_arm[src.condition] = per_seed
        ref = per_arm[baseline]
        for src in arms:
            per_seed = per_arm[src.condition]
            seeds = sorted(set(per_seed) & set(ref))
            if not seeds:
                continue
            asr = np.array([per_seed[s][outcome].mean() for s in seeds])
            ref_asr = np.array([ref[s][outcome].mean() for s in seeds])
            row = {"dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
                   "condition": src.condition, "seeds": ",".join(map(str, seeds)),
                   "n_per_seed": int(len(per_seed[seeds[0]][outcome])),
                   f"{outcome}_mean": float(asr.mean()), f"{outcome}_min": float(asr.min()),
                   f"{outcome}_max": float(asr.max()),
                   "delta_pp_vs_reference": float(100 * (asr.mean() - ref_asr.mean()))}
            for f in secondary:
                row[f"{f}_mean"] = float(np.mean([per_seed[s][f].mean() for s in seeds]))
            summary.append(row)
            if src.condition == baseline or REF_SEED not in seeds:
                continue
            a, b = per_seed[REF_SEED], ref[REF_SEED]
            if not np.array_equal(a["sample_id"], b["sample_id"]):
                raise AssertionError(f"{g}: {src.condition} rows are not aligned with reference")
            x, r = a[outcome].astype(bool), b[outcome].astype(bool)
            both, only_x, only_r = int((x & r).sum()), int((x & ~r).sum()), int((~x & r).sum())
            neither = int((~x & ~r).sum())
            test = mcnemar_test(only_x, only_r)
            lo, hi = newcombe_paired_ci(both, only_x, only_r, neither)
            tests.append({"dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
                          "condition": src.condition, "outcome": outcome, "seed": REF_SEED,
                          "n": int(len(x)), "condition_asr": float(x.mean()),
                          "reference_asr": float(r.mean()),
                          "diff_pp": float(100 * (x.mean() - r.mean())),
                          "ci95_lo_pp": float(100 * lo), "ci95_hi_pp": float(100 * hi),
                          "condition_only": only_x, "reference_only": only_r,
                          "test": test["test_variant"], "p_value": float(test["p_value"])})
    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        tests_df["p_holm"] = holm_adjust(tests_df["p_value"].tolist())
    return pd.DataFrame(summary), tests_df


def _seeds(results_dir: Path, g, condition: str) -> set[int]:
    seeds = set()
    for path in (results_dir / g.dataset / "artifacts").glob(
            f"{g.victim}__*__{g.budget_label}__{condition}__seed*.npz"):
        seeds.add(int(path.stem.rsplit("seed", 1)[1]))
    return seeds


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.2f}%"


def summary_markdown(summary: pd.DataFrame, tests: pd.DataFrame, *, outcome: str,
                     outcome_label: str, secondary_labels: dict[str, str] | None = None,
                     baseline: str = REFERENCE.name) -> str:
    """One table per dataset: ASR mean (seed range), delta vs reference, seed-42 McNemar."""
    secondary_labels = secondary_labels or {}
    lines = []
    test_idx = ({(r.dataset, r.victim, r.budget, r.condition): r
                 for r in tests.itertuples(index=False)} if not tests.empty else {})
    for dataset, df in summary.groupby("dataset", sort=True):
        lines += [f"### {dataset}", "",
                  "| victim | budget | condition | " + outcome_label + " mean (min-max) | "
                  "Δ vs ref (pp) | " + "".join(f"{v} | " for v in secondary_labels.values())
                  + "seed-42 cond-only / ref-only | p (Holm) |",
                  "|---|---|---|---|---|" + "---|" * len(secondary_labels) + "---|---|"]
        for r in df.itertuples(index=False):
            t = test_idx.get((r.dataset, r.victim, r.budget, r.condition))
            d = getattr(r, f"{outcome}_mean")
            cell = (f"{pct(d)} ({pct(getattr(r, f'{outcome}_min'))}-"
                    f"{pct(getattr(r, f'{outcome}_max'))})")
            sec = "".join(f"{pct(getattr(r, f'{k}_mean'))} | " for k in secondary_labels)
            test = ("-" if t is None else f"{t.condition_only} / {t.reference_only}")
            p = "-" if t is None else f"{t.p_holm:.3g}"
            delta = "-" if r.condition == baseline else f"{r.delta_pp_vs_reference:+.2f}"
            lines.append(f"| {r.victim} | {r.budget} | {r.condition} | {cell} | {delta} | "
                         f"{sec}{test} | {p} |")
        lines.append("")
    return "\n".join(lines)


def write_standard_report(results_dir: Path, reference_dir: Path, conditions: list[str], *,
                          title: str, outcome: str = "valid_success",
                          outcome_label: str = "Valid ASR",
                          secondary: dict[str, str] | None = None,
                          extra_sections: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    secondary = secondary or {"raw_success": "Raw ASR"}
    summary, tests = compare_to_reference(results_dir, reference_dir, conditions,
                                          outcome=outcome, secondary=tuple(secondary))
    summary.to_csv(results_dir / "summary.csv", index=False)
    tests.to_csv(results_dir / "tests.csv", index=False)
    body = [f"# {title}", "",
            f"Outcome: **{outcome_label}** over the frozen clean-correct flows (4 classes x 800 per "
            "attack seed). Mean over attack seeds 42/2024/2026 with the seed range. Paired "
            "McNemar vs the reference arm at seed 42, Holm over this table's comparisons.", "",
            summary_markdown(summary, tests, outcome=outcome, outcome_label=outcome_label,
                             secondary_labels=secondary)]
    body += extra_sections or []
    (results_dir / "report.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print("\n".join(body))
    return summary, tests


PROFILE_FIELDS = ("valid_success", "normalized_cost", "relative_duration_change", "added_bytes",
                  "p", "delay", "shape")


def success_profile(results_dir: Path, reference_dir: Path, conditions: list[str],
                    *, mask_field: str = "valid_success") -> list[str]:
    """Cost profile of the successes (all seeds and classes pooled per group)."""
    rows = []
    for g in _groups(results_dir, reference_dir, conditions):
        for cond in [REFERENCE.name] + conditions:
            src = Source(reference_dir if cond == REFERENCE.name else results_dir, cond)
            fields = tuple(dict.fromkeys(PROFILE_FIELDS + (mask_field,)))
            parts = [pooled_rows(src, g.dataset, g.victim, g.budget_label, s, fields)
                     for s in sorted(_seeds(src.results_dir, g, cond))]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            r = {f: np.concatenate([p[f] for p in parts]) for f in fields}
            s = r[mask_field].astype(bool)
            med = lambda f: float(np.median(r[f][s])) if s.any() else float("nan")  # noqa: E731
            rows.append({"dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
                         "condition": cond, "successes": int(s.sum()),
                         "median_cost": med("normalized_cost"),
                         "median_rel_duration": med("relative_duration_change"),
                         "median_delay_us": med("delay"), "median_shape": med("shape"),
                         "frac_padding": float((r["p"][s] > 0).mean()) if s.any() else float("nan"),
                         "median_added_bytes": med("added_bytes")})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / f"success_profile_{mask_field}.csv", index=False)
    if df.empty:
        return []
    f3 = lambda x: "n/a" if x != x else f"{x:.3g}"  # noqa: E731
    lines = [f"## Cost profile of the successes (`{mask_field}`, all seeds pooled)", "",
             "| dataset | victim | budget | condition | successes | median norm. cost | "
             "median rel. duration change | median delay (µs) | median shape | "
             "frac. using padding | median added bytes |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget} | {r.condition} | {r.successes} | "
                     f"{f3(r.median_cost)} | {f3(r.median_rel_duration)} | "
                     f"{f3(r.median_delay_us)} | {f3(r.median_shape)} | {f3(r.frac_padding)} | "
                     f"{f3(r.median_added_bytes)} |")
    return lines + [""]


def append_report(results_dir: Path, lines: list[str]) -> None:
    with (results_dir / "report.md").open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
