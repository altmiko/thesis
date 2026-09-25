"""Paired old-vs-new PrimAttack comparison on the frozen full-evaluation roster.

Compares the upgraded search (``outputs/full_adv_eval_primattack_v2``, ``prim_search_*``) with
the replaced Adam/sigmoid ``(p, alpha)`` optimizer (``outputs/full_adv_eval``, ``prim_opt_*``)
and with the in-run random-feasible control (``prim_rand_*``). Rows are joined by victim,
class, seed, and ordered ``sample_id``; any selection or row mismatch aborts.

Primary tests use the reference seed, pool the four malicious classes WITHIN a victim (each
paired unit is one distinct flow), and never pool victims. McNemar: exact binomial when
b+c<25, else continuity-corrected chi-square. CIs: Newcombe square-and-add paired 95% CI.
Holm correction runs within each outcome family over all primary tests.

Writes ``outputs/full_adv_eval_primattack_v2/optimizer_comparison.json`` and
``PRIMATTACK_V2_OPTIMIZER_COMPARISON.md`` (repo root).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.paired_validity_gap import holm_adjust, mcnemar_test, newcombe_paired_ci

REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGETS = ("p50", "p75", "unb")
BUDGET_TEXT = {"p50": "intermediate (p50)", "p75": "maximum-evaluated (p75)", "unb": "unbounded"}
MODES = ("joint", "timing", "padding")
OUTCOMES = {
    "targeted_valid": "targeted ∧ hybrid_valid",
    "targeted_valid_feasible": "targeted ∧ valid ∧ primitive_feasible",
    "semantic_pass": "targeted ∧ valid ∧ feasible ∧ SP",
    "true_idsr": "True-IDSR: targeted ∧ valid ∧ in_distribution",
}
CONTRASTS = (
    ("search_vs_old", "prim_search_{mode}_{budget}", "new", "prim_opt_{mode}_{budget}", "old"),
    ("search_vs_random", "prim_search_{mode}_{budget}", "new", "prim_rand_{mode}_{budget}", "new"),
)


def _outcome(data: dict[str, np.ndarray], outcome: str) -> np.ndarray:
    success = data["targeted_success"].astype(bool) & data["domain_valid"].astype(bool)
    if outcome == "targeted_valid":
        return success
    if outcome == "true_idsr":
        return success & data["in_dist"].astype(bool)
    feasible = data["primitive_feasible"]
    if feasible.dtype != bool and int(feasible.min()) < 0:
        raise ValueError("primitive outcome requested for a non-primitive artifact")
    success &= feasible.astype(bool)
    if outcome == "targeted_valid_feasible":
        return success
    return success & data["semantic_pass"].astype(bool)


class Runs:
    def __init__(self, old_dir: Path, new_dir: Path):
        self.dirs = {"old": old_dir, "new": new_dir}
        self.selection = {
            key: json.loads((path / "selection.json").read_text(encoding="utf-8"))
            for key, path in self.dirs.items()
        }
        self.config = json.loads((new_dir / "config.json").read_text(encoding="utf-8"))
        self._cache: dict[tuple, dict[str, np.ndarray]] = {}

    def check_selection(self) -> dict[str, dict[str, str]]:
        hashes = {}
        for victim in self.config["victims"]:
            hashes[victim] = {}
            for class_name in self.config["classes"]:
                old = self.selection["old"][victim][class_name]["sha256_sample_ids"]
                new = self.selection["new"][victim][class_name]["sha256_sample_ids"]
                if old != new:
                    raise AssertionError(f"frozen selection mismatch for {victim}/{class_name}")
                hashes[victim][class_name] = new
        return hashes

    def load(self, run: str, victim: str, class_name: str, attack: str, seed: int):
        key = (run, victim, class_name, attack, seed)
        if key not in self._cache:
            path = self.dirs[run] / "artifacts" / f"{victim}__{class_name}__{attack}__seed{seed}.npz"
            with np.load(path, allow_pickle=True) as data:
                loaded = {name: data[name] for name in (
                    "sample_id", "targeted_success", "domain_valid",
                    "primitive_feasible", "semantic_pass", "in_dist",
                )}
            expected = np.asarray(
                self.selection["new"][victim][class_name]["sample_ids"], dtype="U128"
            )
            if not np.array_equal(loaded["sample_id"].astype("U128"), expected):
                raise AssertionError(f"row order mismatch in {path}")
            self._cache[key] = loaded
        return self._cache[key]

    def outcome(self, run, victim, classes, attack, seed, outcome):
        return np.concatenate([
            _outcome(self.load(run, victim, class_name, attack, seed), outcome)
            for class_name in classes
        ])


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    n11 = int((a & b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n00 = int((~a & ~b).sum())
    test = mcnemar_test(n10, n01)
    lower, upper = newcombe_paired_ci(n11, n10, n01, n00)
    return {
        "n": int(a.size), "n11": n11, "n10_A_only": n10, "n01_B_only": n01, "n00": n00,
        "rate_A": float(a.mean()), "rate_B": float(b.mean()),
        "diff_A_minus_B": float(a.mean() - b.mean()), "ci95": [lower, upper],
        "test_variant": test["test_variant"], "p_value": float(test["p_value"]),
    }


def analyze(runs: Runs, reference_seed: int) -> dict:
    victims, classes, seeds = runs.config["victims"], runs.config["classes"], runs.config["seeds"]
    rows = []
    for contrast, a_tpl, a_run, b_tpl, b_run in CONTRASTS:
        for victim in victims:
            for budget in BUDGETS:
                for mode in MODES:
                    a_name = a_tpl.format(mode=mode, budget=budget)
                    b_name = b_tpl.format(mode=mode, budget=budget)
                    for outcome in OUTCOMES:
                        a = runs.outcome(a_run, victim, classes, a_name, reference_seed, outcome)
                        b = runs.outcome(b_run, victim, classes, b_name, reference_seed, outcome)
                        row = {
                            "contrast": contrast, "victim": victim, "budget": budget,
                            "mode": mode, "outcome": outcome, "A": a_name, "B": b_name,
                            **paired(a, b),
                        }
                        per_seed = []
                        for seed in seeds:
                            sa = runs.outcome(a_run, victim, classes, a_name, seed, outcome)
                            sb = runs.outcome(b_run, victim, classes, b_name, seed, outcome)
                            per_seed.append({
                                "seed": seed, "rate_A": float(sa.mean()),
                                "rate_B": float(sb.mean()),
                                "b": int((sa & ~sb).sum()), "c": int((~sa & sb).sum()),
                            })
                        row["per_seed"] = per_seed
                        row["per_class_ref_seed"] = {
                            class_name: {
                                "A": int(runs.outcome(a_run, victim, [class_name], a_name,
                                                      reference_seed, outcome).sum()),
                                "B": int(runs.outcome(b_run, victim, [class_name], b_name,
                                                      reference_seed, outcome).sum()),
                            }
                            for class_name in classes
                        }
                        rows.append(row)
    for contrast, *_ in CONTRASTS:
        for outcome in OUTCOMES:
            family = [r for r in rows if r["contrast"] == contrast and r["outcome"] == outcome]
            for row, adjusted in zip(family, holm_adjust([r["p_value"] for r in family])):
                row["holm_p"] = float(adjusted)
    return {"reference_seed": reference_seed, "rows": rows}


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _seed_mean(row, key):
    values = np.asarray([s[key] for s in row["per_seed"]], dtype=float)
    return f"{values.mean() * 100:.2f}±{values.std(ddof=1) * 100:.2f}%" if values.size > 1 \
        else _pct(values.mean())


def write_report(path: Path, runs: Runs, result: dict, hashes: dict) -> None:
    cfg = runs.config
    ref = result["reference_seed"]
    L = [
        "# PrimAttack v2 — paired optimizer comparison (CICIDS2017-DistriNet)",
        "",
        "Upgraded PrimAttack search (`prim_search_*`, `outputs/full_adv_eval_primattack_v2`) versus",
        "the replaced Adam/sigmoid `(p, α)` optimizer (`prim_opt_*`, `outputs/full_adv_eval`) and",
        "versus the in-run random-feasible control (`prim_rand_*`).",
        "",
        "## Design",
        "",
        f"- Victims: {', '.join(cfg['victims'])} (never pooled). Classes: {', '.join(cfg['classes'])}.",
        f"- Rows: the frozen clean-correct selection, {cfg['n_per_class_cap']} flows per victim/class; "
        "every victim/class `sha256_sample_ids` matches `outputs/full_adv_eval/selection.json`, and row "
        "order is asserted per artifact.",
        f"- Primary paired unit: one flow at reference seed {ref}; the four classes are pooled within a "
        f"victim (n = {len(cfg['classes'])}×{cfg['n_per_class_cap']}). Seed means ± SD over "
        f"{cfg['seeds']} are descriptive only.",
        "- McNemar: exact binomial when b+c<25, otherwise continuity-corrected χ². 95% CI: Newcombe "
        "square-and-add paired interval for rate(A) − rate(B). Holm correction within each contrast × "
        "outcome family (27 tests = 3 victims × 3 budgets × 3 modes).",
        "- Outcomes: targeted ∧ hybrid_valid ⊇ ∧ primitive_feasible ⊇ ∧ semantic PASS (nested); "
        "True-IDSR = targeted ∧ valid ∧ in_distribution (per-class VAE Mahalanobis gate, val-anchored "
        "p95; realism, kept outside structural validity).",
        "- The old and new runs differ in timing primitive as well as optimizer: old timing is "
        "proportional dilation `α`; new timing is `(delay, shape)`, which contains `α` as `shape=0`. "
        "Padding-only rows isolate the optimizer change.",
        "- **Post-hoc caveat:** the frozen roster was inspected diagnostically (CNN/DoS p75 exhaustive "
        "padding) before the upgrade was designed, so these are post-hoc paired comparisons on the same "
        "rows, not a held-out confirmation.",
        "",
    ]
    for contrast, title in (("search_vs_old", "New search vs old optimizer"),
                            ("search_vs_random", "New search vs random-feasible control")):
        L += [f"## {title}", ""]
        for outcome, label in OUTCOMES.items():
            L += [f"### {label}", "",
                  "| Victim | Budget | Mode | A (seed ref) | B (seed ref) | A only | B only | Δ [95% CI] | "
                  "p | Holm p | A seeds mean±SD | B seeds mean±SD |",
                  "|---|---|---|--:|--:|--:|--:|---|--:|--:|--:|--:|"]
            for row in result["rows"]:
                if row["contrast"] != contrast or row["outcome"] != outcome:
                    continue
                lo, hi = row["ci95"]
                L.append(
                    f"| {row['victim']} | {row['budget']} | {row['mode']} | {_pct(row['rate_A'])} | "
                    f"{_pct(row['rate_B'])} | {row['n10_A_only']} | {row['n01_B_only']} | "
                    f"{row['diff_A_minus_B'] * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}] | "
                    f"{row['p_value']:.2e} | {row['holm_p']:.2e} | {_seed_mean(row, 'rate_A')} | "
                    f"{_seed_mean(row, 'rate_B')} |"
                )
            L.append("")
    L += ["## Per-class counts, joint p75, reference seed (targeted ∧ valid)", "",
          "| Victim | Class | new search | old optimizer | new random |", "|---|---|--:|--:|--:|"]
    for victim in cfg["victims"]:
        old_row = next(r for r in result["rows"] if r["contrast"] == "search_vs_old"
                       and r["victim"] == victim and r["budget"] == "p75"
                       and r["mode"] == "joint" and r["outcome"] == "targeted_valid")
        rand_row = next(r for r in result["rows"] if r["contrast"] == "search_vs_random"
                        and r["victim"] == victim and r["budget"] == "p75"
                        and r["mode"] == "joint" and r["outcome"] == "targeted_valid")
        for class_name in cfg["classes"]:
            L.append(
                f"| {victim} | {class_name} | {old_row['per_class_ref_seed'][class_name]['A']} | "
                f"{old_row['per_class_ref_seed'][class_name]['B']} | "
                f"{rand_row['per_class_ref_seed'][class_name]['B']} |"
            )
    L += ["", "## Selection hashes", "", "| Victim | Class | sha256_sample_ids |", "|---|---|---|"]
    for victim, per_class in hashes.items():
        for class_name, digest in per_class.items():
            L.append(f"| {victim} | {class_name} | `{digest[:16]}…` |")
    L += ["", "## Claim boundary", "",
          "Feature-space proxy on aggregate CICFlowMeter rows: no PCAP is edited or replayed, and the "
          "affine delay allocation is checked only through flow summaries. Validity means schema/"
          "extractor/protocol/mined consistency; primitive feasibility means the projected integer "
          "controls lie inside the train-calibrated box; semantic PASS is a flow-level proxy and is "
          "NOT_FULLY_TESTABLE for Recon/BruteForce. Unbounded rows are not budget-compliant claims.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-dir", type=Path, default=REPO_ROOT / "outputs/full_adv_eval")
    parser.add_argument("--new-dir", type=Path,
                        default=REPO_ROOT / "outputs/full_adv_eval_primattack_v2")
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument("--report", type=Path,
                        default=REPO_ROOT / "PRIMATTACK_V2_OPTIMIZER_COMPARISON.md")
    args = parser.parse_args()
    runs = Runs(args.old_dir, args.new_dir)
    hashes = runs.check_selection()
    result = analyze(runs, args.reference_seed)
    result["selection_sha256"] = hashes
    (args.new_dir / "optimizer_comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    write_report(args.report, runs, result, hashes)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
