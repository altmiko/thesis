"""Paired analysis of the PrimAttack validity-gate ablation over every FINAL PrimAttack stage.

The FINAL suite ran PrimAttack with validator_v2 ``hybrid_valid`` inside the search's success
predicate (``RealizedSearch``: success = objective hit AND valid). The ablation reruns every FINAL
PrimAttack configuration with ``run_primattack_optimizer_ablation.py --no-validity-gate`` (search
success = objective hit only). Both sides use the same frozen source flows, victims, seeds
42/2024/2026, budgets, hyperparameters and environment (thesis env, CUDA,
``CUBLAS_WORKSPACE_CONFIG=:4096:8``), and validator_v2 is re-computed post hoc on both sides, so
``valid_success`` means the same thing.

Inference follows the FINAL protocol: one paired outcome per source flow on the reference attack
seed 42, classes pooled within a victim, exact/continuity-corrected McNemar, Holm over the three
victims of each (dataset, configuration). Seeds 2024/2026 are descriptive (mean +- SD).

Ablation runs (``<ds>`` in {cicids2017, cicids2018}; victims ``mlp,cnn,ft_transformer`` or
``mlp-s42,cnn-s42,ft_transformer-s42``), each:

    python scripts/run_primattack_optimizer_ablation.py --dataset <ds> --device cuda \
        --selection-from FINAL_OUTPUTS/runs/<ds>_distrinet/baselines_untargeted/selection.json \
        --split test --seeds 42,2024,2026 --classes DoS,DDoS,Recon,BruteForce --victims <victims> \
        --eval-budget 256 --hybrid-steps 40 --hybrid-lr 0.1 --pgd-restarts 3 --pgd-step 0.05 \
        --pgd-momentum 0.75 --cw-stages 3 --cw-lr 0.5 --cw-c 1.0 --cw-kappa 0.0 \
        --no-validity-gate --objective <obj> --methods <m> --budgets <b> --modes <modes> \
        --output-dir outputs/primattack_nogate_ablation/<ds>_distrinet/<dir>

with (<obj>, <m>, <b>, <modes>, <dir>) as listed in ``RUNS_NOGATE`` below. Then:

    python scripts/analyze_gate_ablation.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from evaluation.paired_validity_gap import holm_adjust, mcnemar_test  # noqa: E402

RUNS = REPO_ROOT / "FINAL_OUTPUTS" / "runs"
ABL = REPO_ROOT / "outputs" / "primattack_nogate_ablation"
DATASETS = {
    "cicids2017_distrinet": ("mlp", "cnn", "ft_transformer"),
    "cicids2018_distrinet": ("mlp-s42", "cnn-s42", "ft_transformer-s42"),
}
CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
SEEDS = (42, 2024, 2026)
REF_SEED = 42

# Ablation output directories: (objective, methods, budgets, modes).
RUNS_NOGATE = {
    "untargeted_nogate": ("untargeted", "pgd", "maximum-evaluated,unbounded", "joint"),
    "targeted_nogate": ("targeted", "pgd", "maximum-evaluated,unbounded", "joint"),
    "targeted_optimizers_nogate": ("targeted", "hybrid,cw", "maximum-evaluated", "joint"),
    "targeted_budgets_p50_nogate": ("targeted", "hybrid,pgd", "intermediate", "joint"),
    "targeted_budgets_hybrid_unb_nogate": ("targeted", "hybrid", "unbounded", "joint"),
    "hybrid_objective_untargeted_nogate": ("untargeted", "hybrid", "maximum-evaluated", "joint"),
    "untargeted_modes_nogate": ("untargeted", "pgd", "maximum-evaluated",
                                "timing-only,padding-only"),
    "untargeted_random_null_nogate": ("untargeted", "random", "maximum-evaluated", "joint"),
}

# One row per FINAL PrimAttack configuration:
# (FINAL stage, ablation dir, objective, budget tag, artifact method tag).
CONFIGS = [
    ("primattack_untargeted", "untargeted_nogate", "untargeted", "p75", "pgd"),
    ("primattack_untargeted", "untargeted_nogate", "untargeted", "unb", "pgd"),
    ("primattack_untargeted_modes", "untargeted_modes_nogate", "untargeted", "p75", "pgd-timing-only"),
    ("primattack_untargeted_modes", "untargeted_modes_nogate", "untargeted", "p75", "pgd-padding-only"),
    ("primattack_untargeted_random_null", "untargeted_random_null_nogate", "untargeted", "p75", "random"),
    ("primattack_hybrid_objective_untargeted", "hybrid_objective_untargeted_nogate", "untargeted", "p75", "hybrid"),
    ("primattack_targeted_optimizers", "targeted_nogate", "targeted", "p75", "pgd"),
    ("primattack_targeted_optimizers", "targeted_optimizers_nogate", "targeted", "p75", "hybrid"),
    ("primattack_targeted_optimizers", "targeted_optimizers_nogate", "targeted", "p75", "cw"),
    ("primattack_hybrid_objective_targeted", "targeted_optimizers_nogate", "targeted", "p75", "hybrid"),
    ("primattack_targeted_budgets", "targeted_budgets_p50_nogate", "targeted", "p50", "pgd"),
    ("primattack_targeted_budgets", "targeted_budgets_p50_nogate", "targeted", "p50", "hybrid"),
    ("primattack_targeted_budgets", "targeted_nogate", "targeted", "unb", "pgd"),
    ("primattack_targeted_budgets", "targeted_budgets_hybrid_unb_nogate", "targeted", "unb", "hybrid"),
]


def load_pair(dataset: str, stage: str, nogate_dir: str, objective: str, budget: str,
              tag: str, victim: str, seed: int) -> dict:
    """Concatenate the four classes of one victim; assert both sides attack identical flows."""
    out: dict[str, list] = {k: [] for k in ("g_raw", "g_valid", "n_raw", "n_valid", "changed")}
    for cname in CLASSES:
        name = f"{victim}__{cname}__{budget}__{tag}__seed{seed}.npz"
        with np.load(RUNS / dataset / stage / "artifacts" / name) as g, \
             np.load(ABL / dataset / nogate_dir / "artifacts" / name) as n:
            if not np.array_equal(g["sample_id"], n["sample_id"]):
                raise AssertionError(f"source flows differ: {dataset} {stage} {name}")
            if str(g["objective"]) != objective or str(n["objective"]) != objective:
                raise AssertionError(f"objective mismatch: {dataset} {stage} {name}")
            out["g_raw"].append(g["raw_success"]); out["g_valid"].append(g["valid_success"])
            out["n_raw"].append(n["raw_success"]); out["n_valid"].append(n["valid_success"])
            out["changed"].append(np.any(g["adv_raw"] != n["adv_raw"], axis=1))
    return {k: np.concatenate(v).astype(bool) for k, v in out.items()}


def pct(x: np.ndarray) -> float:
    return 100.0 * float(x.mean())


def main() -> None:
    cell_rows, test_rows = [], []
    for dataset, victims in DATASETS.items():
        for stage, nogate_dir, objective, budget, tag in CONFIGS:
            family = []
            for victim in victims:
                per_seed = {s: load_pair(dataset, stage, nogate_dir, objective, budget, tag, victim, s)
                            for s in SEEDS}
                for s, d in per_seed.items():
                    cell_rows.append({
                        "dataset": dataset, "stage": stage, "objective": objective,
                        "budget": budget, "method": tag, "victim": victim, "seed": s,
                        "n": len(d["g_valid"]),
                        "gated_raw_asr": pct(d["g_raw"]), "gated_valid_asr": pct(d["g_valid"]),
                        "nogate_raw_asr": pct(d["n_raw"]), "nogate_valid_asr": pct(d["n_valid"]),
                        "nogate_invalid_hits": int((d["n_raw"] & ~d["n_valid"]).sum()),
                        "gated_only_valid": int((d["g_valid"] & ~d["n_valid"]).sum()),
                        "nogate_only_valid": int((d["n_valid"] & ~d["g_valid"]).sum()),
                        "final_flow_changed": int(d["changed"].sum()),
                    })
                ref = per_seed[REF_SEED]
                b = int((ref["g_valid"] & ~ref["n_valid"]).sum())
                c = int((ref["n_valid"] & ~ref["g_valid"]).sum())
                t = mcnemar_test(b, c)
                seeds_g = [pct(per_seed[s]["g_valid"]) for s in SEEDS]
                seeds_n = [pct(per_seed[s]["n_valid"]) for s in SEEDS]
                family.append({
                    "dataset": dataset, "stage": stage, "objective": objective, "budget": budget,
                    "method": tag, "victim": victim, "n": len(ref["g_valid"]),
                    "gated_valid_mean": float(np.mean(seeds_g)),
                    "gated_valid_sd": float(np.std(seeds_g, ddof=1)),
                    "nogate_valid_mean": float(np.mean(seeds_n)),
                    "nogate_valid_sd": float(np.std(seeds_n, ddof=1)),
                    "nogate_raw_mean": float(np.mean([pct(per_seed[s]["n_raw"]) for s in SEEDS])),
                    "ref_gated_only": b, "ref_nogate_only": c,
                    "ref_diff_pp": 100.0 * (c - b) / len(ref["g_valid"]),
                    "test": t["test_variant"], "p_value": t["p_value"],
                })
            for row, p in zip(family, holm_adjust([r["p_value"] for r in family])):
                row["p_holm"] = p
            test_rows.extend(family)

    ABL.mkdir(parents=True, exist_ok=True)
    for path, rows in ((ABL / "gate_ablation_cells.csv", cell_rows),
                       (ABL / "gate_ablation_tests.csv", test_rows)):
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)

    lines = [
        "# PrimAttack validity-gate ablation (all FINAL PrimAttack stages)",
        "",
        "Gated = canonical FINAL cells (search success = hit AND validator_v2). No gate = "
        "`--no-validity-gate` (search success = hit only). Same frozen flows, victims, seeds "
        "42/2024/2026, budgets, hyperparameters and environment; validator_v2 is re-computed post "
        "hoc on both sides. Valid ASR = mean +- SD over the 3 attack seeds (n = 3200 flows per "
        "victim, 4 classes pooled). McNemar on seed 42 (one outcome per flow), Holm over the 3 "
        "victims of each dataset x configuration.",
        "",
        "| Dataset | FINAL stage | Objective | Budget | Method | Victim | Valid ASR gated "
        "| Valid ASR no gate | Raw ASR no gate | Gated-only / no-gate-only (s42) | McNemar p (Holm) "
        "| Final flows changed (3 seeds) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    changed = {}
    for r in cell_rows:
        k = (r["dataset"], r["stage"], r["budget"], r["method"], r["victim"])
        changed[k] = changed.get(k, 0) + r["final_flow_changed"]
    for r in test_rows:
        k = (r["dataset"], r["stage"], r["budget"], r["method"], r["victim"])
        lines.append(
            f"| {r['dataset'].split('_')[0]} | {r['stage']} | {r['objective']} | {r['budget']} "
            f"| {r['method']} | {r['victim']} "
            f"| {r['gated_valid_mean']:.2f}% +- {r['gated_valid_sd']:.2f} "
            f"| {r['nogate_valid_mean']:.2f}% +- {r['nogate_valid_sd']:.2f} "
            f"| {r['nogate_raw_mean']:.2f}% | {r['ref_gated_only']} / {r['ref_nogate_only']} "
            f"| {r['p_holm']:.3g} | {changed[k]} |")
    tot = {k: sum(r[k] for r in cell_rows) for k in
           ("n", "nogate_invalid_hits", "gated_only_valid", "nogate_only_valid", "final_flow_changed")}
    lines += [
        "",
        f"Totals over all {len(cell_rows)} victim x seed x configuration cells "
        f"({tot['n']} flow attacks): invalid hits kept by the ungated search "
        f"{tot['nogate_invalid_hits']}; flows valid only with the gate {tot['gated_only_valid']}; "
        f"flows valid only without the gate {tot['nogate_only_valid']}; final adversarial flow "
        f"differs between the two runs on {tot['final_flow_changed']} flows.",
        "",
        "Non-canonical ablation outside the locked protocol; feature-space proxy only "
        "(CLAUDE.md claim boundary).",
    ]
    (ABL / "gate_ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
