"""PrimAttack Hybrid: targeted (-> Benign) vs untargeted Valid ASR on identical source flows.

Reads the two stages written by ``scripts/run_primattack_optimizer_ablation.py`` with
``--methods hybrid --budgets maximum-evaluated --modes joint`` and the locked suite arguments:

* ``runs/<dataset>/primattack_hybrid_objective_untargeted``  (success: pred != source class)
* ``runs/<dataset>/primattack_hybrid_objective_targeted``    (success: pred == Benign)

Every artifact is audited by ``analyze_final_suite.Store`` (canonical sample IDs and order,
clean-input and checkpoint hashes, seeds {42, 2024, 2026}, raw success recomputed from the stored
prediction, validator_v2 re-run on the stored final flow, victim re-prediction) and the two
conditions must be paired flow-for-flow. Valid ASR = (objective met AND validator_v2
``hybrid_valid``) / attempted clean-correct flows. Statistics follow the locked protocol:
paired McNemar on reference seed 42 per (dataset, victim), Holm over the six comparisons;
seeds 2024 / 2026 descriptive (mean ± SD, per-seed discordant counts). The targeted stage is a
re-run of the Exp B Hybrid p75 cell and is checked against it flow-for-flow.

Writes ``FINAL_OUTPUTS/hybrid_targeted_vs_untargeted/`` (never touches Exp A-F outputs).

    python scripts/analyze_hybrid_objective_comparison.py [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_final_suite import (  # noqa: E402
    ALPHA, DATASETS, DS_LABEL, FINAL, REF_SEED, SEEDS, Store, assert_paired, fmt_p, md_table,
    prim_cond, ref_vector, seed_level, table_level,
)
from evaluation.paired_validity_gap import (  # noqa: E402
    holm_adjust, mcnemar_test, newcombe_paired_ci,
)

OUT = FINAL / "hybrid_targeted_vs_untargeted"
STAGE = "primattack_hybrid_objective_{}"
LABEL = {"untargeted": "Hybrid untargeted (p75)", "targeted": "Hybrid targeted→Benign (p75)"}
EXP_B_STAGE = "primattack_targeted_optimizers"


def display(victim: str) -> str:
    return victim.removesuffix("-s42")


def paired_tests(unt: pd.DataFrame, tgt: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, victims in DATASETS.items():
        for victim in victims:
            rec = {"dataset": dataset, "victim": victim, "outcome": "valid_success",
                   "seed_used": REF_SEED}
            for seed in SEEDS:
                u, ids_u = ref_vector(unt, dataset, victim, "valid_success", seed)
                t, ids_t = ref_vector(tgt, dataset, victim, "valid_success", seed)
                if not np.array_equal(ids_u, ids_t):
                    raise AssertionError(f"{dataset}/{victim}/seed {seed}: flows not paired")
                u_only, t_only = int((u & ~t).sum()), int((~u & t).sum())
                both, neither = int((u & t).sum()), int((~u & ~t).sum())
                diff = (u.mean() - t.mean()) * 100
                if seed == REF_SEED:
                    test = mcnemar_test(u_only, t_only)
                    lo, hi = newcombe_paired_ci(both, u_only, t_only, neither)
                    rec.update({
                        "n_paired": len(u), "untargeted_valid_successes": int(u.sum()),
                        "targeted_valid_successes": int(t.sum()),
                        "untargeted_valid_asr": float(u.mean()),
                        "targeted_valid_asr": float(t.mean()),
                        "diff_pp": diff, "diff_ci_lo_pp": 100 * lo, "diff_ci_hi_pp": 100 * hi,
                        "untargeted_only": u_only, "targeted_only": t_only,
                        "both": both, "neither": neither,
                        "test_variant": test["test_variant"],
                        "statistic": test["test_statistic"], "p_value": test["p_value"],
                    })
                else:
                    rec.update({f"diff_pp_seed{seed}": diff,
                                f"untargeted_only_seed{seed}": u_only,
                                f"targeted_only_seed{seed}": t_only})
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out.p_value.tolist())
    out["significant_holm"] = out.p_holm < ALPHA
    out["holm_family"] = "6 dataset x victim comparisons"
    return out


def reproduces_exp_b(store: Store, tgt: pd.DataFrame) -> pd.DataFrame:
    """Flow-level agreement of the targeted re-run with the Exp B Hybrid p75 cell."""
    exp_b = store.frame(prim_cond(EXP_B_STAGE, "hybrid", "p75", "targeted", "Exp B Hybrid"))
    key = ["dataset", "victim", "source_class", "seed", "sample_id"]
    m = tgt[key + ["valid_success", "raw_success"]].merge(
        exp_b[key + ["valid_success", "raw_success"]], on=key, suffixes=("", "_exp_b"),
        validate="one_to_one")
    if len(m) != len(tgt):
        raise AssertionError("targeted re-run and Exp B cells do not cover the same flows")
    return (m.assign(valid_differs=m.valid_success != m.valid_success_exp_b,
                     raw_differs=m.raw_success != m.raw_success_exp_b)
            .groupby(["dataset", "victim", "seed"], as_index=False)
            .agg(n=("sample_id", "size"), valid_success_differs=("valid_differs", "sum"),
                 raw_success_differs=("raw_differs", "sum")))


def pm(row: pd.Series, metric: str) -> str:
    return f"{100 * row[f'{metric}_mean']:.2f}% ± {100 * row[f'{metric}_sd']:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-recheck-predictions", action="store_true")
    args = ap.parse_args()

    store = Store(recheck_predictions=not args.no_recheck_predictions, device=args.device)
    unt = store.frame(prim_cond(STAGE.format("untargeted"), "hybrid", "p75", "untargeted",
                                LABEL["untargeted"]))
    tgt = store.frame(prim_cond(STAGE.format("targeted"), "hybrid", "p75", "targeted",
                                LABEL["targeted"]))
    assert_paired([unt, tgt], "Hybrid targeted vs untargeted")
    per_sample = pd.concat([unt, tgt], ignore_index=True)
    seed_df = seed_level(per_sample)
    table = table_level(seed_df)
    tests = paired_tests(unt, tgt)
    repro = reproduces_exp_b(store, tgt)

    victim_tbl = table[table.scope == "victim"].set_index(["dataset", "victim", "objective"])
    seed_v = seed_df[seed_df.scope == "victim"]
    final_rows, per_seed_rows = [], []
    for _, r in tests.iterrows():
        u = victim_tbl.loc[(r.dataset, r.victim, "untargeted")]
        t = victim_tbl.loc[(r.dataset, r.victim, "targeted")]
        diffs = np.array([
            100 * (u[f"valid_asr_seed{s}"] - t[f"valid_asr_seed{s}"]) for s in SEEDS])
        final_rows.append({
            "Dataset": DS_LABEL[r.dataset], "Victim": display(r.victim),
            "Untargeted Valid ASR": pm(u, "valid_asr"),
            "Targeted→Benign Valid ASR": pm(t, "valid_asr"),
            "Difference (Untargeted−Targeted)":
                f"{diffs.mean():+.2f} ± {diffs.std(ddof=1):.2f} pp",
            "Seed-42 discordance Untargeted-only/Targeted-only":
                f"{r.untargeted_only} / {r.targeted_only}",
            "Holm p": fmt_p(r.p_holm),
        })
        for obj, row in (("untargeted", u), ("targeted", t)):
            per_seed_rows.append({
                "Dataset": DS_LABEL[r.dataset], "Victim": display(r.victim),
                "Objective": "targeted→Benign" if obj == "targeted" else obj,
                "Valid ASR 42 / 2024 / 2026 (%)": " / ".join(
                    f"{100 * row[f'valid_asr_seed{s}']:.2f}" for s in SEEDS),
                "Valid ASR mean ± SD": pm(row, "valid_asr"),
                "Raw ASR 42 / 2024 / 2026 (%)": " / ".join(
                    f"{100 * row[f'raw_asr_seed{s}']:.2f}" for s in SEEDS),
                "Raw ASR mean ± SD": pm(row, "raw_asr"),
                "Validity gap (pp)": f"{row.validity_gap_pp_mean:.2f} ± "
                                     f"{row.validity_gap_pp_sd:.2f}",
            })
    final = pd.DataFrame(final_rows)

    test_rows = [{
        "Dataset": DS_LABEL[r.dataset], "Victim": display(r.victim), "N": r.n_paired,
        "Untargeted / targeted valid successes (seed 42)":
            f"{r.untargeted_valid_successes} / {r.targeted_valid_successes}",
        "Both / neither": f"{r.both} / {r.neither}",
        "Untargeted-only / targeted-only (42; 2024; 2026)":
            f"{r.untargeted_only}/{r.targeted_only}; "
            f"{r.untargeted_only_seed2024}/{r.targeted_only_seed2024}; "
            f"{r.untargeted_only_seed2026}/{r.targeted_only_seed2026}",
        "Δ seed 42 [Newcombe 95% CI]":
            f"{r.diff_pp:+.2f} pp [{r.diff_ci_lo_pp:+.2f}, {r.diff_ci_hi_pp:+.2f}]",
        "Test": r.test_variant,
        "Statistic": "—" if pd.isna(r.statistic) else f"{r.statistic:.3f}",
        "p": fmt_p(r.p_value), "Holm p": fmt_p(r.p_holm),
        "Significant (Holm, α = 0.05)": "yes" if r.significant_holm else "no",
    } for _, r in tests.iterrows()]

    cls = table[table.scope == "class"]
    class_rows = []
    for (d, v, c), g in cls.groupby(["dataset", "victim", "source_class"], sort=False):
        g = g.set_index("objective")
        class_rows.append({
            "Dataset": DS_LABEL[d], "Victim": display(v), "Class": c,
            "Untargeted Valid ASR": pm(g.loc["untargeted"], "valid_asr"),
            "Targeted→Benign Valid ASR": pm(g.loc["targeted"], "valid_asr"),
        })

    repro_ok = int(repro.valid_success_differs.sum()) == 0 and int(
        repro.raw_success_differs.sum()) == 0

    OUT.mkdir(parents=True, exist_ok=True)
    per_sample.to_parquet(OUT / "per_sample.parquet", index=False)
    seed_df.to_csv(OUT / "seed_level.csv", index=False)
    table.to_csv(OUT / "table_level.csv", index=False)
    tests.to_csv(OUT / "paired_tests.csv", index=False)
    final.to_csv(OUT / "final_table.csv", index=False)
    repro.to_csv(OUT / "targeted_rerun_vs_exp_b.csv", index=False)
    (OUT / "audit.json").write_text(json.dumps({
        **store.audit, "stages": [STAGE.format("untargeted"), STAGE.format("targeted")],
        "targeted_rerun_reproduces_exp_b": repro_ok,
    }, indent=2), encoding="utf-8")
    (OUT / "final_table.md").write_text(md_table(final) + "\n", encoding="utf-8")

    lines = [
        "# PrimAttack Hybrid — targeted vs untargeted Valid ASR",
        "",
        "Hybrid Search, p75 budget, joint mode, capability-aware padding, 256 victim evaluations "
        "per flow, validator_v2 `hybrid_valid`; the locked final-suite configuration "
        "(`runs/final_suite_config.json`, `00_PROTOCOL.md`). Both objectives attack the same "
        "canonical clean-correct source flows (`runs/<dataset>/baselines_untargeted/"
        "selection.json`, 800 per class, N = 3,200 per victim) with attack seeds 42, 2024, 2026.",
        "",
        "- Untargeted success: final prediction ≠ original malicious class.",
        "- Targeted success: final prediction = Benign.",
        "- Valid ASR = (success ∧ validator-valid final flow) / attempted clean-correct malicious "
        "flows. Mean ± SD (ddof = 1) over the three seed-level rates; classes pooled within a "
        "victim; datasets and victims never pooled.",
        "- Test: paired McNemar on reference seed 42 (exact binomial if discordant pairs < 25, "
        "else continuity-corrected χ²), Holm over the six dataset-victim comparisons. Seeds "
        "2024 / 2026 are descriptive only.",
        "- Each objective optimizes its own margin, so the two conditions are two separate "
        "searches on the same flows, not one search scored two ways.",
        "",
        "## Final table",
        "",
        md_table(final),
        "",
        "## Per-seed values (Raw ASR kept for reference)",
        "",
        md_table(pd.DataFrame(per_seed_rows)),
        "",
        "## Paired tests (Valid success, seed 42)",
        "",
        md_table(pd.DataFrame(test_rows)),
        "",
        "## Per class (Valid ASR, mean ± SD over seeds)",
        "",
        md_table(pd.DataFrame(class_rows)),
        "",
        "## Provenance and checks",
        "",
        f"- Integrity audit: {store.audit['npz_files']} artifacts, {store.audit['rows']:,} rows; "
        f"validator_v2 re-run on {store.audit['validator_rechecked_rows']:,} stored final flows; "
        f"victim re-prediction on {store.audit['prediction_rechecked_rows']:,} flows with "
        f"{store.audit['prediction_mismatches']} mismatches; identical sample IDs, order, "
        "clean-input and checkpoint hashes across both objectives and all seeds.",
        "- Targeted re-run vs the existing Exp B Hybrid p75 cell "
        f"(`{EXP_B_STAGE}`): flows whose valid / raw success differs = "
        f"{int(repro.valid_success_differs.sum())} / {int(repro.raw_success_differs.sum())} "
        f"({'reproduced exactly' if repro_ok else 'NOT identical, see targeted_rerun_vs_exp_b.csv'}).",
        f"- Run stages: `runs/<dataset>/{STAGE.format('untargeted')}`, "
        f"`runs/<dataset>/{STAGE.format('targeted')}`. This comparison is an added analysis "
        "(not part of the locked Exp A-F families); Exp D remains the pre-registered "
        "targeted-vs-untargeted test for the selected optimizer (Prim-PGD).",
        "- Feature-space proxy on CICFlowMeter aggregates; no PCAP edited or replayed.",
        "",
        "Files: `final_table.{md,csv}`, `paired_tests.csv`, `seed_level.csv`, `table_level.csv`, "
        "`per_sample.parquet`, `targeted_rerun_vs_exp_b.csv`, `audit.json`.",
    ]
    (OUT / "hybrid_targeted_vs_untargeted.md").write_text("\n".join(lines) + "\n",
                                                          encoding="utf-8")
    print(md_table(final))
    print(f"[ok] -> {OUT}")


if __name__ == "__main__":
    main()
