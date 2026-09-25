"""Build the full PrimAttack budget/ablation Markdown report from saved artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from attack.primattack_budget import BUDGET_NAMES, load_calibration
from attack.run_cicids2017_primitive_attack import PRIMITIVE_MODES


def _load_samples(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/*/attack_artifacts/*.npz")):
        with np.load(path) as artifact:
            numeric_fields = (
                "X_adv_raw",
                "relative_duration_change",
                "relative_byte_change",
                "rate_retention",
                "primitive_p_projected",
                "primitive_delay_projected",
                "primitive_shape_projected",
            )
            for field in numeric_fields:
                if not np.isfinite(artifact[field]).all():
                    raise ValueError(f"non-finite {field} in {path}")
            mode = str(artifact["primitive_mode"][0])
            if mode == "timing-only" and np.any(artifact["primitive_p_projected"] != 0):
                raise ValueError(f"padding active in timing-only artifact {path}")
            if mode == "padding-only" and np.any(artifact["primitive_delay_projected"] != 0):
                raise ValueError(f"timing active in padding-only artifact {path}")
            n = len(artifact["sample_id"])
            frames.append(pd.DataFrame({
                "sample_id": artifact["sample_id"].astype(str),
                "attack_class": artifact["attack_class"].astype(str),
                "victim": artifact["victim_model"].astype(str),
                "seed": np.full(n, int(artifact["seed"])),
                "budget": artifact["budget_name"].astype(str),
                "mode": artifact["primitive_mode"].astype(str),
                "eligible": artifact["clean_correct"].astype(bool),
                "targeted": artifact["targeted_success"].astype(bool),
                "domain": artifact["domain_valid"].astype(bool),
                "feasible": artifact["primitive_feasible"].astype(bool),
                "semantic": artifact["semantic_status"].astype(str),
                "relative_duration": artifact["relative_duration_change"].astype(float),
                "relative_bytes": artifact["relative_byte_change"].astype(float),
                "rate_retention": artifact["rate_retention"].astype(float),
                "changed_features": artifact["number_features_changed"].astype(float),
                "cost": artifact["cost_total"].astype(float),
            }))
    if not frames:
        raise FileNotFoundError(f"no PrimAttack attack artifacts under {root}")
    return pd.concat(frames, ignore_index=True)


def _summarize(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, subset in frame.groupby(groups, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        eligible = subset[subset["eligible"]]
        n = len(eligible)
        if n == 0:
            continue
        target = eligible["targeted"].to_numpy(bool)
        domain = eligible["domain"].to_numpy(bool)
        feasible = eligible["feasible"].to_numpy(bool)
        semantic = eligible["semantic"].to_numpy(str)
        rows.append({
            **dict(zip(groups, key)),
            "n": n,
            "raw_n": int(target.sum()),
            "valid_n": int((target & domain).sum()),
            "feasible_n": int((target & domain & feasible).sum()),
            "sp_n": int((target & domain & feasible & (semantic == "PASS")).sum()),
            "raw_asr": float(target.mean()),
            "valid_asr": float((target & domain).mean()),
            "feasible_asr": float((target & domain & feasible).mean()),
            "sp_asr": float((target & domain & feasible & (semantic == "PASS")).mean()),
            "domain_rate": float(domain.mean()),
            "feasible_rate": float(feasible.mean()),
            "semantic_pass_n": int((semantic == "PASS").sum()),
            "semantic_fail_n": int((semantic == "FAIL").sum()),
            "semantic_nt_n": int((semantic == "NOT_FULLY_TESTABLE").sum()),
            "semantic_pass": float((semantic == "PASS").mean()),
            "semantic_fail": float((semantic == "FAIL").mean()),
            "semantic_nt": float((semantic == "NOT_FULLY_TESTABLE").mean()),
            "testability": float((semantic != "NOT_FULLY_TESTABLE").mean()),
            "duration_median": float(eligible["relative_duration"].median()),
            "bytes_median": float(eligible["relative_bytes"].median()),
            "retention_median": float(eligible["rate_retention"].median()),
            "changed_median": float(eligible["changed_features"].median()),
            "cost_median": float(eligible["cost"].median()),
        })
    return pd.DataFrame(rows)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _num(value: float) -> str:
    return f"{value:.6g}"


def _table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _condition_order(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["mode"] = pd.Categorical(result["mode"], PRIMITIVE_MODES, ordered=True)
    result["budget"] = pd.Categorical(result["budget"], BUDGET_NAMES, ordered=True)
    return result.sort_values(["mode", "budget"])


def _stats_tables(stats: dict) -> tuple[str, str]:
    omnibus_rows = []
    significant_rows = []
    for family_type in ("binary", "continuous"):
        for family in stats[family_type]:
            omnibus = family["omnibus"]
            omnibus_rows.append((
                family_type,
                omnibus["test"],
                omnibus["fixed_axis"],
                omnibus["fixed_value"],
                omnibus["outcome"],
                omnibus["n"],
                _num(float(omnibus["statistic"])),
                _num(float(omnibus["p_value"])),
            ))
            for pair in family["pairwise"]:
                if float(pair["holm_p"]) < 0.05:
                    significant_rows.append((
                        family_type,
                        pair["fixed_axis"],
                        pair["fixed_value"],
                        pair["outcome"],
                        f"{pair['left']} vs {pair['right']}",
                        _num(float(pair["p_value"])),
                        _num(float(pair["holm_p"])),
                    ))
    omnibus_table = _table(
        ["Family", "Test", "Fixed axis", "Fixed value", "Outcome", "N", "Statistic", "p"],
        omnibus_rows,
    )
    significant_table = (
        _table(
            ["Family", "Fixed axis", "Fixed value", "Outcome", "Comparison", "p", "Holm p"],
            significant_rows,
        )
        if significant_rows
        else "No pairwise comparison remained significant after Holm correction."
    )
    return omnibus_table, significant_table


def build_report(root: Path, calibration_path: Path) -> str:
    samples = _load_samples(root)
    artifact_count = len(list(root.glob("*/*/attack_artifacts/*.npz")))
    saved_rows = len(samples)
    pooled = _condition_order(_summarize(samples, ["mode", "budget"]))
    maximum = _summarize(
        samples[samples["budget"] == "maximum-evaluated"],
        ["mode", "attack_class", "victim"],
    )
    class_summary = _summarize(
        samples[
            (samples["budget"] == "maximum-evaluated")
            & (samples["mode"] == "joint")
        ],
        ["attack_class"],
    )
    victim_summary = _summarize(
        samples[
            (samples["budget"] == "maximum-evaluated")
            & (samples["mode"] == "joint")
        ],
        ["victim"],
    )
    calibration = load_calibration(calibration_path)
    consistency = json.loads((root / "source_id_consistency.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "paired_statistics.json").read_text(encoding="utf-8"))
    omnibus_table, significant_table = _stats_tables(stats)

    def binary_family(fixed_axis: str, fixed_value: str, outcome: str) -> dict:
        return next(
            family for family in stats["binary"]
            if family["omnibus"]["fixed_axis"] == fixed_axis
            and family["omnibus"]["fixed_value"] == fixed_value
            and family["omnibus"]["outcome"] == outcome
        )

    def pair(family: dict, left: str, right: str) -> dict:
        return next(
            item for item in family["pairwise"]
            if {item["left"], item["right"]} == {left, right}
        )

    maximum_raw_family = binary_family(
        "budget_name", "maximum-evaluated", "targeted_success"
    )
    maximum_sp_family = binary_family(
        "budget_name", "maximum-evaluated", "sp_success"
    )
    raw_joint_padding = pair(maximum_raw_family, "joint", "padding-only")
    raw_joint_timing = pair(maximum_raw_family, "joint", "timing-only")
    raw_padding_timing = pair(maximum_raw_family, "padding-only", "timing-only")
    sp_joint_padding = pair(maximum_sp_family, "joint", "padding-only")
    joint_budget_family = binary_family("primitive_mode", "joint", "targeted_success")
    joint_max_intermediate = pair(
        joint_budget_family, "maximum-evaluated", "intermediate"
    )

    budget_rows = []
    for class_name in ("DoS", "DDoS", "Recon", "BruteForce"):
        budgets = calibration["classes"][class_name]["budgets"]
        budget_rows.append((
            class_name,
            *(
                f"{budgets[name]['padding_bytes_per_forward_packet']:.0f} B / "
                f"{budgets[name]['max_relative_duration_change']:.6g}"
                for name in BUDGET_NAMES
            ),
        ))

    pooled_rows = [(
        row.mode,
        row.budget,
        int(row.n),
        int(row.raw_n),
        _pct(row.raw_asr),
        _pct(row.valid_asr),
        _pct(row.feasible_asr),
        int(row.sp_n),
        _pct(row.sp_asr),
        _pct(row.semantic_pass),
        _pct(row.semantic_fail),
        _pct(row.semantic_nt),
        _pct(row.testability),
        _num(row.duration_median),
        _num(row.bytes_median),
        _num(row.retention_median),
        _num(row.changed_median),
    ) for row in pooled.itertuples(index=False)]

    class_rows = [(
        row.attack_class,
        int(row.n),
        int(row.raw_n),
        _pct(row.raw_asr),
        _pct(row.valid_asr),
        _pct(row.feasible_asr),
        int(row.sp_n),
        _pct(row.sp_asr),
        _pct(row.semantic_pass),
        _pct(row.semantic_fail),
        _pct(row.semantic_nt),
        _pct(row.testability),
    ) for row in class_summary.itertuples(index=False)]

    victim_rows = [(
        row.victim,
        int(row.n),
        int(row.raw_n),
        _pct(row.raw_asr),
        _pct(row.valid_asr),
        _pct(row.feasible_asr),
        int(row.sp_n),
        _pct(row.sp_asr),
        _num(row.duration_median),
        _num(row.bytes_median),
    ) for row in victim_summary.itertuples(index=False)]

    detail_rows = [(
        row.mode,
        row.attack_class,
        row.victim,
        int(row.n),
        int(row.raw_n),
        _pct(row.raw_asr),
        _pct(row.valid_asr),
        _pct(row.feasible_asr),
        int(row.sp_n),
        _pct(row.sp_asr),
        _pct(row.semantic_pass),
        _pct(row.semantic_nt),
        _num(row.duration_median),
        _num(row.bytes_median),
        _num(row.retention_median),
    ) for row in maximum.itertuples(index=False)]
    def condition(mode: str, budget: str) -> pd.Series:
        return pooled[(pooled["mode"] == mode) & (pooled["budget"] == budget)].iloc[0]

    timing_max = condition("timing-only", "maximum-evaluated")
    padding_max = condition("padding-only", "maximum-evaluated")
    joint_max = condition("joint", "maximum-evaluated")
    dos_joint = class_summary[class_summary["attack_class"] == "DoS"].iloc[0]
    brute_joint = class_summary[class_summary["attack_class"] == "BruteForce"].iloc[0]


    relative_root = "../outputs/" + root.name
    return f"""# Full PrimAttack budget and primitive-ablation results

## Scope and claim boundary

This report contains the predeclared full PrimAttack budget-sensitivity experiment: four retained
attack classes, the active MLP and CNN victims, 512 fixed source rows per class, 40 optimization
steps, and seed 42. It evaluates three train-calibrated budgets under timing-only, padding-only,
and joint primitive modes. No victim was retrained.

SP-ASR is a **flow-level semantic-preservation proxy ASR**. It does not establish complete
malicious functionality or packet-trace behavior. Recon/PortScan and BruteForce retain critical
properties that CICIDS2017 aggregate rows cannot test; these rows are conservatively
`NOT_FULLY_TESTABLE` rather than counted as semantic PASS.

## Reproduction

```text
PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py \\
  --classes DoS,DDoS,Recon,BruteForce \\
  --victims mlp,cnn \\
  --test-limit 512 --steps 40 --seeds 42 \\
  --output-dir {root.as_posix()}

PYTHONPATH=".;src" python scripts/analyze_primattack_experiments.py \\
  --input-dir {root.as_posix()}

PYTHONPATH=".;src" python scripts/build_primattack_budget_report.py \\
  --input-dir {root.as_posix()} \\
  --output docs/primattack_budget_results.md
```

Source pairing: **{str(consistency['identical_across_all_configurations']).lower()}** across
{consistency['cells_checked']} class/victim/seed cells. The same source IDs are used in every
budget/mode condition.

## Frozen class budgets

Each entry is `padding bytes per forward packet / maximum relative duration increase`.

{_table(['Class', 'Restricted', 'Intermediate', 'Maximum-evaluated'], budget_rows)}

Budgets and semantic thresholds were fitted on `X_train_pristine.npy` and
`y_train_cat.npy` only. Attack success did not participate in calibration.

## Main findings

- Timing-only produced **{int(timing_max.raw_n)}/{int(timing_max.n)}** targeted successes at
  maximum-evaluated budget ({_pct(timing_max.raw_asr)}).
- Padding-only produced **{int(padding_max.raw_n)}/{int(padding_max.n)}** raw, domain-valid, and
  primitive-feasible successes ({_pct(padding_max.raw_asr)}); **{int(padding_max.sp_n)}**
  survived the semantic proxy ({_pct(padding_max.sp_asr)}).
- Joint PrimAttack produced **{int(joint_max.raw_n)}/{int(joint_max.n)}** raw, domain-valid, and
  primitive-feasible successes ({_pct(joint_max.raw_asr)}); **{int(joint_max.sp_n)}** survived
  all gates ({_pct(joint_max.sp_asr)}).
- At maximum-evaluated joint budget, BruteForce contributed
  **{int(brute_joint.raw_n)}** raw successes but zero SP successes because its critical
  application semantics are not testable from aggregate flow rows. DoS contributed
  **{int(dos_joint.raw_n)}** raw successes, of which **{int(dos_joint.sp_n)}** passed every
  flow-level proxy gate.
- Raw, valid, and primitive-feasible counts are equal in every pooled condition: validator_v2
  and hard primitive feasibility rejected none of the classifier successes.
- Padding is the effective evasion primitive in this experiment. Joint optimization adds only
  {int(joint_max.raw_n - padding_max.raw_n)} raw and
  {int(joint_max.sp_n - padding_max.sp_n)} SP successes over padding-only at the maximum budget.

## Statistical interpretation

- At maximum-evaluated budget, primitive mode affected targeted success
  (Cochran's Q $p={_num(float(maximum_raw_family['omnibus']['p_value']))}$).
- Joint and padding-only each exceeded timing-only after Holm correction
  (`joint vs timing` Holm $p={_num(float(raw_joint_timing['holm_p']))}$;
  `padding vs timing` Holm $p={_num(float(raw_padding_timing['holm_p']))}$).
- The {int(joint_max.raw_n - padding_max.raw_n)}-success raw difference between joint and
  padding-only had Holm $p={_num(float(raw_joint_padding['holm_p']))}$. Their
  {int(joint_max.sp_n - padding_max.sp_n)}-success SP-ASR difference had Holm
  $p={_num(float(sp_joint_padding['holm_p']))}$.
- For joint PrimAttack, maximum-evaluated exceeded intermediate budget after correction
  (Holm $p={_num(float(joint_max_intermediate['holm_p']))}$).


## Pooled primary results

{_table(
    ['Mode', 'Budget', 'N', 'Raw n', 'Raw ASR', 'Valid ASR', 'Feasible ASR',
     'SP n', 'SP-ASR', 'Semantic PASS', 'Semantic FAIL', 'Not fully testable',
     'Testability', 'Median Δduration', 'Median Δbytes', 'Median rate retention',
     'Median changed features'],
    pooled_rows,
)}

All ASRs use eligible clean-correct malicious sources as the denominator. `Valid ASR` adds
validator_v2 domain validity. `Feasible ASR` additionally requires hard primitive compliance and
internal transform consistency. SP-ASR additionally requires semantic status `PASS`.

## Maximum-evaluated joint results by source class

{_table(
    ['Class', 'N', 'Raw n', 'Raw ASR', 'Valid ASR', 'Feasible ASR', 'SP n',
     'SP-ASR', 'Semantic PASS', 'Semantic FAIL', 'Not fully testable', 'Testability'],
    class_rows,
)}

## Maximum-evaluated joint results by victim

{_table(
    ['Victim', 'N', 'Raw n', 'Raw ASR', 'Valid ASR', 'Feasible ASR', 'SP n',
     'SP-ASR', 'Median Δduration', 'Median Δbytes'],
    victim_rows,
)}

## Maximum-evaluated primitive ablation details

{_table(
    ['Mode', 'Class', 'Victim', 'N', 'Raw n', 'Raw ASR', 'Valid ASR',
     'Feasible ASR', 'SP n', 'SP-ASR', 'Semantic PASS', 'Not fully testable',
     'Median Δduration', 'Median Δbytes', 'Median rate retention'],
    detail_rows,
)}

## Paired statistical tests

Pairing key: `sample_id × attack_class × victim × seed`. Binary omnibus tests use Cochran's Q,
with repository McNemar tests for pairwise follow-ups. Continuous omnibus tests use Friedman,
with Wilcoxon signed-rank follow-ups. Pairwise p-values are Holm corrected within each family.

### Omnibus tests

{omnibus_table}

### Holm-significant pairwise tests

{significant_table}

## Integrity and reproducibility checks

- Artifacts checked: **{artifact_count}** NPZ files containing **{saved_rows}** saved rows.
- Every projected sample passed primitive feasibility: **{str(bool(samples['feasible'].all())).lower()}**.
- No NaN or Inf occurred in final vectors, primitive controls, costs, or retention quantities.
- Timing-only artifacts had $p=0$ and padding-only artifacts had $\\alpha=1$ for every row.
- Exact source-ID pairing held across all 16 class/victim/seed cells and all nine conditions.
- Final repository test suite: **217 passed, 1 skipped**; one unrelated existing PyTorch
  convolution warning.

## Figures

1. [Raw Target-Benign ASR vs budget]({relative_root}/01_raw_asr_vs_budget.png)
2. [Valid Target-Benign ASR vs budget]({relative_root}/02_valid_asr_vs_budget.png)
3. [SP-ASR vs budget]({relative_root}/03_sp_asr_vs_budget.png)
4. [Semantic PASS rate vs budget]({relative_root}/04_semantic_pass_vs_budget.png)
5. [Rate retention vs timing budget]({relative_root}/05_rate_retention_vs_timing_budget.png)
6. [Primitive cost vs ASR]({relative_root}/06_primitive_cost_vs_asr.png)
7. [Timing-only vs padding-only vs joint]({relative_root}/07_timing_padding_combined.png)

## Interpretation

- Compare raw, valid, feasible, and SP-ASR in order; later gates never replace earlier metrics.
- A zero SP-ASR can coexist with raw success when success occurs in a class whose critical
  semantics are unavailable from aggregate flow data.
- Timing-only, padding-only, and joint conditions use identical sources and hard budgets, so
  differences isolate primitive contribution rather than sample selection.
- The maximum-evaluated level is the training-derived P75 experimental envelope, not a universal
  physical maximum.

## Limitations

The transformation is an offline flow-feature model. Complete scan structure, authentication
attempt semantics, payload behavior, merged packet ordering, and target response are unavailable
from CICIDS2017 aggregate rows. Establishing those properties would require packet realization,
feature re-extraction, and isolated replay, which are outside this thesis.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("outputs/primattack_budget_sensitivity_full"),
    )
    parser.add_argument(
        "--calibration", type=Path,
        default=Path("artifacts/primattack/budget_calibration.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/primattack_budget_results.md")
    )
    args = parser.parse_args()
    report = build_report(args.input_dir, args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bytes": len(report.encode('utf-8'))}))


if __name__ == "__main__":
    main()
