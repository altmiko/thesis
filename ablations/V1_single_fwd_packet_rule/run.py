"""V1 - Toggle-able validator rule: no forward IAT on single-forward-packet flows.

Ablation D6 found a validator_v2 gap: forward delay added to a flow with one forward packet
(``Fwd IAT Total`` 0 -> 1.2e7 µs with ``Total Fwd Packet = 1``) passes all four layers. This
experiment adds the definitional CICFlowMeter rule

    Total Fwd Packet <= 1  =>  Fwd IAT Total = Mean = Std = Max = Min = 0

(``ablations/common/extra_rules.py``, ``single_fwd_packet_no_fwd_iat``) as a toggle on top of
validator_v2. validator_v2 itself is the locked FINAL validator and is not modified. With
``--rule on`` the rule is ANDed into the search success predicate and into validity; with
``--rule off`` it is not. Every arm stores the rule's per-row verdict, so all arms report

* ``extended_valid_success`` = objective met ∧ validator_v2 ∧ rule (primary);
* ``valid_success`` = objective met ∧ validator_v2 (as in every other ablation).

PrimAttack variants (``--capability``):

* ``aware``   - canonical capability-aware PrimAttack (timing needs >= 2 forward packets, so the
                rule is expected never to bind);
* ``ablated`` - D6's capability-ablated PrimAttack (u = (p, D, s)), where the gap appears.

Arms: ``capaware_rule_off`` (= the shared reference configuration), ``capaware_rule_on``,
``nocap_rule_off`` (= D6 ``no_capability``), ``nocap_rule_on``. Shared protocol otherwise:
p75 + unbounded, both datasets, three victims, four classes, attack seeds 42/2024/2026.

The analysis also checks the rule on every genuine flow of the train, val and test splits of
both datasets (a sound rule must accept them all), and confirms that the rule-off arms reproduce
``ablations/reference`` and ``ablations/thesis_ablations/D6_capability_inference`` flow-for-flow.

    python ablations/V1_single_fwd_packet_rule/run.py --device cuda                  # all arms
    python ablations/V1_single_fwd_packet_rule/run.py --rule on --capability ablated  # one arm
"""
from __future__ import annotations

import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ablations.common.analysis import (  # noqa: E402
    Source, _groups, _seeds, compare_to_reference, pooled_rows, summary_markdown,
)
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.extra_rules import FWD_IAT_COLUMNS, single_fwd_packet_no_fwd_iat  # noqa: E402
from ablations.common.runner import (  # noqa: E402
    DATASETS, REFERENCE, REFERENCE_DIR, REPO_ROOT, Condition,
)
from datasets import get_adapter  # noqa: E402

RULE = "single_fwd_packet_no_fwd_iat"
CONDITIONS = [
    Condition("capaware_rule_off", "capability-aware PrimAttack, validator_v2 only"),
    Condition("capaware_rule_on", "capability-aware PrimAttack, validator_v2 + rule",
              extra_rules=(RULE,)),
    Condition("nocap_rule_off", "capability-ablated PrimAttack (u=(p,D,s)), validator_v2 only",
              capability_aware=False),
    Condition("nocap_rule_on", "capability-ablated PrimAttack (u=(p,D,s)), validator_v2 + rule",
              capability_aware=False, extra_rules=(RULE,)),
]
FAMILIES = (("aware", "capaware_rule_off", "capaware_rule_on"),
            ("ablated", "nocap_rule_off", "nocap_rule_on"))
SAME_AS = {"capaware_rule_off": (REFERENCE_DIR / "results", REFERENCE.name),
           "nocap_rule_off": (REPO_ROOT / "ablations" / "thesis_ablations"
                              / "D6_capability_inference" / "results", "no_capability")}


def add_arguments(ap) -> None:
    ap.add_argument("--rule", choices=("on", "off", "both"), default="both",
                    help="toggle the single-forward-packet IAT rule")
    ap.add_argument("--capability", choices=("aware", "ablated", "both"), default="both",
                    help="PrimAttack variant: capability-aware (canonical) or capability-ablated")


def select(args, conditions: list[Condition]) -> list[Condition]:
    rules = {"on": (True,), "off": (False,), "both": (True, False)}[args.rule]
    caps = {"aware": (True,), "ablated": (False,), "both": (True, False)}[args.capability]
    return [c for c in conditions
            if bool(c.extra_rules) in rules and c.capability_aware in caps]


def genuine_flow_check(results_dir: Path) -> list[str]:
    """The rule on every genuine flow of every split (must accept all of them)."""
    rows = []
    for dataset, spec in DATASETS.items():
        adapter = get_adapter(spec["cli"])
        index = {n: k for k, n in enumerate(adapter.feature_manifest().names)}
        names = adapter.class_mapping().names
        for split in ("train", "val", "test"):
            X = np.load(adapter._processed / f"X_{split}_pristine.npy", mmap_mode="r")
            y = np.load(adapter._processed / f"y_{split}_cat.npy")
            names_used = ("Total Fwd Packet",) + FWD_IAT_COLUMNS
            sub = np.asarray(X[:, [index[c] for c in names_used]])
            local = {c: k for k, c in enumerate(names_used)}
            ok = single_fwd_packet_no_fwd_iat(sub, local)
            single = sub[:, 0] <= 1
            for cid, cname in enumerate(names):
                m = y == cid
                rows.append({"dataset": dataset, "split": split, "class": cname,
                             "flows": int(m.sum()), "single_fwd_packet": int((m & single).sum()),
                             "violations": int((m & ~ok).sum())})
    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "genuine_flow_check.csv", index=False)
    tot = df.groupby(["dataset", "split"])[["flows", "single_fwd_packet", "violations"]].sum()
    lines = ["## Rule on genuine flows (all splits, all classes incl. Benign)", "",
             "| dataset | split | flows | single-forward-packet flows | violations |",
             "|---|---|---|---|---|"]
    for (dataset, split), r in tot.iterrows():
        lines.append(f"| {dataset} | {split} | {r.flows} | {r.single_fwd_packet} | "
                     f"{r.violations} |")
    return lines + ["", "Per-class counts: `results/genuine_flow_check.csv`.", ""]


def reproduction(results_dir: Path) -> list[str]:
    lines = ["## Rule-off arms reproduce the earlier runs", "",
             "| arm | same as | dataset | cells | flows | identical adversarial flow |",
             "|---|---|---|---|---|---|"]
    for arm, (other_dir, other_name) in SAME_AS.items():
        for ds_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
            cells = flows = same = 0
            for npz in sorted((ds_dir / "artifacts").glob(f"*__{arm}__seed*.npz")):
                other = (other_dir / ds_dir.name / "artifacts"
                         / npz.name.replace(f"__{arm}__", f"__{other_name}__"))
                if not other.exists():
                    continue
                with np.load(npz, allow_pickle=True) as a, np.load(other, allow_pickle=True) as b:
                    n = len(a["sample_id"])  # --limit-rows smoke runs use a prefix of the rows
                    if not np.array_equal(a["sample_id"], b["sample_id"][:n]):
                        raise AssertionError(f"{npz.name}: rows differ from {other}")
                    eq = np.all(a["adv_raw"] == b["adv_raw"][:n], axis=1)
                cells += 1
                flows += len(eq)
                same += int(eq.sum())
            if cells:
                lines.append(f"| {arm} | {other_dir.parent.name}/{other_name} | {ds_dir.name} | "
                             f"{cells} | {flows} | {same} |")
    return lines + [""]


def gap_audit(results_dir: Path) -> list[str]:
    """validator_v2 successes that the rule rejects, per arm (all seeds pooled)."""
    fields = ("valid_success", "extended_valid_success", f"rule_{RULE}_valid")
    lines = ["## validator_v2 successes the rule rejects (all seeds and classes pooled)", "",
             "| dataset | victim | budget | arm | validator_v2 successes | rejected by the rule | "
             "validator_v2 + rule successes |", "|---|---|---|---|---|---|---|"]
    rows = []
    for g in _groups(results_dir, results_dir, [c.name for c in CONDITIONS]):
        for cond in CONDITIONS:
            src = Source(results_dir, cond.name)
            parts = [pooled_rows(src, g.dataset, g.victim, g.budget_label, s, fields)
                     for s in sorted(_seeds(results_dir, g, cond.name))]
            parts = [p for p in parts if p is not None]
            if not parts:
                continue
            r = {k: np.concatenate([p[k] for p in parts]) for k in fields}
            vs = r["valid_success"].astype(bool)
            rej = int((vs & ~r[f"rule_{RULE}_valid"].astype(bool)).sum())
            rows.append({"dataset": g.dataset, "victim": g.victim, "budget": g.budget_label,
                         "arm": cond.name, "validator_v2_successes": int(vs.sum()),
                         "rejected_by_rule": rej,
                         "extended_successes": int(r["extended_valid_success"].sum())})
            lines.append(f"| {g.dataset} | {g.victim} | {g.budget_label} | {cond.name} | "
                         f"{int(vs.sum())} | {rej} | {int(r['extended_valid_success'].sum())} |")
    pd.DataFrame(rows).to_csv(results_dir / "gap_audit.csv", index=False)
    return lines + [""]


def analyze(results_dir: Path, _reference: Path) -> None:
    body = ["# V1 - Toggle-able validator rule: no forward IAT on single-forward-packet flows", "",
            "Primary outcome: **Valid ASR under validator_v2 + rule** (`extended_valid_success`). "
            "Mean over attack seeds 42/2024/2026 (seed range); paired McNemar rule-on vs rule-off "
            "at seed 42, Holm within each PrimAttack variant.", ""]
    summaries, tests = [], []
    for variant, off, on in FAMILIES:
        if not any((results_dir / d / "artifacts").glob(f"*__{on}__*") for d in DATASETS):
            continue
        summary, t = compare_to_reference(
            results_dir, results_dir, [on], baseline=off, outcome="extended_valid_success",
            secondary=("valid_success", "raw_success"))
        summaries.append(summary.assign(variant=variant))
        tests.append(t.assign(variant=variant))
        body += [f"## PrimAttack variant: capability-{variant} ({off} vs {on})", "",
                 summary_markdown(summary, t, outcome="extended_valid_success",
                                  outcome_label="Valid ASR (validator_v2 + rule)",
                                  secondary_labels={"valid_success": "Valid ASR (validator_v2)",
                                                    "raw_success": "Raw ASR"},
                                  baseline=off)]
    if summaries:
        pd.concat(summaries).to_csv(results_dir / "summary.csv", index=False)
        pd.concat(tests).to_csv(results_dir / "tests.csv", index=False)
    body += gap_audit(results_dir) + reproduction(results_dir) + genuine_flow_check(results_dir)
    (results_dir / "report.md").write_text("\n".join(body) + "\n", encoding="utf-8")
    print("\n".join(body))


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__,
                    add_arguments=add_arguments, select=select)
