"""Run the minimum paired statistical protocol from stored final-suite artifacts.

No attack is executed. Every loaded artifact is checked against its dataset's canonical
``selection.json`` before any statistic is computed. Outputs are written to
``FINAL_OUTPUTS/statistics``.

The canonical budget names are fixed by ``artifacts/primattack/budget_calibration.json``:
restricted=p25, intermediate=p50, maximum-evaluated=p75. A missing condition is never
silently replaced by another budget.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.paired_validity_gap import cochran_q, holm_adjust, mcnemar_test  # noqa: E402

FINAL = REPO_ROOT / "FINAL_OUTPUTS"
RUNS = FINAL / "runs"
OUT = FINAL / "statistics"
ALPHA = 0.05
REFERENCE_SEED = 42
EXPECTED_SEEDS = (42, 2024, 2026)
FILE_BUDGET = {
    "restricted": "p25",
    "intermediate": "p50",
    "maximum-evaluated": "p75",
}


@dataclass(frozen=True)
class Condition:
    key: str
    label: str
    stage: str
    method: str
    budget: str
    objective: str
    mode: str = "joint"
    primitive: bool = True

    def artifact(self, dataset: str, victim: str, source_class: str, seed: int) -> Path:
        if not self.primitive:
            name = f"{victim}__{source_class}__{self.method}__seed{seed}.npz"
        else:
            tag = self.method if self.mode == "joint" else f"{self.method}-{self.mode}"
            name = f"{victim}__{source_class}__{FILE_BUDGET[self.budget]}__{tag}__seed{seed}.npz"
        return RUNS / dataset / self.stage / "artifacts" / name


@dataclass(frozen=True)
class Outcome:
    ids: np.ndarray
    raw: np.ndarray
    valid: np.ndarray


class ArtifactStore:
    def __init__(self, datasets: dict[str, tuple[str, ...]], classes: tuple[str, ...]) -> None:
        self.datasets = datasets
        self.classes = classes
        self.canonical: dict[tuple[str, str, str], dict] = {}
        self.cache: dict[tuple[str, str, str, int], Outcome] = {}
        self.checked_files = 0
        self.checked_rows = 0
        self._load_canonical()

    def _load_canonical(self) -> None:
        for dataset, victims in self.datasets.items():
            path = RUNS / dataset / "baselines_untargeted" / "selection.json"
            selection = json.loads(path.read_text(encoding="utf-8"))
            for victim in victims:
                for source_class in self.classes:
                    entry = selection[victim][source_class]
                    ids = np.asarray(entry["sample_ids"], dtype="U128")
                    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
                        raise AssertionError(
                            f"{dataset}/{victim}/{source_class}: canonical IDs are empty or duplicated")
                    digest = hashlib.sha256("\n".join(ids.tolist()).encode("utf-8")).hexdigest()
                    if digest != entry["sha256_sample_ids"]:
                        raise AssertionError(
                            f"{dataset}/{victim}/{source_class}: canonical sample-ID hash mismatch")
                    self.canonical[(dataset, victim, source_class)] = {
                        "ids": ids,
                        "positions": np.asarray(entry["positional_idx"], dtype=np.int64),
                        "clean_hash": entry["clean_raw_sha256"],
                    }

    def condition_available(self, condition: Condition) -> tuple[bool, list[Path]]:
        missing = []
        for dataset, victims in self.datasets.items():
            for victim in victims:
                for source_class in self.classes:
                    for seed in EXPECTED_SEEDS:
                        path = condition.artifact(dataset, victim, source_class, seed)
                        if not path.exists():
                            missing.append(path)
        return not missing, missing

    def load(self, condition: Condition, dataset: str, victim: str, seed: int) -> Outcome:
        cache_key = (condition.key, dataset, victim, seed)
        if cache_key in self.cache:
            return self.cache[cache_key]

        ids_parts: list[np.ndarray] = []
        raw_parts: list[np.ndarray] = []
        valid_parts: list[np.ndarray] = []
        for source_class in self.classes:
            path = condition.artifact(dataset, victim, source_class, seed)
            if not path.exists():
                raise FileNotFoundError(f"missing stored attack artifact: {path.relative_to(REPO_ROOT)}")
            with np.load(path, allow_pickle=True) as artifact:
                canonical = self.canonical[(dataset, victim, source_class)]
                ids = np.asarray(artifact["sample_id"], dtype="U128")
                positions = np.asarray(artifact["positional_idx"], dtype=np.int64)
                raw = np.asarray(artifact["raw_success"], dtype=bool)
                valid = np.asarray(artifact["valid_success"], dtype=bool)
                where = path.relative_to(REPO_ROOT)

                if len(ids) != len(raw) or len(ids) != len(valid):
                    raise AssertionError(f"{where}: outcome lengths differ")
                if len(np.unique(ids)) != len(ids):
                    raise AssertionError(f"{where}: duplicate sample IDs")
                if not np.array_equal(ids, canonical["ids"]):
                    raise AssertionError(f"{where}: sample IDs differ from canonical selection")
                if not np.array_equal(positions, canonical["positions"]):
                    raise AssertionError(f"{where}: positional indices differ from canonical selection")
                if str(np.asarray(artifact["clean_raw_sha256"]).item()) != canonical["clean_hash"]:
                    raise AssertionError(f"{where}: clean-input hash differs from canonical selection")
                if int(np.asarray(artifact["seed"]).item()) != seed:
                    raise AssertionError(f"{where}: stored seed differs from expected seed {seed}")
                if str(np.asarray(artifact["objective"]).item()) != condition.objective:
                    raise AssertionError(f"{where}: stored objective differs from {condition.objective}")
                if str(np.asarray(artifact["method"]).item()) != condition.method:
                    raise AssertionError(f"{where}: stored optimizer differs from {condition.method}")
                if condition.primitive:
                    if str(np.asarray(artifact["budget"]).item()) != condition.budget:
                        raise AssertionError(f"{where}: stored budget differs from {condition.budget}")
                    if str(np.asarray(artifact["primitive_mode"]).item()) != condition.mode:
                        raise AssertionError(f"{where}: stored primitive mode differs from {condition.mode}")
                if np.any(valid & ~raw):
                    raise AssertionError(f"{where}: valid_success is not a subset of raw_success")

            qualified_ids = np.char.add(f"{source_class}\x1f", ids)
            ids_parts.append(qualified_ids)
            raw_parts.append(raw)
            valid_parts.append(valid)
            self.checked_files += 1
            self.checked_rows += len(ids)

        outcome = Outcome(
            ids=np.concatenate(ids_parts),
            raw=np.concatenate(raw_parts),
            valid=np.concatenate(valid_parts),
        )
        self.cache[cache_key] = outcome
        return outcome

    def preload_and_verify(self, conditions: Iterable[Condition]) -> None:
        """Verify canonical IDs first; no statistical function is called before this completes."""
        conditions = tuple(conditions)
        for condition in conditions:
            for dataset, victims in self.datasets.items():
                for victim in victims:
                    for seed in EXPECTED_SEEDS:
                        self.load(condition, dataset, victim, seed)

        for dataset, victims in self.datasets.items():
            for victim in victims:
                for seed in EXPECTED_SEEDS:
                    reference = self.load(conditions[0], dataset, victim, seed).ids
                    for condition in conditions[1:]:
                        candidate = self.load(condition, dataset, victim, seed).ids
                        if not np.array_equal(reference, candidate):
                            raise AssertionError(
                                f"pairing broken for {dataset}/{victim}/seed{seed}: "
                                f"{conditions[0].label} vs {condition.label}")


def load_design() -> tuple[dict[str, tuple[str, ...]], tuple[str, ...], str]:
    config = json.loads((RUNS / "final_suite_config.json").read_text(encoding="utf-8"))
    datasets = {
        dataset: tuple(v.strip() for v in spec["victims"].split(",") if v.strip())
        for dataset, spec in config["datasets"].items()
    }
    classes = tuple(c.strip() for c in config["classes"].split(",") if c.strip())
    selection = json.loads((RUNS / "optimizer_selection.json").read_text(encoding="utf-8"))
    return datasets, classes, str(selection["selected"])


def condition_mean_sd(store: ArtifactStore, condition: Condition, dataset: str,
                      victim: str, field: str) -> tuple[float, float]:
    values = np.asarray([
        getattr(store.load(condition, dataset, victim, seed), field).mean()
        for seed in EXPECTED_SEEDS
    ], dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def counts_and_rates(vectors: list[np.ndarray]) -> tuple[str, str]:
    counts = [int(v.sum()) for v in vectors]
    rates = [float(v.mean()) for v in vectors]
    return ";".join(map(str, counts)), ";".join(f"{rate:.8f}" for rate in rates)


def base_row(analysis: str, test: str, dataset: str, victim: str, objective: str,
             optimizer: str, conditions: list[str], vectors: list[np.ndarray],
             outcome: str) -> dict:
    counts, rates = counts_and_rates(vectors)
    return {
        "analysis": analysis,
        "test": test,
        "dataset": dataset,
        "victim": victim,
        "objective": objective,
        "optimizer": optimizer,
        "conditions": ";".join(conditions),
        "outcome": outcome,
        "reference_seed": REFERENCE_SEED,
        "n_paired": len(vectors[0]),
        "success_counts": counts,
        "success_rates": rates,
        "test_statistic": "",
        "df": "",
        "p_value": "",
        "p_holm": "",
        "significant_alpha_0_05": "",
        "a_succeeds_b_fails": "",
        "a_fails_b_succeeds": "",
        "status": "performed",
    }


def mcnemar_row(analysis: str, dataset: str, victim: str, objective: str, optimizer: str,
                 label_a: str, label_b: str, a: np.ndarray, b: np.ndarray,
                 outcome: str) -> dict:
    if len(a) != len(b):
        raise AssertionError(f"{analysis}/{dataset}/{victim}: paired lengths differ")
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    result = mcnemar_test(a_only, b_only)
    row = base_row(analysis, "McNemar", dataset, victim, objective, optimizer,
                   [label_a, label_b], [a, b], outcome)
    row.update({
        "test_statistic": ("" if result["test_statistic"] is None
                           else float(result["test_statistic"])),
        "p_value": float(result["p_value"]),
        "significant_alpha_0_05": "yes" if result["p_value"] < ALPHA else "no",
        "a_succeeds_b_fails": a_only,
        "a_fails_b_succeeds": b_only,
        "status": f"performed ({result['test_variant']})",
    })
    return row


def omnibus_family(analysis: str, dataset: str, victim: str, objective: str, optimizer: str,
                    conditions: list[Condition], vectors: list[np.ndarray],
                    planned_pairs: tuple[tuple[int, int], ...]) -> list[dict]:
    result = cochran_q(np.column_stack(vectors))
    q_row = base_row(analysis, "Cochran's Q", dataset, victim, objective, optimizer,
                     [c.label for c in conditions], vectors, "valid_success")
    q_row.update({
        "test_statistic": float(result["Q"]),
        "df": int(result["df"]),
        "p_value": float(result["p"]),
        "significant_alpha_0_05": "yes" if result["p"] < ALPHA else "no",
    })
    rows = [q_row]
    if result["p"] >= ALPHA:
        for a_idx, b_idx in planned_pairs:
            row = base_row(analysis, "McNemar", dataset, victim, objective, optimizer,
                           [conditions[a_idx].label, conditions[b_idx].label],
                           [vectors[a_idx], vectors[b_idx]], "valid_success")
            row["status"] = "not performed: omnibus Cochran's Q not significant"
            rows.append(row)
        return rows

    pairs = [
        mcnemar_row(analysis, dataset, victim, objective, optimizer,
                     conditions[a_idx].label, conditions[b_idx].label,
                     vectors[a_idx], vectors[b_idx], "valid_success")
        for a_idx, b_idx in planned_pairs
    ]
    adjusted = holm_adjust([float(row["p_value"]) for row in pairs])
    for row, adjusted_p in zip(pairs, adjusted):
        row["p_holm"] = float(adjusted_p)
        row["significant_alpha_0_05"] = "yes" if adjusted_p < ALPHA else "no"
    rows.extend(pairs)
    return rows


def fmt_p(value: object) -> str:
    if value == "" or value is None:
        return "—"
    number = float(value)
    if number == 0:
        return "<1×10⁻³⁰⁰"
    return f"{number:.3g}"


def fmt_mean_sd(mean: float, sd: float) -> str:
    return f"{100 * mean:.2f}% ± {100 * sd:.2f}%"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |",
                "|" + "|".join("---" for _ in headers) + "|"]
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def write_markdown(path: Path, rows: list[dict], store: ArtifactStore,
                   datasets: dict[str, tuple[str, ...]], mode_conditions: list[Condition],
                   available_budget: list[Condition], raw_valid_conditions: list[Condition],
                   missing_budget: dict[str, list[Path]], optimizer: str) -> None:
    mode_summary = []
    for dataset, victims in datasets.items():
        for victim in victims:
            for condition in mode_conditions:
                mean, sd = condition_mean_sd(store, condition, dataset, victim, "valid")
                ref = store.load(condition, dataset, victim, REFERENCE_SEED).valid
                mode_summary.append([dataset, victim, condition.label, len(ref), int(ref.sum()),
                                     f"{100 * ref.mean():.2f}%", fmt_mean_sd(mean, sd)])

    budget_summary = []
    for dataset, victims in datasets.items():
        for victim in victims:
            for condition in available_budget:
                mean, sd = condition_mean_sd(store, condition, dataset, victim, "valid")
                ref = store.load(condition, dataset, victim, REFERENCE_SEED).valid
                budget_summary.append([dataset, victim, condition.label, len(ref), int(ref.sum()),
                                       f"{100 * ref.mean():.2f}%", fmt_mean_sd(mean, sd)])

    q_rows = [row for row in rows if row["test"] == "Cochran's Q"]
    pair_rows = [row for row in rows if row["test"] == "McNemar" and row["status"].startswith("performed")]

    raw_valid_summary = []
    for dataset, victims in datasets.items():
        for victim in victims:
            for condition in raw_valid_conditions:
                raw_mean, raw_sd = condition_mean_sd(store, condition, dataset, victim, "raw")
                valid_mean, valid_sd = condition_mean_sd(store, condition, dataset, victim, "valid")
                ref = store.load(condition, dataset, victim, REFERENCE_SEED)
                raw_valid_summary.append([
                    dataset, victim, condition.label, len(ref.raw),
                    f"{int(ref.raw.sum())} ({100 * ref.raw.mean():.2f}%)",
                    f"{int(ref.valid.sum())} ({100 * ref.valid.mean():.2f}%)",
                    fmt_mean_sd(raw_mean, raw_sd), fmt_mean_sd(valid_mean, valid_sd),
                    f"{100 * (raw_mean - valid_mean):.2f} pp",
                ])
    test_table = []
    for row in q_rows + pair_rows:
        statistic = "—" if row["test_statistic"] == "" else f"{float(row['test_statistic']):.3f}"
        test_table.append([
            row["analysis"], row["dataset"], row["victim"], row["test"],
            row["conditions"].replace(";", " vs " if row["test"] == "McNemar" else " / "),
            row["n_paired"], row["success_counts"].replace(";", " / "),
            " / ".join(f"{100 * float(x):.2f}%" for x in row["success_rates"].split(";")),
            statistic, fmt_p(row["p_value"]), fmt_p(row["p_holm"]),
            row["significant_alpha_0_05"],
            ("—" if row["test"] != "McNemar" else
             f"{row['a_succeeds_b_fails']} / {row['a_fails_b_succeeds']}")
        ])

    raw_rows = [row for row in pair_rows if row["analysis"] == "Raw-vs-Valid validity gap"]
    significant_gaps = sum(row["significant_alpha_0_05"] == "yes" for row in raw_rows)
    no_gap = sum(int(row["a_succeeds_b_fails"]) == 0 for row in raw_rows)

    if missing_budget:
        missing_count = sum(len(paths) for paths in missing_budget.values())
        budget_status = (
            f"**Not computed.** The required restricted (train-p25) condition has no stored "
            f"per-sample artifacts ({missing_count} expected files are absent). The stored final "
            f"suite contains intermediate (p50), maximum-evaluated (p75), and an unbounded "
            f"sensitivity condition. Unbounded was not substituted for restricted because these "
            f"are different perturbation budgets. No attack was rerun."
        )
    else:
        budget_status = (
            "Computed separately for each dataset and victim using restricted, intermediate, and "
            "maximum-evaluated paired `valid_success` outcomes."
        )

    lines = [
        "# Minimum paired statistical tests for the final experiments",
        "",
        "## Protocol",
        "",
        "- Paired unit: one canonical clean-correct source flow; four source classes are pooled "
        "within each victim (800 per class, N = 3,200). Datasets and victims are never pooled.",
        "- Inference uses reference attack seed 42 only. Seeds 42, 2024, and 2026 are used only "
        "for descriptive mean ± sample SD of rates; seed-level means are not inferential units.",
        "- Primary inferential outcome: `valid_success`. The only exception is the dedicated "
        "paired `raw_success` versus `valid_success` comparison.",
        "- α = 0.05. Cochran’s Q gates the two planned McNemar comparisons in each three-condition "
        "family. Holm correction is applied only across those two comparisons.",
        "- McNemar uses an exact two-sided binomial test for fewer than 25 discordant pairs; "
        "otherwise it uses the continuity-corrected χ² statistic.",
        f"- Selected canonical PrimAttack optimizer: `{optimizer}`.",
        "",
        "## Pairing and provenance audit",
        "",
        f"Before any statistic was computed, {store.checked_files} stored NPZ artifacts "
        f"({store.checked_rows:,} per-sample rows) were checked against the canonical "
        "`selection.json` records. Checks covered exact sample-ID order, duplicate IDs, positional "
        "indices, clean-input SHA-256, seed, objective, attack method and, where applicable, budget "
        "and primitive mode, outcome lengths, and `valid_success ⊆ raw_success`. Every "
        "multi-condition family was checked for identical paired IDs for all three seeds. Any "
        "mismatch aborts before output writing.",
        "",
        "## 1. Primitive-mode ablation",
        "",
        "This test asks whether valid untargeted success differs across timing-only, padding-only, "
        "and joint primitive control at the maximum-evaluated budget. A significant Q establishes "
        "a condition effect; the gated joint-vs-timing and joint-vs-padding McNemar tests identify "
        "which planned contrasts differ.",
        "",
        markdown_table(["Dataset", "Victim", "Mode", "N", "Successes (seed 42)",
                        "Rate (seed 42)", "Mean ± SD (3 seeds)"], mode_summary),
        "",
        "## 2. Budget sensitivity",
        "",
        budget_status,
        "",
    ]
    if budget_summary:
        lines.extend([
            markdown_table(["Dataset", "Victim", "Budget", "N", "Successes (seed 42)",
                            "Rate (seed 42)", "Mean ± SD (3 seeds)"], budget_summary),
            "",
        ])
    lines.extend([
        "## 3. Raw-vs-Valid validity gap",
        "",
        "For each canonical headline attack condition (the five inferential Exp-A methods and the "
        "three maximum-evaluated targeted PrimAttack optimizers), McNemar compares `raw_success` "
        "with `valid_success` on the same seed-42 flows. Because valid success is a subset of raw "
        "success, the directional count `Raw succeeds / Valid fails` is the number of apparent "
        "successes removed by validity enforcement; the reverse discordance must be zero.",
        "",
        markdown_table(["Dataset", "Victim", "Condition", "N", "Raw seed 42",
                        "Valid seed 42", "Raw mean ± SD", "Valid mean ± SD",
                        "Mean validity gap"], raw_valid_summary),
        "",
        "## Inferential results",
        "",
        markdown_table(["Analysis", "Dataset", "Victim", "Test", "Conditions", "N",
                        "Success counts", "Success rates", "Statistic", "p", "Holm p",
                        "Significant", "A-only / B-only"], test_table),
        "",
        "The machine-readable version is `statistical_tests.csv`. Rows whose planned McNemar tests "
        "were gated off by a non-significant Cochran’s Q remain in the CSV with status "
        "`not performed`; they are not included in the table above.",
        "",
        "## Interpretation: statistical versus practical significance",
        "",
        f"Among {len(raw_rows)} performed Raw-vs-Valid comparisons, {significant_gaps} show a "
        f"statistically significant loss after validity enforcement and {no_gap} have no seed-42 "
        "raw-success-but-invalid discordance.",
        "",
        "Statistical significance addresses whether the paired outcome difference is unlikely under "
        "the null of equal marginal success probabilities. It does not measure the size or thesis "
        "importance of that difference. Practical significance must be read from the success-rate "
        "difference and directional discordant count. With N = 3,200, a small rate difference can "
        "be statistically significant; conversely, a non-significant result does not establish "
        "equivalence. Mean ± SD across attack seeds describes run-to-run variability only and is "
        "not uncertainty for the paired hypothesis tests.",
        "",
        "The primitive-mode inferential analysis was not part of the originally locked final-suite "
        "plan, where that ablation was descriptive. It must therefore be identified as an added "
        "post-run analysis in the thesis rather than described as pre-registered.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    datasets, classes, optimizer = load_design()
    store = ArtifactStore(datasets, classes)

    mode_conditions = [
        Condition("mode_timing", "timing-only", "primattack_untargeted_modes", optimizer,
                  "maximum-evaluated", "untargeted", "timing-only"),
        Condition("mode_padding", "padding-only", "primattack_untargeted_modes", optimizer,
                  "maximum-evaluated", "untargeted", "padding-only"),
        Condition("mode_joint", "joint", "primattack_untargeted", optimizer,
                  "maximum-evaluated", "untargeted", "joint"),
    ]
    budget_conditions = [
        Condition("budget_restricted", "restricted", "primattack_targeted_budgets", optimizer,
                  "restricted", "targeted"),
        Condition("budget_intermediate", "intermediate", "primattack_targeted_budgets", optimizer,
                  "intermediate", "targeted"),
        Condition("budget_maximum", "maximum-evaluated", "primattack_targeted_optimizers", optimizer,
                  "maximum-evaluated", "targeted"),
    ]
    raw_valid_conditions = [
        Condition("raw_prim_untargeted", f"PrimAttack-{optimizer} untargeted maximum-evaluated",
                  "primattack_untargeted", optimizer, "maximum-evaluated", "untargeted"),
        Condition("raw_pgd", "Input PGD untargeted", "baselines_untargeted",
                  "pgd_untargeted", "native", "untargeted", primitive=False),
        Condition("raw_cw", "Input C&W untargeted", "baselines_untargeted",
                  "cw_untargeted", "native", "untargeted", primitive=False),
        Condition("raw_capgd", "CAPGD-PrimSupport untargeted", "baselines_untargeted",
                  "capgd_prim_support", "native", "untargeted", primitive=False),
        Condition("raw_cpgd", "C-PGD-PrimSupport untargeted", "baselines_untargeted",
                  "cpgd_prim_support", "native", "untargeted", primitive=False),
        *[
            Condition(f"raw_prim_{method}_targeted",
                      f"PrimAttack-{method} targeted maximum-evaluated",
                      "primattack_targeted_optimizers", method, "maximum-evaluated", "targeted")
            for method in ("hybrid", "pgd", "cw")
        ],
    ]

    availability = {condition.key: store.condition_available(condition) for condition in budget_conditions}
    available_budget = [condition for condition in budget_conditions if availability[condition.key][0]]
    missing_budget = {
        condition.label: availability[condition.key][1]
        for condition in budget_conditions if not availability[condition.key][0]
    }
    available_conditions = mode_conditions + available_budget + raw_valid_conditions

    # Required ordering: complete all canonical-ID and cross-condition pairing checks before tests.
    store.preload_and_verify(available_conditions)

    rows: list[dict] = []
    for dataset, victims in datasets.items():
        for victim in victims:
            mode_vectors = [store.load(c, dataset, victim, REFERENCE_SEED).valid
                            for c in mode_conditions]
            # Pair order: joint vs timing-only, joint vs padding-only.
            rows.extend(omnibus_family(
                "Primitive-mode ablation", dataset, victim, "untargeted", optimizer,
                mode_conditions, mode_vectors, ((2, 0), (2, 1))))

            if not missing_budget:
                budget_vectors = [store.load(c, dataset, victim, REFERENCE_SEED).valid
                                  for c in budget_conditions]
                # Pair order: restricted vs intermediate, intermediate vs maximum-evaluated.
                rows.extend(omnibus_family(
                    "Budget sensitivity", dataset, victim, "targeted", optimizer,
                    budget_conditions, budget_vectors, ((0, 1), (1, 2))))

            for condition in raw_valid_conditions:
                outcome = store.load(condition, dataset, victim, REFERENCE_SEED)
                row = mcnemar_row(
                    "Raw-vs-Valid validity gap", dataset, victim, condition.objective, condition.method,
                    f"{condition.label}:raw_success", f"{condition.label}:valid_success",
                    outcome.raw, outcome.valid, "raw_success vs valid_success")
                if int(row["a_fails_b_succeeds"]) != 0:
                    raise AssertionError(
                        f"{dataset}/{victim}/{condition.label}: valid success exceeds raw success")
                rows.append(row)

    fields = [
        "analysis", "test", "dataset", "victim", "objective", "optimizer", "conditions",
        "outcome", "reference_seed", "n_paired", "success_counts", "success_rates",
        "test_statistic", "df", "p_value", "p_holm", "significant_alpha_0_05",
        "a_succeeds_b_fails", "a_fails_b_succeeds", "status",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "statistical_tests.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(OUT / "statistical_summary.md", rows, store, datasets, mode_conditions,
                   available_budget, raw_valid_conditions, missing_budget, optimizer)

    performed = sum(row["status"].startswith("performed") for row in rows)
    print(json.dumps({
        "output_dir": str(OUT),
        "performed_tests": performed,
        "csv_rows": len(rows),
        "paired_artifacts_checked": store.checked_files,
        "paired_rows_checked": store.checked_rows,
        "budget_sensitivity_computed": not missing_budget,
        "missing_budget_conditions": {
            name: len(paths) for name, paths in missing_budget.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
