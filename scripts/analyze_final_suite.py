"""Build every final table, statistic, figure and report from ``FINAL_OUTPUTS/runs``.

Reads only the per-sample artifacts written by ``scripts/run_final_suite.py`` (no attack is
re-run) and writes, per experiment directory under ``FINAL_OUTPUTS``:

* ``per_sample.parquet``     one row per (dataset, victim, class, sample, seed, condition)
* ``seed_level.csv``         per-seed rates (victim-pooled and per-class rows)
* ``table_level.csv``        mean / SD over seeds 42, 2024, 2026 plus every seed value
* ``statistical_tests.csv``  Cochran's Q / McNemar / Holm rows (planned tests only)
* ``<report>.md`` + ``plots/*.png``

Fail-loud audit before any aggregation: every expected cell exists; sample IDs match the
canonical selection exactly (order, no duplicates); the clean-input hash, labels, clean
predictions and victim checkpoint hash are identical across conditions; seeds are exactly
{42, 2024, 2026}; denominators are equal; stored validator verdicts are recomputed from the
stored final adversarial flows; raw/valid success flags are recomputed from the stored
predictions.

    python scripts/analyze_final_suite.py [--final]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from comparisons.capgd_cicids2017 import fit_train_minmax  # noqa: E402
from datasets import get_adapter  # noqa: E402
from evaluation.paired_validity_gap import cochran_q, holm_adjust, mcnemar_test  # noqa: E402
from run_final_suite import CLASSES as SUITE_CLASSES, DATASETS as SUITE_DATASETS  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402
from validation import load_validator  # noqa: E402
from validation.attack_interface import structural_masks  # noqa: E402

FINAL = REPO_ROOT / "FINAL_OUTPUTS"
RUNS = FINAL / "runs"
SEEDS = (42, 2024, 2026)
REF_SEED = 42
ALPHA = 0.05
CLASSES = tuple(SUITE_CLASSES.split(","))
DATASETS = {d: tuple(spec["victims"].split(",")) for d, spec in SUITE_DATASETS.items()}
DS_LABEL = {"cicids2017_distrinet": "CICIDS2017", "cicids2018_distrinet": "CICIDS2018"}
BASELINE_LABEL = {"pgd_untargeted": "PGD", "cw_untargeted": "C&W",
                  "capgd_prim_support": "CAPGD-PrimSupport",
                  "cpgd_prim_support": "C-PGD-PrimSupport"}
OPT_LABEL = {"hybrid": "Hybrid Search", "pgd": "Prim-PGD", "cw": "Prim-C&W"}
FILE_BUDGET = {"p50": "p50", "p75": "p75", "unbounded": "unb"}
EXP_DIRS = {
    "A": ("A_primary_baseline_comparison", "primary_baseline_comparison.md"),
    "B": ("B_optimizer_selection", "primattack_optimizer_selection.md"),
    "C": ("C_budget_sensitivity", "primattack_budget_sensitivity.md"),
    "D": ("D_objective_sensitivity", "objective_sensitivity.md"),
    "E": ("E_paired_validity_gap", "paired_validity_gap_analysis.md"),
    "F": ("F_validator_evaluation", "validator_evaluation.md"),
}
BASELINE_NATIVE_BUDGET = {
    "pgd_untargeted": "L∞ ε=0.5 (RobustScaler space), 79 features",
    "cw_untargeted": "L2 penalty (unbounded), 79 features",
    "capgd_prim_support": "L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask",
    "cpgd_prim_support": "L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask",
}


# ----------------------------------------------------------------------------- conditions
@dataclass(frozen=True)
class Cond:
    key: str
    stage: str
    method: str
    budget: str
    objective: str
    label: str

    @property
    def prim(self) -> bool:
        return self.stage.startswith("primattack")

    def npz(self, dataset: str, victim: str, cls: str, seed: int) -> Path:
        art = RUNS / dataset / self.stage / "artifacts"
        if self.prim:
            return art / f"{victim}__{cls}__{FILE_BUDGET[self.budget]}__{self.method}__seed{seed}.npz"
        return art / f"{victim}__{cls}__{self.method}__seed{seed}.npz"


def base_cond(name: str) -> Cond:
    return Cond(f"{name}", "baselines_untargeted", name, "native", "untargeted",
                BASELINE_LABEL[name])


def prim_cond(stage: str, opt: str, budget: str, objective: str, label: str | None = None) -> Cond:
    return Cond(f"prim_{opt}_{budget}_{objective}", stage, opt, budget, objective,
                label or f"{OPT_LABEL[opt]} ({budget}, {objective})")


# ----------------------------------------------------------------------------- loading
class Store:
    """Loads + audits per-sample artifacts once per condition."""

    def __init__(self, recheck_predictions: bool, device: str) -> None:
        self.recheck_predictions = recheck_predictions
        self.device = device
        self.frames: dict[str, pd.DataFrame] = {}
        self.canonical: dict[str, dict] = {}
        self.victim_meta: dict[str, dict] = {}
        self.clean_raw: dict[str, np.ndarray] = {}
        self.scale: dict[str, np.ndarray] = {}
        self.center: dict[str, np.ndarray] = {}
        self.minmax_span: dict[str, np.ndarray] = {}
        self.victims: dict[tuple[str, str], torch.nn.Module] = {}
        self.audit = {"npz_files": 0, "rows": 0, "validator_rechecked_rows": 0,
                      "prediction_rechecked_rows": 0, "prediction_mismatches": 0,
                      "raw_success_recomputed_rows": 0}
        for dataset in DATASETS:
            self._load_canonical(dataset)

    # -- canonical selection -----------------------------------------------------------
    def _load_canonical(self, dataset: str) -> None:
        base = RUNS / dataset / "baselines_untargeted"
        sel = json.loads((base / "selection.json").read_text(encoding="utf-8"))
        cfg = json.loads((base / "config.json").read_text(encoding="utf-8"))
        adapter = get_adapter(dataset)
        processed = adapter._processed
        meta = pd.read_parquet(processed / "test.parquet", columns=["sample_id"])
        ids_all = meta["sample_id"].astype(str).to_numpy(dtype="U128")
        raw_all = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
        transform = adapter.feature_transform()
        self.scale[dataset] = np.asarray(transform.scale, dtype=np.float64)
        self.center[dataset] = np.asarray(transform.center, dtype=np.float64)
        low, high = fit_train_minmax(processed / "X_train_pristine.npy")
        span = high - low
        self.minmax_span[dataset] = np.where(span > 0, span, 1.0)
        mapping = adapter.class_mapping()
        canon = {}
        for victim in DATASETS[dataset]:
            if victim not in sel:
                raise KeyError(f"{dataset}: victim {victim} missing from canonical selection")
            for cls in CLASSES:
                entry = sel[victim][cls]
                idx = np.asarray(entry["positional_idx"], dtype=np.int64)
                sids = np.asarray(entry["sample_ids"], dtype="U128")
                if len(np.unique(sids)) != len(sids):
                    raise AssertionError(f"{dataset}/{victim}/{cls}: duplicate sample IDs")
                if not np.array_equal(ids_all[idx], sids):
                    raise AssertionError(f"{dataset}/{victim}/{cls}: positional_idx does not map "
                                         "to the recorded sample IDs in test.parquet")
                sha_ids = hashlib.sha256("\n".join(sids.tolist()).encode("utf-8")).hexdigest()
                if sha_ids != entry["sha256_sample_ids"]:
                    raise AssertionError(f"{dataset}/{victim}/{cls}: sample-ID hash mismatch")
                raw = np.ascontiguousarray(np.asarray(raw_all[idx], dtype=np.float32))
                if hashlib.sha256(raw.tobytes()).hexdigest() != entry["clean_raw_sha256"]:
                    raise AssertionError(f"{dataset}/{victim}/{cls}: clean-input hash mismatch")
                if entry["class_id"] != int(mapping.name_to_id[cls]):
                    raise AssertionError(f"{dataset}/{victim}/{cls}: class id mismatch")
                self.clean_raw[f"{dataset}|{victim}|{cls}"] = raw
                canon[(victim, cls)] = {"sample_ids": sids, "positional_idx": idx,
                                        "clean_raw_sha256": entry["clean_raw_sha256"],
                                        "class_id": int(entry["class_id"]),
                                        "n_eligible_total": int(entry["n_eligible_total"]),
                                        "n_class_test": int(entry["n_class_test"])}
        self.canonical[dataset] = canon
        self.victim_meta[dataset] = cfg["victims"]

    def victim(self, dataset: str, victim: str):
        key = (dataset, victim)
        if key not in self.victims:
            meta = self.victim_meta[dataset][victim]
            self.victims[key] = load_category_victim(
                Path(meta["checkpoint"]), adapter=get_adapter(dataset),
                expected_model_type=meta["arch"], device=self.device)
        return self.victims[key]

    # -- per-condition frame -----------------------------------------------------------
    def frame(self, cond: Cond) -> pd.DataFrame:
        """All rows of one condition; artifacts are read + audited once per condition key, and
        the ``method`` column carries this experiment's label for ``cond``."""
        if cond.key not in self.frames:
            parts = []
            for dataset, victims in DATASETS.items():
                for victim in victims:
                    for cls in CLASSES:
                        for seed in SEEDS:
                            path = cond.npz(dataset, victim, cls, seed)
                            if not path.exists():
                                raise FileNotFoundError(
                                    "incomplete experiment: missing cell "
                                    f"{path.relative_to(REPO_ROOT)}")
                            parts.append(self._read(path, cond, dataset, victim, cls, seed))
            self.frames[cond.key] = pd.concat(parts, ignore_index=True)
        frame = self.frames[cond.key]
        if frame["method"].iloc[0] != cond.label:
            frame = frame.assign(method=cond.label)
        return frame

    def _read(self, path: Path, cond: Cond, dataset: str, victim: str, cls: str,
              seed: int) -> pd.DataFrame:
        with np.load(path, allow_pickle=True) as d:
            data = {k: d[k] for k in d.files}
        canon = self.canonical[dataset][(victim, cls)]
        cid = canon["class_id"]
        where = f"{path.relative_to(RUNS)}"
        sids = data["sample_id"].astype("U128")
        n = len(sids)
        if not np.array_equal(sids, canon["sample_ids"]):
            raise AssertionError(f"{where}: sample IDs differ from the canonical list")
        if not np.array_equal(np.asarray(data["positional_idx"], np.int64), canon["positional_idx"]):
            raise AssertionError(f"{where}: positional indices differ from the canonical list")
        if str(data["clean_raw_sha256"]) != canon["clean_raw_sha256"]:
            raise AssertionError(f"{where}: clean-input hash differs")
        if str(data["checkpoint_sha256"]) != self.victim_meta[dataset][victim]["checkpoint_sha256"]:
            raise AssertionError(f"{where}: victim checkpoint hash differs")
        if int(np.asarray(data["seed"])) != seed:
            raise AssertionError(f"{where}: stored seed differs")
        true = np.asarray(data["true_class"], np.int64)
        clean_pred = np.asarray(data["clean_pred"], np.int64)
        adv_pred = np.asarray(data["adv_pred"], np.int64)
        if not (np.all(true == cid) and np.all(clean_pred == cid)):
            raise AssertionError(f"{where}: labels / clean predictions not all the source class")
        stored_objective = str(data["objective"])
        expected_objective = cond.objective if cond.prim else "untargeted"
        if stored_objective != expected_objective:
            raise AssertionError(f"{where}: objective {stored_objective} != {expected_objective}")
        raw_success = (adv_pred == 0) if cond.objective == "targeted" else (adv_pred != cid)
        if not np.array_equal(raw_success, np.asarray(data["raw_success"], bool)):
            raise AssertionError(f"{where}: stored raw_success disagrees with adv_pred")
        validator_pass = np.asarray(data["validator_pass"], bool)
        adv_raw = np.asarray(data["adv_raw"], np.float32)
        layers = structural_masks(adv_raw, dataset=dataset)
        recheck = layers["hybrid_valid"]
        if not np.array_equal(recheck, validator_pass):
            raise AssertionError(f"{where}: validator_v2 recheck on the stored final flow differs")
        valid_success = raw_success & validator_pass
        if not np.array_equal(valid_success, np.asarray(data["valid_success"], bool)):
            raise AssertionError(f"{where}: stored valid_success != raw_success AND validator_pass")
        outside = np.asarray(data["n_modified_outside_primattack_mask"], np.int64)
        if (cond.prim or cond.method in ("capgd_prim_support", "cpgd_prim_support")) and outside.any():
            raise AssertionError(f"{where}: features changed outside the PrimAttack support")
        if self.recheck_predictions:
            victim_m = self.victim(dataset, victim)
            scale = torch.tensor(self.scale[dataset], dtype=torch.float32, device=self.device)
            center = torch.tensor(self.center[dataset], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                pred = victim_m((torch.as_tensor(adv_raw, device=self.device) - center)
                                / scale).argmax(1).cpu().numpy()
            self.audit["prediction_rechecked_rows"] += n
            self.audit["prediction_mismatches"] += int((pred != adv_pred).sum())
        self.audit["npz_files"] += 1
        self.audit["rows"] += n
        self.audit["validator_rechecked_rows"] += n
        self.audit["raw_success_recomputed_rows"] += n

        clean = self.clean_raw[f"{dataset}|{victim}|{cls}"].astype(np.float64)
        diff = adv_raw.astype(np.float64) - clean
        l2 = np.linalg.norm(diff / self.scale[dataset], axis=1)
        linf = np.abs(diff / self.scale[dataset]).max(axis=1)
        l2_minmax = np.linalg.norm(diff / self.minmax_span[dataset], axis=1)
        nan = np.full(n, np.nan)
        if cond.prim:
            evals = np.asarray(data["total_evaluations"], np.float64)
            accounting = "exact per flow: realized + surrogate victim forwards"
            runtime = float(np.asarray(data["elapsed_seconds"]))
            extra = {
                "optimizer": cond.method, "primitive_mode": "joint",
                "primitive_p": np.asarray(data["p"], np.float64),
                "primitive_delay": np.asarray(data["delay"], np.float64),
                "primitive_shape": np.asarray(data["shape"], np.float64),
                "primitive_p_hi": np.asarray(data["p_hi"], np.float64),
                "primitive_delay_hi": np.asarray(data["delay_hi"], np.float64),
                "primitive_normalized_cost": np.asarray(data["normalized_cost"], np.float64),
                "objective_margin": np.asarray(data["objective_margin"], np.float64),
                "candidate_source": data["candidate_source"].astype(str),
                "failure_category": data["failure"].astype(str),
                "backward_evaluations": np.asarray(data["backward_evaluations"], np.float64),
                "first_success_evaluation": np.asarray(data["first_success_evaluation"], np.float64),
                "iterations": np.full(n, int(np.asarray(data["iterations"]))),
                "restarts": np.full(n, int(np.asarray(data["restarts"]))),
                "attack_parameters": np.full(n, json.dumps(self._prim_cfg(dataset, cond))),
            }
            allowed = np.full(n, 23)
        else:
            per_row = np.asarray(data["model_evaluations"], np.float64)
            forward_rows = int(np.asarray(data["victim_forward_rows"]))
            if (per_row < 0).any():
                evals = np.full(n, forward_rows / n)
                accounting = "batch mean per flow: victim forward hook (rows / flows)"
            else:
                evals = per_row
                accounting = "exact per flow"
            runtime = float(np.asarray(data["runtime_seconds"]))
            extra = {
                "optimizer": "n/a", "primitive_mode": "n/a (direct feature space)",
                "primitive_p": nan, "primitive_delay": nan, "primitive_shape": nan,
                "primitive_p_hi": nan, "primitive_delay_hi": nan,
                "primitive_normalized_cost": nan, "objective_margin": nan,
                "candidate_source": np.full(n, "n/a"), "failure_category": np.full(n, "n/a"),
                "backward_evaluations": nan, "first_success_evaluation": nan,
                "iterations": np.asarray(data["iterations"], np.int64),
                "restarts": np.full(n, -1),
                "attack_parameters": np.full(n, str(data["attack_parameters"])),
            }
            allowed = np.asarray(data["n_allowed_primattack_support_features"], np.int64)
        return pd.DataFrame({
            "dataset": dataset, "victim": victim,
            "victim_arch": self.victim_meta[dataset][victim]["arch"],
            "checkpoint_sha256": self.victim_meta[dataset][victim]["checkpoint_sha256"],
            "source_class": cls, "true_class": true, "sample_id": sids,
            "positional_idx": np.asarray(data["positional_idx"], np.int64), "seed": seed,
            "run_stage": cond.stage, "condition": cond.key, "method": cond.label,
            "method_id": cond.method, "budget": cond.budget, "objective": cond.objective,
            "clean_pred": clean_pred, "adv_pred": adv_pred, "raw_success": raw_success,
            "validator_pass": validator_pass, "valid_success": valid_success,
            "schema_pass": layers["schema_valid"], "extractor_pass": layers["extractor_valid"],
            "protocol_pass": layers["protocol_valid"], "mined_pass": layers["mined_valid"],
            "targeted_success": adv_pred == 0, "untargeted_success": adv_pred != cid,
            "n_allowed_features": allowed,
            "n_features_modified": np.asarray(data["n_features_modified"], np.int64),
            "n_modified_outside_primattack_mask": outside,
            "l2_robust_scaled": l2, "linf_robust_scaled": linf, "l2_train_minmax": l2_minmax,
            "model_evaluations": evals,
            "model_evaluation_accounting": accounting,
            "cell_runtime_seconds": runtime, "cell_n": n,
            "clean_raw_sha256": canon["clean_raw_sha256"],
            **extra,
        })

    def _prim_cfg(self, dataset: str, cond: Cond) -> dict:
        cfg = json.loads((RUNS / dataset / cond.stage / "config.json").read_text(encoding="utf-8"))
        return {"method": cond.method, "budget": cond.budget, "objective": cond.objective,
                "eval_budget_per_flow": cfg["eval_budget_per_flow"],
                "hyperparameters": cfg["method_configs"][cond.method]}


# ----------------------------------------------------------------------------- pairing
def assert_paired(frames: list[pd.DataFrame], what: str) -> None:
    """Identical (dataset, victim, class, seed) cells with identical sample order / n / hashes."""
    ref = None
    for f in frames:
        seeds = set(f["seed"].unique())
        if seeds != set(SEEDS):
            raise AssertionError(f"{what}: seed set {sorted(seeds)} != {list(SEEDS)}")
        key = f[["dataset", "victim", "source_class", "seed", "sample_id",
                 "clean_raw_sha256", "checkpoint_sha256", "true_class", "clean_pred"]]
        if key.duplicated(["dataset", "victim", "source_class", "seed", "sample_id"]).any():
            raise AssertionError(f"{what}: duplicate sample IDs within a cell")
        if ref is None:
            ref = key.reset_index(drop=True)
        elif not ref.equals(key.reset_index(drop=True)):
            raise AssertionError(f"{what}: paired sample sets / order / hashes differ between "
                                 f"conditions ({f['method'].iloc[0]})")


def ref_vector(frame: pd.DataFrame, dataset: str, victim: str, column: str,
               seed: int = REF_SEED) -> tuple[np.ndarray, np.ndarray]:
    sub = frame[(frame.dataset == dataset) & (frame.victim == victim) & (frame.seed == seed)]
    ordered = pd.concat([sub[sub.source_class == c] for c in CLASSES])
    return ordered[column].to_numpy(bool), ordered["sample_id"].to_numpy()


# ----------------------------------------------------------------------------- aggregation
GROUP = ["dataset", "victim", "condition", "method", "objective", "budget"]


def seed_level(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope in ("victim", "class"):
        keys = GROUP + (["source_class"] if scope == "class" else []) + ["seed"]
        g = frame.groupby(keys, sort=False)
        agg = g.agg(n=("raw_success", "size"), raw_successes=("raw_success", "sum"),
                    valid_successes=("valid_success", "sum"),
                    validator_passes=("validator_pass", "sum"),
                    mean_model_evaluations=("model_evaluations", "mean"),
                    mean_l2_robust_scaled=("l2_robust_scaled", "mean"),
                    mean_linf_robust_scaled=("linf_robust_scaled", "mean"),
                    mean_l2_train_minmax=("l2_train_minmax", "mean"),
                    mean_features_modified=("n_features_modified", "mean"),
                    max_modified_outside_mask=("n_modified_outside_primattack_mask", "max"),
                    ).reset_index()
        cell_keys = keys if scope == "class" else keys[:-1] + ["source_class", "seed"]
        runtime = (frame.groupby(cell_keys, sort=False)["cell_runtime_seconds"].first()
                   .groupby(keys, sort=False).sum().rename("runtime_seconds").reset_index())
        agg = agg.merge(runtime, on=keys)
        vs = frame[frame.valid_success].groupby(keys, sort=False)
        agg = agg.merge(
            vs.agg(median_primitive_cost_valid=("primitive_normalized_cost", "median"),
                   median_p_valid=("primitive_p", "median"),
                   median_delay_valid=("primitive_delay", "median")).reset_index(),
            on=keys, how="left")
        agg["scope"] = scope
        if scope == "victim":
            agg["source_class"] = "ALL"
        rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out["raw_asr"] = out.raw_successes / out.n
    out["valid_asr"] = out.valid_successes / out.n
    out["validity_gap_pp"] = 100 * (out.raw_asr - out.valid_asr)
    out["validator_pass_rate"] = out.validator_passes / out.n
    out["ms_per_flow"] = 1000 * out.runtime_seconds / out.n
    cols = ["scope"] + GROUP + ["source_class", "seed", "n", "raw_successes", "valid_successes",
                                "validator_passes", "raw_asr", "valid_asr", "validity_gap_pp",
                                "validator_pass_rate", "mean_model_evaluations",
                                "mean_l2_robust_scaled", "mean_linf_robust_scaled",
                                "mean_l2_train_minmax",
                                "mean_features_modified", "max_modified_outside_mask",
                                "median_primitive_cost_valid", "median_p_valid",
                                "median_delay_valid", "runtime_seconds", "ms_per_flow"]
    return out[cols]


TABLE_METRICS = ("raw_asr", "valid_asr", "validity_gap_pp", "validator_pass_rate",
                 "mean_model_evaluations", "mean_l2_robust_scaled", "mean_linf_robust_scaled",
                 "mean_l2_train_minmax", "mean_features_modified",
                 "median_primitive_cost_valid", "ms_per_flow")


def table_level(seed_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["scope"] + GROUP + ["source_class"]
    recs = []
    for key, g in seed_df.groupby(keys, sort=False):
        seeds = sorted(g.seed.tolist())
        if seeds != list(SEEDS):
            raise AssertionError(f"{key}: seeds {seeds} != {list(SEEDS)}")
        if g.n.nunique() != 1:
            raise AssertionError(f"{key}: denominators differ across seeds {g.n.tolist()}")
        rec = dict(zip(keys, key))
        rec["n_per_seed"] = int(g.n.iloc[0])
        rec["max_modified_outside_mask"] = int(g.max_modified_outside_mask.max())
        g = g.set_index("seed").loc[list(SEEDS)]
        for m in TABLE_METRICS:
            vals = g[m].to_numpy(dtype=float)
            rec[f"{m}_mean"] = float(np.mean(vals))
            rec[f"{m}_sd"] = float(np.std(vals, ddof=1))
            for s, v in zip(SEEDS, vals):
                rec[f"{m}_seed{s}"] = float(v)
        recs.append(rec)
    return pd.DataFrame(recs)


def pm(mean: float, sd: float, pct: bool = True) -> str:
    if pct:
        return f"{100 * mean:.2f}% ± {100 * sd:.2f}%"
    return f"{mean:.2f} ± {sd:.2f}"


def fnum(value: float, spec: str = ".3f", suffix: str = "") -> str:
    return "—" if value is None or pd.isna(value) else f"{value:{spec}}{suffix}"


def pm_pp(mean: float, sd: float) -> str:
    return f"{mean:.2f} ± {sd:.2f} pp"


def seeds_str(row: pd.Series, metric: str, pct: bool = True) -> str:
    vals = [row[f"{metric}_seed{s}"] for s in SEEDS]
    if pct:
        return " / ".join(f"{100 * v:.2f}" for v in vals)
    return " / ".join(f"{v:.2f}" for v in vals)


def fmt_p(p: float | None) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p == 0.0:
        return "<1e-300"
    return f"{p:.3g}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- statistics
def mcnemar_row(exp: str, family: str, dataset: str, victim: str, outcome: str,
                a_label: str, b_label: str, vecs_a: dict, vecs_b: dict) -> dict:
    a, ids_a = vecs_a[REF_SEED]
    b, ids_b = vecs_b[REF_SEED]
    if not np.array_equal(ids_a, ids_b):
        raise AssertionError(f"{exp}/{dataset}/{victim}: {a_label} vs {b_label} not paired")
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    test = mcnemar_test(n10, n01)
    rec = {"experiment": exp, "family": family, "dataset": dataset, "victim": victim,
           "test": "McNemar", "comparison": f"{a_label} vs {b_label}", "outcome": outcome,
           "seed_used": REF_SEED, "n_paired": int(a.size), "rate_A": float(a.mean()),
           "rate_B": float(b.mean()), "diff_pp": 100 * float(a.mean() - b.mean()),
           "A_only": n10, "B_only": n01, "both": int((a & b).sum()),
           "neither": int((~a & ~b).sum()), "variant": test["test_variant"],
           "statistic": test["test_statistic"], "p_value": float(test["p_value"]),
           "p_holm": np.nan, "Q": np.nan, "df": np.nan, "k": 2}
    for s in SEEDS:
        if s == REF_SEED:
            continue
        sa, ia = vecs_a[s]
        sb, ib = vecs_b[s]
        if not np.array_equal(ia, ib):
            raise AssertionError(f"{exp}: seed {s} not paired")
        rec[f"diff_pp_seed{s}"] = 100 * float(sa.mean() - sb.mean())
        rec[f"A_only_seed{s}"] = int((sa & ~sb).sum())
        rec[f"B_only_seed{s}"] = int((~sa & sb).sum())
    return rec


def cochran_row(exp: str, family: str, dataset: str, victim: str, outcome: str,
                labels: list[str], vectors: list[np.ndarray]) -> dict:
    res = cochran_q(np.stack(vectors, axis=1))
    sig = res["p"] < ALPHA
    return {"experiment": exp, "family": family, "dataset": dataset, "victim": victim,
            "test": "Cochran's Q", "comparison": " / ".join(labels), "outcome": outcome,
            "seed_used": REF_SEED, "n_paired": int(vectors[0].size), "k": len(vectors),
            "Q": res["Q"], "df": res["df"], "p_value": res["p"], "p_holm": np.nan,
            "statistic": res["Q"],
            "variant": "Cochran's Q (chi-square, k-1 df)",
            "interpretation": (
                f"Valid success differs among the {len(vectors)} paired conditions "
                f"(Q = {res['Q']:.1f}, p = {fmt_p(res['p'])}); planned McNemar tests follow."
                if sig else
                f"No evidence that valid success differs among the {len(vectors)} conditions "
                f"(Q = {res['Q']:.2f}, p = {fmt_p(res['p'])}); planned McNemar tests not performed.")}


def holm_family(rows: list[dict]) -> None:
    adj = holm_adjust([r["p_value"] for r in rows])
    for r, p in zip(rows, adj):
        r["p_holm"] = p
        r["interpretation"] = mcnemar_text(r, p, "Holm-adjusted")


def mcnemar_text(r: dict, p: float, kind: str) -> str:
    a, b = r["comparison"].split(" vs ")
    d = r["diff_pp"]
    if p < ALPHA:
        direction = "higher" if d > 0 else "lower"
        return (f"{a} has {direction} {r['outcome']} than {b} by {abs(d):.2f} pp "
                f"({r['A_only']} vs {r['B_only']} discordant flows; {kind} p = {fmt_p(p)}).")
    return (f"No significant difference ({kind} p = {fmt_p(p)}; Δ = {d:+.2f} pp, "
            f"{r['A_only']} vs {r['B_only']} discordant flows).")


def not_performed(exp: str, family: str, dataset: str, victim: str, outcome: str,
                  comparisons: list[str]) -> list[dict]:
    return [{"experiment": exp, "family": family, "dataset": dataset, "victim": victim,
             "test": "McNemar", "comparison": c, "outcome": outcome, "seed_used": REF_SEED,
             "interpretation": "Not performed: the omnibus Cochran's Q was not significant."}
            for c in comparisons]


def vectors(frame: pd.DataFrame, dataset: str, victim: str, column: str) -> dict:
    return {s: ref_vector(frame, dataset, victim, column, s) for s in SEEDS}


# ----------------------------------------------------------------------------- plotting
VICTIM_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]
METHOD_COLORS = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860",
                 "#da8bc3"]


def grouped_bars(ax, groups: list[str], series: list[str], means, sds, colors, ylabel: str,
                 percent: bool = True) -> None:
    x = np.arange(len(groups))
    width = 0.8 / max(len(series), 1)
    for i, s in enumerate(series):
        m = np.asarray([means[(g, s)] for g in groups], float)
        e = np.asarray([sds[(g, s)] for g in groups], float)
        scale = 100 if percent else 1
        ax.bar(x + (i - (len(series) - 1) / 2) * width, m * scale, width, yerr=e * scale,
               capsize=2, label=s, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)


def per_dataset_bars(table: pd.DataFrame, group_col: str, series_col: str, metric: str,
                     title: str, ylabel: str, path: Path, group_order: dict, series_order: dict,
                     colors, percent: bool = True) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        t = table[table.dataset == dataset]
        groups = [g for g in group_order[dataset] if g in set(t[group_col])]
        series = [s for s in series_order[dataset] if s in set(t[series_col])]
        means = {(r[group_col], r[series_col]): r[f"{metric}_mean"] for _, r in t.iterrows()}
        sds = {(r[group_col], r[series_col]): r[f"{metric}_sd"] for _, r in t.iterrows()}
        grouped_bars(ax, groups, series, means, sds, colors, ylabel, percent)
        ax.set_title(DS_LABEL[dataset])
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.suptitle(title + "  (mean ± SD over seeds 42/2024/2026)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- outputs
def write_outputs(exp: str, per_sample: pd.DataFrame, seed_df: pd.DataFrame,
                  table_df: pd.DataFrame, stats: pd.DataFrame | None) -> Path:
    out = FINAL / EXP_DIRS[exp][0]
    (out / "plots").mkdir(parents=True, exist_ok=True)
    per_sample.to_parquet(out / "per_sample.parquet", index=False)
    seed_df.to_csv(out / "seed_level.csv", index=False)
    table_df.to_csv(out / "table_level.csv", index=False)
    if stats is not None:
        stats.to_csv(out / "statistical_tests.csv", index=False)
    return out


def include_interpretation(out: Path, final: bool) -> str:
    path = out / "interpretation.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    if final:
        raise FileNotFoundError(f"--final requires the hand-written {path.relative_to(REPO_ROOT)}")
    return "_Interpretation not yet written (`interpretation.md`)._"


def victims_order() -> dict:
    return {d: list(v) for d, v in DATASETS.items()}


def stats_md(stats: pd.DataFrame, exp_cols: list[str] | None = None) -> str:
    rows = []
    for _, r in stats.iterrows():
        rows.append({
            "Dataset": DS_LABEL[r["dataset"]], "Victim": r["victim"],
            "Family": r.get("family", ""), "Test": r["test"], "Comparison": r["comparison"],
            "n": "" if pd.isna(r.get("n_paired", np.nan)) else int(r["n_paired"]),
            "A-only": "" if pd.isna(r.get("A_only", np.nan)) else int(r["A_only"]),
            "B-only": "" if pd.isna(r.get("B_only", np.nan)) else int(r["B_only"]),
            "Δ (pp)": "" if pd.isna(r.get("diff_pp", np.nan)) else f"{r['diff_pp']:+.2f}",
            "Variant": ("" if pd.isna(r.get("variant", np.nan)) else
                        "Cochran χ²" if r["test"] == "Cochran's Q" else
                        "exact binomial" if "exact" in r["variant"] else "χ² (cc)"),
            "Statistic": ("" if r.get("statistic") is None or pd.isna(r.get("statistic"))
                          else f"{r['statistic']:.2f}"),
            "p": fmt_p(r.get("p_value")),
            "Holm p": fmt_p(r.get("p_holm")),
            "Interpretation": r.get("interpretation", ""),
        })
    df = pd.DataFrame(rows)
    if exp_cols:
        df = df[exp_cols]
    return md_table(df)


STAT_METHOD_TEXT = (
    "Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only "
    "(one outcome per flow, n = attempted flows of one victim, classes pooled within the "
    "victim), so the three seeded runs of a flow are never treated as independent "
    "observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired "
    "difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). "
    "McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic "
    "shown). α = 0.05. Holm correction only within the planned family of one experiment and "
    "one (dataset, victim).")


# ----------------------------------------------------------------------------- experiment A
def experiment_a(store: Store, selection: dict, final: bool) -> dict:
    sel = selection["selected"]
    prim_label = f"PrimAttack ({OPT_LABEL[sel]}, p75)"
    prim = prim_cond("primattack_untargeted", sel, "p75", "untargeted", prim_label)
    prim_unb = prim_cond("primattack_untargeted", sel, "unbounded", "untargeted",
                         f"PrimAttack ({OPT_LABEL[sel]}, unbounded)")
    bases = [base_cond(n) for n in BASELINE_LABEL]
    conds = [prim] + bases
    frames = [store.frame(c) for c in conds + [prim_unb]]
    assert_paired(frames, "Experiment A")
    per_sample = pd.concat(frames, ignore_index=True)
    seed_df = seed_level(per_sample)
    table = table_level(seed_df)

    stats = []
    labels = [c.label for c in conds]
    for dataset, victims in DATASETS.items():
        for victim in victims:
            vecs = {c.key: vectors(store.frame(c), dataset, victim, "valid_success") for c in conds}
            q = cochran_row("A", "A: 5 untargeted attacks", dataset, victim, "Valid ASR", labels,
                            [vecs[c.key][REF_SEED][0] for c in conds])
            stats.append(q)
            comps = [f"{prim.label} vs {b.label}" for b in bases]
            if q["p_value"] < ALPHA:
                fam = [mcnemar_row("A", "A: PrimAttack vs baselines (Holm over 4)", dataset,
                                   victim, "Valid ASR", prim.label, b.label, vecs[prim.key],
                                   vecs[b.key]) for b in bases]
                holm_family(fam)
                stats += fam
            else:
                stats += not_performed("A", "A: PrimAttack vs baselines (Holm over 4)", dataset,
                                       victim, "Valid ASR", comps)
    stats_df = pd.DataFrame(stats)
    out = write_outputs("A", per_sample, seed_df, table, stats_df)

    # diagnostics
    vt = table[table.scope == "victim"]
    diag_rows = []
    for c in [bases[2], bases[3], prim]:
        for dataset, victims in DATASETS.items():
            for victim in victims:
                r = vt[(vt.dataset == dataset) & (vt.victim == victim) & (vt.condition == c.key)].iloc[0]
                diag_rows.append({
                    "dataset": dataset, "victim": victim, "method": c.label,
                    "allowed_downstream_features": 23,
                    "mean_features_modified": r.mean_features_modified_mean,
                    "max_modified_outside_mask": r.max_modified_outside_mask,
                    "raw_asr_mean": r.raw_asr_mean, "raw_asr_sd": r.raw_asr_sd,
                    "valid_asr_mean": r.valid_asr_mean, "valid_asr_sd": r.valid_asr_sd,
                    "validity_gap_pp_mean": r.validity_gap_pp_mean,
                    "validity_gap_pp_sd": r.validity_gap_pp_sd,
                    "validator_pass_rate_mean": r.validator_pass_rate_mean,
                    "validator_pass_rate_sd": r.validator_pass_rate_sd,
                    "mean_model_evaluations": r.mean_model_evaluations_mean,
                })
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(out / "constrained_baseline_diagnostics.csv", index=False)
    prim_cost = primitive_cost_table(per_sample[per_sample.condition.isin([prim.key, prim_unb.key])])
    prim_cost.to_csv(out / "primattack_primitive_costs.csv", index=False)

    # plots
    method_order = {d: [c.label for c in conds] for d in DATASETS}
    vo = victims_order()
    per_dataset_bars(vt[vt.condition.isin([c.key for c in conds])], "method", "victim", "raw_asr",
                     "Exp A — Raw ASR by attack (untargeted)", "Raw ASR (%)",
                     out / "plots" / "A1_raw_asr_by_attack.png", method_order, vo, VICTIM_COLORS)
    per_dataset_bars(vt[vt.condition.isin([c.key for c in conds])], "method", "victim", "valid_asr",
                     "Exp A — Valid ASR by attack (untargeted)", "Valid ASR (%)",
                     out / "plots" / "A2_valid_asr_by_attack.png", method_order, vo, VICTIM_COLORS)
    per_dataset_bars(vt[vt.condition.isin([c.key for c in conds])], "victim", "method", "valid_asr",
                     "Exp A — Model-wise Valid ASR", "Valid ASR (%)",
                     out / "plots" / "A4_modelwise_valid_asr.png", vo, method_order, METHOD_COLORS)
    per_dataset_bars(vt[vt.condition.isin([c.key for c in conds])], "method", "victim",
                     "validity_gap_pp", "Exp A — Validity Gap (Raw − Valid ASR) by attack",
                     "Validity gap (pp)", out / "plots" / "A5_validity_gap_by_attack.png",
                     method_order, vo, VICTIM_COLORS, percent=False)
    classwise_heatmap(table[(table.scope == "class") & table.condition.isin([c.key for c in conds])],
                      [c.label for c in conds], out / "plots" / "A3_classwise_valid_asr.png")
    p75_vs_unbounded_plot(vt, prim, prim_unb, out / "plots" / "A6_primattack_p75_vs_unbounded.png")

    report_a(out, table, stats_df, diag, prim_cost, conds, prim, prim_unb, store, final)
    return {"table": table, "stats": stats_df, "prim": prim, "bases": bases}


def primitive_cost_table(frame: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for (dataset, victim, method, seed), g in frame.groupby(["dataset", "victim", "method", "seed"], sort=False):
        v = g[g.valid_success]
        recs.append({"dataset": dataset, "victim": victim, "method": method, "seed": seed,
                     "n_valid_successes": int(len(v)),
                     "median_p_bytes_valid": float(v.primitive_p.median()) if len(v) else np.nan,
                     "median_delay_us_valid": float(v.primitive_delay.median()) if len(v) else np.nan,
                     "median_normalized_cost_valid": float(v.primitive_normalized_cost.median()) if len(v) else np.nan,
                     "share_valid_using_padding": float((v.primitive_p > 0).mean()) if len(v) else np.nan,
                     "share_valid_using_timing": float((v.primitive_delay > 0).mean()) if len(v) else np.nan,
                     "median_p_hi_bytes": float(g.primitive_p_hi.median()),
                     "median_delay_hi_us": float(g.primitive_delay_hi.median()),
                     "share_no_headroom": float((g.failure_category == "no_headroom").mean())})
    df = pd.DataFrame(recs)
    agg = df.groupby(["dataset", "victim", "method"], sort=False).agg(
        n_valid_successes_mean=("n_valid_successes", "mean"),
        median_p_bytes_valid=("median_p_bytes_valid", "mean"),
        median_delay_us_valid=("median_delay_us_valid", "mean"),
        median_normalized_cost_valid=("median_normalized_cost_valid", "mean"),
        share_valid_using_padding=("share_valid_using_padding", "mean"),
        share_valid_using_timing=("share_valid_using_timing", "mean"),
        median_p_hi_bytes=("median_p_hi_bytes", "mean"),
        median_delay_hi_us=("median_delay_hi_us", "mean"),
        share_no_headroom=("share_no_headroom", "mean")).reset_index()
    return agg


def classwise_heatmap(table: pd.DataFrame, method_order: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(8 * len(DATASETS), 3.8), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        t = table[table.dataset == dataset]
        cols = [(v, c) for v in DATASETS[dataset] for c in CLASSES]
        mat = np.full((len(method_order), len(cols)), np.nan)
        for i, m in enumerate(method_order):
            for j, (v, c) in enumerate(cols):
                r = t[(t.method == m) & (t.victim == v) & (t.source_class == c)]
                if len(r):
                    mat[i, j] = 100 * r.valid_asr_mean.iloc[0]
        im = ax.imshow(mat, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color="white" if v < 60 else "black")
        ax.set_yticks(range(len(method_order)))
        ax.set_yticklabels(method_order, fontsize=7)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{v}\n{c}" for v, c in cols], fontsize=6, rotation=90)
        ax.set_title(f"{DS_LABEL[dataset]} — class-wise Valid ASR (%), seed mean")
        fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def p75_vs_unbounded_plot(vt: pd.DataFrame, prim: Cond, prim_unb: Cond, path: Path) -> None:
    rows = []
    for c, name in ((prim, "p75"), (prim_unb, "unbounded")):
        for _, r in vt[vt.condition == c.key].iterrows():
            for metric, mname in (("raw_asr", "Raw"), ("valid_asr", "Valid")):
                rows.append({"dataset": r.dataset, "victim": r.victim, "series": f"{mname} ASR, {name}",
                             "v_mean": r[f"{metric}_mean"], "v_sd": r[f"{metric}_sd"]})
    df = pd.DataFrame(rows).rename(columns={"v_mean": "x_mean", "v_sd": "x_sd"})
    order = ["Raw ASR, p75", "Valid ASR, p75", "Raw ASR, unbounded", "Valid ASR, unbounded"]
    per_dataset_bars(df, "victim", "series", "x", "Exp A — PrimAttack (untargeted) p75 vs unbounded budget",
                     "ASR (%)", path, victims_order(), {d: order for d in DATASETS},
                     ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"])


def fairness_guide_a() -> str:
    return "\n".join([
        "### Fairness guide — what is held constant and what differs",
        "",
        "**Held constant for all five attacks:** the canonical source flows (identical sample IDs, "
        "clean inputs, labels and clean predictions, asserted), the victim checkpoints (SHA-256 "
        "asserted), preprocessing/scaling, the chronological test split, the untargeted objective "
        "(success = prediction ≠ true source class), the seeds 42/2024/2026, the success "
        "definitions, the independent validator_v2 `hybrid_valid` verdict on the final adversarial "
        "flow, and the aggregation rules (one denominator = attempted clean-correct flows).",
        "",
        "**Inherently different (not forced to be equal):**",
        "",
        "| Attack | Attack space / feature mask | Norm / budget | Iterations / restarts | Loss | "
        "Projection / clamping | Stopping rule | Evaluations per flow | Implementation |",
        "|---|---|---|---|---|---|---|---|---|",
        "| PGD | all 79 features, victim RobustScaler space | L∞ ε = 0.5 | 40 steps, α = 0.05, "
        "1 random start | CE (untargeted) | L∞ ball only (no box/type/mask) | fixed 40 steps; final "
        "iterate | 40 fwd+bwd (exact) | `attack/input_baselines.py` |",
        "| C&W | all 79 features, victim RobustScaler space | L2 penalty, no hard ε | ≤ 60 Adam "
        "steps, lr 0.01, λ = 1, κ = 0; 1 run | max(z_true − max z_other + κ, 0)·λ + ‖δ‖² | none | "
        "convergence (Δδ < 1e-5) or 60 steps; lowest-L2 success kept | 2 fwd per step (exact) | "
        "`attack/input_baselines.py` |",
        "| CAPGD-PrimSupport | 23-feature `primattack_joint_feature_mask`; all other 56 features "
        "bitwise unchanged (asserted) | L2 ε = 0.5 in train min-max space | 10 steps, 2 restarts "
        "(TabularBench CAPGD) | CE (untargeted) | L2 ball + train box + mask + integer-type repair "
        "| fixed steps | forward-hook batch mean | frozen `external/tabularbench` via "
        "`comparisons/capgd_cicids2017.py` |",
        "| C-PGD-PrimSupport | same 23-feature mask (asserted) | L2 ε = 0.5 in train min-max space "
        "| 40 steps, step 0.05, 1 random start | CE − 1.0·differentiable relation penalty | L2 ball "
        "+ train box + mask + integer-type repair | fixed steps | exact per flow | "
        "`comparisons/cpgd_prim_support.py` (Simonetto et al., IJCAI 2022) |",
        "| PrimAttack | primitive controls only: padding p (bytes/fwd packet, increase-only), "
        "added forward delay (µs) + shape; downstream changes only through the canonical "
        "recomputation φ onto the same 23-feature support | train-calibrated per-class p75 box "
        "(joint) | per-flow cap of 256 victim evaluations | untargeted margin "
        "z_true − max z_other on realized flows | integer bytes/µs projection, capability gates, "
        "quantized recomputation | incumbent: success > failure, lowest primitive cost among "
        "successes, best margin among failures | exact per flow | `attack/primitive_optimizer.py` |",
        "",
        "**Validator in the loop.** PrimAttack's search success predicate includes validator_v2 "
        "(it keeps the cheapest *valid* success). PGD, C&W and CAPGD optimize without the "
        "validator; C-PGD optimizes a differentiable subset of flow relations (penalty), not the "
        "validator. Validator access is part of the PrimAttack threat model, not a shared "
        "setting.",
        "",
        "**Matched support is not matched feasibility.** CAPGD/C-PGD may move any of the 23 "
        "coordinates independently within their norm ball. PrimAttack reaches the same coordinates "
        "only through two coupled, increase-only primitives. The mask gives a controlled "
        "matched-support comparison. It does not give CAPGD/C-PGD packet-level realizability.",
    ])


def report_a(out: Path, table: pd.DataFrame, stats: pd.DataFrame, diag: pd.DataFrame,
             prim_cost: pd.DataFrame, conds: list[Cond], prim: Cond, prim_unb: Cond,
             store: Store, final: bool) -> None:
    vt = table[table.scope == "victim"]
    main_rows = []
    for dataset, victims in DATASETS.items():
        for victim in victims:
            for c in conds + [prim_unb]:
                r = vt[(vt.dataset == dataset) & (vt.victim == victim) & (vt.condition == c.key)].iloc[0]
                if c.prim:
                    budget = f"primitive box {c.budget} (joint), ≤256 evals/flow"
                    pert = (f"median normalized primitive cost of valid successes "
                            f"{fnum(r.median_primitive_cost_valid_mean)} (p/p_hi + delay/delay_hi)")
                elif c.method == "pgd_untargeted":
                    budget = BASELINE_NATIVE_BUDGET[c.method]
                    pert = (f"mean L∞ (RobustScaler) {r.mean_linf_robust_scaled_mean:.3f}; "
                            f"{r.mean_features_modified_mean:.1f} features modified")
                elif c.method == "cw_untargeted":
                    budget = BASELINE_NATIVE_BUDGET[c.method]
                    pert = (f"mean L2 (RobustScaler) {r.mean_l2_robust_scaled_mean:.3f}; "
                            f"{r.mean_features_modified_mean:.1f} features modified")
                else:
                    budget = BASELINE_NATIVE_BUDGET[c.method]
                    pert = (f"mean L2 (train min-max) {r.mean_l2_train_minmax_mean:.3f}; "
                            f"{r.mean_features_modified_mean:.1f} of 23 features modified")
                main_rows.append({
                    "Dataset": DS_LABEL[dataset], "Victim": victim,
                    "Attack": c.label + (" †" if c is prim_unb else ""),
                    "Budget / configuration": budget, "n/seed": r.n_per_seed,
                    "Raw ASR": pm(r.raw_asr_mean, r.raw_asr_sd),
                    "Valid ASR": pm(r.valid_asr_mean, r.valid_asr_sd),
                    "Validity Gap": pm_pp(r.validity_gap_pp_mean, r.validity_gap_pp_sd),
                    "Raw per seed (42/2024/2026, %)": seeds_str(r, "raw_asr"),
                    "Valid per seed (%)": seeds_str(r, "valid_asr"),
                    "Perturbation / budget metric": pert})
    diag_md = pd.DataFrame([{
        "Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Attack": r.method,
        "Allowed downstream features": r.allowed_downstream_features,
        "Mean modified features": f"{r.mean_features_modified:.2f}",
        "Max modified outside mask": int(r.max_modified_outside_mask),
        "Raw ASR": pm(r.raw_asr_mean, r.raw_asr_sd),
        "Valid ASR": pm(r.valid_asr_mean, r.valid_asr_sd),
        "Validity Gap": pm_pp(r.validity_gap_pp_mean, r.validity_gap_pp_sd),
        "Validator pass rate": pm(r.validator_pass_rate_mean, r.validator_pass_rate_sd),
        "Mean victim evals / flow": f"{r.mean_model_evaluations:.1f}",
    } for _, r in diag.iterrows()])
    cost_md = pd.DataFrame([{
        "Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Configuration": r.method,
        "Valid successes / seed": f"{r.n_valid_successes_mean:.1f}",
        "Median p (bytes/pkt) of valid successes": fnum(r.median_p_bytes_valid, ".1f"),
        "Median added delay (µs)": fnum(r.median_delay_us_valid, ".0f"),
        "Median normalized primitive cost": fnum(r.median_normalized_cost_valid),
        "Valid successes using padding / timing": (
            f"{fnum(100 * r.share_valid_using_padding, '.0f', '%')} / "
            f"{fnum(100 * r.share_valid_using_timing, '.0f', '%')}"),
        "Median per-flow cap p_hi (bytes) / delay_hi (µs)": f"{r.median_p_hi_bytes:.1f} / {r.median_delay_hi_us:.0f}",
        "Flows with no primitive headroom": f"{100 * r.share_no_headroom:.1f}%",
    } for _, r in prim_cost.iterrows()])
    classwise = table[(table.scope == "class") & table.condition.isin([c.key for c in conds])]
    cw_md = pd.DataFrame([{
        "Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Class": r.source_class,
        "Attack": r.method, "n/seed": r.n_per_seed, "Raw ASR": pm(r.raw_asr_mean, r.raw_asr_sd),
        "Valid ASR": pm(r.valid_asr_mean, r.valid_asr_sd),
        "Gap": pm_pp(r.validity_gap_pp_mean, r.validity_gap_pp_sd)} for _, r in classwise.iterrows()])
    lines = [
        "# Final Experiment A — Primary baseline comparison (untargeted)",
        "",
        "Five attacks run on identical paired source flows under one **untargeted** objective "
        "(success = prediction ≠ original malicious class). Metrics: Raw ASR = successes / "
        "attempted flows; Valid ASR = (success ∧ validator_v2 `hybrid_valid` on the same final "
        "flow) / attempted flows; Validity Gap = Raw − Valid (pp). Values are mean ± SD over "
        "attack seeds 42/2024/2026. Classes are pooled within a victim. Victims and datasets are "
        "never pooled. Protocol: `../00_PROTOCOL.md`.",
        "",
        f"PrimAttack configuration: optimizer **{prim.label}** (selected by the pre-registered "
        "Exp B rule; see `../B_optimizer_selection/`), joint mode, p75 budget. † = descriptive "
        "p75-vs-unbounded row, not part of the inferential comparison.",
        "",
        fairness_guide_a(),
        "",
        "## Main baseline table",
        "",
        md_table(pd.DataFrame(main_rows)),
        "",
        "## Constrained-baseline diagnostic table",
        "",
        "`Allowed downstream features` = size of the canonical `primattack_joint_feature_mask` "
        "(23 of 79). For CAPGD/C-PGD it is the set of directly optimized coordinates. For "
        "PrimAttack it is the potential write-support of its recomputation φ. `Max modified "
        "outside mask` = 0 confirms that no feature outside the mask changed in any flow of any "
        "seed (also asserted at run time and in this analysis). A modified-feature count is "
        "**not** a PrimAttack primitive cost. PrimAttack's primitive cost is reported separately "
        "below. Evaluation counts: C-PGD and PrimAttack are exact per flow. CAPGD is the "
        "forward-hook batch mean.",
        "",
        md_table(diag_md),
        "",
        "### PrimAttack primitive-domain cost / budget information",
        "",
        "Normalized primitive cost = p/p_hi + delay/delay_hi (the incumbent's cost rule). The "
        "values are medians over valid successes, averaged over seeds.",
        "",
        md_table(cost_md),
        "",
        "## Statistical analysis (Valid Success)",
        "",
        STAT_METHOD_TEXT,
        "",
        "Cochran's Q across the five paired attacks per (dataset, victim). Only if it is "
        "significant: the four planned McNemar comparisons PrimAttack vs each baseline, "
        "Holm-corrected over those four. A = PrimAttack, B = baseline, Δ = Valid ASR(A) − "
        "Valid ASR(B) in pp at seed 42.",
        "",
        stats_md(stats),
        "",
        "## Class-wise results",
        "",
        md_table(cw_md),
        "",
        "## Plots",
        "",
        "- `plots/A1_raw_asr_by_attack.png` — Raw ASR by attack",
        "- `plots/A2_valid_asr_by_attack.png` — Valid ASR by attack",
        "- `plots/A3_classwise_valid_asr.png` — class-wise Valid ASR",
        "- `plots/A4_modelwise_valid_asr.png` — model-wise Valid ASR",
        "- `plots/A5_validity_gap_by_attack.png` — Validity Gap by attack",
        "- `plots/A6_primattack_p75_vs_unbounded.png` — PrimAttack p75 vs unbounded",
        "",
        "## Machine-readable outputs",
        "",
        "`per_sample.parquet` (every flow × seed × attack, incl. clean/adversarial prediction, raw "
        "success, validator pass, valid success, evaluations, primitive controls, attack "
        "parameters), `seed_level.csv`, `table_level.csv`, `statistical_tests.csv`, "
        "`constrained_baseline_diagnostics.csv`, `primattack_primitive_costs.csv`.",
        "",
        "## Interpretation",
        "",
        include_interpretation(out, final),
        "",
    ]
    (out / EXP_DIRS["A"][1]).write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- experiment B
def experiment_b(store: Store, selection: dict, final: bool) -> dict:
    conds = [prim_cond("primattack_targeted_optimizers", m, "p75", "targeted", OPT_LABEL[m])
             for m in ("hybrid", "pgd", "cw")]
    frames = [store.frame(c) for c in conds]
    assert_paired(frames, "Experiment B")
    per_sample = pd.concat(frames, ignore_index=True)
    # Recompute the pre-registered selection from per-sample data and check the frozen choice.
    agg = per_sample.groupby("method_id").agg(s=("valid_success", "sum"), n=("valid_success", "size"),
                                              e=("model_evaluations", "mean"))
    order = ["hybrid", "pgd", "cw"]
    ranking = sorted(order, key=lambda m: (-agg.loc[m, "s"] / agg.loc[m, "n"], agg.loc[m, "e"],
                                           order.index(m)))
    if ranking != selection["ranking"]:
        raise AssertionError(f"optimizer ranking from per-sample data {ranking} != frozen "
                             f"{selection['ranking']}")
    seed_df = seed_level(per_sample)
    table = table_level(seed_df)
    stats = []
    labels = [c.label for c in conds]
    pairs = [(0, 1), (0, 2), (1, 2)]
    for dataset, victims in DATASETS.items():
        for victim in victims:
            vecs = {c.key: vectors(store.frame(c), dataset, victim, "valid_success") for c in conds}
            q = cochran_row("B", "B: 3 optimizers", dataset, victim, "Valid Targeted ASR", labels,
                            [vecs[c.key][REF_SEED][0] for c in conds])
            stats.append(q)
            comps = [f"{conds[i].label} vs {conds[j].label}" for i, j in pairs]
            if q["p_value"] < ALPHA:
                fam = [mcnemar_row("B", "B: optimizer pairs (Holm over 3)", dataset, victim,
                                   "Valid Targeted ASR", conds[i].label, conds[j].label,
                                   vecs[conds[i].key], vecs[conds[j].key]) for i, j in pairs]
                holm_family(fam)
                stats += fam
            else:
                stats += not_performed("B", "B: optimizer pairs (Holm over 3)", dataset, victim,
                                       "Valid Targeted ASR", comps)
    stats_df = pd.DataFrame(stats)
    out = write_outputs("B", per_sample, seed_df, table, stats_df)
    (out / "optimizer_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    vt = table[table.scope == "victim"]
    lo = {d: labels for d in DATASETS}
    vo = victims_order()
    per_dataset_bars(vt, "victim", "method", "raw_asr", "Exp B — Raw Targeted ASR by optimizer (p75)",
                     "Raw targeted ASR (%)", out / "plots" / "B1_raw_asr_by_optimizer.png", vo, lo,
                     METHOD_COLORS)
    per_dataset_bars(vt, "victim", "method", "valid_asr", "Exp B — Valid Targeted ASR by optimizer (p75)",
                     "Valid targeted ASR (%)", out / "plots" / "B2_valid_asr_by_optimizer.png", vo, lo,
                     METHOD_COLORS)
    runtime_plot(vt, labels, out / "plots" / "B3_runtime_and_evaluations.png")
    report_b(out, table, stats_df, selection, labels, final)
    return {"table": table, "stats": stats_df}


def runtime_plot(vt: pd.DataFrame, labels: list[str], path: Path) -> None:
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(6.5 * len(DATASETS), 7), squeeze=False)
    for j, dataset in enumerate(DATASETS):
        t = vt[vt.dataset == dataset]
        for i, (metric, ylab) in enumerate((("mean_model_evaluations", "victim evaluations / flow"),
                                            ("ms_per_flow", "runtime (ms / flow)"))):
            means = {(r.victim, r.method): r[f"{metric}_mean"] for _, r in t.iterrows()}
            sds = {(r.victim, r.method): r[f"{metric}_sd"] for _, r in t.iterrows()}
            grouped_bars(axes[i][j], list(DATASETS[dataset]), labels, means, sds, METHOD_COLORS,
                         ylab, percent=False)
            axes[i][j].set_title(f"{DS_LABEL[dataset]} — {ylab}")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Exp B — cost of the three PrimAttack optimizers (mean ± SD over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_b(out: Path, table: pd.DataFrame, stats: pd.DataFrame, selection: dict,
             labels: list[str], final: bool) -> None:
    vt = table[table.scope == "victim"]
    rows = [{"Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Optimizer": r.method,
             "n/seed": r.n_per_seed,
             "Raw Targeted ASR": pm(r.raw_asr_mean, r.raw_asr_sd),
             "Valid Targeted ASR": pm(r.valid_asr_mean, r.valid_asr_sd),
             "Validity Gap": pm_pp(r.validity_gap_pp_mean, r.validity_gap_pp_sd),
             "Valid per seed (42/2024/2026, %)": seeds_str(r, "valid_asr"),
             "Evals / flow": pm(r.mean_model_evaluations_mean, r.mean_model_evaluations_sd, False),
             "ms / flow": pm(r.ms_per_flow_mean, r.ms_per_flow_sd, False),
             "Median primitive cost (valid)": fnum(r.median_primitive_cost_valid_mean)}
            for _, r in vt.iterrows()]
    sel_rows = [{"Rank": i + 1, "Optimizer": OPT_LABEL[m],
                 "Valid targeted successes": selection["aggregate"][m]["valid_targeted_successes"],
                 "Attempts": selection["aggregate"][m]["attempts"],
                 "Aggregate Valid Targeted ASR": f"{100 * selection['aggregate'][m]['aggregate_valid_targeted_asr']:.3f}%",
                 "Mean evals / flow": f"{selection['aggregate'][m]['mean_evaluations_per_flow']:.1f}"}
                for i, m in enumerate(selection["ranking"])]
    cw = table[table.scope == "class"]
    cw_md = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Class": r.source_class,
                           "Optimizer": r.method, "Raw": pm(r.raw_asr_mean, r.raw_asr_sd),
                           "Valid": pm(r.valid_asr_mean, r.valid_asr_sd)} for _, r in cw.iterrows()])
    lines = [
        "# Final Experiment B — PrimAttack optimizer selection (targeted → Benign, p75)",
        "",
        "Three optimizers over the identical PrimAttack attack space (`RealizedSearch`: the same "
        "primitive parameterization, joint mode, per-class p75 box, recomputation φ, integer "
        "rounding, victim, validator_v2 gate and success predicate; every reported candidate is a "
        "realized, quantized, recomputed flow; per-flow incumbent: success > failure, cheapest "
        "success, best-margin failure). Matched per-flow cap of 256 victim evaluations. Success = "
        "prediction == Benign. Valid success additionally requires validator_v2 `hybrid_valid`. "
        "Mean ± SD over seeds 42/2024/2026.",
        "",
        "| Optimizer | Hyperparameters (frozen; Prim-PGD/C&W tuned on the CICIDS2017 validation split) |",
        "|---|---|",
        "| Hybrid Search | exact integer padding enumeration, then adaptive projected sign-momentum "
        "refinement (40 steps/restart, lr 0.1, momentum 0.75, stall halving), restarts until the "
        "256-evaluation budget |",
        "| Prim-PGD | 3 restarts (clean + 2 uniform) × 42 steps, α = 0.05, momentum 0.75 |",
        "| Prim-C&W | 3 binary-search stages × 42 Adam steps, lr 0.5, c₀ = 1, κ = 0 |",
        "",
        "## Optimizer selection (pre-registered criterion)",
        "",
        f"Rule: {selection['rule']}. **Selected: {OPT_LABEL[selection['selected']]}.** "
        f"Budget-sensitivity optimizers (top 2): "
        f"{', '.join(OPT_LABEL[m] for m in selection['budget_sensitivity_optimizers'])}. "
        "The selection is based on the aggregate Valid Targeted ASR, not on p-values. The tests "
        "below are supporting evidence.",
        "",
        md_table(pd.DataFrame(sel_rows)),
        "",
        "## Results per dataset and victim",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        "## Statistical analysis (Valid Targeted Success)",
        "",
        STAT_METHOD_TEXT,
        "",
        "Cochran's Q across the three optimizers per (dataset, victim). If significant: Hybrid vs "
        "Prim-PGD, Hybrid vs Prim-C&W and Prim-PGD vs Prim-C&W, Holm over the three.",
        "",
        stats_md(stats),
        "",
        "## Class-wise results",
        "",
        md_table(cw_md),
        "",
        "## Plots",
        "",
        "- `plots/B1_raw_asr_by_optimizer.png`",
        "- `plots/B2_valid_asr_by_optimizer.png`",
        "- `plots/B3_runtime_and_evaluations.png`",
        "",
        "## Interpretation",
        "",
        include_interpretation(out, final),
        "",
    ]
    (out / EXP_DIRS["B"][1]).write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- experiment C
def experiment_c(store: Store, selection: dict, final: bool) -> dict:
    opts = selection["budget_sensitivity_optimizers"]
    conds = {}
    for m in opts:
        conds[m] = [
            prim_cond("primattack_targeted_budgets", m, "p50", "targeted", f"{OPT_LABEL[m]} p50"),
            prim_cond("primattack_targeted_optimizers", m, "p75", "targeted", f"{OPT_LABEL[m]} p75"),
            prim_cond("primattack_targeted_budgets", m, "unbounded", "targeted",
                      f"{OPT_LABEL[m]} unbounded"),
        ]
    allc = [c for m in opts for c in conds[m]]
    frames = [store.frame(c) for c in allc]
    assert_paired(frames, "Experiment C")
    per_sample = pd.concat(frames, ignore_index=True)
    # the p75 label differs from Exp B's; rewrite for this experiment's tables only
    seed_df = seed_level(per_sample)
    table = table_level(seed_df)
    stats = []
    for m in opts:
        cs = conds[m]
        for dataset, victims in DATASETS.items():
            for victim in victims:
                vecs = {c.key: vectors(store.frame(c), dataset, victim, "valid_success") for c in cs}
                fam_name = f"C: {OPT_LABEL[m]} adjacent budgets (Holm over 2)"
                q = cochran_row("C", f"C: {OPT_LABEL[m]} 3 budgets", dataset, victim,
                                "Valid Targeted ASR", [c.label for c in cs],
                                [vecs[c.key][REF_SEED][0] for c in cs])
                stats.append(q)
                if q["p_value"] < ALPHA:
                    fam = [mcnemar_row("C", fam_name, dataset, victim, "Valid Targeted ASR",
                                       cs[1].label, cs[0].label, vecs[cs[1].key], vecs[cs[0].key]),
                           mcnemar_row("C", fam_name, dataset, victim, "Valid Targeted ASR",
                                       cs[2].label, cs[1].label, vecs[cs[2].key], vecs[cs[1].key])]
                    holm_family(fam)
                    stats += fam
                else:
                    stats += not_performed("C", fam_name, dataset, victim, "Valid Targeted ASR",
                                           [f"{cs[1].label} vs {cs[0].label}",
                                            f"{cs[2].label} vs {cs[1].label}"])
    stats_df = pd.DataFrame(stats)
    out = write_outputs("C", per_sample, seed_df, table, stats_df)
    vt = table[table.scope == "victim"]
    for metric, name, ylab, pct in (("raw_asr", "C1_raw_asr_vs_budget", "Raw targeted ASR (%)", True),
                                    ("valid_asr", "C2_valid_asr_vs_budget", "Valid targeted ASR (%)", True),
                                    ("validity_gap_pp", "C3_validity_gap_vs_budget", "Validity gap (pp)", False)):
        budget_lines(vt, opts, metric, ylab, pct, out / "plots" / f"{name}.png")
    report_c(out, table, stats_df, opts, final)
    return {"table": table, "stats": stats_df}


def budget_lines(vt: pd.DataFrame, opts: list[str], metric: str, ylab: str, pct: bool, path: Path) -> None:
    budgets = ["p50", "p75", "unbounded"]
    styles = ["-", "--"]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 4.2), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for vi, victim in enumerate(DATASETS[dataset]):
            for oi, m in enumerate(opts):
                t = vt[(vt.dataset == dataset) & (vt.victim == victim) & (vt.method.str.startswith(OPT_LABEL[m]))]
                t = t.set_index("budget").loc[budgets]
                scale = 100 if pct else 1
                ax.errorbar(range(3), scale * t[f"{metric}_mean"], yerr=scale * t[f"{metric}_sd"],
                            color=VICTIM_COLORS[vi], linestyle=styles[oi], marker="o", capsize=2,
                            label=f"{victim} — {OPT_LABEL[m]}")
        ax.set_xticks(range(3))
        ax.set_xticklabels(budgets)
        ax.set_xlabel("PrimAttack budget")
        ax.set_ylabel(ylab)
        ax.set_title(DS_LABEL[dataset])
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=6)
    fig.suptitle(f"Exp C — {ylab} vs budget (targeted → Benign; mean ± SD over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def report_c(out: Path, table: pd.DataFrame, stats: pd.DataFrame, opts: list[str], final: bool) -> None:
    vt = table[table.scope == "victim"]
    side = []
    for dataset, victims in DATASETS.items():
        for victim in victims:
            for b in ("p50", "p75", "unbounded"):
                rec = {"Dataset": DS_LABEL[dataset], "Victim": victim, "Budget": b}
                for m in opts:
                    r = vt[(vt.dataset == dataset) & (vt.victim == victim) & (vt.budget == b)
                           & vt.method.str.startswith(OPT_LABEL[m])].iloc[0]
                    rec[f"{OPT_LABEL[m]} Raw"] = pm(r.raw_asr_mean, r.raw_asr_sd)
                    rec[f"{OPT_LABEL[m]} Valid"] = pm(r.valid_asr_mean, r.valid_asr_sd)
                    rec[f"{OPT_LABEL[m]} Gap"] = pm_pp(r.validity_gap_pp_mean, r.validity_gap_pp_sd)
                side.append(rec)
    seeds_rows = [{"Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Condition": r.method,
                   "Raw per seed (42/2024/2026, %)": seeds_str(r, "raw_asr"),
                   "Valid per seed (%)": seeds_str(r, "valid_asr"),
                   "Median primitive cost (valid)": fnum(r.median_primitive_cost_valid_mean)}
                  for _, r in vt.iterrows()]
    cw = table[table.scope == "class"]
    cw_md = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Class": r.source_class,
                           "Condition": r.method, "Raw": pm(r.raw_asr_mean, r.raw_asr_sd),
                           "Valid": pm(r.valid_asr_mean, r.valid_asr_sd)} for _, r in cw.iterrows()])
    lines = [
        "# Final Experiment C — PrimAttack budget sensitivity (targeted → Benign)",
        "",
        "Budgets are train-only per-class calibrations (`artifacts/primattack/budget_calibration*.json`, "
        "`fit_split = train`): **p50** (intermediate) and **p75** (maximum-evaluated) cap padding "
        "bytes and relative duration change at the class's train percentiles. **unbounded** "
        "removes those caps and keeps only the train-p99 feature envelope and the DoS/DDoS "
        "min-rate floor. All other settings are identical (joint mode, 256 evaluations/flow, same "
        "flows, seeds, victims, validator, success predicate). Optimizers: the top two of the Exp B "
        f"ranking ({', '.join(OPT_LABEL[m] for m in opts)}). Their p75 cells are the Exp B cells.",
        "",
        "## Valid / Raw ASR by budget (side by side per optimizer)",
        "",
        md_table(pd.DataFrame(side)),
        "",
        "## Per-seed values",
        "",
        md_table(pd.DataFrame(seeds_rows)),
        "",
        "## Statistical analysis (Valid Targeted Success)",
        "",
        STAT_METHOD_TEXT,
        "",
        "Per optimizer and (dataset, victim): Cochran's Q across p50 / p75 / unbounded. If "
        "significant: the adjacent comparisons p75 vs p50 and unbounded vs p75, Holm over the two. "
        "Direct optimizer-vs-optimizer tests per budget are not part of the planned suite.",
        "",
        stats_md(stats),
        "",
        "## Class-wise results",
        "",
        md_table(cw_md),
        "",
        "## Plots",
        "",
        "- `plots/C1_raw_asr_vs_budget.png`",
        "- `plots/C2_valid_asr_vs_budget.png`",
        "- `plots/C3_validity_gap_vs_budget.png`",
        "",
        "## Interpretation",
        "",
        include_interpretation(out, final),
        "",
    ]
    (out / EXP_DIRS["C"][1]).write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- experiment D
def experiment_d(store: Store, selection: dict, final: bool) -> dict:
    sel = selection["selected"]
    tgt = prim_cond("primattack_targeted_optimizers", sel, "p75", "targeted",
                    f"{OPT_LABEL[sel]} targeted→Benign")
    unt = prim_cond("primattack_untargeted", sel, "p75", "untargeted",
                    f"{OPT_LABEL[sel]} untargeted")
    frames = [store.frame(tgt), store.frame(unt)]
    assert_paired(frames, "Experiment D")
    per_sample = pd.concat(frames, ignore_index=True)
    seed_df = seed_level(per_sample)
    table = table_level(seed_df)
    stats = []
    for dataset, victims in DATASETS.items():
        for victim in victims:
            r = mcnemar_row("D", "D: single planned comparison (no Holm)", dataset, victim,
                            "Valid ASR", tgt.label, unt.label,
                            vectors(store.frame(tgt), dataset, victim, "valid_success"),
                            vectors(store.frame(unt), dataset, victim, "valid_success"))
            r["interpretation"] = mcnemar_text(r, r["p_value"], "McNemar")
            stats.append(r)
    stats_df = pd.DataFrame(stats)
    out = write_outputs("D", per_sample, seed_df, table, stats_df)
    vt = table[table.scope == "victim"]
    lo = {d: [tgt.label, unt.label] for d in DATASETS}
    per_dataset_bars(vt, "victim", "method", "raw_asr", "Exp D — Raw ASR: targeted vs untargeted (p75)",
                     "Raw ASR (%)", out / "plots" / "D1_raw_asr_targeted_vs_untargeted.png",
                     victims_order(), lo, ["#3182bd", "#e6550d"])
    per_dataset_bars(vt, "victim", "method", "valid_asr", "Exp D — Valid ASR: targeted vs untargeted (p75)",
                     "Valid ASR (%)", out / "plots" / "D2_valid_asr_targeted_vs_untargeted.png",
                     victims_order(), lo, ["#3182bd", "#e6550d"])
    rows = [{"Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Objective": r.method,
             "n/seed": r.n_per_seed, "Raw ASR": pm(r.raw_asr_mean, r.raw_asr_sd),
             "Valid ASR": pm(r.valid_asr_mean, r.valid_asr_sd),
             "Validity Gap": pm_pp(r.validity_gap_pp_mean, r.validity_gap_pp_sd),
             "Valid per seed (42/2024/2026, %)": seeds_str(r, "valid_asr")} for _, r in vt.iterrows()]
    st = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "n": r.n_paired,
                        "Targeted-only valid": r.A_only, "Untargeted-only valid": r.B_only,
                        "Δ Valid ASR (targeted − untargeted, pp)": f"{r.diff_pp:+.2f}",
                        "Test": r.variant, "Statistic": "" if r.statistic is None or pd.isna(r.statistic) else f"{r.statistic:.2f}",
                        "McNemar p": fmt_p(r.p_value),
                        "Δ seed 2024 / 2026 (pp)": f"{r.diff_pp_seed2024:+.2f} / {r.diff_pp_seed2026:+.2f}",
                        "Interpretation": r.interpretation} for _, r in stats_df.iterrows()])
    cw = table[table.scope == "class"]
    cw_md = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Class": r.source_class,
                           "Objective": r.method, "Raw": pm(r.raw_asr_mean, r.raw_asr_sd),
                           "Valid": pm(r.valid_asr_mean, r.valid_asr_sd)} for _, r in cw.iterrows()])
    lines = [
        "# Final Experiment D — Objective sensitivity (targeted → Benign vs untargeted)",
        "",
        f"PrimAttack with the selected optimizer ({OPT_LABEL[sel]}), joint mode, p75 budget, on "
        "identical flows and seeds. Targeted success = prediction == Benign. Untargeted success "
        "= prediction ≠ source class. Both arms use the same validator gate and the same "
        "incumbent rule. Only the objective margin differs. The targeted arm is the Exp B cell and "
        "the untargeted arm is the Exp A PrimAttack cell.",
        "",
        "## Results",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        "## Statistical analysis (one planned McNemar per (dataset, victim), no Holm)",
        "",
        STAT_METHOD_TEXT,
        "",
        md_table(st),
        "",
        "Raw targeted and raw untargeted ASR are descriptive only. No Raw-Success test is run.",
        "",
        "## Class-wise results",
        "",
        md_table(cw_md),
        "",
        "## Plots",
        "",
        "- `plots/D1_raw_asr_targeted_vs_untargeted.png`",
        "- `plots/D2_valid_asr_targeted_vs_untargeted.png`",
        "",
        "## Interpretation",
        "",
        include_interpretation(out, final),
        "",
    ]
    (out / EXP_DIRS["D"][1]).write_text("\n".join(lines), encoding="utf-8")
    return {"table": table, "stats": stats_df}


# ----------------------------------------------------------------------------- experiment E
def experiment_e(store: Store, selection: dict, a_conds: list[Cond], final: bool) -> dict:
    b_conds = [prim_cond("primattack_targeted_optimizers", m, "p75", "targeted",
                         f"{OPT_LABEL[m]} (targeted, p75)") for m in ("hybrid", "pgd", "cw")]
    groups = [("A (untargeted)", c) for c in a_conds] + [("B (targeted→Benign)", c) for c in b_conds]
    frames = []
    stats = []
    for exp_group, c in groups:
        f = store.frame(c).copy()
        f["experiment_group"] = exp_group
        frames.append(f)
        for dataset, victims in DATASETS.items():
            for victim in victims:
                fr = store.frame(c)
                r = mcnemar_row("E", f"E: {exp_group}", dataset, victim, "success",
                                "Raw", "Valid", vectors(fr, dataset, victim, "raw_success"),
                                vectors(fr, dataset, victim, "valid_success"))
                if r["B_only"] != 0:
                    raise AssertionError("valid success must be a subset of raw success")
                r["comparison"] = f"{c.label}: Raw vs Valid"
                r["condition"] = c.label
                r["raw_but_invalid"] = r["A_only"]
                r["valid_successes"] = r["both"]
                if r["A_only"] == 0:
                    r["interpretation"] = "No raw-success-but-invalid example: no validity gap at seed 42."
                elif r["p_value"] < ALPHA:
                    r["interpretation"] = (f"{r['A_only']} of {r['A_only'] + r['both']} raw successes "
                                           f"fail the validator ({r['diff_pp']:.2f} pp lost); the paired "
                                           f"loss is systematic (p = {fmt_p(r['p_value'])}).")
                else:
                    r["interpretation"] = (f"{r['A_only']} raw successes fail the validator "
                                           f"({r['diff_pp']:.2f} pp); not significant "
                                           f"(p = {fmt_p(r['p_value'])}).")
                stats.append(r)
    per_sample = pd.concat(frames, ignore_index=True)
    seed_df = seed_level(per_sample)
    table = table_level(seed_df)
    stats_df = pd.DataFrame(stats)
    out = write_outputs("E", per_sample, seed_df, table, stats_df)
    vt = table[table.scope == "victim"]
    labels = [c.label for _, c in groups]
    scatter_raw_valid(vt, labels, out / "plots" / "E1_raw_vs_valid_asr.png")
    per_dataset_bars(vt, "method", "victim", "validity_gap_pp", "Exp E — Validity gap by attack / method",
                     "Validity gap (pp)", out / "plots" / "E2_validity_gap_by_method.png",
                     {d: labels for d in DATASETS}, victims_order(), VICTIM_COLORS, percent=False)
    rows = []
    for group, c in groups:
        for dataset, victims in DATASETS.items():
            for victim in victims:
                r = vt[(vt.dataset == dataset) & (vt.victim == victim) & (vt.condition == c.key)].iloc[0]
                s = stats_df[(stats_df.dataset == dataset) & (stats_df.victim == victim)
                             & (stats_df.condition == c.label)].iloc[0]
                rows.append({"Group": group, "Dataset": DS_LABEL[dataset], "Victim": victim,
                             "Condition": c.label, "n (seed 42)": s.n_paired,
                             "Raw ASR": pm(r.raw_asr_mean, r.raw_asr_sd),
                             "Valid ASR": pm(r.valid_asr_mean, r.valid_asr_sd),
                             "Validity Gap": pm_pp(r.validity_gap_pp_mean, r.validity_gap_pp_sd),
                             "Raw-success-but-invalid (seed 42)": s.raw_but_invalid,
                             "Valid successes (seed 42)": s.valid_successes,
                             "Gap seed 42 (pp)": f"{s.diff_pp:.2f}",
                             "Test": s.variant.replace("McNemar ", ""),
                             "Statistic": "" if s.statistic is None or pd.isna(s.statistic) else f"{s.statistic:.1f}",
                             "McNemar p": fmt_p(s.p_value), "Interpretation": s.interpretation})
    cat_rows = []
    ref = per_sample[(per_sample.seed == REF_SEED) & per_sample.raw_success & ~per_sample.validator_pass]
    for group, c in groups:
        for dataset, victims in DATASETS.items():
            for victim in victims:
                g = ref[(ref.condition == c.key) & (ref.method == c.label) & (ref.dataset == dataset)
                        & (ref.victim == victim)]
                rec = {"group": group, "dataset": dataset, "victim": victim, "condition": c.label,
                       "raw_success_but_invalid": int(len(g))}
                for cat in ("schema", "extractor", "protocol", "mined"):
                    rec[f"share_failing_{cat}"] = float((~g[f"{cat}_pass"]).mean()) if len(g) else np.nan
                cat_rows.append(rec)
    cats = pd.DataFrame(cat_rows)
    cats.to_csv(out / "rejection_categories_of_invalid_successes.csv", index=False)
    cats_md = pd.DataFrame([{
        "Group": r.group, "Dataset": DS_LABEL[r.dataset], "Victim": r.victim, "Condition": r.condition,
        "Raw-success-but-invalid (seed 42)": r.raw_success_but_invalid,
        **{f"Failing {cat.upper()}": fnum(100 * r[f"share_failing_{cat}"], ".1f", "%")
           for cat in ("schema", "extractor", "protocol", "mined")}}
        for _, r in cats.iterrows() if r.raw_success_but_invalid > 0])
    lines = [
        "# Final Experiment E — Paired validity-gap analysis",
        "",
        "The research question: how much classifier-level attack success disappears when "
        "domain validity is required, and is that paired loss systematic? For every main "
        "condition, Raw Success and Valid Success are two binary outcomes of the **same** final "
        "adversarial example. Valid ⊆ Raw, so the only possible discordant cell is raw success = 1, "
        "valid success = 0 (fools the classifier, fails validator_v2). McNemar's test on that "
        "paired table is the one test where Raw Success is tested directly. There is no "
        "Cochran's Q. Each (condition, dataset, victim) test stands alone, organized by the "
        "experiment it belongs to, with no cross-thesis correction.",
        "",
        STAT_METHOD_TEXT,
        "",
        "## Paired validity gap per condition",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        "## Why raw successes are rejected (descriptive)",
        "",
        "For every raw-success-but-invalid example at seed 42: the share that violates at least "
        "one rule of each validator_v2 category. An example can fail several categories. SCHEMA, "
        "EXTRACTOR and PROTOCOL are general flow-consistency rules. MINED rules are train-mined, "
        "dataset-specific invariants. Also in `rejection_categories_of_invalid_successes.csv`.",
        "",
        md_table(cats_md),
        "",
        "## Plots",
        "",
        "- `plots/E1_raw_vs_valid_asr.png` — Raw ASR vs Valid ASR (points below the diagonal = gap)",
        "- `plots/E2_validity_gap_by_method.png` — Validity gap by attack / method",
        "",
        "## Interpretation",
        "",
        include_interpretation(out, final),
        "",
    ]
    (out / EXP_DIRS["E"][1]).write_text("\n".join(lines), encoding="utf-8")
    return {"table": table, "stats": stats_df, "groups": groups}


def scatter_raw_valid(vt: pd.DataFrame, labels: list[str], path: Path) -> None:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6.5 * len(DATASETS), 5.2), squeeze=False)
    for ax, dataset in zip(axes[0], DATASETS):
        for mi, lab in enumerate(labels):
            for vi, victim in enumerate(DATASETS[dataset]):
                t = vt[(vt.dataset == dataset) & (vt.victim == victim) & (vt.method == lab)]
                if not len(t):
                    continue
                r = t.iloc[0]
                ax.errorbar(100 * r.raw_asr_mean, 100 * r.valid_asr_mean, xerr=100 * r.raw_asr_sd,
                            yerr=100 * r.valid_asr_sd, marker=markers[mi % len(markers)],
                            color=VICTIM_COLORS[vi], linestyle="none", capsize=2,
                            label=None)
        ax.plot([0, 100], [0, 100], color="grey", linewidth=0.8)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xlabel("Raw ASR (%)")
        ax.set_ylabel("Valid ASR (%)")
        ax.set_title(f"{DS_LABEL[dataset]} (colour = victim: " + ", ".join(DATASETS[dataset]) + ")",
                     fontsize=8)
        ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker=markers[i % len(markers)], color="black",
                          linestyle="none", label=lab) for i, lab in enumerate(labels)]
    handles += [plt.Line2D([], [], marker="s", color=VICTIM_COLORS[i], linestyle="none",
                           label=f"victim {i + 1}: " + " / ".join(v[i] for v in DATASETS.values()))
                for i in range(3)]
    axes[0][0].legend(handles=handles, fontsize=6, loc="upper left")
    fig.suptitle("Exp E — Raw vs Valid ASR of the same adversarial examples (mean ± SD over seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------- experiment F
def experiment_f(final: bool) -> dict:
    out = FINAL / EXP_DIRS["F"][0]
    out.mkdir(parents=True, exist_ok=True)
    acc_rows, cat_rows, rule_rows, class_rows, rule_inventory = [], [], [], [], []
    per_flow = []
    for dataset in DATASETS:
        adapter = get_adapter(dataset)
        v = load_validator(dataset)
        names = list(adapter.class_mapping().names)
        counts = pd.Series([r.source_type for r in v.rules]).value_counts()
        for st in ("SCHEMA", "EXTRACTOR", "PROTOCOL", "MINED"):
            rule_inventory.append({"dataset": dataset, "source_type": st,
                                   "layer": "dataset-specific (train-mined)" if st == "MINED"
                                   else "general flow consistency",
                                   "n_rules": int(counts.get(st, 0))})
        for split in ("val", "test"):
            X = np.asarray(np.load(adapter._processed / f"X_{split}_pristine.npy", mmap_mode="r"))
            y = np.load(adapter._processed / f"y_{split}_cat.npy").astype(np.int64)
            b = v.validate_batch(X)
            n = int(b.n)
            general = b.hard_structural_valid
            ids = pd.read_parquet(adapter._processed / f"{split}.parquet",
                                  columns=["sample_id"])["sample_id"].astype(str).to_numpy()
            if len(ids) != n or len(y) != n:
                raise AssertionError(f"{dataset}/{split}: row count mismatch across arrays")
            per_flow.append(pd.DataFrame({
                "dataset": dataset, "split": split, "sample_id": ids,
                "class": np.asarray(names)[y], "schema_pass": b.schema_valid,
                "extractor_pass": b.extractor_valid, "protocol_pass": b.protocol_valid,
                "mined_pass": b.mined_valid, "general_only_pass": general,
                "hybrid_valid": b.hybrid_valid}))
            hybrid = b.hybrid_valid
            for layer, mask in (("general only (SCHEMA ∧ EXTRACTOR ∧ PROTOCOL)", general),
                                ("general + dataset-specific (hybrid_valid)", hybrid)):
                acc_rows.append({"dataset": dataset, "split": split, "validator": layer,
                                 "n_genuine": n, "accepted": int(mask.sum()),
                                 "rejected": int((~mask).sum()),
                                 "acceptance_rate": float(mask.mean()),
                                 "rejection_rate": float((~mask).mean())})
            groups = {"SCHEMA": b.schema_valid, "EXTRACTOR": b.extractor_valid,
                      "PROTOCOL": b.protocol_valid, "MINED": b.mined_valid}
            for st, mask in groups.items():
                others_accept = np.all([m for s2, m in groups.items() if s2 != st], axis=0)
                cat_rows.append({"dataset": dataset, "split": split, "rejection_category": st,
                                 "n_genuine": n, "rejected_by_category": int((~mask).sum()),
                                 "rate": float((~mask).mean()),
                                 "rejected_only_by_this_category": int(
                                     ((~mask) & others_accept).sum())})
            by_id = {r.id: r for r in v.rules}
            for rid, c in sorted(b.rules_rejecting_any().items(), key=lambda kv: -kv[1]):
                r = by_id[rid]
                rule_rows.append({"dataset": dataset, "split": split, "rule_id": rid,
                                  "source_type": r.source_type, "expression": r.expression(),
                                  "rejected": int(c), "rate": c / n})
            for cid, cname in enumerate(names):
                m = y == cid
                if not m.any():
                    continue
                class_rows.append({"dataset": dataset, "split": split, "class": cname,
                                   "n_genuine": int(m.sum()),
                                   "general_only_acceptance": float(general[m].mean()),
                                   "hybrid_acceptance": float(hybrid[m].mean()),
                                   "hybrid_rejected": int((~hybrid[m]).sum())})
    acc, cat, rules, cls, inv = map(pd.DataFrame, (acc_rows, cat_rows, rule_rows, class_rows,
                                                    rule_inventory))
    acc.to_csv(out / "validator_acceptance.csv", index=False)
    pd.concat(per_flow, ignore_index=True).to_parquet(out / "per_sample.parquet", index=False)
    cat.to_csv(out / "rejection_categories.csv", index=False)
    rules.to_csv(out / "rule_rejections.csv", index=False)
    cls.to_csv(out / "classwise_acceptance.csv", index=False)
    inv.to_csv(out / "rule_inventory.csv", index=False)
    acc_md = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Split": r.split, "Validator": r.validator,
                            "Genuine flows": f"{r.n_genuine:,}", "Accepted": f"{r.accepted:,}",
                            "Rejected": f"{r.rejected:,}",
                            "Acceptance": f"{100 * r.acceptance_rate:.4f}%",
                            "Rejection": f"{100 * r.rejection_rate:.4f}%"} for _, r in acc.iterrows()])
    cat_md = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Split": r.split,
                            "Rejection category": r.rejection_category,
                            "Rejected (≥1 rule of category)": r.rejected_by_category,
                            "Rate": f"{100 * r.rate:.4f}%",
                            "Rejected only by this category": r.rejected_only_by_this_category}
                           for _, r in cat.iterrows()])
    rules_md = (pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Split": r.split, "Rule": r.rule_id,
                               "Category": r.source_type, "Expression": f"`{r.expression}`",
                               "Rejected": r.rejected, "Rate": f"{100 * r.rate:.4f}%"}
                              for _, r in rules.iterrows()])
                if len(rules) else pd.DataFrame([{"Note": "no rule rejects any genuine flow"}]))
    cls_md = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Split": r.split, "Class": r["class"],
                            "Genuine flows": f"{r.n_genuine:,}",
                            "General-only acceptance": f"{100 * r.general_only_acceptance:.4f}%",
                            "General + dataset-specific acceptance": f"{100 * r.hybrid_acceptance:.4f}%",
                            "Rejected (hybrid)": r.hybrid_rejected} for _, r in cls.iterrows()])
    inv_md = pd.DataFrame([{"Dataset": DS_LABEL[r.dataset], "Category": r.source_type,
                            "Layer": r.layer, "Rules": r.n_rules} for _, r in inv.iterrows()])
    lines = [
        "# Final Experiment F — Validator evaluation (descriptive)",
        "",
        "validator_v2 is the independent domain validator used by every experiment. It combines "
        "general network-flow consistency constraints (SCHEMA domain/type rules, CICFlowMeter "
        "EXTRACTOR identities, PROTOCOL rules) with automatically mined, train-only "
        "dataset-specific invariants (MINED). This experiment applies it to **every genuine "
        "held-out flow** of both datasets (validation and test splits, all classes including "
        "Benign). Neither split was used for rule mining or tolerance fitting. No hypothesis test "
        "is used. The results are counts and percentages.",
        "",
        "## Rule inventory",
        "",
        md_table(inv_md),
        "",
        "## Acceptance / rejection of genuine held-out flows",
        "",
        md_table(acc_md),
        "",
        "## Rejection categories",
        "",
        "A flow counts under a category when it violates at least one rule of that category. "
        "`Rejected only by this category` counts flows that every other category accepts.",
        "",
        md_table(cat_md),
        "",
        "## Rules that reject genuine flows",
        "",
        md_table(rules_md),
        "",
        "## Class-wise acceptance",
        "",
        md_table(cls_md),
        "",
        "## Machine-readable outputs",
        "",
        "`per_sample.parquet` (one verdict row per genuine flow, per category), "
        "`validator_acceptance.csv`, `rejection_categories.csv`, `rule_rejections.csv`, "
        "`classwise_acceptance.csv`, `rule_inventory.csv`.",
        "",
        "## Interpretation",
        "",
        include_interpretation(out, final),
        "",
    ]
    (out / EXP_DIRS["F"][1]).write_text("\n".join(lines), encoding="utf-8")
    return {"acceptance": acc, "categories": cat, "rules": rules}


# ----------------------------------------------------------------------------- summary
def write_summary(store: Store, selection: dict, a: dict, b: dict, c: dict, d: dict, e: dict,
                  f: dict, final: bool) -> None:
    at = a["table"]
    n_lo = int(at[at.scope == "class"].n_per_seed.min())
    n_hi = int(at[at.scope == "class"].n_per_seed.max())
    n_txt = f"{n_lo}" if n_lo == n_hi else f"{n_lo}–{n_hi}"
    vt = at[(at.scope == "victim") & ~at.method.str.contains("unbounded")]
    head = []
    for dataset, victims in DATASETS.items():
        for victim in victims:
            rec = {"Dataset": DS_LABEL[dataset], "Victim": victim}
            for _, r in vt[(vt.dataset == dataset) & (vt.victim == victim)].iterrows():
                rec[r.method] = f"{pm(r.raw_asr_mean, r.raw_asr_sd)} → {pm(r.valid_asr_mean, r.valid_asr_sd)}"
            head.append(rec)
    n_tests = {k: int(len(v["stats"].dropna(subset=["p_value"]))) for k, v in
               (("A", a), ("B", b), ("C", c), ("D", d), ("E", e))}
    audit = store.audit
    lines = [
        "# Final experiment summary",
        "",
        "Bachelor's thesis: constrained adversarial attacks against NIDS classifiers. This is the "
        "definitive final suite defined in `master_experiments.md`, run under the locked protocol "
        "`00_PROTOCOL.md`. Every number below is regenerated from `runs/` by "
        "`scripts/analyze_final_suite.py`. All results are feature-space proxies on CICFlowMeter "
        "aggregates. No PCAP is edited or replayed, and no packet-level realizability or complete "
        "malicious functionality is claimed.",
        "",
        "## Reports",
        "",
        "| Experiment | Report |",
        "|---|---|",
        "| A — primary baseline comparison (untargeted) | `A_primary_baseline_comparison/primary_baseline_comparison.md` |",
        "| B — PrimAttack optimizer selection | `B_optimizer_selection/primattack_optimizer_selection.md` |",
        "| C — budget sensitivity | `C_budget_sensitivity/primattack_budget_sensitivity.md` |",
        "| D — objective sensitivity | `D_objective_sensitivity/objective_sensitivity.md` |",
        "| E — paired validity gap | `E_paired_validity_gap/paired_validity_gap_analysis.md` |",
        "| F — validator evaluation | `F_validator_evaluation/validator_evaluation.md` |",
        "",
        "## Scope",
        "",
        "- Datasets: CICIDS2017-DistriNet, CSE-CIC-IDS-2018-DistriNet.",
        "- Victims: MLP, CNN and FT-Transformer category classifiers, one frozen checkpoint per "
        "architecture and dataset (2018: training-seed-42 replicates).",
        f"- Source classes: DoS, DDoS, Recon, BruteForce. There are {n_txt} canonical "
        "clean-correct test flows per (dataset, victim, class), identical for every method.",
        "- Seeds 42, 2024 and 2026 are attack seeds used for every run.",
        "- Metrics: Raw ASR, Valid ASR (success ∧ validator_v2 `hybrid_valid`, same "
        "denominator) and Validity Gap = Raw − Valid (pp).",
        "",
        "## Statistical methodology",
        "",
        "- mean ± SD across seeds 42, 2024 and 2026 for run-to-run variability;",
        "- paired sample-level binary inference (one outcome per source flow: reference seed 42; "
        "seeds are never treated as independent observations and seed means are never the "
        "statistical sample);",
        "- Cochran's Q for 3+ paired conditions (Exp A: 5 attacks; Exp B: 3 optimizers; Exp C: "
        "3 budgets);",
        "- planned McNemar comparisons for two-condition contrasts (A: PrimAttack vs each baseline; "
        "B: 3 optimizer pairs; C: adjacent budgets; D: targeted vs untargeted; E: raw vs valid);",
        "- Holm correction only within logical families of multiple planned McNemar comparisons "
        "(A: 4, B: 3, C: 2 per dataset × victim);",
        "- alpha = 0.05;",
        "- Valid Success is the primary inferential outcome;",
        "- Raw ASR is descriptive except in the dedicated validity-gap analysis (E);",
        "- F is descriptive (no test).",
        "",
        f"Inferential tests actually computed: A {n_tests['A']}, B {n_tests['B']}, C {n_tests['C']}, "
        f"D {n_tests['D']}, E {n_tests['E']} (see each `statistical_tests.csv`).",
        "",
        "## Headline: Experiment A (Raw ASR → Valid ASR, untargeted, mean ± SD)",
        "",
        md_table(pd.DataFrame(head)),
        "",
        f"Selected PrimAttack optimizer (Exp B, pre-registered aggregate-Valid-Targeted-ASR rule): "
        f"**{OPT_LABEL[selection['selected']]}**. Ranking: "
        f"{' > '.join(OPT_LABEL[m] for m in selection['ranking'])}.",
        "",
        "## Thesis contribution mapping",
        "",
        "| Contribution | Evidence |",
        "|---|---|",
        "| 1. PrimAttack: constrained adversarial attack framework over attacker-controllable "
        "packet-size and timing primitives | Exp A (`primary_baseline_comparison.md`), Exp B "
        "(`primattack_optimizer_selection.md`) |",
        "| 2. Evaluation methodology separating attack objective, domain validity and "
        "constraint-valid success under controlled perturbation conditions | Exp A, Exp C "
        "(`primattack_budget_sensitivity.md`), Exp D (`objective_sensitivity.md`) |",
        "| 3. Paired validity-gap analysis (raw vs constraint-valid success on identical source "
        "samples) | `paired_validity_gap_analysis.md`; paired Raw/Valid/Gap columns in every "
        "report |",
        "| 4. Independent domain-validation framework (general flow-consistency + automatically "
        "derived dataset-specific constraints) | `validator_evaluation.md` |",
        "| 5. Controlled PrimAttack perturbation-budget analysis | `primattack_budget_sensitivity.md` |",
        "",
        "## Provenance and integrity audit",
        "",
        f"- Per-sample artifacts read: {audit['npz_files']} files, {audit['rows']:,} attacked "
        "flow-instances.",
        f"- validator_v2 recomputed on the stored final adversarial flow for "
        f"{audit['validator_rechecked_rows']:,} rows: 0 mismatches (analysis aborts on any).",
        f"- Raw/valid success recomputed from stored predictions for "
        f"{audit['raw_success_recomputed_rows']:,} rows: 0 mismatches (analysis aborts on any).",
        f"- Victim re-prediction of the stored final flows: {audit['prediction_rechecked_rows']:,} "
        f"rows, {audit['prediction_mismatches']} mismatches"
        + (" (batch-composition float noise at the decision boundary; the stored run-time "
           "predictions are reported)." if audit["prediction_mismatches"] else "."),
        "- Pairing asserted per experiment: identical canonical sample IDs (order, no "
        "duplicates), clean-input SHA-256, labels, clean predictions, victim checkpoint SHA-256, "
        "seed set {42, 2024, 2026} and equal denominators.",
        "- Run configurations with every hyperparameter: `runs/final_suite_config.json`, "
        "`runs/<dataset>/<stage>/config.json`; run logs: `runs/<dataset>/logs/`.",
        "",
        "## Interpretation",
        "",
        include_interpretation(FINAL, final),
        "",
    ]
    (FINAL / "final_experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-recheck-predictions", action="store_true")
    ap.add_argument("--final", action="store_true",
                    help="require every hand-written interpretation.md")
    args = ap.parse_args()
    selection = json.loads((RUNS / "optimizer_selection.json").read_text(encoding="utf-8"))
    store = Store(recheck_predictions=not args.no_recheck_predictions, device=args.device)
    b = experiment_b(store, selection, args.final)
    a = experiment_a(store, selection, args.final)
    c = experiment_c(store, selection, args.final)
    d = experiment_d(store, selection, args.final)
    e = experiment_e(store, selection, [a["prim"]] + a["bases"], args.final)
    f = experiment_f(args.final)
    write_summary(store, selection, a, b, c, d, e, f, args.final)
    (FINAL / "analysis_audit.json").write_text(json.dumps(store.audit, indent=2), encoding="utf-8")
    print(json.dumps(store.audit, indent=2))


if __name__ == "__main__":
    main()
