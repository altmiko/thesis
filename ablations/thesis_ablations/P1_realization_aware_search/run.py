"""P1 - Realization-aware search vs continuous-state search (``NoRealizedSearch``).

Question: does scoring and selecting REALIZED integer primitive states during the search
contribute to the final Valid ASR?

* ``reference``: the FINAL-suite Hybrid Search, targeted -> Benign. Every candidate (identity,
  exact padding enumeration, each refinement iterate) is projected to integer bytes / µs inside
  the per-flow box and M(x), mapped through φ with quantization, scored by the victim, and the
  incumbent (success first, then lowest normalized cost; best margin among failures) is
  selected on these realized flows.
* ``no_realized_search`` (``HybridConfig(realization_aware_search=False)``): the same optimizer
  (padding enumeration, 40-step adaptive refinement, momentum, step halving, restarts until the
  budget is spent, same gradient on the same relaxation), but every candidate is scored on the
  CONTINUOUS primitive state: controls clamped to the continuous box ``[0, bounds]`` and M(x),
  φ without quantization (``ablations.common.hybrid.ContinuousSearch``). The search success is
  the objective hit on that continuous flow, and the step-size checkpoints, restart bests and
  incumbent use the continuous margin and cost. Nothing is rounded during the search. The
  returned continuous candidate is realized exactly once by the canonical code path
  (``RealizedSearch._score``: ``project_controls`` -> integer bytes / µs, box, M(x) -> φ
  ``quantize=True`` -> victim) and this realized flow is what Raw / Valid ASR count.

Budget: both arms have at most 256 victim evaluations per flow. The continuous search runs with
255; the final realization is the 256th. validator_v2 is not in the continuous search's success
predicate (it judges realized, integer flows; amendment A6 showed the gate never changed a FINAL
flow). Both arms' final flows are scored by the same full validator_v2 (source-conditioned).

The shared ablation reference arm (``ablations/reference``) is untargeted, so P1 runs its own
targeted reference arm and checks it flow-for-flow against the FINAL targeted Hybrid cells
(``primattack_hybrid_objective_targeted`` p75, ``primattack_targeted_budgets`` unbounded).

    python ablations/thesis_ablations/P1_realization_aware_search/run.py --device cuda
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from ablations.common.analysis import REF_SEED, BUDGET_ORDER, load_cells, pct  # noqa: E402
from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.hybrid import HybridConfig  # noqa: E402
from ablations.common.runner import (  # noqa: E402
    CLASSES, DATASETS, FINAL_RUNS, REFERENCE, Condition, _sha_ids,
)
from attack.primattack_budget import (  # noqa: E402
    class_calibration, load_calibration, unbounded_calibration,
)
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from datasets import get_adapter  # noqa: E402
from evaluation.paired_validity_gap import holm_adjust, mcnemar_test, newcombe_paired_ci  # noqa: E402
from run_full_adversarial_eval import DATASET_DEFAULTS  # noqa: E402
from validation.attack_interface import structural_masks  # noqa: E402

OBJECTIVE = "targeted"
A3 = "no_realized_search"
CONDITIONS = [
    REFERENCE,
    Condition(A3, "Hybrid Search scored and selected on the continuous primitive state; the "
              "returned candidate is realized once (integer bytes/µs, box, M(x), quantized φ)",
              HybridConfig(validity_in_search=False, realization_aware_search=False)),
]
EVAL_CAP = 256
FINAL_STAGE = {"p75": "primattack_hybrid_objective_targeted", "unb": "primattack_targeted_budgets"}
BUDGET_NAME = {"p75": "maximum-evaluated", "unb": "unbounded"}

COMMON = ("sample_id", "positional_idx", "adv_pred", "raw_success", "valid_success",
          "validator_pass", "search_success", "schema_valid", "extractor_valid",
          "protocol_valid", "mined_valid", "p", "delay", "shape", "requested_p",
          "requested_delay", "requested_shape", "normalized_cost", "relative_duration_change",
          "added_bytes", "total_evaluations", "realized_evaluations", "surrogate_evaluations",
          "pad_allowed", "timing_allowed", "p_hi", "delay_hi", "row_primitive_mode",
          "objective_margin")
CONTINUOUS = ("continuous_success", "continuous_pred", "continuous_margin", "continuous_cost",
              "continuous_p", "continuous_delay", "continuous_shape",
              "continuous_validator_pass", "continuous_schema_valid",
              "continuous_extractor_valid", "continuous_protocol_valid",
              "continuous_mined_valid", "continuous_candidate_evaluations",
              "continuous_gradient_evaluations", "final_realization_evaluations")


# ----------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------
def _npz(results_dir: Path, dataset: str, victim: str, cname: str, budget: str, cond: str,
         seed: int) -> Path:
    return results_dir / dataset / "artifacts" / f"{victim}__{cname}__{budget}__{cond}__seed{seed}.npz"


def _cells(results_dir: Path) -> list[tuple[str, str, str, int]]:
    cells = load_cells(results_dir)
    keys = cells[["dataset", "victim", "budget_label", "seed"]].drop_duplicates()
    out = []
    for k in keys.itertuples(index=False):
        if all(_npz(results_dir, k.dataset, k.victim, c, k.budget_label, cond, k.seed).exists()
               for c in CLASSES for cond in (REFERENCE.name, A3)):
            out.append((k.dataset, k.victim, k.budget_label, int(k.seed)))
    victim_order = {v: i for spec in DATASETS.values() for i, v in enumerate(spec["victims"])}
    return sorted(out, key=lambda k: (k[0], victim_order.get(k[1], 9), BUDGET_ORDER[k[2]], k[3]))


def load_pair(results_dir: Path, dataset: str, victim: str, budget: str, seed: int):
    """Per-flow arrays of both arms, the four classes concatenated in ``CLASSES`` order."""
    arms = {}
    for cond, fields in ((REFERENCE.name, COMMON + ("adv_raw",)),
                         (A3, COMMON + ("adv_raw",) + CONTINUOUS)):
        parts = []
        for cname in CLASSES:
            with np.load(_npz(results_dir, dataset, victim, cname, budget, cond, seed),
                         allow_pickle=True) as z:
                d = {f: z[f] for f in fields}
            d["attack_class"] = np.full(len(d["sample_id"]), cname)
            parts.append(d)
        arms[cond] = {f: np.concatenate([p[f] for p in parts]) for f in parts[0]}
    return arms[REFERENCE.name], arms[A3]


def _median(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(x)) if x.size else float("nan")


# ----------------------------------------------------------------------------------------
# per-flow / per-seed tables
# ----------------------------------------------------------------------------------------
def per_flow_frame(dataset, victim, budget, seed, ref, a3) -> pd.DataFrame:
    df = pd.DataFrame({
        "dataset": dataset, "victim": victim, "budget": budget, "seed": seed,
        "attack_class": ref["attack_class"], "sample_id": ref["sample_id"],
        "positional_idx": ref["positional_idx"], "row_primitive_mode": ref["row_primitive_mode"],
        "pad_allowed": ref["pad_allowed"], "timing_allowed": ref["timing_allowed"],
        "p_hi": ref["p_hi"], "delay_hi": ref["delay_hi"],
    })
    for tag, arm in (("ref", ref), ("p1", a3)):
        for f in ("adv_pred", "raw_success", "validator_pass", "valid_success", "p", "delay",
                  "shape", "normalized_cost", "relative_duration_change", "total_evaluations",
                  "objective_margin"):
            df[f"{tag}_{f}"] = arm[f]
    for f in ("continuous_success", "continuous_pred", "continuous_validator_pass",
              "continuous_p", "continuous_delay", "continuous_shape", "continuous_cost",
              "continuous_margin", "continuous_candidate_evaluations",
              "continuous_gradient_evaluations", "final_realization_evaluations"):
        df[f"p1_{f}"] = a3[f]
    df["p1_prediction_changed_by_realization"] = a3["continuous_pred"] != a3["adv_pred"]
    return df


def per_seed_row(dataset, victim, budget, seed, ref, a3) -> dict:
    rv, av = ref["valid_success"].astype(bool), a3["valid_success"].astype(bool)
    rr, ar = ref["raw_success"].astype(bool), a3["raw_success"].astype(bool)
    ch = a3["continuous_success"].astype(bool)
    apass = a3["validator_pass"].astype(bool)
    n = len(rv)
    return {
        "dataset": dataset, "victim": victim, "budget": budget, "seed": seed, "n": n,
        "ref_raw_asr": rr.mean(), "ref_valid_asr": rv.mean(),
        "p1_continuous_asr": ch.mean(), "p1_raw_asr": ar.mean(), "p1_valid_asr": av.mean(),
        "delta_valid_pp": 100 * (av.mean() - rv.mean()),
        "delta_raw_pp": 100 * (ar.mean() - rr.mean()),
        "continuous_hits": int(ch.sum()),
        "continuous_hit_realized_miss": int((ch & ~ar).sum()),
        "continuous_hit_realized_hit_invalid": int((ch & ar & ~apass).sum()),
        "continuous_hit_realized_valid": int((ch & av).sum()),
        "continuous_miss_realized_hit": int((~ch & ar).sum()),
        "prediction_changed_by_realization": int((a3["continuous_pred"] != a3["adv_pred"]).sum()),
        "continuous_validator_pass_rate": a3["continuous_validator_pass"].mean(),
        # What the single final rounding changes (P1): controls, margin, validity.
        "continuous_hits_validator_pass": int((ch & a3["continuous_validator_pass"]).sum()),
        "continuous_hits_schema_pass": int((ch & a3["continuous_schema_valid"]).sum()),
        "max_abs_delay_rounding_us": float(np.abs(a3["continuous_delay"] - a3["delay"]).max()),
        "max_abs_p_rounding_bytes": float(np.abs(a3["continuous_p"] - a3["p"]).max()),
        "max_abs_margin_change": float(
            np.abs(a3["continuous_margin"] - a3["objective_margin"]).max()),
        "min_abs_continuous_margin_hits": (float(np.abs(a3["continuous_margin"][ch]).min())
                                           if ch.any() else float("nan")),
        "final_flow_identical": int(np.all(a3["adv_raw"] == ref["adv_raw"], axis=1).sum()),
        "ref_valid_final_flow_identical": int(
            (np.all(a3["adv_raw"] == ref["adv_raw"], axis=1) & rv).sum()),
        "ref_validator_pass_among_raw": apass_rate(ref),
        "p1_validator_pass_among_raw": apass_rate(a3),
        "valid_p1_only": int((av & ~rv).sum()), "valid_ref_only": int((~av & rv).sum()),
        "valid_both": int((av & rv).sum()),
        "raw_p1_only": int((ar & ~rr).sum()), "raw_ref_only": int((~ar & rr).sum()),
        "ref_valid_cost_median": _median(ref["normalized_cost"][rv]),
        "p1_valid_cost_median": _median(a3["normalized_cost"][av]),
        "ref_valid_delay_us_median": _median(ref["delay"][rv]),
        "p1_valid_delay_us_median": _median(a3["delay"][av]),
        "ref_valid_rel_duration_median": _median(ref["relative_duration_change"][rv]),
        "p1_valid_rel_duration_median": _median(a3["relative_duration_change"][av]),
        "ref_evals_mean": ref["total_evaluations"].mean(),
        "ref_evals_max": int(ref["total_evaluations"].max()),
        "p1_evals_mean": a3["total_evaluations"].mean(),
        "p1_evals_max": int(a3["total_evaluations"].max()),
        "p1_continuous_candidate_evals_mean": a3["continuous_candidate_evaluations"].mean(),
        "p1_gradient_evals_mean": a3["continuous_gradient_evaluations"].mean(),
        "p1_final_realization_evals_mean": a3["final_realization_evaluations"].mean(),
        "ref_realized_evals_mean": ref["realized_evaluations"].mean(),
        "ref_gradient_evals_mean": ref["surrogate_evaluations"].mean(),
    }


def apass_rate(arm) -> float:
    raw = arm["raw_success"].astype(bool)
    return float(arm["validator_pass"][raw].mean()) if raw.any() else float("nan")


# ----------------------------------------------------------------------------------------
# sanity checks
# ----------------------------------------------------------------------------------------
class Realizer:
    """Recomputes the canonical realization of stored controls (integer projection + φ)."""

    def __init__(self, dataset: str, device: str) -> None:
        adapter = get_adapter(DATASETS[dataset]["cli"])
        self.device = device
        self.model = CICIDS2017PrimitiveModel(adapter.feature_manifest())
        self.raw_all = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
        self.calibration = load_calibration(DATASET_DEFAULTS[dataset]["calibration"])
        self.dataset = dataset

    def check(self, arm: dict, budget: str) -> dict:
        out = {"flows": 0, "controls_integer": 0, "controls_equal_projection": 0,
               "adv_equal_recomputed": 0, "validator_equal_recomputed": 0,
               "capability_equal_recomputed": 0}
        for cname in CLASSES:
            m = arm["attack_class"] == cname
            idx = arm["positional_idx"][m]
            raw = torch.tensor(np.ascontiguousarray(self.raw_all[idx]), dtype=torch.float32,
                               device=self.device)
            caps = self.model.infer_capabilities(raw)
            ccfg = (unbounded_calibration(self.calibration, cname) if budget == "unb"
                    else class_calibration(self.calibration, cname, BUDGET_NAME[budget]))
            bounds = self.model.per_flow_bounds(raw, ccfg.bounds_config(), capabilities=caps)
            t = lambda k: torch.as_tensor(arm[k][m], dtype=torch.float32, device=self.device)  # noqa: E731
            requested = {"p": t("requested_p"), "delay": t("requested_delay"),
                         "shape": t("requested_shape")}
            stored = {"p": t("p"), "delay": t("delay"), "shape": t("shape")}
            with torch.no_grad():
                proj = self.model.project_controls(raw, requested, bounds, capabilities=caps)
                adv = self.model.generate(raw, stored, quantize=True, capabilities=caps)
            p, d = arm["p"][m], arm["delay"][m]
            out["flows"] += int(m.sum())
            out["controls_integer"] += int(((p == np.round(p)) & (d == np.round(d))).sum())
            out["controls_equal_projection"] += int(torch.stack(
                [proj[k] == stored[k] for k in ("p", "delay", "shape")]).all(0).sum())
            out["adv_equal_recomputed"] += int(
                (adv.cpu().numpy() == arm["adv_raw"][m]).all(1).sum())
            valid = structural_masks(arm["adv_raw"][m], dataset=self.dataset,
                                     source_raw=raw.cpu().numpy())["hybrid_valid"]
            out["validator_equal_recomputed"] += int(
                (np.asarray(valid, bool) == arm["validator_pass"][m].astype(bool)).sum())
            out["capability_equal_recomputed"] += int(
                ((caps.pad_allowed.cpu().numpy() == arm["pad_allowed"][m])
                 & (caps.timing_allowed.cpu().numpy() == arm["timing_allowed"][m])).sum())
        return out


def reproduction_vs_final(dataset, victim, budget, seed, ref) -> dict:
    art = FINAL_RUNS / dataset / FINAL_STAGE[budget] / "artifacts"
    out = {"flows": 0, "adv_identical": 0, "pred_identical": 0, "valid_identical": 0,
           "valid_success_identical": 0, "missing": 0}
    for cname in CLASSES:
        path = art / f"{victim}__{cname}__{budget}__hybrid__seed{seed}.npz"
        m = ref["attack_class"] == cname
        if not path.exists():
            out["missing"] += 1
            continue
        with np.load(path, allow_pickle=True) as b:
            n = int(m.sum())  # < 800 only in --limit-rows smoke runs (FINAL row order is kept)
            if not np.array_equal(ref["sample_id"][m], b["sample_id"][:n]):
                raise AssertionError(f"{path.name}: sample order differs from FINAL")
            out["flows"] += n
            out["adv_identical"] += int(
                np.all(ref["adv_raw"][m] == b["adv_raw"][:n], axis=1).sum())
            out["pred_identical"] += int((ref["adv_pred"][m] == b["adv_pred"][:n]).sum())
            out["valid_identical"] += int(
                (ref["validator_pass"][m] == b["validator_pass"][:n]).sum())
            out["valid_success_identical"] += int(
                (ref["valid_success"][m] == b["valid_success"][:n]).sum())
    return out


def frozen_ids_ok(dataset: str, victim: str, ids: np.ndarray, classes: np.ndarray) -> bool:
    sel = json.loads((FINAL_RUNS / dataset / "baselines_untargeted" / "selection.json")
                     .read_text(encoding="utf-8"))
    return all(_sha_ids(ids[classes == c]) == sel[victim][c]["sha256_sample_ids"] for c in CLASSES)


# ----------------------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------------------
def _mean_sd(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()), float(x.std(ddof=1)) if x.size > 1 else float("nan")


def _pm(m: float, s: float, scale: float = 100.0, digits: int = 2) -> str:
    if m != m:
        return "n/a"
    return f"{scale * m:.{digits}f} ± {scale * s:.{digits}f}" if s == s else f"{scale * m:.{digits}f}"


def analyze(results_dir: Path, _reference_dir: Path) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = _cells(results_dir)
    if not keys:
        raise FileNotFoundError(f"no complete P1 cells under {results_dir}")
    flows, seeds_rows, sanity, tests = [], [], [], []
    realizers: dict[str, Realizer] = {}
    for dataset, victim, budget, seed in keys:
        ref, a3 = load_pair(results_dir, dataset, victim, budget, seed)
        if not np.array_equal(ref["sample_id"], a3["sample_id"]):
            raise AssertionError(f"{dataset}/{victim}/{budget}/{seed}: arms not aligned")
        flows.append(per_flow_frame(dataset, victim, budget, seed, ref, a3))
        seeds_rows.append(per_seed_row(dataset, victim, budget, seed, ref, a3))
        realizer = realizers.setdefault(dataset, Realizer(dataset, device))
        full = len(ref["sample_id"]) == 800 * len(CLASSES)
        chk = {"dataset": dataset, "victim": victim, "budget": budget, "seed": seed,
               "flows": len(ref["sample_id"]),
               "same_source_ids": bool(np.array_equal(ref["sample_id"], a3["sample_id"])),
               "frozen_selection_sha256_ok": (frozen_ids_ok(dataset, victim, ref["sample_id"],
                                                            ref["attack_class"])
                                              if full else None),
               "same_capability": bool(np.array_equal(ref["pad_allowed"], a3["pad_allowed"])
                                       and np.array_equal(ref["timing_allowed"],
                                                          a3["timing_allowed"])),
               "same_box": bool(np.array_equal(ref["p_hi"], a3["p_hi"])
                                and np.array_equal(ref["delay_hi"], a3["delay_hi"])),
               "ref_evals_max": int(ref["total_evaluations"].max()),
               "p1_evals_max": int(a3["total_evaluations"].max()),
               "p1_final_realization_evals_all_1": bool(
                   (a3["final_realization_evaluations"] == 1).all()
                   and (a3["realized_evaluations"] == 1).all()),
               # P1 success counted on the realized flow only: search_success is the realized
               # objective hit, never the continuous one.
               "p1_raw_success_is_realized_hit": bool(
                   np.array_equal(a3["raw_success"], a3["adv_pred"] == 0)
                   and np.array_equal(a3["search_success"], a3["raw_success"])),
               "p1_continuous_only_hits_not_counted": int(
                   (a3["continuous_success"] & ~a3["raw_success"] & a3["valid_success"]).sum()),
               "ref_raw_success_is_realized_hit": bool(
                   np.array_equal(ref["raw_success"], ref["adv_pred"] == 0))}
        for tag, arm in (("ref", ref), ("p1", a3)):
            for k, v in realizer.check(arm, budget).items():
                chk[f"{tag}_{k}"] = v
        for k, v in reproduction_vs_final(dataset, victim, budget, seed, ref).items():
            chk[f"final_{k}"] = v
        sanity.append(chk)
        if seed == REF_SEED:
            x, r = a3["valid_success"].astype(bool), ref["valid_success"].astype(bool)
            both, only_x, only_r = int((x & r).sum()), int((x & ~r).sum()), int((~x & r).sum())
            neither = int((~x & ~r).sum())
            t = mcnemar_test(only_x, only_r)
            lo, hi = newcombe_paired_ci(both, only_x, only_r, neither)
            tests.append({"dataset": dataset, "victim": victim, "budget": budget,
                          "seed": REF_SEED, "n": int(len(x)), "outcome": "valid_success",
                          "p1_valid_asr": float(x.mean()), "ref_valid_asr": float(r.mean()),
                          "diff_pp": float(100 * (x.mean() - r.mean())),
                          "ci95_lo_pp": float(100 * lo), "ci95_hi_pp": float(100 * hi),
                          "p1_only": only_x, "ref_only": only_r, "test": t["test_variant"],
                          "p_value": float(t["p_value"])})
    tests_df = pd.DataFrame(tests)
    if not tests_df.empty:
        tests_df["p_holm"] = holm_adjust(tests_df["p_value"].tolist())

    per_seed = pd.DataFrame(seeds_rows)
    group = ["dataset", "victim", "budget"]
    summary_rows = []
    for (dataset, victim, budget), g in per_seed.groupby(group, sort=False):
        row = {"dataset": dataset, "victim": victim, "budget": budget,
               "seeds": ",".join(map(str, g["seed"]))}
        for col in ("ref_raw_asr", "ref_valid_asr", "p1_continuous_asr", "p1_raw_asr",
                    "p1_valid_asr", "delta_valid_pp", "delta_raw_pp",
                    "continuous_hit_realized_miss", "continuous_hit_realized_hit_invalid",
                    "continuous_miss_realized_hit", "prediction_changed_by_realization",
                    "ref_valid_cost_median", "p1_valid_cost_median",
                    "ref_valid_delay_us_median", "p1_valid_delay_us_median",
                    "ref_evals_mean", "p1_evals_mean"):
            row[f"{col}_mean"], row[f"{col}_sd"] = _mean_sd(g[col])
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)

    loss = (per_seed.groupby(group, sort=False)[
        ["n", "continuous_hits", "continuous_hit_realized_valid",
         "continuous_hit_realized_hit_invalid", "continuous_hit_realized_miss",
         "continuous_miss_realized_hit", "prediction_changed_by_realization",
         "continuous_hits_validator_pass", "continuous_hits_schema_pass",
         "final_flow_identical", "ref_valid_final_flow_identical", "valid_both",
         "valid_p1_only", "valid_ref_only"]]
        .sum().reset_index())
    loss["realized_miss_share_of_continuous_hits"] = (
        loss["continuous_hit_realized_miss"] / loss["continuous_hits"].where(loss["continuous_hits"] > 0))
    extremes = per_seed.groupby(group, sort=False).agg(
        max_abs_delay_rounding_us=("max_abs_delay_rounding_us", "max"),
        max_abs_p_rounding_bytes=("max_abs_p_rounding_bytes", "max"),
        max_abs_margin_change=("max_abs_margin_change", "max"),
        min_abs_continuous_margin_hits=("min_abs_continuous_margin_hits", "min")).reset_index()
    loss = loss.merge(extremes, on=group, sort=False)
    budget_df = per_seed[group + ["seed", "ref_evals_mean", "ref_evals_max",
                                  "ref_realized_evals_mean", "ref_gradient_evals_mean",
                                  "p1_evals_mean", "p1_evals_max",
                                  "p1_continuous_candidate_evals_mean", "p1_gradient_evals_mean",
                                  "p1_final_realization_evals_mean"]]
    sanity_df = pd.DataFrame(sanity)

    pd.concat(flows, ignore_index=True).to_csv(results_dir / "per_flow.csv.gz", index=False)
    per_seed.to_csv(results_dir / "per_seed.csv", index=False)
    summary.to_csv(results_dir / "summary.csv", index=False)
    tests_df.to_csv(results_dir / "tests.csv", index=False)
    loss.to_csv(results_dir / "realization_loss.csv", index=False)
    budget_df.to_csv(results_dir / "budget.csv", index=False)
    sanity_df.to_csv(results_dir / "sanity_checks.csv", index=False)

    lines = report_lines(summary, tests_df, loss, per_seed, budget_df, sanity_df)
    (results_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def report_lines(summary, tests, loss, per_seed, budget_df, sanity) -> list[str]:
    t_idx = {(r.dataset, r.victim, r.budget): r for r in tests.itertuples(index=False)}
    lines = ["# P1 - Realization-aware search vs continuous-state search", "",
             "Targeted -> Benign, Hybrid Search, frozen clean-correct flows (4 classes x 800 per "
             "attack seed). Mean ± SD over attack seeds 42/2024/2026 (%). `P1 cont` = objective "
             "hit of the continuous candidate before realization (never counted as success). "
             "Paired McNemar on valid success at seed 42 (exact binomial below 25 discordant "
             "pairs), Holm over the 12 cells.", ""]
    for dataset, df in summary.groupby("dataset", sort=True):
        lines += [f"## {dataset}", "",
                  "| victim | budget | ref Raw | ref Valid | P1 cont | P1 Raw | P1 Valid | "
                  "Δ Valid (pp) | seed-42 P1-only / ref-only | p (Holm) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in df.itertuples(index=False):
            t = t_idx.get((r.dataset, r.victim, r.budget))
            lines.append(
                f"| {r.victim} | {r.budget} | {_pm(r.ref_raw_asr_mean, r.ref_raw_asr_sd)} | "
                f"{_pm(r.ref_valid_asr_mean, r.ref_valid_asr_sd)} | "
                f"{_pm(r.p1_continuous_asr_mean, r.p1_continuous_asr_sd)} | "
                f"{_pm(r.p1_raw_asr_mean, r.p1_raw_asr_sd)} | "
                f"{_pm(r.p1_valid_asr_mean, r.p1_valid_asr_sd)} | "
                f"{r.delta_valid_pp_mean:+.2f} ± {r.delta_valid_pp_sd:.2f} | "
                + ("- | - |" if t is None else f"{t.p1_only} / {t.ref_only} | {t.p_holm:.3g} |"))
        lines.append("")
    lines += ["## Continuous -> realized (P1, all three seeds summed)", "",
              "| dataset | victim | budget | flows | cont hits | -> realized valid | -> realized "
              "hit, invalid | -> realized miss | share lost | cont miss -> realized hit | "
              "prediction changed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in loss.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget} | {r.n} | {r.continuous_hits} | "
                     f"{r.continuous_hit_realized_valid} | "
                     f"{r.continuous_hit_realized_hit_invalid} | "
                     f"{r.continuous_hit_realized_miss} | "
                     f"{pct(r.realized_miss_share_of_continuous_hits)} | "
                     f"{r.continuous_miss_realized_hit} | "
                     f"{r.prediction_changed_by_realization} |")
    lines += ["", "## What the single final rounding changes (P1, all three seeds)", "",
              "Continuous flows are judged by the same validator_v2 for diagnosis only. The "
              "margin is the targeted margin max(non-Benign) - Benign logit (negative = hit).", "",
              "| dataset | victim | budget | max abs Δdelay (µs) | max abs Δp (bytes) | max abs "
              "Δmargin | min abs margin of cont hits | cont hits passing validator_v2 (SCHEMA) "
              "unrealized | final flow identical to ref (all / ref valid successes) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in loss.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget} | "
                     f"{r.max_abs_delay_rounding_us:.3f} | {r.max_abs_p_rounding_bytes:.3f} | "
                     f"{r.max_abs_margin_change:.2e} | {r.min_abs_continuous_margin_hits:.2e} | "
                     f"{r.continuous_hits_validator_pass}/{r.continuous_hits} "
                     f"({r.continuous_hits_schema_pass}) | {r.final_flow_identical}/{r.n} / "
                     f"{r.ref_valid_final_flow_identical}/{r.valid_both + r.valid_ref_only} |")
    lines += ["", "## Per seed", "",
              "| dataset | victim | budget | seed | ref Raw | ref Valid | P1 cont | P1 Raw | "
              "P1 Valid | P1-only / ref-only valid | cont hit -> miss | pred changed | "
              "median cost valid ref / P1 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_seed.itertuples(index=False):
        lines.append(f"| {r.dataset} | {r.victim} | {r.budget} | {r.seed} | {pct(r.ref_raw_asr)} | "
                     f"{pct(r.ref_valid_asr)} | {pct(r.p1_continuous_asr)} | {pct(r.p1_raw_asr)} | "
                     f"{pct(r.p1_valid_asr)} | {r.valid_p1_only} / {r.valid_ref_only} | "
                     f"{r.continuous_hit_realized_miss} | {r.prediction_changed_by_realization} | "
                     f"{r.ref_valid_cost_median:.3f} / {r.p1_valid_cost_median:.3f} |")
    lines += ["", "## Victim evaluations per flow (mean over seeds; max over all flows)", "",
              "| dataset | victim | budget | ref mean | ref max | P1 mean | P1 max | P1 cont "
              "candidates | P1 gradient | P1 final realization |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for (d, v, b), g in budget_df.groupby(["dataset", "victim", "budget"], sort=False):
        lines.append(f"| {d} | {v} | {b} | {g.ref_evals_mean.mean():.1f} | {g.ref_evals_max.max()} | "
                     f"{g.p1_evals_mean.mean():.1f} | {g.p1_evals_max.max()} | "
                     f"{g.p1_continuous_candidate_evals_mean.mean():.1f} | "
                     f"{g.p1_gradient_evals_mean.mean():.1f} | "
                     f"{g.p1_final_realization_evals_mean.mean():.2f} |")
    s = sanity
    flows = int(s["flows"].sum())
    lines += ["", "## Sanity checks", "",
              f"* cells x seeds checked: {len(s)}; flows per arm: {flows}",
              f"* identical source ids in both arms: {int(s.same_source_ids.sum())}/{len(s)}; "
              f"frozen selection SHA-256 ok: {int(s.frozen_selection_sha256_ok.fillna(False).sum())}"
              f"/{len(s)}",
              f"* identical capability mask M(x) / per-flow box in both arms: "
              f"{int(s.same_capability.sum())}/{len(s)} / {int(s.same_box.sum())}/{len(s)}; "
              f"M(x) recomputed from the source flows equals the stored mask: ref "
              f"{int(s.ref_capability_equal_recomputed.sum())}/{flows}, P1 "
              f"{int(s.p1_capability_equal_recomputed.sum())}/{flows}",
              f"* max victim evaluations per flow: ref {int(s.ref_evals_max.max())}, P1 "
              f"{int(s.p1_evals_max.max())} (cap {EVAL_CAP}); P1 final realization = 1 "
              f"evaluation per flow in {int(s.p1_final_realization_evals_all_1.sum())}/{len(s)}",
              f"* final validator_v2 recomputed on the stored realized flows equals the stored "
              f"verdict: ref {int(s.ref_validator_equal_recomputed.sum())}/{flows}, P1 "
              f"{int(s.p1_validator_equal_recomputed.sum())}/{flows}",
              f"* realized controls integer (bytes, µs): ref {int(s.ref_controls_integer.sum())}"
              f"/{flows}, P1 {int(s.p1_controls_integer.sum())}/{flows}; stored controls = "
              f"canonical projection of the requested controls: ref "
              f"{int(s.ref_controls_equal_projection.sum())}/{flows}, P1 "
              f"{int(s.p1_controls_equal_projection.sum())}/{flows}; stored flow = quantized φ "
              f"of the stored controls (bit-identical): ref "
              f"{int(s.ref_adv_equal_recomputed.sum())}/{flows}, P1 "
              f"{int(s.p1_adv_equal_recomputed.sum())}/{flows}",
              f"* P1 raw success = realized victim prediction == Benign and search success = raw "
              f"success: {int(s.p1_raw_success_is_realized_hit.sum())}/{len(s)}; valid successes "
              f"of P1 whose realized flow misses (continuous-only): "
              f"{int(s.p1_continuous_only_hits_not_counted.sum())}",
              f"* reference vs FINAL targeted Hybrid (`{FINAL_STAGE['p75']}` p75, "
              f"`{FINAL_STAGE['unb']}` unb): flows compared {int(s.final_flows.sum())}, adv flow "
              f"identical {int(s.final_adv_identical.sum())}, prediction identical "
              f"{int(s.final_pred_identical.sum())}, validity identical "
              f"{int(s.final_valid_identical.sum())}, valid success identical "
              f"{int(s.final_valid_success_identical.sum())}; FINAL artifacts missing "
              f"{int(s.final_missing.sum())}", ""]
    return lines


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__, objective=OBJECTIVE)
