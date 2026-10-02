r"""P2 - Coupled feature-recomputation ablation (``NoCoupledPhi``).

Question: is PrimAttack's coupled recomputation φ needed to produce valid adversarial feature
states, or would changing only each primitive's direct statistics do as well?

Reduced path. ``ablations/common/phi_mapping.py`` traces the data flow of the canonical
``CICIDS2017PrimitiveModel.generate`` (φ) and classifies each of its 23 write sites: DIRECT when
the written value is a function of the primitives and source columns only, DERIVED when φ
computes it from another value it has written. The result (``results/phi_mapping.md``) is

    p            -> Total Length of Fwd Packet, Fwd Packet Length Min/Max          (direct)
    delay, shape -> Fwd IAT Total, Fwd IAT Std/Max/Min                             (direct)
    16 derived features (fwd/combined length means, variance, std, min/max, Fwd IAT Mean,
    Flow Duration, Flow IAT Mean/Max, the four rates) -> held at the source value in P2.

Arms (both targeted -> Benign; Hybrid Search with the FINAL ``PRIM_ARGS``; joint mode;
capability-aware M(x); p75 and envelope-only unbounded train-fit budgets; both datasets; one
victim per architecture; the frozen 800 clean-correct flows per class; attack seeds
42/2024/2026; 256 victim evaluations per flow):

* ``full_phi`` (reference, ``recompute_mode="full_phi"``): the FINAL Hybrid configuration,
  validator_v2 in the search success predicate. Must reproduce the FINAL targeted Hybrid cells
  (``primattack_hybrid_objective_targeted`` p75, ``primattack_targeted_budgets`` unbounded) flow
  for flow.
* ``direct_only`` (P2, ``recompute_mode="direct_only"``): the same search, but every candidate it
  scores and every gradient it takes go through the direct-only map. The search success is the
  objective hit on the reduced flow; validator_v2 is not in the predicate because it would judge
  the reduced flow, which breaks φ's identities by construction. Amendment A6 showed that
  removing the gate leaves every FINAL Hybrid flow bit-identical (targeted, p75 and unbounded,
  both datasets, all seeds), so the gate setting does not separate the two arms.

The primitives ``(p, D, s)`` a ``direct_only`` search returns are realized again through
canonical φ (``phi_mapping.realize_full_phi``; same projection, rounding and M(x)). That full-φ
flow is scored by the victim and the unchanged validator_v2 and is the only P2 outcome compared
with the reference. The reduced flow, its prediction and its validator verdict are kept as
diagnostics.

    python ablations/thesis_ablations/P2_coupled_phi/run.py --device cuda
    python ablations/thesis_ablations/P2_coupled_phi/run.py --skip-run          # re-analyze
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

from ablations.common.cli import experiment_main  # noqa: E402
from ablations.common.hybrid import HybridConfig  # noqa: E402
from ablations.common.phi_mapping import (  # noqa: E402
    DirectOnlyPrimitiveModel, derive_phi_mapping, empirical_check, mapping_markdown,
)
from ablations.common.runner import (  # noqa: E402
    CLASSES, DATASETS, FINAL_RUNS, LAYERS, SEEDS, Condition, layer_masks,
)
from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel  # noqa: E402
from attack.realizability.validator import RealizabilityValidator  # noqa: E402
from datasets import get_adapter  # noqa: E402
from evaluation.paired_validity_gap import holm_adjust, mcnemar_test, newcombe_paired_ci  # noqa: E402
from run_full_adversarial_eval import victim_checkpoint  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402

REF, P2 = "full_phi", "direct_only"
CONDITIONS = [
    Condition(REF, "canonical φ (FINAL targeted Hybrid configuration, validator_v2 gate in the "
                   "search, capability-aware)"),
    Condition(P2, "search sees only φ's direct writes; derived features stay at source values; "
                  "returned primitives re-realized through canonical φ",
              HybridConfig(validity_in_search=False), recompute_mode="direct_only"),
]
BUDGETS = ("p75", "unb")
FINAL_STAGE = {"p75": "primattack_hybrid_objective_targeted", "unb": "primattack_targeted_budgets"}
REF_SEED = 42
BENIGN = 0
ROW_FIELDS = ("sample_id", "positional_idx", "adv_pred", "raw_success", "valid_success",
              "validator_pass", "p", "delay", "shape", "p_hi", "delay_hi", "normalized_cost",
              "total_evaluations", "pad_capability_violation", "timing_capability_violation",
              "pad_allowed", "timing_allowed") + tuple(f"{k}_valid" for k in LAYERS)
P2_FIELDS = ("reduced_success", "reduced_pred", "reduced_validator_pass", "reduced_logits",
             "full_logits") + tuple(f"reduced_{k}_valid" for k in LAYERS)


# ------------------------------------------------------------------------ mapping (before run)
def write_mapping(results_dir: Path, datasets: list[str]) -> dict[str, dict]:
    results_dir.mkdir(parents=True, exist_ok=True)
    out, lines = {}, ["# P2 reduced recomputation path: definition", "",
                      "Derived from code by `ablations/common/phi_mapping.py` before any P2 run. "
                      "The P2 search (`recompute_mode=\"direct_only\"`) writes the direct features "
                      "exactly as φ does and leaves every derived feature at the source flow's "
                      "value. The reference (`recompute_mode=\"full_phi\"`) is φ unchanged.", ""]
    for dataset in datasets:
        adapter = get_adapter(DATASETS[dataset]["cli"])
        manifest = adapter.feature_manifest()
        mapping = derive_phi_mapping(manifest)
        X = np.load(adapter._processed / "X_train_pristine.npy", mmap_mode="r")
        raw = torch.tensor(np.ascontiguousarray(X[:200_000]), dtype=torch.float32)
        mapping["empirical_check"] = empirical_check(CICIDS2017PrimitiveModel(manifest), mapping,
                                                     raw)
        mapping["empirical_check"]["flows_from"] = "first 200,000 TRAIN flows (pristine)"
        out[dataset] = mapping
        lines += [f"# {dataset}", ""] + mapping_markdown(mapping, mapping["empirical_check"])
    names = {json.dumps([m["direct_features"], m["derived_features"]]) for m in out.values()}
    if len(names) != 1:
        raise AssertionError("the φ mapping differs between datasets")
    (results_dir / "phi_mapping.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (results_dir / "phi_mapping.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ------------------------------------------------------------------------------- data loading
def _npz(results_dir: Path, dataset: str, victim: str, cname: str, budget: str, cond: str,
         seed: int) -> Path:
    return results_dir / dataset / "artifacts" / f"{victim}__{cname}__{budget}__{cond}__seed{seed}.npz"


def _cells(results_dir: Path):
    for dataset in DATASETS:
        if not (results_dir / dataset / "cells.json").exists():
            continue
        for victim in DATASETS[dataset]["victims"]:
            for budget in BUDGETS:
                for seed in SEEDS:
                    paths = [(c, _npz(results_dir, dataset, victim, c, budget, cond, seed))
                             for c in CLASSES for cond in (REF, P2)]
                    if all(p.exists() for _, p in paths):
                        yield dataset, victim, budget, seed


def load_cell(results_dir: Path, dataset, victim, budget, seed, *, vectors=False) -> dict:
    """Both arms of one (dataset, victim, budget, seed), classes concatenated."""
    out = {}
    for cond, fields in ((REF, ROW_FIELDS + (("adv_raw",) if vectors else ())),
                         (P2, ROW_FIELDS + P2_FIELDS
                          + (("adv_raw", "reduced_adv_raw") if vectors else ()))):
        parts = []
        for cname in CLASSES:
            with np.load(_npz(results_dir, dataset, victim, cname, budget, cond, seed),
                         allow_pickle=True) as z:
                d = {f: z[f] for f in fields}
            d["class"] = np.full(len(d["sample_id"]), cname)
            parts.append(d)
        out[cond] = {f: np.concatenate([p[f] for p in parts]) for f in parts[0]}
    return out


# ------------------------------------------------------------------------------- per-flow table
def per_flow_table(results_dir: Path) -> pd.DataFrame:
    frames = []
    for dataset, victim, budget, seed in _cells(results_dir):
        c = load_cell(results_dir, dataset, victim, budget, seed)
        r, a = c[REF], c[P2]
        if not np.array_equal(r["sample_id"], a["sample_id"]):
            raise AssertionError(f"{dataset}/{victim}/{budget}/{seed}: arms not row-aligned")
        frames.append(pd.DataFrame({
            "dataset": dataset, "victim": victim, "budget": budget, "seed": seed,
            "class": r["class"], "sample_id": r["sample_id"],
            "ref_pred": r["adv_pred"], "ref_raw_success": r["raw_success"],
            "ref_validator_pass": r["validator_pass"], "ref_valid_success": r["valid_success"],
            "ref_p": r["p"], "ref_delay": r["delay"], "ref_shape": r["shape"],
            "ref_cost": r["normalized_cost"], "ref_evaluations": r["total_evaluations"],
            "p2_reduced_pred": a["reduced_pred"], "p2_reduced_success": a["reduced_success"],
            "p2_reduced_validator_pass": a["reduced_validator_pass"],
            "p2_full_pred": a["adv_pred"], "p2_full_raw_success": a["raw_success"],
            "p2_full_validator_pass": a["validator_pass"],
            "p2_full_valid_success": a["valid_success"],
            "p2_p": a["p"], "p2_delay": a["delay"], "p2_shape": a["shape"],
            "p2_cost": a["normalized_cost"], "p2_evaluations": a["total_evaluations"],
            "p_hi": r["p_hi"], "delay_hi": r["delay_hi"],
        }))
    if not frames:
        raise FileNotFoundError(f"no complete cells under {results_dir}")
    df = pd.concat(frames, ignore_index=True)
    for col in df.columns:
        if col.endswith(("_success", "_pass")):
            df[col] = df[col].astype(bool)
    return df


def _med(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(np.median(x)) if x.size else float("nan")


def per_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, victim, budget, seed), g in df.groupby(["dataset", "victim", "budget", "seed"],
                                                          sort=False):
        n = len(g)
        red, full, valid = g.p2_reduced_success, g.p2_full_raw_success, g.p2_full_valid_success
        ref = g.ref_valid_success
        changed = g.p2_reduced_pred != g.p2_full_pred
        moved = (g.p2_p > 0) | (g.p2_delay > 0)
        rows.append({
            "dataset": dataset, "victim": victim, "budget": budget, "seed": seed, "n": n,
            "ref_raw_asr": g.ref_raw_success.mean(), "ref_valid_asr": ref.mean(),
            "p2_reduced_raw_asr": red.mean(), "p2_full_raw_asr": full.mean(),
            "p2_full_valid_asr": valid.mean(),
            "reduced_to_full_raw_loss_pp": 100 * (red.mean() - full.mean()),
            "reduced_to_full_valid_loss_pp": 100 * (red.mean() - valid.mean()),
            "valid_delta_vs_ref_pp": 100 * (valid.mean() - ref.mean()),
            "apparent_successes": int(red.sum()),
            "apparent_surviving_valid": int((red & valid).sum()),
            "apparent_invalidated": int((red & ~valid).sum()),
            "apparent_invalidated_pct": 100 * (red & ~valid).sum() / max(int(red.sum()), 1),
            "apparent_lost_by_prediction": int((red & ~full).sum()),
            "apparent_lost_by_validator": int((red & full & ~valid).sum()),
            "full_hit_not_apparent": int((~red & full).sum()),
            "prediction_changed": int(changed.sum()),
            "prediction_changed_pct": 100 * changed.mean(),
            "prediction_changed_among_apparent": int((changed & red).sum()),
            "prediction_changed_among_apparent_pct":
                100 * (changed & red).sum() / max(int(red.sum()), 1),
            "rows_moved": int(moved.sum()),
            "reduced_validator_pass_rate_moved":
                float(g.p2_reduced_validator_pass[moved].mean()) if moved.any() else float("nan"),
            "full_validator_pass_rate": g.p2_full_validator_pass.mean(),
            "ref_validator_pass_rate": g.ref_validator_pass.mean(),
            "p2_median_cost_valid": _med(g.p2_cost[valid]),
            "ref_median_cost_valid": _med(g.ref_cost[ref]),
            "both_valid": int((valid & ref).sum()), "p2_only_valid": int((valid & ~ref).sum()),
            "ref_only_valid": int((~valid & ref).sum()),
            "neither_valid": int((~valid & ~ref).sum()),
            "p2_evals_mean": g.p2_evaluations.mean(), "p2_evals_max": int(g.p2_evaluations.max()),
            "ref_evals_mean": g.ref_evaluations.mean(),
            "ref_evals_max": int(g.ref_evaluations.max()),
        })
    return pd.DataFrame(rows)


AGG_COLS = ("ref_raw_asr", "ref_valid_asr", "p2_reduced_raw_asr", "p2_full_raw_asr",
            "p2_full_valid_asr", "reduced_to_full_valid_loss_pp", "apparent_invalidated_pct",
            "prediction_changed_among_apparent_pct", "p2_median_cost_valid",
            "ref_median_cost_valid")


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, victim, budget), g in per_seed.groupby(["dataset", "victim", "budget"],
                                                         sort=False):
        row = {"dataset": dataset, "victim": victim, "budget": budget,
               "seeds": ",".join(map(str, sorted(g.seed))), "n_per_seed": int(g.n.iloc[0])}
        for c in AGG_COLS:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else float("nan")
        for c in ("apparent_successes", "apparent_invalidated", "apparent_lost_by_prediction",
                  "apparent_lost_by_validator", "prediction_changed",
                  "prediction_changed_among_apparent", "full_hit_not_apparent"):
            row[f"{c}_total"] = int(g[c].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def mcnemar_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, victim, budget, seed): P2 full-φ Valid vs reference Valid (primary) and P2
    reduced-space hit vs P2 full-φ Valid (secondary). Holm within each (comparison, seed)
    family of 12 cells; the seed-42 P2-vs-reference family is the primary one."""
    rows = []
    for (dataset, victim, budget, seed), g in df.groupby(["dataset", "victim", "budget", "seed"],
                                                          sort=False):
        for comparison, x, r in (
                ("p2_full_valid_vs_ref_valid", g.p2_full_valid_success, g.ref_valid_success),
                ("p2_reduced_hit_vs_p2_full_valid", g.p2_reduced_success,
                 g.p2_full_valid_success)):
            x, r = x.to_numpy(bool), r.to_numpy(bool)
            both, only_x, only_r = int((x & r).sum()), int((x & ~r).sum()), int((~x & r).sum())
            neither = int((~x & ~r).sum())
            t = mcnemar_test(only_x, only_r)
            lo, hi = newcombe_paired_ci(both, only_x, only_r, neither)
            rows.append({"comparison": comparison, "dataset": dataset, "victim": victim,
                         "budget": budget, "seed": seed, "n": len(x),
                         "a_asr": x.mean(), "b_asr": r.mean(),
                         "diff_pp": 100 * (x.mean() - r.mean()),
                         "ci95_lo_pp": 100 * lo, "ci95_hi_pp": 100 * hi,
                         "a_only": only_x, "b_only": only_r, "test": t["test_variant"],
                         "p_value": float(t["p_value"])})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = np.nan
    for _, idx in tests.groupby(["comparison", "seed"]).groups.items():
        tests.loc[idx, "p_holm"] = holm_adjust(tests.loc[idx, "p_value"].tolist())
    tests["primary"] = (tests.comparison == "p2_full_valid_vs_ref_valid") & (tests.seed == REF_SEED)
    return tests


# --------------------------------------------------------------------------------- diagnostics
def diagnostics(results_dir: Path, mapping: dict, device: str) -> dict:
    """Lost apparent successes (reduced hit, no full-φ valid success): vectors, logits,
    validator outcome; single-group / single-feature substitutions between the two flows."""
    out_dir = results_dir / "diagnostics"
    out_dir.mkdir(exist_ok=True)
    groups = mapping["derived_groups"]
    derived = mapping["derived_features"]
    feature_records, group_records, lost_rows = [], [], []
    vec = {"reduced": [], "full": [], "key": []}
    verdicts = {"rows": 0, "reduced_hit_reproduced": 0, "full_verdict_reproduced": 0}
    by_dv: dict[tuple[str, str], list] = {}
    for dataset, victim, budget, seed in _cells(results_dir):
        by_dv.setdefault((dataset, victim), []).append((budget, seed))
    for (dataset, victim), cells in by_dv.items():
        adapter = get_adapter(DATASETS[dataset]["cli"])
        names = list(adapter.feature_manifest().names)
        fidx = {n: names.index(n) for n in derived}
        gidx = {g: [names.index(f) for f in fs] for g, fs in groups.items()}
        transform = adapter.feature_transform()
        center = torch.tensor(transform.center, dtype=torch.float32, device=device)
        scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
        ckpt, arch, _ = victim_checkpoint(dataset, victim)
        net = load_category_victim(ckpt, adapter=adapter, expected_model_type=arch, device=device)

        @torch.no_grad()
        def hit(x: np.ndarray) -> np.ndarray:
            t = torch.tensor(x, dtype=torch.float32, device=device)
            return (net((t - center) / scale).argmax(1) == BENIGN).cpu().numpy()

        for budget, seed in cells:
            c = load_cell(results_dir, dataset, victim, budget, seed, vectors=True)[P2]
            red_hit = c["reduced_success"].astype(bool)
            lost = red_hit & ~c["valid_success"].astype(bool)
            if not lost.any():
                continue
            red, full = c["reduced_adv_raw"][lost], c["adv_raw"][lost]
            by_pred = ~c["raw_success"].astype(bool)[lost]
            # reproduce both verdicts from the stored vectors (batch composition differs from
            # the run, so a borderline argmax could in principle flip: counted, not assumed)
            red_rescored, full_rescored = hit(red), hit(full)
            verdicts["rows"] += len(red)
            verdicts["reduced_hit_reproduced"] += int(red_rescored.sum())
            verdicts["full_verdict_reproduced"] += int((full_rescored == ~by_pred).sum())
            diff = red != full
            scale_np = scale.cpu().numpy()
            for k, j in enumerate(np.flatnonzero(lost)):
                lost_rows.append({
                    "dataset": dataset, "victim": victim, "budget": budget, "seed": seed,
                    "class": c["class"][j], "sample_id": c["sample_id"][j],
                    "p": c["p"][j], "delay": c["delay"][j], "shape": c["shape"][j],
                    "lost_by": "prediction" if by_pred[k] else "validator",
                    "reduced_pred": int(c["reduced_pred"][j]), "full_pred": int(c["adv_pred"][j]),
                    **{f"reduced_logit_{i}": float(v) for i, v in enumerate(c["reduced_logits"][j])},
                    **{f"full_logit_{i}": float(v) for i, v in enumerate(c["full_logits"][j])},
                    "full_hybrid_valid": bool(c["validator_pass"][j]),
                    **{f"full_{x}_valid": bool(c[f"{x}_valid"][j]) for x in LAYERS},
                    "reduced_hybrid_valid": bool(c["reduced_validator_pass"][j]),
                    "n_changed_features": int(diff[k].sum()),
                    "changed_features": ";".join(names[i] for i in np.flatnonzero(diff[k])),
                })
                vec["key"].append(f"{dataset}|{victim}|{budget}|{seed}|{c['class'][j]}|"
                                  f"{c['sample_id'][j]}")
            vec["reduced"].append(red)
            vec["full"].append(full)
            # substitutions only on rows whose two verdicts re-score exactly (hit / miss)
            keep = by_pred & red_rescored & ~full_rescored
            if not keep.any():
                continue
            rp, fp = red[keep], full[keep]
            dz = np.abs((fp - rp) / scale_np)  # change in the victim's (scaled) input space
            base = {"dataset": dataset, "victim": victim, "budget": budget, "seed": seed}
            for g, cols in gidx.items():
                x = rp.copy(); x[:, cols] = fp[:, cols]
                y = fp.copy(); y[:, cols] = rp[:, cols]
                group_records.append({**base, "group": g, "rows": len(rp),
                                      "changed_rows": int((rp[:, cols] != fp[:, cols]).any(1).sum()),
                                      "alone_breaks_success": int((~hit(x)).sum()),
                                      "reverting_restores_success": int(hit(y).sum())})
            for f, j in fidx.items():
                x = rp.copy(); x[:, j] = fp[:, j]
                y = fp.copy(); y[:, j] = rp[:, j]
                feature_records.append({**base, "feature": f, "rows": len(rp),
                                        "changed_rows": int((rp[:, j] != fp[:, j]).sum()),
                                        "median_abs_scaled_change": float(np.median(dz[:, j])),
                                        "alone_breaks_success": int((~hit(x)).sum()),
                                        "reverting_restores_success": int(hit(y).sum())})
        del net
    lost_df = pd.DataFrame(lost_rows)
    lost_df.to_csv(out_dir / "lost_successes.csv", index=False)
    if vec["key"]:
        np.savez_compressed(out_dir / "lost_successes_vectors.npz",
                            key=np.asarray(vec["key"]), feature_names=np.asarray(names),
                            reduced_adv_raw=np.concatenate(vec["reduced"]),
                            full_phi_adv_raw=np.concatenate(vec["full"]))

    def rank(records, key):
        if not records:
            return pd.DataFrame()
        d = pd.DataFrame(records)
        agg = d.groupby(["dataset", key], sort=False)[
            ["rows", "changed_rows", "alone_breaks_success", "reverting_restores_success"]].sum()
        agg = agg.reset_index()
        agg["alone_breaks_pct"] = 100 * agg.alone_breaks_success / agg.rows
        agg["reverting_restores_pct"] = 100 * agg.reverting_restores_success / agg.rows
        agg["changed_pct"] = 100 * agg.changed_rows / agg.rows
        if "median_abs_scaled_change" in d:
            med = d.groupby(["dataset", key], sort=False).median_abs_scaled_change.median()
            agg = agg.merge(med.rename("median_abs_scaled_change_of_cells").reset_index())
        return agg.sort_values(["dataset", "alone_breaks_success", "reverting_restores_success"],
                               ascending=[True, False, False])

    group_rank, feature_rank = rank(group_records, "group"), rank(feature_records, "feature")
    group_rank.to_csv(out_dir / "group_ranking.csv", index=False)
    feature_rank.to_csv(out_dir / "feature_ranking.csv", index=False)
    pd.DataFrame(group_records).to_csv(out_dir / "group_substitution_per_cell.csv", index=False)
    pd.DataFrame(feature_records).to_csv(out_dir / "feature_substitution_per_cell.csv",
                                         index=False)
    (out_dir / "verdict_reproduction.json").write_text(json.dumps(verdicts, indent=2),
                                                       encoding="utf-8")
    return {"lost": lost_df, "group_rank": group_rank, "feature_rank": feature_rank,
            "verdicts": verdicts}


# --------------------------------------------------------------------- reduced-flow audit
def reduced_flow_audit(results_dir: Path, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """validator_v2 vs φ's own consistency checks on the reduced flows the P2 search saw.

    Rows with a non-zero primitive. ``attack/realizability/validator.py`` (Level B: the exact
    identities ``CICIDS2017PrimitiveModel.algebraic_identities`` + timing order checks) is run on
    the reduced and on the full-φ flow; per-identity violations are counted among the reduced
    flows validator_v2 accepted."""

    rows, ident_rows, models, raws = [], [], {}, {}
    for dataset, victim, budget, seed in _cells(results_dir):
        if dataset not in models:
            adapter = get_adapter(DATASETS[dataset]["cli"])
            model = CICIDS2017PrimitiveModel(adapter.feature_manifest())
            models[dataset] = (model, RealizabilityValidator(model))
            raws[dataset] = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
        model, rv = models[dataset]
        a = load_cell(results_dir, dataset, victim, budget, seed, vectors=True)[P2]
        moved = (a["p"] > 0) | (a["delay"] > 0)
        if not moved.any():
            continue
        src = torch.tensor(np.ascontiguousarray(raws[dataset][a["positional_idx"][moved]]),
                           dtype=torch.float32, device=device)
        red = torch.tensor(a["reduced_adv_raw"][moved], device=device)
        full = torch.tensor(a["adv_raw"][moved], device=device)
        rr, fr = rv.validate(red, src), rv.validate(full, src)
        np_ = lambda t: t.cpu().numpy()  # noqa: E731
        v2 = a["reduced_validator_pass"][moved].astype(bool)
        alg, tim = np_(rr.categories["algebraic_dependency_fail"]), np_(rr.categories["timing_fail"])
        rows.append({"dataset": dataset, "victim": victim, "budget": budget, "seed": seed,
                     "moved_rows": int(moved.sum()), "reduced_v2_accept": int(v2.sum()),
                     "reduced_identity_fail": int(alg.sum()), "reduced_timing_order_fail": int(tim.sum()),
                     "reduced_levelB_fail": int(np_(rr.any_fail).sum()),
                     "v2_accepted_identity_fail": int((v2 & alg).sum()),
                     "v2_accepted_timing_order_fail": int((v2 & tim).sum()),
                     "full_phi_levelB_fail": int(np_(fr.any_fail).sum()),
                     "full_phi_v2_reject": int((~a["validator_pass"][moved].astype(bool)).sum())})
        c = lambda n: red[:, model.i[n]]  # noqa: E731
        for chk in model.algebraic_identities():
            expected = chk.fn(c)
            bad = np_((c(chk.target) - expected).abs() > (chk.atol + chk.rtol * expected.abs()))
            ident_rows.append({"dataset": dataset, "budget": budget, "identity": chk.target,
                               "violations": int(bad.sum()), "v2_accepted_violations":
                               int((bad & v2).sum()), "moved_rows": int(moved.sum())})
    audit = pd.DataFrame(rows)
    idents = (pd.DataFrame(ident_rows).groupby(["dataset", "budget", "identity"], sort=False)
              .sum(numeric_only=True).reset_index())
    audit.to_csv(results_dir / "reduced_flow_audit_per_seed.csv", index=False)
    idents.to_csv(results_dir / "reduced_flow_identity_violations.csv", index=False)
    return audit, idents


# ------------------------------------------------------------------------------ sanity checks
def sanity_checks(results_dir: Path, mapping: dict, device: str) -> dict:
    checks: dict = {"reproduction_vs_final": [], "pairing": {}, "primitive_identity": {},
                    "capability": {}, "validator": {}, "final_counting": {}}
    totals = {k: 0 for k in ("flows", "adv_identical", "pred_identical", "valid_identical",
                             "success_identical")}
    n_pair = n_bounds = n_cells = 0
    prim = {"rows": 0, "full_phi_recomputed_identical": 0, "reduced_recomputed_identical": 0,
            "reprojection_identical": 0, "reduced_full_differ_only_on_derived": 0}
    cap = {"ref_violations": 0, "p2_violations": 0, "cap_mask_matches_inference": 0, "rows": 0}
    val = {"rows": 0, "p2_validator_recomputed_identical": 0, "ref_validator_recomputed_identical": 0}
    count = {"p2_valid_not_full_hit": 0, "p2_valid_not_full_validator": 0,
             "p2_raw_success_not_full_benign": 0, "ref_valid_not_hit_and_valid": 0}
    models, raws, masks = {}, {}, {}
    for dataset, victim, budget, seed in _cells(results_dir):
        if dataset not in models:
            adapter = get_adapter(DATASETS[dataset]["cli"])
            model = CICIDS2017PrimitiveModel(adapter.feature_manifest())
            models[dataset] = (model, DirectOnlyPrimitiveModel(model, mapping))
            raws[dataset] = np.load(adapter._processed / "X_test_pristine.npy", mmap_mode="r")
            masks[dataset] = models[dataset][1].derived_mask.numpy()
        model, reduced_model = models[dataset]
        c = load_cell(results_dir, dataset, victim, budget, seed, vectors=True)
        r, a = c[REF], c[P2]
        n_cells += 1
        n_pair += int(np.array_equal(r["sample_id"], a["sample_id"])
                      and np.array_equal(r["positional_idx"], a["positional_idx"]))
        n_bounds += int(np.array_equal(r["p_hi"], a["p_hi"])
                        and np.array_equal(r["delay_hi"], a["delay_hi"]))
        # reproduction of FINAL (reference arm)
        for cname in CLASSES:
            final = (FINAL_RUNS / dataset / FINAL_STAGE[budget] / "artifacts"
                     / f"{victim}__{cname}__{budget}__hybrid__seed{seed}.npz")
            with np.load(_npz(results_dir, dataset, victim, cname, budget, REF, seed),
                         allow_pickle=True) as x, np.load(final, allow_pickle=True) as y:
                n = len(x["sample_id"])
                if not np.array_equal(x["sample_id"], y["sample_id"][:n]):
                    raise AssertionError(f"{final.name}: sample order differs from FINAL")
                rec = {"dataset": dataset, "victim": victim, "budget": budget, "seed": seed,
                       "class": cname, "flows": n,
                       "adv_identical": int(np.all(x["adv_raw"] == y["adv_raw"][:n], 1).sum()),
                       "pred_identical": int((x["adv_pred"] == y["adv_pred"][:n]).sum()),
                       "valid_identical": int((x["validator_pass"] == y["validator_pass"][:n]).sum()),
                       "success_identical": int((x["valid_success"] == y["valid_success"][:n]).sum())}
            checks["reproduction_vs_final"].append(rec)
            for k in totals:
                totals[k] += rec[k]
        # primitive identity: stored primitives -> canonical φ / direct-only map
        raw = torch.tensor(np.ascontiguousarray(raws[dataset][a["positional_idx"]]),
                           dtype=torch.float32, device=device)
        caps = model.infer_capabilities(raw)
        u = {k: torch.tensor(a[k], dtype=torch.float32, device=device)
             for k in ("p", "delay", "shape")}
        bounds = {"p": torch.tensor(a["p_hi"], device=device),
                  "delay": torch.tensor(a["delay_hi"], device=device),
                  "shape": caps.timing_allowed.to(torch.float32)}
        rp = model.project_controls(raw, u, bounds, capabilities=caps)
        same_proj = np.ones(len(raw), bool)
        for k in u:
            same_proj &= (rp[k] == u[k]).cpu().numpy()
        full = model.generate(raw, u, quantize=True, capabilities=caps).cpu().numpy()
        red = reduced_model.generate(raw, u, quantize=True, capabilities=caps).cpu().numpy()
        prim["rows"] += len(raw)
        prim["reprojection_identical"] += int(same_proj.sum())
        prim["full_phi_recomputed_identical"] += int(np.all(full == a["adv_raw"], 1).sum())
        prim["reduced_recomputed_identical"] += int(np.all(red == a["reduced_adv_raw"], 1).sum())
        prim["reduced_full_differ_only_on_derived"] += int(
            (~(a["reduced_adv_raw"] != a["adv_raw"])[:, ~masks[dataset]].any(1)).sum())
        # capability inference active in both arms
        pad, tim = caps.pad_allowed.cpu().numpy(), caps.timing_allowed.cpu().numpy()
        cap["rows"] += len(raw)
        cap["ref_violations"] += int((r["pad_capability_violation"]
                                      | r["timing_capability_violation"]).sum())
        cap["p2_violations"] += int((a["pad_capability_violation"]
                                     | a["timing_capability_violation"]).sum())
        cap["cap_mask_matches_inference"] += int(((a["pad_allowed"] == pad)
                                                  & (a["timing_allowed"] == tim)).sum())
        # final validator unchanged: recompute validator_v2 on the stored flows
        src = raws[dataset][a["positional_idx"]]
        for arm, d, key in ((P2, a, "p2_validator_recomputed_identical"),
                            (REF, r, "ref_validator_recomputed_identical")):
            m = layer_masks(d["adv_raw"], src, dataset)
            hv = np.logical_and.reduce([m[k] for k in LAYERS])
            val[key] += int((hv == d["validator_pass"]).sum())
        val["rows"] += len(raw)
        # no P2 success counted unless full φ hit AND full validator
        benign = a["full_logits"].argmax(1) == BENIGN
        vs, rs = a["valid_success"].astype(bool), a["raw_success"].astype(bool)
        count["p2_valid_not_full_hit"] += int((vs & ~benign).sum())
        count["p2_valid_not_full_validator"] += int((vs & ~a["validator_pass"].astype(bool)).sum())
        count["p2_raw_success_not_full_benign"] += int((rs != benign).sum())
        count["ref_valid_not_hit_and_valid"] += int(
            (r["valid_success"].astype(bool)
             != (r["raw_success"].astype(bool) & r["validator_pass"].astype(bool))).sum())
    checks["reproduction_totals"] = totals
    checks["pairing"] = {"cells": n_cells, "sample_ids_identical_cells": n_pair,
                         "budget_boxes_identical_cells": n_bounds}
    checks["primitive_identity"] = prim
    checks["capability"] = cap
    checks["validator"] = val
    checks["final_counting"] = count
    ok = (totals["flows"] == totals["adv_identical"] == totals["success_identical"]
          and n_pair == n_cells == n_bounds
          and all(v == prim["rows"] for v in prim.values())
          and cap["ref_violations"] == cap["p2_violations"] == 0
          and cap["cap_mask_matches_inference"] == cap["rows"]
          and val["rows"] == val["p2_validator_recomputed_identical"]
          == val["ref_validator_recomputed_identical"]
          and not any(count.values()))
    checks["all_passed"] = bool(ok)
    (results_dir / "sanity_checks.json").write_text(json.dumps(checks, indent=2, default=int),
                                                    encoding="utf-8")
    return checks


# ------------------------------------------------------------------------------------ report
def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.2f}%"


def msd(row, col: str, scale: float = 100.0, unit: str = "%") -> str:
    m, s = row[f"{col}_mean"], row[f"{col}_sd"]
    if m != m:
        return "n/a"
    return f"{scale * m:.2f} ± {scale * s:.2f}{unit}"


def report(results_dir: Path, mapping: dict, agg: pd.DataFrame, per_seed: pd.DataFrame,
           tests: pd.DataFrame, diag: dict, checks: dict, audit: pd.DataFrame,
           idents: pd.DataFrame) -> None:
    L = ["# P2 - Coupled feature-recomputation ablation: tables", "",
         "Targeted → Benign. Per (dataset, victim, budget) the four classes are pooled "
         "(3,200 frozen flows per seed). Mean ± SD over attack seeds 42/2024/2026. "
         "`full_phi` = reference (canonical φ). `direct_only` = P2: the search sees φ's direct "
         "writes only; its primitives are then realized through canonical φ and judged by "
         "validator_v2. Definition of the reduced path: `phi_mapping.md`.", "",
         "## Reference vs P2 (Valid = full-φ flow ∧ validator_v2)", "",
         "| dataset | victim | budget | Ref Raw | Ref Valid | P2 reduced-space Raw | P2 full-φ Raw "
         "| P2 full-φ Valid | reduced→full-φ Valid loss (pp) | seed-42 P2-only / ref-only | "
         "p (Holm) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    t = tests[tests.primary].set_index(["dataset", "victim", "budget"])
    for _, r in agg.iterrows():
        k = t.loc[(r.dataset, r.victim, r.budget)]
        L.append(f"| {r.dataset} | {r.victim} | {r.budget} | {msd(r, 'ref_raw_asr')} | "
                 f"{msd(r, 'ref_valid_asr')} | {msd(r, 'p2_reduced_raw_asr')} | "
                 f"{msd(r, 'p2_full_raw_asr')} | {msd(r, 'p2_full_valid_asr')} | "
                 f"{msd(r, 'reduced_to_full_valid_loss_pp', 1.0, '')} | "
                 f"{int(k.a_only)} / {int(k.b_only)} | {k.p_holm:.3g} |")
    L += ["", "## Apparent (reduced-space) successes after full φ, all seeds pooled", "",
          "| dataset | victim | budget | apparent | invalidated (%) | lost: prediction | "
          "lost: validator | prediction changed among apparent (%) | full-φ hits not apparent | "
          "median cost of valid successes P2 / ref |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        app = r.apparent_successes_total
        L.append(f"| {r.dataset} | {r.victim} | {r.budget} | {app} | "
                 f"{r.apparent_invalidated_total} ({100 * r.apparent_invalidated_total / max(app, 1):.1f}%) | "
                 f"{r.apparent_lost_by_prediction_total} | {r.apparent_lost_by_validator_total} | "
                 f"{r.prediction_changed_among_apparent_total} "
                 f"({100 * r.prediction_changed_among_apparent_total / max(app, 1):.1f}%) | "
                 f"{r.full_hit_not_apparent_total} | "
                 f"{r.p2_median_cost_valid_mean:.3g} / {r.ref_median_cost_valid_mean:.3g} |")
    L += ["", "## McNemar (P2 full-φ Valid vs reference Valid), all seeds", "",
          "Holm within each seed's family of 12 cells; seed 42 is the primary family.", "",
          "| dataset | victim | budget | seed | P2 | ref | diff (pp) [95% CI] | P2-only / ref-only "
          "| test | p | p (Holm) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in tests[tests.comparison == "p2_full_valid_vs_ref_valid"].iterrows():
        L.append(f"| {r.dataset} | {r.victim} | {r.budget} | {r.seed} | {pct(r.a_asr)} | "
                 f"{pct(r.b_asr)} | {r.diff_pp:+.2f} [{r.ci95_lo_pp:+.2f}, {r.ci95_hi_pp:+.2f}] | "
                 f"{r.a_only} / {r.b_only} | {r.test} | {r.p_value:.3g} | {r.p_holm:.3g} |")
    L += ["", "## Reduced-space hit vs full-φ Valid within P2 (same primitives), seed 42", "",
          "| dataset | victim | budget | reduced hit | full-φ Valid | reduced-only / full-only | "
          "p (Holm) |", "|---|---|---|---|---|---|---|"]
    for _, r in tests[(tests.comparison == "p2_reduced_hit_vs_p2_full_valid")
                      & (tests.seed == REF_SEED)].iterrows():
        L.append(f"| {r.dataset} | {r.victim} | {r.budget} | {pct(r.a_asr)} | {pct(r.b_asr)} | "
                 f"{r.a_only} / {r.b_only} | {r.p_holm:.3g} |")
    L += ["", "## Consistency of the reduced flows (rows with a non-zero primitive, all seeds)", "",
          "validator_v2 = the final validator (reduced flow judged given its source flow). "
          "Level B = `attack/realizability/validator.py`: φ's exact identities "
          "(`algebraic_identities`) and timing order checks (e.g. Fwd IAT Total ≤ Flow "
          "Duration). Diagnostic only; no P2 result is counted on a reduced flow.", "",
          "| dataset | victim | budget | moved rows | reduced: v2 accepts | reduced: identity "
          "violated | v2-accepted with identity violated | v2-accepted with timing order "
          "violated | full-φ: Level-B fail | full-φ: v2 rejects |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for (d, v, b), g in audit.groupby(["dataset", "victim", "budget"], sort=False):
        s = g.sum(numeric_only=True)
        m = max(int(s.moved_rows), 1)
        L.append(f"| {d} | {v} | {b} | {int(s.moved_rows)} | "
                 f"{100 * s.reduced_v2_accept / m:.1f}% | {100 * s.reduced_identity_fail / m:.1f}% | "
                 f"{int(s.v2_accepted_identity_fail)} | {int(s.v2_accepted_timing_order_fail)} | "
                 f"{int(s.full_phi_levelB_fail)} | {int(s.full_phi_v2_reject)} |")
    L += ["", "Per-identity violations on the reduced flows (all victims and seeds):", "",
          "| dataset | budget | identity target | violated | of which accepted by validator_v2 |",
          "|---|---|---|---|---|"]
    for _, r in idents[idents.violations > 0].iterrows():
        L.append(f"| {r.dataset} | {r.budget} | {r.identity} | {r.violations} | "
                 f"{r.v2_accepted_violations} |")
    gr, fr = diag["group_rank"], diag["feature_rank"]
    L += ["", "## Which recomputed groups account for the lost successes (lost by prediction)", "",
          "Single substitutions between the reduced and the full-φ flow of the same primitives. "
          "*alone breaks*: reduced flow + this group's full-φ values → no longer Benign. "
          "*reverting restores*: full-φ flow with this group reset to source values → Benign "
          "again. Descriptive of the victims' decisions on these flows; interactions between "
          "groups are not separated.", ""]
    if not gr.empty:
        L += ["| dataset | φ code block (derived group) | lost rows | rows where it changed | "
              "alone breaks | reverting restores |", "|---|---|---|---|---|---|"]
        for _, r in gr.iterrows():
            L.append(f"| {r.dataset} | {r.group} | {r.rows} | {r.changed_rows} | "
                     f"{r.alone_breaks_success} ({r.alone_breaks_pct:.1f}%) | "
                     f"{r.reverting_restores_success} ({r.reverting_restores_pct:.1f}%) |")
        L += ["", "| dataset | feature | lost rows | changed | median abs. change (scaled) | "
              "alone breaks | reverting restores |", "|---|---|---|---|---|---|---|"]
        for _, r in fr.iterrows():
            L.append(f"| {r.dataset} | {r.feature} | {r.rows} | {r.changed_rows} | "
                     f"{r.median_abs_scaled_change_of_cells:.3g} | "
                     f"{r.alone_breaks_success} ({r.alone_breaks_pct:.1f}%) | "
                     f"{r.reverting_restores_success} ({r.reverting_restores_pct:.1f}%) |")
    else:
        L.append("No apparent success was lost by prediction.")
    tot = checks["reproduction_totals"]
    L += ["", "## Sanity checks", "",
          f"* full_phi reproduces FINAL targeted Hybrid: {tot['adv_identical']:,} / "
          f"{tot['flows']:,} realized flows identical, {tot['pred_identical']:,} predictions, "
          f"{tot['valid_identical']:,} validator verdicts, {tot['success_identical']:,} valid-"
          "success outcomes (p75 vs `primattack_hybrid_objective_targeted`, unbounded vs "
          "`primattack_targeted_budgets`).",
          f"* pairing: {checks['pairing']}",
          f"* primitive identity (stored primitives → φ / direct-only map): "
          f"{checks['primitive_identity']}",
          f"* capability inference: {checks['capability']}",
          f"* validator_v2 recomputed on stored flows: {checks['validator']}",
          f"* final counting (must be all 0): {checks['final_counting']}",
          f"* lost-success verdicts re-scored from stored vectors: {diag['verdicts']}",
          f"* all passed: **{checks['all_passed']}**", ""]
    (results_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def analyze(results_dir: Path, _reference: Path) -> None:
    mapping_all = json.loads((results_dir / "phi_mapping.json").read_text(encoding="utf-8"))
    mapping = next(iter(mapping_all.values()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = per_flow_table(results_dir)
    df.to_parquet(results_dir / "per_flow.parquet", index=False)
    per_seed = per_seed_metrics(df)
    per_seed.to_csv(results_dir / "per_seed.csv", index=False)
    agg = aggregate(per_seed)
    agg.to_csv(results_dir / "aggregate.csv", index=False)
    agg[["dataset", "victim", "budget"] + [c for c in agg.columns if c.startswith(
        ("p2_reduced_raw_asr", "p2_full_raw_asr", "p2_full_valid_asr", "reduced_to_full",
         "apparent", "prediction_changed", "full_hit_not_apparent"))]].to_csv(
        results_dir / "reduced_vs_full.csv", index=False)
    tests = mcnemar_tests(df)
    tests.to_csv(results_dir / "tests.csv", index=False)
    diag = diagnostics(results_dir, mapping, device)
    checks = sanity_checks(results_dir, mapping, device)
    audit, idents = reduced_flow_audit(results_dir, device)
    report(results_dir, mapping, agg, per_seed, tests, diag, checks, audit, idents)


def _write_mapping_first(args, conditions):
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if not args.skip_run or not (args.results_dir / "phi_mapping.json").exists():
        write_mapping(args.results_dir, datasets)
    return conditions


if __name__ == "__main__":
    experiment_main(EXP_DIR, CONDITIONS, analyze, __doc__, select=_write_mapping_first,
                    objective="targeted")
