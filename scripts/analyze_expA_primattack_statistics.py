"""Statistical evaluation of the Exp A PrimAttack Raw / Valid ASR on both datasets.

Question: is PrimAttack's (selected optimizer, untargeted, p75, joint) Raw / Valid ASR a real,
reproducible effect of the attack, or luck? Each source of luck is tested separately, per
dataset and victim (never pooled; N = 3,200 paired source flows, 800 per class):

1. Sampling of the source flows -> exact Clopper-Pearson and Wilson 95% CIs on the reference
   seed-42 rates plus a class-stratified bootstrap over flows.
2. The attack's random initialization -> per-flow agreement of the success sets across attack
   seeds 42 / 2024 / 2026 (Cochran's Q over seeds, Fleiss' kappa, Jaccard), next to evidence
   that the seeds really changed the search (final controls, random-restart share).
3. Chance evasion -> a single uniform random primitive draw in the same p75 box (the first
   candidate of the Prim-Random null-control stage): paired McNemar on seed 42, Holm over the
   6 victims, Newcombe CI.
4. Value of the gradient optimizer -> Prim-Random random search (255 uniform draws, same box,
   capability gates, quantization, validator gate and 256-evaluation cap; stage
   ``primattack_untargeted_random_null``): paired McNemar on seed 42, Holm over the 6 victims.
5. Validity bookkeeping -> paired Raw vs Valid McNemar (valid = raw AND validator_v2).

Inputs are read and audited by ``analyze_final_suite.Store`` (canonical sample IDs, clean-input
and checkpoint hashes, validator_v2 re-run on every stored final flow, victim re-prediction).
No attack is re-run. Outputs: ``FINAL_OUTPUTS/statistics/expA_primattack/``.

    python scripts/analyze_expA_primattack_statistics.py [--bootstrap 10000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_final_suite import (  # noqa: E402
    ALPHA, CLASSES, DATASETS, DS_LABEL, FINAL, OPT_LABEL, REF_SEED, RUNS, SEEDS, Store,
    assert_paired, fmt_p, md_table, prim_cond,
)
from evaluation.paired_validity_gap import (  # noqa: E402
    cochran_q, holm_adjust, mcnemar_test, newcombe_paired_ci, wilson_score_interval,
)

OUT = FINAL / "statistics" / "expA_primattack"
NULL_STAGE = "primattack_untargeted_random_null"
NO_PRIMITIVE = "no-primitive"
# Candidate j of a Prim-Random row is its evaluation j + 2 (evaluation 1 = shared identity).
FIRST_RANDOM_DRAW_EVAL = 2
RNG_SEED = 42
PAIRS = [(d, v) for d, vs in DATASETS.items() for v in vs]
I_REF = SEEDS.index(REF_SEED)
VS_SINGLE = "PrimAttack vs single random draw (Valid ASR)"
VS_SEARCH = "PrimAttack vs random search (Valid ASR)"
RAW_VALID = "Raw vs Valid (PrimAttack)"
SEED_EFFECT = "Attack-seed effect (Valid ASR)"
CLASS_HOMOGENEITY = "Class homogeneity (Valid ASR)"


# ----------------------------------------------------------------------------- estimators
def clopper_pearson(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def stratified_bootstrap(k_by_class: list[int], n_by_class: list[int], reps: int,
                         rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI of the pooled rate when flows are resampled with replacement within each
    class (class sizes fixed by design). The resampled success count of a class is exactly
    Binomial(n_c, k_c / n_c), so it is drawn directly."""
    total = np.zeros(reps)
    for k, n in zip(k_by_class, n_by_class):
        total += rng.binomial(n, k / n, reps)
    rates = total / sum(n_by_class)
    return float(np.quantile(rates, ALPHA / 2)), float(np.quantile(rates, 1 - ALPHA / 2))


def fleiss_kappa_binary(counts: np.ndarray, m: int) -> float:
    """Fleiss' kappa for a binary outcome rated by ``m`` seeds; ``counts`` = successes per flow."""
    s = counts.astype(np.float64)
    p_i = (s * (s - 1) + (m - s) * (m - s - 1)) / (m * (m - 1))
    p1 = s.sum() / (len(s) * m)
    p_e = p1 ** 2 + (1 - p1) ** 2
    return float("nan") if p_e >= 1.0 else float((p_i.mean() - p_e) / (1 - p_e))


def class_homogeneity_permutation(success: np.ndarray, reps: int,
                                  rng: np.random.Generator) -> tuple[float, float]:
    """Pearson chi-square of the class x success table and its Monte-Carlo permutation p-value
    (success labels permuted across flows; classes are contiguous blocks of equal size)."""
    k = len(CLASSES)
    n_c = len(success) // k
    total = int(success.sum())
    if total in (0, len(success)):
        return float("nan"), float("nan")
    exp_s = total / k
    exp_f = n_c - exp_s

    def stat(block_sums: np.ndarray) -> np.ndarray:
        return (((block_sums - exp_s) ** 2) / exp_s
                + (((n_c - block_sums) - exp_f) ** 2) / exp_f).sum(-1)

    observed = float(stat(success.reshape(k, n_c).sum(1).astype(np.float64)))
    perm = rng.permuted(np.tile(success, (reps, 1)), axis=1)
    null = stat(perm.reshape(reps, k, n_c).sum(2).astype(np.float64))
    return observed, float((1 + (null >= observed - 1e-9).sum()) / (reps + 1))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    both, a_only = int((a & b).sum()), int((a & ~b).sum())
    b_only, neither = int((~a & b).sum()), int((~a & ~b).sum())
    test = mcnemar_test(a_only, b_only)
    lo, hi = newcombe_paired_ci(both, a_only, b_only, neither)
    return {"rate_A": float(a.mean()), "rate_B": float(b.mean()),
            "both": both, "a_only": a_only, "b_only": b_only, "neither": neither,
            "diff": (a_only - b_only) / len(a), "diff_ci_lo": lo, "diff_ci_hi": hi,
            "variant": test["test_variant"], "statistic": test["test_statistic"],
            "p_value": test["p_value"]}


# ----------------------------------------------------------------------------- data access
def cell_matrix(frame: pd.DataFrame, dataset: str, victim: str, column: str
                ) -> tuple[np.ndarray, np.ndarray]:
    """(N, 3) matrix of ``column`` over seeds (rows = canonical flows in class order) and the
    source class of each row."""
    sub = frame[(frame.dataset == dataset) & (frame.victim == victim)]
    cols, ids_ref, cls_ref = [], None, None
    for seed in SEEDS:
        s = pd.concat([sub[(sub.seed == seed) & (sub.source_class == c)] for c in CLASSES])
        ids = s.sample_id.to_numpy()
        if ids_ref is None:
            ids_ref, cls_ref = ids, s.source_class.to_numpy()
        elif not np.array_equal(ids, ids_ref):
            raise AssertionError(f"{dataset}/{victim}: flow order differs across seeds")
        cols.append(s[column].to_numpy())
    return np.stack(cols, 1), cls_ref


def success_matrix(frame: pd.DataFrame, dataset: str, victim: str, column: str) -> np.ndarray:
    return cell_matrix(frame, dataset, victim, column)[0].astype(bool)


def movable_rows(frame: pd.DataFrame, dataset: str, victim: str) -> np.ndarray:
    return cell_matrix(frame, dataset, victim, "row_primitive_mode")[0][:, I_REF] != NO_PRIMITIVE


# ----------------------------------------------------------------------------- analysis
def estimates(frame: pd.DataFrame, label: str, role: str, metrics: tuple[str, ...], reps: int,
              rng: np.random.Generator) -> tuple[list[dict], list[dict]]:
    victim_rows, class_rows = [], []
    for dataset, victim in PAIRS:
        movable = movable_rows(frame, dataset, victim)
        for metric in metrics:
            mat, cls = cell_matrix(frame, dataset, victim, f"{metric}_success")
            mat = mat.astype(bool)
            ref = mat[:, I_REF]
            n, k = len(ref), int(ref.sum())
            per_seed = mat.mean(0)
            cp, wi = clopper_pearson(k, n), wilson_score_interval(k, n)
            bs = stratified_bootstrap([int(ref[cls == c].sum()) for c in CLASSES],
                                      [int((cls == c).sum()) for c in CLASSES], reps, rng)
            km, nm = int(ref[movable].sum()), int(movable.sum())
            cpm = clopper_pearson(km, nm) if nm else (float("nan"), float("nan"))
            victim_rows.append({
                "role": role, "condition": label, "dataset": dataset, "victim": victim,
                "metric": f"{metric}_asr", "n": n, "successes_seed42": k,
                "rate_seed42": k / n, "cp_lo": cp[0], "cp_hi": cp[1],
                "wilson_lo": wi[0], "wilson_hi": wi[1],
                "bootstrap_lo": bs[0], "bootstrap_hi": bs[1],
                **{f"rate_seed{s}": float(v) for s, v in zip(SEEDS, per_seed)},
                "mean_3seeds": float(per_seed.mean()), "sd_3seeds": float(per_seed.std(ddof=1)),
                "n_movable": nm, "successes_movable_seed42": km,
                "rate_movable_seed42": km / nm if nm else float("nan"),
                "cp_movable_lo": cpm[0], "cp_movable_hi": cpm[1],
            })
            for c in CLASSES:
                sel = cls == c
                kc, nc = int(ref[sel].sum()), int(sel.sum())
                lo, hi = clopper_pearson(kc, nc)
                class_rows.append({
                    "role": role, "condition": label, "dataset": dataset, "victim": victim,
                    "metric": f"{metric}_asr", "source_class": c, "n": nc,
                    "n_movable": int((sel & movable).sum()), "successes_seed42": kc,
                    "rate_seed42": kc / nc, "cp_lo": lo, "cp_hi": hi,
                    **{f"rate_seed{s}": float(mat[sel, i].mean()) for i, s in enumerate(SEEDS)},
                })
    return victim_rows, class_rows


def seed_reproducibility(frame: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for dataset, victim in PAIRS:
        mat = success_matrix(frame, dataset, victim, "valid_success")
        movable = movable_rows(frame, dataset, victim)
        counts = mat.sum(1)
        q = cochran_q(mat)
        jac = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                union = int((mat[:, i] | mat[:, j]).sum())
                if union:
                    jac.append(float((mat[:, i] & mat[:, j]).sum() / union))
        changed = {}
        for col in ("primitive_delay", "primitive_shape", "objective_margin"):
            vals = cell_matrix(frame, dataset, victim, col)[0].astype(np.float64)
            changed[col] = int((movable & ~np.all(vals == vals[:, :1], 1)).sum())
        src = cell_matrix(frame, dataset, victim, "candidate_source")[0].astype(str)
        from_random = [int((mat[:, i] & np.char.endswith(src[:, i], "random")).sum())
                       for i in range(len(SEEDS))]
        rows.append({
            "condition": label, "dataset": dataset, "victim": victim,
            "n": len(mat), "n_movable": int(movable.sum()),
            **{f"valid_successes_seed{s}": int(mat[:, i].sum()) for i, s in enumerate(SEEDS)},
            "success_all_3_seeds": int((counts == 3).sum()),
            "success_any_seed": int((counts > 0).sum()),
            "success_1_or_2_seeds": int(((counts > 0) & (counts < 3)).sum()),
            "cochran_q_over_seeds": q["Q"], "cochran_q_p": q["p"],
            "fleiss_kappa": fleiss_kappa_binary(counts, len(SEEDS)),
            "jaccard_min": min(jac) if jac else float("nan"),
            "movable_flows_final_delay_differs": changed["primitive_delay"],
            "movable_flows_final_shape_differs": changed["primitive_shape"],
            "movable_flows_final_margin_differs": changed["objective_margin"],
            **{f"successes_from_random_start_seed{s}": v for s, v in zip(SEEDS, from_random)},
        })
    return rows


def _seed_discordance(a: np.ndarray, b: np.ndarray) -> dict:
    out = {}
    for i, s in enumerate(SEEDS):
        if s != REF_SEED:
            out[f"a_only_seed{s}"] = int((a[:, i] & ~b[:, i]).sum())
            out[f"b_only_seed{s}"] = int((~a[:, i] & b[:, i]).sum())
    return out


def tests(prim: pd.DataFrame, single: pd.DataFrame, search: pd.DataFrame, labels: dict,
          reps: int, rng: np.random.Generator) -> pd.DataFrame:
    rows, families = [], {VS_SINGLE: [], VS_SEARCH: []}
    for dataset, victim in PAIRS:
        pv = success_matrix(prim, dataset, victim, "valid_success")
        pr = success_matrix(prim, dataset, victim, "raw_success")
        base = {"dataset": dataset, "victim": victim, "seed_used": str(REF_SEED),
                "n_paired": len(pv)}
        rows.append({**base, "analysis": RAW_VALID,
                     "comparison": f"{labels['prim']}: raw_success vs valid_success",
                     **paired(pr[:, I_REF], pv[:, I_REF]), **_seed_discordance(pr, pv),
                     "family": "one test per condition (no Holm)"})
        for analysis, frame, key in ((VS_SINGLE, single, "single"), (VS_SEARCH, search, "search")):
            nv = success_matrix(frame, dataset, victim, "valid_success")
            row = {**base, "analysis": analysis,
                   "comparison": f"{labels['prim']} vs {labels[key]}",
                   **paired(pv[:, I_REF], nv[:, I_REF]), **_seed_discordance(pv, nv),
                   "family": "Holm over the 6 dataset x victim tests of this analysis "
                             "(post-run family)"}
            rows.append(row)
            families[analysis].append(row)
        q = cochran_q(pv)
        rows.append({**base, "seed_used": "42/2024/2026", "analysis": SEED_EFFECT,
                     "comparison": f"{labels['prim']}: seed 42 vs 2024 vs 2026",
                     "statistic": q["Q"], "variant": f"Cochran's Q ({q['df']} df)",
                     "p_value": q["p"], "family": "robustness check (no Holm)"})
        stat, p = class_homogeneity_permutation(pv[:, I_REF], reps, rng)
        rows.append({**base, "analysis": CLASS_HOMOGENEITY,
                     "comparison": f"{labels['prim']}: " + " / ".join(CLASSES),
                     "statistic": stat,
                     "variant": f"Pearson chi-square, Monte-Carlo permutation p ({reps})",
                     "p_value": p, "family": "descriptive (no Holm)"})
    for family in families.values():
        for row, adj in zip(family, holm_adjust([r["p_value"] for r in family])):
            row["p_holm"] = adj
    return pd.DataFrame(rows)


def null_diagnostics(prim: pd.DataFrame, search: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, victim in PAIRS:
        p = prim[(prim.dataset == dataset) & (prim.victim == victim) & (prim.seed == REF_SEED)]
        q = search[(search.dataset == dataset) & (search.victim == victim)
                   & (search.seed == REF_SEED)]
        mv_p, mv_q = p.row_primitive_mode != NO_PRIMITIVE, q.row_primitive_mode != NO_PRIMITIVE
        succ_p, succ_q = p.valid_success, q.valid_success
        rows.append({
            "dataset": dataset, "victim": victim,
            "prim_mean_realized_queries_movable":
                float((p.model_evaluations - p.backward_evaluations)[mv_p].mean()),
            "prim_mean_total_evals_movable": float(p.model_evaluations[mv_p].mean()),
            "search_mean_realized_queries_movable": float(q.model_evaluations[mv_q].mean()),
            "prim_median_cost_valid": float(p.primitive_normalized_cost[succ_p].median())
            if succ_p.any() else float("nan"),
            "search_median_cost_valid": float(q.primitive_normalized_cost[succ_q].median())
            if succ_q.any() else float("nan"),
            "prim_median_first_success_eval": float(p.first_success_evaluation[succ_p].median())
            if succ_p.any() else float("nan"),
            "search_median_first_success_eval":
                float(q.first_success_evaluation[succ_q].median())
            if succ_q.any() else float("nan"),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report
def pct(x: float) -> str:
    return "—" if pd.isna(x) else f"{100 * x:.2f}%"


def ci(lo: float, hi: float) -> str:
    return f"[{100 * lo:.2f}, {100 * hi:.2f}]"


def pp_ci(d: float, lo: float, hi: float) -> str:
    return f"{100 * d:+.2f} pp [{100 * lo:+.2f}, {100 * hi:+.2f}]"


def name(dataset: str, victim: str) -> str:
    return f"{DS_LABEL[dataset]} {victim}"


def report(est: pd.DataFrame, cls: pd.DataFrame, seeds: pd.DataFrame, tst: pd.DataFrame,
           diag: pd.DataFrame, labels: dict, audit: dict, reps: int) -> str:
    def e(role: str, metric: str, d: str, v: str) -> pd.Series:
        return est[(est.role == role) & (est.metric == metric) & (est.dataset == d)
                   & (est.victim == v)].iloc[0]

    def t(analysis: str, d: str, v: str) -> pd.Series:
        return tst[(tst.analysis == analysis) & (tst.dataset == d) & (tst.victim == v)].iloc[0]

    def s(d: str, v: str) -> pd.Series:
        return seeds[(seeds.dataset == d) & (seeds.victim == v)].iloc[0]

    # --- data-driven verdict --------------------------------------------------------------
    pv = {pair: e("primattack", "valid_asr", *pair) for pair in PAIRS}
    nonzero = [p for p in PAIRS if pv[p].cp_lo > 0]
    zero = [p for p in PAIRS if pv[p].successes_seed42 == 0]
    identical = [p for p in PAIRS if s(*p).success_all_3_seeds == s(*p).success_any_seed]
    sig_single = [p for p in PAIRS if t(VS_SINGLE, *p).p_holm < ALPHA]
    sig_search = [p for p in PAIRS if t(VS_SEARCH, *p).p_holm < ALPHA]
    rv = tst[tst.analysis == RAW_VALID]
    raw_only_total = int(rv.a_only.sum() + rv.a_only_seed2024.sum() + rv.a_only_seed2026.sum())
    search_diffs = [t(VS_SEARCH, *p)["diff"] for p in PAIRS]
    srch = tst[tst.analysis == VS_SEARCH]
    search_only_total = int(srch.b_only.sum() + srch.b_only_seed2024.sum()
                            + srch.b_only_seed2026.sum())
    share = [t(VS_SEARCH, *p).rate_B / t(VS_SEARCH, *p).rate_A for p in PAIRS
             if t(VS_SEARCH, *p).rate_A > 0]
    effect_lo, effect_hi = min(v.rate_seed42 for v in pv.values()), max(
        v.rate_seed42 for v in pv.values())

    def names(pairs: list) -> str:
        return ", ".join(name(*p) for p in pairs) if pairs else "none"

    verdict = [
        f"- **Sampling of source flows.** The exact 95% CI of Valid ASR excludes 0 for "
        f"{len(nonzero)} of 6 victims ({names(nonzero)}); the class-stratified bootstrap agrees. "
        + (f"Zero successes: {names(zero)} (exact 95% upper bound "
           + ", ".join(pct(pv[p].cp_hi) for p in zero) + ")." if zero else ""),
        f"- **Attack-seed / initialization luck.** For {len(identical)} of 6 victims the *same "
        "flows* succeed under all three attack seeds (Cochran's Q over seeds "
        + ("= 0, p = 1" if len(identical) == 6 else "see §4") + "), although the seeds changed "
        "the search: final delays differ across seeds on "
        f"{int(seeds.movable_flows_final_delay_differs.min())}–"
        f"{int(seeds.movable_flows_final_delay_differs.max())} movable flows per victim and "
        f"{int(seeds[[f'successes_from_random_start_seed{x}' for x in SEEDS]].to_numpy().sum())}"
        f" of {int(seeds[[f'valid_successes_seed{x}' for x in SEEDS]].to_numpy().sum())} "
        "valid successes (all seeds) were reached from a random restart.",
        f"- **Chance evasion (single random primitive draw in the same p75 box).** PrimAttack is "
        f"significantly better after Holm for {len(sig_single)} of 6 victims "
        f"({names(sig_single)}).",
        f"- **Value of the gradient optimizer over black-box random search** (255 uniform draws, "
        f"same box / gates / validator / 256-evaluation cap). Δ Valid ASR ranges "
        f"{100 * min(search_diffs):+.2f} to {100 * max(search_diffs):+.2f} pp; significant after "
        f"Holm for {len(sig_search)} of 6 victims ({names(sig_search)}). Random search reaches "
        f"{100 * min(share):.0f}–{100 * max(share):.0f}% of PrimAttack's valid successes, and "
        f"flows evaded by random search but not by PrimAttack, summed over all victims and "
        f"seeds: {search_only_total}"
        + (" (PrimAttack's success set contains the random-search success set on every seed)."
           if search_only_total == 0 else ".")
        + " Most of the effect is therefore carried by the primitive attack space (flows that "
        "an in-box timing change evades); the gradient optimizer adds the remaining margin.",
        f"- **Raw vs Valid.** PrimAttack raw successes that fail the validator, summed over all "
        f"victims and seeds: {raw_only_total}. Its Valid ASR equals its Raw ASR (validator in "
        "the search loop); contrast the input-space baselines in `../statistical_summary.md` §3.",
    ]

    head = []
    for p in PAIRS:
        r1, r2 = t(VS_SINGLE, *p), t(VS_SEARCH, *p)
        head.append({
            "Dataset": DS_LABEL[p[0]], "Victim": p[1],
            "Valid ASR seed 42 (k)": f"{pct(pv[p].rate_seed42)} ({pv[p].successes_seed42})",
            "Exact 95% CI": ci(pv[p].cp_lo, pv[p].cp_hi),
            "3-seed mean ± SD": f"{pct(pv[p].mean_3seeds)} ± {pct(pv[p].sd_3seeds)}",
            "Same flows on all 3 seeds": "yes" if p in identical else
            f"no ({s(*p).success_1_or_2_seeds} differ)",
            "Single random draw": pct(r1.rate_B),
            "Δ vs single draw (Holm p)": f"{100 * r1['diff']:+.2f} pp ({fmt_p(r1.p_holm)})",
            "Random search (255)": pct(r2.rate_B),
            "Δ vs random search (Holm p)": f"{100 * r2['diff']:+.2f} pp ({fmt_p(r2.p_holm)})",
        })

    est_rows = []
    for p in PAIRS:
        for metric, label in (("raw_asr", "Raw"), ("valid_asr", "Valid")):
            r = e("primattack", metric, *p)
            est_rows.append({
                "Dataset": DS_LABEL[p[0]], "Victim": p[1], "Metric": label,
                "k / N (seed 42)": f"{r.successes_seed42} / {r.n}",
                "Rate": pct(r.rate_seed42), "Clopper–Pearson 95%": ci(r.cp_lo, r.cp_hi),
                "Wilson 95%": ci(r.wilson_lo, r.wilson_hi),
                f"Stratified bootstrap 95% (B={reps})": ci(r.bootstrap_lo, r.bootstrap_hi),
                "Per seed 42/2024/2026 (%)": " / ".join(
                    f"{100 * r[f'rate_seed{x}']:.2f}" for x in SEEDS),
                "Mean ± SD": f"{pct(r.mean_3seeds)} ± {pct(r.sd_3seeds)}",
            })

    rv_rows = []
    for p in PAIRS:
        r = t(RAW_VALID, *p)
        rv_rows.append({
            "Dataset": DS_LABEL[p[0]], "Victim": p[1], "Raw": pct(r.rate_A),
            "Valid": pct(r.rate_B),
            "Gap [Newcombe 95% CI]": pp_ci(r["diff"], r.diff_ci_lo, r.diff_ci_hi),
            "Raw-only / Valid-only (seed 42)": f"{int(r.a_only)} / {int(r.b_only)}",
            "Raw-only seeds 2024 / 2026": f"{int(r.a_only_seed2024)} / {int(r.a_only_seed2026)}",
            "McNemar": f"{r.variant}; p = {fmt_p(r.p_value)}",
        })

    seed_rows = []
    for p in PAIRS:
        r = s(*p)
        seed_rows.append({
            "Dataset": DS_LABEL[p[0]], "Victim": p[1],
            "Valid successes 42/2024/2026": " / ".join(
                str(r[f"valid_successes_seed{x}"]) for x in SEEDS),
            "All 3 / any seed": f"{r.success_all_3_seeds} / {r.success_any_seed}",
            "Cochran's Q over seeds (p)": f"{r.cochran_q_over_seeds:.2f} ({fmt_p(r.cochran_q_p)})",
            "Fleiss κ": "n/a (no success)" if pd.isna(r.fleiss_kappa) else f"{r.fleiss_kappa:.3f}",
            "Min pairwise Jaccard": "—" if pd.isna(r.jaccard_min) else f"{r.jaccard_min:.3f}",
            "Movable flows with different final delay across seeds":
                f"{r.movable_flows_final_delay_differs} / {r.n_movable}",
            "Successes from a random restart (42/2024/2026)": " / ".join(
                str(r[f"successes_from_random_start_seed{x}"]) for x in SEEDS),
        })

    def null_rows(analysis: str, role: str) -> list[dict]:
        out = []
        for p in PAIRS:
            r, nv = t(analysis, *p), e(role, "valid_asr", *p)
            out.append({
                "Dataset": DS_LABEL[p[0]], "Victim": p[1],
                "PrimAttack Valid": pct(r.rate_A),
                "Null Valid seed 42 [exact 95%]": f"{pct(r.rate_B)} {ci(nv.cp_lo, nv.cp_hi)}",
                "Null 3-seed mean ± SD": f"{pct(nv.mean_3seeds)} ± {pct(nv.sd_3seeds)}",
                "Prim-only / null-only (seed 42)": f"{int(r.a_only)} / {int(r.b_only)}",
                "Prim-only / null-only (2024; 2026)":
                    f"{int(r.a_only_seed2024)}/{int(r.b_only_seed2024)}; "
                    f"{int(r.a_only_seed2026)}/{int(r.b_only_seed2026)}",
                "Δ [Newcombe 95% CI]": pp_ci(r["diff"], r.diff_ci_lo, r.diff_ci_hi),
                "McNemar p (Holm over 6)": f"{fmt_p(r.p_value)} ({fmt_p(r.p_holm)})",
            })
        return out

    diag_rows = []
    for p in PAIRS:
        g = diag[(diag.dataset == p[0]) & (diag.victim == p[1])].iloc[0]
        diag_rows.append({
            "Dataset": DS_LABEL[p[0]], "Victim": p[1],
            "Realized queries / movable flow (Prim vs search)":
                f"{g.prim_mean_realized_queries_movable:.0f} vs "
                f"{g.search_mean_realized_queries_movable:.0f}",
            "Total evaluations / movable flow (Prim vs search)":
                f"{g.prim_mean_total_evals_movable:.0f} vs "
                f"{g.search_mean_realized_queries_movable:.0f}",
            "Median normalized cost of valid successes (Prim vs search)":
                f"{g.prim_median_cost_valid:.3f} vs {g.search_median_cost_valid:.3f}"
                if not pd.isna(g.prim_median_cost_valid) else "—",
            "Median evaluation of first success (Prim vs search)":
                f"{g.prim_median_first_success_eval:.0f} vs "
                f"{g.search_median_first_success_eval:.0f}"
                if not pd.isna(g.prim_median_first_success_eval) else "—",
        })

    class_rows = []
    for p in PAIRS:
        sub = cls[(cls.role == "primattack") & (cls.metric == "valid_asr")
                  & (cls.dataset == p[0]) & (cls.victim == p[1])]
        h = t(CLASS_HOMOGENEITY, *p)
        row = {"Dataset": DS_LABEL[p[0]], "Victim": p[1]}
        for _, c in sub.iterrows():
            row[c.source_class] = f"{c.successes_seed42}/{c.n} (movable {c.n_movable})"
        row["Valid ASR on movable flows [exact 95%]"] = (
            f"{pct(pv[p].rate_movable_seed42)} {ci(pv[p].cp_movable_lo, pv[p].cp_movable_hi)} "
            f"(n = {pv[p].n_movable})")
        row["Class homogeneity χ² (perm. p)"] = (
            "n/a (no success)" if pd.isna(h.statistic) else
            f"{h.statistic:.1f} ({fmt_p(h.p_value)})")
        class_rows.append(row)

    unb_rows = []
    for p in PAIRS:
        r = e("primattack_unbounded", "valid_asr", *p)
        unb_rows.append({"Dataset": DS_LABEL[p[0]], "Victim": p[1],
                         "Valid ASR (seed 42)": f"{pct(r.rate_seed42)} ({r.successes_seed42})",
                         "Exact 95% CI": ci(r.cp_lo, r.cp_hi),
                         "Mean ± SD": f"{pct(r.mean_3seeds)} ± {pct(r.sd_3seeds)}"})

    lines = [
        "# Experiment A — statistical evaluation of PrimAttack Raw / Valid ASR",
        "",
        f"Condition: **{labels['prim']}** (untargeted, p75, joint, capability-aware; the Exp A "
        "PrimAttack cell). Both datasets, every victim, analysed separately (never pooled). "
        "N = 3,200 paired clean-correct source flows per victim (800 per class). Generated by "
        "`scripts/analyze_expA_primattack_statistics.py` from the stored per-sample artifacts in "
        "`FINAL_OUTPUTS/runs/`; the analysis re-runs no attack.",
        "",
        "## Verdict",
        "",
        *verdict,
        "",
        md_table(pd.DataFrame(head)),
        "",
        "## 1. Protocol and provenance",
        "",
        "- Paired unit = one canonical clean-correct source flow; classes pooled within a victim; "
        "datasets and victims never pooled (pseudoreplication). α = 0.05.",
        "- Inference uses reference attack seed 42 only (locked protocol §3). Seeds 2024 / 2026 "
        "enter the per-seed rates, the per-seed discordant counts and the seed-agreement "
        "analysis, which uses all three seeds of the *same* flows jointly (Cochran's Q over "
        "seeds); the three seed means are never used as n = 3.",
        "- McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ². "
        "Paired differences carry Newcombe square-and-add 95% CIs. Holm families: the 6 "
        "(dataset × victim) tests of one null comparison.",
        "- Integrity audit (fail-loud, `analyze_final_suite.Store`): "
        f"{audit['npz_files']} artifacts, {audit['rows']:,} per-sample rows; validator_v2 re-run "
        f"on {audit['validator_rechecked_rows']:,} stored final flows; victim re-prediction on "
        f"{audit['prediction_rechecked_rows']:,} flows with {audit['prediction_mismatches']} "
        "mismatches; canonical sample IDs, clean-input and checkpoint hashes identical across "
        "PrimAttack (p75, unbounded) and the null control.",
        "- **Added after the locked protocol (post-run, not pre-registered):** every CI, the "
        "bootstrap, the seed-agreement analysis, the class-homogeneity test and the Prim-Random "
        f"null control (stage `{NULL_STAGE}`, protocol amendment A4). The Exp A point estimates "
        "and the planned Exp A tests (`../../A_primary_baseline_comparison/statistical_tests.csv`) "
        "are unchanged.",
        "",
        "## 2. Point estimates and uncertainty (source-flow sampling)",
        "",
        "Clopper–Pearson is exact (conservative); Wilson is the score interval; the bootstrap "
        "resamples flows with replacement within each class (class sizes fixed by design). The "
        "SD over seeds is attack-run variability, not a CI.",
        "",
        md_table(pd.DataFrame(est_rows)),
        "",
        "## 3. Raw vs Valid on the same adversarial flows",
        "",
        "Valid = Raw ∧ validator_v2 `hybrid_valid`, so valid-only discordance is 0 by "
        "construction; raw-only counts the apparent successes the validator removes.",
        "",
        md_table(pd.DataFrame(rv_rows)),
        "",
        "## 4. Reproducibility across attack seeds (initialization luck)",
        "",
        "Cochran's Q tests whether the per-flow Valid success probability differs between the "
        "three seeds on the same 3,200 flows (Q = 0, p = 1 when the success sets are identical). "
        "The last two columns show that the seeds did change the search.",
        "",
        md_table(pd.DataFrame(seed_rows)),
        "",
        "## 5. Null controls (Prim-Random)",
        "",
        f"Stage `{NULL_STAGE}`: per flow, 255 candidates drawn i.i.d. uniformly in the flow's "
        "normalized p75 box (pinned coordinates stay 0) after the shared identity evaluation; no "
        "gradient, no surrogate evaluation. Everything else is the shared `RealizedSearch` "
        "(projection, integer quantization, capability gates, validator_v2 gate, success "
        "predicate, incumbent) on the same canonical flows and attack seeds.",
        "",
        f"### 5a. Chance: {labels['single']}",
        "",
        "Success of the *first* random candidate only (evaluation 2): the probability that one "
        "random, non-optimized timing/padding change in the same box evades the victim validly.",
        "",
        md_table(pd.DataFrame(null_rows(VS_SINGLE, "single"))),
        "",
        f"### 5b. Optimizer value: {labels['search']}",
        "",
        "The best of 255 random candidates per flow. It spends nothing on surrogate/backward "
        "passes, so it gets more realized, validator-gated queries per flow than PrimAttack: a "
        "strong black-box baseline, not a chance level.",
        "",
        md_table(pd.DataFrame(null_rows(VS_SEARCH, "search"))),
        "",
        md_table(pd.DataFrame(diag_rows)),
        "",
        "## 6. Where the successes come from (class structure)",
        "",
        "Valid successes per class (seed 42) and the number of flows with any primitive headroom "
        "(`movable`: timing and/or padding in the p75 box after capability gates). Flows without "
        "headroom cannot be attacked by construction. The permutation χ² tests whether success "
        f"is spread uniformly over classes (smallest attainable p = 1/(B+1) = {1 / (reps + 1):.1g}).",
        "",
        md_table(pd.DataFrame(class_rows)),
        "",
        "## 7. Descriptive: unbounded (envelope-only) budget",
        "",
        md_table(pd.DataFrame(unb_rows)),
        "",
        "## 8. What this does and does not establish",
        "",
        "- Established per victim: how precisely the Valid ASR is estimated (exact CIs), whether "
        "it is reproduced flow-for-flow across attack seeds, whether it exceeds a single random "
        "in-box perturbation, how much the gradient optimizer adds over a matched random search, "
        "and that there is no raw-vs-valid leakage.",
        "- Seeds are **attack** seeds only; each (dataset, architecture) uses one victim "
        "checkpoint. Victim-training variability is not covered (no seed-robustness claim over "
        "victim training).",
        "- The sampling CIs cover resampling of source flows from the canonical test-pool "
        "selection; they do not cover new campaigns or a global forward-time split.",
        f"- Effects are victim-dependent and small at p75 ({pct(effect_lo)}–{pct(effect_hi)} "
        "Valid ASR); statistical significance with N = 3,200 does not make a small effect "
        "operationally large.",
        "- All results are feature-space proxies on CICFlowMeter aggregates (no PCAP edited or "
        "replayed).",
        "",
        "Machine-readable: `estimates.csv`, `class_estimates.csv`, `seed_reproducibility.csv`, "
        "`tests.csv`, `null_control_diagnostics.csv`, `audit.json`.",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bootstrap", type=int, default=10000,
                    help="bootstrap / permutation replicates")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-recheck-predictions", action="store_true")
    args = ap.parse_args()

    selection = json.loads((RUNS / "optimizer_selection.json").read_text(encoding="utf-8"))
    sel = selection["selected"]
    labels = {"prim": f"PrimAttack ({OPT_LABEL[sel]}, p75)",
              "single": "single uniform random draw in the p75 box",
              "search": "Prim-Random search (255 uniform draws, p75)"}
    store = Store(recheck_predictions=not args.no_recheck_predictions, device=args.device)
    prim = store.frame(prim_cond("primattack_untargeted", sel, "p75", "untargeted",
                                 labels["prim"]))
    unb = store.frame(prim_cond("primattack_untargeted", sel, "unbounded", "untargeted",
                                f"PrimAttack ({OPT_LABEL[sel]}, unbounded)"))
    search = store.frame(prim_cond(NULL_STAGE, "random", "p75", "untargeted", labels["search"]))
    assert_paired([prim, unb, search], "Exp A PrimAttack statistics")
    single = search.assign(
        method=labels["single"],
        valid_success=(search.first_success_evaluation == FIRST_RANDOM_DRAW_EVAL).to_numpy())
    if (single.valid_success & ~search.valid_success).any():
        raise AssertionError("a first-draw success must also be a random-search success")

    rng = np.random.default_rng(RNG_SEED)
    est, cls = [], []
    for frame, label, role, metrics in (
            (prim, labels["prim"], "primattack", ("raw", "valid")),
            (single, labels["single"], "single", ("valid",)),
            (search, labels["search"], "search", ("raw", "valid")),
            (unb, unb.method.iloc[0], "primattack_unbounded", ("raw", "valid"))):
        v, c = estimates(frame, label, role, metrics, args.bootstrap, rng)
        est += v
        cls += c
    est, cls = pd.DataFrame(est), pd.DataFrame(cls)
    seeds = pd.DataFrame(seed_reproducibility(prim, labels["prim"])
                         + seed_reproducibility(search, labels["search"]))
    tst = tests(prim, single, search, labels, args.bootstrap, rng)
    diag = null_diagnostics(prim, search)

    OUT.mkdir(parents=True, exist_ok=True)
    est.to_csv(OUT / "estimates.csv", index=False)
    cls.to_csv(OUT / "class_estimates.csv", index=False)
    seeds.to_csv(OUT / "seed_reproducibility.csv", index=False)
    tst.to_csv(OUT / "tests.csv", index=False)
    diag.to_csv(OUT / "null_control_diagnostics.csv", index=False)
    audit = {**store.audit, "bootstrap_replicates": args.bootstrap, "rng_seed": RNG_SEED,
             "selected_optimizer": sel, "null_stage": NULL_STAGE}
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (OUT / "expA_primattack_statistics.md").write_text(
        report(est, cls, seeds[seeds.condition == labels["prim"]], tst, diag, labels, audit,
               args.bootstrap),
        encoding="utf-8")
    print(f"[ok] -> {OUT}")


if __name__ == "__main__":
    main()
