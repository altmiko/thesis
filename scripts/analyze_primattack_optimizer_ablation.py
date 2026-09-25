"""Analyze the PrimAttack optimizer ablation (Hybrid vs Prim-PGD vs Prim-C&W).

Reads ``<root>/cells.json`` + per-row npz from ``run_primattack_optimizer_ablation.py`` and
writes ``<root>/analysis/``: ``cells_raw.csv``, ``aggregate_*.csv``, ``paired_tests.json``,
``tables.md`` (every table in markdown), and ``plots/*.png``.

Statistics: the paired unit is one source flow. Primary tests use the reference seed and pool
the four classes WITHIN a victim (victims and budgets are never pooled); the other seeds are
reported as replications. McNemar: exact binomial when b+c<25, else continuity-corrected
chi-square; Newcombe square-and-add paired 95% CI; Holm across the primary family. Cochran's Q
is the 3-method omnibus. Historical rows from the canonical campaign
(``outputs/adv_campaign_noidr``) share the frozen selection and are paired row-by-row.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from evaluation.paired_validity_gap import holm_adjust, mcnemar_test, newcombe_paired_ci  # noqa: E402

METHODS = ("hybrid", "pgd", "cw")
LABEL = {"hybrid": "Hybrid", "pgd": "Prim-PGD", "cw": "Prim-C&W",
         "hybrid_canonical": "Hybrid (R=2 prefix)", "prefix_hybrid": "Pre-fix Hybrid (campaign)",
         "rand1": "Random-feasible (1 draw, campaign)"}
COLOR = {"hybrid": "#1b9e77", "pgd": "#d95f02", "cw": "#7570b3"}
BUDGETS = ("p50", "p75", "unb")
CLASSES = ("DoS", "DDoS", "Recon", "BruteForce")
CONTRASTS = (("hybrid", "pgd"), ("hybrid", "cw"), ("pgd", "cw"))


class Rows:
    def __init__(self, root: Path):
        self.root = root
        self.cells = pd.DataFrame(json.loads((root / "cells.json").read_text(encoding="utf-8")))
        self.config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self._cache: dict = {}

    def load(self, victim, cls, budget, seed, method):
        key = (victim, cls, budget, seed, method)
        if key not in self._cache:
            path = self.root / "artifacts" / f"{victim}__{cls}__{budget}__{method}__seed{seed}.npz"
            with np.load(path, allow_pickle=False) as d:
                self._cache[key] = {k: d[k] for k in d.files}
        return self._cache[key]

    def outcome(self, victim, classes, budget, seed, method, *, canonical=False):
        parts = []
        for c in classes:
            d = self.load(victim, c, budget, seed, method)
            s = d["final_success"].astype(bool)
            if canonical:
                s = s & (d["first_success_phase"] < 2)
            parts.append(s)
        return np.concatenate(parts)

    def sample_ids(self, victim, classes, budget, seed, method):
        return np.concatenate([self.load(victim, c, budget, seed, method)["sample_id"]
                               for c in classes])


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    n11 = int((a & b).sum()); n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum()); n00 = int((~a & ~b).sum())
    test = mcnemar_test(n10, n01)
    lo, hi = newcombe_paired_ci(n11, n10, n01, n00)
    return {"n": int(a.size), "n11": n11, "A_only": n10, "B_only": n01, "n00": n00,
            "rate_A": float(a.mean()), "rate_B": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "ci95": [lo, hi],
            "test": test["test_variant"], "p": float(test["p_value"])}


def cochran_q(matrix: np.ndarray) -> dict:
    """Cochran's Q for k related binary samples; matrix is (n, k)."""
    x = matrix.astype(np.int64)
    k = x.shape[1]
    col, row = x.sum(0), x.sum(1)
    denom = k * row.sum() - (row ** 2).sum()
    if denom == 0:
        return {"Q": 0.0, "df": k - 1, "p": 1.0}
    q = (k - 1) * (k * (col ** 2).sum() - col.sum() ** 2) / denom
    return {"Q": float(q), "df": k - 1, "p": float(chi2.sf(q, k - 1))}


def pct(v) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{100 * v:.2f}%"


def num(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{digits}g}" if abs(v) < 1e5 else f"{v:.3e}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------
def unique_success_columns(rows: Rows, cells: pd.DataFrame) -> pd.DataFrame:
    uniq = []
    for _, c in cells.iterrows():
        mine = rows.load(c.victim, c["class"], c.budget_label, c.seed, c.method)["final_success"]
        others = [rows.load(c.victim, c["class"], c.budget_label, c.seed, m)["final_success"]
                  for m in METHODS if m != c.method and m in set(cells.method)]
        other_any = np.logical_or.reduce(others) if others else np.zeros_like(mine)
        uniq.append(int((mine & ~other_any).sum()))
    cells = cells.copy()
    cells["unique_successes"] = uniq
    return cells


def aggregate(rows: Rows, cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Victim x budget x method (classes pooled per seed) and per-class aggregates."""
    recs = []
    for (v, b, m), g in cells.groupby(["victim", "budget_label", "method"], sort=False):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            data = [rows.load(v, c, b, seed, m) for c in gs["class"]]
            cat = lambda k: np.concatenate([d[k] for d in data])  # noqa: E731
            s = cat("final_success").astype(bool)
            fs = cat("first_success_evaluation")
            per_seed.append({
                "asr": s.mean(), "raw": cat("targeted").mean(),
                "sp": (s & cat("primitive_feasible") & cat("semantic_pass")).mean(),
                "canon": (s & (cat("first_success_phase") < 2)).mean() if m == "hybrid" else np.nan,
                "cost_mean": cat("normalized_cost")[s].mean() if s.any() else np.nan,
                "cost_median": np.median(cat("normalized_cost")[s]) if s.any() else np.nan,
                "evals_mean": cat("total_evaluations").mean(),
                "evals_median": np.median(cat("total_evaluations")),
                "first_median": np.median(fs[fs > 0]) if (fs > 0).any() else np.nan,
                "inv": (cat("failure") == "invalid").mean(),
                "exh": (cat("failure") == "exhausted").mean(),
                "noh": (cat("failure") == "no_headroom").mean(),
                "runtime": gs.elapsed_seconds.sum(),
                "ms_per_flow": 1e3 * gs.elapsed_seconds.sum() / len(s),
                "n": len(s),
            })
        ps = pd.DataFrame(per_seed)
        rec = {"victim": v, "budget": b, "method": m, "n_seeds": len(ps), "n_per_seed": int(ps.n.iloc[0])}
        for k in ps.columns:
            if k == "n":
                continue
            rec[k] = ps[k].mean()
            rec[k + "_sd"] = ps[k].std(ddof=1) if len(ps) > 1 else np.nan
        recs.append(rec)
    agg = pd.DataFrame(recs)
    per_class = (cells.groupby(["victim", "class", "budget_label", "method"], sort=False)
                 .agg(asr=("asr_valid", "mean"), asr_sd=("asr_valid", "std"),
                      raw=("asr_raw", "mean"), sp=("sp_asr", "mean"),
                      cost_median=("cost_median", "mean"), evals_mean=("evals_mean", "mean"),
                      runtime=("elapsed_seconds", "mean"), unique=("unique_successes", "mean"),
                      n=("n", "first"), n_movable=("n_movable", "first"))
                 .reset_index())
    return agg, per_class


def paired_tests(rows: Rows, cells: pd.DataFrame, ref_seed: int) -> dict:
    victims = list(dict.fromkeys(cells.victim))
    seeds_by_victim = {v: sorted(cells[cells.victim == v].seed.unique()) for v in victims}
    methods = [m for m in METHODS if m in set(cells.method)]
    primary, per_class, omnibus, overlap = [], [], [], []
    for v in victims:
        seeds = seeds_by_victim[v]
        ref = ref_seed if ref_seed in seeds else seeds[0]
        for b in BUDGETS:
            if not ((cells.victim == v) & (cells.budget_label == b)).any():
                continue
            out = {m: rows.outcome(v, CLASSES, b, ref, m) for m in methods}
            ids = {m: rows.sample_ids(v, CLASSES, b, ref, m) for m in methods}
            for m in methods[1:]:
                assert np.array_equal(ids[m], ids[methods[0]]), "pairing broken"
            omnibus.append({"victim": v, "budget": b, "seed": int(ref),
                            **cochran_q(np.stack([out[m] for m in methods], 1))})
            H, P, C = (out.get(m) for m in ("hybrid", "pgd", "cw"))
            overlap.append({
                "victim": v, "budget": b, "seed": int(ref), "n": int(H.size),
                "all3": int((H & P & C).sum()), "hybrid_only": int((H & ~P & ~C).sum()),
                "pgd_only": int((~H & P & ~C).sum()), "cw_only": int((~H & ~P & C).sum()),
                "hybrid_pgd_only": int((H & P & ~C).sum()), "hybrid_cw_only": int((H & ~P & C).sum()),
                "pgd_cw_only": int((~H & P & C).sum()), "none": int((~H & ~P & ~C).sum()),
                "pairwise": {f"{a}&{bb}": int((out[a] & out[bb]).sum())
                             for a, bb in combinations(methods, 2)},
            })
            for a, bb in CONTRASTS:
                rec = {"victim": v, "budget": b, "seed": int(ref), "A": a, "B": bb,
                       **paired(out[a], out[bb])}
                rep = []
                for s in seeds:
                    if s == ref:
                        continue
                    r = paired(rows.outcome(v, CLASSES, b, s, a), rows.outcome(v, CLASSES, b, s, bb))
                    rep.append({"seed": int(s), "diff": r["diff"], "p": r["p"]})
                rec["replications"] = rep
                primary.append(rec)
                for c in CLASSES:
                    r = paired(rows.outcome(v, [c], b, ref, a), rows.outcome(v, [c], b, ref, bb))
                    per_class.append({"victim": v, "budget": b, "class": c, "seed": int(ref),
                                      "A": a, "B": bb, **r})
    for fam in (primary, per_class):
        adj = holm_adjust([r["p"] for r in fam]) if fam else []
        for r, p in zip(fam, adj):
            r["p_holm"] = p
    adj = holm_adjust([r["p"] for r in omnibus]) if omnibus else []
    for r, p in zip(omnibus, adj):
        r["p_holm"] = p
    return {"reference_seed": ref_seed, "primary": primary, "per_class": per_class,
            "omnibus": omnibus, "overlap": overlap}


def historical(rows: Rows, cells: pd.DataFrame, campaign: Path, ref_seed: int) -> dict:
    """Pair ablation Hybrid with the canonical campaign's pre-fix Hybrid and 1-draw random."""
    out = []
    if not campaign.exists():
        return {"available": False, "rows": out}
    for v in dict.fromkeys(cells.victim):
        seeds = sorted(cells[cells.victim == v].seed.unique())
        for b in BUDGETS:
            for seed in seeds:
                try:
                    new = rows.outcome(v, CLASSES, b, seed, "hybrid")
                    ids = rows.sample_ids(v, CLASSES, b, seed, "hybrid")
                except FileNotFoundError:
                    continue
                for tag, atk in (("prefix_hybrid", f"prim_search_joint_{b}"),
                                 ("rand1", f"prim_rand_joint_{b}")):
                    parts, pid = [], []
                    for c in CLASSES:
                        path = campaign / "artifacts" / f"{v}__{c}__{atk}__seed{seed}.npz"
                        with np.load(path, allow_pickle=True) as d:
                            parts.append(d["targeted_success"].astype(bool) & d["domain_valid"].astype(bool))
                            pid.append(d["sample_id"].astype("U128"))
                    old = np.concatenate(parts)
                    if not np.array_equal(np.concatenate(pid), ids.astype("U128")):
                        raise AssertionError(f"campaign rows differ for {v}/{b}/{seed}")
                    out.append({"victim": v, "budget": b, "seed": int(seed), "B": tag,
                                "primary": seed == (ref_seed if ref_seed in seeds else seeds[0]),
                                **paired(new, old)})
    prim = [r for r in out if r["primary"]]
    for r, p in zip(prim, holm_adjust([r["p"] for r in prim]) if prim else []):
        r["p_holm"] = p
    return {"available": True, "rows": out}


def old_adam(old_dir: Path) -> pd.DataFrame:
    """Descriptive ASR of the replaced (p, alpha) Adam optimizer (different head selection)."""
    path = old_dir / "cells.json"
    if not path.exists():
        return pd.DataFrame()
    c = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    c = c[c.attack.str.match(r"prim_opt_joint_(p50|p75|unb)$")].copy()
    if c.empty:
        return c
    c["budget"] = c.attack.str.extract(r"_(p50|p75|unb)$")[0]
    c["valid"] = c["valid_targeted_benign"]
    per_seed = c.groupby(["victim", "budget", "seed"]).valid.mean().reset_index()
    return per_seed.groupby(["victim", "budget"]).valid.agg(["mean", "std"]).reset_index()


# ------------------------------------------------------------------------------------------
def anytime_curves(rows: Rows, cells: pd.DataFrame, budget_cap: int) -> dict:
    grid = np.arange(1, budget_cap + 1)
    curves = {}
    for (v, b, m), g in cells.groupby(["victim", "budget_label", "method"], sort=False):
        per_seed = []
        for seed, gs in g.groupby("seed"):
            fs = np.concatenate([rows.load(v, c, b, seed, m)["first_success_evaluation"]
                                 for c in gs["class"]])
            fs = np.where(fs > 0, fs, np.inf)
            per_seed.append((fs[None, :] <= grid[:, None]).mean(1))
        curves[(v, b, m)] = np.mean(per_seed, 0)
    return {"grid": grid, "curves": curves}


def make_plots(out: Path, agg, per_class, tests, curves, victims):
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    methods = [m for m in METHODS if m in set(agg.method)]
    paths = {}

    # 1. ASR by optimizer/model/class/budget
    fig, axes = plt.subplots(len(CLASSES), len(victims), figsize=(3.2 * len(victims), 2.6 * len(CLASSES)),
                             sharey="row", squeeze=False)
    w = 0.8 / len(methods)
    for i, c in enumerate(CLASSES):
        for j, v in enumerate(victims):
            ax = axes[i, j]
            for k, m in enumerate(methods):
                sub = per_class[(per_class.victim == v) & (per_class["class"] == c) & (per_class.method == m)]
                vals = [sub[sub.budget_label == b].asr.mean() * 100 for b in BUDGETS]
                sds = [np.nan_to_num(sub[sub.budget_label == b].asr_sd.mean() * 100) for b in BUDGETS]
                ax.bar(np.arange(3) + (k - (len(methods) - 1) / 2) * w, vals, w, yerr=sds,
                       color=COLOR[m], label=LABEL[m], capsize=2)
            ax.set_xticks(range(3), BUDGETS)
            ax.set_title(f"{v} / {c}", fontsize=9)
            if j == 0:
                ax.set_ylabel("valid ASR (%)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    paths["asr_by_optimizer"] = plots / "asr_by_optimizer_model_class_budget.png"
    fig.savefig(paths["asr_by_optimizer"], dpi=130); plt.close(fig)

    # 2. ASR vs classifier-evaluation budget
    grid = curves["grid"]
    fig, axes = plt.subplots(len(victims), 3, figsize=(12, 3 * len(victims)), squeeze=False)
    for i, v in enumerate(victims):
        for j, b in enumerate(BUDGETS):
            ax = axes[i, j]
            for m in methods:
                key = (v, b, m)
                if key in curves["curves"]:
                    ax.plot(grid, curves["curves"][key] * 100, color=COLOR[m], label=LABEL[m])
            ax.set_xscale("log"); ax.set_title(f"{v} / {b}", fontsize=9)
            ax.set_xlabel("victim evaluations per flow"); ax.set_ylabel("valid ASR (%)")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    paths["asr_vs_evals"] = plots / "asr_vs_evaluation_budget.png"
    fig.savefig(paths["asr_vs_evals"], dpi=130); plt.close(fig)

    # 3. cost vs ASR ; 4. evaluations / runtime vs ASR
    markers = {"p50": "o", "p75": "s", "unb": "^"}
    for name, xkey, xlabel in (("cost_vs_asr", "cost_median", "median normalized primitive cost of successes"),
                               ("evals_vs_asr", "evals_mean", "mean victim evaluations per flow"),
                               ("runtime_vs_asr", "ms_per_flow", "runtime per flow (ms)")):
        fig, axes = plt.subplots(1, len(victims), figsize=(4 * len(victims), 3.4), squeeze=False)
        for j, v in enumerate(victims):
            ax = axes[0, j]
            for m in methods:
                for b in BUDGETS:
                    r = agg[(agg.victim == v) & (agg.method == m) & (agg.budget == b)]
                    if len(r):
                        ax.scatter(r[xkey], r.asr * 100, color=COLOR[m], marker=markers[b], s=45,
                                   label=f"{LABEL[m]} {b}")
            ax.set_title(v, fontsize=9); ax.set_xlabel(xlabel); ax.set_ylabel("valid ASR (%)")
            ax.grid(alpha=0.3)
            if name == "runtime_vs_asr":
                ax.set_xscale("log")
        axes[0, 0].legend(fontsize=6, ncol=1)
        fig.tight_layout()
        paths[name] = plots / f"{name}.png"
        fig.savefig(paths[name], dpi=130); plt.close(fig)

    # 5. pairwise success overlap (Venn regions, reference seed)
    ov = pd.DataFrame(tests["overlap"])
    regions = ["all3", "hybrid_pgd_only", "hybrid_cw_only", "pgd_cw_only",
               "hybrid_only", "pgd_only", "cw_only"]
    colors = ["#444444", "#66a61e", "#1f78b4", "#e7298a", COLOR["hybrid"], COLOR["pgd"], COLOR["cw"]]
    fig, ax = plt.subplots(figsize=(max(8, 0.9 * len(ov)), 4))
    bottom = np.zeros(len(ov))
    for reg, col in zip(regions, colors):
        vals = ov[reg].to_numpy() / ov.n.to_numpy() * 100
        ax.bar(range(len(ov)), vals, bottom=bottom, color=col, label=reg)
        bottom += vals
    ax.set_xticks(range(len(ov)), [f"{r.victim}\n{r.budget}" for r in ov.itertuples()], fontsize=7)
    ax.set_ylabel("% of flows with a valid success"); ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    paths["overlap"] = plots / "success_overlap.png"
    fig.savefig(paths["overlap"], dpi=130); plt.close(fig)

    # 6. per-class heatmaps
    fig, axes = plt.subplots(len(victims), 3, figsize=(11, 2.4 * len(victims)), squeeze=False)
    for i, v in enumerate(victims):
        for j, b in enumerate(BUDGETS):
            ax = axes[i, j]
            mat = np.array([[per_class[(per_class.victim == v) & (per_class.budget_label == b)
                                       & (per_class.method == m) & (per_class["class"] == c)].asr.mean() * 100
                             for c in CLASSES] for m in methods])
            ax.imshow(mat, cmap="viridis", vmin=0, vmax=100, aspect="auto")
            for (r, cc), val in np.ndenumerate(mat):
                ax.text(cc, r, f"{val:.1f}", ha="center", va="center",
                        color="white" if val < 60 else "black", fontsize=7)
            ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=7)
            ax.set_yticks(range(len(methods)), [LABEL[m] for m in methods], fontsize=7)
            ax.set_title(f"{v} / {b} (valid ASR %)", fontsize=9)
    fig.tight_layout()
    paths["per_class"] = plots / "per_class_asr_heatmap.png"
    fig.savefig(paths["per_class"], dpi=130); plt.close(fig)
    return paths


# ------------------------------------------------------------------------------------------
def write_tables(out: Path, cells, agg, per_class, tests, hist, adam, curves, paths, victims):
    L = []
    methods = [m for m in METHODS if m in set(cells.method)]

    L.append("### Aggregated results (classes pooled within victim; mean ± SD over attack seeds)\n")
    t = agg.copy()
    tab = pd.DataFrame({
        "Victim": t.victim, "Budget": t.budget, "Method": t.method.map(LABEL),
        "n/seed": t.n_per_seed, "seeds": t.n_seeds,
        "valid ASR": [f"{pct(a)} ± {pct(s) if not np.isnan(s) else '—'}" for a, s in zip(t.asr, t.asr_sd)],
        "raw targeted": t.raw.map(pct), "SP-ASR": t.sp.map(pct),
        "Hybrid R=2 prefix": t.canon.map(pct),
        "cost mean/median": [f"{num(a)}/{num(b)}" for a, b in zip(t.cost_mean, t.cost_median)],
        "evals mean/median": [f"{num(a, 4)}/{num(b, 4)}" for a, b in zip(t.evals_mean, t.evals_median)],
        "median evals→1st success": t.first_median.map(lambda x: num(x, 4)),
        "fail invalid": t.inv.map(pct), "fail exhausted": t.exh.map(pct),
        "no headroom": t.noh.map(pct),
        "runtime s (sum/seed)": t.runtime.map(lambda x: f"{x:.1f}"),
        "ms/flow": t.ms_per_flow.map(lambda x: f"{x:.2f}"),
    })
    L.append(md_table(tab) + "\n")

    L.append("### Per-class results (mean over attack seeds)\n")
    pc = per_class.copy()
    tab = pd.DataFrame({
        "Victim": pc.victim, "Class": pc["class"], "Budget": pc.budget_label,
        "Method": pc.method.map(LABEL), "n": pc.n, "movable": pc.n_movable,
        "valid ASR": pc.asr.map(pct), "SD": pc.asr_sd.map(pct), "raw": pc.raw.map(pct),
        "SP-ASR": pc.sp.map(pct), "median cost": pc.cost_median.map(num),
        "mean evals": pc.evals_mean.map(lambda x: num(x, 4)),
        "unique succ.": pc.unique.map(lambda x: f"{x:.1f}"),
        "runtime s": pc.runtime.map(lambda x: f"{x:.2f}"),
    })
    L.append(md_table(tab) + "\n")

    L.append(f"### Paired tests, primary family (reference seed, classes pooled within victim, "
             f"Holm over {len(tests['primary'])} tests)\n")
    tab = pd.DataFrame([{
        "Victim": r["victim"], "Budget": r["budget"], "A vs B": f"{LABEL[r['A']]} vs {LABEL[r['B']]}",
        "n": r["n"], "A": pct(r["rate_A"]), "B": pct(r["rate_B"]),
        "A only": r["A_only"], "B only": r["B_only"],
        "Δ [95% CI] (pp)": f"{100 * r['diff']:+.2f} [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}]",
        "p": f"{r['p']:.2e}", "Holm p": f"{r['p_holm']:.2e}",
        "replications Δpp (p)": "; ".join(f"s{x['seed']}: {100 * x['diff']:+.2f} ({x['p']:.1e})"
                                          for x in r["replications"]) or "—",
    } for r in tests["primary"]])
    L.append(md_table(tab) + "\n")

    L.append("### Cochran's Q omnibus (3 methods, reference seed, Holm-adjusted)\n")
    tab = pd.DataFrame([{"Victim": r["victim"], "Budget": r["budget"], "Q": f"{r['Q']:.2f}",
                         "df": r["df"], "p": f"{r['p']:.2e}", "Holm p": f"{r['p_holm']:.2e}"}
                        for r in tests["omnibus"]])
    L.append(md_table(tab) + "\n")

    L.append("### Success overlap (reference seed, valid successes, classes pooled)\n")
    tab = pd.DataFrame([{"Victim": r["victim"], "Budget": r["budget"], "n": r["n"],
                         "all 3": r["all3"], "H∧P only": r["hybrid_pgd_only"],
                         "H∧C only": r["hybrid_cw_only"], "P∧C only": r["pgd_cw_only"],
                         "Hybrid only": r["hybrid_only"], "PGD only": r["pgd_only"],
                         "C&W only": r["cw_only"], "none": r["none"],
                         **{k.replace("hybrid", "H").replace("pgd", "P").replace("cw", "C"): v
                            for k, v in r["pairwise"].items()}}
                        for r in tests["overlap"]])
    L.append(md_table(tab) + "\n")

    L.append(f"### Per-class paired tests (reference seed, Holm over {len(tests['per_class'])} tests)\n")
    tab = pd.DataFrame([{
        "Victim": r["victim"], "Budget": r["budget"], "Class": r["class"],
        "A vs B": f"{LABEL[r['A']]} vs {LABEL[r['B']]}", "A": pct(r["rate_A"]), "B": pct(r["rate_B"]),
        "A only": r["A_only"], "B only": r["B_only"],
        "Δ [95% CI] (pp)": f"{100 * r['diff']:+.2f} [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}]",
        "p": f"{r['p']:.2e}", "Holm p": f"{r['p_holm']:.2e}",
    } for r in tests["per_class"]])
    L.append(md_table(tab) + "\n")

    L.append("### ASR at matched evaluation budgets (from per-flow first-success index, seed mean)\n")
    qs = [q for q in (1, 8, 16, 32, 64, 128, 256) if q <= curves["grid"][-1]]
    recs = []
    for (v, b, m), cur in curves["curves"].items():
        recs.append({"Victim": v, "Budget": b, "Method": LABEL[m],
                     **{f"ASR@{q}": pct(cur[q - 1]) for q in qs}})
    L.append(md_table(pd.DataFrame(recs)) + "\n")

    if hist["available"]:
        L.append("### Historical: ablation Hybrid (validity-aware, budget-filled) vs canonical "
                 "campaign rows on the same flows\n")
        tab = pd.DataFrame([{
            "Victim": r["victim"], "Budget": r["budget"], "Seed": r["seed"],
            "Comparator": LABEL[r["B"]], "Hybrid": pct(r["rate_A"]), "Comparator ASR": pct(r["rate_B"]),
            "Hybrid only": r["A_only"], "Comp. only": r["B_only"],
            "Δ [95% CI] (pp)": f"{100 * r['diff']:+.2f} [{100 * r['ci95'][0]:+.2f}, {100 * r['ci95'][1]:+.2f}]",
            "p": f"{r['p']:.2e}", "Holm p (primary seed)": f"{r['p_holm']:.2e}" if "p_holm" in r else "—",
        } for r in hist["rows"]])
        L.append(md_table(tab) + "\n")
    if len(adam):
        L.append("### Historical: replaced (p, α) Adam optimizer, joint mode (head selection — "
                 "NOT the same flows; descriptive only)\n")
        tab = pd.DataFrame({"Victim": adam.victim, "Budget": adam.budget,
                            "valid ASR mean": adam["mean"].map(pct), "SD over seeds": adam["std"].map(pct)})
        L.append(md_table(tab) + "\n")

    L.append("### Complete raw per-cell results\n")
    c = cells.sort_values(["victim", "class", "budget_label", "seed", "method"])
    tab = pd.DataFrame({
        "victim": c.victim, "class": c["class"], "budget": c.budget_label, "seed": c.seed,
        "method": c.method.map(LABEL), "attempted": c.n, "movable": c.n_movable,
        "succ.": c.successes, "valid ASR": c.asr_valid.map(pct), "raw ASR": c.asr_raw.map(pct),
        "SP-ASR": c.sp_asr.map(pct),
        "cost mean/med": [f"{num(a)}/{num(b)}" for a, b in zip(c.cost_mean, c.cost_median)],
        "p mean/med": [f"{num(a)}/{num(b)}" for a, b in zip(c.p_mean, c.p_median)],
        "delay µs mean/med": [f"{num(a)}/{num(b)}" for a, b in zip(c.delay_mean, c.delay_median)],
        "shape mean/med": [f"{num(a)}/{num(b)}" for a, b in zip(c.shape_mean, c.shape_median)],
        "evals mean/med": [f"{num(a, 4)}/{num(b, 4)}" for a, b in zip(c.evals_mean, c.evals_median)],
        "med evals→1st": c.first_success_median.map(lambda x: num(x, 4)),
        "fail inv/exh/noh": [f"{a}/{b}/{d}" for a, b, d in zip(c.fail_invalid, c.fail_exhausted, c.fail_no_headroom)],
        "unique": c.unique_successes, "iters": c.iterations, "restarts": c.restarts,
        "runtime s": c.elapsed_seconds.map(lambda x: f"{x:.2f}"),
    })
    L.append(md_table(tab) + "\n")

    L.append("### Plots\n")
    for k, p in paths.items():
        shown = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
        L.append(f"- `{shown.as_posix()}` ({k})")
    (out / "tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path,
                    default=REPO_ROOT / "outputs/primattack_optimizer_ablation/cicids2017_distrinet")
    ap.add_argument("--campaign", type=Path, default=None,
                    help="canonical campaign dir with the same selection (default: "
                         "outputs/adv_campaign_noidr/<dataset>)")
    ap.add_argument("--old-adam", type=Path, default=REPO_ROOT / "outputs/full_adv_eval")
    ap.add_argument("--reference-seed", type=int, default=42)
    args = ap.parse_args()

    rows = Rows(args.root)
    dataset = rows.config["dataset"]
    global CLASSES
    CLASSES = tuple(rows.config["classes"])
    campaign = args.campaign or (REPO_ROOT / "outputs/adv_campaign_noidr" / dataset)
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    cells = unique_success_columns(rows, rows.cells)
    cells.to_csv(out / "cells_raw.csv", index=False)
    victims = list(dict.fromkeys(cells.victim))
    agg, per_class = aggregate(rows, cells)
    agg.to_csv(out / "aggregate_victim_budget_method.csv", index=False)
    per_class.to_csv(out / "aggregate_per_class.csv", index=False)
    tests = paired_tests(rows, cells, args.reference_seed)
    hist = historical(rows, cells, campaign, args.reference_seed) if rows.config.get("limit_rows") is None \
        else {"available": False, "rows": []}
    adam = old_adam(args.old_adam) if dataset == "cicids2017_distrinet" else pd.DataFrame()
    curves = anytime_curves(rows, cells, int(rows.config["eval_budget_per_flow"]))
    pd.DataFrame([{"victim": v, "budget": b, "method": m, **{f"q{q}": c[q - 1] for q in curves["grid"]}}
                  for (v, b, m), c in curves["curves"].items()]).to_csv(out / "asr_vs_evaluations.csv", index=False)
    (out / "paired_tests.json").write_text(json.dumps(
        {"tests": tests, "historical": hist,
         "old_adam": adam.to_dict(orient="records")}, indent=2, default=float), encoding="utf-8")
    paths = make_plots(out, agg, per_class, tests, curves, victims)
    write_tables(out, cells, agg, per_class, tests, hist, adam, curves, paths, victims)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
