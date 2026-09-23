"""Synthetic corruption tests (§14).

Starts from genuine clean flows that validator_v2 accepts, injects ONE controlled
violation at a time (including subtle ones), and measures the known-invalid
detection rate (KIDR) overall and per corruption, plus which rules fire.

Run:  python -m validation.evaluation.synthetic_violations
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from validation import load_validator
from validation.mining import data_access as da
from validation.metrics import known_invalid_detection_rate

PKG = Path(__file__).resolve().parents[1]
N = 20000


def _corruptions(idx):
    def col(X, f):
        return idx[f]

    C = []

    def add(name, desc, layer, fn):
        C.append({"name": name, "desc": desc, "layer": layer, "fn": fn})

    def min_gt_max(X, i):
        c = X[:, idx["Fwd Packet Length Min"]]
        app = X[:, idx["Fwd Packet Length Max"]] > 0
        X[app, idx["Fwd Packet Length Min"]] = X[app, idx["Fwd Packet Length Max"]] + 10.0
        return app
    add("min_gt_max", "Fwd Packet Length Min set above Max", "MINED (order chain)", min_gt_max)

    def mean_outside_range(X, i):
        app = np.ones(X.shape[0], bool)
        X[:, idx["Packet Length Mean"]] = X[:, idx["Packet Length Max"]] + 100.0
        return app
    add("mean_above_max", "Packet Length Mean pushed above Max", "MINED (order chain)", mean_outside_range)

    def neg_count(X, i):
        app = np.ones(X.shape[0], bool)
        X[:, idx["Total Fwd Packet"]] = -1.0
        return app
    add("negative_count", "Total Fwd Packet set negative", "PROTOCOL (nonnegativity)", neg_count)

    def variance_broken(X, i):
        app = X[:, idx["Packet Length Std"]] > 1.0
        X[app, idx["Packet Length Variance"]] = (X[app, idx["Packet Length Std"]] ** 2) * 3.0
        return app
    add("variance_neq_std_sq", "Packet Length Variance != Std^2 (3x)", "EXTRACTOR (square)", variance_broken)

    def variance_subtle(X, i):
        app = X[:, idx["Packet Length Std"]] > 1.0
        X[app, idx["Packet Length Variance"]] = (X[app, idx["Packet Length Std"]] ** 2) * 1.02
        return app
    add("variance_subtle_2pct", "Variance off by a subtle 2%", "EXTRACTOR (square)", variance_subtle)

    def flow_pkts_inconsistent(X, i):
        app = (X[:, idx["Fwd Packets/s"]] + X[:, idx["Bwd Packets/s"]]) > 0
        X[app, idx["Flow Packets/s"]] = (X[app, idx["Fwd Packets/s"]] + X[app, idx["Bwd Packets/s"]]) * 2 + 5
        return app
    add("flow_pkts_inconsistent", "Flow Packets/s inconsistent with Fwd+Bwd", "EXTRACTOR (sum)", flow_pkts_inconsistent)

    def avg_size_broken(X, i):
        app = np.ones(X.shape[0], bool)
        X[:, idx["Average Packet Size"]] = X[:, idx["Packet Length Mean"]] + 50.0
        return app
    add("avg_size_neq_mean", "Average Packet Size != Packet Length Mean", "EXTRACTOR (equality)", avg_size_broken)

    def totlen_broken(X, i):
        app = X[:, idx["Total Fwd Packet"]] > 0
        X[app, idx["Total Length of Fwd Packet"]] = (
            X[app, idx["Total Fwd Packet"]] * X[app, idx["Fwd Packet Length Mean"]] * 1.5 + 20.0)
        return app
    add("totlen_neq_count_mean", "Total Length Fwd != count*mean (1.5x)", "EXTRACTOR (product)", totlen_broken)

    def integer_fractional(X, i):
        app = np.ones(X.shape[0], bool)
        X[:, idx["Total Fwd Packet"]] = X[:, idx["Total Fwd Packet"]] + 0.5
        return app
    add("count_fractional", "Integer packet count made fractional (+0.5)", "SCHEMA (integer)", integer_fractional)

    def mean_below_min_subtle(X, i):
        app = X[:, idx["Packet Length Min"]] > 0.01
        X[app, idx["Packet Length Mean"]] = X[app, idx["Packet Length Min"]] - 0.01
        return app
    add("mean_below_min_subtle", "Mean set 0.01 below Min (subtle)", "MINED (order chain)", mean_below_min_subtle)

    return C


def run(dataset: str = "cicids2017_distrinet", n: int = N) -> dict:
    v = load_validator("cicids2017_distrinet")
    X = da.load_split("test", allow_test_for_final_reporting=True)
    Xs = da.sample_rows(X, n, seed=11)
    # keep only clean rows that v2 accepts, so the only violation is the injected one
    base_valid = v.validate_batch(Xs).structurally_valid
    clean = Xs[base_valid]
    results = []
    for c in _corruptions(v.idx):
        Xc = clean.copy()
        app = c["fn"](Xc, v.idx)
        Xc = Xc[app]
        b = v.validate_batch(Xc)
        detected = ~b.structurally_valid
        det_rate = float(detected.mean()) if Xc.shape[0] else 0.0
        # which rules fired on detected rows
        fired = Counter()
        rej = b.rules_rejecting_any()
        by_id = {r.id: r for r in v.rules}
        for rid, cnt in rej.items():
            fired[f"{by_id[rid].source_type}:{by_id[rid].expression()}"] += cnt
        results.append({"name": c["name"], "desc": c["desc"], "layer": c["layer"],
                        "applied": int(Xc.shape[0]), "detection_rate": det_rate,
                        "top_rules": fired.most_common(3)})
    overall = float(np.mean([r["detection_rate"] for r in results])) if results else 0.0
    md = _report(dataset, clean.shape[0], results, overall)
    out = PKG / "reports" / dataset / "synthetic_violation_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return {"overall_kidr": overall, "clean_base": int(clean.shape[0]),
            "per_corruption": {r["name"]: r["detection_rate"] for r in results}}


def _report(dataset, n_base, results, overall) -> str:
    L = [f"# Synthetic violation report — {dataset}", "",
         f"Clean accepted base flows: **{n_base:,}** (validator_v2-accepted `test` flows). "
         f"Each corruption injects ONE controlled violation.", "",
         f"**Overall known-invalid detection rate (mean over corruptions): {overall:.4f}**", "",
         "| corruption | target layer | applied | detection rate | top firing rule |",
         "|---|---|---|---|---|"]
    for r in results:
        top = r["top_rules"][0][0] if r["top_rules"] else "-"
        L.append(f"| {r['name']} ({r['desc']}) | {r['layer']} | {r['applied']:,} | "
                 f"{r['detection_rate']:.4f} | `{top}` |")
    L.append("")
    L.append("Detection = fraction of corrupted rows for which `structurally_valid` becomes FALSE.")
    return "\n".join(L)


if __name__ == "__main__":
    print(run())
