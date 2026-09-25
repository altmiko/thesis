"""Train-only Layer-2 rule miner -> ``old_constraints/<dataset>/mined.json``.

Implements the procedure documented in ``docs/methods/constraint_miner.md``:

1. enumerate candidates from the manifest's semantic feature families --
   every ``<P> Min / <P> Mean / <P> Max`` triple -> ``MonotoneNondecreasing``;
   every ``<P> Std`` with a Min/Max pair -> ``HalfRangeBound``; the CICFlowMeter
   flow-statistic identities (``Variance = Std^2``, ``Avg = Mean``, ``Total = Count*Mean``)
   -> ``ProductEquality``;
2. measure each candidate's violation rate on the full TRAIN split with the constraint's
   own ``validate`` (``product_rtol=0.05``, ``monotone_tol=1e-6``, half-range ``rtol=0.05``);
3. keep a rule iff its train violation rate <= ``keep_if_violation_rate_lte`` (0.01);
4. serialize survivors with :func:`constraints.layer2.dump_layer2` plus a ``mining`` block.

Validation/test rows are never read.

Run:  python -m constraints.mine_layer2 --dataset cicids2018
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from constraints.layer1 import HalfRangeBound, MonotoneNondecreasing, ProductEquality
from constraints.layer2 import dump_layer2
from datasets import get_adapter
from datasets.feature_manifest import FeatureManifest

KEEP_IF_VIOLATION_RATE_LTE = 0.01
PRODUCT_RTOL = 0.05
MONOTONE_TOL = 1e-6
HALF_RANGE_RTOL = 0.05

# CICFlowMeter flow-statistic identities (target, factors, rule name).
_IDENTITY_TEMPLATES = (
    ("Packet Length Variance", ("Packet Length Std", "Packet Length Std"),
     "packet_variance_eq_std_squared"),
    ("Average Packet Size", ("Packet Length Mean",), "avg_packet_size_eq_packet_mean"),
    ("Fwd Segment Size Avg", ("Fwd Packet Length Mean",), "fwd_segment_avg_eq_fwd_mean"),
    ("Bwd Segment Size Avg", ("Bwd Packet Length Mean",), "bwd_segment_avg_eq_bwd_mean"),
    ("Total Length of Fwd Packet", ("Total Fwd Packet", "Fwd Packet Length Mean"),
     "fwd_total_eq_count_mean"),
    ("Total Length of Bwd Packet", ("Total Bwd packets", "Bwd Packet Length Mean"),
     "bwd_total_eq_count_mean"),
)


def _slug(prefix: str) -> str:
    return prefix.lower().replace(" ", "_").replace("/", "_per_")


def candidates(manifest: FeatureManifest) -> list:
    names = set(manifest.names)
    prefixes = sorted({n[: -len(" Mean")] for n in names if n.endswith(" Mean")})
    out: list = []
    for p in prefixes:
        lo, mid, hi = f"{p} Min", f"{p} Mean", f"{p} Max"
        if {lo, mid, hi} <= names:
            out.append(MonotoneNondecreasing(manifest, [lo, mid, hi], tol=MONOTONE_TOL,
                                             name=f"{_slug(p)}_order"))
            if f"{p} Std" in names:
                out.append(HalfRangeBound(manifest, f"{p} Std", lo, hi, rtol=HALF_RANGE_RTOL,
                                          name=f"{_slug(p)}_std_half_range"))
    for target, factors, name in _IDENTITY_TEMPLATES:
        if target in names and set(factors) <= names:
            out.append(ProductEquality(manifest, target, factors, rtol=PRODUCT_RTOL, name=name))
    return out


def violation_rates(rules: list, x_train: np.ndarray, chunk: int = 262_144) -> list[float]:
    fails = np.zeros(len(rules), dtype=np.int64)
    for start in range(0, x_train.shape[0], chunk):
        xb = torch.as_tensor(np.asarray(x_train[start:start + chunk], dtype=np.float64))
        for k, rule in enumerate(rules):
            fails[k] += int((~rule.validate(xb)).sum())
    return (fails / x_train.shape[0]).tolist()


def mine(dataset: str) -> dict:
    adapter = get_adapter(dataset)
    manifest = adapter.feature_manifest()
    source = adapter._processed / "X_train_pristine.npy"
    x_train = np.load(source, mmap_mode="r")
    rules = candidates(manifest)
    rates = violation_rates(rules, x_train)
    kept = [r for r, v in zip(rules, rates) if v <= KEEP_IF_VIOLATION_RATE_LTE]
    rejected = [{"type": type(r).__name__, "name": r.name, "train_violation_rate": v}
                for r, v in zip(rules, rates) if v > KEEP_IF_VIOLATION_RATE_LTE]
    payload = dump_layer2(kept, manifest.dataset_name)
    payload["mining"] = {
        "fit_split": "train",
        "fit_rows": int(x_train.shape[0]),
        "source": str(source.name),
        "miner": "constraints.mine_layer2",
        "keep_if_violation_rate_lte": KEEP_IF_VIOLATION_RATE_LTE,
        "product_rtol": PRODUCT_RTOL,
        "monotone_tol": MONOTONE_TOL,
        "half_range_rtol": HALF_RANGE_RTOL,
        "candidates_tested": len(rules),
        "retained_rules": len(kept),
        "retained_violation_rates": {r.name: v for r, v in zip(rules, rates)
                                     if v <= KEEP_IF_VIOLATION_RATE_LTE},
        "rejected": rejected,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="datasets.get_adapter name")
    ap.add_argument("--output", type=Path, default=None,
                    help="default: old_constraints/<dataset_name>/mined.json")
    args = ap.parse_args()
    payload = mine(args.dataset)
    out = args.output or (Path(__file__).resolve().parents[2] / "old_constraints"
                          / payload["dataset"] / "mined.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "retained": payload["mining"]["retained_rules"],
                      "rejected": [r["name"] for r in payload["mining"]["rejected"]]}))


if __name__ == "__main__":
    main()
