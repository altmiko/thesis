"""Exp A PrimAttack on the FULL clean-correct test pool + resampling of the attacked rows.

The FINAL Exp A attacks 800 clean-correct test flows per (victim, class), drawn by one seeded
permutation (selection seed 42). This script removes that row choice: it runs the identical
Exp A PrimAttack configuration (selected optimizer, untargeted, joint, p75, validator_v2 gate,
locked ``PRIM_ARGS``, attack seed 42) on EVERY clean-correct test flow of every attack class,
through the unchanged canonical runner ``scripts/run_primattack_optimizer_ablation.py`` (called
in-process, in row chunks to bound GPU memory).

Analysis (per victim):
* full-pool Raw / Valid ASR (class-balanced = mean of per-class rates, i.e. the 800-per-class
  design; and prevalence-weighted = all successes / all flows);
* row-resampling distribution: B random draws of 800 flows per class (without replacement,
  canonical rule) from the full pool, using the per-flow outcomes -- equivalent to re-running
  on a new row selection because a flow's outcome does not depend on the other rows or on the
  attack seed (verified by the multi-seed runs); reports mean, SD, 2.5/97.5 percentiles and the
  percentile of the canonical FINAL sample;
* consistency: full-pool outcomes on the canonical 800 rows vs the FINAL seed-42 artifacts.

Outputs (non-canonical): ``outputs/expA_primattack_full_pool/<dataset>/<victim>/chunk_<k>/`` and
``outputs/expA_primattack_full_pool/row_resampling.{json,md}``.

    python scripts/run_expA_primattack_full_pool.py --device cuda
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

import run_final_suite as final  # noqa: E402

OUT_ROOT = REPO_ROOT / "outputs" / "expA_primattack_full_pool"
STAGE = "primattack_untargeted"
BUDGET = "maximum-evaluated"
ATTACK_SEED = 42
N_PER_CLASS = 800
CHUNK_ROWS = 4000


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_primattack_optimizer_ablation",
        REPO_ROOT / "scripts" / "run_primattack_optimizer_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _classes() -> list[str]:
    return [c for c in final.CLASSES.split(",") if c]


def build_pools(runner, dataset: str, device: str) -> dict:
    """victim -> class -> clean-correct test positional indices (test order). Uses the runner's
    exact clean-correctness computation (victim on ``(raw[idx] - center) / scale`` per chunk)."""
    spec = final.DATASETS[dataset]
    adapter = runner.get_adapter(spec["cli"])
    transform = adapter.feature_transform()
    mapping = adapter.class_mapping()
    center = torch.tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.tensor(transform.scale, dtype=torch.float32, device=device)
    processed = adapter._processed
    raw_all_t = torch.tensor(np.ascontiguousarray(
        np.load(processed / "X_test_pristine.npy", mmap_mode="r")), dtype=torch.float32,
        device=device)
    y = np.load(processed / "y_test_cat.npy").astype(np.int64)
    pools = {}
    for vname in spec["victims"].split(","):
        ckpt, arch, _ = runner.victim_checkpoint(adapter.name, vname)
        victim = runner.load_category_victim(ckpt, adapter=adapter, expected_model_type=arch,
                                             device=device)
        pools[vname] = {}
        def predict(idx: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return victim((raw_all_t[torch.as_tensor(idx, device=device)] - center)
                              / scale).argmax(1).cpu().numpy()

        for cname in _classes():
            cid = int(mapping.name_to_id[cname])
            cand = np.flatnonzero(y == cid)
            keep = np.concatenate([cand[s:s + CHUNK_ROWS][predict(cand[s:s + CHUNK_ROWS]) == cid]
                                   for s in range(0, len(cand), CHUNK_ROWS)])
            # The runner re-checks clean-correctness on exactly each final chunk; batch
            # composition can flip a borderline row, so iterate until every chunk passes as-is.
            while True:
                chunks = [keep[s:s + CHUNK_ROWS] for s in range(0, len(keep), CHUNK_ROWS)]
                ok = [c[predict(c) == cid] for c in chunks]
                if all(len(a) == len(b) for a, b in zip(ok, chunks)):
                    break
                keep = np.concatenate(ok)
            pools[vname][cname] = keep
        del victim
        torch.cuda.empty_cache()
    del raw_all_t
    torch.cuda.empty_cache()
    return pools


def run_dataset(runner, dataset: str, device: str, method: str) -> dict:
    spec = final.DATASETS[dataset]
    adapter = runner.get_adapter(spec["cli"])
    sample_ids_all = runner.pd.read_parquet(
        adapter._processed / "test.parquet", columns=["sample_id"]
    )["sample_id"].astype(str).to_numpy(dtype="U128")
    pools = build_pools(runner, dataset, device)
    for vname, by_class in pools.items():
        n_chunks = max(-(-len(ix) // CHUNK_ROWS) for ix in by_class.values())
        for k in range(n_chunks):
            out = OUT_ROOT / dataset / vname / f"chunk_{k:02d}"
            sel, classes = {vname: {}}, []
            for cname, ix in by_class.items():
                idx = ix[k * CHUNK_ROWS:(k + 1) * CHUNK_ROWS]
                if not len(idx):
                    continue
                sids = sample_ids_all[idx]
                sel[vname][cname] = {"positional_idx": idx.tolist(), "sample_ids": sids.tolist(),
                                     "sha256_sample_ids": runner._sha_ids(sids),
                                     "n_used": int(len(idx))}
                classes.append(cname)
            out.mkdir(parents=True, exist_ok=True)
            sel_path = out / "pool_selection.json"
            sel_path.write_text(json.dumps(sel), encoding="utf-8")
            sys.argv = [
                "run_primattack_optimizer_ablation.py", "--dataset", spec["cli"],
                "--device", device, "--selection-from", str(sel_path), "--split", "test",
                "--seeds", str(ATTACK_SEED), "--objective", "untargeted", "--victims", vname,
                "--classes", ",".join(classes), "--budgets", BUDGET, "--methods", method,
                "--modes", "joint", *final.PRIM_ARGS, "--output-dir", str(out), "--resume",
            ]
            print(f"[pool] {dataset}/{vname} chunk {k + 1}/{n_chunks} classes {classes}",
                  flush=True)
            runner.main()
            torch.cuda.empty_cache()
    return {v: {c: int(len(ix)) for c, ix in bc.items()} for v, bc in pools.items()}


def _load_outcomes(dataset: str, vname: str, method: str) -> dict[str, dict[str, np.ndarray]]:
    parts: dict[str, dict[str, list]] = {}
    for chunk in sorted((OUT_ROOT / dataset / vname).glob("chunk_*")):
        for cname in _classes():
            f = chunk / "artifacts" / f"{vname}__{cname}__p75__{method}__seed{ATTACK_SEED}.npz"
            if not f.exists():
                continue
            with np.load(f, allow_pickle=True) as d:
                p = parts.setdefault(cname, {"sid": [], "raw": [], "valid": []})
                p["sid"].append(d["sample_id"].astype("U128"))
                p["raw"].append(d["raw_success"].astype(bool))
                p["valid"].append(d["valid_success"].astype(bool))
    return {c: {k: np.concatenate(v) for k, v in p.items()} for c, p in parts.items()}


def analyze(datasets: list[str], method: str, n_boot: int) -> dict:
    rng = np.random.default_rng(20261001)
    result = {}
    for dataset in datasets:
        canon_dir = final.RUNS / dataset / STAGE / "artifacts"
        result[dataset] = {}
        for vname in final.DATASETS[dataset]["victims"].split(","):
            outc = _load_outcomes(dataset, vname, method)
            if set(outc) != set(_classes()):
                raise KeyError(f"{dataset}/{vname}: missing classes {set(_classes()) - set(outc)}")
            per_class, mismatch, canon_valid, canon_raw = {}, 0, 0, 0
            for cname, o in outc.items():
                if len(set(o["sid"])) != len(o["sid"]):
                    raise AssertionError(f"duplicate rows in pool {dataset}/{vname}/{cname}")
                per_class[cname] = {"n_pool": int(len(o["sid"])),
                                    "raw_asr": float(o["raw"].mean()),
                                    "valid_asr": float(o["valid"].mean())}
                name = f"{vname}__{cname}__p75__{method}__seed{ATTACK_SEED}.npz"
                with np.load(canon_dir / name, allow_pickle=True) as d:
                    csid = d["sample_id"].astype("U128")
                    cval = d["valid_success"].astype(bool)
                    craw = d["raw_success"].astype(bool)
                pos = {s: i for i, s in enumerate(o["sid"])}
                j = np.asarray([pos[s] for s in csid])
                mismatch += int((o["valid"][j] != cval).sum() + (o["raw"][j] != craw).sum())
                canon_valid += int(cval.sum()); canon_raw += int(craw.sum())
            n_canon = N_PER_CLASS * len(outc)
            # Row resampling: 800 per class without replacement, pooled over classes.
            boot = {"raw": np.empty(n_boot), "valid": np.empty(n_boot)}
            for b in range(n_boot):
                tot_r = tot_v = 0
                for o in outc.values():
                    j = rng.choice(len(o["sid"]), size=min(N_PER_CLASS, len(o["sid"])),
                                   replace=False)
                    tot_r += int(o["raw"][j].sum()); tot_v += int(o["valid"][j].sum())
                boot["raw"][b] = tot_r / n_canon; boot["valid"][b] = tot_v / n_canon
            n_all = sum(len(o["sid"]) for o in outc.values())
            entry = {"per_class": per_class, "n_pool_total": int(n_all),
                     "canonical_vs_pool_outcome_mismatches": mismatch}
            for m, canon_k in (("raw", canon_raw), ("valid", canon_valid)):
                dist, canon = boot[m], canon_k / n_canon
                entry[m] = {
                    "pool_class_balanced": float(np.mean([c[f"{m}_asr"] for c in per_class.values()])),
                    "pool_prevalence_weighted": float(sum(o[m].sum() for o in outc.values()) / n_all),
                    "canonical_sample": canon,
                    "resample_mean": float(dist.mean()), "resample_variance": float(dist.var(ddof=1)),
                    "resample_sd": float(dist.std(ddof=1)),
                    "resample_p2_5": float(np.percentile(dist, 2.5)),
                    "resample_p97_5": float(np.percentile(dist, 97.5)),
                    "canonical_percentile": float(100 * np.mean(dist <= canon)),
                }
            result[dataset][vname] = entry
    return result


def write_report(res: dict, method: str, n_boot: int) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "row_resampling.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pct = lambda v: f"{100 * v:.2f}%"  # noqa: E731
    L = ["# Exp A PrimAttack: full clean-correct test pool and row resampling", "",
         f"Exp A PrimAttack configuration (optimizer `{method}`, untargeted, joint, p75, "
         f"validator_v2 gate, locked PRIM_ARGS, attack seed {ATTACK_SEED}) run on EVERY "
         "clean-correct test flow of DoS/DDoS/Recon/BruteForce. Resampling: "
         f"B = {n_boot} random draws of {N_PER_CLASS} flows per class (without replacement) from "
         "the pool's per-flow outcomes; ASR pooled over the 4 classes per draw. Canonical = the "
         "FINAL 800-per-class sample (selection seed 42).", ""]
    for m in ("valid", "raw"):
        L += [f"## {m.capitalize()} ASR", "",
              "| Dataset | Victim | Pool flows | Full pool (class-balanced) | Full pool "
              "(prevalence-weighted) | Canonical sample | Resample mean | Resample variance | "
              "Resample SD (pp) | Resample 95% range | Canonical percentile |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for dataset, ds in res.items():
            for vname, e in ds.items():
                r = e[m]
                L.append(f"| {dataset} | {vname} | {e['n_pool_total']} | "
                         f"{pct(r['pool_class_balanced'])} | {pct(r['pool_prevalence_weighted'])} | "
                         f"{pct(r['canonical_sample'])} | {pct(r['resample_mean'])} | "
                         f"{r['resample_variance']:.3e} | {100 * r['resample_sd']:.3f} | "
                         f"{pct(r['resample_p2_5'])}–{pct(r['resample_p97_5'])} | "
                         f"{r['canonical_percentile']:.1f} |")
        L.append("")
    L += ["## Per-class full-pool Valid ASR", "",
          "| Dataset | Victim | " + " | ".join(_classes()) + " |",
          "|---|---|" + "---|" * len(_classes())]
    for dataset, ds in res.items():
        for vname, e in ds.items():
            L.append(f"| {dataset} | {vname} | " + " | ".join(
                f"{pct(e['per_class'][c]['valid_asr'])} (n={e['per_class'][c]['n_pool']})"
                for c in _classes()) + " |")
    L += ["", "Consistency (full-pool vs FINAL seed-42 outcomes on the canonical rows, raw+valid "
          "mismatches): " + ", ".join(f"{d}/{v}: {e['canonical_vs_pool_outcome_mismatches']}"
                                      for d, ds in res.items() for v, e in ds.items()), "",
          "Non-canonical; feature-space proxy only (CLAUDE.md claim boundary). Covers the choice "
          "of attacked test rows; not victim-training variability or new campaigns.", ""]
    (OUT_ROOT / "row_resampling.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--datasets", default=",".join(final.DATASETS))
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    method = json.loads((final.RUNS / "optimizer_selection.json").read_text("utf-8"))["selected"]
    if not args.analyze_only:
        runner = _load_runner()
        for dataset in datasets:
            run_dataset(runner, dataset, args.device, method)
    write_report(analyze(datasets, method, args.n_boot), method, args.n_boot)


if __name__ == "__main__":
    main()
