"""Non-canonical PrimAttack source-row sensitivity: one random cohort per attack seed.

Only the locked untargeted Prim-PGD, joint, p75 and envelope-only unbounded arms run.
The FINAL suite and its single frozen cohort remain untouched. All artifacts live below
ablations/seeded_rows/results. Seed 42's existing cells are copied after verifying that
this selector reproduces the FINAL cohort; seeds 2024/2026 execute on new rows.

    python ablations/seeded_rows/run.py --device cuda
    python ablations/seeded_rows/analyze.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import get_adapter  # noqa: E402
from run_final_suite import CLASSES, DATASETS, N_PER_CLASS, PRIM_ARGS, RUNS, SEEDS  # noqa: E402
from run_full_adversarial_eval import _sha_ids, file_sha256, victim_checkpoint  # noqa: E402
from src.classifiers.cicids2017d_victims import load_category_victim  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BUDGETS = ("maximum-evaluated", "unbounded")
METHOD = "pgd"
STAGE = "primattack_untargeted"


def _sha_raw(raw: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float32).tobytes()).hexdigest()


def prepare(dataset: str, seeds: list[int], device: str, results: Path) -> None:
    """Replicate the canonical class-permutation selector, varying only its seed."""
    spec = DATASETS[dataset]
    adapter = get_adapter(spec["cli"])
    processed = adapter._processed
    raw_disk = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
    labels = np.load(processed / "y_test_cat.npy").astype(np.int64)
    sids = pd.read_parquet(processed / "test.parquet", columns=["sample_id"])[
        "sample_id"].astype(str).to_numpy(dtype="U128")
    if len(labels) != len(sids) or len(labels) != len(raw_disk):
        raise AssertionError(f"{dataset}: test features, labels and IDs misaligned")
    transform = adapter.feature_transform()
    center = torch.as_tensor(transform.center, dtype=torch.float32, device=device)
    scale = torch.as_tensor(transform.scale, dtype=torch.float32, device=device)
    raw_tensor = torch.tensor(np.ascontiguousarray(raw_disk), dtype=torch.float32, device=device)
    scaled = (raw_tensor - center) / scale
    class_names = CLASSES.split(",")
    mapping = adapter.class_mapping()
    cohorts: dict[int, dict] = {seed: {} for seed in seeds}
    for victim_name in spec["victims"].split(","):
        ckpt, arch, _ = victim_checkpoint(dataset, victim_name)
        victim = load_category_victim(ckpt, adapter=adapter, expected_model_type=arch,
                                      device=device)
        with torch.no_grad():
            predicted = np.concatenate([
                victim(scaled[start:start + 16384]).argmax(1).cpu().numpy()
                for start in range(0, len(labels), 16384)
            ])
        del victim
        for seed in seeds:
            cohorts[seed][victim_name] = {}
        for name in class_names:
            cid = int(mapping.name_to_id[name])
            class_rows = np.flatnonzero(labels == cid)
            correct = (predicted == cid) & (labels == cid)
            eligible = int(correct[class_rows].sum())
            if eligible < N_PER_CLASS:
                raise ValueError(f"{dataset}/{victim_name}/{name}: only {eligible} eligible")
            for seed in seeds:
                # Same permutation/filter/sort rule as run_full_adversarial_eval.py:476-479.
                perm = np.random.default_rng(seed + cid).permutation(class_rows)
                idx = np.sort(perm[correct[perm]][:N_PER_CLASS])
                ids = sids[idx]
                entry = {
                    "class_id": cid, "n_class_test": int(len(class_rows)),
                    "n_eligible_total": eligible, "n_used": N_PER_CLASS,
                    "sha256_sample_ids": _sha_ids(ids),
                    "clean_raw_sha256": _sha_raw(raw_tensor[idx].cpu().numpy()),
                    "positional_idx": idx.tolist(), "sample_ids": ids.tolist(),
                }
                cohorts[seed][victim_name][name] = entry
                if seed == 42:
                    original = json.loads((RUNS / dataset / "baselines_untargeted" /
                                           "selection.json").read_text(encoding="utf-8"))[
                                               victim_name][name]
                    for field in ("class_id", "n_used", "n_eligible_total", "positional_idx",
                                  "sample_ids", "sha256_sample_ids", "clean_raw_sha256"):
                        if entry[field] != original[field]:
                            raise AssertionError(f"{dataset}/{victim_name}/{name}: seed-42 "
                                                 f"selection differs from FINAL ({field})")
    for seed, selection in cohorts.items():
        out = results / dataset / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        selection_path = out / "selection.json"
        serialized = json.dumps(selection, indent=2)
        if selection_path.exists() and json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise AssertionError(f"{selection_path}: selection changed on resume")
        selection_path.write_text(serialized, encoding="utf-8")
    del raw_tensor, scaled
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def reuse_reference(dataset: str, out: Path, selection: Path) -> None:
    """Copy the verified identical seed-42 cells, never hardlink canonical NPZ files."""
    source = RUNS / dataset / STAGE
    source_selection = RUNS / dataset / "baselines_untargeted" / "selection.json"
    generated = json.loads(selection.read_text(encoding="utf-8"))
    canonical = json.loads(source_selection.read_text(encoding="utf-8"))
    fields = ("class_id", "n_used", "n_eligible_total", "positional_idx",
              "sample_ids", "sha256_sample_ids", "clean_raw_sha256")
    for victim in DATASETS[dataset]["victims"].split(","):
        for name in CLASSES.split(","):
            for field in fields:
                if generated[victim][name][field] != canonical[victim][name][field]:
                    raise AssertionError(f"seed-42 selector differs from FINAL: "
                                         f"{dataset}/{victim}/{name}/{field}")
    cells = json.loads((source / "cells.json").read_text(encoding="utf-8"))
    subset = [cell for cell in cells if cell["seed"] == 42 and cell["method"] == METHOD
              and cell["budget"] in BUDGETS and cell["objective"] == "untargeted"
              and cell["mode"] == "joint"]
    if len(subset) != 2 * len(DATASETS[dataset]["victims"].split(",")) * len(CLASSES.split(",")):
        raise AssertionError(f"{dataset}: incomplete seed-42 FINAL reference cells")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    if (config["eval_budget_per_flow"] != 256
            or "AND validator_v2 hybrid_valid" not in config["success"]
            or config["method_configs"][METHOD]["restarts"] != 3
            or config["method_configs"][METHOD]["step_size"] != 0.05):
        raise AssertionError(f"{dataset}: FINAL PrimAttack config differs from locked settings")
    artifacts = out / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for cell in subset:
        filename = (f"{cell['victim']}__{cell['class']}__{cell['budget_label']}__"
                    f"{METHOD}__seed42.npz")
        src, dst = source / "artifacts" / filename, artifacts / filename
        if not src.exists():
            raise FileNotFoundError(src)
        if dst.exists():
            if file_sha256(dst) != file_sha256(src):
                raise AssertionError(f"{dst}: existing copy differs from FINAL")
        else:
            shutil.copy2(src, dst)
    cells_path = out / "cells.json"
    if cells_path.exists() and json.loads(cells_path.read_text(encoding="utf-8")) != subset:
        raise AssertionError(f"{cells_path}: resumed seed-42 cells differ from FINAL")
    cells_path.write_text(json.dumps(subset, indent=2), encoding="utf-8")
    shutil.copy2(source / "config.json", out / "source_config.json")
    (out / "provenance.json").write_text(json.dumps({
        "source": str(source), "selection": str(source_selection),
        "source_config_sha256": file_sha256(source / "config.json"),
        "source_cells_sha256": file_sha256(source / "cells.json"),
        "copied_cells": len(subset), "copied_artifacts": len(subset),
    }, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default=SEEDS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--run-only", action="store_true",
                        help="use previously verified per-seed selection files")
    args = parser.parse_args()
    if args.prepare_only and args.run_only:
        parser.error("--prepare-only and --run-only are mutually exclusive")
    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(set(seeds)) != len(seeds) or not seeds or set(seeds) - {42, 2024, 2026}:
        raise ValueError("seeds must be distinct and drawn from 42,2024,2026")
    if set(datasets) - set(DATASETS):
        raise ValueError(f"unknown dataset(s) {set(datasets) - set(DATASETS)}")
    root = args.results_dir.resolve()
    for dataset in datasets:
        if not args.run_only:
            prepare(dataset, seeds, args.device, root)
        if args.prepare_only:
            continue
        spec = DATASETS[dataset]
        for seed in seeds:
            selection = root / dataset / f"seed{seed}" / "selection.json"
            if not selection.is_file():
                raise FileNotFoundError(f"{selection}: prepare cohorts first")
            out = selection.parent / "primattack"
            if seed == 42:
                reuse_reference(dataset, out, selection)
                print(f"[reuse] {dataset}/seed42: copied verified canonical p75/unbounded", flush=True)
                continue
            command = [sys.executable, str(ROOT / "scripts" /
                                         "run_primattack_optimizer_ablation.py"),
                       "--dataset", spec["cli"], "--device", args.device,
                       "--selection-from", str(selection), "--split", "test",
                       "--seeds", str(seed), "--objective", "untargeted",
                       "--victims", spec["victims"], "--classes", CLASSES,
                       "--budgets", ",".join(BUDGETS), "--methods", METHOD,
                       "--modes", "joint", *PRIM_ARGS, "--output-dir", str(out), "--resume"]
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            log = selection.parent / "run.log"
            print(f"[run] {dataset}/seed{seed}: Prim-PGD p75 and unbounded -> {out}", flush=True)
            with log.open("a", encoding="utf-8") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                               stderr=subprocess.STDOUT, check=True)
            print(f"[done] {dataset}/seed{seed}", flush=True)


if __name__ == "__main__":
    main()
