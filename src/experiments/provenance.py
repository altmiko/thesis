"""Immutable provenance helpers for CICIDS2017 experiment runners."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import sklearn
import torch


PROVENANCE_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(repo_root: str | Path, roots: Iterable[str] = ("src", "scripts")) -> str:
    """Hash source paths and bytes, independent of Git tracking state."""
    repo = Path(repo_root)
    digest = hashlib.sha256()
    files: list[Path] = []
    for root in roots:
        base = repo / root
        if base.exists():
            files.extend(
                path for path in base.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    for path in sorted(files, key=lambda value: value.relative_to(repo).as_posix()):
        relative = path.relative_to(repo).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def git_state(repo_root: str | Path) -> dict[str, object]:
    repo = Path(repo_root)
    try:
        commit = _git(repo, "rev-parse", "HEAD")
        status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Git identity is required for an experiment run") from exc
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def deterministic_runtime(seed: int) -> dict[str, object]:
    """Seed every RNG used by attacks and request deterministic torch kernels."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": int(seed),
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def load_row_ids(processed_dir: str | Path, split: str = "test") -> np.ndarray:
    path = Path(processed_dir) / f"{split}.parquet"
    values = pd.read_parquet(path, columns=["sample_id"])["sample_id"].astype(str)
    return np.asarray(values.tolist(), dtype="U128")


def build_provenance(
    *,
    repo_root: str | Path,
    dataset: str,
    method_id: str,
    config: Mapping[str, object],
    preprocessing_manifest: str | Path,
    scaler: str | Path,
    checkpoints: Mapping[str, str | Path],
) -> dict[str, object]:
    repo = Path(repo_root)
    checkpoint_hashes = {
        name: {"path": str(Path(path)), "sha256": sha256_file(path)}
        for name, path in sorted(checkpoints.items())
    }
    state = git_state(repo)
    payload: dict[str, object] = {
        "provenance_version": PROVENANCE_VERSION,
        "dataset": dataset,
        "method_id": method_id,
        "git": state,
        "source_tree_sha256": source_tree_sha256(repo),
        "preprocessing_manifest": {
            "path": str(Path(preprocessing_manifest)),
            "sha256": sha256_file(preprocessing_manifest),
        },
        "scaler": {"path": str(Path(scaler)), "sha256": sha256_file(scaler)},
        "checkpoints": checkpoint_hashes,
        "config": dict(config),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": int(torch.cuda.device_count()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_id"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return payload


def ensure_fresh_output_dir(output_dir: str | Path) -> Path:
    """Refuse silent result replacement; callers must archive/remove old runs explicitly."""
    path = Path(output_dir)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty experiment directory {path}; "
            "archive it or choose a new --output-dir"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_provenance_arrays(
    provenance: Mapping[str, object],
    *,
    row_ids: np.ndarray,
    class_name: str,
    victim: str,
    method_id: str,
    seed: int,
    checkpoint_ids: Mapping[str, str],
) -> dict[str, np.ndarray]:
    """NPZ-safe metadata. Static values are scalar Unicode arrays; row IDs are per sample."""
    return {
        "row_id": np.asarray(row_ids, dtype="U128"),
        "dataset": np.asarray(str(provenance["dataset"])),
        "class_name": np.asarray(class_name),
        "victim": np.asarray(victim),
        "method": np.asarray(method_id),
        "seed": np.asarray(int(seed), dtype=np.int64),
        "git_commit": np.asarray(str(provenance["git"]["commit"])),
        "git_dirty": np.asarray(bool(provenance["git"]["dirty"])),
        "source_tree_sha256": np.asarray(str(provenance["source_tree_sha256"])),
        "run_id": np.asarray(str(provenance["run_id"])),
        "config_json": np.asarray(json.dumps(provenance["config"], sort_keys=True)),
        "checkpoint_identifiers_json": np.asarray(json.dumps(dict(checkpoint_ids), sort_keys=True)),
    }
