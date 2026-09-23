"""Golden regression for the canonical CICIDS2017 DoS/LSTM latent attack.

The fixture is intentionally immutable.  If this test changes, report the affected row IDs and
fields; never regenerate ``tests/data/golden_cicids2017_dos_lstm_seed42.npz`` silently.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attack.realizability.cicids2017 import CICIDS2017PrimitiveModel
from attack.realizability.validator import RealizabilityValidator
from attack.run_cicids2017_primitive_attack import train_envelope
from attack.vae_latent_primitive import LatentAttackConfig, LatentPrimitiveAttack
from datasets.cicids2017 import CICIDS2017Adapter
from src.classifiers.cicids2017d_victims import load_category_victim
from vae.cicids2017_stage_a import load_stage_a

GOLDEN = REPO / "tests/data/golden_cicids2017_dos_lstm_seed42.npz"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_indices(y_test: np.ndarray) -> np.ndarray:
    class_rows = np.flatnonzero(y_test == 1)  # DoS
    picks = np.random.default_rng(43).choice(len(class_rows), 1024, replace=False)
    return class_rows[np.sort(picks)]


def test_golden_provenance_and_row_ids_are_unchanged():
    with np.load(GOLDEN, allow_pickle=False) as expected:
        artifacts = {
            "manifest_sha256": REPO / "data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json",
            "scaler_sha256": REPO / "data/processed/CICIDS_2017_Distrinet/scaler.pkl",
            "victim_sha256": REPO / "outputs/cicids2017distrinet/models/lstm_category.pt",
            "vae_sha256": REPO / "outputs/cicids2017_vae_stage_a/vae_DoS.pt",
        }
        changed = {
            name: {"expected": str(expected[name]), "actual": _sha256(path)}
            for name, path in artifacts.items()
            if str(expected[name]) != _sha256(path)
        }
        assert not changed, f"golden dependencies changed: {changed}"

        processed = REPO / "data/processed/CICIDS_2017_Distrinet"
        y_test = np.load(processed / "y_test_cat.npy")
        indices = _canonical_indices(y_test)[:50]
        assert np.array_equal(indices, expected["test_index"])
        rows = pd.read_parquet(processed / "test.parquet", columns=["sample_id"])
        row_ids = rows.iloc[indices]["sample_id"].astype(str).to_numpy()
        assert np.array_equal(row_ids, expected["row_id"])


def test_golden_attack_outputs_are_unchanged():
    with np.load(GOLDEN, allow_pickle=False) as expected:
        adapter = CICIDS2017Adapter(REPO)
        manifest = adapter.feature_manifest()
        primitive = CICIDS2017PrimitiveModel(manifest)
        transform = adapter.feature_transform()
        center = torch.tensor(transform.center, dtype=torch.float32)
        scale = torch.tensor(transform.scale, dtype=torch.float32)

        processed = adapter._processed
        y_test = np.load(processed / "y_test_cat.npy")
        indices = _canonical_indices(y_test)
        raw_test = np.load(processed / "X_test_pristine.npy", mmap_mode="r")
        raw = torch.tensor(np.ascontiguousarray(raw_test[indices], dtype=np.float32))
        raw_train = np.load(processed / "X_train_pristine.npy", mmap_mode="r")
        envelope = train_envelope(raw_train, primitive.i)
        bounds = primitive.per_flow_bounds(
            raw,
            {
                "p_max": 1460.0,
                "alpha_max": 100.0,
                "mtu_cap": 0.0,
                **{f"env_{name}": value for name, value in envelope.items()},
            },
        )
        vae, _ = load_stage_a(
            adapter, REPO / "outputs/cicids2017_vae_stage_a/vae_DoS.pt",
            expected_class_name="DoS", device="cpu",
        )
        victim = load_category_victim(
            REPO / "outputs/cicids2017distrinet/models/lstm_category.pt",
            adapter=adapter, expected_model_type="lstm", device="cpu",
        )
        idr = np.load(REPO / "outputs/cicids2017_vae_stage_a/idr_DoS.npz")
        realism = {
            "mean": torch.tensor(idr["mean"], dtype=torch.float32),
            "precision": torch.tensor(idr["precision"], dtype=torch.float32),
            "threshold_sq": torch.tensor(float(idr["threshold_sq"]), dtype=torch.float32),
        }
        config_values = json.loads(str(expected["config_json"]))
        target_class = int(config_values.pop("target_class", 0))
        attack_config = {
            key: value
            for key, value in config_values.items()
            if key in LatentAttackConfig.__dataclass_fields__
        }
        config = LatentAttackConfig(**attack_config)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        result = LatentPrimitiveAttack(primitive, config).attack(
            vae, victim, raw, center, scale, bounds, target_class=target_class, realism=realism
        )

        realized_valid = (
            RealizabilityValidator(primitive)
            .validate(result.x_adv_realized_raw, raw)
            .valid.numpy()
        )
        actual = {
            "y_pred_clean": result.clean_logits.argmax(1).numpy()[:50],
            "z0": result.z0.numpy()[:50],
            "z_adv": result.z_adv.numpy()[:50],
            "y_pred_adv": result.realized_logits.argmax(1).numpy()[:50],
            "p": result.controls_realized["p"].numpy()[:50],
            "alpha": result.controls_realized["alpha"].numpy()[:50],
            "realizable": realized_valid[:50],
            "target_success": (result.realized_logits.argmax(1) == 0).numpy()[:50],
            "cost_total": result.primitive_cost.numpy()[:50],
        }
        exact_fields = ("y_pred_clean", "y_pred_adv", "p", "realizable", "target_success")
        changed = [name for name in exact_fields if not np.array_equal(actual[name], expected[name])]
        assert not changed, f"golden exact fields changed for rows {expected['row_id'].tolist()}: {changed}"

        tolerances = {
            "z0": 1e-6,
            "z_adv": 3e-5,  # measured CPU-vs-CUDA max was 2.01e-5
            "alpha": 3e-6,
            "cost_total": 1e-6,
        }
        drift = {
            name: float(np.max(np.abs(actual[name] - expected[name])))
            for name in tolerances
            if not np.allclose(actual[name], expected[name], rtol=1e-6, atol=tolerances[name])
        }
        assert not drift, f"golden numeric fields drifted: {drift}"
