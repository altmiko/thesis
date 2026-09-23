# 09 — End-to-End Pipeline, Row Selection, Seeds & Training Configs

This is the "put it all together" document: the exact order of operations, **which rows are
attacked and how they are chosen**, every seed, the VAE and victim training configurations,
and the provenance/artifact contract.

---

## 1. The full pipeline (offline, per experiment)

```
STAGE 0  Preprocessing (already done, not re-run by the attack)
  raw CICFlowMeter CSVs
   -> leakage-safe forward temporal split by shard  (train / val / test)
   -> RobustScaler fit on TRAIN only  (scaler.pkl)
   -> data/processed/CICIDS_2017_Distrinet/
        X_{train,val,test}.npy            (RobustScaler space)
        X_{train,val,test}_pristine.npy   (raw units)   <-- attacks read pristine
        y_{train,val,test}_cat.npy
        preprocessing_manifest.json  (modelling_feature_names, 79)
        label_encoders.json          (Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4)

STAGE A  Per-class beta-VAEs (vae/cicids2017_stage_a.py)   [victim-independent]
  for class in {DoS, DDoS, Recon, BruteForce}:
     train on TRAIN rows of that class; select checkpoint on VAL loss;
     calibrate IDR (Mahalanobis) on VAL rows -> vae_<class>.pt, idr_<class>.npz

STAGE CFF  Conditional Feature Freedom (preprocessing/conditional_feature_freedom.py)
  rank perturbable features per class on TRAIN only -> outputs/cff_.../masks/*.npy
  (consumed only by the masked ladder run_cicids2017_vae_attacks.py)

STAGE C  Mined constraints (constraints/cicids2017_distrinet/mined.json)  [static, TRAIN-mined]

ATTACK  (primitive / latent / input-baseline / masked)
  for class -> pick rows -> load VAE+IDR -> build A4 engine -> per-flow bounds
    for victim (mlp,cnn,lstm,serial) -> for seed (42,43,44):
        optimize -> project/quantize -> generate -> validators + IDR -> metrics + NPZ
```

The attack never fits anything on val/test: it *reads* the train-fit scaler, the train-mined
CFF masks and rules, the train-fit Stage-A VAEs, and the val-calibrated IDR gate.

---

## 2. Which rows are attacked, and how they are picked

**Population.** Only rows of the **pristine test split** (`X_test_pristine.npy`) whose label is
one of the four attack classes. Benign test rows are never attacked (Benign is the target).

**Selection function** (`run_cicids2017_primitive_attack.py::_class_rows`, reused by the latent
and input-baseline runners):

```python
def _class_rows(y, class_id, limit, seed):
    idx = np.flatnonzero(y == class_id)          # all test rows of this class
    if limit is not None and len(idx) > limit:
        pick = np.random.default_rng(seed).choice(len(idx), limit, replace=False)
        idx = idx[np.sort(pick)]                  # sorted -> stable order
    return idx
```

Called as `_class_rows(test.y, cid, test_limit, seed=42 + cid)`:

- **`test_limit = 1024` rows per class** (default). If a class has fewer, all are used.
- **The sampling seed is `42 + class_id`, NOT the attack seed.** Consequences:
  - the *same* 1024 rows are attacked by every victim and every optimization seed (42/43/44),
    so success differences reflect the *attack*, not different samples;
  - each class draws a different subset (`42+1 … 42+4`), so classes are independent;
  - sampling is **without replacement** and the indices are **sorted**, so the row order is
    deterministic and reproducible.
- Row identities are preserved end-to-end: `load_row_ids` reads `test.parquet:sample_id`, and
  `all_row_ids[idx]` is written into every NPZ (`row_id`) so any adversarial can be traced back
  to its exact source flow.

**Masked ladder** (`run_cicids2017_vae_attacks.py`): test rows selected the same way
(`_class_x(test, class_id, test_limit, 42 + class_id)`); Stage-B training rows (for the A6
victim-guided head) are drawn from **train** with a different seed `142 + class_id` to avoid
overlap with anything test-side.

---

## 3. Every seed in one place

| Purpose | Seed | Code |
|---------|------|------|
| Global repo seed | `42` | `config/paths.py::SEED` |
| Test-row sampling (per class) | `42 + class_id` | `_class_rows` |
| Stage-B train-row sampling | `142 + class_id` | masked runner `_class_x` |
| Attack optimization | `42, 43, 44` | `--seeds` |
| Per-step determinism | current attack seed | `deterministic_runtime(seed)` |
| Stage-A VAE training | `42` | `StageAConfig.seed` |
| DataLoader shuffling (Stage A) | config seed | `torch.Generator().manual_seed` |
| CFF sampling / KFold / LightGBM | `42` | `default_rng(42)`, `KFold(random_state=42)`, `random_state=42` |

`deterministic_runtime(seed)` (`experiments/provenance.py`) seeds `random`, `numpy`, `torch`,
CUDA, sets `cudnn.deterministic=True`, `cudnn.benchmark=False`,
`torch.use_deterministic_algorithms(True, warn_only=True)`, and returns the resulting flags for
the provenance record.

---

## 4. Stage-A per-class β-VAE training config (`vae/cicids2017_stage_a.py::StageAConfig`)

Victim-independent. Trained on TRAIN rows of one attack class; VAL selects the checkpoint and
calibrates the realism gate; **test is never loaded**.

| Hyper-parameter | Value |
|-----------------|-------|
| `latent_dim` | 16 |
| `encoder_input_transform` | `asinh` |
| `encoder_hidden` / `decoder_hidden` | `(128, 64)` / `(64, 128)` |
| `epochs` | 30 (early stop `patience = 5` on val loss) |
| `batch_size` | 2048 |
| `learning_rate` / `weight_decay` | `1e-3` / `1e-5` (AdamW) |
| `beta_target` / `beta_warmup_epochs` | `0.5` / 10 (linear warmup via `BetaScheduler`) |
| `free_bits_lambda` | 0.1 (anti-collapse) |
| `continuous_likelihood` | `laplace` |
| `grad_clip` | 5.0 |
| `seed` | 42 |
| constraint penalties during training | **off** (`constraint_l1_weight = l2_weight = 0`) |

Extra details:
- **Loss feature weights** (`_fit_loss_feature_weights`): per-feature `1/max(RMS, 1)`
  normalized to mean 1, fit on TRAIN — prevents zero-IQR rare columns dominating the ELBO.
- **Checkpoint selection:** lowest val loss (`compute_manifold_elbo`), `best_state` restored
  before saving; the manifest content hash is stored and re-checked on load
  (`load_stage_a` fails on class/id/hash mismatch).
- **IDR calibration** (`fit_idr`): encode VAL rows → `mean`, `cov + 1e-4·I`, `precision`;
  `threshold_sq` = 95th percentile of Mahalanobis `d²` (val-empirical p95). Saved to
  `idr_<class>.npz` with `val_in_distribution_rate` and `fit_rows`.

---

## 5. Victim models (`classifiers/cicids2017d_victims.py`)

Four pre-trained category classifiers loaded as frozen victims that still pass input
gradients: `mlp`, `cnn`, `lstm`, `serial` (custom PyTorch, `classifiers/models.py::get_model`).
Each maps a 79-feature RobustScaler-space vector → 5 category logits
(`Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`).

`load_category_victim` is defensively strict — it fails on any mismatch of:
preprocessing-manifest SHA-256, category class order, feature count, model type, or class
count (against `classifier_run_manifest.json` and the checkpoint's own metadata). Parameters
are frozen (`requires_grad_(False)`); a `_GradientSafeVictim` wrapper disables cuDNN for
recurrent victims (LSTM/serial) so CUDA input gradients stay deterministic.

---

## 6. Attack hyper-parameters side by side

| | Direct primitive (doc 07) | VAE latent (doc 08) | Input PGD baseline |
|--|--|--|--|
| Optimizer var | `(p, alpha)` | `z_adv` | 79 scaled features |
| Steps | 40 | 120 | (CLI) |
| LR | 0.1 (Adam) | 0.08 (Adam) | `alpha` step (PGD) |
| Objective | CE→Benign + cost | CW (or CE) + latent/cost/realism | CE→Benign (L∞) |
| Bounds | per-flow train envelope caps | `epsilon_z=10` L2 ball on z + caps | L∞ `epsilon` ball |
| Rows/class | 1024 | 1024 | 1024 |
| Seeds | 42,43,44 | 42,43,44 | 42,43,44 |
| Realizability layer | yes | yes | **no** |

`p_max = 1460` (≈ one Ethernet MTU of forward padding), `alpha_max = 100`, `cost_weight = 0.01`
shared by the two realizability-aware methods.

---

## 7. The A0–A6 ablation ladder (`experiments/ablations.py`)

One code path, selected by preset, used by the masked runner and to build the A4 validator
engine everywhere:

| Preset | decoder | L0 | L1 | L2 | residual head | Stage-B |
|--------|---------|----|----|----|--------------|---------|
| A0 | legacy structured | – | – | – | – | – |
| A1 | typed | – | – | – | – | – |
| A2 | typed | ✔ | – | – | – | – |
| A3 | typed | ✔ | ✔ | – | – | – |
| **A4** | typed | ✔ | ✔ | ✔ | – | – |
| A5 | typed | ✔ | ✔ | ✔ | ✔ | – |
| A6 | typed | ✔ | ✔ | ✔ | ✔ | ✔ (victim-guided) |

`build_ablation("A4", ...)` is what every primitive/latent runner calls to obtain the
independent mined validator engine (Layer 0+1+2). Layer 1 is refit on `raw_train[:200000]`
(coverage 0.99); Layer 2 loads `mined.json`.

---

## 8. Provenance & artifacts (`experiments/provenance.py`)

Every run is reproducible by construction:

- `ensure_fresh_output_dir` refuses to overwrite a non-empty output dir (no silent
  clobbering).
- `build_provenance` records: dataset, `method_id`, git commit + dirty flag +
  `status_sha256`, a **source-tree SHA-256** (hashes all `src/`+`scripts/` `.py` bytes,
  independent of git), the preprocessing-manifest hash, the scaler hash, **per-checkpoint
  SHA-256** (each victim, VAE, IDR), the full config, the environment (python/numpy/torch/
  sklearn/CUDA/cuDNN versions), and a 20-char `run_id` = hash of the canonical payload.
- Each NPZ embeds `artifact_provenance_arrays`: per-row `row_id`, class/victim/method/seed,
  git commit, source-tree hash, run id, config JSON, and checkpoint identifiers — so a single
  `.npz` is a fully self-describing evidence file.

`load_row_ids(processed_dir, "test")` reads `test.parquet:sample_id` and the `[idx]` slice ties
each adversarial row back to its exact source flow.

---

## 9. Reproducing a run (commands)

```bash
# Direct primitive-domain attack (baseline)
python -m attack.run_cicids2017_primitive_attack \
    --classes DoS,DDoS,Recon,BruteForce --victims mlp,cnn,lstm,serial \
    --test-limit 1024 --steps 40 --learning-rate 0.1 --seeds 42,43,44 \
    --output-dir outputs/cicids2017_primitive_attack

# VAE latent primitive-constrained attack (proposed)
python -m attack.run_cicids2017_vae_latent_attack \
    --variant full --steps 120 --learning-rate 0.08 --objective cw --epsilon-z 10 \
    --seeds 42,43,44 --output-dir outputs/cicids2017_vae_latent_attack

# CFF ranking (train-only)
python -m preprocessing.conditional_feature_freedom \
    --input data/processed/CICIDS_2017_Distrinet --output-dir outputs/cff_cicids2017distrinet
```

Metrics land in `<output-dir>/attack_results.json`; raw evidence in
`<output-dir>/attack_artifacts/*.npz`; provenance in `<output-dir>/run_manifest.json`.
