# 10. Reproducibility & Experiment Provenance (MODERATE DETAIL)

## 10.1 Seeds
- Global `SEED = 42` set in `config/paths.py:44` (CICIoT2023-scoped). CICIDS2017 code
  sources `42` independently (preprocessing imports `config.paths.SEED`; attack runners
  seed torch/numpy per run via `experiments.provenance.deterministic_runtime`).
- Preprocessing manifest `seed: 42`; all committed attack artifacts are `_seed42`.
- **Single seed** in the final PrimAttack sweep (runner default is `42,43,44`; only 42
  ran). No across-seed variance (`00_OPEN_ISSUES.md#B8`).

## 10.2 Software / hardware
- `environment.yml`: python **3.11**, torch **2.5.1+cu121**, numpy 2.4.4, scikit-learn
  1.9.0, ART 1.20.1, torchattacks 3.5.1. `old_root_files/requirements.txt` adds xgboost
  3.2.0 and non-portable `file://` pins (not reproducible; prefer `environment.yml`).
- **Victims trained**: python 3.11.15 + CUDA on an RTX 4070 Ti SUPER.
- **Attacks / calibration**: python 3.12.3 + **CPU** (recorded in run manifests) — a
  train-vs-attack environment mismatch to disclose.

## 10.3 Checkpoints
- Victims: `outputs/cicids2017distrinet/models/{mlp,cnn,ft_transformer}_{binary,category}.pt`
  (FT also in `cicids2017distrinet_ft/models/`, smoke in `_ft_smoke/`).
- Generators: `outputs/cicids2017_vae_stage_a/{vae_*.pt, idr_*.npz, stage_a_summary.json}`.
- Calibration: `artifacts/primattack/budget_calibration.json`.

## 10.4 Hashes / manifests
- **Consistency anchor**: `preprocessing_manifest.json` sha256 (`df93d07c…`) and
  `scaler.pkl` sha256 (`c04ed703…`) are **identical** across the classifier, attack, and
  calibration run manifests — one preprocessing lineage everywhere.
- PrimAttack `run_manifest.json` (`experiments.provenance.build_provenance`) records:
  git commit + `dirty` flag, config (`test_limit_per_class=512, attack_steps=40, lr=0.1,
  cost_weight=0.01, seeds=[42], init_noise=0.5, optimizer=optimized,
  calibration_fit_split=train`), and per-checkpoint sha256 (victim, VAE, IDR, calibration,
  preprocessing manifest, scaler). npz artifacts embed provenance arrays
  (`artifact_provenance_arrays`): row ids, class, victim, method, seed, checkpoint sha256s.
- Calibration artifact records source feature/label sha256 and `fit_split="train"` with
  prohibited-inputs list.
- The three joint budget manifests are byte-identical except `budget_name`/`run_id`.

## 10.5 Identity guards (`tests/test_experiment_identity.py`)
Asserts: `n_features == 79`; `scaler.n_features_in_ == 79`; X vs pristine shapes match;
run-manifest sha256 == live preprocessing hash; victim checkpoints carry
`model_type/num_features=79/num_classes=5`; VAE carries `class_name/class_id/
manifest_hash/latent_dim=16/beta_target=0.5`; loaders reject wrong identity. This is the
guardrail that a run used the expected data/model.

## 10.6 Final vs superseded / archived
- **Final CICIDS2017**: preprocessing under `data/processed/CICIDS_2017_Distrinet/`;
  victims `outputs/cicids2017distrinet/`; generators `outputs/cicids2017_vae_stage_a/`;
  PrimAttack `outputs/primattack_budget_sensitivity_full/`; PGD/C&W baselines
  `outputs/cicids2017_baseline_pgd_cw/`; validator artifacts `validation/rules/…`,
  `validation/schema/…`.
- **Superseded/smoke**: `outputs/primattack_smoke`, `primattack_random_control` (older
  commit `378aaf25`, MLP-only), `primattack_budget_sensitivity` (non-full),
  `cicids2017_vae_attacks_{masked,pave_run}` (smoke npz), `_ft_smoke`, `cff*`.
- **Archived/retired**: `old_root_files/retired_lstm_cnn_lstm_2026-09-24/` (57 snapshots
  of retired `LSTMOnly`/`SerialCNNLSTM` victims + their attack results),
  `old_constraints/`, `run_phase{0..4}_*`, CICIoT runners.

## 10.7 Git / version mismatches
- Git HEAD `3380e4a0…` (`main`, 2026-09-24, "Made validator v2 + fixed primattack"); all
  runs record `dirty: true`.
- `outputs/*`, `checkpoints/*`, `*.pkl`, `data/*`, `CLAUDE.md` are **git-ignored** → no
  `VERSION` file → **artifact reproducibility relies on the manifest SHA-256 chain, not on
  git**. Some runs were produced at older commits (e.g. `primattack_random_control`).

## 10.8 Claims
- **Can claim**: deterministic `SEED=42`, full SHA-256 provenance chain from raw inputs →
  preprocessing → scaler → checkpoints → attack artifacts, with identity-guard tests.
- **Must NOT claim**: bit-for-bit reproducibility across machines (train GPU/py3.11 vs
  attack CPU/py3.12; dirty trees; git-ignored artifacts), or that superseded/archived
  (LSTM/serial, CICIoT) numbers reflect the current pipeline.
