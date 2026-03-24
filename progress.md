# Progress Tracker

## Code Review & Bug Fixes (2026-03-22)

### Files Reviewed
| File | Status | Issues Found |
|------|--------|-------------|
| `src/train_lightgbm.py` | Fixed | 3 bugs |
| `src/train_cnn-lstm.py` | Fixed | 1 bug |
| `src/train_vae.py` | Clean | None |
| `src/run_attacks.py` | Fixed | 1 critical bug + 1 minor |
| `src/debug.py` | Clean | None |
| `data/ciciot2023/preprocess.py` | Clean | None |

### Bugs Found & Fixed

#### 1. `train_lightgbm.py` — Undefined variables & missing save
- **Bug**: Lines 64-68 used `lgbm` instead of `clf` (NameError at runtime)
- **Bug**: Line 72 used `NIDS_DIR` instead of `MODEL_DIR` (NameError)
- **Bug**: Line 64 `lgbm.booster.num_trees()` — wrong attribute path, changed to `clf.n_classes_`
- **Missing**: Script did not save sklearn pickle (`nids_lgbm_sklearn.pkl`), which `info.md` specifies as the canonical way to load the model for label alignment. Added `joblib.dump(clf, ...)`.
- **Fix**: Corrected all variable references, added `import joblib`, added sklearn pickle save.

#### 2. `train_cnn-lstm.py` — Missing `map_location`
- **Bug**: Line 129 `torch.load(best_path)` missing `map_location=DEVICE`. Would fail when loading a GPU-saved checkpoint on a CPU machine.
- **Fix**: Added `map_location=DEVICE`.

#### 3. `run_attacks.py` — VAE encoder input order mismatch (CRITICAL)
- **Bug**: The VAE was trained in `train_vae.py` on *reordered* input `[continuous_features | binary_features]` (i.e., features 0-22 then 38 then 23-37). But `run_attacks.py` fed the original 39-feature order directly into the encoder. This means:
  - Position 23: encoder expected Protocol Type (continuous) but received HTTP (binary)
  - Positions 24-37: encoder expected binary features shifted by one position
  - Position 38: encoder expected LLC (binary) but received Protocol Type (continuous)
  - The decoder output was correctly reassembled via `decode_to_full()`, so only encoding was affected.
- **Impact**: Degraded latent representations → suboptimal attack performance. Existing ASR results in `info.md` were obtained with this bug, so they may improve after re-running attacks.
- **Fix**: Added feature reordering in `encode()`:
  ```python
  x_reordered = torch.cat([x[:, self.continuous_idx], x[:, self.binary_idx]], dim=1)
  ```
- **Minor**: Also fixed dropout mismatch (was 0.1 in run_attacks.py vs 0.2 in train_vae.py). No functional impact in eval mode but corrected for consistency.

### Action Required
- **Re-run `run_attacks.py`** after the VAE encoder fix — existing results are invalid.
- **Re-run `train_lightgbm.py`** to generate the missing `nids_lgbm_sklearn.pkl`.

---

## Pipeline Status

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| Preprocessing | `preprocess.py` | Done | |
| LightGBM Training | `train_lightgbm.py` | Done (needs re-run) | Now saves sklearn pickle |
| CNN-LSTM Training | `train_cnn-lstm.py` | Done | |
| VAE Training | `train_vae.py` | Done | |
| VAE Attacks | `run_attacks.py` | Done (needs re-run) | Critical encoder fix applied |
| PGD Baseline | `baseline_pgd.py` | TODO | |
| Evaluation | `evaluate.py` | TODO | ASR, Protocol Validity, IDSR |

## Preliminary Results (pre-fix, N=500 per class)

| Class | ASR CNN-LSTM | ASR LightGBM |
|-------|-------------|-------------|
| DDoS  | 0.944       | 0.986       |
| DoS   | 0.108       | 1.000*      |
| Mirai | 0.012       | 0.798       |
| Recon | 0.434       | 1.000*      |

*LightGBM results pending label alignment fix.
