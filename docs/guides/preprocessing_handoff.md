# Preprocessing Handoff

For the teammates owning dataset preprocessing. Goal: turn raw NIDS captures into
**leakage-safe, model-ready splits** that both the NIDS classifiers and the per-class
β-VAE can load without further work. Read this first; dataset-specific detail lives in
`cicids2017_preprocessing_prompt.md` and `src/preprocessing/ciciot2023/`.

Two datasets:
- **CICIoT2023** — pipeline already exists. Your job is to *run and verify* it, not rewrite.
- **CICIDS2017** — new. Follow `cicids2017_preprocessing_prompt.md`, hitting the same
  output contract below.

---

## 1. What preprocessing does, and why

Five stages, in this order. The order is the point — it prevents data leakage.

1. **Split FIRST (temporal, leakage-safe).** Forward-chaining split by time/shard:
   earliest flows → train, latest → **test**. `val = 0.10`, `test = 0.20`, `train = 0.70`.
   *Why:* the model must never see "future" traffic; and every statistic fitted later
   must be fitted on train only, so the split has to exist before anything is fitted.

2. **Clean (per-row + train-fitted bounds).** Drop NaN/Inf and blank rows; clip
   continuous/count extremes at the train 99.99th percentile; round integer
   counts; map positive protocol/service/flag presence values to `1`.
   *Why:* flow features have Inf/overflow and heavy tails, while ordinary
   rounding of sparse fractional indicators silently erases valid positives.

3. **Scale (fit on TRAIN only).** `RobustScaler`, fit on train, then *transform* val/test
   with those same parameters. *Why:* comparability across the two datasets, robustness
   to outliers, and the VAE needs an **invertible, unbounded** transform (see §3).

4. **Balance TRAIN only.** Undersample majority classes / handle rare classes on the
   train split. **Never touch val/test** — they must reflect the real distribution so
   metrics are honest.

5. **Emit artifacts + a manifest.** Save arrays, scaler, encoders, class weights, and a
   run manifest recording every count, bound, and hash for reproducibility.

**The one rule that governs all of it:** *everything fitted (scaler, clip bounds,
imbalance sampling, class weights, imputation) is computed on the TRAIN split only.*
Any statistic touching val/test before the split = leakage = results are invalid.

Global seed `SEED = 42` (`config/paths.py`) everywhere for determinism.

---

## 2. Output contract (what you must produce)

Write to `data/processed/` (CICIoT2023) or `data/processed/CIC-IDS-2017/` (CICIDS2017).
Column order in every array MUST equal the frozen schema order.

| Artifact | Type | Used by |
|---|---|---|
| `X_{train,val,test}` | `(N, F)` float32, **scaled** | VAE + NIDS |
| `y_{split}` (multiclass) | int | NIDS |
| `y_{split}_cat` / `label_category` | int, coarse groups | **VAE (per-class)** |
| `y_{split}_bin` | int, benign=0/attack=1 | NIDS |
| `scaler.pkl` | fitted `RobustScaler` | **VAE (required)** |
| `label_encoder` / `category_encoder` | sklearn encoders + name JSON | both |
| `class_weights_*` | float32, balanced, from sampled train | NIDS |
| schema/manifest | feature order, typing, **partition**, constraints, mask | everything |
| run/preprocessing report | per-step counts, bounds, hashes | reproducibility |

CICIoT2023 file names are exactly what `src/vae/train.py` loads
(`X_train.npy`, `y_train.npy`, `y_train_cat.npy`, `scaler.pkl`, …). Match them.
CICIDS2017 additionally keeps an **unscaled "pristine" matrix** + a metadata sidecar
(Flow ID, IPs, Timestamp) for the adversarial validity checks.

---

## 3. What the VAE requires (the hard bits)

The VAE is `MixedInputBetaVAE`, trained as **one β-VAE per category**. It consumes:

1. **Fixed-order scaled float32 matrix.** Reordering columns silently corrupts the
   scaler and every checkpoint. Freeze the feature order once.
2. **A per-split integer category array** (`y_*_cat` / `label_category`). The VAE subsets
   `X` by this, so **each category must have enough train rows to fit** (CICIoT2023 floor
   ≈ 500). Tiny classes can't get their own VAE — group or exclude them (keep them for
   the NIDS/binary labels).
3. **A train-fitted, invertible `RobustScaler`.** The VAE inverse-transforms columns to
   build raw-space targets and maps categorical values into scaled space. Do **not** use
   a bounded/non-invertible transform (no Min-Max squash on top) unless the VAE owner
   changes the decoder output activation to match.
4. **A feature partition** — index lists by role: `continuous_idx`,
   `independent_binary_idx` (strict {0,1}), and categorical column index(es). This is the
   analog of `src/vae/schema.py:get_partition`; ship it in the schema artifact.

> **CICIDS2017 caveat (owner's call, not preprocessing's):** today the model hardcodes
> the CICIoT2023 shape (39 features, protocol allowlist, derived TCP/UDP/ICMP/IGMP
> columns). CICIDS2017 has a different feature set and a single `Protocol` column, so the
> model must be generalized before it can train on it. **Freeze the CICIDS2017
> schema/partition only after the VAE owner confirms the target shape.**

---

## 4. Acceptance checklist (assert before "done")

- Column order in every saved array == schema/manifest order.
- No NaN/Inf in any model-ready matrix.
- `scaler.inverse_transform(X_train)` round-trips to raw within tolerance.
- Scaler / clip bounds / class weights provably fit on the **train mask only** (manifest).
- Split is chronological and non-overlapping; **val/test are not resampled**.
- Row counts reconcile (dropped + kept == raw total).
- `label_category` present per split, every group non-empty in train **and** val, and
  each VAE-trained group is large enough to fit.
- Categorical/protocol values ⊆ the allowed set; binary columns ∈ {0,1} exactly.

---

## 5. Pointers

- CICIoT2023 pipeline: `src/preprocessing/ciciot2023/pipeline.py` (`main()`),
  schema `src/preprocessing/schema.py`, paths `config/paths.py`.
- CICIDS2017 full spec: `cicids2017_preprocessing_prompt.md`.
- VAE side (reference for the contract): `src/vae/{model,dataset,train,config,schema}.py`.
- Project invariants: `CLAUDE.md` ("REFACTOR PRINCIPLE", locked decisions).
