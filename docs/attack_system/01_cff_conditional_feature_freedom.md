# 01 — Conditional Feature Freedom (CFF)

**Source:** `src/preprocessing/conditional_feature_freedom.py` (one standalone command
module, ~1127 lines). **Companion (older) doc:** `docs/CFF.md`.

CFF answers one question, per traffic class, from **training data only**:

> *For attack class c, which features are "free" (can vary a lot without being pinned down
> by the other features), and which are effectively determined by the rest of the flow?*

It produces a **ranking** and a set of boolean `.npy` **candidate-perturbation masks**
(top-10 / top-25 / top-50 %). These masks feed the *masked* VAE attack
(`run_cicids2017_vae_attacks.py::load_cff_mask`) as the attacker's degrees of freedom.

CFF is **not** a constraint and does **not** prove that a feature is physically mutable — it
is a *data-driven proxy for conditional freedom*. That caveat is stated in the module
docstring (lines 13–16) and repeated in the report language.

---

## 1. The intuition

If you can predict feature `X_i` almost perfectly from all the other features, then `X_i` is
*structurally constrained* — an attacker cannot move it independently without breaking the
flow's internal consistency. If, on the other hand, `X_i` still has a large unexplained
spread after conditioning on everything else, it has genuine **conditional freedom**.

CFF measures exactly this: fit a regressor `f_i` that predicts `X_i` from `X_{-i}` (every
*other* feature), and compare the **spread of the residual** to the **natural spread of the
feature**.

---

## 2. The math (exactly as implemented)

For each class `c` and feature `X_i` (`compute_feature_cff`, lines 443–497):

```
r_i        = X_i - f_i(X_{-i})                         # out-of-fold residual
W_r(i)     = Q_0.95(r_i) - Q_0.05(r_i)                 # robust residual width  (residual_span)
W_x(i)     = Q_0.95(X_i) - Q_0.05(X_i)                 # robust natural width    (feature_span)
CFF_raw    = W_r(i) / (W_x(i) + epsilon)
CFF_i      = clip(CFF_raw, 0, 1)                        # only the final score is clipped
```

Also recorded per (class, feature): `cv_r2` (out-of-fold R²; may be very negative if the
model is worse than a constant), `normalized_mae = MAE / (W_x + eps)`, `residual_q05`,
`residual_q95`, and `suspicious_model_fit = (cv_r2 < -1.0)`.

- **High CFF** → large unexplained residual relative to the feature's own range → high
  conditional freedom → good perturbation candidate.
- **Low CFF** → feature is largely predictable from the others → structurally constrained.

`epsilon = 1e-12` is a *pure arithmetic guard*, not the eligibility criterion (see §4).

---

## 3. The estimator (fixed, deterministic)

`make_lgbm_regressor` (matches `docs/CFF.md`, LightGBM 4.6.0):

```python
LGBMRegressor(
    objective="regression", n_estimators=150, learning_rate=0.05,
    num_leaves=31, max_depth=-1, min_child_samples=20,
    reg_alpha=0.0, reg_lambda=1.0, subsample=1.0, colsample_bytree=1.0,
    random_state=42, n_jobs=-1, verbosity=-1, deterministic=True, force_col_wise=True,
)
```

- **Out-of-fold predictions** (`_oof_predictions`) via `KFold(n_splits=cv, shuffle=True,
  random_state=42)` — the residual is *never* an in-sample residual, so an over-fit model
  cannot fake low residual spread.
- No hyper-parameter search, no SHAP, no classifier gradient, no VAE, no correlation score.
  One fixed estimator across all classes and target features.
- **No fallback estimator:** if LightGBM is not importable, `require_lightgbm()` fails
  loudly (the thesis pins `lightgbm==4.6.0` in `environment.yml`).
- `--estimator-check` optionally re-runs the first class with `ExtraTreesRegressor` and
  reports the Spearman rank correlation between the two estimators (a robustness sanity
  check, not part of the main score).

---

## 4. Eligibility gating (which features even get a score)

Before fitting, every (class, feature) is screened (`_base_feature_record` +
`detect_degenerate_feature`). A feature receives `cff_score = 0`, `eligible = False`, and
skips LightGBM if **any** of these hold:

| Reason | Condition |
|--------|-----------|
| **Hard-excluded** | feature ∈ `DEFAULT_HARD_EXCLUDE` (IDs, timestamps, ports, `Protocol Type`, label) or `--exclude-feature` |
| **Constant** | `n_unique <= 1` |
| **Near-constant** | `dominant_fraction >= 0.995` (`--near-constant-threshold`) |
| **Span-degenerate** | `W_x(i) < 1e-3` (`--min-absolute-span`) **or** `W_x(i) < 0.02 * W_g(i)` (`--span-degeneracy-ratio`) |

where `W_g(i)` is the **pooled** robust span over deterministic samples from *all requested
classes* (`global_feature_spans`, computed once in `compute_all_classes`). Span-degeneracy
catches a feature that is multi-valued but tightly clustered *within one class* compared to
its cross-class scale.

Hard exclusion controls whether a feature can be *selected as a target*; it does **not**
remove that feature from the *predictor set* (a hard-excluded feature can still help predict
another feature).

---

## 5. Sampling, folds, determinism

`deterministic_class_sample` draws at most `sample_per_class` rows **without replacement**
per class using `numpy.random.default_rng(42)`, then **sorts** the selected indices before
materialization (order-stable). Defaults:

| Setting | Normal | `--fast` |
|---|---:|---:|
| Rows per class (`--sample-per-class`) | 20,000 | 10,000 |
| LightGBM estimators (`--n-estimators`) | 150 | 75 |
| OOF folds (`--cv`) | 3 | 2 |
| Seed (`--seed`) | 42 | 42 |
| `--near-constant-threshold` | 0.995 | 0.995 |
| `--span-degeneracy-ratio` | 0.02 | 0.02 |
| `--min-absolute-span` | 1e-3 | 1e-3 |
| `--learning-rate` | 0.05 | 0.05 |
| `--num-leaves` | 31 | 31 |
| `--min-child-samples` | 20 | 20 |

**Train-only guarantee:** the module opens only the *training* artifacts
(`X_train.npy` / `X_train_pristine`-space, `y_train_cat.npy`, `category_names.json` /
label encoders). `_validate_training_path` rejects an explicit val/test path. Validation
and test data never touch CFF.

The default representation is model-ready RobustScaler space; the CFF width *ratio* is
invariant under positive affine scaling, so scaling does not change the ranking.

---

## 6. Ranking and selection (`_rank_class_frame`, `select_top_fraction`)

1. Eligible rows are sorted by **descending `cff_score`**, with `feature_index` as a
   deterministic tie-break (`kind="mergesort"`, stable).
2. Ranks `1..K` are assigned only to eligible features; ineligible features have `rank = NA`.
3. For each fraction in `SELECTION_FRACTIONS = {top10:0.10, top25:0.25, top50:0.50}`:
   `selected_count = ceil(fraction * n_eligible)`; the top `selected_count` eligible
   features are flagged `selected_top{10,25,50} = True`.

The denominator is the **eligible count**, not 79. There is no absolute magic threshold like
"CFF > 0.5". **`top25` is the primary experimental CFF mask.**

---

## 7. Outputs (`save_results`)

A run writes (under `outputs/cff/` by default; the masked attack reads
`outputs/cff_cicids2017distrinet/`):

```
cff_scores_all.csv            # every (class, feature) record
cff_scores_<Class>.csv        # per-class
cff_masks.json                # boolean selections in one JSON
feature_order.json            # the exact 79-name order (asserted against the manifest)
cff_run_metadata.json         # config + seed + LightGBM version + timing
cff_summary.md                # human-readable summary
masks/<Class>_eligible.npy    # bool[n_features]
masks/<Class>_top10.npy
masks/<Class>_top25.npy
masks/<Class>_top50.npy
```

`build_feature_mask` validates each mask is a bool array in exact model-feature order
before saving.

---

## 8. How CFF enters the attack

CFF masks are consumed **only by the masked VAE attack ladder**
(`run_cicids2017_vae_attacks.py`), via `load_cff_mask(repo, manifest, class_name, tier)`:

```python
order = json.load(cff/"feature_order.json"); manifest.assert_names_match(order)
mask  = np.load(cff/"masks"/f"{class_name}_{tier}.npy").astype(bool)
mask[manifest.derived_indices()] = False    # never let the attacker directly move a DERIVED feature
return torch.tensor(mask, bool)
```

The result becomes the `mutable_mask` handed to the residual/latent generator: **only CFF-
selected, non-derived features receive a direct perturbation**; everything else is frozen or
recomputed.

> The **direct primitive attack** and the **VAE latent primitive attack**
> (docs 07–08) do **not** use CFF masks. They perturb the two primitives `(p, alpha)`
> and the *realizability model's roles* decide what changes. CFF governs the *masked
> ablation ladder* (`run_cicids2017_vae_attacks.py`, `--mask-source cff`). This distinction
> matters for the report: CFF is one of two independent ways the codebase constrains the
> attacker's degrees of freedom.

---

## 9. Self-test

`--self-test` (`run_synthetic_self_test`, seeded by `SEED`) builds synthetic features with
known behaviour — a predictable feature (low CFF), an independent feature (high CFF), a
degenerate feature (excluded), and an excluded name — and asserts CFF ranks them correctly.
Use it as a regression check after touching the module.
