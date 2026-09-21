# Conditional Feature Freedom (CFF)

## Status

Implemented in one standalone command module:

```text
src/preprocessing/conditional_feature_freedom.py
```

The implementation does not change preprocessing, classifiers, VAEs, validators, attacks, or their existing masks. It reads saved training artifacts and writes an independent CFF artifact bundle. The same module supports the repository-native CICIoT2023 and CICIDS2017-DistriNet feature contracts.

## Repository findings before implementation

### Relevant files

| Concern | Repository source |
|---|---|
| Immutable 39-feature model order | `src/preprocessing/schema.py::FEATURE_NAMES` |
| Fine-label column and 34-to-8 mapping | `src/preprocessing/schema.py::LABEL_COLUMN`, `CATEGORY_MAP` |
| Split, cleaning, scaling, and saved arrays | `src/preprocessing/ciciot2023/pipeline.py` |
| Forward source-order split logic | `src/preprocessing/ciciot2023/splitter.py` |
| Current model-ready arrays | `data/processed/X_{train,val,test}.npy` |
| Current encoded labels | `data/processed/y_{train,val,test}{,_cat,_bin}.npy` |
| Coarse category names | `data/processed/category_names.json` |
| Train-fitted scaler | `data/processed/scaler.pkl` |
| Existing attack mask representation | `src/attack/latent_infra.py::PerturbationMask` |
| Preprocessing provenance | `data/processed/run_manifest.json` |

### Canonical training data

The CFF command defaults to `data/processed/` and opens only:

```text
data/processed/X_train.npy
data/processed/y_train_cat.npy
data/processed/category_names.json
```

`X_train.npy` has shape `(1,226,533, 39)` and dtype `float32`. It is the saved, sampled training partition in model-ready `RobustScaler` space. The preprocessing scaler was fitted on the complete natural training partition before the saved training rows were sampled; validation and test rows did not fit the scaler. The CFF command does not open `X_val.npy`, `X_test.npy`, or either holdout's labels.

The source preprocessing pipeline constructs the split before fitted preprocessing. Classes with at least three source shards use forward source-order assignment; two-shard classes use an ordered hybrid; one-shard classes use contiguous ordered blocks. CFF does not reconstruct or redefine this split—it consumes the already materialized training partition.

### Labels and selected classes

The labelled source Parquet contains:

- `Label`: the 34-class fine attack label;
- `category`: the configured 8-class grouping used by the 8-class NIDS and per-class VAE workflow.

The repository-native CFF input uses `y_train_cat.npy`, decoded by `category_names.json`. The configured category order is:

```text
Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web
```

When `--classes` is omitted, CFF detects observed training categories and selects every non-benign category:

```text
BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web
```

Class names passed through `--classes` are exact and case-sensitive. Scores are computed independently per selected class; classes are never pooled for the main score.

### Canonical model feature order

All masks use this exact order from `src.preprocessing.schema.FEATURE_NAMES`:

1. `Header_Length`
2. `Protocol Type`
3. `Time_To_Live`
4. `Rate`
5. `fin_flag_number`
6. `syn_flag_number`
7. `rst_flag_number`
8. `psh_flag_number`
9. `ack_flag_number`
10. `ece_flag_number`
11. `cwr_flag_number`
12. `ack_count`
13. `syn_count`
14. `fin_count`
15. `rst_count`
16. `HTTP`
17. `HTTPS`
18. `DNS`
19. `Telnet`
20. `SMTP`
21. `SSH`
22. `IRC`
23. `TCP`
24. `UDP`
25. `DHCP`
26. `ARP`
27. `ICMP`
28. `IGMP`
29. `IPv`
30. `LLC`
31. `Tot sum`
32. `Min`
33. `Max`
34. `AVG`
35. `Std`
36. `Tot size`
37. `IAT`
38. `Number`
39. `Variance`

The implementation imports the list; it does not duplicate it. Every selected feature is resolved through an exact `feature name -> canonical index` mapping before a mask is saved.

### Existing mask representation and deliberate non-integration

The live attack code represents one global three-tier mask as `PerturbationMask(full_indices, partial_indices, frozen_indices, partial bounds)`. CFF produces class-specific boolean candidate-selection masks. Converting a Boolean CFF selection into the existing Full/Partial/Frozen policy would require an additional policy decision that CFF does not justify. Therefore the implementation exports easy-to-load Boolean `.npy` masks but does not rewrite or silently replace `PerturbationMask`.

Historical documentation refers to a numeric `perturbation_mask.npy`, but the current preprocessing pipeline does not emit that artifact. No semantically misleading numeric compatibility file was added.

## Method

For each selected traffic class $c$ and canonical feature $X_i$:

1. deterministically sample at most `sample_per_class` rows from the saved training partition for each requested class;
2. concatenate those class samples and compute a pooled training-only robust span for every feature;
3. reject targets whose within-class span is tiny absolutely or relative to the pooled span;
4. use every other configured model feature $X_{-i}$ as predictors;
5. produce explicit shuffled-KFold out-of-fold predictions with a fixed LightGBM regressor;
6. calculate residuals $r_i = X_i - \hat X_i$;
7. normalize the robust residual width by the feature's class-local robust natural width; and
8. rank only eligible features.

The equations are:

$$
\hat X_i = f_i(X_{-i}), \qquad r_i = X_i - \hat X_i
$$

$$
W_r(i) = Q_{0.95}(r_i)-Q_{0.05}(r_i)
$$

$$
W_x(i) = Q_{0.95}(X_i)-Q_{0.05}(X_i)
$$

The pooled reference width over deterministic samples from all requested classes is:

$$
W_g(i) = Q_{0.95}(X_i^{\mathrm{pooled}})-Q_{0.05}(X_i^{\mathrm{pooled}}).
$$

A class-feature pair is span-degenerate when:

$$
W_x(i) < 10^{-3}
\quad\lor\quad
W_x(i) < 0.02\,W_g(i).
$$

The thresholds are configurable with `--min-absolute-span` and `--span-degeneracy-ratio`. Span-degenerate targets receive CFF zero and skip LightGBM fitting.

$$
\operatorname{CFF}_i = \operatorname{clip}\left(\frac{W_r(i)}{W_x(i)+\epsilon},0,1\right)
$$

Only CFF is clipped. OOF $R^2$ remains negative when the model performs worse than a constant baseline. `normalized_mae` is:

$$
\frac{\operatorname{MAE}(X_i,\hat X_i)}{W_x(i)+\epsilon}.
$$

`residual_q05` and `residual_q95` are also saved. They can later define an approximate conditional interval around a fitted prediction, but this implementation does not build a constraint engine or integrate such intervals into PGD, C&W, or the VAE.

## Scientific interpretation

CFF is a data-driven proxy for ranking candidate perturbable features within each traffic class. LightGBM approximates the conditional mean of one feature given the remaining model features, and normalized out-of-fold residual spread measures how much variation remains unexplained. High CFF indicates greater observed class-conditional freedom, while low CFF indicates greater conditional predictability or structural constraint. CFF does not prove attacker accessibility, packet-level mutability, functionality preservation, or causal controllability; those require separate problem-space validation.

## Estimator configuration

The default `LGBMRegressor` configuration is fixed across classes and target features:

```python
LGBMRegressor(
    objective="regression",
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    reg_alpha=0.0,
    reg_lambda=1.0,
    subsample=1.0,
    colsample_bytree=1.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
    deterministic=True,
    force_col_wise=True,
)
```

Features are processed sequentially. LightGBM owns CPU parallelism through `n_jobs=-1`; there is no joblib feature-level pool, so nested all-core parallelism is avoided. No grid search, randomized search, Bayesian optimization, per-feature tuning, SHAP, classifier gradient, VAE, or correlation score is involved.

Installed and exercised LightGBM version: **4.6.0**.

`environment.yml` declares `lightgbm==4.6.0` in its pip dependencies, matching the installed and verified thesis environment. If the command is run in an environment without LightGBM, it fails clearly with:

```text
LightGBM is required for CFF. Install it in the repository environment with:
    python -m pip install lightgbm
```

There is no estimator fallback.

## Sampling, folds, and determinism

Defaults:

| Setting | Normal | `--fast` when not explicitly overridden |
|---|---:|---:|
| Rows per class | 20,000 | 10,000 |
| LightGBM estimators | 150 | 75 |
| Internal OOF folds | 3 | 2 |
| Random seed | 42 | 42 |
| Span-degeneracy ratio | 0.02 | 0.02 |
| Minimum absolute span | 0.001 | 0.001 |

Each class is sampled with `numpy.random.default_rng(42)` without replacement, and selected row indices are sorted before materialization. Internal folds use `KFold(n_splits=..., shuffle=True, random_state=42)`. “Validation fold” in this context means a fold drawn entirely from `X_train`; it is not the NIDS validation partition.

The default representation is model-ready scaled values. LightGBM does not require scaling, and the CFF width ratio is invariant under ordinary positive affine scaling. No representation is mixed within a run.

## Semantic exclusions and predictors

`DEFAULT_HARD_EXCLUDE` contains small, configurable semantic/context exclusions:

```text
Flow ID, Src IP, Source IP, Dst IP, Destination IP, Timestamp, Label,
Protocol, Protocol Type, Destination Port, Dst Port
```

Only `Protocol Type` exists in the canonical 39-feature model schema. It is therefore assigned `cff_score=0`, `hard_excluded=True`, and `eligible=False` by default. Destination port and identifier fields are absent from the model matrix and are not predictors.

Hard exclusion controls whether a legitimate model feature can be selected as a perturbation target. It does not remove a legitimate model input from the conditioning set. Consequently, canonical `Protocol Type` may predict another feature even though it cannot itself be selected. For tabular input, only the canonical 39 model features enter predictors; labels and extra metadata are excluded and the ignored column names are logged.

Additional exact exclusions are repeatable:

```powershell
--exclude-feature "Feature A" --exclude-feature "Feature B"
```

Absent default exclusions do not cause failure. An explicitly requested exclusion absent from the canonical schema produces a warning.

## Degenerate-feature handling

For every class-feature pair:

```text
n_unique = number of observed values
dominant_fraction = largest value count / sampled class rows
constant = n_unique <= 1
near constant = not constant and dominant_fraction >= 0.995
class span = class Q95 - class Q05
global span = pooled requested-class Q95 - pooled requested-class Q05
span degenerate = class span < 0.001 or class span < 0.02 * global span
```

Near-constant detection catches concentration on one repeated value. Span degeneracy catches a feature with multiple distinct values that are nonetheless tightly clustered within one class compared with its pooled training-class scale. Constant, near-constant, and span-degenerate targets are not fitted; they receive `cff_score=0` and `eligible=False`. `epsilon=1e-12` remains a final arithmetic safeguard, not the criterion for deciding whether a class-local denominator is meaningful.

## Ranking and selection

Eligible rows are sorted by descending `cff_score`, with canonical `feature_index` as a deterministic tie-breaker. Ineligible features have no rank. The selected size is:

```text
ceil(fraction * number_of_eligible_features)
```

for fractions 10%, 25%, and 50%. Thus the denominator contains only features that are not hard-excluded, constant, near-constant, or class-span-degenerate. No unsupported absolute threshold such as `CFF > 0.5` is imposed. The top-25% mask is the primary CFF experimental mask.

## Output files

A normal run under `outputs/cff/` writes:

```text
outputs/cff/
├── cff_scores_all.csv
├── cff_scores_<Class>.csv
├── cff_masks.json
├── feature_order.json
├── cff_run_metadata.json
├── cff_summary.md
└── masks/
    ├── <Class>_eligible.npy
    ├── <Class>_top10.npy
    ├── <Class>_top25.npy
    └── <Class>_top50.npy
```

Each score row contains:

```text
class, feature, feature_index,
cff_score, cff_raw,
residual_q05, residual_q95, residual_span,
feature_q05, feature_q95, feature_span, global_feature_span,
cv_r2, normalized_mae,
n_unique, dominant_fraction,
hard_excluded, is_constant, is_near_constant, is_span_degenerate, eligible,
suspicious_model_fit, rank,
selected_top10, selected_top25, selected_top50
```

`suspicious_model_fit=True` records `cv_r2 < -1` as a diagnostic; it does not change CFF or eligibility.

Every `.npy` mask has:

```text
shape = (39,)
dtype = bool
```

The nth element corresponds to the nth canonical model feature. `cff_masks.json` stores corresponding human-readable feature lists, and `feature_order.json` records the exact positional contract. `cff_run_metadata.json` records the timestamp, seed, input and label sources, representation, selected classes, row counts, folds, LightGBM version and parameters, degeneracy thresholds, pooled `global_feature_spans`, exclusions, feature count, parallelism policy, optional estimator check, and runtime.

`cff_summary.md` reports per-class source/sample counts, exclusion and degeneracy counts, eligible count, eligible-score diagnostics, top and bottom eligible features, and the primary top-25% feature list. Its language deliberately says “candidate perturbable features” and “conditional freedom,” not physical controllability.

Optional `--plots` writes a top-20 horizontal ranking plot per class. Optional `--estimator-check` recomputes one class with `ExtraTreesRegressor` and records Spearman rank correlation without changing the primary LightGBM ranking.

The optional estimator check was also exercised on the 200-row `BruteForce` smoke sample with five trees per estimator and two folds. It completed successfully and reported Spearman $\rho=0.592208$. This tiny robustness-check value is implementation evidence only, not a thesis result.

## Commands

### Synthetic self-test

```powershell
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.conditional_feature_freedom --self-test
```

Observed result:

```text
Self-test passed: CFF(x3)=0.0958 < CFF(x4)=1.0000; R²(x3)=0.9817 > R²(x4)=-0.1094; constant, class-span-degenerate, and hard-excluded scores are zero
```

The test uses tolerant logical assertions, not exact expected scores.

### Actual-training smoke test performed

```powershell
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.conditional_feature_freedom `
  --input data/processed `
  --classes BruteForce `
  --sample-per-class 200 `
  --n-estimators 5 `
  --cv 2 `
  --output-dir outputs/cff_smoke
```

The smoke test processed all 39 target positions for 200 actual `BruteForce` training rows. It wrote and validated four `(39,)` Boolean masks. The initial cold run reported approximately **1.9 seconds** of CFF/runtime and **3.0 seconds** end-to-end; a cached final run reported approximately **0.6 seconds** and **1.7 seconds** end-to-end. This deliberately tiny run verifies mechanics only and must not be treated as a thesis ranking result.

A second identical run produced byte-identical `cff_scores_all.csv` and top-25% mask outputs.

### Exact full run for the current CICIoT2023 training artifacts

```powershell
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.conditional_feature_freedom `
  --input data/processed `
  --sample-per-class 20000 `
  --cv 3 `
  --n-estimators 150 `
  --span-degeneracy-ratio 0.02 `
  --min-absolute-span 0.001 `
  --output-dir outputs/cff
```

Omitting `--classes` selected all seven observed non-benign categories. The corrected full run used LightGBM 4.6.0, seed 42, three OOF folds, 150 estimators, a 0.02 pooled-span ratio, and a 0.001 absolute span floor. It reported **127.84 seconds** of CFF runtime and **129.07 seconds** end-to-end while the CICIDS run was executing concurrently.

| Class | Training rows | Sampled | Span-degenerate | Eligible | Top-25% |
|---|---:|---:|---:|---:|---:|
| BruteForce | 9,146 | 9,146 | 19 | 19 | 5 |
| DDoS | 200,000 | 20,000 | 17 | 21 | 6 |
| DoS | 200,000 | 20,000 | 22 | 16 | 4 |
| Mirai | 200,000 | 20,000 | 25 | 13 | 4 |
| Recon | 200,000 | 20,000 | 17 | 21 | 6 |
| Spoofing | 200,000 | 20,000 | 19 | 19 | 5 |
| Web | 17,387 | 17,387 | 20 | 18 | 5 |

Corrected top-25% selections:

- **BruteForce:** HTTPS, rst_count, fin_count, Rate, Time_To_Live.
- **DDoS:** HTTP, Time_To_Live, Rate, ack_count, Header_Length, Max.
- **DoS:** HTTP, Time_To_Live, ack_count, rst_count.
- **Mirai:** Max, Time_To_Live, ack_count, Min.
- **Recon:** HTTPS, fin_count, Time_To_Live, Rate, Header_Length, syn_count.
- **Spoofing:** Time_To_Live, fin_count, HTTPS, psh_flag_number, Header_Length.
- **Web:** HTTPS, Time_To_Live, fin_count, syn_count, Header_Length.

The validated result contains 273 class-feature score rows and 28 Boolean masks. Every mask has shape `(39,)` and dtype `bool`; `feature_order.json` exactly equals `FEATURE_NAMES`.

The primary masks were saved as:

```text
outputs/cff/masks/BruteForce_top25.npy
outputs/cff/masks/DDoS_top25.npy
outputs/cff/masks/DoS_top25.npy
outputs/cff/masks/Mirai_top25.npy
outputs/cff/masks/Recon_top25.npy
outputs/cff/masks/Spoofing_top25.npy
outputs/cff/masks/Web_top25.npy
```

### Fast exploratory run

```powershell
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.conditional_feature_freedom `
  --input data/processed `
  --fast `
  --output-dir outputs/cff_fast
```

### Explicit class subset

```powershell
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.conditional_feature_freedom `
  --input data/processed `
  --classes DDoS DoS Recon BruteForce `
  --output-dir outputs/cff_subset
```

## Input alternatives

The repository-native NumPy artifact directory is preferred because it guarantees the actual model feature order and training-only preprocessing. A train-only Parquet or CSV is also accepted when its parent artifact directory contains the corresponding feature contract:

```powershell
python -m src.preprocessing.conditional_feature_freedom `
  --input path/to/train.parquet `
  --label-column category `
  --output-dir outputs/cff_table
```

For safety, an explicit tabular input filename must identify a train split. Its columns must match the configured feature order—39 CICIoT features from `FEATURE_NAMES` or 79 CICIDS features from `preprocessing_manifest.json`—plus the chosen label column. Extra columns are named in the log and never enter the predictor matrix. Non-finite values cause a clear failure; CFF does not invent an imputation policy. CICIDS `train.parquet` uses `--label-column category_label`.

## CICIDS2017-DistriNet support

### Training contract

For CICIDS2017-DistriNet, CFF reads only:

```text
data/processed/CICIDS_2017_Distrinet/X_train.npy
data/processed/CICIDS_2017_Distrinet/y_train_cat.npy
data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json
data/processed/CICIDS_2017_Distrinet/label_encoders.json
```

`X_train.npy` has shape `(1,456,265, 79)` and dtype `float32`. It is the exact model-ready `RobustScaler` representation produced by `scripts/preprocess_cicids2017_distrinet.py`; the scaler was fitted on all 1,456,265 training rows only. The preprocessing pipeline performs a chronological 70/15/15 split independently within each retained source attack label before fitting the scaler. CFF does not open CICIDS validation/test matrices, labels, Parquet files, timestamps, or metadata.

The 79-feature order is loaded from `preprocessing_manifest.json::modelling_feature_names`, not copied into the CFF source. `label_encoders.json::category` supplies the configured class order:

```text
Benign, DoS, DDoS, Recon, BruteForce
```

Default CFF execution therefore scores the four non-benign classes independently in this order:

```text
DoS, DDoS, Recon, BruteForce
```

The following preprocessing metadata remains outside the CFF predictor matrix: `sample_id`, `record_id`, source file/day/order/row fields, `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, numeric timestamp, original/source/category labels, attempted-attack marker, and binary label.

### Semantic exclusions

The shared hard-exclusion policy finds two fields in the CICIDS model schema:

```text
Dst Port
Protocol
```

Both receive score zero and cannot enter a selected mask, but remain legitimate conditioning predictors for other model features. `Src Port` remains eligible by default: the original CFF specification explicitly identifies destination/service port and protocol as semantic exclusions, while source port can be an ephemeral flow statistic. It can be excluded in a separate threat-model ablation with `--exclude-feature "Src Port"`; the completed primary run does not add that unrequested exclusion.

### CICIDS command

```powershell
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.conditional_feature_freedom `
  --input data/processed/CICIDS_2017_Distrinet `
  --sample-per-class 20000 `
  --cv 3 `
  --n-estimators 150 `
  --span-degeneracy-ratio 0.02 `
  --min-absolute-span 0.001 `
  --output-dir outputs/cff_cicids2017distrinet
```

The corrected-source-split run completed in **53.89 seconds** of CFF runtime and produced 316 class-feature rows and 16 masks:

| Class | Training rows | Sampled | Constant | Near-constant | Span-degenerate | Eligible | Top-25% |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoS | 120,093 | 20,000 | 11 | 0 | 27 | 52 | 13 |
| DDoS | 66,568 | 20,000 | 22 | 3 | 36 | 43 | 11 |
| Recon | 111,311 | 20,000 | 16 | 41 | 70 | 8 | 2 |
| BruteForce | 4,862 | 4,862 | 20 | 5 | 35 | 44 | 11 |

Corrected top-25% selections:

- **DoS:** Bwd IAT Min, Src Port, Fwd IAT Min, Bwd Bulk Rate Avg, FIN Flag Count, Flow IAT Min, ACK Flag Count, RST Flag Count, Subflow Fwd Bytes, Fwd Packet Length Std, Fwd Packet Length Max, Down/Up Ratio, Bwd Packet Length Std.
- **DDoS:** Bwd IAT Min, Src Port, Fwd IAT Min, Bwd Bulk Rate Avg, Flow IAT Min, FIN Flag Count, Bwd IAT Mean, Bwd IAT Total, Bwd IAT Std, Flow Bytes/s, Flow IAT Std.
- **Recon:** Src Port, Fwd Packets/s.
- **BruteForce:** Fwd IAT Min, Bwd IAT Min, Src Port, Flow IAT Min, Fwd IAT Max, Down/Up Ratio, Fwd IAT Std, Flow Duration, Fwd IAT Total, Bwd IAT Total, Bwd IAT Std.

The corrected run removes the broad flat-one artifact from locally collapsed features: span-degenerate rows are scored zero without fitting. A few eligible heavy-tailed timing features still have high normalized MAE; their class spans are comparable to pooled spans, so those are estimator/data diagnostics rather than small-denominator cases.

Full-run masks were validated as shape `(79,)`, dtype `bool`, and exact manifest order. They are dataset-specific and must not be loaded into the 39-feature CICIoT models. Primary masks are:

```text
outputs/cff_cicids2017distrinet/masks/DoS_top25.npy
outputs/cff_cicids2017distrinet/masks/DDoS_top25.npy
outputs/cff_cicids2017distrinet/masks/Recon_top25.npy
outputs/cff_cicids2017distrinet/masks/BruteForce_top25.npy
```

## Implementation map

| Function | Responsibility |
|---|---|
| `require_lightgbm` | Lazy dependency check and version discovery |
| `_directory_contract` | Dataset-specific feature order and category-ID discovery |
| `load_training_data` | Training-only NumPy/Parquet/CSV loading and contract checks |
| `resolve_feature_columns` | Exact configured tabular feature selection |
| `resolve_classes` | Exact requested classes or configured non-benign order |
| `deterministic_class_sample` | Seeded per-class sampling without replacement |
| `detect_degenerate_feature` | Constant and dominant-value near-constant diagnostics |
| `make_lgbm_regressor` | One fixed deterministic estimator configuration |
| `_oof_predictions` | Explicit internal KFold fit/predict loop |
| `compute_feature_cff` | One class-feature metric record |
| `compute_class_cff` | Sequential feature scoring and timing |
| `compute_all_classes` | Deterministic sampling, pooled global spans, and per-class execution |
| `select_top_fraction` | Eligible-only fraction selection |
| `build_feature_mask` | Exact-name canonical Boolean mask construction |
| `save_results` | CSV, JSON, NumPy mask, metadata, summary, and plot output |
| `generate_summary` | Conservative report-oriented Markdown |
| `plot_rankings` | Optional top-20 plots |
| `run_estimator_check` | Optional one-class ExtraTrees rank comparison |
| `run_synthetic_self_test` | Predictable, independent, constant, span-degenerate, and excluded checks |
| `parse_args`, `main` | Independent CLI |

## Assumptions

1. CICIoT2023 uses `src.preprocessing.schema.FEATURE_NAMES`; CICIDS2017-DistriNet uses `preprocessing_manifest.json::modelling_feature_names`. Each is the immutable order for its own saved arrays, models, and masks.
2. Each dataset directory's `X_train.npy` and `y_train_cat.npy` are aligned products of the same preprocessing run.
3. CICIoT uses its configured eight-category grouping; CICIDS uses its configured five-category grouping. Benign is excluded from default CFF target classes in both datasets.
4. The saved scaled representation is acceptable because it is the exact model input representation and the normalized width ratio is invariant to positive affine transforms.
5. `Protocol Type` for CICIoT and `Protocol`/`Dst Port` for CICIDS are semantic perturbation-target exclusions but remain legitimate conditioning predictors because they are visible to their models.
6. Pooled global spans use the same deterministic per-class samples and only the requested training classes; changing `--classes` intentionally changes this reference population.
7. Top fractions use `ceil`, ensuring a non-empty selection whenever at least one eligible feature exists.
8. CFF ranks statistical candidates only; validity constraints, threat-model permissions, and packet realization remain separate gates.

## Explicit implementation audit

| Risk | Audit result |
|---|---|
| Train/test leakage | Repository mode opens only dataset-local `X_train.npy`, `y_train_cat.npy`, and contract metadata. No test artifact is read. |
| Validation/test contamination | OOF folds are drawn only from sampled training rows. No NIDS validation artifact is read. |
| Label leakage | Labels select one class before regression and never enter `X_-i`. Predictors are configured model feature columns only. |
| Identifier inclusion | Identifier/context fields are absent from both model matrices. Tabular extras are excluded and logged. |
| Feature-order mismatch | CICIoT imports `FEATURE_NAMES`; CICIDS loads `modelling_feature_names` from its manifest. Masks assert length, Boolean dtype, and unique exact mapping. |
| Nested parallelism | Feature targets are sequential; only the active LightGBM fit uses `n_jobs=-1`. |
| Nondeterministic seeds | Sampling, KFold, LightGBM, and optional ExtraTrees use seed 42. LightGBM deterministic/column-wise flags are enabled. |
| Constant divide-by-zero | Constant targets skip fitting; all normalized denominators retain positive `epsilon`. |
| Near-constant handling | `dominant_fraction >= 0.995` skips fitting and sets score/eligibility to zero/false. |
| Class-span degeneracy | Every saved flag exactly matches `feature_span < 0.001 or feature_span < 0.02 * global_feature_span`; flagged rows have no residual/R²/MAE because no estimator was fitted. |
| Global reference leakage | Pooled spans come only from deterministic samples of requested training classes; no validation/test rows participate. |
| Ranking denominator | Only `eligible=True` rows determine ranks and top 10/25/50 sizes; span-degenerate rows are excluded. |
| LightGBM warnings/errors | Corrected synthetic, smoke, CICIoT full, and CICIDS full runs completed without LightGBM warnings/errors. |
| Mask dimensionality | CICIoT masks validate as `(39,)`; CICIDS masks validate as `(79,)`; all use `bool`. |
| Dataset separation | Each output stores `feature_order.json`; CICIDS and CICIoT masks are not dimensionally interchangeable. |
| Reproducibility | Seeded sampling, folds, estimators, and deterministic LightGBM settings are recorded in each run metadata file. |

## Scope boundaries

CFF does not modify the attack pipeline, assign Full versus Partial perturbation tiers, prove physical mutability, derive packet-level actions, fit on validation/test data, tune estimators per feature, or promote CFF selections into the canonical model schema. The generated masks are ablation inputs for a later explicitly configured experiment.
