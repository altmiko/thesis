# CICIoT2023 Preprocessing Report

> **Current corrected run.** This report describes the presence-preserving
> preprocessing under `outputs/ciciot2023_fixed/`. The original
> `outputs/ciciot2023/` bundle remains available as the before side of the
> audit. End-to-end classifier effects are recorded in
> `outputs/ciciot2023_fixed/fixes_report.md`.

## 1. Purpose

This document records the preprocessing run that produced the corrected model-ready CICIoT2023 artifacts in `outputs/ciciot2023_fixed/`. It describes the source build, leakage-safe split, presence-preserving cleaning, train-only scaling and downsampling, saved array contract, and full-holdout verification.

Run identity:

| Field | Value |
|---|---|
| Dataset | CICIoT2023 |
| Schema | Modified Schema A, 39 ordered features |
| Seed | 42 |
| Run timestamp | `2026-09-18T12:52:53.449176+00:00` |
| Git commit | `9bb60e3e99f5feafd349a15b138d9f8f4dcb1349` |
| Runtime | 175.4 seconds |
| Output root | `outputs/ciciot2023_fixed/` |
| Run manifest | `outputs/ciciot2023_fixed/run_manifest.json` |

## 2. Holdout decision

Validation and test are retained **in full**. They are not sampled, balanced, clustered, capped, or otherwise resampled.

This is intentional. The user clarified that “full” means keeping every validation and test row. It also preserves the repository’s evaluation contract: training may be balanced, but validation and test must retain their natural temporal prevalence so reported metrics remain representative of the held-out traffic.

Consequences:

- `X_val.npy` contains all 5,555,150 validation rows.
- `X_test.npy` contains all 8,248,312 test rows.
- Corresponding fine-label, category-label, and binary-label arrays have exactly the same lengths.
- No auxiliary sampled validation/test arrays were created.
- Only the training partition is downsampled.

## 3. Canonical inputs

The run consumes:

| Input | Role |
|---|---|
| `data/raw/CICIoT2023_CSV_DOWNLOADED/` | Official 34-folder, 309-shard CSV distribution |
| `data/processed/ciciot2023_labeled_full.parquet` | Labelled 43-column source used by the pipeline |
| `data/processed/ciciot2023_labeled_full_manifest.json` | Source-build accounting and provenance |
| `src/preprocessing/schema.py` | Frozen feature order, feature typing, and 34-to-8 category map |
| `src/preprocessing/ciciot2023/splitter.py` | Forward-chaining and contiguous-block split logic |
| `src/preprocessing/ciciot2023/sampler.py` | Cluster-proportional-floor training sampler |
| `src/preprocessing/ciciot2023/pipeline.py` | End-to-end preprocessing implementation |

The canonical labelled Parquet remains under `data/processed/` because it is an input. Corrected model-ready artifacts are under `outputs/ciciot2023_fixed/`.

## 4. Source build accounting

The labelled source was constructed from 309 raw CSV shards across 34 class folders.

| Measure | Count |
|---|---:|
| Raw rows read | 46,776,700 |
| Rows retained | 46,775,660 |
| Rows dropped for NaN or ±infinity | 1,040 |
| Drop rate | 0.0022% |
| Fine labels | 34 |
| Coarse categories | 8 |
| Features | 39 |
| Labelled Parquet columns | 43 |

The Parquet layout is:

```text
39 ordered features
+ Label
+ category
+ source_csv_filename
+ source_folder
= 43 columns
```

At the source-build stage, the only cleaning is replacement of positive/negative infinity with NaN followed by row removal. No percentile clipping, rounding, scaling, or sampling occurs before the temporal split.

### 4.1 Full-source category distribution

| Category | Rows | Share |
|---|---:|---:|
| DDoS | 33,983,922 | 72.6530% |
| DoS | 7,844,894 | 16.7713% |
| Mirai | 2,633,870 | 5.6309% |
| Benign | 1,098,126 | 2.3476% |
| Recon | 690,521 | 1.4762% |
| Spoofing | 486,435 | 1.0399% |
| Web | 24,828 | 0.0531% |
| BruteForce | 13,064 | 0.0279% |
| **Total** | **46,775,660** | **100%** |

The source is dominated by DDoS and DoS traffic. This imbalance motivates training-only downsampling while making full validation/test retention essential for honest evaluation.

## 5. Frozen feature schema

`src/preprocessing/schema.py::FEATURE_NAMES` is the sole authoritative feature order:

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

Reordering these features would invalidate saved arrays, the scaler, trained models, attacks, and downstream interpretation.

Feature roles used by cleaning:

- 22 presence-valued fields: the 15 protocol/service indicators plus the seven
  `*_flag_number` fields;
- five integer count fields: `ack_count`, `syn_count`, `fin_count`, `rst_count`,
  and `Number`;
- `Protocol Type` is retained as its numeric protocol value;
- all remaining fields are continuous.

## 6. Leakage-safe temporal split

The split is computed before any percentile, scaler, clustering, sampling, or class-weight statistic is fitted.

Default fractions:

| Partition | Target fraction |
|---|---:|
| Train | 70% |
| Validation | 10% |
| Test | 20% |

The exact counts differ from simple global percentages because splitting is performed per fine class using complete temporal shards where possible.

### 6.1 Split protocols

| Protocol | Fine classes | Procedure |
|---|---:|---|
| `forward_chain` | 17 | Natural-sort shards; earliest train, later validation, latest test |
| `two_shard_hybrid` | 2 | Entire later shard to test; earlier shard split contiguously into train/validation |
| `block` | 15 | One shard divided into ordered contiguous train/validation/test blocks |

The splitter asserts shard disjointness for forward-chain classes and ordered, non-overlapping ranges for block/hybrid classes.

### 6.2 Natural split counts before training sampling

| Split | Rows | Sampling status |
|---|---:|---|
| Train | 32,972,198 | Eligible for train-only sampling |
| Validation | 5,555,150 | Retained in full |
| Test | 8,248,312 | Retained in full |
| **Total** | **46,775,660** | — |

The counts reconcile exactly with the retained labelled source.

## 7. Train-derived cleaning

Cleaning occurs only after split assignment.

For every feature, the pipeline computes the 99.99th percentile from the
natural training partition only. Continuous and integer-count features use that
bound directly. Presence-valued protocol/service/flag features floor the
effective upper bound at `1`, preventing rare positive values from being
clipped below their canonical positive state. Validation and test influence
none of the bounds.

### 7.1 Effective train-derived upper clipping bounds

| Feature | Upper bound | Feature | Upper bound |
|---|---:|---|---:|
| Header_Length | 60 | Protocol Type | 47 |
| Time_To_Live | 248 | Rate | 626,015.5 |
| fin_flag_number | 1 | syn_flag_number | 1 |
| rst_flag_number | 1 | psh_flag_number | 1 |
| ack_flag_number | 1 | ece_flag_number | 1 |
| cwr_flag_number | 1 | ack_count | 100 |
| syn_count | 100 | fin_count | 100 |
| rst_count | 100 | HTTP | 1 |
| HTTPS | 1 | DNS | 1 |
| Telnet | 1 | SMTP | 1 |
| SSH | 1 | IRC | 1 |
| TCP | 1 | UDP | 1 |
| DHCP | 1 | ARP | 1 |
| ICMP | 1 | IGMP | 1 |
| IPv | 1 | LLC | 1 |
| Tot sum | 132,694.78125 | Min | 1,514 |
| Max | 13,098 | AVG | 3,117.35596 |
| Std | 3,793.11523 | Tot size | 3,117.35596 |
| IAT | 0.08072246 | Number | 100 |
| Variance | 14,387,724 | — | — |

After clipping:

- integer count fields are rounded and constrained to non-negative values;
- any positive protocol/service/flag presence value maps to `1`, otherwise `0`;
- continuous fields retain floating-point values;
- `Protocol Type` is clipped but not canonicalized by this routine.

### 7.2 Why presence canonicalization changed

The vendor CSV contains fractional positive values in nominal indicator and flag
columns. Ordinary rounding, combined with sub-one percentile bounds, previously
collapsed nine nonconstant source features to zero:
`ece_flag_number`, `cwr_flag_number`, `Telnet`, `SMTP`, `SSH`, `IRC`, `DHCP`,
`ARP`, and `IGMP`.

The corrected rule preserves occurrence information. For example, raw `ARP` is
positive in 3,297,486 natural training rows and 775,049 test rows; the corrected
validation/test arrays retain every positive occurrence as `1`.

Corrected EDA checks found zero non-finite, invalid binary, or invalid integer
training values. Constant training features fell from 9 to 0, and undefined
Spearman cells fell from 621 to 0.

## 8. Train-only scaling

The pipeline fits one `sklearn.preprocessing.RobustScaler` on all 32,972,198 cleaned natural training rows:

```python
scaler.fit(X[train_mask])
```

That fitted scaler transforms the sampled training rows and every full validation/test row. The scaler is not refitted on sampled training data, a category subset, validation, or test.

Persisted scaler hash:

```text
dbd8dc3680cdf2bfbcb9194b1461c99167766bb45fdeb27a462299cbd7cc46a0
```

## 9. Train-only downsampling

Only the training partition is sampled. The method is intra-category MiniBatchKMeans followed by cluster-proportional allocation with a floor and seeded uniform selection of real rows within each cluster.

Parameters:

| Setting | Value |
|---|---|
| Category cap | 200,000 rows |
| Per-cluster floor | 500 rows |
| Selection mode | `random_within` |
| Seed | 42 |
| Cluster features | 23 non-binary, non-protocol features |
| Rare categories kept whole | BruteForce, Web |

### 9.1 Training category results

| Category | Natural train | Saved train | Policy | Clusters |
|---|---:|---:|---|---:|
| Benign | 657,907 | 200,000 | Cluster-proportional-floor | 10 |
| BruteForce | 9,146 | 9,146 | Kept whole | — |
| DDoS | 24,271,179 | 200,000 | Cluster-proportional-floor | 20 |
| DoS | 5,286,875 | 200,000 | Cluster-proportional-floor | 20 |
| Mirai | 1,904,377 | 200,000 | Cluster-proportional-floor | 10 |
| Recon | 483,369 | 200,000 | Cluster-proportional-floor | 15 |
| Spoofing | 341,958 | 200,000 | Cluster-proportional-floor | 15 |
| Web | 17,387 | 17,387 | Kept whole | — |
| **Total** | **32,972,198** | **1,226,533** | — | — |

No synthetic centroid or generated row is saved. Every retained training observation maps to a real global Parquet row index stored in `train_kept_indices.npy`.

## 10. Labels and class weights

Three target views are emitted for every split:

| Target | File pattern | Meaning |
|---|---|---|
| Fine | `y_<split>.npy` | 34-class encoded CICIoT label |
| Category | `y_<split>_cat.npy` | 8-class category label |
| Binary | `y_<split>_bin.npy` | Benign = 0, attack = 1 |

The label encoders and their ordered class-name lists are persisted. Class weights are computed from the **sampled training labels**, because those are the observations presented to model fitting.

Observed class-weight ranges:

| Target | Minimum | Maximum |
|---|---:|---:|
| 34-class | 0.18037 | 152.21307 |
| 8-class | 0.76658 | 16.76324 |
| Binary | 0.59742 | 3.06633 |

## 11. Generated artifacts

All corrected preprocessing artifacts are under `outputs/ciciot2023_fixed/`.

### 11.1 Model arrays

| Artifact | Shape | Type | Meaning |
|---|---|---|---|
| `X_train.npy` | 1,226,533 × 39 | float32 | Cleaned, scaled, sampled training rows |
| `X_val.npy` | 5,555,150 × 39 | float32 | Cleaned, scaled, full validation rows |
| `X_test.npy` | 8,248,312 × 39 | float32 | Cleaned, scaled, full test rows |
| `y_train.npy` | 1,226,533 | int32 | Sampled 34-class training labels |
| `y_val.npy` | 5,555,150 | int32 | Full 34-class validation labels |
| `y_test.npy` | 8,248,312 | int32 | Full 34-class test labels |
| `y_*_cat.npy` | Split length | int32 | Eight-category labels |
| `y_*_bin.npy` | Split length | int32 | Binary labels |

### 11.2 Supporting artifacts

- `train_kept_indices.npy`
- `class_weights_34.npy`
- `class_weights_8.npy`
- `class_weights_2.npy`
- `scaler.pkl`
- `label_encoder.pkl`
- `category_encoder.pkl`
- `class_names.json`
- `category_names.json`
- `class_to_category.json`
- `run_manifest.json`
- `ciciot2023_labeled_full_manifest.json`
- `preprocessing_verification.txt`
- `bundle_manifest.json`

The output bundle also contains the EDA report and EDA artifacts, but those are downstream analyses rather than preprocessing transformations.

### 11.3 Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `X_train.npy` | `3c44a81a882ca2d6a78a7ad030d2f8a6f376fdf667e79cd89f84c3b05d801bc7` |
| `X_val.npy` | `60fcd73f501852b8e6d24fde5971ddba8ba28e41bce985d0798d3b8697fd194f` |
| `X_test.npy` | `f646ab0b0b7a675131fa87a253ce1079b490cfd22b307d513fb6237fc71782d3` |
| Scaler center + scale | `dbd8dc3680cdf2bfbcb9194b1461c99167766bb45fdeb27a462299cbd7cc46a0` |

The changed matrix hashes distinguish the corrected presence-preserving arrays
from the original preprocessing run. The scaler hash is unchanged because
`RobustScaler` medians and IQRs for these sparse columns remain the same.

## 12. Verification results

The redirected output bundle passed the preprocessing verifier:

```text
LEAKAGE GUARDS PASSED:
  [OK] no shard appears in two splits
  [OK] every class present in train/validation/test
  [OK] no test row precedes a training row within a class
  [OK] validation/test untouched by sampler
  [OK] sampled training total matches manifest and arrays
```

Verified counts:

```text
train: 32,972,198 natural -> 1,226,533 sampled
validation: 5,555,150 full rows
test: 8,248,312 full rows
```

Additional corrected-run checks passed:

- manifest feature order matches the frozen schema;
- every fine class and category occurs in all splits;
- no non-finite training values;
- no invalid binary or integer training values;
- all 22 presence-valued columns remain exactly in `{0, 1}`;
- all audited validation/test positive-occurrence counts match the raw source;
- constant processed training features: 0;
- preprocessing regression suite: 17 passed.

## 13. Reproduction commands

Run preprocessing into the isolated output directory:

```bash
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe \
  -m src.preprocessing.ciciot2023.pipeline \
  --output-dir outputs/ciciot2023_fixed
```

Verify the redirected arrays and manifest:

```bash
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe \
  -m src.preprocessing.ciciot2023.reports verify \
  --processed-dir outputs/ciciot2023_fixed
```

Regenerate EDA against the redirected preprocessing artifacts:

```bash
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe \
  -m src.evaluation.ciciot2023_eda \
  --processed-dir outputs/ciciot2023_fixed \
  --output-dir outputs/ciciot2023_fixed/eda
```

## 14. Interpretation requirements

- Treat `X_train.npy` as the sampled training distribution, not natural prevalence.
- Treat `X_val.npy` and `X_test.npy` as full temporal holdouts.
- Do not report training category shares as source-dataset prevalence.
- Do not fit scalers, clipping thresholds, imputers, feature policies, or clustering parameters on validation/test.
- Do not replace the full holdouts with balanced samples for headline metrics.
- If a later experiment needs a small evaluation subset, select it explicitly at experiment time, preserve the full arrays, store selected indices, and label the resulting metric as sampled evaluation.

## 15. Related documentation

- Full build and implementation guide: `docs/data/ciciot2023_full_build.md`
- EDA report: `docs/data/ciciot2023_eda.md`
- Downsampling implementation log: `docs/data/ciciot2023_downsampling_implementation.md`
- Raw labelling and provenance: `docs/data/ciciot2023_building_from_download.md`
- General preprocessing handoff: `docs/preprocessing_handoff.md`
- Corrected end-to-end fixes report: `outputs/ciciot2023_fixed/fixes_report.md`
