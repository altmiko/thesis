# Two-Dataset Training-Split Feature Audit

## Scope

This document consolidates the feature audits for:

1. **CICIoT2023** — Modified Schema A, 39 features, eight-category scope.
2. **CICIDS2017-DistriNet** — 77-feature CICFlowMeter audit scope, restricted to Benign, DoS, DDoS, Recon, and BruteForce.

Only training data was used. Validation and test arrays were not loaded or used. No feature was removed, transformed, or otherwise modified by the audit.

The detailed reports and machine-readable results are stored at:

- `outputs/ciciot2023/feature_audit/feature_audit_report.md`
- `outputs/cicids2017distrinet/feature_audit/feature_audit_report.md`

Each output directory contains:

- `audit_manifest.json`
- `degeneracy.csv`
- `missing_invalid_values.csv`
- `spearman_correlation.csv`
- `spearman_correlation_heatmap.png`
- `high_correlation_pairs.csv`
- `structural_relationship_checks.csv`
- `univariate_predictive_power.csv`
- `permutation_importance.csv`
- `recommendations.csv`
- `feature_audit_report.md`

## Spearman heatmaps and EDA codebase inventory

Current audit heatmaps:

- CICIoT2023: `outputs/ciciot2023/feature_audit/spearman_correlation_heatmap.png`
- CICIDS2017-DistriNet: `outputs/cicids2017distrinet/feature_audit/spearman_correlation_heatmap.png`

The heatmaps use the signed `[-1, 1]` Spearman scale and are rendered directly from each audit's `spearman_correlation.csv`. Gray cells represent undefined correlations caused by constant features.

Current CICIoT2023 EDA implementations:

- `src/evaluation/ciciot2023_eda.py` — leakage-safe sampled-training EDA, including distributions, PCA, and a Spearman heatmap.
- `src/preprocessing/ciciot2023/feature_analysis.py` — semantics-corrected training-only Pearson/Spearman, variability, mutual information, dependency groups, and optional classifier-ablation analysis.

Current CICIoT2023 EDA outputs:

- Semantics-corrected analysis: `outputs/ciciot2023_semantics_corrected/eda/`
- Current feature audit: `outputs/ciciot2023/feature_audit/`
- Legacy pre-fix EDA: `outputs/ciciot2023/eda/`

CICIDS2017-DistriNet does not have a dedicated executable EDA module equivalent to `ciciot2023_eda.py`. Its current analysis is split across:

- preprocessing and source-data audit code: `scripts/preprocess_cicids2017_distrinet.py`;
- full EDA/preprocessing write-up: `docs/cicids2017_preprocessing.md`;
- processed manifests and leakage/duplicate audits: `data/processed/CICIDS_2017_Distrinet/`;
- current feature audit, matrix, and heatmap: `outputs/cicids2017distrinet/feature_audit/`.

Historical CICIoT2023 EDA scripts also exist at `src/evaluation/eda_tables.py`, `src/evaluation/eda_figures_part1.py`, `src/evaluation/eda_figures_part2.py`, and `old_root_files/generate_eda_figures.py`. They retain historical paths or pre-correction semantics and are not the canonical implementations for the current audit.

## Methodology

### Degeneracy

For each audited feature, the audit calculated:

- variance in raw feature units;
- unique-value count;
- modal value and modal percentage.

A feature was flagged as near-zero variance when a single value accounted for **more than 99.9%** of the training observations.

### Missing and invalid values

The processed training matrices were checked feature by feature for:

- NaN;
- positive infinity;
- negative infinity.

For CICIDS2017-DistriNet, raw source rows in the current category-specific training time windows were also inspected before the existing finite/negative-row cleaner and deduplication. Negative values were counted per feature rather than being silently represented as valid values.

### Redundancy

Spearman matrices were calculated on deterministic, category-stratified training samples:

- CICIoT2023: 326,533 rows;
- CICIDS2017-DistriNet: 311,431 rows.

Every pair with `|rho| > 0.95` was reported and classified as one of:

- exact structural relationship;
- known formula or constraint family;
- coincidental high correlation with no known formula link.

Formula and constraint checks were evaluated separately so that strong rank correlation was not automatically treated as proof of an exact identity.

### Leakage and generation-artifact screen

For every feature and both prediction heads, a class-balanced depth-1 decision tree was fitted and scored on a deterministic training-only sample. The reported score is:

- binary ROC AUC for the binary head;
- macro one-vs-rest ROC AUC for the category head.

A feature was flagged for leakage/artifact review when either:

1. its AUC exceeded `Q3 + 3 × IQR` within its head and was at least 0.75; or
2. it was the rank-1 feature, had AUC at least 0.85, and exceeded the rank-2 feature by at least 0.05 AUC.

### Existing-checkpoint permutation importance

Permutation importance used the validation-selected `SerialCNNLSTM` checkpoint for each head. Importance is the decrease in balanced accuracy after independently permuting a feature, averaged over three deterministic repeats on 50,000 training rows.

The bottom 10% comprises:

- four features per CICIoT2023 head;
- eight features per CICIDS2017-DistriNet head.

### Recommendation rules

The **classifier-candidate-cut list** is the union of features that are:

- near-zero variance;
- selected as the lower-univariate-value member of a coincidental `|rho| > 0.95` pair; or
- in the bottom permutation-importance decile for either head.

The **VAE-must-keep list** contains every feature involved in a formula, identity, logical relationship, or Min/Mean/Max/Std constraint family, plus every feature flagged for leakage/artifact review. This list overrides classifier cut signals. A structurally required feature that also has weak classifier evidence is labelled:

> keep for structural modeling, low value for classification

## CICIoT2023 audit

### Provenance

- Current audit bundle: `outputs/ciciot2023_semantics_corrected`
- Training rows: 1,226,533
- Features: 39
- Scope: eight categories
- Preprocessing: semantics-corrected, no clipping
- Spearman sample: 326,533 training rows
- Univariate sample: 176,533 training rows
- Permutation sample: 50,000 training rows

### Degeneracy findings

Two features exceed the strict near-zero-variance threshold:

| Feature | Raw-unit variance | Unique values | Modal percentage |
|---|---:|---:|---:|
| Telnet | 3.96799e-06 | 7 | 99.9465% |
| SMTP | 4.50735e-06 | 9 | 99.9499% |

### Clip/round-bug recheck

All nine previously affected features now have nonzero variance and more than one observed value in the semantics-corrected saved training split.

| Feature | Raw-unit variance | Unique values | Modal percentage | Still above 99.9% |
|---|---:|---:|---:|---|
| ece_flag_number | 5.92945e-05 | 20 | 99.7966% | No |
| cwr_flag_number | 2.88990e-05 | 18 | 99.8717% | No |
| Telnet | 3.96799e-06 | 7 | 99.9465% | Yes |
| SMTP | 4.50735e-06 | 9 | 99.9499% | Yes |
| SSH | 0.00102770 | 17 | 98.8918% | No |
| IRC | 8.85178e-06 | 6 | 99.8789% | No |
| DHCP | 0.000251573 | 41 | 98.7417% | No |
| ARP | 0.00320669 | 177 | 82.9639% | No |
| IGMP | 1.84538e-05 | 9 | 99.7961% | No |

The clip/round fix restored observable variation in all nine features. It did not make `Telnet` or `SMTP` useful high-frequency signals: both remain effectively near-constant under the audit threshold.

### Missing and invalid values

The processed CICIoT2023 training matrix contains:

| Invalid value | Count |
|---|---:|
| NaN | 0 |
| Positive infinity | 0 |
| Negative infinity | 0 |
| Total infinity | 0 |

### Redundancy findings

Ten pairs have `|rho| > 0.95`. All ten have a known formula, identity, logical, or structural-family link.

| Feature A | Feature B | Spearman rho | Classification |
|---|---|---:|---|
| AVG | Tot size | 1.000000 | Exact structural relationship |
| ARP | IPv | -1.000000 | Exact structural relationship |
| IPv | LLC | 1.000000 | Exact structural relationship |
| ARP | LLC | -1.000000 | Exact structural relationship |
| Std | Variance | 1.000000 | Exact structural relationship |
| rst_flag_number | rst_count | 0.999751 | Exact structural relationship |
| fin_flag_number | fin_count | 0.999479 | Exact structural relationship |
| syn_flag_number | syn_count | 0.998093 | Exact structural relationship |
| Rate | IAT | -0.995758 | Known formula family; not exact locally |
| ack_flag_number | ack_count | 0.980904 | Exact structural relationship |

Verified exact relationships include:

- `AVG = Tot size`;
- `Variance = Std²`;
- `Tot sum = Number × AVG`;
- `fin_count = fin_flag_number × Number`;
- `syn_count = syn_flag_number × Number`;
- `rst_count = rst_flag_number × Number`;
- `ack_count = ack_flag_number × Number`;
- `LLC = IPv`;
- `ARP + IPv = 1`;
- `ARP + LLC = 1`;
- `Min <= AVG <= Max`.

`Rate` and `IAT` have a near-perfect inverse rank relationship, but `Rate × IAT = 1` holds within the audit tolerance for only 18.9962% of the sampled rows. The pair is structurally related but must not be described as an exact identity for this local release.

### Leakage and artifact findings

No CICIoT2023 feature met the suspicious univariate-outlier rule.

Top binary-head features:

| Rank | Feature | Stump AUC |
|---:|---|---:|
| 1 | HTTPS | 0.782907 |
| 2 | Header_Length | 0.765228 |
| 3 | ack_flag_number | 0.758561 |
| 4 | ack_count | 0.752464 |
| 5 | Number | 0.747385 |

Top category-head features:

| Rank | Feature | Stump AUC |
|---:|---|---:|
| 1 | Number | 0.775796 |
| 2 | ack_flag_number | 0.739706 |
| 3 | IAT | 0.725228 |
| 4 | Rate | 0.721902 |
| 5 | ack_count | 0.711679 |

### Permutation-importance limitation

The only available CICIoT2023 classifier checkpoints were trained before the semantics correction. Using those checkpoints directly on corrected inputs is not a valid importance experiment:

| Head | Matching legacy training baseline | Corrected-input compatibility probe |
|---|---:|---:|
| Binary | 0.8854–0.8915 balanced accuracy | 0.6103 |
| Category | 0.6791–0.6842 balanced accuracy | 0.4351 |

Permutation importance was therefore computed on the matching legacy pre-fix training bundle. These values describe the existing trained models but are **historical evidence only** and must not be interpreted as importance for a future classifier retrained on semantics-corrected inputs.

Bottom-decile features were:

| Head | Feature | Mean balanced-accuracy decrease |
|---|---|---:|
| Binary | IPv | 0 |
| Binary | LLC | 0 |
| Binary | SMTP | 0 |
| Binary | Telnet | 7.97e-06 |
| Category | ICMP | -0.000797 |
| Category | IRC | -0.000169 |
| Category | SMTP | approximately 0 |
| Category | IPv | 0 |

Negative importance means the permuted sample scored slightly better than the unpermuted sample; it is evidence of no useful importance at this resolution, not evidence that randomization is beneficial.

### CICIoT2023 classifier-candidate-cut list

| Feature | Reason |
|---|---|
| Telnet | Near-zero variance; bottom-decile binary importance |
| SMTP | Near-zero variance; bottom-decile importance for both heads |
| IRC | Bottom-decile category importance |
| ICMP | Bottom-decile category importance |

This is a review list only. The permutation evidence is historical because corrected CICIoT2023 checkpoints do not yet exist.

### CICIoT2023 VAE-must-keep list

The following 21 features participate in structural formulas, identities, logical constraints, or distribution-statistic families and are excluded from the classifier cut list:

- `Rate`
- `fin_flag_number`
- `syn_flag_number`
- `rst_flag_number`
- `ack_flag_number`
- `ack_count`
- `syn_count`
- `fin_count`
- `rst_count`
- `ARP`
- `IPv`
- `LLC`
- `Tot sum`
- `Min`
- `Max`
- `AVG`
- `Std`
- `Tot size`
- `IAT`
- `Number`
- `Variance`

`IPv` and `LLC` are explicitly labelled **keep for structural modeling, low value for classification** because they are structurally required despite bottom-decile legacy-model importance.

## CICIDS2017-DistriNet audit

### Provenance and 77-feature scope

- Processed training rows: 1,456,264
- Checkpoint input width: 79
- Audited feature count: 77
- Categories: Benign, DoS, DDoS, Recon, BruteForce
- Spearman sample: 311,431 training rows
- Univariate sample: 164,863 training rows
- Permutation sample: 50,000 training rows

The processed bundle and trained checkpoints use 79 inputs. To implement the requested 77-feature CICFlowMeter scope, the audit excludes:

- `Src Port`;
- `Protocol`.

`Dst Port` remains in scope because Destination Port is the historical CICIDS2017 artifact under review. During checkpoint permutation, `Src Port` and `Protocol` remain fixed while each of the 77 audited features is permuted independently.

### Degeneracy findings

| Feature | Raw-unit variance | Unique values | Modal percentage |
|---|---:|---:|---:|
| Fwd URG Flags | 0 | 1 | 100% |
| Bwd URG Flags | 0 | 1 | 100% |
| URG Flag Count | 0 | 1 | 100% |
| CWR Flag Count | 0.000537498 | 5 | 99.9622% |
| ECE Flag Count | 0.00126582 | 3 | 99.9648% |
| Subflow Bwd Packets | 2.33469e-05 | 2 | 99.9977% |

The first three features are exactly constant. The remaining three cross the strict >99.9% modal threshold.

### Missing, infinite, and negative values

The processed training matrix contains no NaN or infinite values. That result reflects the existing cleaner, which removes an entire row when any numeric field is non-finite or negative.

To expose the underlying source artifacts, the audit examined 1,469,083 supported raw rows in the current category-specific training time windows before finite/negative cleaning and deduplication. A total of 2,328 rows contained at least one audited NaN, infinity, or negative value.

| Feature | NaN | Infinite | Negative |
|---|---:|---:|---:|
| Flow Bytes/s | 131 | 133 | 0 |
| Flow Packets/s | 0 | 264 | 0 |
| Flow IAT Mean | 163 | 0 | 0 |
| Flow IAT Std | 163 | 0 | 0 |
| Flow IAT Max | 163 | 0 | 0 |
| Flow IAT Min | 163 | 0 | 2,064 |

The 2,064 negative `Flow IAT Min` observations are consistent with the known CICFlowMeter clock-artifact failure mode. They are reported explicitly rather than interpreted as valid IATs or silently converted to zero.

### Redundancy findings

Ninety-one pairs have `|rho| > 0.95`. The complete classified list is in:

- `outputs/cicids2017distrinet/feature_audit/high_correlation_pairs.csv`

Examples of exact structural relationships include:

- `Packet Length Mean = Average Packet Size`;
- `Fwd Segment Size Avg = Fwd Packet Length Mean`;
- `Bwd Segment Size Avg = Bwd Packet Length Mean`;
- `Fwd IAT Total = Fwd IAT Mean × (Total Fwd Packet − 1)`;
- `Bwd IAT Total = Bwd IAT Mean × (Total Bwd packets − 1)`.

Known formula or constraint families include:

- flow/forward/backward packet rates, duration, and packet counts;
- forward and backward bulk byte/packet/rate averages;
- directional PSH flags and aggregate PSH count;
- forward, backward, and combined packet-length Min/Mean/Max/Std families;
- Flow, Fwd, and Bwd IAT Min/Mean/Max/Std/Total families;
- Active and Idle Min/Mean/Max/Std families.

Not every textbook CICFlowMeter relationship is exact in this local release:

| Expected relationship | Satisfaction fraction |
|---|---:|
| Flow Bytes/s from byte totals and duration | 70.5794% |
| Flow Packets/s from packet totals and duration | 79.3531% |
| Fwd Packets/s from count and duration | 79.3514% |
| Bwd Packets/s from count and duration | 79.3495% |
| Packet Length Variance = Packet Length Std² | 72.4051% |
| Subflow Fwd Packets = Total Fwd Packet | 0.0010% |
| Subflow Fwd Bytes = Total Length of Fwd Packet | 26.8933% |
| Subflow Bwd Packets = Total Bwd packets | 0.3250% |
| Subflow Bwd Bytes = Total Length of Bwd Packet | 28.1754% |

These discrepancies are preserved as audit findings. The involved fields remain structurally relevant, but the validator or VAE must not enforce these expected formulas as exact equalities without a dataset-specific mined rule.

### Leakage and generation-artifact findings

`RST Flag Count` is a suspicious binary-head outlier:

| Feature | Head | Stump AUC | Rank | Gap to rank 2 |
|---|---|---:|---:|---:|
| RST Flag Count | Binary | 0.947057 | 1 | 0.118498 |

It is therefore flagged for leakage/generation-artifact review and placed on the VAE-must-keep list rather than the classifier cut list.

Top category-head features were:

| Rank | Feature | Stump AUC |
|---:|---|---:|
| 1 | Fwd Packet Length Mean | 0.808187 |
| 2 | Fwd Segment Size Avg | 0.808187 |
| 3 | Subflow Fwd Bytes | 0.807743 |
| 4 | Total Length of Fwd Packet | 0.806761 |
| 5 | Bwd Packet Length Std | 0.805925 |

### Destination Port result

The historical original-CICIDS2017 Destination Port artifact does **not** persist under the audit rule in the DistriNet-corrected training scope.

| Head | Dst Port stump AUC | Rank | Suspicious outlier |
|---|---:|---:|---|
| Binary | 0.781361 | 27 | No |
| Category | 0.615050 | 56 | No |

`Dst Port` remains available for review, but it is neither a classifier cut candidate nor a leakage outlier in this audit.

### Permutation importance

The DistriNet checkpoints match the current processed 79-input training bundle. Baseline balanced accuracy on the permutation sample was:

| Head | Baseline balanced accuracy |
|---|---:|
| Binary | 0.999446 |
| Category | 0.998244 |

Bottom-decile features were:

| Head | Feature | Mean balanced-accuracy decrease |
|---|---|---:|
| Binary | Bwd URG Flags | 0 |
| Binary | CWR Flag Count | 0 |
| Binary | ECE Flag Count | 0 |
| Binary | Fwd URG Flags | 0 |
| Binary | SYN Flag Count | 0 |
| Binary | Subflow Bwd Packets | 0 |
| Binary | Subflow Fwd Packets | 0 |
| Binary | URG Flag Count | 0 |
| Category | Bwd URG Flags | 0 |
| Category | CWR Flag Count | 0 |
| Category | ECE Flag Count | 0 |
| Category | Fwd URG Flags | 0 |
| Category | Subflow Bwd Packets | 0 |
| Category | Subflow Fwd Packets | 0 |
| Category | URG Flag Count | 0 |
| Category | Bwd Header Length | 3.37e-06 |

### CICIDS2017-DistriNet classifier-candidate-cut list

| Feature | Reason |
|---|---|
| Fwd URG Flags | Constant; bottom-decile importance for both heads |
| Bwd URG Flags | Constant; bottom-decile importance for both heads |
| Bwd Header Length | Bottom-decile category importance |
| SYN Flag Count | Bottom-decile binary importance |
| URG Flag Count | Constant; bottom-decile importance for both heads |
| CWR Flag Count | Near-zero variance; bottom-decile importance for both heads |
| ECE Flag Count | Near-zero variance; bottom-decile importance for both heads |

### CICIDS2017-DistriNet VAE-must-keep policy

Sixty-one of the 77 audited features participate in a structural relationship family or require leakage/artifact review. The complete list and per-feature reasons are in:

- `outputs/cicids2017distrinet/feature_audit/recommendations.csv`

Important keep decisions include:

- all byte/packet total and rate formula participants;
- all packet-length Min/Mean/Max/Std/Variance families;
- all Flow/Fwd/Bwd IAT family members;
- all Active and Idle family members;
- all bulk byte/packet/rate family members;
- subflow fields, even where expected local identities failed validation;
- directional PSH fields and `PSH Flag Count`;
- `RST Flag Count`, because its binary stump AUC requires artifact review.

`Subflow Bwd Packets` is explicitly labelled **keep for structural modeling, low value for classification**: it is near-constant and bottom-decile for both classifier heads, but remains part of the subflow structural model.

## Recommendations

1. Do not change either canonical feature schema from this audit alone.
2. Treat both classifier-candidate-cut lists as ablation proposals, not approved removals.
3. Keep the full feature sets available to the VAE, validator, perturbation mask, and structural-rule mining pipeline.
4. Investigate the DistriNet `RST Flag Count` generation path before relying on its high binary predictive power.
5. Do not revive the original CICIDS2017 Destination Port leakage claim for this DistriNet scope without new evidence; the current training-only stump results do not support it.
6. Do not enforce expected CICFlowMeter formulas that failed the local relationship checks as hard validator rules. Mine tolerances or conditional applicability from training data.
7. Retrain CICIoT2023 classifiers on the semantics-corrected bundle before using permutation importance for a production feature-removal decision.

## Verification

The generated artifacts were checked for:

- exactly 39 and 77 audited feature rows;
- square `39 × 39` and `77 × 77` Spearman matrices;
- binary and category univariate results for every feature;
- binary and category permutation results for every feature;
- correct bottom-decile counts;
- disjoint classifier-candidate-cut and VAE-must-keep sets;
- nonzero variance and multiple observed values for all nine corrected CICIoT2023 clip/round features;
- `Dst Port` not being flagged as a suspicious outlier;
- `RST Flag Count` being flagged for binary-head artifact review;
- matching SHA-256 hashes for every artifact recorded by each audit manifest.
