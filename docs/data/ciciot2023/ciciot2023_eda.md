# CICIoT2023 Exploratory Data Analysis Report

## 1. Purpose and scope

This report records the exploratory data analysis (EDA) performed on the current, leakage-safe CICIoT2023 preprocessing run. It covers:

- source-dataset size and schema;
- temporal split and train-only preprocessing provenance;
- full-source, sampled-training, validation, and test class distributions;
- raw-space feature summaries for the saved training matrix;
- protocol-value frequencies;
- feature correlation and constant-feature findings;
- principal component analysis (PCA);
- data-quality and artifact-contract checks;
- limitations that govern interpretation.

The EDA implementation is `src/evaluation/ciciot2023_eda.py`. Generated machine-readable tables, figures, PCA coordinates, and the EDA manifest are under `data/processed/ciciot2023_eda/`.

This report describes the run recorded at `2026-09-17T15:04:07.480417+00:00`, from Git commit `db65d87a3161b4c37f12608c55cfe7429a108a61`, with global seed 42.

## 2. Data and methodological provenance

### 2.1 Canonical inputs

The analysis uses the following canonical repository artifacts:

- schema: `src/preprocessing/schema.py`;
- labelled source: `data/processed/ciciot2023_labeled_full.parquet`;
- labelled build manifest: `data/processed/ciciot2023_labeled_full_manifest.json`;
- model-ready arrays: `data/processed/X_{train,val,test}.npy` and corresponding label arrays;
- train-fitted scaler: `data/processed/scaler.pkl`;
- preprocessing run record: `data/processed/run_manifest.json`.

The fixed feature schema contains 39 ordered features. This order is a data-format invariant: saved arrays, the scaler, models, attacks, and evaluation code all index features by position.

### 2.2 Source build accounting

| Measure | Value |
|---|---:|
| Raw CSV shards | 309 |
| Fine-grained labels | 34 |
| Coarse categories | 8 |
| Rows read | 46,776,700 |
| Rows retained | 46,775,660 |
| Rows dropped for NaN or infinity | 1,040 |
| Drop rate | 0.0022% |
| Labelled Parquet columns | 43 |
| Model features | 39 |

The labelled Parquet adds `Label`, `category`, `source_csv_filename`, and `source_folder` to the 39 features. The 1,040 discarded rows contained at least one NaN or positive/negative infinity. No clipping, scaling, or sampling occurs during this labelling stage.

### 2.3 Leakage-safe preprocessing order

The core pipeline performs these operations in order:

1. compute a forward-chaining temporal split from source-shard identity and natural-numeric shard order;
2. derive the 99.99th-percentile clipping limits from training rows only;
3. apply non-negative clipping and canonical integer/binary rounding;
4. fit `RobustScaler` on training rows only;
5. perform clustering-based undersampling on training rows only;
6. leave validation and test at their natural prevalence;
7. save arrays, encoders, class weights, kept row indices, hashes, and the run manifest.

The split protocols used across the 34 fine classes were:

| Protocol | Classes | Meaning |
|---|---:|---|
| Forward-chain by complete shards | 17 | Earliest shards train, later shards validation, latest shards test |
| Two-shard hybrid | 2 | Later shard test; earlier shard split contiguously into train/validation |
| One-shard contiguous block | 15 | Ordered train/validation/test blocks within the only available shard |

The preprocessing verifier confirmed that no shard crosses incompatible splits, every class occurs in every split, no test segment precedes a training segment within a class, and validation/test counts were untouched by sampling.

### 2.4 Saved split sizes

| Split | Rows | Feature shape | Population status |
|---|---:|---|---|
| Natural train before sampling | 32,972,198 | — | Temporal training partition before balancing |
| Saved train | 1,226,533 | 1,226,533 × 39 | Cluster-proportional-floor sampled |
| Validation | 5,555,150 | 5,555,150 × 39 | Natural holdout |
| Test | 8,248,312 | 8,248,312 × 39 | Natural holdout |

Six majority categories were capped at 200,000 saved training rows each: Benign, DDoS, DoS, Mirai, Recon, and Spoofing. BruteForce and Web were retained whole, contributing 9,146 and 17,387 training rows respectively. Therefore, training prevalence is intentionally different from validation/test prevalence.

## 3. Class distribution

### 3.1 Full-source category imbalance

The complete retained source is strongly attack-dominated:

| Category | Full source rows | Full source share | Sampled train | Train share | Validation | Validation share | Test | Test share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Benign | 1,098,126 | 2.3476% | 200,000 | 16.3061% | 310,395 | 5.5875% | 129,824 | 1.5739% |
| BruteForce | 13,064 | 0.0279% | 9,146 | 0.7457% | 1,306 | 0.0235% | 2,612 | 0.0317% |
| DDoS | 33,983,922 | 72.6530% | 200,000 | 16.3061% | 3,836,353 | 69.0594% | 5,876,390 | 71.2435% |
| DoS | 7,844,894 | 16.7713% | 200,000 | 16.3061% | 1,039,705 | 18.7161% | 1,518,314 | 18.4076% |
| Mirai | 2,633,870 | 5.6309% | 200,000 | 16.3061% | 247,012 | 4.4465% | 482,481 | 5.8495% |
| Recon | 690,521 | 1.4762% | 200,000 | 16.3061% | 69,050 | 1.2430% | 138,102 | 1.6743% |
| Spoofing | 486,435 | 1.0399% | 200,000 | 16.3061% | 48,850 | 0.8794% | 95,627 | 1.1594% |
| Web | 24,828 | 0.0531% | 17,387 | 1.4176% | 2,479 | 0.0446% | 4,962 | 0.0602% |
| **Total** | **46,775,660** | **100%** | **1,226,533** | **100%** | **5,555,150** | **100%** | **8,248,312** | **100%** |

DDoS and DoS together account for 89.42% of the full source, 87.78% of validation, and 89.65% of test. The training sampler prevents those categories from overwhelming model fitting while preserving natural prevalence in both holdouts.

The Benign share falls from 5.59% in validation to 1.57% in test. Since the test partition is temporally later, this is evidence of prevalence shift across the forward-chaining holdout rather than a consequence of resampling.

Binary target counts are:

| Split | Benign | Attack | Benign share | Attack share |
|---|---:|---:|---:|---:|
| Sampled train | 200,000 | 1,026,533 | 16.3061% | 83.6939% |
| Validation | 310,395 | 5,244,755 | 5.5875% | 94.4125% |
| Test | 129,824 | 8,118,488 | 1.5739% | 98.4261% |

![Category distribution by split](../../data/processed/ciciot2023_eda/category_distribution_by_split.png)

### 3.2 Fine-label distribution

The table distinguishes the complete labelled source from the **sampled** training matrix. Training counts should not be added to validation/test counts to reconstruct the full source because majority training categories were undersampled.

| Fine label | Full source | Sampled train | Validation | Test |
|---|---:|---:|---:|---:|
| BACKDOOR_MALWARE | 3,218 | 2,254 | 321 | 643 |
| BENIGN | 1,098,126 | 200,000 | 310,395 | 129,824 |
| BROWSERHIJACKING | 5,859 | 4,103 | 585 | 1,171 |
| COMMANDINJECTION | 5,409 | 3,788 | 540 | 1,081 |
| DDOS-ACK_FRAGMENTATION | 285,045 | 1,851 | 25,879 | 54,984 |
| DDOS-HTTP_FLOOD | 28,790 | 236 | 2,879 | 5,758 |
| DDOS-ICMP_FLOOD | 7,200,436 | 40,925 | 802,028 | 1,337,934 |
| DDOS-ICMP_FRAGMENTATION | 452,444 | 3,001 | 44,990 | 90,562 |
| DDOS-PSHACK_FLOOD | 4,094,727 | 23,845 | 533,641 | 614,984 |
| DDOS-RSTFINFLOOD | 4,045,248 | 24,126 | 532,479 | 599,941 |
| DDOS-SLOWLORIS | 23,425 | 217 | 2,342 | 4,685 |
| DDOS-SYNONYMOUSIP_FLOOD | 3,598,100 | 21,989 | 265,189 | 663,430 |
| DDOS-SYN_FLOOD | 4,059,097 | 23,815 | 534,739 | 669,651 |
| DDOS-TCP_FLOOD | 4,497,546 | 25,233 | 536,332 | 893,693 |
| DDOS-UDP_FLOOD | 5,412,169 | 32,834 | 533,737 | 879,973 |
| DDOS-UDP_FRAGMENTATION | 286,895 | 1,928 | 22,118 | 60,795 |
| DICTIONARYBRUTEFORCE | 13,064 | 9,146 | 1,306 | 2,612 |
| DNS_SPOOFING | 178,893 | 73,212 | 17,889 | 35,778 |
| DOS-HTTP_FLOOD | 71,857 | 1,848 | 5,085 | 31,173 |
| DOS-SYN_FLOOD | 2,028,791 | 48,624 | 265,509 | 468,898 |
| DOS-TCP_FLOOD | 2,671,363 | 78,643 | 259,654 | 316,666 |
| DOS-UDP_FLOOD | 3,072,883 | 70,885 | 509,457 | 701,577 |
| MIRAI-GREETH_FLOOD | 991,774 | 71,345 | 103,227 | 205,331 |
| MIRAI-GREIP_FLOOD | 751,589 | 59,639 | 70,863 | 115,899 |
| MIRAI-UDPPLAIN | 890,507 | 69,016 | 72,922 | 161,251 |
| MITM-ARPSPOOFING | 307,542 | 126,788 | 30,961 | 59,849 |
| RECON-HOSTDISCOVERY | 134,377 | 38,468 | 13,437 | 26,875 |
| RECON-OSSCAN | 98,255 | 28,638 | 9,825 | 19,651 |
| RECON-PINGSWEEP | 2,262 | 634 | 226 | 452 |
| RECON-PORTSCAN | 82,283 | 23,807 | 8,228 | 16,456 |
| SQLINJECTION | 5,244 | 3,672 | 524 | 1,048 |
| UPLOADING_ATTACK | 1,252 | 877 | 125 | 250 |
| VULNERABILITYSCAN | 373,344 | 108,453 | 37,334 | 74,668 |
| XSS | 3,846 | 2,693 | 384 | 769 |

All 34 labels remain present in all three saved splits. The smallest sampled-training pools are DDOS-SLOWLORIS (217), DDOS-HTTP_FLOOD (236), RECON-PINGSWEEP (634), and UPLOADING_ATTACK (877). These labels are available for 34-class learning, but their small sample sizes imply higher uncertainty than the large flood classes.

## 4. Feature-level analysis

### 4.1 Scope of feature statistics

Feature statistics below cover all 1,226,533 saved training rows after cleaning and training-only sampling. Values were inverse-transformed using the scaler fitted on the natural training partition. They do **not** describe the pristine raw source or the natural validation/test distributions.

| Feature | Type | Min | Q1 | Median | Q3 | Max | Mean | Zero % |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Header_Length | continuous | 4.76837e-07 | 8 | 20 | 25.6 | 60 | 17.7316 | 9.16 |
| Protocol Type | protocol | 0 | 6 | 6 | 17 | 47 | 12.798 | 0.35 |
| Time_To_Live | continuous | 0 | 64 | 64 | 92.2 | 248 | 84.206 | 0.01 |
| Rate | continuous | 0.000678071 | 178.453 | 2,982.3 | 16,387.8 | 626,015 | 16,643.9 | 0.00 |
| fin_flag_number | integer | 0 | 0 | 0 | 0 | 1 | 0.0197524 | 98.02 |
| syn_flag_number | integer | 0 | 0 | 0 | 0 | 1 | 0.0892646 | 91.07 |
| rst_flag_number | integer | 0 | 0 | 0 | 0 | 1 | 0.0308202 | 96.92 |
| psh_flag_number | integer | 0 | 0 | 0 | 0 | 1 | 0.040016 | 96.00 |
| ack_flag_number | integer | 0 | 0 | 0 | 1 | 1 | 0.362629 | 63.74 |
| ece_flag_number | integer | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| cwr_flag_number | integer | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| ack_count | integer | 0 | 0 | 2 | 8 | 100 | 6.01689 | 41.52 |
| syn_count | integer | 0 | 0 | 0 | 0 | 100 | 7.79024 | 75.98 |
| fin_count | integer | 0 | 0 | 0 | 0 | 100 | 2.07519 | 88.98 |
| rst_count | integer | 0 | 0 | 0 | 0 | 100 | 2.45673 | 92.48 |
| HTTP | binary | 0 | 0 | 0 | 0 | 1 | 0.0504242 | 94.96 |
| HTTPS | binary | 0 | 0 | 0 | 0 | 1 | 0.23213 | 76.79 |
| DNS | binary | 0 | 0 | 0 | 0 | 1 | 0.00152707 | 99.85 |
| Telnet | binary | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| SMTP | binary | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| SSH | binary | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| IRC | binary | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| TCP | binary | 0 | 0 | 1 | 1 | 1 | 0.574938 | 42.51 |
| UDP | binary | 0 | 0 | 0 | 0 | 1 | 0.227904 | 77.21 |
| DHCP | binary | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| ARP | binary | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| ICMP | binary | 0 | 0 | 0 | 0 | 1 | 0.0359151 | 96.41 |
| IGMP | binary | 0 | 0 | 0 | 0 | 0 | 0 | 100.00 |
| IPv | binary | 0 | 1 | 1 | 1 | 1 | 0.998456 | 0.15 |
| LLC | binary | 0 | 1 | 1 | 1 | 1 | 0.998456 | 0.15 |
| Tot sum | continuous | 120 | 1,484 | 6,000 | 12,296 | 132,695 | 14,324 | 0.00 |
| Min | continuous | 42 | 60 | 60 | 66 | 1,514 | 134.442 | 0.00 |
| Max | continuous | 46 | 66 | 276 | 724 | 13,098 | 719.378 | 0.00 |
| AVG | continuous | 46 | 60.06 | 121.8 | 557.68 | 3,117.36 | 350.01 | 0.00 |
| Std | continuous | 0 | 0 | 39.7996 | 148.416 | 3,793.12 | 199.42 | 36.02 |
| Tot size | continuous | 46 | 60.06 | 121.8 | 557.68 | 3,117.36 | 350.01 | 0.00 |
| IAT | continuous | 1.58947e-07 | 6.3529e-05 | 0.00035882 | 0.0063549 | 0.0807225 | 0.0054475 | 0.01 |
| Number | integer | 2 | 10 | 10 | 100 | 100 | 53.9174 | 0.00 |
| Variance | continuous | 0 | 0 | 1,584.01 | 22,027.4 | 1.43877e+07 | 194,852 | 36.02 |

### 4.2 Distribution shape

Several continuous features remain strongly right-skewed after the train-derived 99.99th-percentile clipping:

- `Rate`: median 2,982.3, mean 16,643.9, maximum 626,015;
- `Variance`: median 1,584.0, mean 194,851.9, maximum 14,387,723.8;
- `Max`: median 276, mean 719.4, maximum 13,098;
- `IAT`: median 0.000359, mean 0.00545, maximum 0.08072.

The mean substantially exceeds the median for each, indicating that large upper-tail observations remain important even after clipping. `Std` and `Variance` are exactly zero for 36.02% of the sampled training matrix. Most flag and service indicators are sparse. `IPv` and `LLC`, by contrast, equal one in 99.85% of training rows.

![Training feature distributions](../../data/processed/ciciot2023_eda/train_feature_distributions.png)

### 4.3 Protocol values

The inverse-transformed training matrix contains six observed `Protocol Type` values:

| Protocol value | Rows | Share |
|---:|---:|---:|
| 0 | 4,314 | 0.3517% |
| 1 | 44,474 | 3.6260% |
| 2 | 3 | 0.0002% |
| 6 | 750,960 | 61.2262% |
| 17 | 297,061 | 24.2196% |
| 47 | 129,721 | 10.5762% |

The value 6 dominates the sampled training matrix, followed by 17 and 47. The categorical protocol column is reported numerically because the pipeline preserves the CIC-shipped protocol values; no new semantic relabelling is introduced by the EDA.

### 4.4 Constant features after current preprocessing

Nine features have zero variance in the saved training matrix:

- `ece_flag_number`
- `cwr_flag_number`
- `Telnet`
- `SMTP`
- `SSH`
- `IRC`
- `DHCP`
- `ARP`
- `IGMP`

This finding applies to the **current cleaned training artifacts**, not necessarily to the pristine source CSVs. The current preprocessing computes upper clipping limits from training data and then rounds integer/binary features. Very rare positive values can consequently collapse to zero. These features contribute no discrimination to models trained on this saved matrix and produce undefined correlations.

## 5. Correlation structure

Spearman correlation was computed on a deterministic sample of 40,000 training rows: at most 5,000 rows from each of the eight categories. Validation and test were not used. The category-stratified design prevents the correlation matrix from being dominated only by DDoS prevalence, but it also means the matrix is not a prevalence-weighted population correlation estimate.

The strongest non-diagonal relationships are:

| Feature A | Feature B | Spearman ρ | Interpretation |
|---|---|---:|---|
| IPv | LLC | 1.0000 | Identical rank ordering in the sample |
| AVG | Tot size | 1.0000 | Redundant rank ordering |
| Std | Variance | 1.0000 | Expected monotonic dispersion relationship |
| Rate | IAT | -0.9958 | Near-perfect inverse relationship |
| Max | AVG | 0.9355 | Strong packet-size relationship |
| Max | Tot size | 0.9355 | Strong packet-size relationship |
| ack_flag_number | ack_count | 0.8506 | Flag/count coupling |
| Header_Length | TCP | 0.8334 | Strong protocol/header association |
| Max | Variance | 0.8160 | Larger maxima track higher dispersion |
| Max | Std | 0.8160 | Larger maxima track higher dispersion |
| Header_Length | ack_count | 0.8032 | Header/count association |
| Header_Length | ack_flag_number | 0.7978 | Header/flag association |

The exact or near-exact relationships indicate material redundancy. This does not require deleting columns from the locked 39-feature schema, but it matters when interpreting feature importance, attribution, covariance estimates, and generated samples.

There are 621 undefined cells in the 39 × 39 correlation matrix. They arise when at least one member of a pair is one of the nine constant training features. The heatmap marks these cells gray rather than treating them as zero correlation.

![Training Spearman correlation](../../data/processed/ciciot2023_eda/train_spearman_correlation.png)

## 6. Principal component analysis

PCA was fitted on the same 40,000-row category-stratified training sample. Inputs were the saved RobustScaler-space values clipped to `[-10, 10]` for visualization only. The scaler itself remained the train-fitted preprocessing scaler; validation and test did not participate.

| Component | Explained variance |
|---|---:|
| PC1 | 42.40% |
| PC2 | 24.13% |
| **PC1 + PC2** | **66.52%** |

The largest absolute loadings were:

| PC1 feature | Loading | PC2 feature | Loading |
|---|---:|---|---:|
| Tot sum | 0.4817 | Tot sum | 0.7226 |
| Number | 0.3691 | Max | 0.3296 |
| IAT | -0.3342 | AVG | 0.2832 |
| Std | -0.3177 | Tot size | 0.2832 |
| Variance | -0.3111 | Min | 0.2718 |
| Time_To_Live | -0.3013 | Protocol Type | 0.2097 |
| AVG | -0.2512 | Std | 0.1930 |
| Tot size | -0.2512 | Variance | 0.1879 |

The first two components are dominated by traffic-volume, packet-size, dispersion, timing, and protocol structure. The projection shows category overlap rather than clean linear separation. This is descriptive evidence only: the balanced plotting sample and two-dimensional projection must not be interpreted as classifier accuracy or proof of class separability.

![Training PCA by category](../../data/processed/ciciot2023_eda/train_pca_by_category.png)

## 7. Data-quality and contract checks

All automated EDA checks passed:

| Check | Result |
|---|---|
| Manifest feature order equals `schema.py` | Passed |
| Saved array counts equal `run_manifest.json` | Passed |
| All 34 fine classes present in train/validation/test | Passed |
| All 8 categories present in train/validation/test | Passed |
| Non-finite values in saved training matrix | 0 |
| Invalid binary values in inverse-transformed training matrix | 0 |
| Invalid integer values in inverse-transformed training matrix | 0 |
| Leakage/split verification | Passed |
| Validation/test sampled | No |

The generated figures were visually checked after execution. Axes, legends, labels, colorbars, and all expected panels rendered correctly. Undefined correlations are explicitly distinguished from zero correlations.

## 8. Main findings

1. **The dataset is extremely imbalanced.** DDoS alone contributes 72.65% of the full source. DDoS and DoS together contribute 89.42%.
2. **Temporal prevalence shifts are substantial.** Benign traffic is 5.59% of validation but only 1.57% of the later test partition. Test results therefore measure a materially more attack-heavy operating period.
3. **Train-only sampling changes prevalence by design.** Six majority categories have exactly 200,000 saved rows, while BruteForce and Web remain whole. Training shares must not be presented as natural dataset prevalence.
4. **All labels survive the pipeline.** Every fine label and coarse category occurs in train, validation, and test, although several fine labels have fewer than 1,000 saved training rows.
5. **Continuous features remain heavy-tailed.** Rate, IAT, packet-size statistics, and Variance retain large upper tails after 99.99th-percentile clipping.
6. **Nine processed training features are constant.** Their lack of variance is a property of the current clipping-and-rounding output and should be considered in downstream modelling and manifest-based feature-policy work.
7. **Several feature pairs are redundant or almost deterministic.** IPv/LLC, AVG/Tot size, Std/Variance, and Rate/IAT show absolute Spearman correlations near one.
8. **Two PCA components retain 66.52% of sampled variance.** Packet-volume, packet-size, timing, dispersion, and protocol variables dominate these components, but categories still overlap in two dimensions.
9. **The saved artifacts satisfy their basic numerical contract.** No NaN/Inf, invalid binary, or invalid integer values were found in the inverse-transformed training matrix.

## 9. Interpretation limits

- Full-source and holdout class counts are exact, but feature statistics describe the **cleaned, sampled training matrix**, not the pristine 46.8-million-row source.
- Correlation and PCA use a category-stratified training sample of 5,000 rows per category. This is appropriate for cross-category visualization but intentionally differs from natural prevalence.
- PCA inputs are clipped to `[-10, 10]` in scaler space for visual stability. This clipping is EDA-only and does not alter saved model inputs.
- Spearman correlation describes monotonic association, not causation or independent predictive value.
- PCA is a linear projection and cannot establish classifier separability.
- Seventeen classes have complete-shard forward chaining. Two use a two-shard hybrid, and fifteen single-shard classes rely on ordered within-shard blocks, which are a weaker temporal proxy.
- Constant-feature findings are run-specific and should be recomputed if clipping, rounding, sampling, or source data change.
- No validation or test observations were used to fit clipping limits, the scaler, correlation transforms, or PCA.

## 10. Reproducibility

Run preprocessing and leakage verification first:

```bash
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.pipeline
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.preprocessing.ciciot2023.reports verify
```

Regenerate the EDA:

```bash
C:/Users/user6/.local/share/mamba/envs/thesis/python.exe -m src.evaluation.ciciot2023_eda
```

The EDA command writes:

- `eda_report.json`
- `fine_class_distribution_by_split.csv`
- `category_distribution_by_split.csv`
- `train_feature_summary_raw.csv`
- `train_protocol_distribution.csv`
- `train_spearman_correlation.csv`
- `train_pca_loadings.csv`
- `train_pca_coordinates.npz`
- four PNG figures

All outputs are stored in `data/processed/ciciot2023_eda/`. The JSON report records the run timestamp, source/build counts, preprocessing provenance, array shapes, fit provenance, PCA variance, constant features, numerical checks, and generated-output inventory.
