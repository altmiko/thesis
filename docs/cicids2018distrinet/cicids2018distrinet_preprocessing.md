# CSE-CIC-IDS-2018 DistriNet preprocessing

This document defines the production preprocessing pipeline for the thesis's secondary dataset: the corrected DistriNet CSE-CIC-IDS-2018 release.

- Pipeline: `src/preprocessing/preprocess_cicids2018_distrinet.py`
- Adapter: `src/datasets/cicids2018.py` (`get_adapter("cicids2018")`)
- Tests: `src/preprocessing/tests/test_preprocess_cicids2018_distrinet.py`, `src/datasets/tests/test_cicids2018.py`
- Processed data: `data/processed/CSECICIDS_2018_Distrinet/`
- Generated report and figures: `outputs/cicids2018distrinet/preprocessing/`
- EDA evidence: `docs/data/cicids2018distrinet/eda_explained.md`
- Classifier results on this data: `outputs/cicids2018distrinet/classifiers_multiseed/cicids2018_multiseed_classifier_results.md`

All counts below come from the full production run over 63,195,145 rows with seed 42 and `--class-row-targets Benign=250000,DoS=200000,DDoS=200000`. The run completed in 368 seconds with 10 workers. `preprocessing_manifest.json` is the machine-readable source of truth.

> After the chronological split, Benign, DoS and DDoS were deterministically undersampled to fixed totals of 250,000, 200,000 and 200,000 rows. Each total was apportioned 70/15/15 across train, validation and test and sampled independently inside each partition, stratified by source label, source file and UTC hour. Recon and BruteForce retain every row. No observation is duplicated or synthesised.

> The resulting partitions do not reproduce the original CICIDS2018 class prior and should therefore be interpreted as controlled experimental datasets rather than estimates of real-world attack prevalence.

---

## 1. Scope and claim boundary

- Input: ten corrected/relabelled DistriNet CSVs, 36 GB, 91 columns, 63,195,145 rows.
- Targets: binary Benign/Attack and five categories: Benign, DoS, DDoS, Recon, BruteForce.
- Feature-space only: the pipeline reads CICFlowMeter aggregates, not PCAPs.
- Split: chronological 70/15/15 within each retained source label. This preserves closed-set coverage but remains a within-campaign split; it is not global forward-time or new-campaign evaluation.
- Class composition: controlled by fixed per-class row targets (≈30% Benign, 24% DoS, 24% DDoS, 11% Recon, 11% BruteForce in every split). None reproduces the original ≈94.6% Benign prior.

## 2. Processing order

```text
raw corrected CICIDS2018
→ validate files and schema
→ cleaning
→ label mapping
→ float32 canonicalisation
→ global feature+class deduplication
→ chronological 70/15/15 split within retained source label
→ class-size reduction (Benign, DoS, DDoS) independently inside train/validation/test
→ fit RobustScaler on final training rows only
→ transform validation/test with the frozen training scaler
→ final arrays, Parquet, audits, manifest, reports, figures
```

The chronological split is fixed before any sampling. Sampling only removes rows of a targeted class that are already in a partition; it never moves a row between partitions.

The 36 GB source is processed in two streaming passes. Pass 1 stores compact provenance, timestamp, label, header flags and 128-bit feature hashes. Deduplication, splitting and sampling operate on those keys. Pass 2 rereads the CSVs and writes only the selected rows into float32 memory-mapped arrays, recomputing hashes to detect any pass-to-pass drift.

## 3. Running the pipeline

```powershell
$Env:PATH = $Env:PATH -replace 'C:\\Program Files\\Tailscale"', 'C:\Program Files\Tailscale'
conda activate C:\Users\user6\.local\share\mamba\envs\thesis
python src/preprocessing/preprocess_cicids2018_distrinet.py `
  --class-row-targets Benign=250000,DoS=200000,DDoS=200000 `
  --workers 10
```

| Option | Default | Meaning |
|---|---:|---|
| `--class-row-targets` | `Benign=250000,DoS=200000,DDoS=200000` | total rows per listed class over all splits; unlisted classes keep every row |
| `--stratum-hours` | `1` | UTC time-bin width crossed with source label and source file |
| `--seed` | `42` | deterministic within-stratum sampling seed |
| `--workers` | `8` | files processed concurrently; `1` runs in-process |
| `--input-dir` | `data/raw/CSECICIDS2018_Distrinet` | exactly the ten corrected DistriNet files |
| `--output-dir` | `data/processed/CSECICIDS_2018_Distrinet` | processed artifacts |
| `--report-dir` | `outputs/cicids2018distrinet/preprocessing` | generated report and figures |
| `--max-rows-per-file` | none | smoke limit; marks output non-production |

## 4. Schema and 79-feature contract

The pipeline requires identical 91-column headers and exactly the ten expected file names. It takes the modelling feature names and order from the CICIDS2017 preprocessing manifest and requires all 79 names in the same relative order.

Excluded from `X`:

- metadata/identifiers: `id`, `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, `Label`, `Attempted Category`;
- CICIDS2018-only fields: `Fwd RST Flags`, `Bwd RST Flags`, `ICMP Code`, `ICMP Type`, `Total TCP Flow Time`.

Excluding the ICMP fields removes their `-1` non-ICMP sentinel from the modelling schema. Provenance is kept separately: source file, source `id`, original CSV row, Flow ID, endpoint IPs, parsed UTC timestamp, original label, Attempted Category and mapped labels.

Features are parsed as numeric, checked for finiteness, cast to float32 and saved in the CICIDS2017 order. `-0.0` is folded into `+0.0` before hashing.

## 5. Label mapping

| Final class | Retained source labels before Attempted handling |
|---|---|
| Benign | `BENIGN`; every label ending ` - Attempted` |
| DoS | DoS Hulk, DoS GoldenEye, DoS Slowloris |
| DDoS | DDoS-HOIC, DDoS-LOIC-HTTP, DDoS-LOIC-UDP |
| Recon | Infiltration - NMAP Portscan |
| BruteForce | SSH-BruteForce |

All ten observed Attempted labels map to source label `BENIGN` and final class Benign. Their `original_label`, `Attempted Category` and `is_attempted` fields stay as metadata. They take part in the Benign sampling of whichever chronological split they already belong to.

Dropped labels: Botnet Ares, Web Attack - Brute Force, Web Attack - XSS, Web Attack - SQL, Infiltration - Dropbox Download, and Infiltration - Communication Victim Attacker. Unknown labels abort the run rather than falling into an implicit bucket.

## 6. Cleaning

Rules are deterministic and row-local; none uses a fitted statistic.

| Rule | Removed rows |
|---|---:|
| non-finite value in any of the 79 model features | 57 |
| missing/unparseable timestamp | 0 |
| negative value outside Fwd/Bwd Header Length | 0 |
| unsupported final class | 143,493 |
| **retained after cleaning/mapping** | **63,051,595** |

The 57 non-finite rows are all Benign zero-duration flows with `+inf` in `Flow Bytes/s` and `Flow Packets/s`. They are dropped, not imputed. No imputation, winsorisation or clipping is used.

### Header-length extractor artefact

`Fwd Header Length` and `Bwd Header Length` span the signed 16-bit range and wrap on long flows. Rows are kept and flagged rather than dropped:

- negative flags: the stored value is `< 0`;
- overflow flags: packets × minimum transport header (TCP 20 B, UDP 8 B) exceeds 32,767, so the true header count cannot fit in the stored field.

DDoS-LOIC-UDP is the most affected: 2,432 of its 2,527 rows must have wrapped, but only 1,209 became negative. Dropping negative rows would remove 47.8% of this source label and still leave 1,223 wrapped positive values. The wrap count is unknown, so no correction is defensible. The raw extractor values stay in `X`; four flags stay in metadata only.

## 7. Deduplication and conflicts

Before splitting, the pipeline deduplicates globally by exact equality of:

```text
canonical float32 vector of all 79 modelling features + mapped final class
```

The earliest `(timestamp, source-file order, id)` occurrence survives. `Flow ID` is never used.

| Measure | Rows |
|---|---:|
| before deduplication | 63,051,595 |
| duplicates removed | 657,839 (1.043%) |
| after deduplication | 62,393,756 |

Duplicates removed by class: Benign 657,351; DoS 469; Recon 19; DDoS 0; BruteForce 0. The 128-bit deduplication hash is re-checked against the saved arrays, and shared fingerprints are compared byte-exactly.

Before reduction, one feature vector carried two classes (Recon and Benign). The key includes the class, so this is a label conflict, not a duplicate. The final outputs contain zero feature-only or feature+class overlaps across splits.

## 8. Chronological split

Rows are sorted by the parsed UTC timestamp, with source-file order and source `id` as deterministic tie-breakers. Filename dates do not determine chronology.

For each of the nine retained source labels independently:

1. select its cleaned, deduplicated rows;
2. keep chronological order;
3. allocate 70/15/15 by largest remainder, with at least one row per split;
4. cut the ordered block into train, validation and test;
5. merge source labels into the five final categories.

Class counts immediately after splitting, before reduction:

| Split | Benign | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|---:|
| Train | 41,301,621 | 1,283,619 | 961,904 | 62,549 | 65,938 |
| Validation | 8,850,347 | 275,061 | 206,122 | 13,403 | 14,130 |
| Test | 8,850,347 | 275,061 | 206,122 | 13,403 | 14,129 |

These memberships remain authoritative. Reduction only changes which members of a targeted class are written out.

## 9. Class-size reduction

### 9.1 Targets

| Class | Total target | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Benign | 250,000 | 175,000 | 37,500 | 37,500 |
| DoS | 200,000 | 140,000 | 30,000 | 30,000 |
| DDoS | 200,000 | 140,000 | 30,000 | 30,000 |
| Recon | all rows | 62,549 | 13,403 | 13,403 |
| BruteForce | all rows | 65,938 | 14,130 | 14,129 |

Each total is apportioned 70/15/15 by the same largest-remainder rule used for the chronological split (`allocate_class_counts`). Every split therefore gets the same class composition. If a split held fewer rows than its target, all would be kept and `target_reached=false` recorded; all nine production targets were reached.

### 9.2 Sampling algorithm

Within each split and each targeted class:

1. Take only that class's rows in that split.
2. Stratum = `(source label, source file, floor(UTC timestamp / 1 hour))`. Including the source label keeps subtype shares (e.g. DoS Hulk vs GoldenEye vs Slowloris) proportional.
3. Allocate the split target by Hamilton/largest-remainder apportionment in proportion to stratum size.
4. If the target is at least the number of strata, a stratum rounded to zero receives one row, taken from the most over-allocated stratum. No stratum is ever asked for more rows than it holds; this is asserted.
5. Sample uniformly without replacement with `default_rng([seed, split_index, class_index, source_index, file_index, time_bin])`.
6. Put the selected rows back in chronological order.

### 9.3 Before/after counts

| Split | Class | Before | After | Removed | Status |
|---|---|---:|---:|---:|---|
| Train | Benign | 41,301,621 | 175,000 | 41,126,621 | reduced |
| Train | DoS | 1,283,619 | 140,000 | 1,143,619 | reduced |
| Train | DDoS | 961,904 | 140,000 | 821,904 | reduced |
| Train | Recon | 62,549 | 62,549 | 0 | unchanged |
| Train | BruteForce | 65,938 | 65,938 | 0 | unchanged |
| Validation | Benign | 8,850,347 | 37,500 | 8,812,847 | reduced |
| Validation | DoS | 275,061 | 30,000 | 245,061 | reduced |
| Validation | DDoS | 206,122 | 30,000 | 176,122 | reduced |
| Validation | Recon | 13,403 | 13,403 | 0 | unchanged |
| Validation | BruteForce | 14,130 | 14,130 | 0 | unchanged |
| Test | Benign | 8,850,347 | 37,500 | 8,812,847 | reduced |
| Test | DoS | 275,061 | 30,000 | 245,061 | reduced |
| Test | DDoS | 206,122 | 30,000 | 176,122 | reduced |
| Test | Recon | 13,403 | 13,403 | 0 | unchanged |
| Test | BruteForce | 14,129 | 14,129 | 0 | unchanged |

### 9.4 Source-label composition after reduction

| Source label | Train | Validation | Test |
|---|---:|---:|---:|
| BENIGN (incl. Attempted) | 175,000 | 37,500 | 37,500 |
| DoS Hulk | 137,665 | 29,500 | 29,499 |
| DoS GoldenEye | 1,722 | 369 | 370 |
| DoS Slowloris | 613 | 131 | 131 |
| DDoS-HOIC | 110,265 | 23,628 | 23,628 |
| DDoS-LOIC-HTTP | 29,477 | 6,317 | 6,317 |
| DDoS-LOIC-UDP | 258 | 55 | 55 |
| Infiltration - NMAP Portscan | 62,549 | 13,403 | 13,403 |
| SSH-BruteForce | 65,938 | 14,130 | 14,129 |

Because sampling is proportional, rare subtypes shrink with their class. GoldenEye, Slowloris and LOIC-UDP keep their original share of DoS/DDoS but have few absolute rows (LOIC-UDP: 55 in validation and 55 in test). Their per-subtype estimates are unstable.

### 9.5 Temporal coverage

| Split | Class | Available | Selected | Sampling fraction | Strata | Represented after |
|---|---|---:|---:|---:|---:|---:|
| Train | Benign | 41,301,621 | 175,000 | 0.424% | 116 | 116 |
| Train | DoS | 1,283,619 | 140,000 | 10.91% | 3 | 3 |
| Train | DDoS | 961,904 | 140,000 | 14.55% | 4 | 4 |
| Validation | Benign | 8,850,347 | 37,500 | 0.424% | 19 | 19 |
| Validation | DoS | 275,061 | 30,000 | 10.91% | 3 | 3 |
| Validation | DDoS | 206,122 | 30,000 | 14.55% | 4 | 4 |
| Test | Benign | 8,850,347 | 37,500 | 0.424% | 20 | 20 |
| Test | DoS | 275,061 | 30,000 | 10.91% | 4 | 4 |
| Test | DDoS | 206,122 | 30,000 | 14.55% | 4 | 4 |

Every stratum stays represented. The minimum-one-row rule moved 17 rows in train Benign and 1 row each in validation Benign and test DoS.

### 9.6 Attempted-derived Benign kept

| Split | Original Attempted label | Before | After |
|---|---|---:|---:|
| Train | FTP-BruteForce - Attempted | 87,236 | 353 |
| Train | DoS Slowloris - Attempted | 2,280 | 18 |
| Train | DoS GoldenEye - Attempted | 4,301 | 11 |
| Train | DDoS-LOIC-UDP - Attempted | 251 | 2 |
| Train | Web Attack - Brute Force / DoS Hulk / Web Attack - SQL / Web Attack - XSS - Attempted | 137 / 86 / 14 / 4 | 0 / 0 / 0 / 0 |
| Validation | Infiltration - Dropbox Download - Attempted | 28 | 1 |
| Test | Botnet Ares - Attempted | 262 | 1 |

Strata are temporal, not per original label, so rare Attempted labels can drop to zero. Full provenance remains in `row_index.parquet`.

## 10. Final datasets

| Split | Total | Benign | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|---:|---:|
| Train | **583,487** | 175,000 (29.99%) | 140,000 (23.99%) | 140,000 (23.99%) | 62,549 (10.72%) | 65,938 (11.30%) |
| Validation | **125,033** | 37,500 (29.99%) | 30,000 (23.99%) | 30,000 (23.99%) | 13,403 (10.72%) | 14,130 (11.30%) |
| Test | **125,032** | 37,500 (29.99%) | 30,000 (23.99%) | 30,000 (23.99%) | 13,403 (10.72%) | 14,129 (11.30%) |

The largest-to-smallest class ratio is now 2.8 : 1 (was ≈660 : 1 in the natural data and ≈20 : 1 in the previous 30%-Benign design).

## 11. Scaling and leakage boundary

`RobustScaler` is fitted on the **final reduced training matrix only** (583,487 × 79). Its median/IQR parameters are frozen and applied unchanged to validation and test.

The same boundary applies downstream:

- scaler, feature-selection statistics, CFF masks, mined constraints, validator training profile, VAE and PrimAttack budget calibration: training only;
- model/attack selection and calibration: validation;
- final reporting: test.

Sampling validation/test rows is not a learned transformation; it uses only each partition's own membership, timestamps, source identifiers and the configured seed.

Reference class weights from the final training labels (`n / (5 × count)`):

- category: `[0.6668, 0.8336, 0.8336, 1.8657, 1.7698]`;
- binary: `[1.6671, 0.7142]`.

`asinh` remains a differentiable model-side transform applied after robust scaling, exactly as in CICIDS2017. It is not applied twice.

## 12. Controlled-evaluation interpretation and metrics

Accuracy is no longer dominated by the ≈94.6% Benign prior, but the composition is still a design choice, not an estimate of operational prevalence.

Report:

- per-class precision, recall and F1;
- macro-F1;
- balanced accuracy;
- confusion matrix;
- ordinary accuracy only as a supplementary metric.

Do not read softmax outputs as calibrated real-world probabilities. Any deployment-prior analysis needs explicit prior correction or calibration against data with the intended prevalence.

## 13. Assertions and tests

Production assertions and permanent tests establish:

1. only Benign, DoS and DDoS rows are removed during reduction;
2. every Recon and BruteForce row survives, per class, per split and per row;
3. every targeted class reaches exactly min(available, target) in every split, and its three split counts sum to the total target;
4. sampled rows never move between partitions;
5. final splits are disjoint by `source_file:id`;
6. no feature+class duplicate exists within or across final splits;
7. every temporal stratum stays represented whenever the target allows one row per stratum;
8. no source label disappears from a split during sampling when the target allows it;
9. the same seed and input produce identical rows and arrays;
10. changing the seed changes only the sampled subsets, not untargeted rows, class counts or split membership;
11. the saved scaler centre equals the median of the final training rows, not of all three splits;
12. no NaN/Inf remains, and negative values appear only in the two header columns;
13. every retained source label is present in every split, and chronological boundaries stay ordered.

## 14. Outputs

`data/processed/CSECICIDS_2018_Distrinet/` contains:

- `X_{train,val,test}.npy`, `X_{split}_pristine.npy`;
- `y_{split}_cat.npy`, `y_{split}_bin.npy`;
- `{train,val,test}.parquet` with provenance and pristine features (`sampling_stratum` metadata column);
- `timestamp_epoch_{seconds,us}_{split}.npy`;
- `scaler.pkl`, `scaler_parameters.json`, encoders and class weights;
- `row_index.parquet` for all 63,051,595 cleaned/mapped rows;
- cleaning, header, duplicate and leakage audits;
- `class_reduction_by_split.csv`, `source_label_reduction_by_split.csv`, `split_composition.csv`;
- `attempted_benign_by_split.csv`;
- `sampling_strata.csv`, `sampling_audit.json`;
- `preprocessing_manifest.json`.

`outputs/cicids2018distrinet/preprocessing/figures/` contains:

1. `class_distribution_before_reduction.png`: the original retained split distributions;
2. `class_distribution_{train,val,test}_after_reduction.png`;
3. `sampling_strata_{Benign,DoS,DDoS}.png`: population vs selected rows in every stratum;
4. `spearman_train_sample.png` and `pca_train_stratified.png` (PC1 60.2%, PC2 15.2%): descriptive only.

Every after-reduction figure states that the class counts are a design choice, not natural CICIDS2018 prevalence.

## 15. Compatibility and limitations

The final file names, dtypes, class ids and 79-feature order match the CICIDS2017 processed layout. `CICIDS2018Adapter` changes the processed path and the header-length bounds and otherwise reuses the generic `DatasetAdapter → FeatureManifest → FeatureTransform` interface. The six CICIDS2017 derived identities have zero violations on all 583,487 final training rows and all 125,033 validation rows.

Limitations:

- attack campaigns are not independent across partitions;
- the sampling uses one seed (42), so sampling variance is not quantified;
- class composition is controlled and unsuitable for prevalence claims or naive probability calibration;
- DoS and DDoS are downsampled to 10.9% and 14.6% of their rows; each campaign spans only 3–4 hourly strata, so temporal diversity within these classes was already small;
- rare subtypes are small after proportional sampling (LOIC-UDP 55 validation / 55 test rows; Slowloris 131 / 131);
- rare Attempted labels can have zero selected observations;
- header-length overflow and attacker-tool fingerprints remain learnable artefacts;
- CICIDS2018-specific VAE, validator profile/rules, CFF masks and PrimAttack budgets still need training-only fitting before adversarial evaluation;
- aggregate features remain a proxy; no packet-realisability claim follows.
