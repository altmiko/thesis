**Chapter 3**

**Data Preprocessing and Dataset Engineering for CIC-IDS-2017**

# **3.1 Chapter Overview and Pipeline Demarcation**

Network Intrusion Detection Systems (NIDS) built on deep neural networks and generative models are highly sensitive to the statistical fidelity, distribution, and integrity of their training representations. In benchmark NIDS evaluations, improper preprocessing frequently introduces subtle forms of data leakage, unrealistic feature artifacts, or unphysical numerical quantities. Such flaws degrade the generalization ability of downstream classifiers and undermine the validity of adversarial robustness and generative modeling experiments built upon them. This chapter documents, in full methodological detail, the data engineering pipeline that transforms the raw CIC-IDS-2017 capture files into the leakage-controlled, normalized dataset used throughout this thesis.

## **3.1.1 Division of Responsibilities: DistriNet versus This Thesis Pipeline**

To preserve academic transparency, this chapter explicitly demarcates the boundary between two distinct bodies of work: (i) the upstream, network-layer re-extraction of the dataset performed by DistriNet (Engelen et al., 2021; Liu et al., 2022), and (ii) the downstream, machine-learning-specific preprocessing pipeline designed and implemented by the author for this thesis. The former corrects flow-reassembly and labeling errors in the original CICFlowMeter output; the latter performs data hygiene, leakage-controlled partitioning, scaling, and class-imbalance handling in preparation for neural classifier and generative-model training. Table 3.0 summarizes this division of responsibility for every stage of the pipeline.

**_Table 3.0 — Pipeline Responsibility Matrix: DistriNet versus Author's Implementation_**

| **Preprocessing Stage**                          | **DistriNet** | **Author (This Thesis)** | **Implementation Reference**                                          |
| ------------------------------------------------ | ------------- | ------------------------ | --------------------------------------------------------------------- |
| Raw PCAP parsing & flow reassembly               | Yes           | No                       | DistriNet corrected CICFlowMeter                                      |
| TCP session termination fixes (FIN/RST)          | Yes           | No                       | DistriNet CNS 2022 codebase                                           |
| Bidirectional flow direction correction          | Yes           | No                       | DistriNet CNS 2022 codebase                                           |
| Attack-specific PCAP-to-CSV labeling             | Yes           | No                       | DistriNet ground-truth match rules                                    |
| Tagging incomplete attacks ("- Attempted")       | Yes           | No                       | DistriNet CSV export generator                                        |
| SHA-256 raw file validation                      | No            | Yes                      | preprocess_cicids2017_distrinet.py – verify_raw_inventory             |
| Decoupling IP/metadata from 79 features          | No            | Yes                      | preprocess_cicids2017_distrinet.py – clean_single_file                |
| Attempted-flow policy (Attempted → Benign)       | No            | Yes                      | preprocess_cicids2017_distrinet.py – map_labels                       |
| NaN / ±∞ filtering (793 rows)                    | No            | Yes                      | preprocess_cicids2017_distrinet.py – filter_invalid_rows              |
| Negative physical metric pruning (2,922 rows)    | No            | Yes                      | preprocess_cicids2017_distrinet.py – filter_invalid_rows              |
| Multi-class taxonomy mapping (25 → 5)            | No            | Yes                      | preprocess_cicids2017_distrinet.py – filter_supported_categories      |
| Global float32 deduplication (15,754 rows)       | No            | Yes                      | preprocess_cicids2017_distrinet.py – remove_global_duplicates         |
| Leakage-controlled chronological 70/15/15 split  | No            | Yes                      | preprocess_cicids2017_distrinet.py – chronological_per_category_split |
| Train-only RobustScaler fitting                  | No            | Yes                      | preprocess_cicids2017_distrinet.py – fit_and_apply_scaler             |
| Train-constant feature handling                  | No            | Yes                      | preprocess_cicids2017_distrinet.py – fit_and_apply_scaler             |
| Class-balanced loss weights (β = 0.999)          | No            | Yes                      | cicids2017d_experiments.py – compute_effective_number_weights         |
| Export of Parquet, NumPy, and scaler artifacts   | No            | Yes                      | preprocess_cicids2017_distrinet.py – save_production_outputs          |
| Automated leakage, coverage & monotonicity audit | No            | Yes                      | preprocess_cicids2017_distrinet.py – run_leakage_and_chronology_audit |

**_Figure 3.1 — End-to-End Pipeline Architecture_**

The pipeline proceeds in two sequential phases, illustrated schematically below.

**Phase 1 (DistriNet, upstream):** Raw UNB PCAPs (3–7 July 2017) → patched CICFlowMeter engine → TCP handshake/termination correction → attack-specific schedule matching → tagging of incomplete attacks with the "- Attempted" suffix → export of five raw CSV files (84 columns, 2,100,814 rows).

**Phase 2 (author, downstream):** SHA-256 verification and schema normalization → decoupling of IP/metadata from the 79 modeling features → remapping of Attempted flows to Benign with provenance retained → removal of 793 non-finite and 2,922 negative-metric rows → consolidation of 25 raw label strings into 5 macro-classes → global float32 deduplication (15,754 exact duplicates removed) → within-category chronological 70/15/15 split → RobustScaler fitting strictly on the 1.45M training rows → projection of validation/test sets with unit-scale assignment for constant flags → computation of effective-number class-loss weights (β = 0.999) → export of Parquet tables, NumPy arrays, the fitted scaler, and automated leakage audits.

# **3.2 Dataset Provenance and the DistriNet Correction**

_Contribution scope: upstream network-level flow re-extraction (Engelen et al., 2021; Liu et al., 2022)._

## **3.2.1 Limitations of the Original CIC-IDS-2017 Release**

The University of New Brunswick (UNB) generated the original CIC-IDS-2017 dataset by capturing network traffic from Monday, 3 July 2017, to Friday, 7 July 2017. While widely adopted across the intrusion-detection literature, subsequent audits revealed critical flaws in the original MachineLearningCSV release produced by CICFlowMeter. As established by Engelen et al. (2021) and Liu et al. (2022), these flaws include:

- Premature flow termination — flows were frequently terminated upon observing the initial FIN flag rather than the completion of the full TCP four-way handshake.
- RST handling inconsistencies — reset (RST) packets did not consistently terminate flow sessions, producing split records and corrupted session metrics.
- Directional reversal and header duplication — certain bidirectional flow statistics suffered from reversed initiator/responder direction and duplicated header-size columns.
- Coarse time/IP-based labeling — non-malicious or uncompleted connection attempts were broadly marked as active attacks based purely on time windows and static target IP addresses, creating severe label noise and shortcut-learning vectors.

## **3.2.2 The DistriNet Re-Extraction Release**

To eliminate these systematic artifacts, this research adopts the corrected five-file DistriNet CIC-IDS-2017 release, generated using corrected network flow-reassembly and attack-specific verification logic. Table 3.1 lists the exact raw input files, byte sizes, raw record counts, capture spans, and cryptographic SHA-256 hashes used as the immutable foundation of this pipeline.

**_Table 3.1 — Raw Input Inventory (SHA-256 checksums recorded in the provenance manifest; omitted here for brevity)_**

| **File Name**              | **Size (Bytes)** | **Raw Rows** | **Capture Window (UTC)**       |
| -------------------------- | ---------------- | ------------ | ------------------------------ |
| Monday-WorkingHours.csv    | 208,244,807      | 371,749      | 2017-07-03 13:55:58 – 22:01:34 |
| Tuesday-WorkingHours.csv   | 178,453,210      | 322,003      | 2017-07-04 13:53:44 – 22:00:30 |
| Wednesday-WorkingHours.csv | 291,529,516      | 496,779      | 2017-07-05 13:42:42 – 22:10:14 |
| Thursday-WorkingHours.csv  | 187,707,525      | 362,368      | 2017-07-06 13:59:00 – 22:04:36 |
| Friday-WorkingHours.csv    | 282,607,319      | 547,915      | 2017-07-07 13:59:50 – 22:02:40 |
| Total Pipeline Foundation  | 1,148,542,377    | 2,100,814    | 5 full capture days            |

## **3.2.3 Treatment of Attempted Flows**

The corrected DistriNet tooling introduced explicit "- Attempted" suffixes for flows where an attack was initiated but failed to produce a completed malicious payload exchange (e.g., the target port was closed, or the connection timed out). DistriNet's official guidance mandates that Attempted flows must not be modeled as distinct attack classes. In alignment with standard intrusion-detection semantics, the pipeline implements the following deterministic policy:

_Label(raw) = "&lt;Attack&gt; - Attempted" ⇒ Category = Benign_ (3.1)

Across the 2,100,814 raw records, exactly 9,144 rows (0.4353%) represent Attempted attacks. These flows are mapped to Benign, while their raw ground-truth label is preserved in the provenance metadata table for auditability.

# **3.3 Feature Schema and Representation**

_Contribution scope: author's pipeline design (feature isolation and taxonomy)._

The raw CSV headers comprise 84 columns. To prevent shortcut learning and identity memorization, non-behavioral context fields are decoupled from the statistical modeling features:

1. Metadata and identity columns, excluded from X: Flow ID (textual 5-tuple string; see Section 3.4.3); Src IP and Dst IP (network host identifiers, excluded to prevent the model from memorizing specific host subnets); Timestamp (wall-clock string, parsed and converted to integer epoch seconds and retained solely for temporal ordering); and Label (the categorical target string).
2. Feature matrix, X ∈ ℝ^(N×79): exactly 79 continuous and discrete numeric features capturing flow volume, timing dynamics, packet-size distributions, TCP flag states, and subflow transmission metrics.

**_Table 3.2 — The 79 Canonical Modeling Features (Ordered Representation)_**

| **Index Range** | **Subsystem / Feature Group** | **Included Features**                                                                                |
| --------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| 00–02           | Endpoint & Transport          | Src Port, Dst Port, Protocol                                                                         |
| 03–07           | Flow Lifetime & Volume        | Flow Duration, Total Fwd Packet, Total Bwd Packets, Total Length of Fwd/Bwd Packet                   |
| 08–15           | Directional Packet Lengths    | Fwd/Bwd Packet Length {Max, Min, Mean, Std}                                                          |
| 16–21           | Flow Rates & Global IAT       | Flow Bytes/s, Flow Packets/s, Flow IAT {Mean, Std, Max, Min}                                         |
| 22–31           | Directional IAT               | Fwd/Bwd IAT {Total, Mean, Std, Max, Min}                                                             |
| 32–39           | Header & Directional Flags    | Fwd/Bwd PSH & URG Flags, Fwd/Bwd Header Length, Fwd/Bwd Packets/s                                    |
| 40–44           | Global Packet Statistics      | Packet Length {Min, Max, Mean, Std, Variance}                                                        |
| 45–52           | TCP Control Flags             | FIN, SYN, RST, PSH, ACK, URG, CWR, ECE Flag Counts                                                   |
| 53–63           | Aggregates & Bulk Rates       | Down/Up Ratio, Average Packet Size, Fwd/Bwd Segment Size Avg, Fwd/Bwd {Bytes, Packet, Rate}/Bulk Avg |
| 64–70           | Subflows & TCP Window         | Subflow Fwd/Bwd {Packets, Bytes}, Fwd/Bwd Init Win Bytes, Fwd Act Data Pkts, Fwd Seg Size Min        |
| 71–78           | Active & Idle Timers          | Active {Mean, Std, Max, Min}, Idle {Mean, Std, Max, Min}                                             |

# **3.4 Exploratory Data Analysis and Data Cleansing**

_Contribution scope: author's pipeline implementation (preprocess_cicids2017_distrinet.py)._

## **3.4.1 Handling of Non-Finite and Missing Values**

A detailed scan across all 2,100,814 raw records identified non-finite entries concentrated within derived rate and inter-arrival-time metrics: Flow Bytes/s contained 602 NaN and 191 ±∞ entries; Flow Packets/s contained 793 ±∞ entries; and each of Flow IAT Mean, Std, Max, and Min contained 252 NaN entries. These anomalies occur when zero-duration flows (Duration = 0) cause division-by-zero singularities. Rather than imputing artificial synthetic medians that would distort physical flow correlations, all 793 non-finite rows were removed.

## **3.4.2 Physical Invariant Verification (Negative Values)**

Network metrics such as duration, packet counts, byte rates, and inter-arrival times are strictly non-negative. An inspection revealed 2,922 additional records containing negative timing values: Flow IAT Min exhibited negative values (e.g., −1 µs, −12 µs) attributable to clock-synchronization drift during packet capture, and 31 of these records also exhibited negative Flow Duration and Flow Packets/s. All 2,922 physically invalid rows were pruned.

## **3.4.3 Flow ID Multiplicity and Record Identity**

In standard network terminology, a 5-tuple (Src IP, Dst IP, Src Port, Dst Port, Protocol) defines a flow. However, across the multi-day captures, the Flow ID string is not a unique record identifier: only 1,085,372 distinct Flow ID strings appear across 2,100,814 records, with 1,015,442 repeated instances and a maximum multiplicity of 1,184 for persistent service endpoints. Consequently, Flow ID is strictly prohibited as a join or deduplication key. Instead, deterministic provenance is maintained via a composite key:

_Sample ID = SourceFile : RowIndex_ (3.2)

## **3.4.4 Precision-Aware Global Duplicate Removal**

Duplicate records in tabular NIDS benchmarks cause severe optimistic performance bias when the same feature vector appears in both training and test partitions. The pipeline enforces deduplication on the exact 79-dimensional float32 feature vector concatenated with the target category:

_Deduplication Key = ( x ∈ ℝ⁷⁹ float32 , y_cat )_ (3.3)

Casting to float32 prior to deduplication is mandatory: high-precision ASCII float strings that differ beyond single-precision float limits would otherwise escape pre-split deduplication and collide only after conversion. As reported in Table 3.3, 15,754 exact duplicate rows (0.7516%) were eliminated, retaining the earliest chronological occurrence.

**_Table 3.3 — Data Cleansing and Attrition Breakdown_**

| **Processing Stage**           | **Rows Remaining** | **Rows Removed** | **Description of Attrition**                                     |
| ------------------------------ | ------------------ | ---------------- | ---------------------------------------------------------------- |
| Raw DistriNet Ingestion        | 2,100,814          | 0                | Unfiltered raw capture across 5 files                            |
| Non-Finite Pruning             | 2,100,021          | 793              | Division-by-zero NaN / ±∞ values in rates and IAT                |
| Physical Non-Negativity Filter | 2,097,099          | 2,922            | Negative inter-arrival times (< 0 µs) from clock drift           |
| Unsupported Category Pruning   | 2,096,133          | 966              | Pruning ultra-rare / degenerate classes (Heartbleed, SQLi, etc.) |
| Global Float32 Deduplication   | 2,080,379          | 15,754           | Exact identical (x float32, y_cat) records                       |
| Final Curated Benchmark        | 2,080,379          | 20,435 total     | Overall attrition rate = 0.9727%                                 |

# **3.5 Taxonomy and Class Consolidation**

_Contribution scope: author's pipeline design (consolidation and dual-head mapping)._

The raw DistriNet dataset contains 25 distinct label strings. To support both coarse-grained anomaly detection and fine-grained threat taxonomy, labels are consolidated into a standardized five-category schema alongside a corresponding binary detection schema.

## **3.5.1 The 5-Category Consolidation Mapping**

- Benign: standard traffic, clean administrative sessions, and Attempted flows.
- DoS (Denial of Service): DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest.
- DDoS (Distributed Denial of Service): DDoS flood attacks.
- Recon (Reconnaissance & Probing): PortScan.
- BruteForce: FTP-Patator, SSH-Patator.

## **3.5.2 Exclusion of Unsupported Classes**

A total of 966 clean records (0.046%) belonging to extremely rare or structurally degenerate classes were excluded: Heartbleed (11 rows), Web Attack – SQL Injection (12 rows), Web Attack – XSS (27 rows), Infiltration (32 rows), Web Attack – Brute Force (151 rows), and Bot (738 rows). In a 70/15/15 chronological split, a class with 11 total instances yields fewer than 2 validation and 2 test samples — sample sizes that render empirical precision/recall metrics statistically meaningless and prevent gradient convergence during deep neural network optimization.

**_Table 3.4 — Final Label Mapping and Dual-Head Target Encoding_**

| **Raw Label String**                            | **Raw Count** | **Consolidated Category** | **5-Class ID** | **Binary ID** |
| ----------------------------------------------- | ------------- | ------------------------- | -------------- | ------------- |
| BENIGN + all \*-Attempted                       | 1,666,837     | Benign                    | 0              | 0             |
| DoS Hulk / GoldenEye / slowloris / Slowhttptest | 171,779       | DoS                       | 1              | 1             |
| DDoS                                            | 95,123        | DDoS                      | 2              | 1             |
| PortScan                                        | 159,151       | Recon                     | 3              | 1             |
| FTP-Patator, SSH-Patator                        | 6,953         | BruteForce                | 4              | 1             |

# **3.6 Split Design: Leakage-Controlled Chronological Partitioning**

_Contribution scope: author's algorithmic design and implementation._

## **3.6.1 Limitations of Alternative Splitting Strategies**

1. The flaw of random splitting: in network intrusion datasets, network sessions from the same attack-tool invocation exhibit strong temporal autocorrelation. A random stratified shuffle scatters packets from the same physical TCP session across both training and test folds, producing artificial metric inflation and enabling models to achieve near-100% accuracy through session memorization rather than the learning of generalized threat signatures.
2. The infeasibility of pure day-by-day splitting: in the UNB capture schedule, attack families were executed on distinct days (Tuesday = Patator, Wednesday = DoS, Thursday = Web Attacks, Friday = DDoS/PortScan). Partitioning Monday–Wednesday for training, Thursday for validation, and Friday for testing creates an open-set scenario in which Friday's attacks (DDoS, PortScan) were never observed during training. While valuable for zero-day generalization research, this design violates the closed-set assumption required for training supervised multi-class NIDS discriminators and generative VAEs.

## **3.6.2 The Per-Category Chronological Partitioning Protocol**

To guarantee that every evaluated attack class is represented in training while strictly preventing temporal look-ahead leakage, the pipeline applies within-category chronological partitioning: each category's records are sorted by ascending timestamp, and the split boundaries are applied independently within each category before the folds are recombined.

1. Allocate indices using exact 70% train / 15% validation / 15% test proportions via the largest-remainder apportionment algorithm.
2. Assign the earliest N_train samples of each category to the training fold.
3. Assign the subsequent N_val samples to the validation fold.
4. Assign the final N_test samples to the test fold.
5. Concatenate the respective category folds to construct the global training, validation, and test sets.

**_Table 3.5 — Final Sample Counts across Splits and Categories (Binary split: Benign/Attack — Train 1,153,431/302,833; Val 247,164/64,894; Test 247,164/64,893)_**

| **Category** | **Train (70%)** | **Validation (15%)** | **Test (15%)** | **Total Cleaned Records** | **Class Share (%)** |
| ------------ | --------------- | -------------------- | -------------- | ------------------------- | ------------------- |
| Benign       | 1,153,431       | 247,164              | 247,164        | 1,647,759                 | 79.2047             |
| DoS          | 120,091         | 25,734               | 25,734         | 171,559                   | 8.2465              |
| DDoS         | 66,568          | 14,265               | 14,265         | 95,098                    | 4.5712              |
| Recon        | 111,311         | 23,853               | 23,852         | 159,016                   | 7.6436              |
| BruteForce   | 4,863           | 1,042                | 1,042          | 6,947                     | 0.3339              |
| Total (X)    | 1,456,264       | 312,058              | 312,057        | 2,080,379                 | 100.0000            |

# **3.7 Feature Normalization: Train-Fitted Robust Scaling**

_Contribution scope: author's pipeline implementation (fit_and_apply_scaler)._

## **3.7.1 Distributional Skew and Justification of RobustScaler**

Network traffic measurements exhibit extreme heavy-tailed distributions. As shown in Table 3.6, metrics such as Flow Duration, Flow Bytes/s, and Total Length of Bwd Packet span up to eight orders of magnitude between their median and maximum values.

**_Table 3.6 — Selected Raw Feature Quantiles Illustrating Heavy-Tailed Skewness_**

| **Feature Name**               | **Median (Q2)** | **95th Percentile** | **99th Percentile** | **Maximum Observed** |
| ------------------------------ | --------------- | ------------------- | ------------------- | -------------------- |
| Flow Duration (µs)             | 61,815          | 115,308,900         | 118,104,000         | 119,999,998          |
| Total Length of Fwd Packet (B) | 80              | 1,992               | 7,024               | 12,870,252           |
| Total Length of Bwd Packet (B) | 218             | 11,595              | 101,535             | 655,452,323          |
| Flow Bytes/s                   | 3,867.93        | 1,666,667           | 2,673,077           | 600,000,000          |
| Flow Packets/s                 | 73.74           | 46,511.63           | 500,000             | 3,000,000            |
| Active Mean (µs)               | 0               | 474,229             | 3,001,590           | 110,097,488          |
| Idle Mean (µs)                 | 0               | 19,250,850          | 68,883,810          | 119,999,735          |

Under such skewness, standard z-score normalization is corrupted because extreme outliers inflate the empirical mean and variance, compressing over 95% of typical network flows into near-zero values. Min-max scaling likewise causes catastrophic collapse due to single anomalous maximum spikes. The pipeline therefore applies Robust Scaling, standardizing each feature j via its empirical median and interquartile range (IQR), computed exclusively on the training partition:

_x̃(i,j) = ( x(i,j) − Median_j(X_train) ) / IQR_j(X_train), where IQR_j = Q3(X_train,j) − Q1(X_train,j)_ (3.4)

## **3.7.2 Strict Train-Only Fitting Protocol**

To ensure zero statistical data leakage, the scaling protocol enforces the following:

1. The median and IQR vectors are computed exclusively from the 1,456,264 training records.
2. Validation and test sets are transformed using the fixed training parameters (never refit on validation or test data).
3. Train-constant feature handling: three TCP flag features (Fwd URG Flags, Bwd URG Flags, URG Flag Count) are constant zero throughout the training set (IQR = 0). The scaler assigns unit scaling (scale = 1.0) to these features to prevent division-by-zero errors while preserving column definitions for model compatibility.
4. The fitted scaler is serialized to scaler.pkl to allow exact numerical inversion during adversarial physical-validity audits.

# **3.8 Class Imbalance Mitigation: Effective Number of Samples**

_Contribution scope: author's training formulation (cicids2017d_experiments.py)._

Because the Benign category comprises 79.2% of the dataset while the BruteForce class accounts for only 0.33%, standard cross-entropy loss leads to majority-class over-indexing. To counteract this, the training engine applies the Class-Balanced Loss formulation based on the Effective Number of Samples (Cui et al., 2019). The effective volume E_n occupied by n training instances is defined as:

_E_n = (1 − βⁿ) / (1 − β), with hyperparameter β = 0.999_ (3.5)

The cost-sensitive loss weight w_c for class c is given by:

_w_c = 1 / E_(n_c) = (1 − β) / (1 − β^(n_c))_ (3.6)

**_Table 3.7 — Training Counts and Computed Effective-Number Class Weights_**

| **Target Head** | **Class Label** | **Training Samples** | **Effective Number** | **Loss Weight** |
| --------------- | --------------- | -------------------- | -------------------- | --------------- |
| Binary          | 0 (Benign)      | 1,153,431            | 1,000.00             | 0.0010000000    |
| Binary          | 1 (Attack)      | 302,833              | 1,000.00             | 0.0010000000    |
| 5-Category      | 0 (Benign)      | 1,153,431            | 1,000.00             | 0.0010000000    |
| 5-Category      | 1 (DoS)         | 120,091              | 1,000.00             | 0.0010000000    |
| 5-Category      | 2 (DDoS)        | 66,568               | 1,000.00             | 0.0010000000    |
| 5-Category      | 3 (Recon)       | 111,311              | 1,000.00             | 0.0010000000    |
| 5-Category      | 4 (BruteForce)  | 4,863                | 992.29               | 0.0010077683    |

These weights are applied directly to the weighted cross-entropy loss:

_L_CE(ŷ, y) = − Σ_c w_c · y_c · log(ŷ_c)_ (3.7)

# **3.9 Data Leakage Verification and Partition Auditing**

_Contribution scope: author's integrity and audit test suite (run_leakage_and_chronology_audit)._

Following split generation, an automated verification suite executed comprehensive integrity checks across all partitions.

## **3.9.1 Record Identity Disjointness**

_I_train ∩ I_val = ∅, I_train ∩ I_test = ∅, I_val ∩ I_test = ∅_ (3.8)

Verification confirmed zero shared sample IDs across all pairs of splits.

## **3.9.2 Exact Feature-Label Collision Audit**

_{(x, y_cat) ∈ D_train} ∩ {(x, y_cat) ∈ D_val} = ∅ (0 collisions)_ (3.9a)

_{(x, y_cat) ∈ D_train} ∩ {(x, y_cat) ∈ D_test} = ∅ (0 collisions)_ (3.9b)

## **3.9.3 Strict Temporal Monotonicity**

_max t(D_c,train) ≤ min t(D_c,val) and max t(D_c,val) ≤ min t(D_c,test) ∀ c_ (3.10)

All category partitions satisfied temporal monotonicity, with no temporal inversions detected.

## **3.9.4 Label Ambiguity Collision Analysis**

The leakage audit detected exactly one identical 79-dimensional feature vector occurring under conflicting category labels: Record A (validation), Thursday-WorkingHours.csv row 230785, timestamp 2017-07-06T20:27:47Z, category Benign; and Record B (test), Friday-WorkingHours.csv row 321856, timestamp 2017-07-07T20:08:44Z, category Recon. Because these records originate from different capture days, have distinct sample provenance, and represent legitimate feature ambiguity (a single-packet exchange matching baseline reconnaissance traffic patterns), neither record was removed. Aside from this single, explainable instance, zero feature-only or feature-plus-label collisions exist between the training set and either evaluation fold.

# **3.10 Generated Output Artifacts**

The final output directory data/processed/CICIDS_2017_Distrinet/ contains the standardized, production-ready dataset artifacts summarized in Table 3.8.

**_Table 3.8 — Summary of Preprocessed Dataset Artifacts_**

| **Artifact**                   | **Data Type / Structure** | **Dimensions**           | **Role**                                                                        |
| ------------------------------ | ------------------------- | ------------------------ | ------------------------------------------------------------------------------- |
| train/val/test.parquet         | Apache Parquet (ZSTD)     | 1.45M / 312k / 312k rows | Complete tables including 79 pristine features, timestamps, and full provenance |
| X_train/val/test.npy           | float32 NumPy array       | (N, 79)                  | RobustScaled feature matrices for neural classifier and VAE training            |
| X_\*\_pristine.npy             | float32 NumPy array       | (N, 79)                  | Unscaled physical-unit feature matrices for domain and physics-rule validation  |
| y_\*\_cat.npy                  | int64 NumPy array         | (N,)                     | Consolidated 5-category ground-truth labels (0–4)                               |
| y_\*\_bin.npy                  | int64 NumPy array         | (N,)                     | Binary detection labels (0 = Benign, 1 = Attack)                                |
| timestamp_epoch_seconds_\*.npy | int64 NumPy array         | (N,)                     | Aligned Unix timestamps in integer seconds                                      |
| scaler.pkl                     | Scikit-Learn object       | serialized               | Training-fitted RobustScaler containing center and IQR vectors                  |
| preprocessing_manifest.json    | JSON audit record         | key-value mapping        | Cryptographic hashes, exact schema, feature order, and execution metadata       |

# **3.11 Chapter Summary and Methodological Scope**

This chapter presented the comprehensive data-engineering protocol applied to the corrected DistriNet release of the CIC-IDS-2017 benchmark. By executing rigorous deterministic data cleaning — eliminating 20,435 invalid, non-finite, or duplicate samples — establishing a 79-feature normalized schema, enforcing within-category chronological partitioning, and isolating all learned scaling transformations to the training set, this pipeline establishes a scientifically sound, leakage-controlled experimental foundation for the classification and generative modeling experiments presented in subsequent chapters.

## **Methodological Scope and Limitations**

1. Within-campaign evaluation: because CIC-IDS-2017 executed each specific attack tool during a single designated time window, chronological partitioning isolates early phases of an attack campaign for training and later phases for testing. This design evaluates robust detection against evolving campaign flows but does not claim zero-day generalization across entirely unseen attack families.
2. DoS Hulk implementation anomaly: in accordance with the DistriNet audit, the DoS Hulk traffic in CIC-IDS-2017 contains a script misconfiguration (Connection: close). While this traffic exhibits unmistakable statistical anomalies that the models learn to identify, evasion results against this specific class should be interpreted with this artifact in mind.

# **3.12 Summary of Preprocessing Steps, With Ownership Demarcation**

## **Phase 1 — Upstream Dataset Correction (DistriNet)**

1. Raw packet flow re-extraction: re-running patched CICFlowMeter on the original UNB PCAP files.
2. TCP session termination fix: enforcing complete four-way FIN and valid RST connection teardown.
3. Flow direction and header repair: rectifying reversed flow directions and eliminating duplicate header-length columns.
4. Attack payload matching and re-labeling: replacing coarse time/IP-based labeling with explicit payload/schedule matching.
5. Attempted-flow classification: tagging incomplete or unresponsive attack flows with the "- Attempted" suffix.
6. Raw CSV publication: exporting five unpartitioned, unscaled daily CSV files (84 columns, 2,100,814 rows).

## **Phase 2 — Machine Learning Preprocessing and Leakage Control (Author)**

1. Cryptographic checksum verification: enforcing SHA-256 hash validation across all five ingested CSV files.
2. Schema validation and column-header normalization: verifying the 84-column structure and stripping naming artifacts.
3. Metadata and feature decoupling: extracting five metadata fields into a provenance sidecar and isolating the 79 numeric model features.
4. Sample provenance ID generation: creating deterministic tracking keys (source_file:source_row).
5. Timestamp parsing and conversion: converting string dates to UTC epoch integer seconds for temporal ordering.
6. Attempted-attack flow remapping: mapping \*-Attempted flows to Benign while preserving provenance.
7. Non-finite value pruning: eliminating 793 rows containing division-by-zero NaN / ±∞ singularities.
8. Physical invariant enforcement: pruning 2,922 rows with negative durations and inter-arrival times.
9. Taxonomy consolidation and minority pruning: mapping 25 strings to 5 macro-classes and discarding 966 degenerate-class rows.
10. Canonical float32 type casting: enforcing single-precision floating-point representation prior to deduplication.
11. Global precision-aware deduplication: removing 15,754 exact duplicate (x_float32, y_cat) records across all five days.
12. Class-conditional chronological sorting: sorting records chronologically within each category with deterministic tie-breaking.
13. Deterministic 70/15/15 split apportionment: constructing train, validation, and test folds via largest-remainder apportionment.
14. Dual-head target encoding: generating aligned binary (y_bin ∈ {0,1}) and 5-category (y_cat ∈ {0–4}) ground-truth arrays.
15. Train-only robust scaling: estimating median and IQR statistics exclusively on the 1,456,264 training records.
16. Zero-leakage validation/test transformation: transforming validation and test partitions using frozen training parameters.
17. Train-constant feature handling: assigning unit scale to zero-variance TCP flags (e.g., Fwd URG Flags).
18. Class-balanced loss weight formulation: computing Cui et al. effective-number weights (β = 0.999) to counter class imbalance.
19. Multi-format serialization: exporting compressed Parquet tables, scaled .npy arrays, pristine physical-unit .npy arrays, and the scaler pickle.
20. Automated partition leakage auditing: running formal assertion tests confirming zero ID overlap, zero cross-split duplicates, and strict timestamp monotonicity.