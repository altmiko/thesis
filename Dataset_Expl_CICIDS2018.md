**Chapter 3**

**Data Preprocessing and Dataset Engineering for CSE-CIC-IDS-2018**

# **3.1 Chapter Overview and Pipeline Demarcation**

CSE-CIC-IDS-2018 is the secondary benchmark of this thesis. It is roughly thirty times larger than CIC-IDS-2017 (63.2 million flow records, 36 GB of CSV) and exhibits an even more extreme class prior (≈94.6% Benign after cleaning). Preprocessing it with the CIC-IDS-2017 protocol unchanged is not possible: the release carries additional columns with sentinel semantics, a flow-meter integer-overflow artefact, sub-second timestamps whose file order is not chronological, and attack classes that differ by five orders of magnitude in size. This chapter documents, in full methodological detail, the data engineering pipeline that transforms the corrected DistriNet CSE-CIC-IDS-2018 CSV files into the leakage-controlled, normalized, and class-controlled dataset used for the secondary experiments of this thesis. Wherever possible, the pipeline reuses the CIC-IDS-2017 decisions (feature schema, category taxonomy, split protocol, scaler) so that both datasets share one processed layout.

## **3.1.1 Division of Responsibilities: DistriNet versus This Thesis Pipeline**

As for CIC-IDS-2017, this chapter demarcates (i) the upstream, network-layer re-extraction and relabelling of the dataset performed by DistriNet (Liu et al., 2022) from (ii) the downstream, machine-learning-specific preprocessing pipeline designed and implemented by the author. The former produces corrected flow records and attack-specific labels; the latter performs data hygiene, leakage-controlled partitioning, controlled class-size reduction, scaling, and class-imbalance handling. Table 3.0 summarizes this division of responsibility.

**_Table 3.0 — Pipeline Responsibility Matrix: DistriNet versus Author's Implementation_**

| **Preprocessing Stage**                                   | **DistriNet** | **Author (This Thesis)** | **Implementation Reference**                                             |
| --------------------------------------------------------- | ------------- | ------------------------ | ------------------------------------------------------------------------ |
| Raw PCAP parsing & corrected flow re-extraction           | Yes           | No                       | DistriNet CNS 2022 corrected/relabelled release                          |
| Attack-specific relabelling rules                         | Yes           | No                       | DistriNet CSE-CIC-IDS-2018 documentation                                 |
| Tagging incomplete attacks ("- Attempted") + category code | Yes           | No                       | DistriNet CSV export (`Label`, `Attempted Category`)                      |
| SHA-256 recording of raw files                            | No            | Yes                      | preprocess_cicids2018_distrinet.py – scan_file                           |
| File-set and 91-column header validation                  | No            | Yes                      | preprocess_cicids2018_distrinet.py – validate_inventory                  |
| Decoupling metadata and 2018-only columns from 79 features | No            | Yes                      | preprocess_cicids2018_distrinet.py – load_reference_features, scan_file  |
| Attempted-flow policy (Attempted → Benign)                | No            | Yes                      | preprocess_cicids2018_distrinet.py – _slot_tables                        |
| Non-finite filtering (57 rows)                            | No            | Yes                      | preprocess_cicids2018_distrinet.py – scan_file                           |
| Header-length overflow flagging (rows kept)               | No            | Yes                      | preprocess_cicids2018_distrinet.py – scan_file, header_flag_columns      |
| Multi-class taxonomy mapping (25 → 5)                     | No            | Yes                      | preprocess_cicids2018_distrinet.py – _slot_tables, _label_slots          |
| Global float32 deduplication (657,839 rows)               | No            | Yes                      | preprocess_cicids2018_distrinet.py – canonical_float32, deduplicate      |
| Chronological 70/15/15 split within source label          | No            | Yes                      | preprocess_cicids2018_distrinet.py – split_within_source_label           |
| Per-split time-stratified class-size reduction            | No            | Yes                      | preprocess_cicids2018_distrinet.py – allocate_stratum_quotas, time_stratified_sample |
| Train-only RobustScaler fitting                           | No            | Yes                      | preprocess_cicids2018_distrinet.py – main                                |
| Inverse-frequency class weights                           | No            | Yes                      | preprocess_cicids2017_distrinet.py – balanced_class_weights              |
| Export of Parquet, NumPy, scaler and row-index artifacts  | No            | Yes                      | preprocess_cicids2018_distrinet.py – write_split_parquet, write_row_index |
| Automated leakage, coverage & chronology audit            | No            | Yes                      | preprocess_cicids2018_distrinet.py – leakage_audit, assert_chronological |

**_Figure 3.1 — End-to-End Pipeline Architecture_**

The pipeline proceeds in two sequential phases.

**Phase 1 (DistriNet, upstream):** Raw CSE-CIC-IDS-2018 PCAPs (14 February – 2 March 2018) → corrected flow re-extraction → attack-specific relabelling → tagging of incomplete attacks with the "- Attempted" suffix and a numeric `Attempted Category` code → export of ten daily CSV files (91 columns, 63,195,145 rows).

**Phase 2 (author, downstream):** file-set, header, and SHA-256 recording → decoupling of seven metadata and five CICIDS2018-only columns from the 79 CIC-IDS-2017 modelling features → remapping of Attempted flows to Benign with provenance retained → removal of 57 non-finite rows → retention and flagging of 16-bit-wrapped header lengths → consolidation of 25 raw labels into 5 categories (143,493 rows of unsupported classes dropped) → global float32 deduplication (657,839 exact duplicates removed) → chronological 70/15/15 split within each of the nine retained source labels → time-stratified reduction of Benign/DoS/DDoS to fixed row targets inside each split → RobustScaler fitted strictly on the 583,487 final training rows → projection of validation/test with the frozen scaler → inverse-frequency class weights → export of Parquet tables, NumPy arrays, the fitted scaler, a full row index, and automated audits.

Engineering note: the 36 GB source does not fit in memory as float64. The pipeline therefore runs in two streaming passes. Pass 1 stores only compact keys per row (provenance, timestamp, label, header flags, and a 128-bit feature hash); deduplication, splitting, and sampling operate on those keys. Pass 2 rereads the CSVs and writes only the selected rows into float32 memory-mapped arrays, recomputing hashes to detect any pass-to-pass drift. The full production run (seed 42, 10 workers) took 368 seconds.

# **3.2 Dataset Provenance and the DistriNet Correction**

_Contribution scope: upstream network-level flow re-extraction and relabelling (Liu et al., 2022)._

## **3.2.1 Limitations of the Original CSE-CIC-IDS-2018 Release**

The Communications Security Establishment (CSE) and the Canadian Institute for Cybersecurity (CIC) generated CSE-CIC-IDS-2018 on an AWS testbed, capturing ten working days between Wednesday, 14 February 2018, and Friday, 2 March 2018. Liu et al. (2022) extended the error analysis previously applied to CIC-IDS-2017 to this dataset and published a corrected and relabelled release. Two consequences of that audit are directly visible in the corrected data and shape the taxonomy used here:

- No successful FTP brute-force attack exists. Port 21 on the victim was closed, so all 298,874 FTP rows are labelled `FTP-BruteForce - Attempted`; DistriNet reports that the 16 February "FTP" traffic was most likely a misfired DoS Slowhttptest. Consequently, neither a DoS Slowhttptest nor a Heartleech label exists in the release.
- Labels are assigned by attack-specific rules rather than by time window and IP address alone (for example, DistriNet's GoldenEye rules use the per-direction RST flag counts that the release exports as extra columns).

## **3.2.2 The DistriNet Re-Extraction Release**

This research adopts the ten-file DistriNet CSE-CIC-IDS-2018 release. All ten files share one identical 91-column header, and the `id` column equals the 1-based row number in every file, providing a stable per-file row key. Timestamps carry microsecond precision on every row and are UTC according to DistriNet's documentation. Table 3.1 lists the raw input files used as the immutable foundation of the pipeline.

**_Table 3.1 — Raw Input Inventory (SHA-256 checksums recorded in the provenance manifest; omitted here for brevity)_**

| **File Name**             | **Size (Bytes)** | **Raw Rows** | **First – Last Timestamp (UTC)**                 |
| ------------------------- | ---------------- | ------------ | ------------------------------------------------ |
| Wednesday-14-02-2018.csv  | 3,262,583,806    | 5,898,350    | 2018-02-14 12:28:07 – 2018-02-15 00:47:34        |
| Thursday-15-02-2018.csv   | 2,992,535,617    | 5,410,102    | 2018-02-15 12:23:15 – 2018-02-16 01:14:55        |
| Friday-16-02-2018.csv     | 4,211,741,805    | 7,390,266    | 2018-02-16 12:26:55 – 2018-02-16 22:15:24        |
| Tuesday-20-02-2018.csv    | 3,431,246,090    | 6,054,702    | 2018-02-20 12:28:03 – 2018-02-21 00:37:22        |
| Wednesday-21-02-2018.csv  | 3,952,691,068    | 6,962,593    | 2018-02-21 12:28:18 – 2018-02-22 00:36:49        |
| Thursday-22-02-2018.csv   | 3,473,832,504    | 6,071,153    | 2018-02-22 12:22:02 – 2018-02-23 00:36:53        |
| Friday-23-02-2018.csv     | 3,409,376,619    | 5,976,481    | 2018-02-21 12:33:55 – 2018-02-23 23:46:49        |
| Wednesday-28-02-2018.csv  | 3,812,223,732    | 6,568,726    | 2018-02-28 12:20:49 – 2018-03-01 01:05:48        |
| Thursday-01-03-2018.csv   | 3,808,397,458    | 6,551,401    | 2018-03-01 12:15:28 – 2018-03-01 23:40:14        |
| Friday-02-03-2018.csv     | 3,689,837,784    | 6,311,371    | 2018-03-02 12:46:22 – 2018-03-03 00:39:53        |
| Total Pipeline Foundation | 36,044,466,483   | 63,195,145   | 10 capture days                                  |

The file name is not a reliable date: seven files run past midnight UTC, 11,693 rows fall on a date other than the one in their file name, and Friday-23-02-2018.csv begins on 21 February with a low-rate Benign trickle (2,609 rows) that precedes its own capture day. Within files, rows are not in time order (27,820,016 adjacent timestamp reversals in total). Chronology is therefore always derived from the parsed timestamp (Section 3.4.4), never from the file name or row order.

## **3.2.3 Treatment of Attempted Flows**

As in CIC-IDS-2017, the DistriNet tooling appends a "- Attempted" suffix to flows in which an attack was launched but no malicious exchange took place. Unlike the 2017 export, the 2018 release also carries an `Attempted Category` code, which is −1 on every non-Attempted row and 0–6 on Attempted rows (Table 3.2a). DistriNet's guidance is that Attempted flows must not be modelled as attacks. The pipeline applies the same deterministic policy as for CIC-IDS-2017:

_Label(raw) = "&lt;Attack&gt; - Attempted" ⇒ Source label = BENIGN ⇒ Category = Benign_ (3.1)

Across the 63,195,145 raw records, 306,237 rows (0.4846%) carry one of ten distinct Attempted labels; 298,874 of them are `FTP-BruteForce - Attempted`. The raw exploratory analysis supports the policy empirically: before remapping, 53 identical feature vectors (374 rows) carried both `BENIGN` and `FTP-BruteForce - Attempted`, because closed-port SYN/RST exchanges are indistinguishable from benign ones. All Attempted rows keep `original_label`, `Attempted Category`, and an `is_attempted` flag as metadata; none of these enter the feature matrix.

**_Table 3.2a — DistriNet Attempted Category Codes (as defined by DistriNet; occurrences observed in this release)_**

| **Code** | **Meaning (DistriNet)**          | **Main Occurrences**                                   |
| -------- | -------------------------------- | ------------------------------------------------------ |
| 0        | No payload sent by attacker      | DoS Slowloris/Hulk, Botnet Ares, web attacks           |
| 1        | Port/system closed               | FTP-BruteForce (298,844)                               |
| 2        | Attack startup/teardown artefact | Botnet Ares, web attacks                               |
| 3        | No malicious payload             | Web Attack – XSS                                       |
| 4        | Attack artefact                  | DoS GoldenEye (4,248), FTP-BruteForce (30), Dropbox    |
| 5        | Attack implemented incorrectly   | Web Attack – Brute Force (126)                         |
| 6        | Target system unresponsive       | DDoS-LOIC-UDP (251), DoS GoldenEye (53)                |

# **3.3 Feature Schema and Representation**

_Contribution scope: author's pipeline design (feature isolation and cross-dataset schema alignment)._

The raw CSV header comprises 91 columns: 7 non-feature columns and 84 numeric candidates. All 79 CIC-IDS-2017 modelling features are present in the same relative order. The pipeline reads the feature names and order from the CIC-IDS-2017 preprocessing manifest and aborts if any of the 79 names is missing or out of order, so that both datasets share one feature contract and one `FeatureManifest` interface. The columns are partitioned as follows:

1. Metadata and identity columns, excluded from X: `id`, Flow ID, Src IP, Dst IP, Timestamp, Label, and Attempted Category. They are retained in a provenance sidecar together with the source file, the original CSV row, the parsed UTC timestamp, and the mapped labels.
2. CICIDS2018-only columns, excluded from X: Fwd RST Flags, Bwd RST Flags, ICMP Code, ICMP Type, and Total TCP Flow Time. ICMP Code/Type hold a −1 sentinel on every non-ICMP row (63,089,142 rows) and Total TCP Flow Time is 0 on every non-TCP row; excluding these columns keeps protocol-specific sentinel semantics out of the modelling schema and preserves comparability with CIC-IDS-2017.
3. Feature matrix, X ∈ ℝ^(N×79): the 79 CIC-IDS-2017 features, parsed as numeric, checked for finiteness, cast to float32, and stored in the CIC-IDS-2017 order.

**_Table 3.2 — The 79 Canonical Modeling Features (Ordered Representation, identical to CIC-IDS-2017)_**

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

_Contribution scope: author's pipeline implementation (preprocess_cicids2018_distrinet.py)._

All cleaning rules are deterministic and row-local; none uses a fitted statistic. No imputation, winsorization, or clipping is applied.

## **3.4.1 Handling of Non-Finite and Missing Values**

The release contains no nulls or NaNs. Exactly 57 rows contain +∞ in both Flow Bytes/s and Flow Packets/s; all are Benign zero-duration flows, for which the rate features divide by a zero duration. Consistent with the CIC-IDS-2017 policy, these 57 rows are removed rather than imputed. No row has a missing or unparseable timestamp.

## **3.4.2 Physical Invariant Verification and the Header-Length Artefact**

The CIC-IDS-2017 rule "drop any row with a negative value" cannot be reused unchanged. Applied to all 84 numeric columns it would delete 99.8% of the release because of the ICMP −1 sentinel; with the ICMP columns excluded from X (Section 3.3), the remaining negative values lie exclusively in Fwd Header Length and Bwd Header Length. The pipeline therefore enforces non-negativity on every modelling feature except these two columns; this rule removed 0 rows.

The two header-length columns span exactly [−32,768, 32,764], the range of a signed 16-bit integer: the flow meter stores header bytes in a 16-bit field that wraps on long flows. Because the number of wraps is unknown, the true value cannot be recovered, and the sign is not a validity test — wrapped values land positive about as often as negative. Dropping negative rows would remove 1,209 of the 2,527 DDoS-LOIC-UDP rows (47.8% of that source label) while still leaving 1,223 wrapped positive values in place. The pipeline therefore keeps the rows and raw values and records four metadata flags (never features):

- negative flag: the stored value is < 0;
- overflow flag: packets × minimum transport header (TCP 20 B, UDP 8 B) exceeds 32,767, so the true header byte count cannot be represented.

**_Table 3.3a — Header-Length Artefact Flags (rows kept)_**

| **Scope**                       | **Negative (any direction)** | **Overflow (any direction)** |
| ------------------------------- | ---------------------------- | ---------------------------- |
| Benign (retained rows)          | 22,483                       | 27,557                       |
| DDoS (retained rows)            | 1,209                        | 2,432                        |
| DoS, Recon, BruteForce          | 0                            | 0                            |
| Final train / val / test splits | 200 / 46 / 47                | 334 / 70 / 77                |

## **3.4.3 Flow ID Multiplicity and Record Identity**

The Flow ID string is a textual 5-tuple, not a record identifier: only 35,160,510 distinct values appear across 63,195,145 rows, 28,034,635 rows repeat an earlier Flow ID, the maximum multiplicity is 1,231, and 6,249,883 Flow IDs occur in two or more files. Flow ID is therefore never used as a join or deduplication key. Deterministic provenance is maintained via the composite key:

_Sample ID = SourceFile : id_ (3.2)

where `id` is the 1-based row number within the source file.

## **3.4.4 Timestamp Parsing and Chronological Ordering**

Timestamps are ISO-8601 strings (`YYYY-MM-DD HH:MM:SS.ffffff`) without an explicit time zone and are interpreted as UTC per DistriNet's documentation. They are parsed to integer epoch microseconds, and every row is ordered by the key

_Chronological order = ( timestamp_us , source-file order , id )_ (3.3)

where the file and `id` components are deterministic tie-breakers only. Timestamps are used for ordering and stratification; they are never a classifier input.

## **3.4.5 Precision-Aware Global Duplicate Removal**

Deduplication is global, after cleaning and label mapping and before splitting, on the exact float32 feature vector concatenated with the mapped category:

_Deduplication Key = ( x ∈ ℝ⁷⁹ float32 , y_cat )_ (3.4)

Before hashing, −0.0 is folded into +0.0 so that bit-pattern equality coincides with value equality. The key is implemented as a 128-bit hash (two independent 64-bit mixers over the float32 bit patterns; expected false merges ≈ n²/2¹²⁹, negligible); the hash is re-checked against the saved arrays in Pass 2, and shared fingerprints are compared byte-exactly. The earliest occurrence under order (3.3) survives. As reported in Table 3.3, 657,839 exact duplicates (1.0433%) were removed: 657,351 Benign (445,713 `BENIGN` and 211,638 `FTP-BruteForce - Attempted`), 469 DoS (all DoS Slowloris), 19 Recon, and none in DDoS or BruteForce. For 459,946 of the removed rows the surviving copy lies in a different file.

**_Table 3.3 — Data Cleansing and Attrition Breakdown_**

| **Processing Stage**                     | **Rows Remaining** | **Rows Removed** | **Description of Attrition**                                          |
| ---------------------------------------- | ------------------ | ---------------- | --------------------------------------------------------------------- |
| Raw DistriNet Ingestion                  | 63,195,145         | 0                | Unfiltered raw capture across 10 files                                |
| Non-Finite Pruning                       | 63,195,088         | 57               | +∞ rates from zero-duration Benign flows                              |
| Timestamp Validity                       | 63,195,088         | 0                | No missing or unparseable timestamps                                  |
| Non-Negativity (excl. header lengths)    | 63,195,088         | 0                | No negative values outside the flagged header-length columns          |
| Unsupported Category Pruning             | 63,051,595         | 143,493          | Botnet Ares, web attacks, Infiltration sub-attacks other than NMAP    |
| Global Float32 Deduplication             | 62,393,756         | 657,839          | Exact identical (x float32, y_cat) records                            |
| Curated Benchmark (before reduction)     | 62,393,756         | 801,389 total    | Cleaning + deduplication attrition = 1.2681%                          |
| Controlled Class-Size Reduction (§3.7)   | 833,552            | 61,560,204       | Benign/DoS/DDoS undersampled to fixed targets; design choice, not cleaning |

# **3.5 Taxonomy and Class Consolidation**

_Contribution scope: author's pipeline design (consolidation and dual-head mapping)._

The raw release contains 25 distinct label strings (15 attack or Benign labels and 10 Attempted variants). They are consolidated into the same five-category schema and binary schema used for CIC-IDS-2017, so that classifiers, VAEs, and attacks consume identical class ids on both datasets. Unknown labels abort the run rather than falling into an implicit bucket.

## **3.5.1 The 5-Category Consolidation Mapping**

- Benign: `BENIGN` and every label ending in "- Attempted".
- DoS (Denial of Service): DoS Hulk, DoS GoldenEye, DoS Slowloris.
- DDoS (Distributed Denial of Service): DDoS-HOIC, DDoS-LOIC-HTTP, DDoS-LOIC-UDP.
- Recon (Reconnaissance & Probing): Infiltration – NMAP Portscan.
- BruteForce: SSH-BruteForce.

Two semantic differences from CIC-IDS-2017 must be kept in mind. Recon originates from the internal Infiltration campaign (attackers 172.31.69.13/.24 scanning 22 hosts), not from an external scanner as in the 2017 PortScan traffic. BruteForce is SSH-only, because the FTP brute force never succeeded (Section 3.2.1).

## **3.5.2 Exclusion of Unsupported Classes**

A total of 143,493 clean records (0.2271%) belonging to labels outside the five-category schema were excluded: Botnet Ares (142,921 rows), Infiltration – Communication Victim Attacker (204), Web Attack – Brute Force (131), Web Attack – XSS (113), Infiltration – Dropbox Download (85), and Web Attack – SQL (39). The web-attack and Infiltration sub-attack labels have too few rows to support a chronological 70/15/15 split with stable estimates. Botnet Ares is large but has no counterpart in the shared taxonomy; it is excluded, as Bot was for CIC-IDS-2017, to keep the two datasets on one class schema.

**_Table 3.4 — Final Label Mapping and Dual-Head Target Encoding_**

| **Raw Label String**                                 | **Raw Count** | **Consolidated Category** | **5-Class ID** | **Binary ID** |
| ---------------------------------------------------- | ------------- | ------------------------- | -------------- | ------------- |
| BENIGN (59,353,486) + all \*-Attempted (306,237)     | 59,659,723    | Benign                    | 0              | 0             |
| DoS Hulk / GoldenEye / Slowloris                     | 1,834,210     | DoS                       | 1              | 1             |
| DDoS-HOIC / LOIC-HTTP / LOIC-UDP                     | 1,374,148     | DDoS                      | 2              | 1             |
| Infiltration – NMAP Portscan                         | 89,374        | Recon                     | 3              | 1             |
| SSH-BruteForce                                       | 94,197        | BruteForce                | 4              | 1             |

The per-source raw counts are DoS Hulk 1,803,160, DoS GoldenEye 22,560, DoS Slowloris 8,490, DDoS-HOIC 1,082,293, DDoS-LOIC-HTTP 289,328, and DDoS-LOIC-UDP 2,527.

# **3.6 Split Design: Leakage-Controlled Chronological Partitioning**

_Contribution scope: author's algorithmic design and implementation._

## **3.6.1 Limitations of Alternative Splitting Strategies**

1. The flaw of random splitting: flows produced by one attack-tool invocation are strongly temporally autocorrelated. A random shuffle scatters records of the same session and campaign across folds, allowing models to score highly through memorization rather than generalization.
2. The infeasibility of pure day-by-day splitting: every CSE-CIC-IDS-2018 attack label occurs on only one or two capture days, often within windows of minutes (DoS Hulk produced 1.8 million flows in 12.9 minutes). A whole-day split would place entire attack families only in validation or test, violating the closed-set assumption of supervised multi-class NIDS classifiers and per-class generative models.

## **3.6.2 The Per-Source-Label Chronological Partitioning Protocol**

The pipeline applies the CIC-IDS-2017 protocol, chronological partitioning within each retained source label, with no shuffling before assignment. For each of the nine retained source labels (BENIGN including Attempted, DoS Hulk, DoS GoldenEye, DoS Slowloris, DDoS-HOIC, DDoS-LOIC-HTTP, DDoS-LOIC-UDP, Infiltration – NMAP Portscan, SSH-BruteForce) independently:

1. Select its cleaned, deduplicated rows and keep them in the chronological order (3.3).
2. Allocate exact 70% train / 15% validation / 15% test counts via largest-remainder apportionment, with at least one row per split.
3. Assign the earliest N_train rows to training, the subsequent N_val rows to validation, and the final N_test rows to test.
4. Merge the source labels into the five final categories.

Splitting per source label rather than per category guarantees that every DoS and DDoS subtype (for example, the 2,527-row DDoS-LOIC-UDP) is present in every split. The resulting memberships are authoritative: the class-size reduction of Section 3.7 only removes rows inside a partition and never moves a row between partitions.

**_Table 3.5 — Sample Counts after the Chronological Split, before Class-Size Reduction_**

| **Category** | **Train (70%)** | **Validation (15%)** | **Test (15%)** | **Total Cleaned Records** | **Class Share (%)** |
| ------------ | --------------- | -------------------- | -------------- | ------------------------- | ------------------- |
| Benign       | 41,301,621      | 8,850,347            | 8,850,347      | 59,002,315                | 94.5645             |
| DoS          | 1,283,619       | 275,061              | 275,061        | 1,833,741                 | 2.9390              |
| DDoS         | 961,904         | 206,122              | 206,122        | 1,374,148                 | 2.2024              |
| Recon        | 62,549          | 13,403               | 13,403         | 89,355                    | 0.1432              |
| BruteForce   | 65,938          | 14,130               | 14,129         | 94,197                    | 0.1510              |
| Total        | 43,675,631      | 9,359,063            | 9,359,062      | 62,393,756                | 100.0000            |

# **3.7 Controlled Class-Size Reduction**

_Contribution scope: author's algorithmic design and implementation (allocate_stratum_quotas, time_stratified_sample)._

The natural class prior of Table 3.5 is extreme: the largest-to-smallest class ratio is ≈660 : 1, and the 43.7 million-row training partition is impractical for repeated multi-seed victim training and attack evaluation. After the chronological split, Benign, DoS, and DDoS are therefore deterministically undersampled to fixed total row targets, while Recon and BruteForce retain every row. No observation is duplicated or synthesized, and no fitted statistic is involved.

## **3.7.1 Targets**

Each class total is apportioned 70/15/15 by the same largest-remainder rule used for the chronological split, so every split receives the same class composition.

**_Table 3.6 — Class-Size Targets per Split_**

| **Class**  | **Total Target** | **Train** | **Validation** | **Test** |
| ---------- | ---------------- | --------- | -------------- | -------- |
| Benign     | 250,000          | 175,000   | 37,500         | 37,500   |
| DoS        | 200,000          | 140,000   | 30,000         | 30,000   |
| DDoS       | 200,000          | 140,000   | 30,000         | 30,000   |
| Recon      | all rows         | 62,549    | 13,403         | 13,403   |
| BruteForce | all rows         | 65,938    | 14,130         | 14,129   |

If a split held fewer rows than its target, all rows would be kept and a `target_reached = false` flag recorded; all nine production targets were reached.

## **3.7.2 Time-Stratified Sampling Algorithm**

Within each split and each targeted class:

1. Take only that class's rows in that split.
2. Define strata as _Stratum = ( source label , source file , ⌊ timestamp_UTC / 1 hour ⌋ )_ (3.5). Including the source label keeps subtype shares (e.g. DoS Hulk vs GoldenEye vs Slowloris) proportional.
3. Allocate the split target across strata by Hamilton (largest-remainder) apportionment in proportion to stratum size.
4. If the target is at least the number of strata, any stratum rounded to zero receives one row, taken from the most over-allocated stratum. No stratum is ever asked for more rows than it holds (asserted).
5. Sample uniformly without replacement within each stratum using `default_rng([seed, split_index, class_index, source_index, file_index, time_bin])` with seed 42.
6. Return the selected rows to chronological order.

Every stratum remains represented after sampling. The minimum-one-row rule moved 17 rows in train Benign and one row each in validation Benign and test DoS.

**_Table 3.7 — Temporal Coverage of the Reduction_**

| **Split**  | **Class** | **Available** | **Selected** | **Sampling Fraction** | **Strata (all represented)** |
| ---------- | --------- | ------------- | ------------ | --------------------- | ---------------------------- |
| Train      | Benign    | 41,301,621    | 175,000      | 0.424%                | 116                          |
| Train      | DoS       | 1,283,619     | 140,000      | 10.91%                | 3                            |
| Train      | DDoS      | 961,904       | 140,000      | 14.55%                | 4                            |
| Validation | Benign    | 8,850,347     | 37,500       | 0.424%                | 19                           |
| Validation | DoS       | 275,061       | 30,000       | 10.91%                | 3                            |
| Validation | DDoS      | 206,122       | 30,000       | 14.55%                | 4                            |
| Test       | Benign    | 8,850,347     | 37,500       | 0.424%                | 20                           |
| Test       | DoS       | 275,061       | 30,000       | 10.91%                | 4                            |
| Test       | DDoS      | 206,122       | 30,000       | 14.55%                | 4                            |

## **3.7.3 Source-Label Composition and Attempted Rows after Reduction**

**_Table 3.8 — Source-Label Composition of the Final Splits_**

| **Source Label**             | **Train** | **Validation** | **Test** |
| ---------------------------- | --------- | -------------- | -------- |
| BENIGN (incl. Attempted)     | 175,000   | 37,500         | 37,500   |
| DoS Hulk                     | 137,665   | 29,500         | 29,499   |
| DoS GoldenEye                | 1,722     | 369            | 370      |
| DoS Slowloris                | 613       | 131            | 131      |
| DDoS-HOIC                    | 110,265   | 23,628         | 23,628   |
| DDoS-LOIC-HTTP               | 29,477    | 6,317          | 6,317    |
| DDoS-LOIC-UDP                | 258       | 55             | 55       |
| Infiltration – NMAP Portscan | 62,549    | 13,403         | 13,403   |
| SSH-BruteForce               | 65,938    | 14,130         | 14,129   |

Because sampling is proportional, rare subtypes keep their share of DoS/DDoS but shrink in absolute size (DDoS-LOIC-UDP: 55 validation and 55 test rows); per-subtype estimates for GoldenEye, Slowloris, and LOIC-UDP are unstable. Strata are temporal rather than per original label, so rare Attempted labels can drop to zero: of the 87,236 training `FTP-BruteForce - Attempted` rows, 353 remain; DoS Slowloris – Attempted keeps 18 of 2,280, DoS GoldenEye – Attempted 11 of 4,301, and DDoS-LOIC-UDP – Attempted 2 of 251, while the training Web Attack – Brute Force/DoS Hulk/Web Attack – SQL/Web Attack – XSS Attempted rows (137/86/14/4) all drop to zero. Validation keeps one `Infiltration - Dropbox Download - Attempted` row and test one `Botnet Ares - Attempted` row. Full provenance for every cleaned row remains in `row_index.parquet`.

**_Table 3.9 — Final Sample Counts across Splits and Categories (Binary split: Benign/Attack — Train 175,000/408,487; Val 37,500/87,533; Test 37,500/87,532)_**

| **Category** | **Train (70%)** | **Validation (15%)** | **Test (15%)** | **Total Records** | **Class Share (%)** |
| ------------ | --------------- | -------------------- | -------------- | ----------------- | ------------------- |
| Benign       | 175,000         | 37,500               | 37,500         | 250,000           | 29.9921             |
| DoS          | 140,000         | 30,000               | 30,000         | 200,000           | 23.9937             |
| DDoS         | 140,000         | 30,000               | 30,000         | 200,000           | 23.9937             |
| Recon        | 62,549          | 13,403               | 13,403         | 89,355            | 10.7198             |
| BruteForce   | 65,938          | 14,130               | 14,129         | 94,197            | 11.3007             |
| Total (X)    | 583,487         | 125,033              | 125,032        | 833,552           | 100.0000            |

The largest-to-smallest class ratio falls from ≈660 : 1 to 2.8 : 1. These partitions do not reproduce the original CSE-CIC-IDS-2018 class prior and are interpreted as controlled experimental datasets, not as estimates of real-world attack prevalence (Section 3.12).

# **3.8 Feature Normalization: Train-Fitted Robust Scaling**

_Contribution scope: author's pipeline implementation (preprocess_cicids2018_distrinet.py – main)._

## **3.8.1 Distributional Skew and Justification of RobustScaler**

CSE-CIC-IDS-2018 flow statistics are heavy-tailed. Over the full raw release, Flow Bytes/s has median 2,240, 99.9th percentile 1.04 × 10⁶, and maximum 2.92 × 10⁹; Total Length of Fwd Packet has median 97 and maximum 5.4 × 10⁷; and Flow Duration saturates at 120,000,000 µs, the CICFlowMeter 120-second flow timeout. Table 3.10 shows the same skew on the final training partition on which the scaler is fitted.

**_Table 3.10 — Raw Feature Quantiles on the Final Training Partition (583,487 rows)_**

| **Feature Name**               | **Median (Q2)** | **95th Percentile** | **99th Percentile** | **Maximum Observed** |
| ------------------------------ | --------------- | ------------------- | ------------------- | -------------------- |
| Flow Duration (µs)             | 110,973         | 60,015,044          | 117,208,242         | 120,000,000          |
| Total Length of Fwd Packet (B) | 314             | 1,944               | 2,008               | 7,437,183            |
| Total Length of Bwd Packet (B) | 935             | 2,665               | 7,051.42            | 35,023,204           |
| Flow Bytes/s                   | 10,748.40       | 370,207.90          | 582,356.10          | 32,500,000           |
| Flow Packets/s                 | 120.10          | 10,087.77           | 1,000,000           | 2,500,000            |
| Active Mean (µs)               | 0               | 1,254,242.90        | 3,940,028.70        | 111,275,304          |
| Idle Mean (µs)                 | 0               | 29,553,884.40       | 86,104,386.24       | 119,935,696          |

For the reasons given for CIC-IDS-2017, z-score and min-max scaling are dominated by these tails. The pipeline applies Robust Scaling with the per-feature training median and interquartile range (quantile range 25–75):

_x̃(i,j) = ( x(i,j) − Median_j(X_train) ) / IQR_j(X_train), where IQR_j = Q3(X_train,j) − Q1(X_train,j)_ (3.6)

## **3.8.2 Strict Train-Only Fitting Protocol**

1. The median and IQR vectors are computed exclusively from the 583,487 final (post-reduction) training rows. Validation and test rows are never used; a permanent test asserts that the saved scaler centre equals the median of the final training rows.
2. Validation and test sets are transformed with the frozen training parameters.
3. Zero-IQR feature handling: 25 of the 79 features have zero interquartile range on the training partition (among them Dst Port, Protocol, the Fwd/Bwd/Packet Length Min features, the URG flags, SYN Flag Count, all six Bulk features, Subflow Fwd/Bwd Packets, and the eight Active/Idle timers). RobustScaler assigns them unit scale (scale = 1.0), so they are centred but not divided, and every column is retained for model compatibility. Only Bwd URG Flags is constant on the training set.
4. The fitted scaler is serialized to `scaler.pkl`, with its centre and scale vectors mirrored in `scaler_parameters.json`, allowing exact inversion during validity audits.
5. As for CIC-IDS-2017, a differentiable `asinh` transform is applied inside the victims and VAEs after robust scaling; it is not applied in preprocessing and is never applied twice.

The same train-only boundary governs all downstream fitting on this dataset: feature-selection statistics, CFF masks, mined constraints, the validator training profile, the VAEs, and PrimAttack budget calibration are fitted on training rows only; validation is used for model/attack selection and calibration; test is used only for final reporting. Sampling validation/test rows (Section 3.7) is not a learned transformation: it uses only each partition's own membership, timestamps, source identifiers, and the configured seed.

# **3.9 Class Imbalance Mitigation: Inverse-Frequency Class Weights**

_Contribution scope: author's training formulation (preprocess_cicids2017_distrinet.py – balanced_class_weights; cicids2017d_experiments.py, `--class-weighting balanced`)._

After the controlled reduction, the residual imbalance is moderate (2.8 : 1). The CSE-CIC-IDS-2018 victims are trained with standard inverse-frequency ("balanced") class weights computed from the final training labels only:

_w_c = N / ( K · n_c )_ (3.7)

where N is the number of training rows, K the number of classes of the head, and n_c the training count of class c.

**_Table 3.11 — Training Counts and Class Weights_**

| **Target Head** | **Class Label** | **Training Samples** | **Loss Weight** |
| --------------- | --------------- | -------------------- | --------------- |
| Binary          | 0 (Benign)      | 175,000              | 1.6671          |
| Binary          | 1 (Attack)      | 408,487              | 0.7142          |
| 5-Category      | 0 (Benign)      | 175,000              | 0.6668          |
| 5-Category      | 1 (DoS)         | 140,000              | 0.8336          |
| 5-Category      | 2 (DDoS)        | 140,000              | 0.8336          |
| 5-Category      | 3 (Recon)       | 62,549               | 1.8657          |
| 5-Category      | 4 (BruteForce)  | 65,938               | 1.7698          |

These weights enter the weighted cross-entropy loss:

_L_CE(ŷ, y) = − Σ_c w_c · y_c · log(ŷ_c)_ (3.8)

# **3.10 Data Leakage Verification and Partition Auditing**

_Contribution scope: author's integrity and audit suite (leakage_audit, assert_chronological, and permanent tests)._

## **3.10.1 Record Identity Disjointness**

_I_train ∩ I_val = ∅, I_train ∩ I_test = ∅, I_val ∩ I_test = ∅_ (3.9)

With sample IDs defined by (3.2), verification confirmed zero shared sample IDs across all pairs of splits.

## **3.10.2 Exact Feature-Label Collision Audit**

The audit recomputes fingerprints of the saved float32 rows with an independent hash (pandas `hash_pandas_object`) and compares shared fingerprints byte-exactly:

_{(x, y_cat) ∈ D_train} ∩ {(x, y_cat) ∈ D_val} = ∅ (0 collisions)_ (3.10a)

_{(x, y_cat) ∈ D_train} ∩ {(x, y_cat) ∈ D_test} = ∅ (0 collisions)_ (3.10b)

The same holds for validation versus test, for feature-only fingerprints across every pair of splits, and for exact feature-plus-label duplicates within each split (0 in train, validation, and test).

## **3.10.3 Strict Temporal Monotonicity**

_max t(D_s,train) ≤ min t(D_s,val) and max t(D_s,val) ≤ min t(D_s,test) ∀ source labels s_ (3.11)

The condition is asserted per source label and held for all nine labels. Table 3.12 lists the resulting boundaries.

**_Table 3.12 — Chronological Split Boundaries per Source Label (UTC)_**

| **Source Label**             | **Train**                         | **Validation**                    | **Test**                          |
| ---------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| BENIGN (incl. Attempted)     | 02-14 12:28:07 → 02-28 13:59:38   | 02-28 13:59:38 → 03-01 17:35:38   | 03-01 17:35:38 → 03-03 00:39:53   |
| SSH-BruteForce               | 02-14 18:01:50 → 19:05:18         | 02-14 19:05:18 → 19:18:56         | 02-14 19:18:56 → 19:32:29         |
| DoS GoldenEye                | 02-15 13:27:46 → 13:52:26         | 02-15 13:52:26 → 13:55:04         | 02-15 13:55:05 → 14:02:59         |
| DoS Slowloris                | 02-15 15:00:12 → 15:29:00         | 02-15 15:29:00 → 15:35:37         | 02-15 15:35:37 → 15:41:34         |
| DoS Hulk                     | 02-16 17:45:27 → 17:54:26         | 02-16 17:54:26 → 17:56:24         | 02-16 17:56:24 → 17:58:22         |
| DDoS-LOIC-HTTP               | 02-20 14:13:54 → 14:57:56         | 02-20 14:57:56 → 15:07:21         | 02-20 15:07:21 → 15:16:48         |
| DDoS-LOIC-UDP                | 02-20 17:14:17 → 02-21 14:27:33   | 02-21 14:27:34 → 14:35:21         | 02-21 14:35:21 → 14:43:16         |
| DDoS-HOIC                    | 02-21 18:11:08 → 18:44:55         | 02-21 18:44:55 → 18:54:08         | 02-21 18:54:08 → 19:05:54         |
| Infiltration – NMAP Portscan | 02-28 14:46:45 → 03-01 14:40:13   | 03-01 14:40:13 → 18:37:54         | 03-01 18:37:54 → 19:37:52         |

Monotonicity holds per source label, not globally: the final training set extends to 2018-03-01 14:40 (Recon) while the test set begins on 2018-02-14 19:18 (SSH-BruteForce). Most attack campaigns occupy a single window of minutes, so their train, validation, and test partitions are consecutive slices of the same campaign. Benign validation and test rows come exclusively from the last capture days (28 February – 3 March).

## **3.10.4 Label Ambiguity Collision Analysis**

After deduplication and before reduction, exactly one 79-dimensional feature vector occurred under two categories: Benign (`BENIGN`) and Recon (`Infiltration - NMAP Portscan`), involving two rows. Because the deduplication key includes the category, these are distinct samples and both were kept. The final outputs contain zero feature-only and zero feature-plus-label overlaps across splits.

## **3.10.5 Determinism and Sampling Invariants**

Production assertions and permanent tests additionally establish that: only Benign, DoS, and DDoS rows are removed during reduction; every Recon and BruteForce row survives in every split; every targeted class reaches exactly min(available, target) rows in every split, and its three split counts sum to the total target; sampled rows never move between partitions; every temporal stratum and every retained source label stays represented in every split whenever the target allows it; the same seed and input reproduce identical rows and arrays, while a different seed changes only the sampled subsets, not untargeted rows, class counts, or split membership; and no NaN/Inf remains, with negative values appearing only in the two header-length columns.

# **3.11 Generated Output Artifacts**

The output directory `data/processed/CSECICIDS_2018_Distrinet/` uses the same file names, dtypes, class ids, and 79-feature order as the CIC-IDS-2017 processed layout. `CICIDS2018Adapter` (`src/datasets/cicids2018.py`, `get_adapter("cicids2018")`) changes only the processed path and the header-length bounds and otherwise reuses the generic `DatasetAdapter → FeatureManifest → FeatureTransform` interface. The six CIC-IDS-2017 derived-feature identities have zero violations on all 583,487 final training rows and all 125,033 validation rows.

**_Table 3.13 — Summary of Preprocessed Dataset Artifacts_**

| **Artifact**                                   | **Data Type / Structure** | **Dimensions**                 | **Role**                                                                       |
| ---------------------------------------------- | ------------------------- | ------------------------------ | ------------------------------------------------------------------------------ |
| train/val/test.parquet                         | Apache Parquet (ZSTD)     | 583k / 125k / 125k rows        | Pristine features, timestamps, full provenance, header flags, `sampling_stratum` |
| X_train/val/test.npy                           | float32 NumPy array       | (N, 79)                        | RobustScaled feature matrices for classifier and VAE training                  |
| X_\*\_pristine.npy                             | float32 NumPy array       | (N, 79)                        | Unscaled physical-unit feature matrices for domain and rule validation         |
| y_\*\_cat.npy                                  | int8 NumPy array          | (N,)                           | Consolidated 5-category labels (0–4)                                           |
| y_\*\_bin.npy                                  | int8 NumPy array          | (N,)                           | Binary detection labels (0 = Benign, 1 = Attack)                               |
| timestamp_epoch_{seconds,us}_\*.npy            | int64 NumPy array         | (N,)                           | Aligned UTC timestamps in integer seconds and microseconds                     |
| scaler.pkl / scaler_parameters.json            | Scikit-Learn object / JSON | serialized                    | Training-fitted RobustScaler (centre and IQR vectors)                          |
| class_weights_{5,2}.npy, label_encoders.json   | NumPy / JSON              | (5,), (2,)                     | Train-only inverse-frequency weights and exact label-to-id maps                |
| row_index.parquet                              | Apache Parquet            | 63,051,595 rows                | Every cleaned/mapped row: provenance, duplicate links, split and selection     |
| cleaning / header / duplicate / leakage audits | JSON / CSV                | key-value mapping              | Per-rule and per-label attrition, header flags, duplicates, leakage checks     |
| class/source-label reduction, split composition, attempted-by-split, sampling strata/audit | CSV / JSON | tables | Before/after counts and per-stratum sampling evidence                 |
| preprocessing_manifest.json                    | JSON audit record         | key-value mapping              | Input SHA-256 hashes, schema, feature order, policies, and run metadata        |

Report figures are written to `outputs/cicids2018distrinet/preprocessing/figures/`: the class distribution before reduction, the per-split class distributions after reduction, the population versus selected rows in every Benign/DoS/DDoS stratum, and descriptive-only Spearman and PCA plots of the training sample (PC1 60.2%, PC2 15.2%). Every after-reduction figure states that the class counts are a design choice, not natural CSE-CIC-IDS-2018 prevalence.

# **3.12 Chapter Summary and Methodological Scope**

This chapter presented the data-engineering protocol applied to the corrected DistriNet release of CSE-CIC-IDS-2018. By removing 801,389 non-finite, unsupported, or duplicate records, keeping and flagging rather than silently discarding the header-length overflow artefact, aligning the release to the 79-feature CIC-IDS-2017 schema and five-category taxonomy, enforcing chronological partitioning within each source label, reducing the three majority classes to fixed targets by time-stratified sampling inside each partition, and isolating all learned transformations to the final training set, the pipeline produces a leakage-controlled, class-controlled secondary benchmark whose layout is interchangeable with the CIC-IDS-2017 data.

## **Methodological Scope and Limitations**

1. Controlled class composition: every split has ≈30% Benign, 24% DoS, 24% DDoS, 11% Recon, and 11% BruteForce, against a natural ≈94.6% Benign prior. The composition is a design choice and is unsuitable for prevalence claims or naive probability calibration. Evaluation therefore reports per-class precision, recall, and F1, macro-F1, balanced accuracy, and the confusion matrix, with ordinary accuracy only as a supplementary metric; softmax outputs are not read as calibrated real-world probabilities, and any deployment-prior analysis requires explicit prior correction.
2. Within-campaign evaluation: attack campaigns are not independent across partitions. Chronological partitioning separates earlier and later phases of the same campaign; it does not support claims of global forward-time or new-campaign generalization.
3. Single sampling seed: the reduction uses one seed (42), so sampling variance is not quantified.
4. Reduced temporal diversity: DoS and DDoS are reduced to 10.9% and 14.6% of their rows, and each campaign spans only three to four hourly strata, so their temporal diversity was already small.
5. Rare subtypes and Attempted labels: rare subtypes are small after proportional sampling (DDoS-LOIC-UDP 55/55 and DoS Slowloris 131/131 validation/test rows), and rare Attempted labels can have zero selected observations.
6. Learnable artefacts: header-length overflow and attacker-tool fingerprints remain in the data. For example, the median Fwd Init Win Bytes is 26,883 for DoS Hulk, GoldenEye, Slowloris, and SSH-BruteForce and 65,535 for DDoS-HOIC, versus 2,044 for Benign; classifiers can separate these classes by tool or operating-system artefacts rather than attack behaviour, which qualifies any clean-accuracy claim.
7. Pending train-only artefacts: CSE-CIC-IDS-2018-specific VAEs, validator profile and rules, CFF masks, and PrimAttack budgets must still be fitted on this training split before adversarial evaluation.
8. Feature-space proxy: the pipeline operates on CICFlowMeter aggregates, not PCAPs; no packet-realizability claim follows.

# **3.13 Summary of Preprocessing Steps, With Ownership Demarcation**

## **Phase 1 — Upstream Dataset Correction (DistriNet)**

1. Corrected flow re-extraction from the original CSE-CIC-IDS-2018 captures.
2. Attack-specific relabelling replacing the original labels.
3. Attempted-flow classification: tagging failed or incomplete attack flows with the "- Attempted" suffix and a numeric `Attempted Category` code (0–6).
4. Raw CSV publication: exporting ten unpartitioned, unscaled daily CSV files (91 columns, 63,195,145 rows) with UTC microsecond timestamps and a per-file `id` row key.

## **Phase 2 — Machine Learning Preprocessing and Leakage Control (Author)**

1. File-set and schema validation: requiring exactly the ten expected files with identical 91-column headers.
2. Checksum recording: computing and recording SHA-256 hashes of all ten CSV files in the manifest.
3. Feature-contract alignment: loading the 79 feature names and order from the CIC-IDS-2017 manifest and requiring all of them in the same relative order.
4. Metadata and 2018-only column decoupling: moving seven metadata columns to a provenance sidecar and excluding the five CICIDS2018-only columns (including the ICMP −1 sentinel) from X.
5. Sample provenance ID generation: creating deterministic keys (source_file:id).
6. Timestamp parsing: converting UTC timestamp strings to integer epoch microseconds; never inferring the date from the file name.
7. Attempted-attack flow remapping: mapping every \*-Attempted label to source label BENIGN while preserving the original label and Attempted Category.
8. Non-finite value pruning: removing 57 zero-duration Benign rows with +∞ rates.
9. Physical invariant enforcement: requiring non-negativity on all modelling features except the two header-length columns (0 rows removed).
10. Header-length artefact flagging: keeping 16-bit-wrapped header lengths as raw values and recording negative/overflow flags as metadata.
11. Taxonomy consolidation and unsupported-class pruning: mapping 25 raw labels to 5 categories and dropping 143,493 rows of unsupported labels; unknown labels abort the run.
12. Canonical float32 casting: single-precision representation with −0.0 folded into +0.0.
13. Global precision-aware deduplication: removing 657,839 exact (x_float32, y_cat) duplicates, keeping the earliest occurrence.
14. Chronological sorting with deterministic tie-breaking: ordering by (timestamp, source-file order, id).
15. Per-source-label 70/15/15 split: constructing train, validation, and test folds by largest-remainder apportionment within each of the nine retained source labels.
16. Controlled class-size reduction: time-stratified undersampling of Benign, DoS, and DDoS to 250,000/200,000/200,000 total rows inside each existing split; Recon and BruteForce kept whole.
17. Dual-head target encoding: generating aligned binary (y_bin ∈ {0,1}) and 5-category (y_cat ∈ {0–4}) label arrays.
18. Train-only robust scaling: estimating median and IQR exclusively on the 583,487 final training rows, with unit scale for the 25 zero-IQR features.
19. Zero-leakage validation/test transformation: applying the frozen training scaler.
20. Class-weight formulation: computing inverse-frequency weights from the final training labels.
21. Two-pass streaming serialization: exporting ZSTD Parquet tables, scaled and pristine float32 arrays, timestamps, the scaler, the full row index, audit tables, figures, and the manifest, with pass-to-pass hash re-verification.
22. Automated partition auditing: asserting zero ID overlap, zero cross-split and within-split duplicates, per-source-label chronological monotonicity, sampling invariants, and seed determinism.
