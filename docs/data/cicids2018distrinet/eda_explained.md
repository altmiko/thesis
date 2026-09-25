# CSE-CIC-IDS-2018 DistriNet — EDA explained

Guide to the raw-data exploratory analysis of the corrected DistriNet CSE-CIC-IDS-2018
release: what the script computes, how to read each report section, what the current run
found, and which preprocessing decisions those findings force.

- Script: [`src/evaluation/cicids2018_distrinet_eda.py`](../../../src/evaluation/cicids2018_distrinet_eda.py)
- Generated report: `outputs/cicids2018distrinet/eda/cicids2018_distrinet_eda.md`
  (git-ignored; regenerate with the command below)
- CICIDS2017 counterpart (EDA in §3–§5): [`../cicids2017distrinet/cicids2017distrinet_preprocessing.md`](../cicids2017distrinet/cicids2017distrinet_preprocessing.md)

Numbers quoted here come from the full run (63,195,145 rows, seed 42). The generated report is
the source of truth if they ever disagree.

---

## 1. Scope and claim boundary

- **Input:** the ten DistriNet day files under `data/raw/CSECICIDS2018_Distrinet/`
  (36.0 GB, 91 columns). This is DistriNet's relabelled release (Liu et al., CNS 2022),
  not the original CIC/AWS CSVs.
- **Pre-split, descriptive only.** No split exists yet, so every statistic describes the whole
  release. None of it may feed the scaler, CFF masks, mined constraints, VAE, or thresholds:
  thesis policy fits those on the training split only (`CLAUDE.md`, "THE REFACTOR PRINCIPLE").
  The PCA, Spearman correlations, and quantiles in the report are for inspection only.
- **Not a preprocessing pipeline.** The script writes no model arrays. It answers the questions
  a CICIDS2018 preprocessing script must settle first (§5 of this guide).
- **Feature-space only.** No PCAP is read. The global claim boundary in `CLAUDE.md` applies.

## 2. Running it

```powershell
$Env:PATH = $Env:PATH -replace 'C:\\Program Files\\Tailscale"', 'C:\Program Files\Tailscale'
conda activate C:\Users\user6\.local\share\mamba\envs\thesis
python src/evaluation/cicids2018_distrinet_eda.py            # full run, ~6 min with 4 workers
python src/evaluation/cicids2018_distrinet_eda.py --max-rows-per-file 200000 --output-dir <tmp>  # smoke
```

| Option | Default | Meaning |
|---|---|---|
| `--input-dir` | `data/raw/CSECICIDS2018_Distrinet` | must contain exactly the ten expected files |
| `--output-dir` | `outputs/cicids2018distrinet/eda` | report, JSON, tables, figures |
| `--workers` | 4 | files scanned in parallel; `1` runs in-process |
| `--uniform-sample` | 250,000 | bottom-K uniform sample (quantiles, Spearman, identical columns) |
| `--per-label-sample` | 5,000 | bottom-K sample cap per raw label (medians, histograms, PCA) |
| `--seed` | `config.paths.SEED` (42) | sampling keys |
| `--max-rows-per-file` | none | smoke-test cap; report is marked *Limited run*, hashes skipped |
| `--skip-input-hashes` | off | skip SHA-256 of the raw CSVs |

Outputs:

```text
outputs/cicids2018distrinet/eda/
├── cicids2018_distrinet_eda.md      generated report (sections referenced below)
├── eda_report.json                  every scalar result + arguments + elapsed time
├── tables/*.csv                     one CSV per report table (feature_summary, attack_schedule, …)
└── figures/                         label_distribution, attack_timeline, feature_histograms,
                                     spearman_heatmap, pca_stratified (.png)
```

## 3. How the script works

**One streaming pass per file.** Each worker opens a file once. pyarrow's streaming CSV
reader parses it, and a `HashingReader` wrapper feeds exactly the bytes pyarrow consumes into
SHA-256, so the hash costs no second read. Column types are fixed up front: 84 numeric
candidates as `float64`, `id` as `int64`, identifiers and timestamp as strings, and `Label` as
raw bytes, decoded as UTF-8 with a counted cp1252 fallback.

**Block size.** pyarrow's streaming reader slows super-linearly with block size on these files
(measured about 300k rows/s at 2 MiB vs about 25k rows/s at 64 MiB). The script therefore reads
2 MiB blocks and re-batches them into about 250k-row batches (`_rebatched`) before the numpy/pandas work.

Every statistic falls into one of three classes, and the report marks which is which:

| Class | Mechanism | Used for |
|---|---|---|
| **Exact** | streaming counters/min/max/sums over every row | inventory, labels, schedule, protocol/endpoint counts, NaN/Inf, negatives, sentinels, integer/constant columns, min/max/mean |
| **Hashed (exact up to 64-bit collisions)** | `pandas.util.hash_pandas_object` per row, then `np.unique` over all 63.2M hashes | duplicates, feature-vector label conflicts, `Flow ID` reuse; expected collisions ≈ n²/2⁶⁵ ≈ 1e-4 |
| **Sampled** (*sample*) | bottom-K sampling: each row gets a seeded uniform key, and the K smallest keys are kept | quantiles, Spearman, identical columns, class medians, histograms, PCA |

Bottom-K sampling gives an exact uniform sample without replacement. It is also mergeable (the
bottom-K of a union is the bottom-K of the per-file bottom-Ks), so the result does not depend on
worker scheduling. Keys come from `default_rng([seed, file_index])`.

The duplicate definition matches the CICIDS2017 pipeline: exact equality of the **float32**
feature vector plus the label, because float32 is the precision models consume. The report
computes it on all rows and again on rows that survive the 2017 row-local cleaning rule
(§9.1 below).

## 4. Report sections and current findings

### §1 File inventory
Ten files with 5.41M–7.39M rows each, **63,195,145 rows** in total. `id` equals the 1-based row
number in every file, so `file:id` is a stable row key. The report lists a SHA-256 per file.

### §2 Schema
All ten headers are identical: **91 columns**. Seven are non-features (`id`, `Flow ID`,
`Src IP`, `Dst IP`, `Timestamp`, `Label`, `Attempted Category`) and **84 are numeric
candidates**. All 79 CICIDS2017 modelling features are present, in the same relative order.
The 2018 release adds five:

| New column | Observed behaviour |
|---|---|
| `Fwd RST Flags`, `Bwd RST Flags` | per-direction RST counts; DistriNet's GoldenEye labelling rules use them |
| `ICMP Code`, `ICMP Type` | `-1` on exactly every non-ICMP row, real codes on the 106,003 ICMP rows (§9) |
| `Total TCP Flow Time` | 0 on every non-TCP row |

So the 2017-trained schema (`preprocessing_manifest.json:modelling_feature_names`) can be
reproduced on 2018 by column selection. Cross-dataset experiments need no renaming.

### §3 Raw labels, Attempted flows
- **25 literal labels.** BENIGN is 59,353,486 rows (**93.92%**).
- The largest attacks are DoS Hulk 1,803,160, DDoS-HOIC 1,082,293, DDoS-LOIC-HTTP 289,328,
  Botnet Ares 142,921, SSH-BruteForce 94,197 and Infiltration - NMAP Portscan 89,374.
- **Attempted:** 306,237 rows (0.48%) carry an ` - Attempted` label. Unlike the 2017 export,
  2018 includes DistriNet's `Attempted Category` code. The §3.2 crosstab shows it is `-1` on
  every non-Attempted row and 0–6 on Attempted rows. DistriNet's documentation defines the codes:

  | Code | Meaning (DistriNet) | Main occurrences here |
  |---:|---|---|
  | 0 | no payload sent by attacker | DoS Slowloris/Hulk, Botnet Ares, web attacks |
  | 1 | port/system closed | FTP-BruteForce (298,844) |
  | 2 | attack startup/teardown artefact | Botnet Ares, web attacks |
  | 3 | no malicious payload | Web Attack - XSS |
  | 4 | attack artefact | DoS GoldenEye (4,248), FTP-BruteForce (30), Dropbox |
  | 5 | attack implemented incorrectly | Web Attack - Brute Force (126) |
  | 6 | target system unresponsive | DDoS-LOIC-UDP (251, ICMP), DoS GoldenEye (53) |

- **There is no successful FTP brute force.** Port 21 on the victim was closed. All 298,874 FTP
  rows are `FTP-BruteForce - Attempted`, and DistriNet reports that the 16-02 "FTP" traffic was
  most likely a misfired DoS Slowhttptest. No DoS Slowhttptest or Heartleech label exists.
- DistriNet documents an `SSH-BruteForce - Attempted` rule, but this export contains **0**
  such rows.

### §4 Attack schedule
Every attack label occurs on **one or two capture days** (§3.1), within windows of minutes to
hours (§4 table; `figures/attack_timeline.png`). DoS Hulk produced 1.8M flows in 12.9 minutes.
The timeline plots each file's capture window, and each panel title counts the rows outside it.

Consequence: as with CICIDS2017, a whole-day train/val/test split would put entire attack
families only in validation or test. A closed-set classifier needs a chronological split
*within* each source label (the CICIDS2017 protocol), and the same within-campaign caveat applies.

### §5 Protocol and endpoints
- Protocol rows: TCP 38.05M, UDP 24.96M, ICMP 106,003, protocol 0: 77,810.
- **Each attack has 1–10 fixed attacker IPs and usually a single victim**
  (e.g. DoS Hulk: 18.219.193.20 → 172.31.69.25:80, 100%). Almost all DoS/DDoS/web attacks
  target port 80, SSH port 22, FTP port 21, and Botnet Ares port 8080.
- `DDoS-LOIC-UDP - Attempted` is **ICMP from the victims back to the attackers** (destination
  unreachable, destination port 0). This is response traffic, not attacker packets.

Consequence: IP addresses are a perfect shortcut and stay out of `X` (they are non-features
here, as in 2017). `Dst Port` and `Protocol` are model inputs that nearly identify some classes.
Treat them as immutable during attacks, as in the CICIDS2017 manifest.

### §6 Row identity
`Flow ID` is a textual 5-tuple, not a key: 35,160,510 distinct values over 63.2M rows, maximum
multiplicity 1,231, and 6,249,883 `Flow ID`s appear in two or more files. Use `file:id` for
provenance and deduplication; never join on `Flow ID`.

### §7 Timestamps
- 0 missing, 0 unparseable, **microsecond precision on every row** (the 2017 export had whole
  seconds). DistriNet states the times are UTC.
- **File order is not chronological:** 27,820,016 adjacent timestamp reversals. Splits must sort
  by parsed timestamp, as the 2017 pipeline does.
- **11,693 rows fall on a date other than the file name.** Seven files run past midnight UTC
  (last rows 00:36–01:14 the next day). `Friday-23-02-2018.csv` also starts at
  **2018-02-21 12:33**: 2,609 of its rows are a low-rate BENIGN trickle from then until about
  midday on 02-22, a day before its own capture begins (the timeline panel title counts them).
  Preprocessing must not assume "file = date".

### §8 Missing and infinite values
There are no nulls or NaNs. **57 rows** have `+inf` in both `Flow Bytes/s` and `Flow Packets/s`
(zero-duration flows). All are BENIGN. Dropping them is harmless.

### §9 Negative values and sentinels
- **`ICMP Code`/`ICMP Type` = `-1` is a sentinel,** not a physical value. It appears on exactly the
  63,089,142 non-ICMP rows, and never on ICMP rows. The CICIDS2017 rule "drop any row with a
  negative value" would therefore delete 99.8% of the dataset. The sentinel must be exempted,
  or the two columns re-typed (e.g. categorical with an explicit "not ICMP" level).
- **`Fwd Header Length` (10,039 rows) and `Bwd Header Length` (17,509 rows) are negative, with
  range exactly [−32,768, 32,764].** That signature points to a signed 16-bit overflow in the
  flow meter on very long flows [INFERENCE: consistent with the observed range; not confirmed
  against the CICFlowMeter source].
- **§9.1 shows why this matters.** The 2017 rule, even with the ICMP sentinel exempt, removes
  23,750 rows, and they are not random: **47.8% of DDoS-LOIC-UDP** (1,209 of 2,527) would
  disappear, because LOIC-UDP flows are long enough to overflow. Silently reusing the 2017 rule
  would bias that class. Options for the preprocessing decision:
  (a) drop and document the class-specific loss;
  (b) exclude the two header-length columns;
  (c) treat negative header length as a flagged artefact.
  Any correction must not be fitted on non-training data.

### §10 Duplicates and label conflicts
- No two raw rows are identical (all columns except `id`).
- Float32 features + label: **655,614 duplicates (1.04%)** over all rows. These concentrate in
  **FTP-BruteForce - Attempted (70.8% duplicates)** and BENIGN (0.75%). Real attacks have
  essentially none (NMAP Portscan: 19).
- **53 feature vectors (374 rows) carry two labels, always BENIGN vs FTP-BruteForce - Attempted.**
  Closed-port SYN/RST exchanges are indistinguishable from benign ones, which supports mapping
  Attempted to BENIGN. Deduplicate globally before splitting, as in 2017; otherwise these rows
  leak across splits.

### §11 Scale and heavy tails
The data is heavily right-skewed with extreme scale differences:
- `Flow Bytes/s` has median 2,240, 99.9th percentile 1.04e6 and max 2.92e9.
- `Total Length of Fwd Packet` has median 97 and max 5.4e7.
- `Flow Duration` saturates at 120,000,000 µs, the CICFlowMeter 120 s timeout. Long flows such as
  Slowloris are therefore split into several records.

These support the same median/IQR `RobustScaler` + `asinh` treatment as CICIDS2017.

§11.1 medians expose **tool fingerprints**. The median `FWD Init Win Bytes` is 26,883 for
DoS Hulk, GoldenEye, Slowloris and SSH-BruteForce, and 65,535 for DDoS-HOIC, vs 2,044 for
BENIGN. `Fwd Seg Size Min` is 32/40 for several attacks vs 20 for BENIGN. Classifiers can
separate these classes by attacker-OS/tool artefacts rather than attack behaviour. This caveat
belongs with any clean-accuracy claim.

### §12 Column structure
- **Constant:** only `Bwd URG Flags`. `Fwd URG Flags`/`URG Flag Count` are non-zero on 0.0013% of rows.
- **Identical pairs** (sample, relative tolerance 1e-9): `Fwd Packet Length Mean ≡ Fwd Segment
  Size Avg`, `Bwd Packet Length Mean ≡ Bwd Segment Size Avg`, `Packet Length Mean ≡ Average
  Packet Size`, `Fwd URG Flags ≡ URG Flag Count`. Spearman also shows `Packet Length Variance =
  Std²`, the Bulk triplets, and Active/Idle mean/max/min.
- **`Subflow Fwd Packets` and `Subflow Bwd Packets` take only values {0, 1}** (98.8% and 99.9%
  zero), so they are not packet totals in this export.
- 59 of 84 columns are integer-valued over all 63.2M rows.

These are exactly the relationships the train-only constraint miner and `FeatureManifest`
typing should *discover*. Do not hand-author them into the constraint engine.

### §13 Imbalance
The largest-to-smallest label ratio is 14.8M : 1. **12 labels have fewer than 1,000 rows** and
6 have fewer than 100: all web attacks, the Infiltration sub-attacks other than NMAP, and most
Attempted variants. They cannot support a chronological 70/15/15 split with stable estimates.

### §14 PCA
PCA on the per-label stratified sample (signed log1p, standardized, descriptive only):
PC1/PC2 explain 35.4% / 14.8%. The large attacks form tight, tool-specific clusters (DoS Hulk,
HOIC, LOIC-HTTP, NMAP), while BENIGN is diffuse. This is consistent with the fingerprint
finding in §11.1.

## 5. Preprocessing decisions implemented

These follow from the evidence above. They are decided and implemented; see
[`cicids2018distrinet_preprocessing.md`](../../cicids2018distrinet/cicids2018distrinet_preprocessing.md).
The final choices: map as proposed in item 1, Attempted → BENIGN, keep and flag negative/wrapped
header lengths (option c), use the 79 CICIDS2017 features (item 9), and after the chronological
split time-stratify Benign/DoS/DDoS down to 250k/200k/200k total rows (Recon/BruteForce kept whole).

1. **Label → category map** (implemented; mirrors CICIDS2017 `SOURCE_TO_CATEGORY`):

   | Category | 2018 source labels | Rows |
   |---|---|---:|
   | Benign | BENIGN (+ every `- Attempted` under the default policy) | 59,659,723 |
   | DoS | DoS Hulk, DoS GoldenEye, DoS Slowloris | 1,834,210 |
   | DDoS | DDoS-HOIC, DDoS-LOIC-HTTP, DDoS-LOIC-UDP | 1,374,148 |
   | Recon | Infiltration - NMAP Portscan | 89,374 |
   | BruteForce | SSH-BruteForce | 94,197 |

   Unmapped (dropped, as Bot/Web/Infiltration were in 2017): Botnet Ares, Web Attack *,
   Infiltration - Dropbox Download, Infiltration - Communication Victim Attacker.
   Two caveats:
   - Recon comes from the internal *Infiltration* campaign (attackers 172.31.69.13/.24 scanning
     22 hosts), not from an external scanner as in CICIDS2017 PortScan.
   - BruteForce is SSH-only, because FTP never succeeded.
2. **Attempted policy:** default → BENIGN (DistriNet guidance; §10 conflicts support it). Keep
   `original_label` and `Attempted Category` as metadata only.
3. **ICMP sentinel:** exempt `-1` from the negative-value rule, or re-type the ICMP columns.
   Never clip or impute.
4. **Negative header lengths:** choose option (a)/(b)/(c) from §9 and report the per-class
   impact. The default 2017 rule costs 47.8% of DDoS-LOIC-UDP.
5. **Non-finite rows:** drop the 57 BENIGN `+inf` rows (no imputation).
6. **Deduplicate** on float32 features + mapped category, earliest occurrence kept, before
   splitting.
7. **Chronology:** sort by parsed UTC timestamp with `(file order, id)` tie-breakers. Do not
   infer the date from the file name.
8. **Split:** chronological 70/15/15 within each retained source label, as in 2017. Every
   retained label has at least 2,527 rows, so every split is populated.
9. **Feature set:** either the 79 CICIDS2017 columns (for cross-dataset comparability) or all 84.
   The five 2018-only columns carry protocol-specific sentinel/zero semantics that a manifest
   must type explicitly.

## 6. Limitations

- Everything is pre-split and descriptive; no statistic here is valid as a fitted parameter.
- Hash-based counts can be off by expected collisions ≈ 1e-4 rows.
- Sampled sections (quantiles, Spearman, identical columns, medians, PCA) are estimates. Rare
  labels are fully included in the per-label sample (the cap is 5,000 rows).
- The header-length overflow explanation is an inference from the value range.

## 7. References

1. Liu, L., Engelen, G., Lynar, T., Essam, D., and Joosen, W. (2022), *Error Prevalence in NIDS
   Datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018*, IEEE CNS,
   DOI [10.1109/CNS56114.2022.9947235](https://doi.org/10.1109/CNS56114.2022.9947235).
2. DistriNet, [Improved CSE-CIC-IDS 2018 documentation](https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html):
   per-attack labelling logic, Attempted categories, UTC time windows.
3. Canadian Institute for Cybersecurity, [CSE-CIC-IDS2018 dataset description](https://www.unb.ca/cic/datasets/ids-2018.html).
