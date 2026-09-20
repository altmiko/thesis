# CICIoT2023 Full Build and Preprocessing Guide

**Repository:** `E:/Shameem/thesis`  
**Scope:** the downloaded CICIoT2023 CSV tree, the labelled Parquet build, the leakage-safe split, cleaning, scaling, train-only downsampling, and all generated preprocessing artifacts.  
**Code inspected:** `src/preprocessing/schema.py`, `src/preprocessing/ciciot2023/build_labeled_ciciot2023_dataset.py`, `src/preprocessing/ciciot2023/splitter.py`, `src/preprocessing/ciciot2023/sampler.py`, `src/preprocessing/ciciot2023/pipeline.py`, `src/preprocessing/ciciot2023/reports.py`, `config/paths.py`, and the CICIoT2023 tests.  
**On-disk inspection:** 309 raw CSV files, all raw headers, the build manifest, the archived preprocessing run manifest, and the legacy combined CSV were inspected.

> This is a code-and-disk-grounded description of the current repository. Where an older artifact or older document disagrees with the current code, the disagreement is called out explicitly instead of being silently merged into the current procedure.

---

## 1. One-page overview

The current data path is:

```text
CICIoT2023 vendor CSV download
  data/raw/CICIoT2023_CSV_DOWNLOADED/
  34 class folders / 309 unlabelled CSV shards
        |
        | build_labeled_ciciot2023_dataset.py
        | - discover folders and CSVs
        | - folder -> 34-class Label -> 8-class category
        | - validate the 39-feature header
        | - drop NaN and +/-inf rows only
        | - cast features to float32
        | - append audit metadata
        | - stream into one Zstandard Parquet file
        v
Labelled source Parquet
  data/processed/ciciot2023_labeled_full.parquet
  46,775,660 retained rows / 43 columns
  data/processed/ciciot2023_labeled_full_manifest.json
        |
        | pipeline.py
        | 1. load Parquet
        | 2. infer contiguous source-shard runs
        | 3. plan train/validation/test split before fitting anything
        | 4. clean with train-only 99.99th-percentile bounds
        | 5. fit RobustScaler on train only
        | 6. cluster and undersample train majority categories only
        | 7. encode labels and save arrays/metadata
        v
Model-ready artifacts
  X_train.npy, X_val.npy, X_test.npy
  y_* fine-label, category-label, and binary-label arrays
  class weights, scaler, encoders, name maps
  train_kept_indices.npy and run_manifest.json
```

The important ordering invariant is:

```text
split first -> train-only cleaning bounds -> train-only scaler fit
             -> train-only clustering/sampling -> save all splits
```

Validation and test rows are never downsampled. They retain their natural class mix so headline evaluation is not performed on an artificially balanced holdout.

---

## 2. Dataset locations and what each location means

### 2.1 Relevant repository tree

```text
data/
├── raw/
│   ├── CICIoT2023_CSV_DOWNLOADED/
│   │   ├── 34 class folders
│   │   ├── 309 *.pcap.csv files
│   │   ├── README_CSV.pdf
│   │   └── tree.txt                         # inventory text file
│   ├── ciciot2023_full/
│   │   └── ciciot2023_base.csv              # legacy 40-column combined CSV
│   ├── NF-TON-IOT/
│   │   └── ...                               # separate NetFlow ToN-IoT dataset
│   └── .gitkeep
├── processed/
│   ├── ciciot2023_labeled_full.parquet
│   ├── ciciot2023_labeled_full_manifest.json
│   ├── downsampling_diagnostics/
│   │   └── diagnostics JSON/CSV/PNG outputs
│   └── old/
│       └── archived model-ready arrays and an older run manifest
└── ciciot2023_full_build.md                   # this document
```

### 2.2 Do not mix the three raw-data trees

* `data/raw/CICIoT2023_CSV_DOWNLOADED/` is the official CICIoT2023 CSV distribution used by the current labelled-Parquet build.
* `data/raw/ciciot2023_full/ciciot2023_base.csv` is a separate, historical single-file export. It has 39 features plus `Label` (40 columns). There is no current live CSV-export module in `src/preprocessing/ciciot2023/`; the current build target is Parquet.
* `data/raw/NF-TON-IOT/` is the NetFlow V2 ToN-IoT dataset. It is not part of CICIoT2023 and must not be included in the build.

The vendor `README_CSV.pdf` says the CSV distribution refers to a shuffled dataset and describes a `label` feature. The actual files on disk contradict that wording: all 309 inspected headers have exactly 39 feature columns and no `Label` or `label` column. The current builder therefore correctly derives labels from parent-folder names.

---

## 3. Raw CICIoT2023 layout

### 3.1 What a shard is

Each `*.pcap.csv` file is a flow-feature shard exported from captured traffic. The class is implicit in the parent directory. The raw CSV itself contains no target column. A shard is not treated as an independent statistical capture session by the split code; the splitter treats numbered shards as sequential segments of a class capture and uses their numeric suffixes as temporal order.

The suffix-less base file represents sequence `0` for the natural-numeric ordering used by `splitter.py`. For example:

```text
DDoS-ICMP_Flood.pcap.csv
DDoS-ICMP_Flood1.pcap.csv
DDoS-ICMP_Flood2.pcap.csv
...
DDoS-ICMP_Flood26.pcap.csv
```

The build script itself discovers files with ordinary lexical `sorted(...)` ordering. The later split planner independently applies `natkey()` so `...2...` precedes `...10...`. This distinction matters: source rows remain contiguous within each shard, but global Parquet row order should not be interpreted as natural temporal order across numbered shards merely from the build order.

### 3.2 Raw inventory checked on disk

The current raw root contains exactly **34 directories** and **309 CSV shards**. `README_CSV.pdf`, `tree.txt`, and other loose files are not class directories and are skipped by `resolve_shards()`.

The table gives the complete folder inventory, exact shard count, observed retained rows in the labelled manifest, the assigned label/category, and the exact filename sequence. In the filename column, suffix `0` means the file has no numeric suffix; the final name is formed as `<stem>.pcap.csv` for `0`, and `<stem><n>.pcap.csv` for `n > 0`.

| Raw folder | Label | Category | Shards | Retained rows | Exact filename stem and suffixes |
|---|---|---:|---:|---:|---|
| `Backdoor_Malware` | `BACKDOOR_MALWARE` | Web | 1 | 3,218 | `Backdoor_Malware` [0] |
| `Benign_Final` | `BENIGN` | Benign | 4 | 1,098,126 | `BenignTraffic` [0–3] |
| `BrowserHijacking` | `BROWSERHIJACKING` | Web | 1 | 5,859 | `BrowserHijacking` [0] |
| `CommandInjection` | `COMMANDINJECTION` | Web | 1 | 5,409 | `CommandInjection` [0] |
| `DDoS-ACK_Fragmentation` | `DDOS-ACK_FRAGMENTATION` | DDoS | 13 | 285,045 | `DDoS-ACK_Fragmentation` [0–12] |
| `DDoS-HTTP_Flood` | `DDOS-HTTP_FLOOD` | DDoS | 1 | 28,790 | `DDoS-HTTP_Flood-` [0] (actual file has the extra hyphen) |
| `DDoS-ICMP_Flood` | `DDOS-ICMP_FLOOD` | DDoS | 27 | 7,200,436 | `DDoS-ICMP_Flood` [0–26] |
| `DDoS-ICMP_Fragmentation` | `DDOS-ICMP_FRAGMENTATION` | DDoS | 20 | 452,444 | `DDoS-ICMP_Fragmentation` [0–19] |
| `DDoS-PSHACK_FLOOD` | `DDOS-PSHACK_FLOOD` | DDoS | 16 | 4,094,727 | `DDoS-PSHACK_Flood` [0–15] (folder and file capitalization differ) |
| `DDoS-RSTFINFLOOD` | `DDOS-RSTFINFLOOD` | DDoS | 16 | 4,045,248 | `DDoS-RSTFINFlood` [0–15] (folder and file capitalization differ) |
| `DDoS-SYN_Flood` | `DDOS-SYN_FLOOD` | DDoS | 16 | 4,059,097 | `DDoS-SYN_Flood` [0–15] |
| `DDoS-SlowLoris` | `DDOS-SLOWLORIS` | DDoS | 1 | 23,425 | `DDoS-SlowLoris` [0] |
| `DDoS-SynonymousIP_Flood` | `DDOS-SYNONYMOUSIP_FLOOD` | DDoS | 14 | 3,598,100 | `DDoS-SynonymousIP_Flood` [0–13] |
| `DDoS-TCP_Flood` | `DDOS-TCP_FLOOD` | DDoS | 18 | 4,497,546 | `DDoS-TCP_Flood` [0–17] |
| `DDoS-UDP_Flood` | `DDOS-UDP_FLOOD` | DDoS | 21 | 5,412,169 | `DDoS-UDP_Flood` [0–20] |
| `DDoS-UDP_Fragmentation` | `DDOS-UDP_FRAGMENTATION` | DDoS | 13 | 286,895 | `DDoS-UDP_Fragmentation` [0–12] |
| `DNS_Spoofing` | `DNS_SPOOFING` | Spoofing | 1 | 178,893 | `DNS_Spoofing` [0] |
| `DictionaryBruteForce` | `DICTIONARYBRUTEFORCE` | BruteForce | 1 | 13,064 | `DictionaryBruteForce` [0] |
| `DoS-HTTP_Flood` | `DOS-HTTP_FLOOD` | DoS | 2 | 71,857 | `DoS-HTTP_Flood` [0–1] |
| `DoS-SYN_Flood` | `DOS-SYN_FLOOD` | DoS | 8 | 2,028,791 | `DoS-SYN_Flood` [0–7] |
| `DoS-TCP_Flood` | `DOS-TCP_FLOOD` | DoS | 11 | 2,671,363 | `DoS-TCP_Flood` [0–10] |
| `DoS-UDP_Flood` | `DOS-UDP_FLOOD` | DoS | 17 | 3,072,883 | `DoS-UDP_Flood` [0–16] |
| `MITM-ArpSpoofing` | `MITM-ARPSPOOFING` | Spoofing | 2 | 307,542 | `MITM-ArpSpoofing` [0–1] |
| `Mirai-greeth_flood` | `MIRAI-GREETH_FLOOD` | Mirai | 29 | 991,774 | `Mirai-greeth_flood` [0–28] |
| `Mirai-greip_flood` | `MIRAI-GREIP_FLOOD` | Mirai | 22 | 751,589 | `Mirai-greip_flood` [0–21] |
| `Mirai-udpplain` | `MIRAI-UDPPLAIN` | Mirai | 25 | 890,507 | `Mirai-udpplain` [0–24] |
| `Recon-HostDiscovery` | `RECON-HOSTDISCOVERY` | Recon | 1 | 134,377 | `Recon-HostDiscovery` [0] |
| `Recon-OSScan` | `RECON-OSSCAN` | Recon | 1 | 98,255 | `Recon-OSScan` [0] |
| `Recon-PingSweep` | `RECON-PINGSWEEP` | Recon | 1 | 2,262 | `Recon-PingSweep` [0] |
| `Recon-PortScan` | `RECON-PORTSCAN` | Recon | 1 | 82,283 | `Recon-PortScan` [0] |
| `SqlInjection` | `SQLINJECTION` | Web | 1 | 5,244 | `SqlInjection` [0] |
| `Uploading_Attack` | `UPLOADING_ATTACK` | Web | 1 | 1,252 | `Uploading_Attack` [0] |
| `VulnerabilityScan` | `VULNERABILITYSCAN` | Recon | 1 | 373,344 | `VulnerabilityScan` [0] |
| `XSS` | `XSS` | Web | 1 | 3,846 | `XSS` [0] |
| **Total** | **34 labels** | **8 categories** | **309** | **46,775,660** | — |

The ranges in the table are complete for the current disk inventory: there are no missing suffixes within a listed range. The builder does not synthesize missing files and does not infer a class from the filename; it uses the folder name.

### 3.3 Category totals

The build manifest reports 46,775,660 retained rows after the raw-stage NaN/inf drop:

| Category | Fine labels | Rows | Approximate share |
|---|---:|---:|---:|
| DDoS | 12 | 33,983,922 | 72.65% |
| DoS | 4 | 7,844,894 | 16.77% |
| Mirai | 3 | 2,633,870 | 5.63% |
| Benign | 1 | 1,098,126 | 2.35% |
| Recon | 5 | 690,521 | 1.48% |
| Spoofing | 2 | 486,435 | 1.04% |
| BruteForce | 1 | 13,064 | 0.03% |
| Web | 6 | 24,828 | 0.05% |

The distribution is extremely imbalanced. This is why the later sampler caps six majority categories but keeps BruteForce and Web whole.

---

## 4. Canonical schema

### 4.1 The one source of truth

`src/preprocessing/schema.py` owns:

* `FEATURE_NAMES`: the ordered list of 39 features.
* `LABEL_COLUMN`: `Label`.
* `BINARY_FEATURES`: 15 protocol/service indicators.
* `INTEGER_FEATURES`: 12 integer-valued flag/count features.
* `CATEGORY_MAP`: the 34 uppercase fine labels to eight categories.

Feature order is an invariant. Every saved `X_*.npy` array, scaler, VAE, attack, validator, and evaluation script indexes by this order. Reordering the list is a breaking data-format change.

### 4.2 Exact feature order

| Position | Feature | Role in current preprocessing |
|---:|---|---|
| 1 | `Header_Length` | continuous for clustering |
| 2 | `Protocol Type` | categorical-like protocol value; excluded from clustering |
| 3 | `Time_To_Live` | continuous for clustering |
| 4 | `Rate` | continuous for clustering |
| 5 | `fin_flag_number` | integer; continuous clustering input after scaling |
| 6 | `syn_flag_number` | integer; continuous clustering input after scaling |
| 7 | `rst_flag_number` | integer; continuous clustering input after scaling |
| 8 | `psh_flag_number` | integer; continuous clustering input after scaling |
| 9 | `ack_flag_number` | integer; continuous clustering input after scaling |
| 10 | `ece_flag_number` | integer; continuous clustering input after scaling |
| 11 | `cwr_flag_number` | integer; continuous clustering input after scaling |
| 12 | `ack_count` | integer; continuous clustering input after scaling |
| 13 | `syn_count` | integer; continuous clustering input after scaling |
| 14 | `fin_count` | integer; continuous clustering input after scaling |
| 15 | `rst_count` | integer; continuous clustering input after scaling |
| 16 | `HTTP` | binary indicator |
| 17 | `HTTPS` | binary indicator |
| 18 | `DNS` | binary indicator |
| 19 | `Telnet` | binary indicator |
| 20 | `SMTP` | binary indicator |
| 21 | `SSH` | binary indicator |
| 22 | `IRC` | binary indicator |
| 23 | `TCP` | binary indicator |
| 24 | `UDP` | binary indicator |
| 25 | `DHCP` | binary indicator |
| 26 | `ARP` | binary indicator |
| 27 | `ICMP` | binary indicator |
| 28 | `IGMP` | binary indicator |
| 29 | `IPv` | binary indicator |
| 30 | `LLC` | binary indicator |
| 31 | `Tot sum` | continuous for clustering |
| 32 | `Min` | continuous for clustering |
| 33 | `Max` | continuous for clustering |
| 34 | `AVG` | continuous for clustering |
| 35 | `Std` | continuous for clustering |
| 36 | `Tot size` | continuous for clustering |
| 37 | `IAT` | continuous for clustering |
| 38 | `Number` | integer; continuous clustering input after scaling |
| 39 | `Variance` | continuous for clustering |

The raw distribution is the 39-feature CIC-shipped CSV schema. It is a subset of the larger CICFlowMeter feature set; missing source columns cannot be recreated from these CSVs alone.

### 4.3 Raw and Parquet column layout

Every raw CSV has exactly the 39 feature columns above, in that order. The build appends four columns:

```text
39 feature columns
+ Label                 # uppercase 34-class target
+ category              # one of eight category names
+ source_csv_filename   # path relative to raw root, slash-normalized
+ source_folder         # original parent folder name
= 43 Parquet columns
```

`Label` and `category` are converted to pandas categorical values before the PyArrow conversion. The four metadata columns are deliberately retained:

* `Label` supports 34-class learning.
* `category` supports the six-majority-category sampling policy and eight-class learning.
* `source_csv_filename` supports shard-aware splitting and auditing.
* `source_folder` preserves the original human-readable folder identity.

A subtle code detail: `schema.py::EXPECTED_COLUMNS` contains the 39 features plus `Label` (the original labelled-CSV contract), while `process_shard()` separately asserts the actual 43-column Parquet output order. The latter is the authoritative check for the current builder output.

---

## 5. Stage A — building the labelled Parquet

**Implementation:** `src/preprocessing/ciciot2023/build_labeled_ciciot2023_dataset.py`

### 5.1 Defaults and command line

The script defaults to:

```text
raw_dir       = <repo>/data/raw/CICIoT2023_CSV_DOWNLOADED
output        = <repo>/data/processed/ciciot2023_labeled_full.parquet
manifest      = <repo>/data/processed/ciciot2023_labeled_full_manifest.json
chunk_size    = 500,000 rows per pandas read
limit_per_file = unlimited
```

Run the full build with:

```bash
python -m src.preprocessing.ciciot2023.build_labeled_ciciot2023_dataset
```

Useful options:

```bash
python -m src.preprocessing.ciciot2023.build_labeled_ciciot2023_dataset \
  --raw-dir data/raw/CICIoT2023_CSV_DOWNLOADED \
  --output data/processed/ciciot2023_labeled_full.parquet \
  --manifest data/processed/ciciot2023_labeled_full_manifest.json \
  --chunk-size 500000
```

For a bounded smoke run, `--limit-per-file N` limits rows read from each CSV. `--dry-run` performs discovery, validation, counting, and manifest writing without emitting Parquet. The current CLI has no `--force` flag or overwrite refusal; choose output paths carefully before starting a full run.

### 5.2 Step 1: discover and resolve shards

`resolve_shards(raw_dir)` performs these operations:

1. Fail immediately if `raw_dir` does not exist.
2. Iterate over the raw root in sorted directory order.
3. Ignore loose files such as `README_CSV.pdf`; only directories are candidates.
4. Require every directory to exist in `FOLDER_TO_LABEL`.
5. Collect an unknown-folder list and raise one error if any folder is unmapped. This prevents an entire class from silently disappearing.
6. Convert the folder name into the uppercase 34-class label using the inlined `FOLDER_TO_LABEL` dictionary.
7. Verify that the resulting label exists in `CATEGORY_MAP`.
8. Convert the fine label into one of eight category names.
9. Recursively find `*.csv` files under the class folder in sorted lexical order.
10. Store each shard as a `RawShard` containing its filesystem path, slash-normalized path relative to the raw root, folder, label, and category.
11. Fail if no CSV files were found.

The mapping is intentionally explicit rather than guessed from spelling. For example, `Benign_Final` maps to `BENIGN`, not to a label named `BENIGN_FINAL`.

### 5.3 Step 2: read each CSV in bounded chunks

`iter_shard_chunks()` calls:

```python
pd.read_csv(shard.path, chunksize=500_000)
```

For every chunk it:

1. Checks that all 39 canonical feature names are present.
2. Raises if any feature is missing.
3. Raises if the raw file already contains `Label`; this protects against accidentally merging two competing label sources.
4. Reorders the chunk to `FEATURE_NAMES`.
5. Drops any extra raw columns after the required-header check. The current on-disk files have no extras.
6. Applies `--limit-per-file` if requested, truncating only the final emitted chunk for that file.
7. Yields the chunk in the order returned by pandas. No shuffle or sort is applied to rows inside a CSV.

### 5.4 Step 3: minimum raw-stage cleaning

`process_shard()` applies only the cleaning needed to make the Parquet source finite and typed:

```python
chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
chunk = chunk.astype({feature: "float32" for feature in FEATURE_NAMES})
```

Consequences:

* A row with at least one NaN or positive/negative infinity in the 39 feature columns is removed.
* Relative order of surviving rows is preserved by `dropna()`.
* All feature values are stored as `float32`.
* No lower clipping, percentile clipping, integer rounding, binary rounding, scaling, sampling, or label encoding happens here.

Keeping this stage minimal prevents the full pipeline from applying cleaning twice with different thresholds. The authoritative clip and rounding stage is in `pipeline.py`.

The actual build manifest reports:

```text
rows read:       46,776,700
rows retained:   46,775,660
NaN/inf dropped:          1,040
```

### 5.5 Step 4: append labels and audit metadata

After cleaning and casting, the builder adds:

```python
chunk["Label"] = shard.label
chunk["category"] = shard.category
chunk["Label"] = chunk["Label"].astype("category")
chunk["category"] = chunk["category"].astype("category")
chunk["source_csv_filename"] = shard.source
chunk["source_folder"] = shard.folder
```

It then asserts the exact final order:

```text
FEATURE_NAMES + ["Label", "category", "source_csv_filename", "source_folder"]
```

This is the point at which the implicit folder label becomes an explicit dataset column.

### 5.6 Step 5: write Parquet incrementally

The first non-empty chunk creates one `pyarrow.parquet.ParquetWriter` with Zstandard compression. Later chunks reuse that writer. If a later chunk produces a schema that is not exactly equal to the first PyArrow schema, it is cast to the initial schema before writing; this prevents a per-shard dtype drift from corrupting the append operation.

The writer uses `preserve_index=False`, so pandas row indexes are not stored as a data column. There is no global dataframe concatenation, shuffle, aggregation, or sort.

On this dataset the largest retained file has 373,344 rows, below the 500,000-row chunk size, so the observed Parquet has 309 source-shard row groups: one row group per input CSV. The row group boundary is therefore also an efficient audit boundary.

### 5.7 Step 6: maintain counters and write the build manifest

The `Counter` tracks:

* total rows read;
* retained rows;
* NaN/inf rows dropped;
* rows by folder;
* rows by uppercase label;
* rows by category;
* rows by relative source filename;
* number of files seen.

After all shards finish, the builder verifies that the sums of the per-folder, per-label, and per-category counters all equal `kept_rows`. A mismatch raises instead of producing a questionable manifest.

The JSON manifest records the raw path, output path, dry-run setting, chunk size, row limit, feature names, expected columns, complete row-count breakdowns, and the complete folder-to-label mapping. It is the provenance record for the labelled Parquet stage.

---

## 6. Stage B — split before fitting or sampling

**Implementation:** `src/preprocessing/ciciot2023/pipeline.py` and `src/preprocessing/ciciot2023/splitter.py`

The pipeline first loads the complete labelled Parquet. It extracts:

```python
meta = df[["source_csv_filename", "Label", "category"]]
```

`compute_split()` converts the shard and label columns to NumPy arrays, detects contiguous shard runs, builds one split plan per fine label, assigns every global Parquet row a split code, and rejects any unassigned row.

Split codes are:

```text
0 = train
1 = validation
2 = test
```

The default fractions are `val_frac = 0.10` and `test_frac = 0.20`.

### 6.1 Why shard identity is retained

The raw files are size-bounded segments of a class stream, not random independent examples. `source_csv_filename` makes it possible to assign complete source shards to temporal holdouts and to audit whether a shard was split across incompatible boundaries.

`build_shard_runs()` requires each source filename to occupy exactly one contiguous run in Parquet row order. If a shard reappears later, it raises because block assumptions would be invalid. The builder's pure shard-by-shard append makes this invariant true.

### 6.2 Natural-numeric ordering

`natkey()` strips the trailing alphabetic extensions and splits digit runs before sorting. Thus:

```text
...Flood.pcap.csv      -> sequence 0
...Flood1.pcap.csv     -> sequence 1
...Flood2.pcap.csv     -> sequence 2
...Flood10.pcap.csv    -> sequence 10
```

The base file must sort before `1`, and `2` must sort before `10`. Lexical string sorting alone would get one or both cases wrong.

### 6.3 Per-class split dispatch

`plan_class_split()` chooses one of three protocols:

#### Classes with three or more shards: `forward_chain`

1. Sort the class's shard names using `natkey()`.
2. Take the latest `round(n * test_frac)` shards for test, with at least one test shard.
3. Take the preceding `round(n * val_frac)` shards for validation, with at least one validation shard.
4. Put the earliest remaining shards in train.
5. Require at least one train shard.

This creates an earliest-to-latest train/validation/test progression.

#### Classes with two shards: `two_shard_hybrid`

1. Natural-sort the two shards.
2. Assign the later shard entirely to test.
3. Split the earlier shard into contiguous train and validation blocks.
4. Renormalize the validation fraction onto the non-test portion so the validation target remains close to 10% of the whole class.

#### Classes with one shard: `block`

1. Keep the one shard's row order.
2. Use a contiguous `[0, n_train)` block for train.
3. Use the next block for validation.
4. Use the final block for test.
5. Enforce at least one row in every split.

This is weaker than a true shard boundary, but it is the only available temporal proxy for a one-file class.

### 6.4 Leakage guards

`assert_forward_chaining()` verifies:

* forward-chain train shards precede validation shards;
* validation shards precede test shards;
* no test shard is also in train or validation;
* block/hybrid ranges do not overlap and are ordered.

`compute_split()` additionally raises if any Parquet row remains at split code `-1`.

The saved archived run reports these protocol counts:

```text
forward_chain:      17 classes
two_shard_hybrid:    2 classes
block:              15 classes
```

### 6.5 Important row-order nuance

The source build enumerates filenames with lexical `sorted(...)`, while the splitter uses natural-numeric ordering in the plan. The split assignment is still by `source_csv_filename` run, so a shard receives the intended split even if the Parquet's inter-shard order is lexical. Within each individual CSV, pandas row order is preserved. For audits, use the source filename and the split plan rather than assuming every global Parquet row index is chronological across numbered shards.

---

## 7. Stage C — cleaning after the split

`pipeline.clean_features(X, train_mask)` runs after the split has been computed and before scaler fitting.

### 7.1 Train-only percentile bounds

For each feature, the pipeline computes the 99.99th percentile from
`X[train_mask]` only. Continuous/count fields use that bound directly.
Protocol/service/flag presence fields floor the effective upper bound at `1`
so a rare positive state cannot be clipped below its canonical domain value:

```python
q = np.percentile(X[train_mask], 99.99, axis=0).astype(np.float32)
upper = effective_clip_upper(q)
```

Computing these bounds on all rows would leak validation/test extremes into the
training transformation. Effective bounds are saved under `clip_upper_train`.

### 7.2 Lower/upper clipping

Every split is then transformed in place:

```python
np.clip(X, 0.0, upper, out=X)
```

This applies a zero lower bound and the effective feature-specific upper bound
to train, validation, and test.

### 7.3 Integer and presence canonicalization

After clipping:

* integer count fields are rounded and constrained to non-negative values;
* binary protocol/service fields and `*_flag_number` fields map any positive
  vendor value to `1`, otherwise `0`;
* `Protocol Type` is clipped but not canonicalized by this routine.

The positive-presence rule is required because the vendor CSV stores fractional
positive values in nominal indicator columns. Ordinary rounding previously
collapsed nine nonconstant source features to zero.

No row is imputed at this stage. NaN/inf rows were already removed during Parquet construction.

---

## 8. Stage D — train-only RobustScaler

The pipeline constructs `sklearn.preprocessing.RobustScaler()` and fits it only on cleaned training rows:

```python
scaler = RobustScaler()
scaler.fit(X[train_mask])
```

It then transforms each split separately. The resulting saved `X_train.npy`, `X_val.npy`, and `X_test.npy` arrays are scaled model-space features, not raw CSV values.

The scaler's center and scale are persisted as `scaler.pkl`. The run manifest also hashes the scaler's center and scale arrays for reproducibility.

The sampler reuses this same scaler for its continuous clustering features; it does not fit another scaler on a category, validation set, test set, or sampled subset.

---

## 9. Stage E — train-only cluster-proportional-floor sampling

**Implementation:** `pipeline.sample_train()` and `sampler.cluster_proportional_floor_sample()`

### 9.1 Fixed policy parameters

The current constants in `pipeline.py` are:

| Parameter | Value |
|---|---:|
| Category cap for each majority category | 200,000 rows |
| Rare categories kept whole | `BruteForce`, `Web` |
| DDoS KMeans clusters | 20 |
| DoS KMeans clusters | 20 |
| Mirai KMeans clusters | 10 |
| Recon KMeans clusters | 15 |
| Spoofing KMeans clusters | 15 |
| Benign KMeans clusters | 10 |
| Per-cluster floor | 500 rows |
| Within-cluster selection | `random_within` |
| Random seed | `config.paths.SEED = 42` |

The six capped categories are DDoS, DoS, Mirai, Recon, Spoofing, and Benign. BruteForce and Web are small enough to retain in full.

### 9.2 The 23 clustering features

Clustering excludes all 15 binary features and `Protocol Type`. The 23 retained dimensions are:

```text
Header_Length, Time_To_Live, Rate,
fin_flag_number, syn_flag_number, rst_flag_number,
psh_flag_number, ack_flag_number, ece_flag_number, cwr_flag_number,
ack_count, syn_count, fin_count, rst_count,
Tot sum, Min, Max, AVG, Std, Tot size, IAT, Number, Variance
```

For a category selected for clustering:

1. Find only that category's train rows.
2. Select the 23 columns from the cleaned, unscaled feature matrix.
3. Apply the already train-fitted RobustScaler center and scale for those columns.
4. Run seeded `MiniBatchKMeans` with the category's configured `k`, `batch_size=min(8192, n_rows)`, and `random_state=42`.
5. Count rows in each cluster.
6. Allocate the category budget with the floor rule.
7. Pick real source rows inside each cluster.
8. Map the selected category-local indices back to global Parquet row indices.

No synthetic centroid is ever written to a training array.

### 9.3 Allocation algorithm

For a category with `n_train > 200,000`:

1. **SMALL pool:** clusters with `size <= floor` are taken whole. They cannot supply 500 rows without taking all their rows.
2. Subtract those rows from the remaining budget.
3. **LARGE pool:** clusters with `size > floor` receive at least the floor.
4. Distribute the remaining budget proportional to cluster size.
5. Clamp every allocation to `[floor, cluster_size]`.
6. Use bounded water-filling plus integer largest-remainder placement so the allocations sum exactly to the target whenever feasible.
7. If the floor alone meets or exceeds the target, set `cap_not_binding=True`, keep every large cluster at the floor, and warn that the floor guarantee wins over the cap.

### 9.4 Within-cluster row selection

With `selection_mode="random_within"`, the sampler creates a seeded NumPy permutation of each cluster's real row indices and keeps the first allocated count. The final selected indices are sorted globally before they are returned.

The optional `nearest_centroid` mode exists in the sampler but is not the current pipeline setting. It would keep the rows nearest to each KMeans centroid rather than a seeded uniform-within-cluster selection.

### 9.5 What is not sampled

* Validation rows are never passed to KMeans or removed.
* Test rows are never passed to KMeans or removed.
* Rare BruteForce and Web train rows are kept whole.
* Any category at or below its cap is kept whole.
* No synthetic data is generated.
* No category is globally rebalanced after the split.

The archived run's observed totals were:

```text
train before sampling: 32,972,198
train after sampling:   1,226,533
validation:              5,555,150
                         8,248,312 test
```

The train total decomposes as six 200,000-row capped categories plus the full BruteForce and Web train pools:

```text
6 * 200,000 + 9,146 BruteForce + 17,387 Web = 1,226,533
```

These are archived-run values, not a claim that the current top-level output arrays are present today.

---

## 10. Stage F — labels, encodings, and class weights

### 10.1 Three target views

For every saved split, `y_all(idx)` creates:

1. `y34`: `LabelEncoder` output for the 34 uppercase fine labels.
2. `y8`: `LabelEncoder` output for the eight category names.
3. `ybin`: `0` for labels mapped to `Benign`, `1` for every non-benign label.

The fine-label encoder and category encoder are fit on the full label arrays so all known classes have stable names. The actual training arrays use the selected train indices; validation/test use their unsampled split indices.

### 10.2 Class weights

Class weights are computed after train sampling:

```python
compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
```

Three arrays are saved:

```text
class_weights_34.npy
class_weights_8.npy
class_weights_2.npy
```

Therefore the weights describe the distribution actually seen by training, not the original 46.8-million-row distribution and not the natural validation/test distribution.

---

## 11. Stage G — generated model-ready artifacts

By default, `pipeline.py` writes into `config.paths.PROCESSED_DIR`, which resolves to `data/processed/`. Pass `--output-dir <path>` to keep a complete regenerated artifact set in an isolated directory; the canonical labelled-Parquet input remains unchanged.

### 11.1 Split arrays

For each `train`, `val`, and `test` split:

| File pattern | Contents | Shape/type |
|---|---|---|
| `X_<split>.npy` | cleaned and RobustScaler-transformed 39 features | `N x 39`, `float32` |
| `y_<split>.npy` | 34-class encoded label | `N`, `int32` |
| `y_<split>_cat.npy` | 8-category encoded label | `N`, `int32` |
| `y_<split>_bin.npy` | benign/attack target | `N`, `int32` |

`X_train.npy` contains only the sampled train rows. `X_val.npy` and `X_test.npy` contain all natural validation/test rows.

### 11.2 Supporting artifacts

| File | Meaning |
|---|---|
| `train_kept_indices.npy` | global labelled-Parquet row indices retained for sampled train |
| `class_weights_34.npy` | balanced weights for fine-label training |
| `class_weights_8.npy` | balanced weights for category training |
| `class_weights_2.npy` | balanced weights for binary training |
| `scaler.pkl` | train-fitted `RobustScaler` |
| `label_encoder.pkl` | 34-class `LabelEncoder` |
| `category_encoder.pkl` | 8-category `LabelEncoder` |
| `class_names.json` | ordered fine-label encoder classes |
| `category_names.json` | ordered category encoder classes |
| `class_to_category.json` | serialized 34-to-8 mapping |
| `run_manifest.json` | complete run parameters, split plans, clip bounds, sampling details, counts, hashes, and timing |

### 11.3 Run manifest fields

The pipeline manifest records:

* UTC timestamp and current Git commit hash when available;
* seed;
* sampling method and split protocol;
* validation/test fractions;
* caps, rare whole categories, KMeans `k`, floor, and selection mode;
* exact continuous clustering feature list;
* clip percentile and every train-derived upper bound;
* counts of each split protocol;
* per-class shard lists and row totals;
* natural split row counts;
* train count before/after sampling;
* per-category sampling records including cluster sizes and allocations;
* class-weight ranges;
* hashes of `X_train`, `X_val`, `X_test`, and scaler center/scale;
* runtime.

This manifest is the audit point for reproducing a particular pipeline run.

### 11.4 What the current pipeline does not write

The current `pipeline.py` source inspected here does **not** write `perturbation_mask.npy`, `near_zero_iqr_features.json`, or `netdiffuser_categorization.json`. Those files appear in the archived `data/processed/old/` run and in older design documentation, but they are not emitted by the current live pipeline implementation. They must not be described as current outputs unless the relevant code is restored or a separate producer is run.

---

## 12. Diagnostics and verification reports

**Implementation:** `src/preprocessing/ciciot2023/reports.py`

The reports module has four command-line modes:

```bash
python -m src.preprocessing.ciciot2023.reports diagnostics
python -m src.preprocessing.ciciot2023.reports verify
python -m src.preprocessing.ciciot2023.reports evidence
python -m src.preprocessing.ciciot2023.reports sensitivity --device cuda --epochs 4
```

### 12.1 `diagnostics`

This mode is evidence for the fixed sampling parameters. It:

1. loads Parquet metadata;
2. recomputes the split;
3. loads the 23 continuous features;
4. fits a diagnostic RobustScaler on a train-only sample of at most 2,000,000 rows;
5. samples at most 50,000 train rows per majority category;
6. runs KMeans for `k in {10, 15, 20}`;
7. records cluster-size Gini/CV/max-min statistics;
8. computes cluster-to-fine-label crosstab purity;
9. chooses the `k` with highest purity, preferring the first/simpler value on ties;
10. classifies the category as multimodal when the selected cluster Gini is at least 0.30;
11. writes JSON, CSV crosstabs, and PNG figures.

The current `pipeline.py` constants correspond to the existing diagnostics conclusion that all six majority categories use cluster-proportional-floor rather than the random fallback.

Outputs are written under `data/processed/downsampling_diagnostics/`, including:

```text
multimodality.json
clustersize_<category>.png
crosstab_<category>.png
crosstab_<category>_k<k>.csv
```

### 12.2 `verify`

This mode recomputes metadata splits and checks:

1. forward-chain shard disjointness;
2. every fine class appears in train, validation, and test;
3. row-order constraints for block/hybrid classes;
4. validation/test counts equal the unsampled plan totals;
5. `y_train.npy` length equals `run_manifest["train_after_sampling"]`.

It requires a current `data/processed/run_manifest.json` and current model-ready arrays. If only the archived `data/processed/old/` artifacts exist, move/recreate them according to the current path configuration before treating `verify` as a current-run check.

### 12.3 `evidence`

This mode writes:

* per-class natural split count CSV/PNG;
* PCA support-overlap figures comparing full and retained majority-category train rows;
* within-cluster pre/post histograms;
* `fidelity_summary.json` with within-cluster normalized median-shift summaries.

It reloads the saved `train_kept_indices.npy`, reconstructs the same cleaned/scaled space, and uses the same seed/KMeans settings so the figures correspond to the actual selection.

### 12.4 `sensitivity`

This mode compares the forward-chaining split with a row-level shuffled split while holding the model/training comparison settings fixed. It is an optional research diagnostic, not a prerequisite for producing the Parquet or the core arrays.

---

## 13. Tests that cover the split and sampler

The tests are under:

```text
src/preprocessing/ciciot2023/tests/test_splitter.py
src/preprocessing/ciciot2023/tests/test_sampler.py
```

The splitter tests cover:

* natural numeric ordering, including base/1/2/10 behavior;
* latest-shards-to-test forward chaining;
* rejection of two-shard input to the three-way primitive;
* contiguous one-shard block splitting;
* forward-chain plan validity;
* one-shard range coverage and non-overlap;
* two-shard hybrid behavior.

The sampler tests cover the proportional-floor allocation, small-cluster handling, determinism, and cap/floor edge cases.

Run them with:

```bash
pytest src/preprocessing/ciciot2023/tests/ -q
```

These are unit tests for split/sampling primitives. They do not replace a full-data build or a full pipeline run.

---

## 14. Reproduction procedure from an empty processed directory

The safest end-to-end order is:

### Step 0 — check inputs

```bash
python -c "from pathlib import Path; p=Path('data/raw/CICIoT2023_CSV_DOWNLOADED'); print(p.exists(), len(list(p.rglob('*.csv'))))"
```

Expected result: the root exists and 309 CSVs are found. Confirm that the files are the official 39-column, unlabelled shards, not the separate NF-TON-IOT data.

### Step 1 — build the labelled source

```bash
python -m src.preprocessing.ciciot2023.build_labeled_ciciot2023_dataset
```

Expected full-build accounting:

```text
309 files
46,776,700 rows read
1,040 NaN/inf rows dropped
46,775,660 rows retained
```

Inspect both:

```text
data/processed/ciciot2023_labeled_full.parquet
data/processed/ciciot2023_labeled_full_manifest.json
```

### Step 2 — optionally regenerate diagnostics

```bash
python -m src.preprocessing.ciciot2023.reports diagnostics
```

This recomputes evidence for the hardcoded current `k` values and writes diagnostic artifacts. It does not create model arrays.

### Step 3 — run the core pipeline

```bash
python -m src.preprocessing.ciciot2023.pipeline
```

To place all generated arrays and metadata in an isolated output bundle:

```bash
python -m src.preprocessing.ciciot2023.pipeline --output-dir outputs/ciciot2023
```

The pipeline reads the full Parquet, computes the split, cleans, fits the train-only scaler, samples only train, encodes all targets, and writes arrays and metadata to the selected output directory.

### Step 4 — run leakage/contract verification

```bash
python -m src.preprocessing.ciciot2023.reports verify
```

For a redirected artifact bundle, verify that same directory explicitly:

```bash
python -m src.preprocessing.ciciot2023.reports verify --processed-dir outputs/ciciot2023
```

Run this only after Step 3 has produced `run_manifest.json` and arrays in the default or explicitly selected artifact directory.

### Step 5 — optionally generate fidelity evidence

```bash
python -m src.preprocessing.ciciot2023.reports evidence
```

### Step 6 — run the unit tests

```bash
pytest src/preprocessing/ciciot2023/tests/ -q
```

Do not reverse the split, cleaning, scaler, and sampling order. In particular, do not fit the scaler or clip quantiles before the split, and do not sample validation/test.

---

## 15. Current artifacts actually observed in this checkout

The labelled source artifacts currently present at the time of inspection are:

| Artifact | Observed meaning |
|---|---|
| `data/processed/ciciot2023_labeled_full.parquet` | approximately 566.8 MB on disk; full labelled source |
| `data/processed/ciciot2023_labeled_full_manifest.json` | approximately 24 KB; 309 files, 34 folders, 46,775,660 retained rows |
| `data/raw/ciciot2023_full/ciciot2023_base.csv` | approximately 7.0 GB; 40 columns, 39 features plus `Label`; historical combined CSV |
| `data/processed/downsampling_diagnostics/` | diagnostic JSON/CSV/PNG outputs |
| `data/processed/old/` | archived model-ready arrays and older run metadata, not the path selected by current `config/paths.py` |

The saved labelled manifest contains a stale raw path string (`...data\\raw\\CICIoT2023_CSV`) that does not exist in the current checkout; the current code default is `data/raw/CICIoT2023_CSV_DOWNLOADED`. This is a provenance-path mismatch, not evidence that the current raw tree has fewer files. The row counts and folder/file breakdowns in the manifest match the inspected 309-file `_DOWNLOADED` tree.

The archived run manifest under `data/processed/old/run_manifest.json` records a completed older run with seed 42, six 200,000-row category caps, 32,972,198 pre-sampling train rows, 1,226,533 post-sampling train rows, 5,555,150 validation rows, and 8,248,312 test rows. It is useful as historical evidence, but current code writes to `data/processed/`, not `data/processed/old/`.

---

## 16. Data-integrity invariants to preserve

When changing this pipeline, preserve all of the following:

1. **Raw label provenance:** labels come from the explicit folder mapping; never infer a label from a loose filename string.
2. **Schema order:** the 39 feature names and positions remain unchanged.
3. **No hidden raw label merge:** reject raw CSVs that unexpectedly contain `Label`.
4. **Finite Parquet source:** remove NaN/inf rows once during the build and account for them.
5. **Source identity:** retain both `source_csv_filename` and `source_folder`.
6. **Shard contiguity:** append each shard as one contiguous run; never shuffle before split planning.
7. **Temporal split:** use natural-numeric suffix order and keep later segments out of train.
8. **Split first:** no train-derived statistic may see validation/test rows.
9. **Train-only clip bounds:** calculate percentile limits from train only.
10. **Train-only scaler:** fit `RobustScaler` only on cleaned train rows.
11. **Train-only sampling:** KMeans, allocation, and row selection operate only on train.
12. **Real-row sampling:** retain source rows, never synthetic centroids.
13. **Natural holdouts:** validation and test are saved without downsampling.
14. **Stable encoders:** persist encoder objects and their ordered name lists with the arrays.
15. **Auditability:** persist kept global indices, counts, parameters, and hashes in the run manifest.
16. **Path clarity:** distinguish the current `_DOWNLOADED` raw tree, the legacy combined CSV, and the unrelated NF-TON-IOT dataset.

The complete build is therefore not just “concatenate CSVs.” It is a labelled, schema-checked, finite, source-traceable Parquet construction followed by a leakage-controlled temporal split, train-derived feature cleaning/scaling, mode-aware train undersampling, and an explicit artifact contract for every downstream model and evaluation stage.
