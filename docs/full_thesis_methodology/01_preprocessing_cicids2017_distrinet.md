# 1. CICIDS2017-DistriNet Preprocessing (MAXIMUM DETAIL)

Single source file, no hidden dependencies:
**`src/preprocessing/preprocess_cicids2017_distrinet.py`** (≈898 lines). Run from repo root:
`python src/preprocessing/preprocess_cicids2017_distrinet.py`. Reads
`data/raw/CICIDS_2017_Distrinet/*.csv`, writes
`data/processed/CICIDS_2017_Distrinet/`.

Downstream consumers import only the **outputs** via `CICIDS2017Adapter`
(`src/datasets/cicids2017.py`); nothing recomputes statistics on the fly.

---

## What the implementation does

Deterministic pipeline, executed in `main()` (`:687-895`) in this exact order:

1. **Inventory + schema validation** (`validate_inventory :163-193`)
2. **Per-file load + row cleaning + label normalization + category mapping +
   unsupported-class drop + float32 conversion** (`load_clean_file :211-299`)
3. **Merge + chronological ordering + exact-duplicate removal** (`merge_clean_and_deduplicate :302-345`)
4. **Chronological within-source-label split** (`chronological_within_source_label_split :373-446`)
5. **Train-only RobustScaler fit** (`main :730-745`) — *first fitted statistic in the whole pipeline*
6. **Leakage audit** (`build_leakage_audit :464-565`)
7. **Per-split transform + save** (`save_split :581-644`)
8. **Train-only class weights + encoders + manifest** (`main :760-889`)

Order matters: **all filtering, mapping, and split membership happen before any
fitted statistic exists.** The scaler is fit strictly on training rows after the
split is frozen (`main` comment `:729` "All fitted preprocessing begins here, after
immutable split membership exists").

---

## Step-by-step trace

### 1. Inventory + schema validation (`validate_inventory`)
- Requires exactly the five files, keyed by name in `EXPECTED_FILES` (`:36-42`):
  Monday…Friday WorkingHours, with a fixed day-order (0–4) and an expected capture
  date (2017-07-03 … 07). Any missing/extra CSV → `ValueError` (`:168-171`).
- Column names normalized by stripping BOM/whitespace (`normalize_column_name :135`).
- **All five headers must be byte-identical** (`:175-176`); duplicate normalized
  names rejected (`:177-178`); required columns `Flow ID, Src IP, Dst IP, Timestamp,
  Label` must exist (`:180-183`).
- **Modelling columns** = every column except non-features
  (`NON_FEATURE_COLUMNS = {Flow ID, Src IP, Dst IP, Timestamp, Label}`, `:48`) and any
  `Unnamed:` index column (`:185-190`). Result: **79 modelling features**.

### 2. Per-file cleaning (`load_clean_file`)
Reads each CSV, re-checks header equality (`:221-222`), then for every raw row (raw
row numbers start at 2 to match spreadsheet line numbers, `:225`):

- **Label normalization** (`normalize_label :139-147`): strip, canonicalize dash
  variants (`\x96`, en/em dash → `-`), and upper-case-compare to fold `BENIGN`.
- **Attempted policy** (`source_label_for_mapping :150-155`): DistriNet's
  `" - Attempted"` suffix is **not** a separate target. Default
  `--attempted-policy benign` maps `X - Attempted → BENIGN`; the alternative
  `parent` strips the suffix to the parent attack. (Manifest records the choice.)
- **Source→category map** (`SOURCE_TO_CATEGORY :54-64`): 9 source labels → 5
  categories (see table below). Unmapped source labels get `""`.
- **Timestamp parsing** (`parse_timestamps :196-208`): primary format
  `"%d/%m/%Y %I:%M:%S %p"` (UTC), with a `format="mixed", dayfirst=True` fallback;
  reports missing/strict-fail/mixed-recovered/unparseable counts.
- **Numeric coercion**: the 79 modelling columns → `pd.to_numeric(errors="coerce")`
  → float64 (`:236-237`).

**Cleaning rules (exact, per row)** (`:239-245`):
| Predicate | Meaning |
|---|---|
| `finite_numeric` | all 79 features finite (no NaN/Inf) |
| `valid_timestamp` | timestamp parsed |
| `negative_physical` | any feature `< 0` |
| `supported_category` | mapped category ≠ `""` |

`clean = finite_numeric ∧ valid_timestamp ∧ ¬negative_physical`;
`valid = clean ∧ supported_category`; only `valid` rows are kept.
There is **no imputation, no winsorization, no balancing** (manifest
`cleaning_policy`, `:842-849`). Kept features are cast to **float32 before dedup**
(`:271`) — "canonicalize to the exact float32 representation consumed by the models."

Per-file the code also **asserts the observed date equals the expected date**
(`:274-278`) — a strong guard that files are the right day.

Rich metadata is retained per kept row (`:248-268`): `sample_id`
(`"<file>:<row>"`), source file/day/day-order/row, `Flow ID/Src IP/Dst IP`, ISO
timestamp, `timestamp_epoch_seconds`, original/source/category labels, `is_attempted`.

### 3. Merge + order + exact-duplicate removal (`merge_clean_and_deduplicate`)
- Concatenate all files (`:306-308`).
- **Deterministic chronological order** via `np.lexsort` with keys
  `(source_row, source_day_order, timestamp_epoch_seconds)` — primary key is
  timestamp, ties broken by day then raw row (`:311-320`). This global order is what
  makes the later per-label split chronological.
- **Exact-duplicate removal** (`:322-331`): a temporary `__category_label_for_dedup__`
  column is appended so the duplicate key is *every float32 feature **plus** the
  category label*; `DataFrame.duplicated(keep="first")` keeps the earliest
  (chronologically first) occurrence. Definition recorded verbatim in
  `duplicate_audit.json`.
- Assigns a global `record_id`; asserts `sample_id` and `record_id` are globally
  unique (`:332-335`).

**Observed result** (`duplicate_audit.json`): 2,096,133 rows before → **15,754 exact
duplicates removed (0.752%)** → **2,080,379 rows** after.

### 4. Chronological within-source-label split (`chronological_within_source_label_split`)
Key design decision (manifest `split_policy.rationale`, `:851-857`): splitting is
done **within each of the 9 source attack labels**, not per category and not
globally. Rationale in code: aggregating first put *DoS GoldenEye* entirely in test;
closed-set classification needs every source label present in train.

For each source label (in `SOURCE_TO_CATEGORY` order):
- Take its row indices in the already global-chronological order (`:388`).
- `allocate_class_counts(n)` (`:348-370`): **largest-remainder 70/15/15** with **≥1
  row guaranteed per split**. `SPLIT_RATIOS = (0.70, 0.15, 0.15)`. Raises if a label
  has `< 3` rows.
- Slice the chronologically-ordered indices contiguously: first 70% → train, next
  15% → val, last 15% → test (`np.split` on cumulative boundaries, `:392-395`).
  Because indices are time-ordered, **train precedes val precedes test in time
  within every source label**.
- Warns if any partition `< --min-per-split-warning` (default 10).

Global assertions: the three splits are a **disjoint, exhaustive partition** of all
rows (`:425-427`). Category- and source-level distribution reports are built for
`class_distribution.csv` / `source_label_distribution.csv`.

### 5. RobustScaler fit (train only) (`main :730-745`)
- `train_matrix` = train rows × 79 features, C-contiguous float32.
- Train-constant columns are **reported, not removed** (`:735-741`): removing a
  train-constant field could erase a value present only in val/test and manufacture
  new cross-split duplicates. Observed constants:
  `Fwd URG Flags, Bwd URG Flags, URG Flag Count`.
- `sklearn.preprocessing.RobustScaler` fit on `train_matrix` only, pickled to
  `scaler.pkl`. RobustScaler centers on the **median** and scales by the **IQR**
  (q1–q3) per feature — robust to the heavy tails of flow features.

### 6. Leakage audit (`build_leakage_audit`) — **hard asserts**
- Computes 64-bit fingerprints of feature rows and of feature+label rows
  (`hash_pandas_object`, `:449-450`).
- For each split pair (train/val, train/test, val/test): asserts **no shared
  `sample_id`** (`:518-519`) and **no shared feature+label fingerprint**
  (`:520-521`). Feature-only overlaps are *allowed and reported* (an identical numeric
  vector under a *different* label is legitimate; note `:561-564`).
- **Within-source chronology asserted**: `train.max_ts ≤ val.min_ts ≤ test.min_ts`
  for every source label (`:544-552`), and every source label present in all three
  splits (`:533-536`). These raise `AssertionError` if violated → the pipeline
  cannot silently leak.

### 7. Per-split transform + save (`save_split`)
For each split:
- `pristine` = raw float32 feature matrix (unscaled).
- `scaled` = `scaler.transform(pristine)` float32.
- Targets via `encoded_targets` (`:568-571`): `y_cat` int8 (5-class id), `y_bin` int8
  (`category ≠ Benign`).
- **Guards**: no NaN/Inf in pristine or scaled (`:599-600`); no negative pristine
  value (`:601-602`); row alignment (`:603-604`); **scaler round-trip check** — inverse
  transform of ≤10k sampled rows within `1e-3 + 1e-5·max(|x|,scale)` tolerance
  (`:606-615`).
- Saves `X_{split}.npy` (scaled), `X_{split}_pristine.npy` (raw),
  `y_{split}_cat.npy`, `y_{split}_bin.npy`, `timestamp_epoch_seconds_{split}.npy`,
  and a `{split}.parquet` carrying metadata + pristine features.

### 8. Class weights, encoders, manifest (`main :760-889`)
- `balanced_class_weights` (`:574-578`): `w_c = N / (C · count_c)` on **train labels
  only**. Saved `class_weights_5.npy`, `class_weights_2.npy`.
- `label_encoders.json`: `{binary:{Benign:0,Attack:1}, category:CATEGORY_TO_ID}`.
- `preprocessing_manifest.json`: exhaustive provenance — input SHA-256 (unless
  `--skip-input-hashes`), raw+modelling column lists, timestamp/label/cleaning/split
  policies, duplicate + leakage audits, per-split reports, class weights, elapsed time,
  and an explicit `not_claimed: "global forward-time or independent attack-campaign
  generalization"`.

---

## The 79-feature schema and ordering

`FEATURE_NAMES` = the CSV modelling columns in file order (single source of truth is
the manifest `modelling_feature_names`; `FeatureManifest` mirrors it). First ten:
`Src Port, Dst Port, Protocol, Flow Duration, Total Fwd Packet, Total Bwd packets,
Total Length of Fwd Packet, Total Length of Bwd Packet, Fwd Packet Length Max,
Fwd Packet Length Min`; last five: `Active Min, Idle Mean, Idle Std, Idle Max,
Idle Min`. **Reordering corrupts every saved array, the scaler, and every
checkpoint** — the order is load-bearing.

Schema types (from the validator's inferred profile, doc 3): 48 integer, 25 numeric,
3 constant, 2 binary, 1 categorical.

---

## Source labels → final classes

| Source label (`SOURCE_TO_CATEGORY`) | Category | id |
|---|---|---|
| BENIGN | Benign | 0 |
| DoS Hulk / DoS GoldenEye / DoS slowloris / DoS Slowhttptest | DoS | 1 |
| DDoS | DDoS | 2 |
| PortScan | Recon | 3 |
| FTP-Patator / SSH-Patator | BruteForce | 4 |

`CATEGORY_NAMES = ("Benign","DoS","DDoS","Recon","BruteForce")`. Binary head:
`Attack = category ≠ Benign`.

---

## Final artifact shapes and counts (from `preprocessing_manifest.json`)

| Split | X shape | Benign | Attack | DoS | DDoS | Recon | BruteForce | Time span (UTC) |
|---|---|---|---|---|---|---|---|---|
| train | (1,456,265, 79) | 1,153,431 | 302,834 | 120,093 | 66,568 | 111,311 | 4,862 | 07-03 13:55 → 07-07 21:10 |
| val   | (312,058, 79)   | 247,164 | 64,894 | 25,733 | 14,265 | 23,853 | 1,043 | 07-04 15:02 → 07-07 21:13 |
| test  | (312,056, 79)   | 247,164 | 64,892 | 25,733 | 14,265 | 23,852 | 1,042 | 07-04 15:11 → 07-07 22:02 |

Dtypes: `X_*` and `X_*_pristine` float32; `y_*` int8; timestamps int64 epoch-seconds.
Train class weights (5-class): `[0.2525, 2.4252, 4.3753, 2.6166, 59.904]` (BruteForce
heavily up-weighted); binary: `[0.6313, 2.4044]`.

---

## Why it is designed this way

- **Filter/map before split** so the split operates on the final label space and every
  retained source label is representable in all three splits (avoids GoldenEye→test-only).
- **Chronological within-label split** approximates temporal generalization *per attack
  subtype* without starving any class — a compromise between a pure global forward-time
  split (which strands rare labels) and a random split (which leaks time).
- **float32 canonicalization before dedup** so duplicate detection matches exactly what
  the models will see (avoids float64/float32 mismatch producing phantom uniques).
- **RobustScaler** because flow features are heavy-tailed; median/IQR resist outliers.
- **Train-only everything** (scaler, weights) — the refactor principle: no statistic
  crosses the split boundary.

---

## Assumptions

- DistriNet timestamps are UTC and correct enough to order flows (they drive both the
  split and the leakage chronology asserts).
- The five specified files, with byte-identical headers and one calendar day each, are
  the complete dataset. Extra/missing files or header drift abort the run.
- "Attempted" flows are best folded into Benign (default) — a DistriNet-recommended,
  configurable choice, recorded in the manifest.
- Exact float32 + category equality is the right duplicate criterion (identical vectors
  with different labels are kept as legitimately ambiguous).

## Limitations (for threats-to-validity)

- The split is **not** a global forward-time split and **not** an independent-campaign
  split; the manifest explicitly disclaims those generalizations (`:796`). Temporal
  holdout is only *within* each source label; train/val/test windows overlap in
  wall-clock time across labels.
- BruteForce is tiny (4,862 train / ~1k val / ~1k test) → estimates for that class are
  unstable (the code warns on small partitions).
- Train-constant columns are retained (3 of them), i.e. carried as all-zero-variance
  features into the model input and scaler.
- No deduplication *across* label (feature-only duplicates persist by design), so a
  numeric vector can appear in multiple splits under different labels.
- Category mapping collapses attack subtypes (4 DoS variants → DoS; 2 Patators →
  BruteForce), discarding subtype granularity.

## What can safely be claimed

- A leakage-controlled, fully train-fit, deterministic (`SEED=42`) preprocessing with
  hard-asserted disjoint splits, exact-duplicate removal, and per-source-label temporal
  ordering, reproducible from committed SHA-256 provenance.
- 79 CICFlowMeter features, 5 closed-set categories (+binary), ~2.08 M deduplicated flows,
  70/15/15 within-label chronological split.

## What must NOT be claimed

- Do not claim global forward-time generalization or robustness to a new attack campaign.
- Do not claim the split is i.i.d. or that class balance was addressed (it was not; only
  class weights are provided).
- Do not describe the scaler/weights as fit on "the data" — they are **train-only**.
