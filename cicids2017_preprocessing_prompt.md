# CICIDS2017 Preprocessing Prompt / Plan

This document is a **self-contained specification** for preprocessing the CICIDS2017
dataset that already lives in this repo, written so it can be handed to an agent (or
followed by hand) and executed end-to-end. It is modelled on how **adversarial-attack
NIDS papers** preprocess CICIDS2017 (see §2, with sources), then adapted to this
thesis's constraints.

> **Hard requirement (do not violate):** *Keep every timing-based feature.* Timestamp,
> Flow Duration, all IAT features (flow/fwd/bwd), and all Active/Idle timing features
> are **retained**. Do **not** drop them. See §6 for the exact protected list. Many
> published pipelines discard `Timestamp` (and sometimes IATs); this pipeline
> deliberately does not, because they are needed for the temporal-validity and
> validity-gap analysis (see the thesis workflow doc and `CLAUDE.md`).

---

## 1. Objective and context

**Goal.** Turn the raw merged CICIDS2017 CSV into leakage-safe, model-ready,
constraint-annotated splits for: (a) baseline NIDS classifiers, (b) a VAE, and
(c) adversarial-attack + validity-gap analysis (FGSM/PGD/C&W → ASR_raw vs ASR_valid).
This mirrors the CICIoT2023 pipeline already in the repo (`CLAUDE.md`,
`src/preprocessing/`, `src/ciciot2023/`) so the two datasets can be compared.

**Input artifact (already produced — do not re-merge).**
- Path: `data/processed/CIC-IDS-2017/CICIDS2017_merged_raw.csv`
- Shape: **3,119,345 data rows × 86 columns** (85 original CICIDS2017 features +
  `source_file` provenance column).
- Encoding: **`latin-1`** (byte-preserving; the WebAttacks labels contain a raw
  `0x96` CP1252 en-dash — read/write with `latin-1` or normalize labels explicitly).
- Provenance / merge details: `data/processed/CIC-IDS-2017/CICIDS2017_Merging_Process.md`.
- Nothing has been cleaned, encoded, scaled, split, or balanced yet — this document
  covers all of that.

**Design principles (inherited from `CLAUDE.md`).**
1. **Fit on TRAIN only.** Every statistic/threshold/scaler/encoder/imputation value is
   computed on the train split; val/test are only transformed. No global describe(),
   no scaler fit on all data, no cross-split imputation.
2. **Split is computed FIRST**, before any fitting, using a leakage-safe temporal
   scheme (§9).
3. **Single source of truth for schema.** Emit one artifact (e.g.
   `cicids2017_schema.py` / manifest JSON) holding the fixed feature order, typing,
   constraints, and perturbability mask. Reordering it later corrupts every saved
   array/scaler/checkpoint — so freeze it once.
4. **Preserve a pristine (unscaled, unencoded) copy** of the feature matrix alongside
   the model-ready one, so adversarial artifacts can be validity-checked in the
   original units without inversion error (§10, §11).

---

## 2. How adversarial-attack papers preprocess CICIDS2017 (literature)

Common denominator across adversarial / robustness NIDS papers on CICIDS2017:

**Standard cleaning + encoding pipeline (the near-universal core):**
- Merge the 8 `GeneratedLabelledFlows` / `TrafficLabelling` CSVs into one table;
  strip stray whitespace/BOM from column names.
- **Delete missing and infinite values** (`Flow Bytes/s`, `Flow Packets/s` overflow to
  `Inf`/`NaN` when `Flow Duration = 0`).
- **Drop duplicate rows** and drop **zero-variance / constant** columns.
- **Label handling:** either binary (BENIGN vs ATTACK) or the 14/15-class multiclass;
  re-group / relabel attack families; apply a label encoder.
- **Normalization:** Min-Max to `[0,1]`, StandardScaler, or a **power transformer** to
  tame heavy-tailed flow features — *fit on train, applied to val/test*.
- **Class imbalance:** downsampling of BENIGN and/or oversampling (SMOTE) of rare
  classes (Heartbleed=11, Infiltration=36, SQL Injection=21).
  *(Sources: PLOS One 2022 "Adversarial attacks against supervised ML NIDS" —
  "deleting the missing and infinite values, re-grouping by attack type, label encoder
  and power transformer"; arXiv:2401.12262 oversampling + feature embedding.)*

**Feature selection (paper-dependent; this thesis does NOT aggressively select — see §5):**
- Correlation-based removal + PCA; permutation importance (69→10 features);
  RFE (removes ~80% of features); explainable-DL importance ranking.
  *(Sources: "Selection and Performance Analysis of CICIDS2017 Features Importance";
  ID-RDRL RFE model; DRL EMFFS filter ensemble.)*

**Adversarial-specific additions (what separates these from a plain IDS pipeline):**
- **Feature typing + validity/domain constraints** so perturbations stay realistic:
  which features are continuous vs integer vs binary flags vs immutable, plus range
  and relational constraints (e.g. min ≤ mean ≤ max, counts ≥ 0, ports in 0–65535).
  *(Source: "Adaptative Perturbation Patterns / A2PM", arXiv:2203.04234 — constraint-
  and category-aware adversarial perturbations for tabular NIDS.)*
- **Perturbation masks:** partition features into attacker-controllable vs frozen
  (immutable) sets; feature-level perturbation of flow records (VAM).
  *(Source: ACM ARES 2023 "A Case Study with CICIDS2017 on the Robustness of ML…".)*
- **Keep an unmodified reference sample** so the *same* adversarial example can be
  scored for evasion AND for validity (feature/relational/temporal constraints).
  *(Source: repo thesis workflow doc, steps 22–26.)*
- Model-depth / architecture robustness studies keep the full flow-feature set rather
  than reducing it, to study perturbation transfer. *(Source: arXiv:2510.19761.)*

**Takeaway for this thesis:** adopt the standard cleaning + train-only normalization +
imbalance handling, and add the adversarial machinery (typing, constraints, mask,
pristine copy). Diverge from the literature on two points: **(a) keep timing features**
(no dropping Timestamp/IAT/Active/Idle), and **(b) do not do heavy feature selection**
(the validity-gap study needs the native feature space and its inter-feature relations).

---

## 3. Input schema (86 columns, fixed order as merged)

Index → name (from the merged CSV header):

```
0  Flow ID                 22 Flow IAT Mean           44 Min Packet Length         66 Bwd Avg Packets/Bulk
1  Source IP               23 Flow IAT Std            45 Max Packet Length         67 Bwd Avg Bulk Rate
2  Source Port             24 Flow IAT Max            46 Packet Length Mean        68 Subflow Fwd Packets
3  Destination IP          25 Flow IAT Min            47 Packet Length Std         69 Subflow Fwd Bytes
4  Destination Port        26 Fwd IAT Total           48 Packet Length Variance    70 Subflow Bwd Packets
5  Protocol                27 Fwd IAT Mean            49 FIN Flag Count            71 Subflow Bwd Bytes
6  Timestamp               28 Fwd IAT Std             50 SYN Flag Count            72 Init_Win_bytes_forward
7  Flow Duration           29 Fwd IAT Max             51 RST Flag Count            73 Init_Win_bytes_backward
8  Total Fwd Packets       30 Fwd IAT Min             52 PSH Flag Count            74 act_data_pkt_fwd
9  Total Backward Packets  31 Bwd IAT Total           53 ACK Flag Count            75 min_seg_size_forward
10 Total Length of Fwd Pkt 32 Bwd IAT Mean            54 URG Flag Count            76 Active Mean
11 Total Length of Bwd Pkt 33 Bwd IAT Std             55 CWE Flag Count            77 Active Std
12 Fwd Packet Length Max   34 Bwd IAT Max             56 ECE Flag Count            78 Active Max
13 Fwd Packet Length Min   35 Bwd IAT Min             57 Down/Up Ratio             79 Active Min
14 Fwd Packet Length Mean  36 Fwd PSH Flags           58 Average Packet Size       80 Idle Mean
15 Fwd Packet Length Std   37 Bwd PSH Flags           59 Avg Fwd Segment Size      81 Idle Std
16 Bwd Packet Length Max   38 Fwd URG Flags           60 Avg Bwd Segment Size      82 Idle Max
17 Bwd Packet Length Min   39 Bwd URG Flags           61 Fwd Header Length (DUP)   83 Idle Min
18 Bwd Packet Length Mean  40 Fwd Header Length       62 Fwd Avg Bytes/Bulk        84 Label
19 Bwd Packet Length Std   41 Bwd Header Length       63 Fwd Avg Packets/Bulk      85 source_file
20 Flow Bytes/s            42 Fwd Packets/s           64 Fwd Avg Bulk Rate
21 Flow Packets/s          43 Bwd Packets/s           65 Bwd Avg Bytes/Bulk
```

**Known dataset quirks to handle (verify in code, do not assume counts blindly):**
- **Duplicate column name `Fwd Header Length`** appears at **index 40 and index 61**.
  This is a native CICIDS2017 defect. Deduplicate: keep index 40, drop index 61 after
  confirming they are (near-)identical; if they differ, keep both but rename
  (`Fwd Header Length`, `Fwd Header Length.1`) and document.
- **`Flow Bytes/s` (20)` and `Flow Packets/s` (21)`** contain `Inf` and `NaN`.
- **288,602 fully blank rows** (blank `Label`) originate from the WebAttacks file
  (documented in the merge doc §9.3). These are not a real class — drop them.
- **`Label` contains a raw `0x96` byte** in the three Web Attack labels
  (`Web Attack – Brute Force/XSS/Sql Injection`). Normalize to ASCII `-` on load.
- **Suspected zero-variance columns** (all-zero in CICIDS2017; confirm empirically):
  `Bwd PSH Flags`(37), `Fwd URG Flags`(38), `Bwd URG Flags`(39), `CWE Flag Count`(55),
  `Fwd Avg Bytes/Bulk`(62), `Fwd Avg Packets/Bulk`(63), `Fwd Avg Bulk Rate`(64),
  `Bwd Avg Bytes/Bulk`(65), `Bwd Avg Packets/Bulk`(66), `Bwd Avg Bulk Rate`(67).
  Drop only those that are **actually constant on the TRAIN split** (§8). None of these
  are timing features, so dropping them does not conflict with the timing requirement.

---

## 4. Column roles: Metadata / Model Features / Label

Split the 86 columns into three groups (mirrors workflow doc step 5, but timing stays
in Model Features).

- **Metadata (kept in a sidecar table, never fed to the model):**
  `Flow ID`(0), `Source IP`(1), `Destination IP`(3), `source_file`(85).
  These are identifiers/provenance. **`Timestamp`(6) is NOT metadata here** — it is
  retained as a model/analysis feature (see §6). Keep the metadata sidecar row-aligned
  (same index) with the feature matrix so any row can be traced back.
- **Label:** `Label`(84) → produce three arrays: `label_binary` (BENIGN=0, ATTACK=1),
  `label_multiclass` (15 classes), and **`label_category`** — a coarse grouping (the
  analog of CICIoT2023's 34→8 `CATEGORY_MAP`) used by the per-class β-VAE, which trains
  one model per category (see §11.5). Recommended grouping (confirm with the VAE owner):
  `Benign`; `DoS` (Hulk/GoldenEye/slowloris/Slowhttptest); `DDoS`; `PortScan`;
  `BruteForce` (FTP-/SSH-Patator); `WebAttack` (Brute Force/XSS/SQLi); `Bot`.
  `Infiltration`(36) and `Heartbleed`(11) are too small to train their own VAE — fold
  them into a `Rare`/`Other` group or exclude from per-class VAE training (still keep
  them in `label_binary`/`label_multiclass` for the NIDS classifiers).
- **Model Features:** everything else (Protocol, Ports, all flow statistics, all flag
  counts, and **all timing features**), minus dropped duplicate/constant columns.

> Ports (`Source Port`, `Destination Port`) and `Protocol` are kept as features but
> typed as categorical/immutable (§7) — they are network-layer identity, not freely
> perturbable.

---

## 5. Feature selection policy

**Do NOT perform aggressive dimensionality reduction** (no PCA, no RFE-to-10-features).
The validity-gap analysis requires the native feature space and its inter-feature
relations. The only columns removed are:
1. Metadata identifiers (moved to sidecar, §4).
2. The duplicate `Fwd Header Length` (index 61).
3. Columns that are **constant on the train split** (§3 candidates, verified in §8).

Record the exact removed list in the schema artifact with the reason for each.
(Optionally compute correlation/importance diagnostics for the thesis write-up, but do
**not** let them drop features from the modelling matrix.)

---

## 6. PROTECTED timing features — never remove

All of the following are **retained** (this is the user's hard requirement):

- `Timestamp`(6) — parse to a proper datetime; also derive numeric epoch seconds and
  keep the original string. Used for the temporal split (§9) and temporal-validity
  constraints (§10). Not scaled with the flow features.
- `Flow Duration`(7).
- Flow inter-arrival times: `Flow IAT Mean/Std/Max/Min`(22–25).
- Fwd IAT: `Fwd IAT Total/Mean/Std/Max/Min`(26–30).
- Bwd IAT: `Bwd IAT Total/Mean/Std/Max/Min`(31–35).
- Active timing: `Active Mean/Std/Max/Min`(76–79).
- Idle timing: `Idle Mean/Std/Max/Min`(80–83).

These pass through cleaning and scaling like other continuous features (except
`Timestamp`, handled specially), but are **never dropped** by any step — including the
zero-variance filter (a timing column that were somehow constant is still kept and
flagged, not removed).

---

## 7. Feature typing (for constraints, scaling, and the perturbation mask)

Assign each surviving feature a type (drives §8 scaling, §10 constraints, §11 mask).
Derive integer/binary determinations from the **train split**, not by hand-guessing.

- **Immutable / identity (frozen; attacker cannot change):** `Protocol`,
  `Source Port`, `Destination Port`, `Timestamp`. (Ports/Protocol define the flow's
  network identity; changing them changes what the flow *is*.)
- **Integer counts (≥ 0):** `Total Fwd Packets`, `Total Backward Packets`,
  packet-length totals, header lengths, `Subflow *`, `act_data_pkt_fwd`,
  `min_seg_size_forward`, `Init_Win_bytes_forward/backward`, all `* Flag Count`,
  `Fwd/Bwd PSH/URG Flags`, `Down/Up Ratio`.
- **Binary flags (0/1):** `Fwd PSH Flags`, `Bwd PSH Flags`, `Fwd URG Flags`,
  `Bwd URG Flags` (verify domain ⊆ {0,1} on train).
- **Continuous (≥ 0):** all rates (`Flow Bytes/s`, `Flow Packets/s`, `Fwd/Bwd
  Packets/s`), packet-length statistics, segment sizes, `Average Packet Size`, and
  **all timing features** (`Flow Duration`, IATs, Active/Idle).
- **Categorical:** `Protocol` (small finite set: 0/6/17 = HOPOPT/TCP/UDP), ports if
  bucketed.

Emit this as a typed manifest: `{feature: {role, dtype, min, max, non_negative,
integer, binary}}` computed on train.

---

## 8. Cleaning + scaling steps (execution order)

Do these **after** the split (§9) for anything that fits parameters; do row-level
cleaning before or consistently across splits (it is per-row, not fitted).

1. **Load** with `latin-1`; strip whitespace from column names (already done in merge,
   re-assert). Normalize `Label` `0x96` → `-`.
2. **Drop the 288,602 blank rows** (blank `Label` / all-blank feature row).
3. **Deduplicate the `Fwd Header Length` column** (drop index 61 if identical).
4. **Coerce feature columns to numeric.** Non-parseable → `NaN`.
5. **Handle Inf/NaN:** replace `±Inf` with `NaN`; for `Flow Bytes/s`/`Flow Packets/s`,
   the Inf arises when `Flow Duration = 0` — decide policy: either drop those rows or
   impute with a **train-fitted** value (e.g. train max/median) and log the choice.
   All imputation constants come from **train only**.
6. **Drop exact duplicate rows.** (Optional, standard in the literature; document the
   count. Consider deduping within the feature+label space, ignoring metadata.)
7. **Drop train-constant columns** (§3 candidates; timing features exempt, §6).
8. **Label encode:** build `label_binary` (BENIGN=0 else 1), `label_multiclass`, and
   `label_category` (the coarse per-class-VAE grouping, §4). Persist a `LabelEncoder`
   for each; emit the class-name lists as JSON.
9. **Timestamp:** parse to datetime; keep original + epoch-seconds; do **not** feed the
   raw datetime into the scaler (§6).
10. **Scale continuous features (fit on TRAIN only):** use **`RobustScaler`** — match
    the CICIoT2023 side for comparability, and because the VAE relies on the transform
    being **invertible and unbounded** (it inverse-transforms columns to build raw-space
    targets and keeps the decoder output activation `identity`; §11.5). Do **not** stack
    a Min-Max squash on top unless you also change the VAE decoder activation — the two
    must agree. Integers/binaries: leave raw or scale separately per typing. Persist the
    fitted scaler; transform val/test with it. **Also keep the pristine unscaled matrix**
    (§10/§11).

---

## 9. Train / validation / test split (compute FIRST, leakage-safe)

Match the CICIoT2023 scheme so results are comparable:

- **Forward-chaining temporal split** using `Timestamp` (and/or `source_file`
  day-order): earliest flows → train, latest → test, so the model is never trained on
  future traffic. Target ratios **val = 0.10, test = 0.20** (train = 0.70), same as
  `CLAUDE.md`.
- Compute the split **before** fitting any scaler/encoder/imputer/constraint.
- Preserve chronological order within each split; keep the metadata sidecar aligned.
- Report per-split row counts and per-split class distribution (binary + multiclass).
- **Imbalance handling AFTER split, on TRAIN only:** downsample BENIGN and/or oversample
  rare attacks (document `original rows → selected rows → reason`, workflow step 11).
  **Never** resample val/test — they must reflect the real distribution.

---

## 10. Validity constraints (for the adversarial validity gap)

Derive on **train**, store in the manifest. Three families (workflow steps 13, 24):

- **Feature (range/domain) constraints:** `Source/Destination Port ∈ [0, 65535]`;
  `Protocol ∈ {valid set}`; all counts/durations `≥ 0`; binary flags `∈ {0,1}`;
  each feature within `[train_min, train_max]` (or a tolerance band).
- **Relational constraints:** `Min Packet Length ≤ Packet Length Mean ≤ Max Packet
  Length`; `Fwd Packet Length Min ≤ Fwd Packet Length Mean ≤ Fwd Packet Length Max`
  (and Bwd); `Flow IAT Min ≤ Flow IAT Mean ≤ Flow IAT Max`; `Active/Idle Min ≤ Mean ≤
  Max`; totals consistent with counts × sizes where derivable.
- **Temporal constraints (why timing features are kept):** `Flow Duration ≥ 0`;
  `Fwd IAT Total ≤ Flow Duration` (+tolerance); IAT/Active/Idle non-negative and
  internally ordered; `Timestamp` monotic per flow key where applicable.

These become the gate for `ASR_valid` vs `ASR_raw` (validity gap = ASR_raw − ASR_valid).

---

## 11. Perturbability mask (attacker-controllable vs frozen)

For the attack stage, partition features into tiers (mirrors CICIoT2023 mask concept):

- **Frozen / immutable:** `Protocol`, `Source Port`, `Destination Port`, `Timestamp`
  (identity + timing anchor).
- **Full-perturbable:** continuous flow statistics and rates the attacker can plausibly
  influence by reshaping traffic.
- **Partial / dependent:** features functionally derived from others (means/stds,
  segment sizes, subflow aggregates) — perturb only consistently with their sources, or
  recompute post-perturbation.

Store the mask in the same manifest. Keep the **pristine unscaled feature matrix** so a
generated adversarial example can be (a) inverse-checked against §10 constraints in real
units and (b) scored for evasion, on the *same* sample, enabling the paired
2×2 contingency / McNemar analysis (workflow step 28).

---

## 11.5. VAE input contract — what the per-class β-VAE consumes

The generative model is `MixedInputBetaVAE` (`src/vae/`), trained as **one β-VAE per
category** (`src/vae/config.py`). Preprocessing must hand it exactly:

1. **A fixed-order numeric feature matrix** (`X_{train,val,test}`, `float32`, scaled),
   column order == the frozen schema. This order is load-bearing: it is baked into the
   scaler, every saved array, and every checkpoint.
2. **A per-split integer category array** (`label_category`, §4). The VAE subsets `X`
   by this to train each per-class model, so every category fed to a VAE must have
   **enough train rows to fit a 16-dim-latent model** (CICIoT2023 uses a floor of ~500;
   this is why Heartbleed/Infiltration can't get their own VAE — §4).
3. **A train-fitted, invertible `RobustScaler`** (`scaler.pkl`). The VAE inverse-
   transforms individual columns to recover raw-space reconstruction targets and maps
   categorical values into scaled space — a non-invertible or bounded transform breaks
   this (§8 step 10).
4. **A feature partition** — index lists grouping columns by role, the analog of
   `src/vae/schema.py:get_partition`: `continuous_idx`, `independent_binary_idx`
   (strict {0,1}), and the categorical column index(es) (`Protocol`, and `Ports` if
   bucketed). Emit this in the schema artifact.

> **Heads-up for the VAE owner (not the preprocessing team):** the current
> `MixedInputBetaVAE` is **CICIoT2023-specific** — 39 features hardcoded
> (`model.py`), a protocol allowlist `{0,1,2,6,17,47}`, and four *derived* one-hot
> columns (TCP/UDP/ICMP/IGMP) reconstructed from protocol. CICIDS2017 has a single
> `Protocol` column (values ⊆ {0,6,17}) and **no** derived one-hot block. So the model
> must be generalized to be dataset-parameterized (feature count + partition + protocol
> handling) before it can train on CICIDS2017. Freeze the CICIDS2017 schema/partition
> only after this shape is agreed.

---

## 12. Output artifacts

Write to `data/processed/CIC-IDS-2017/` (and code to `src/preprocessing/cicids2017/`,
matching the CICIoT2023 layout under `src/preprocessing/ciciot2023/`):

- `cicids2017_schema.py` **or** `cicids2017_feature_manifest.json` — the single source
  of truth: fixed feature order, typing, **the VAE feature partition (§11.5)**,
  per-feature train stats, constraints, perturbability mask, and the list of removed
  columns + reasons. **Freeze once.**
- `X_train / X_val / X_test` (model-ready, scaled) + `y_*` — for each split emit
  `label_binary`, `label_multiclass`, **and `label_category`** (per-class-VAE grouping);
  `.npy` or parquet, column order == manifest order.
- `X_*_pristine` — unscaled, cleaned feature matrix (real units) for validity checks.
- `metadata_*` sidecar (Flow ID, IPs, Timestamp, source_file) row-aligned to each split.
- Fitted `scaler` / `label_encoder` objects (train-fitted).
- `cicids2017_preprocessing_report.md` — counts at every step (rows dropped: blank,
  dup, Inf; constant columns removed; per-split sizes; class distributions; split
  boundaries/timestamps), for thesis reproducibility.

---

## 13. Validation checks (assert before declaring done)

- Column order in every saved array == manifest order (byte-stable).
- No `NaN`/`Inf` remain in model-ready matrices.
- **All §6 timing features present** in the final feature list (explicit assertion).
- No feature statistic (scaler/impute/constraint) was computed using val/test rows
  (leakage audit).
- Split is chronological and non-overlapping; val/test class distributions unresampled.
- Row counts reconcile: `blank_dropped + dup_dropped + kept == 3,119,345`.
- `label_binary` and `label_multiclass` counts sum to kept rows; BENIGN=0 mapping
  correct.
- Duplicate `Fwd Header Length` resolved; removed-column list matches manifest.

---

## 14. Reproducibility

- Global `SEED = 42` (repo convention, `config/paths.py`); seed numpy/torch and any
  sampler.
- Encapsulate in a rerunnable script (e.g. `scripts/preprocess_cicids2017.py`) that
  reads only the merged raw CSV and regenerates every artifact in §12 plus the report.
- Chunked/streaming reads (the CSV is ~1.28 GB / 3.1 M rows); do not require the whole
  frame in RAM if the environment is constrained (the merge used stdlib `csv`; pandas
  in chunks or polars/pyarrow are fine here).

---

### Sources (adversarial CICIDS2017 preprocessing)
- PLOS One 2022 — *Adversarial attacks against supervised ML NIDS*
  (delete missing/infinite, re-group by attack, label encoder + power transformer; GAN
  evasion/poisoning). https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0275971
- ACM ARES 2023 — *A Case Study with CICIDS2017 on the Robustness of ML against
  Adversarial Attacks in Intrusion Detection* (VAM feature-level flow perturbation).
  https://dl.acm.org/doi/10.1145/3600160.3605031
- arXiv:2203.04234 — *Adaptative Perturbation Patterns (A2PM): Realistic Adversarial
  Learning for Robust Intrusion Detection* (constraint/category-aware perturbations).
- arXiv:2510.19761 — *Exploring the Effect of DNN Depth on Adversarial Attacks in NIDS*.
- arXiv:2401.12262 — *ML-based NIDS for big/imbalanced data (oversampling, stacking
  feature embedding, feature extraction)*.
- *Selection and Performance Analysis of CICIDS2017 Features Importance* (permutation
  importance 69→10). https://dl.acm.org/doi/10.1007/978-3-030-45371-8_4
- ID-RDRL (RFE feature selection, PMC9470692); DRL DDoS EMFFS filter ensemble
  (SSRN 4331696) — feature-selection references (this thesis deliberately does not
  aggressively select; §5).
