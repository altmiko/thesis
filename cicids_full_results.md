# CICIDS2017-DistriNet — Full Experiment Audit & Results (Single Source of Truth)

**Audit date:** 2026-09-24 · **Repo:** `E:/Shameem/thesis` · **Scope:** entire CICIDS2017-DistriNet adversarial-robustness pipeline.
**Method:** every number below was re-read from raw artifacts (JSON/CSV/NPZ/parquet/.pt/.pkl/logs); each is cited to its source path. Nothing is taken from memory. `MISSING` = artifact absent. `[INFERENCE]` = derived, not directly stored.

> **Top-line reality check (read first).**
> 1. Active neural **classifiers** = SimpleMLP, CNNOnly, FT-Transformer (binary + 5-category). All reach ~0.984 test accuracy.
> 2. Active **attack victims** = **mlp + cnn (category head) ONLY**. FT-Transformer is trained but was **not attacked**. `serial`/`lstm` victims are **retired** (only in `old_root_files/`).
> 3. The **only** CICIDS2017 attack with real aggregated results is **PrimAttack** (primitive-domain). Feature-space **PGD/C&W baselines were NOT run** on CICIDS2017.
> 4. **PrimAttack essentially fails**: raw targeted→Benign ASR peaks at **10/4064 = 0.246 %** (padding/joint, maximum-evaluated), and **SP-ASR = 0 in every condition**. This is a *negative/robustness* result, not an attack-success result.
> 5. All experiments use a **single seed = 42** → no mean±std over seeds is derivable anywhere.

---

## 1. Dataset & Experimental Setup

### 1.1 Provenance
- Dataset: **corrected/relabelled DistriNet CIC-IDS-2017 five-file release** (not the original 8-file UNB `MachineLearningCSV`). Source: `data/raw/CICIDS_2017_Distrinet/` (`preprocessing_manifest.json:input_dir`).
- Raw files (processing order, SHA-256, raw rows) — `preprocessing_manifest.json:input_files/file_reports`, `docs/data/cicids2017distrinet/cicids2017distrinet_preprocessing.md`:
  | File | SHA-256 (prefix) | Raw rows |
  |---|---|---:|
  | Monday-WorkingHours.csv | `580bc5b3…` | 371,749 |
  | Tuesday-WorkingHours.csv | `59d60eff…` | 322,003 |
  | Wednesday-WorkingHours.csv | `820446eb…` | 496,779 |
  | Thursday-WorkingHours.csv | `3db32a8e…` | 362,368 |
  | Friday-WorkingHours.csv | `8bf5a792…` | 547,915 |
  Doc total raw ≈ 2,100,814 rows / 1,148,542,377 bytes.

### 1.2 Preprocessing pipeline (exact order)
Source: `scripts/preprocess_cicids2017_distrinet.py` + `preprocessing_manifest.json` (`methodological_description`, `cleaning_policy`, `split_policy`).
1. Validate inventory/schema (5 identical 84-column headers).
2. Row-local deterministic cleaning per file: drop non-finite numerics (`finite_numeric`), drop unparseable timestamps (`valid_timestamp`), drop negative physical values (`~negative_physical`), drop unsupported categories. **No imputation, no winsorization, no balancing** (`cleaning_policy.imputation=null, winsorization=null, balancing=null`).
3. Label normalization; source→category map; `Attempted`→BENIGN (`attempted_policy="benign"`).
4. Discard unsupported categories.
5. Canonical **float32** conversion of 79 features.
6. **Global exact-duplicate removal** on (float32 features + category_label) BEFORE split.
7. **Chronological 70/15/15 split within each source label**.
8. Encode binary + 5-category targets.
9. **Fit RobustScaler + class weights on TRAIN ONLY**; transform val/test.
10. Save parquet/npy/encoders/scaler/weights/audits; assert coverage, chronology, disjointness.
- Per-file drops (`file_reports`, nonfinite / negative_physical / unsupported): Mon 60/517/0, Tue 23/511/0, Wed 71/758/7, Thu 291/621/221, Fri 348/515/738. All bad-timestamp drops = 0.
- No downsampling (`max_rows_per_file=null`). Pipeline `elapsed_seconds=89.657`.

### 1.3 Feature set — **79 features** (authoritative order)
- Count 79 (`preprocessing_manifest.json:modelling_feature_count`; raw_column_count 84). Order source-of-truth = `modelling_feature_names`. Reordering invalidates scaler/arrays/checkpoints.
- Ordered list (idx 0→78): Src Port, Dst Port, Protocol, Flow Duration, Total Fwd Packet, Total Bwd packets, Total Length of Fwd Packet, Total Length of Bwd Packet, Fwd Packet Length Max/Min/Mean/Std, Bwd Packet Length Max/Min/Mean/Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean/Std/Max/Min, Fwd IAT Total/Mean/Std/Max/Min, Bwd IAT Total/Mean/Std/Max/Min, Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length, Bwd Header Length, Fwd Packets/s, Bwd Packets/s, Packet Length Min/Max/Mean/Std/Variance, FIN/SYN/RST/PSH/ACK/URG/CWR/ECE Flag Count, Down/Up Ratio, Average Packet Size, Fwd Segment Size Avg, Bwd Segment Size Avg, Fwd Bytes/Bulk Avg, Fwd Packet/Bulk Avg, Fwd Bulk Rate Avg, Bwd Bytes/Bulk Avg, Bwd Packet/Bulk Avg, Bwd Bulk Rate Avg, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets, Subflow Bwd Bytes, FWD Init Win Bytes, Bwd Init Win Bytes, Fwd Act Data Pkts, Fwd Seg Size Min, Active Mean/Std/Max/Min, Idle Mean/Std/Max/Min.
- Constant columns (reported, **not** removed): Fwd URG Flags, Bwd URG Flags, URG Flag Count (all 0.0).
- Timestamp is **not** a feature (`timestamp_policy.classifier_input=false`); `timestamp_epoch_seconds_*.npy` used only for ordering/audit.

### 1.4 Classes & mapping (9 source labels → 5 categories)
`label_encoders.json` / `preprocessing_manifest.json:label_policy.source_to_category`:
- category ids: **Benign 0, DoS 1, DDoS 2, Recon 3, BruteForce 4**; binary: Benign 0 / Attack 1.
- BENIGN→Benign; DoS GoldenEye/Hulk/Slowhttptest/slowloris→DoS; DDoS→DDoS; PortScan→Recon; FTP-Patator/SSH-Patator→BruteForce.
- Dropped raw categories (not in the 5): Bot, Web Attack (BruteForce/XSS/SQLi), Infiltration, Heartbleed, and all `- Attempted` variants (doc §5.1: 9,144 Attempted rows = 0.4353 %, mapped to Benign).
- **Note:** this is a **9→5** map. The 34→8 map in `CLAUDE.md` belongs to CICIoT2023 (different dataset).

### 1.5 Original per-source-label counts (post-clean, post-dedup) — `source_label_distribution.csv`
| Source label | Category | Total | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| BENIGN | Benign | 1,647,759 | 1,153,431 | 247,164 | 247,164 |
| DoS Hulk | DoS | 158,281 | 110,797 | 23,742 | 23,742 |
| DoS GoldenEye | DoS | 7,562 | 5,294 | 1,134 | 1,134 |
| DoS slowloris | DoS | 3,974 | 2,782 | 596 | 596 |
| DoS Slowhttptest | DoS | 1,742 | 1,220 | 261 | 261 |
| DDoS | DDoS | 95,098 | 66,568 | 14,265 | 14,265 |
| PortScan | Recon | 159,016 | 111,311 | 23,853 | 23,852 |
| FTP-Patator | BruteForce | 3,969 | 2,778 | 596 | 595 |
| SSH-Patator | BruteForce | 2,978 | 2,084 | 447 | 447 |

### 1.6 Final per-class train/val/test counts (**thesis Table 1**)
5-class — `class_distribution.csv`, cross-checked `preprocessing_manifest.json:split_reports`:
| Class | Total | Train | Val | Test |
|---|---:|---:|---:|---:|
| Benign | 1,647,759 | 1,153,431 | 247,164 | 247,164 |
| DoS | 171,559 | 120,093 | 25,733 | 25,733 |
| DDoS | 95,098 | 66,568 | 14,265 | 14,265 |
| Recon | 159,016 | 111,311 | 23,853 | 23,852 |
| BruteForce | 6,947 | 4,862 | 1,043 | 1,042 |
| **Total** | **2,080,379** | **1,456,265** | **312,058** | **312,056** |

Binary — `split_reports.*.binary_counts`:
| Split | Benign | Attack | Rows |
|---|---:|---:|---:|
| Train | 1,153,431 | 302,834 | 1,456,265 |
| Val | 247,164 | 64,894 | 312,058 |
| Test | 247,164 | 64,892 | 312,056 |

### 1.7 Split method, leakage, scaler, seed, hardware, software
- **Split:** ratios 0.70/0.15/0.15 (`split_policy.ratios`), **chronological forward-time within each retained source label**, `shuffle_before_assignment=false`, largest-remainder allocation. Verified per-label `train_max ≤ val_min ≤ test_min` (`leakage_audit.json:source_label_chronology_epoch_seconds`).
- **Leakage prevention:** `membership_is_disjoint_and_exhaustive=true`. Pairwise overlaps (`leakage_audit.json:pairwise`): train↔test 0, train↔val 0, val↔test **1 feature-only fingerprint** (`11782720952135587951`; val=Benign vs test=Recon → **different labels**, feature+label overlap = 0). Exact-duplicate removal (`duplicate_audit.json`): 15,754 removed; 2,096,133→2,080,379 (0.7515744468504623 %), removed globally before split.
- **Scaler:** `sklearn.preprocessing.RobustScaler`, fit on **train only** (`scaler_fit_rows=1,456,265`, 79 features), `fitted_statistics_used_before_split=false`; `data/processed/CICIDS_2017_Distrinet/scaler.pkl` (sha256 `c04ed70…`). Input transform inside models is `asinh` (see §2).
- **Class weights** (`balanced`, `n_samples/(n_classes·class_count)`, train only; `class_weights_{2,5}.npy`, `class_weights.json`):
  - binary: Benign 0.6312752962112427, Attack 2.4043948650360107.
  - category: Benign 0.252510130405426, DoS 2.4252288341522217, DDoS 4.375270366668701, Recon 2.616569757461548, BruteForce 59.90394973754883.
- **Seed:** 42 (`preprocessing_manifest.json:seed`; `config/paths.py:SEED=42`).
- **Hardware (preprocessing):** MISSING in preprocessing manifest (CPU pandas/numpy job; no device string). Downstream training device = `cuda / NVIDIA GeForce RTX 4070 Ti SUPER` (`outputs/cicids2017distrinet/classifier_run_manifest.json:device,gpu`).
- **Software:** `environment.yml` — python 3.11, scikit-learn 1.9.0, numpy 2.4.4, pandas 3.0.3, pyarrow 24.0.0, scipy 1.17.1, torch 2.5.1+cu121, torchvision 0.20.1+cu121, torchattacks 3.5.1, adversarial-robustness-toolbox 1.20.1, lightgbm 4.6.0. Classifier manifest env: torch 2.5.1+cu121, numpy 2.4.4, sklearn 1.9.0, python 3.11.15.

---

## 2. Victim Models

Two artifact groups, **identical preprocessing** (same `preprocessing_manifest_sha256=df93d07…`, `label_encoders_sha256=7aeb1c…`, 79 features, same split rows, same class weights):
- `outputs/cicids2017distrinet/` — SimpleMLP (`mlp`) + CNNOnly (`cnn`), run 2026-09-21.
- `outputs/cicids2017distrinet_ft/` — FT-Transformer (`ft_transformer`), run 2026-09-24.
Each trained for **binary** (2-class) and **category** (5-class). **Single seed = 42** → report as single-seed (no mean±std).

### 2.1 Architectures (`src/classifiers/models.py`, `ft_transformer.py`)
- **SimpleMLP:** asinh → Linear(79→256)→ReLU→Dropout(0.3)→Linear(256→128)→ReLU→Dropout(0.3)→Linear(128→64)→ReLU→Dropout(0.3)→Linear(64→K).
- **CNNOnly:** asinh → (B,1,79) → Conv1d(1→32,k3,same)→ReLU→Conv1d(32→64,k3,same)→ReLU→AdaptiveMaxPool1d(8)→flatten(512)→Linear(512→64)→ReLU→Dropout(0.3)→Linear(64→K).
- **FTTransformer:** asinh → per-feature NumericalFeatureTokenizer (d_token=192) + learned CLS → 3 PreNorm blocks {MHSA(8 heads, attn_dropout 0.2) + ReGLU FFN(d_hidden 256), residual_dropout 0.0; first block skips attn LayerNorm} → Head LayerNorm→ReLU→Linear(192→K).

### 2.2 Architecture + hyperparameters + checkpoint (**thesis Table 2**)
Param counts = code-computed `sum(p.numel())` (stored in each `metrics/*.json`), analytically re-derived and matched to `.pt` sizes.
| Model×Task | Params | Checkpoint | Optimizer | LR | WD | Best epoch | Epochs run | Train sec |
|---|---:|---|---|---|---|---:|---:|---:|
| mlp-binary | 61,762 | `outputs/cicids2017distrinet/models/mlp_binary.pt` | Adam | 1e-3 | 0 | 7 | 10 | 128.60 |
| cnn-binary | 39,298 | `…/models/cnn_binary.pt` | Adam | 1e-3 | 0 | 8 | 10 | 119.32 |
| mlp-category | 61,957 | `…/models/mlp_category.pt` | Adam | 1e-3 | 0 | 9 | 10 | 124.26 |
| cnn-category | 39,493 | `…/models/cnn_category.pt` | Adam | 1e-3 | 0 | 8 | 10 | 118.69 |
| ft-binary | 922,370 | `outputs/cicids2017distrinet_ft/models/ft_transformer_binary.pt` | AdamW | 1e-4 | 1e-5 | 3 | 6 (early-stopped) | 918.60 |
| ft-category | 922,949 | `…_ft/models/ft_transformer_category.pt` | AdamW | 1e-4 | 1e-5 | 7 | 10 | 1523.18 |

Shared training config (`classifier_run_manifest.json`, `cicids2017d_experiments.py`): **batch 2048; epochs 10; early-stopping patience 3; grad-clip norm 5.0; loss = class-weighted CrossEntropyLoss (`balanced`); LR scheduler ReduceLROnPlateau(mode=max, patience=1, factor=0.5) on val macro-F1; checkpoint = best val macro-F1 (val-loss tie-break); seed 42, cudnn deterministic; `checkpoint_reload_verified=true`.** MLP/CNN dropout 0.3. FT: n_blocks 3, d_token 192, heads 8, attn_dropout 0.2, ffn_multiplier 1.3333 (d_hidden 256), ffn_dropout 0.1, residual_dropout 0.0, no-decay AdamW groups for tokenizer/CLS/bias/LayerNorm. *(FT manifest `run_config.learning_rate` shows 1e-3 cosmetically; effective FT lr = 1e-4 per `metrics/*.json` — MINOR mismatch.)*

### 2.3 Clean performance — VALIDATION vs TEST (**thesis Table 3**)
Source: `classifier_metrics_summary.csv` (both dirs). Validation and test are reported **separately** and are strikingly different (see flag below).

**VALIDATION**
| Model×Task | val_acc | val_bal_acc | val_macro_F1 | val_weighted_F1 | val_loss |
|---|---:|---:|---:|---:|---:|
| mlp-binary | 0.8791378525786873 | 0.92368543517112 | 0.8461142893083445 | 0.8877523972032351 | 1.705455 |
| cnn-binary | 0.8829929051650655 | 0.9261247239313026 | 0.8503403894128261 | 0.8911714350708599 | 1.545948 |
| mlp-category | 0.87889430810939 | 0.9694055688642856 | 0.8919290070000659 | 0.9005749105438551 | 2.747857 |
| cnn-category | 0.8615065148145538 | 0.96494533239725 | 0.8834140015856684 | 0.8877250938216706 | 1.682649 |
| ft-binary | 0.8692005973248563 | 0.9174009085643182 | 0.8353685882433658 | 0.8789599180984039 | 1.199383 |
| ft-category | 0.850348332681745 | 0.9621973993897669 | 0.8795917337940541 | 0.8795080366260392 | 1.377408 |

**TEST**
| Model×Task | test_acc | test_bal_acc | test_macro_F1 | test_weighted_F1 | test_loss |
|---|---:|---:|---:|---:|---:|
| mlp-binary | 0.9841759171430768 | 0.98767531270716 | 0.9765242699067992 | 0.9843526958091028 | 0.179932 |
| cnn-binary | 0.9847238957110264 | 0.9890383438157396 | 0.9773601900046478 | 0.98490193809849 | 0.113975 |
| mlp-category | 0.9844771451277976 | 0.9903090159013888 | 0.9772508929461512 | 0.9850375423142064 | 0.212104 |
| cnn-category | 0.9843297356884662 | 0.9891377462127052 | 0.9752776971829124 | 0.984887550316764 | 0.117592 |
| ft-binary | 0.9845508498474633 | 0.9883949814060486 | 0.9770870903276813 | 0.9847255600625521 | 0.141601 |
| ft-category | 0.9845796908247237 | 0.9909884335268953 | 0.9779606780539598 | 0.9851366155083475 | 0.151714 |

> **FLAG (all models, IMPORTANT):** validation is systematically *harder* than test (val macro-F1 ~0.84–0.89 vs test ~0.975–0.978). This is a property of the temporal within-label split (val/test drawn from later, differently-composed time windows). Checkpoints were selected on val macro-F1 yet test is ~0.13 higher everywhere. Macro precision/recall are not stored as scalar columns; compute from per-class table (§2.4) if needed.

### 2.4 Per-class precision/recall/F1 (TEST, category head) — `per_class_metrics.csv`
| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| mlp | Benign | 0.998995 | 0.981506 | 0.990173 | 247,164 |
| mlp | DoS | 0.997933 | 0.994171 | 0.996048 | 25,733 |
| mlp | DDoS | 0.998668 | 0.998458 | 0.998563 | 14,265 |
| mlp | Recon | 0.840024 | 0.996604 | 0.911640 | 23,852 |
| mlp | BruteForce | 0.999022 | 0.980806 | 0.989831 | 1,042 |
| cnn | Benign | 0.998761 | 0.981551 | 0.990081 | 247,164 |
| cnn | DoS | 0.998866 | 0.992500 | 0.995673 | 25,733 |
| cnn | DDoS | 0.998108 | 0.998528 | 0.998318 | 14,265 |
| cnn | Recon | 0.840259 | 0.996143 | 0.911585 | 23,852 |
| cnn | BruteForce | 0.984526 | 0.976967 | 0.980732 | 1,042 |
| ft | Benign | 0.998773 | 0.981741 | 0.990184 | 247,164 |
| ft | DoS | 0.999882 | 0.991800 | 0.995825 | 25,733 |
| ft | DDoS | 1.000000 | 0.999439 | 0.999720 | 14,265 |
| ft | Recon | 0.840625 | 0.997317 | 0.912291 | 23,852 |
| ft | BruteForce | 0.999026 | 0.984645 | 0.991783 | 1,042 |

Binary TEST per-class (`per_class_metrics.csv`): Attack precision ≈ 0.934–0.935, recall ≈ 0.994–0.996, F1 ≈ 0.963–0.964 for all three; Benign precision ≈ 0.998–0.999, recall ≈ 0.9817.
> **Weak class = Recon** (precision ~0.84 test, collapses to 0.34–0.39 on validation because Benign is misread as Recon). Recall stays ~0.996–1.0.

### 2.5 Confusion matrices (TEST) — `confusion_matrices.json` (both dirs)
Binary `[[Ben→Ben, Ben→Atk],[Atk→Ben, Atk→Atk]]`:
- mlp `[[242637,4527],[411,64481]]` · cnn `[[242629,4535],[232,64660]]` · ft `[[242669,4495],[326,64566]]`.
Category `[Benign,DoS,DDoS,Recon,BruteForce]` (rows=true):
- mlp `[[242593,53,2,4515,1],[150,25583,0,0,0],[12,0,14243,10,0],[64,0,17,23771,0],[18,0,0,2,1022]]`
- cnn `[[242604,27,0,4517,16],[193,25540,0,0,0],[21,0,14244,0,0],[64,1,27,23760,0],[23,1,0,0,1018]]`
- ft  `[[242651,2,0,4510,1],[211,25522,0,0,0],[8,0,14257,0,0],[63,1,0,23788,0],[16,0,0,0,1026]]`

### 2.6 Shared DoS-subclass blind spot — `dos_ddos_error_audit.csv`
**DoS Slowhttptest (support 261)** is misclassified as Benign by all three: mlp recall 0.4406 (benign-rate 0.5594), cnn 0.4368 (0.5632), ft 0.5172 (0.4828). Other DoS subclasses recall ≥0.86. Relevant to attack surface (an intrinsically evasive subclass).

### 2.7 Model consistency
`outputs/cicids2017distrinet` vs `…_ft`: identical features(79)/split_rows/class_counts/class_weights and identical preprocessing+label-encoder SHAs → FT is a legitimate drop-in on the same data. Differences are by design (later run; AdamW 1e-4/1e-5 vs Adam 1e-3/0; ~15× params; ~7–12× training time).

---

## 3. Attack Eligibility & Metric Definitions

### 3.1 Roster & targets (actually executed)
- **Attack victims = mlp, cnn (category head, 5 logits).** FT-Transformer is **not** an attack victim (no `victim_ft` in any `run_manifest.json`; test `VICTIMS=('mlp','cnn')`). `serial`/`lstm` retired.
- **Attack (source) classes:** DoS, DDoS, Recon, BruteForce (Benign excluded as source).
- **Target:** **targeted → Benign** (class id 0). `adv_pred==0` = targeted success. (No untargeted CICIDS2017 attack was aggregated.)

### 3.2 Eligibility & denominators
- **Universal denominator (all CICIDS2017 attacks):** clean-correct malicious test rows for the class: `denom = |{clean_pred == class_id}|` (`run_cicids2017_primitive_attack.py::evaluate_cell/_rate`; `run_cicids2017_vae_attacks.py::_evaluate`).
- **PrimAttack full run:** per-cell `test_limit_per_class = 512`, eligible = clean-correct among those. Per-cell eligible (`joint/maximum-evaluated/attack_results.json`): DoS 512/512 (both victims), DDoS 511/511, Recon mlp 511 / cnn 509, BruteForce mlp 500 / cnn 498. **Pooled eligible = 4064** (= mlp 2034 + cnn 2030 across 4 classes) — identical across all budgets/modes.
- **Feature-space PGD/C&W (CICIDS2017):** **no eligibility numbers — not run** (§4).
- **VAE latent attack (CICIDS2017):** **no aggregate** — only 2 smoke NPZ cells, no `attack_results.json`.

### 3.3 Metric definitions (mathematical, as implemented)
Let `C = {clean_pred == class_id}` (clean-correct, `denom=|C|`), `E = {adv_pred != class_id}` (evasion), `B = {adv_pred == 0}` (targeted-Benign).

**PrimAttack (`run_cicids2017_primitive_attack.py::_rate`, eligible = C):**
$$\text{Raw ASR} = \frac{|B \cap C|}{|C|}, \quad \text{Valid ASR} = \frac{|B \cap C \cap V|}{|C|}, \quad \text{Prim-feasible ASR} = \frac{|B \cap C \cap V \cap F|}{|C|}, \quad \text{SP-ASR} = \frac{|B \cap C \cap V \cap F \cap P|}{|C|}$$
where `V` = validator_v2 `hybrid_valid` (domain valid), `F` = primitive_feasible (hard-budget/dependency compliant), `P` = `semantic_status == PASS`. Nesting is strict: SP-ASR ⊆ Prim-feasible ⊆ Valid ⊆ Raw.

**Target-Benign ASR** = Raw ASR here (all PrimAttack successes are by definition `adv_pred==0`). "Raw ASR" in §5–7 tables = raw targeted-Benign ASR.

**VAE attack (`run_cicids2017_vae_attacks.py::_evaluate`, if ever aggregated):** `ASR_raw=|E∩C|/denom`, `targeted_benign=|B∩C|/denom`, layered `ASR_L0 ⊇ ASR_L0_L1 ⊇ ASR_L0_L1_L2`, `ASR_v2_hybrid_valid=|E∩C∩V|/denom`, `True_IDSR=|E∩C∩V∩ID|/denom` (ID via Mahalanobis).

**Feature-space input PGD (`run_cicids2017_input_baseline.py`, defined but not run):** `untargeted_asr=rate(E)`, `targeted_benign_asr=rate(B)`, `targeted_strict_valid_asr=rate(B∩mined_valid)`, `strict_validity=rate(mined_valid)`, `IDR=rate(in_dist)`.

---

## 4. Feature-Space Baseline Attacks (PGD, C&W)

> **CRITICAL FINDING:** **No feature-space PGD or C&W baseline was run on CICIDS2017-DistriNet.** There is therefore no baseline ASR/validity/perturbation result to pair against PrimAttack from the current pipeline.

Evidence:
- `src/attack/run_cicids2017_input_baseline.py` (the only CICIDS2017 feature-space runner; unconstrained targeted PGD) → default `--output-dir outputs/cicids2017_input_baseline` is **ABSENT**. It also has **stale mask keys** (`masks['benign'/'mined_valid'/'realizable']` not emitted by current `evaluate_cell`) → would `KeyError` → never successfully executed against current code. [INFERENCE]
- **No CICIDS2017 feature-space C&W runner exists.**
- `src/attack/input_baselines.py` (`input_pgd_attack`, `input_cw_attack`) and `constrained_input_baselines.py` / `run_constrained_input_baselines.py` are **CICIoT2023-only** (load `mlp_8class.pt`/`cnn_8class.pt`, import `preprocessing.schema` 39-feature + `vae.config`). They never touch CICIDS2017 victims.
- `src/attack/run_cicids2017_vae_attacks.py` produced only **2 smoke NPZ cells** (`outputs/cicids2017_vae_attacks_pave_run/attack_artifacts/DoS_mlp_A4.npz`, `outputs/cicids2017_vae_attacks_masked/attack_artifacts/DoS_mlp_A1.npz`); **no `attack_results.json`** → no aggregated ASR.

**Baseline parameters recorded in code (for reference / if re-run):**
| Attack | iters | ε (Linf, scaled) | step α | optimizer | κ/conf | C&W lr | binary-search | objective | targeted | mask/projection | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 input-PGD (`run_cicids2017_input_baseline.py`) | 50 | 0.5 | 0.05 | signed-grad PGD, random start | — | — | — | CE→Benign, `x←x−α·sign(∇)` | yes (→Benign) | Linf clamp `[x−ε,x+ε]`, all 79 feats | 42,43,44 |
| generic input-PGD (`input_baselines.py`, CICIoT2023 only) | caller | caller | caller | signed-grad, zero-budget passthrough at ε≤0 | — | — | — | untargeted CE ascent | no | Linf clamp | caller |
| generic input-C&W (`input_baselines.py`, CICIoT2023 only) | caller | — | — | Adam over δ | κ (default 0.0) | 0.01 | early-stop on Δδ<conv | `λ·clamp(margin+κ,0)+‖δ‖₂²` | no | — | caller |

**Archived-only baseline numbers (retired roster, use with caution):** the only place Input-PGD / VAE-latent ASR exist is the retired v2 report `old_root_files/retired_lstm_cnn_lstm_2026-09-24/outputs/v2_validity_report/` — evaluated on **mlp/cnn/lstm/serial** (includes retired victims), N=4055/method, denom=clean-correct:
| Method | Untargeted ASR | →Benign ASR | Hybrid-valid | v2-valid →Benign | In-dist | Valid+ID →Benign |
|---|---:|---:|---:|---:|---:|---:|
| Input PGD | 99.73% | 99.58% | **0.00%** | 0.00% | 0.00% | 0.00% |
| Primitive Direct | 60.12% | 54.16% | **100.00%** | 54.16% | 46.71% | 18.74% |
| VAE Latent Primitive | 8.85% | 8.85% | 100.00% | 8.85% | 97.51% | 8.78% |
| VAE Latent Raw | 68.80% | 68.66% | **0.00%** | 0.00% | 0.47% | 0.00% |
| VAE Latent Masked | 4.36% | 2.61% | 99.98% | 2.61% | 96.15% | 1.45% |
> These figures **predate the mlp/cnn-only roster and the current 512/class budget protocol**; they are **NOT** the reported CICIDS2017 result set. Treat as historical context only, or regenerate (see §17).

**Runtime / gradient-evaluation counts for any CICIDS2017 attack: MISSING** (not recorded).

---

## 5. PrimAttack — Configuration & Main Results

### 5.1 Threat model & primitives
White-box, **targeted malicious→Benign** flow-level primitive modification (`run_cicids2017_primitive_attack.py`, `run_manifest.json`). Packet-level verification unavailable (NullPacketBackend; real-world functionality preservation **NOT ESTABLISHED**). **Exactly two primitives:**
- **`p` — forward packet-length augmentation** (`realizability/cicids2017.py`): units bytes/fwd-packet, **discrete integer**, increase-only, identity 0.0, projection `round_to_integer_bytes_then_clamp_to_floor_budget`. Capability gate: Total Fwd Packet ≥1 ∧ Total Length of Fwd Packet >0 ∧ Fwd Packet Length Mean >0 (else masked to identity, fail-closed). Writes **12** dependent features (Total Length of Fwd Packet, Fwd Packet Length Min/Max/Mean, Fwd Segment Size Avg, Packet Length Min/Max/Mean, Average Packet Size, Packet Length Variance/Std, Flow Bytes/s).
- **`alpha` — forward inter-arrival timing dilation**: dimensionless ratio, **continuous**, increase-only (delay only), identity 1.0, projection `clamp_to_continuous_budget_then_quantize_derived_microseconds`. Capability gate: ≥2 fwd packets ∧ positive Fwd IAT Total. Writes **12** dependent features (Fwd IAT Total/Mean/Std/Max/Min, Flow Duration, Flow IAT Mean/Max, Flow Bytes/s, Flow Packets/s, Fwd Packets/s, Bwd Packets/s).
- **Dependency/recomputation `φ(x0,p,alpha)`:** mined algebraic identities, **0 % violation** on 400k–600k train rows. Fwd Packet Length Std is **proven invariant** under uniform shift. `dur_floor_us=1.0`, `SCALER_ATOL=1e-6`.
- **Immutable/held-constant:** all features φ does not write are asserted unchanged each run (`assert allclose(adv[:,frozen], raw[:,frozen], atol=1e-6)`). Frozen (F): ports, protocol, all backward-packet stats, packet counts, flag counts, header lengths, init-win bytes. Level-C held-constant (Fᶜ, unresolved): Fwd Act Data Pkts, Flow IAT Std/Min, Subflow Fwd Bytes, Fwd Bytes/Bulk Avg, Fwd Bulk Rate Avg, Fwd Packet/Bulk Avg, Active Mean/Std/Max/Min, Idle Mean/Std/Max/Min.

### 5.2 Optimizer / objective / budgets (full run)
Adam over 2 unconstrained leaves `u,v` (init `−2.0 + 0.5·N(0,1)`); `p = p_hi·σ(u)·pad_mask`, `alpha = 1 + (alpha_hi−1)·σ(v)·timing_mask`. Objective per sample `L = CE(f(x_adv),Benign) + cost_weight·(σ(u)·pad + σ(v)·timing)`. **Executed full-run config** (`run_manifest.json`): `steps=40, lr=0.1, cost_weight=0.01, init_noise=0.5, seed=42 (single), test_limit_per_class=512, optimizer='optimized'`. Scaling via train RobustScaler; `project_controls` clamps to hard box + integer-quantizes p. *(CLI defaults `seeds=42,43,44`, `test_limit=1024` were NOT used for the full sweep.)*

### 5.3 Budget calibration (train-only) — `src/attack/primattack_budget.py`, `artifacts/primattack/budget_calibration.json` (sha256 `7ce039a8…`)
Fit on `X_train_pristine.npy`+`y_train_cat.npy` ONLY (`fit_split='train'`; `selection_prohibited_inputs` = val/test features, victim preds, success). Levels ↔ train empirical quantiles: **restricted=p25, intermediate=p50, maximum-evaluated=p75**. Padding budget = round(quantile of positive-class `Fwd Packet Length Mean`); timing budget = quantile of `|FlowDuration − classMedian| / classMedian`. Envelope = global train p99 of 9 features.
**Per-class budgets (padding bytes / max relative duration change):**
| Class (n) | restricted (p25) | intermediate (p50) | maximum-evaluated (p75) |
|---|---|---|---|
| DoS (120,093) | 41 / 0.042726 | 47 / 0.567122 | 54 / 1.268226 |
| DDoS (66,568) | 2 / 0.221998 | 2 / 0.435296 | 3 / 0.687425 |
| Recon (111,311) | 2 / 0.085106 | 2 / 0.212766 | 10 / 0.531915 |
| BruteForce (4,862) | 11 / 0.061742 | 12 / 0.119895 | 91 / 0.233668 |

### 5.4 Main PrimAttack results (pooled, eligible = 4064) — `budget_sensitivity.csv/json`
| Mode | Budget | Raw ASR | Valid ASR | Prim-feas ASR | SP-ASR | med Δdur | med Δbyte | med rate-ret | med #feat | med prim-cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| timing-only | restricted | 0.0 | 0.0 | 0.0 | 0.0 | 0.000591 | 0.0 | 0.999409 | 11 | 6.150e-5 |
| timing-only | intermediate | 0.0 | 0.0 | 0.0 | 0.0 | 0.003394 | 0.0 | 0.996617 | 11 | 1.870e-4 |
| timing-only | maximum-evaluated | 0.0 | 0.0 | 0.0 | 0.0 | 0.006232 | 0.0 | 0.993806 | 11 | 4.981e-4 |
| padding-only | restricted | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.000838 | 1.0 | 10 | 0.002436 |
| padding-only | intermediate | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.001205 | 1.0 | 10 | 0.003903 |
| padding-only | maximum-evaluated | **0.0024606** | **0.0024606** | **0.0024606** | **0.0** | 0.0 | 0.002066 | 1.0 | 10 | 0.005951 |
| joint | restricted | 0.0 | 0.0 | 0.0 | 0.0 | 0.000617 | 0.000838 | 0.999384 | 20 | 0.003982 |
| joint | intermediate | 0.0 | 0.0 | 0.0 | 0.0 | 0.003638 | 0.001205 | 0.996376 | 20 | 0.004741 |
| joint | maximum-evaluated | **0.0024606** | **0.0024606** | **0.0024606** | **0.0** | 0.007007 | 0.002066 | 0.993042 | 20 | 0.012697 |

`0.0024606299212598425 = 10/4064`. **Raw = Valid = Prim-feasible in every row** → validator_v2 and hard feasibility rejected NONE of the classifier successes. **SP-ASR = 0.0 in all 9 conditions.** Constant across all conditions: semantic_pass 0.4818, semantic_fail 0.0217, not_fully_testable 0.4966, testability 0.5034.

### 5.5 Per-class × per-victim (maximum-evaluated, joint) — `joint/maximum-evaluated/attack_results.json`
| Class | Victim | Eligible | Raw n | Raw ASR | Valid ASR | Prim-feas | SP-ASR | Sem-pass | Not-testable |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | mlp | 512 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9668 | 0.0 |
| DoS | cnn | 512 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9668 | 0.0 |
| DDoS | mlp | 511 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9472 | 0.0 |
| DDoS | cnn | 511 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.9472 | 0.0 |
| Recon | mlp | 511 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| Recon | cnn | 509 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| BruteForce | mlp | 500 | 6 | 0.012 | 0.012 | 0.012 | 0.0 | 0.0 | 1.0 |
| BruteForce | cnn | 498 | 4 | 0.008032 | 0.008032 | 0.008032 | 0.0 | 0.0 | 1.0 |

**Recomputed:** pooled BruteForce raw ASR = (6+4)/(500+498) = 10/998 = 0.010020; pooled over all classes = 10/4064 = 0.0024606 ✓. Per-victim pooled: mlp 6/2034 = 0.295 %, cnn 4/2030 = 0.197 %. **All 10 successes are BruteForce, which is NOT_FULLY_TESTABLE → 0 survive the semantic gate.**

### 5.6 Random-feasible / smoke controls
`outputs/primattack_random_control/` (random-feasible optimizer, seed 42): ASR 0.0 across all classes. `outputs/primattack_smoke/` (test_limit 16, steps 2): 0.0. Older pilot `outputs/primattack_budget_sensitivity/` (eligible 254, cells_checked 4): 0.0 across all 9 conditions (**superseded** — do not cite).

---

## 6. Budget Experiments

Config names (do NOT rename — code uses these): **`restricted` (p25) · `intermediate` (p50) · `maximum-evaluated` (p75)**, numeric per-class limits in §5.3. Derivation = train empirical quantiles of padding-bytes and relative-duration (train-only, §5.3).

**Budget sensitivity (pooled, thesis Table 10)** — Target-Benign ASR = Valid ASR here (raw=valid always); values from §5.4:
| Budget | timing Raw/Valid/SP | padding Raw/Valid/SP | joint Raw/Valid/SP | sem-pass |
|---|---|---|---|---|
| restricted | 0/0/0 | 0/0/0 | 0/0/0 | 0.4818 |
| intermediate | 0/0/0 | 0/0/0 | 0/0/0 | 0.4818 |
| maximum-evaluated | 0/0/0 | **0.00246/0.00246/0** | **0.00246/0.00246/0** | 0.4818 |

**Primitive cost grows monotonically with budget** (median primitive cost: timing 6.1e-5→5.0e-4; padding 2.4e-3→6.0e-3; joint 4.0e-3→1.27e-2) but success does not — only the largest padding budget flips 10 BruteForce rows. Semantic pass/fail/testability are **budget-invariant** (0.4818/0.0217/0.5034), because failures are rate-retention (timing) and testability is structural (class-based), neither controlled by budget magnitude. Per-class results: only BruteForce nonzero, only at maximum-evaluated (§5.5).

---

## 7. Primitive Ablation (timing-only · padding-only · joint)

**Held constant:** identical source rows (deterministic seed 42+class_id), same victims (mlp,cnn), same target (Benign), same validator_v2 + semantic rules, same train calibration, same steps/lr/cost/init_noise/seed. **What changed:** timing-only forces p=0 (optimize α only); padding-only forces α=1 (optimize p only); joint optimizes both.

**Identical-source confirmation:** `source_id_consistency.json` `identical_across_all_configurations=true, cells_checked=8`; `budget_sweep_primitive.py::_assert_same_source_ids` (no AssertionError); constant `eligible_original_samples=4064`; per-class selection seed independent of mode/budget/victim.

**Ablation result (maximum-evaluated, thesis Table 9):**
| Mode | Raw ASR | Valid ASR | SP-ASR | med #feat changed | med Δbyte | med Δdur | med rate-ret |
|---|---:|---:|---:|---:|---:|---:|---:|
| timing-only | 0.0 | 0.0 | 0.0 | 11 | 0.0 | 0.006232 | 0.993806 |
| padding-only | 0.0024606 | 0.0024606 | 0.0 | 10 | 0.002066 | 0.0 | 1.0 |
| joint | 0.0024606 | 0.0024606 | 0.0 | 20 | 0.002066 | 0.007007 | 0.993042 |

**Conclusion:** **padding is the sole effective primitive; timing contributes nothing.** joint = padding for success (adds 0 raw and 0 SP over padding-only; McNemar b=c=0, Holm p=1.0). timing-only = 0 at every budget (no byte primitive). joint changes ≈ (timing 11 + padding 10) = 20 features (roughly additive). Timing-only leaves bytes unchanged (Δbyte 0); padding-only leaves duration unchanged (Δdur 0, rate-retention 1.0). No optimizer ablation beyond `optimized` vs `random-feasible` (random = 0, §5.6).

---

## 9. Domain Validity (validator_v2)

### 9.1 Spec — `validation/` package (docs/validator_v2_audit.md)
Dataset-agnostic rule-as-data engine; profiles per dataset. **235 rules** across 4 provenance classes (`from_profiles(dataset='cicids2017_distrinet')`):
| source_type | hardness | count | file |
|---|---|---:|---|
| SCHEMA | HARD | 133 | synthesized from `validation/schema/cicids2017_distrinet.yaml` |
| PROTOCOL | PROTOCOL | 79 | `validation/rules/cicids2017_distrinet/protocol_rules.yaml` |
| EXTRACTOR | HARD | 7 | `validation/rules/cicids2017_distrinet/extractor_rules.yaml` |
| MINED | EMPIRICAL | 16 | `validation/rules/cicids2017_distrinet/mined_rules.json` |
| **Total** | | **235** | |

- **Hard vs empirical:** HARD = SCHEMA (finite/type/domain) + EXTRACTOR (7 CICFlowMeter algebraic identities: Var=Std², AvgPktSize=PktLenMean, Fwd/Bwd SegSizeAvg, FlowPkts/s=Fwd+Bwd, TotLen Fwd/Bwd=count×mean). PROTOCOL = 79 nonnegativity facts. EMPIRICAL = 16 MINED invariants (8 monotone_chain Min≤Mean≤Max; 1 sum_equality PSH; 7 implication_zero Total-Bwd-packets==0⇒bwd fields==0).
- **Composition:** `hard_structural_valid = SCHEMA & EXTRACTOR & PROTOCOL`; **`hybrid_valid = hard_structural & MINED`** (the "domain-valid" flag used by attacks). Plausibility/in_distribution is **never** folded into structural validity.
- **Slack/tolerance:** `|obs−exp| ≤ abs + rel·|exp|`; floors 1e-6, caps abs 1e9 / rel 0.10. Approximate mined tolerances from train residual p99.9; logical rules fixed 1e-6.
- **Calibration & train-only:** discovery split = **train (1,456,265)**, confirmation = **validation (312,058)**, **test never used**. Accept thresholds: prefilter ≥0.99, train support ≥0.999, val support ≥0.995. Miner: 7586 candidates → 24 accepted → 16 retained (8 pruned as EXTRACTOR-dominated). SCHEMA facts `inferred_on: X_train_pristine`. Plausibility band fit on train (low_q 0.001/high_q 0.999); train min/max deliberately relegated to plausibility, NOT hard bounds.
- **Rule types (19 templates):** 9 approximate (equality/scaled/square/sqrt/sum/difference/product/ratio_equality, constant) + 10 exact/logical (finite/integer/binary/nonnegative/nonpositive/categorical/le/ge/monotone_chain/implication_zero/implication_pos).

### 9.2 Clean acceptance & synthetic detection
- Clean (20,000 genuine test flows, `clean_acceptance_report.md`): SCHEMA=PROTOCOL=EXTRACTOR=MINED=hard_structural=hybrid = **1.000000**; in_distribution = 0.974900 (separate gate). Zero structural rejections of genuine data.
- Synthetic known-invalid (`synthetic_violation_report.md`): mean detection **0.9998**; all 10 corruptions detect 1.0000 except `variance_subtle_2pct` = 0.9980.

### 9.3 Validity rates per attack
- **PrimAttack (current, mlp+cnn):** raw = valid = feasible in EVERY condition (`budget_sensitivity.csv`) → **validity rate = 100 % of the classifier successes**; validator rejected none. Max padding/joint valid ASR = 0.0024606 (10/4064); all else 0.
- **Feature-space PGD / VAE latent (current roster):** MISSING (not run). Archived-only rates (retired mlp/cnn/lstm/serial): Input PGD hybrid-valid **0.00 %** (fails SCHEMA integer rules on Bwd Init Win Bytes / Bwd Packet Length Max — non-integer perturbations); Primitive Direct & VAE Latent Primitive 100 %; VAE Latent Raw 0.00 %; VAE Latent Masked 99.98 %.

### 9.4 Paired raw-success vs valid-success
- PrimAttack: raw==valid in every cell (paired by construction).
- Archived v2: Input PGD raw 99.73 % untargeted but valid 0.00 % — the canonical "raw succeeds, validity kills it" contrast (retired roster).

---

## 10. Flow-Level Semantic Preservation

### 10.1 Spec — `src/attack/flow_semantics.py`
Four-level model: Level0 evasion → Level1 domain validity (validator_v2 hybrid_valid) → Level2 primitive feasibility → **Level3 SP proxy**. SP is separate from domain validity.
- **Immutable / required checks:** attack-label metadata unchanged; protocol unchanged; service ports unchanged; packet counts unchanged (Total Fwd/Bwd, Fwd Act Data Pkts, Subflow Fwd/Bwd Packets); all `*Flag*` unchanged; flow endpoints/direction unchanged (→NOT_TESTABLE if IP metadata absent); finite generated features; traffic volume not decreased; only declared primitive dependencies changed; primitive budget compliance.
- **Class-specific rules:** RateRetentionRule (DoS, DDoS) — adversarial Flow Packets/s ≥ class train **p05** (DoS 0.6247806906700134, DDoS 1.0654268741607666; `rate_retention_required=true`). ReconRule → NOT_TESTABLE {complete_scanned_port_set, scan_sequence, distinct_connection_attempt_count}. BruteForceRule → NOT_TESTABLE {authentication_attempt_count, credential_or_payload_semantics, server_authentication_outcome}.
- **Tolerance:** atol 1e-6 on unchanged/finite checks; rate/duration thresholds = class train p05/p99 (fixed before attack).
- **Train-only:** constructor raises unless `calibration.fit_split=='train'`; `selection_prohibited_inputs` includes validation features.
- **Status semantics:** per sample over required checks — PASS = all pass; **NOT_FULLY_TESTABLE** = no fail but ≥1 not_testable; **FAIL** = ≥1 required fail (FAIL overrides NOT_FULLY_TESTABLE). `sp_success = evasion ∧ valid ∧ feasible ∧ status==PASS`. **Recon/BruteForce can never PASS** (they always carry NOT_TESTABLE critical properties).

### 10.2 Results (pooled, identical across all 9 conditions) — `budget_sensitivity.csv`
- semantic_pass_rate = **0.48179133858267714** · semantic_fail_rate = **0.021653543307086614** · not_fully_testable_rate = **0.4965551181102362** · testability_rate = **0.5034448818897638** · **SP-ASR = 0.0 everywhere.**

**Per-class (maximum-evaluated, joint)** — `docs/primattack_budget_results.md` / `attack_results.json`:
| Class | Sem PASS | Sem FAIL | Not-testable | Testability | Fail reason |
|---|---:|---:|---:|---:|---|
| DoS | 0.9668 | 0.0332 | 0.0 | 1.0 | RATE_BELOW_TRAIN_P05 (timing dilation drops Flow Packets/s) |
| DDoS | 0.9472 | 0.0528 | 0.0 | 1.0 | RATE_BELOW_TRAIN_P05 |
| Recon | 0.0 | 0.0 | 1.0 | 0.0 | (structurally NOT_TESTABLE — scan structure unavailable) |
| BruteForce | 0.0 | 0.0 | 1.0 | 0.0 | (structurally NOT_TESTABLE — auth/credential semantics unavailable) |

- **NOT_TESTABLE coverage:** only DoS+DDoS are testable → overall testability ≈50.34 %. The 10 BruteForce raw successes are domain-valid and primitive-feasible but conservatively NOT_FULLY_TESTABLE → SP-ASR = 0.

### 10.3 Scope statement (MUST appear in thesis)
> **This is FLOW-LEVEL semantic-preservation testing over aggregate CICFlowMeter statistics, NOT packet-level real-world functionality verification.** Packet realization, feature re-extraction, and isolated replay are **outside scope**. (`docs/primattack_attack_preservation.md`, `docs/primattack_budget_results.md` Limitations, `primattack_sp_budget_explained.md §2–3`, `flow_semantics.py` docstring.) Held-constant Level-C fields must NOT be described as proof that a packet extractor would reproduce identical values.

---

## 11. Statistical Tests

Source: `outputs/primattack_budget_sensitivity_full/paired_statistics.json`; generator `scripts/analyze_primattack_experiments.py`; tests `src/evaluation/paired_validity_gap.py`. Design: **paired** on `[sample_id, attack_class, victim_model, seed]`, `dropna` keeps fully-paired rows, **pooled n = 4064** for every family. Correction = **Holm** (α=0.05). McNemar = exact two-sided binomial when b+c<25 (always here). **Effect sizes and confidence intervals are NOT emitted (MISSING by design)** — Newcombe CI/odds-ratio code exists in `paired_validity_gap.py` but is applied only to the raw-vs-valid ASR report, not here.

**Pairing verified:** `source_id_consistency.json` `identical_across_all_configurations=true, cells_checked=8`; constant n=4064 corroborates identical paired sets. **No broken pairing detected.**

### 11.1 Binary families (Cochran's Q omnibus + McNemar pairwise) — significant ones (**thesis Table 12**)
Null (omnibus): equal success proportion across matched conditions. Null (McNemar): marginal homogeneity (b=c).
| Fixed | Outcome | Condition | Cochran Q | p | Significant pairwise (exact McNemar, Holm) |
|---|---|---|---:|---:|---|
| budget=maximum-evaluated | targeted_success | primitive_mode | 20.0 | 4.539993e-05 | joint vs timing b=10 p=0.001953 holm=0.005859; padding vs timing b=10 p=0.001953 holm=0.005859; joint vs padding b=c=0 p=1.0 |
| primitive_mode=joint | targeted_success | budget | 20.0 | 4.539993e-05 | max vs restricted b=10 holm=0.005859; intermediate vs max c=10 holm=0.005859; intermediate vs restricted p=1.0 |
| primitive_mode=padding-only | targeted_success | budget | 20.0 | 4.539993e-05 | max vs restricted holm=0.005859; intermediate vs max holm=0.005859; intermediate vs restricted p=1.0 |

**Degenerate families (Q=0, p=1.0, all pairwise p=1.0):** budget=intermediate/targeted_success; budget=restricted/targeted_success; timing-only/targeted_success; **all sp_success families** (sp_asr=0 everywhere ⇒ no variance). Interpretation: the only real effect is that **padding (and hence joint) flips 10 BruteForce rows at the maximum budget that timing cannot** — statistically significant (Holm p≈0.0059) but negligible in magnitude, and it **vanishes under the semantic gate** (all sp_success comparisons null).

### 11.2 Continuous families (Friedman + Wilcoxon signed-rank, Pratt zeros)
18 families; representative statistics (`paired_statistics.json`): budgets/modes change **cost** strongly (Friedman χ² ≈ 4270–5306, p→0 for duration/byte/rate) but never success. Notable NS pairwise: joint~padding on relative_byte_change (intermediate p=0.276; restricted p=0.083; max p=8.5e-5 sig). padding-only duration & rate-retention Friedman = 0.0/p=1.0 (padding never alters duration/rate); timing-only byte Friedman = 0.0/p=1.0 (timing never alters bytes). Several Wilcoxon p values underflow to 0.0 (render as `<1e-300`).

**Summary claim supported:** cost metrics differ significantly across budgets/modes; **success and SP do not** (all SP comparisons null).

---

## 12. Seed-Level Reproducibility

**Every experiment uses a single global seed = 42.** No separate model/attack/sample-selection seeds; one `seed`/`seeds:[42]` governs each run. **Zero repeated runs → no mean±std is derivable anywhere.**
| Experiment | seed | repeats | source |
|---|---|---:|---|
| mlp+cnn classifiers | 42 | 1 | `outputs/cicids2017distrinet/classifier_run_manifest.json` |
| FT-Transformer classifier | 42 | 1 | `outputs/cicids2017distrinet_ft/classifier_run_manifest.json` |
| PrimAttack budget-FULL (9 conditions) | 42 | 1 | `outputs/primattack_budget_sensitivity_full/*/*/run_manifest.json` |
| PrimAttack random-control | 42 | 1 | `outputs/primattack_random_control/run_manifest.json` |
| PrimAttack smoke | 42 | 1 | `outputs/primattack_smoke/run_manifest.json` |
| PrimAttack budget (older, superseded) | 42 | 1 | `outputs/primattack_budget_sensitivity/` |

All NPZ artifacts are `*_seed42.npz`. **Any "mean±std over seeds" claim would be UNSUPPORTED.** Statistical power comes from the **4064 paired samples**, not seed replication. Attack config verified in every cell: steps 40, lr 0.1, cost_weight 0.01, init_noise 0.5, test_limit 512, optimizer 'optimized', calibration fit_split 'train'.
**Recompute check:** BruteForce raw ASR mlp 6/500=0.012, cnn 4/498=0.008032128514056224, pooled 10/4064=0.0024606299212598425 — all match stored values ✓.

---

## 13. Consistency Audit

### CRITICAL
- **C1. Feature-space PGD/C&W baselines not run on CICIDS2017 (§4).** The thesis cannot present a *current-roster* baseline-vs-PrimAttack comparison from existing artifacts. The only baseline numbers are from the **retired** v2 report (mlp/cnn/lstm/serial). → Either regenerate on mlp/cnn, or frame the study as PrimAttack-only + archived context. (Not silently fixed.)

### IMPORTANT
- **I1. Two coexisting budget-sensitivity dirs with divergent numbers.** `primattack_budget_sensitivity_full` (n=4064, padding/joint@max raw=0.0024606) is the **final** run; `primattack_budget_sensitivity` (n=254, all raw=0.0, different medians) is a **superseded pilot**. Cite `_full` only.
- **I2. Git commit mismatch.** budget-FULL manifests `git.commit=3380e4a0…`; random_control & smoke `git.commit=378aaf25…` (both dirty, different `source_tree_sha256`). Controls produced at a different tree state → comparability caveat.
- **I3. Train/attack environment split.** Victims trained on cuda / torch 2.5.1+cu121 / py3.11.15; PrimAttack executed on cpu / torch 2.5.1+cpu / py3.12.3. `checkpoint_reload_verified=true`, but cross-device/py numeric drift is a caveat for logit-threshold-sensitive ASR.
- **I4. Validator "domain-valid" is very permissive vs PrimAttack.** raw==valid in all cells means the validator (correctly) never rejects primitive-domain outputs; the *binding* gate is SP (semantic). State that raw/valid coincidence is expected for a realizability-constrained attack, not evidence of validator laxity.

### MINOR
- **M1.** Effect sizes & CIs absent for all 30 paired families (MISSING by design).
- **M2.** Underflow p-values stored as 0.0 (true underflow; render `<1e-300`).
- **M3.** FT checkpoints duplicated into `outputs/cicids2017distrinet/models/` although that manifest lists only mlp/cnn (harmless disk/manifest mismatch; FT has its own dir).
- **M4.** Retired archive `old_root_files/retired_lstm_cnn_lstm_2026-09-24/` mirrors an outputs tree with serial+lstm NPZ; suffix-based path resolution could accidentally reach it. Live artifacts are clean (mlp+cnn only, 72 NPZ, verified via `find`).
- **M5.** val (312,058) and test (312,056) are near-mirror sizes/distributions — symmetric holdout, not leakage.
- **M6.** `run_cicids2017_input_baseline.py` has stale mask keys (would KeyError) — dead/never-run code.

### Passed checks (evidence)
- **Feature count = 79** everywhere (classifier & FT manifests; `test_experiment_identity.py` asserts manifest/scaler/checkpoint all 79).
- Identical **preprocessing sha256** `df93d07…`, **scaler sha256** `c04ed70…`, **label_encoders sha256** `7aeb1c…` across all classifier/FT/PrimAttack manifests.
- Victim checkpoint SHAs consistent (mlp_category `a896ffcc…`, cnn_category `51849974…`).
- Class map consistent (Benign 0…BruteForce 4; 4 attack classes in all cells).
- ASR denominators sound; no test-set leakage (attacks on test, scaler/calibration on train, checkpoint reload verified).
- **No NaN/Inf** (`audit_results._require_finite`; 0.0 p-values are genuine underflow, not NaN). No duplicated paired rows (pivot+dropna unique keys).
- lstm/serial cleanly retired; live roster mlp+cnn only.

---

## 14. Final Thesis Tables (index)
1. **Dataset/class distribution** → §1.6 (5-class + binary).
2. **Victim architecture/hyperparameters** → §2.2 (+ §2.1 architectures).
3. **Clean model performance** → §2.3 (val vs test) + §2.4 (per-class test) + §2.5 (confusion).
4. **Baseline attack parameters** → §4 parameter table (NOTE: not run on CICIDS2017).
5. **Baseline attack results** → §4 archived-only table (retired roster) — **MISSING for current roster**.
6. **PrimAttack parameters** → §5.1–5.3.
7. **Main PrimAttack results** → §5.4 (pooled) + §5.5 (per-class×victim).
8. **Raw vs Valid vs SP-ASR** → below.
9. **Primitive ablation** → §7.
10. **Budget sensitivity** → §6.
11. **Optimizer ablation** → optimized vs random-feasible only (random=0); no other optimizer ablation exists.
12. **Statistical tests** → §11.1.
13. **Semantic-preservation results** → §10.2.
14. **Per-class final comparison** → §5.5 / §10.2.

**Table 8 — Raw vs Valid vs SP-ASR (PrimAttack, maximum-evaluated):**
| Mode | Raw ASR | Valid ASR | Prim-feasible ASR | SP-ASR |
|---|---:|---:|---:|---:|
| timing-only | 0.0 | 0.0 | 0.0 | 0.0 |
| padding-only | 0.0024606 (10/4064) | 0.0024606 | 0.0024606 | **0.0** |
| joint | 0.0024606 (10/4064) | 0.0024606 | 0.0024606 | **0.0** |

---

## 15. Recommended Figures
| # | Figure | Source artifact |
|---|---|---|
| 1 | Clean per-class F1 (mlp/cnn/ft, test) bar chart | `outputs/cicids2017distrinet*/per_class_metrics.csv` |
| 2 | Test confusion matrices (category, 3 models) | `outputs/cicids2017distrinet*/plots/*_category_test_confusion.png` (already generated) |
| 3 | Raw vs Valid vs SP-ASR by budget | `outputs/primattack_budget_sensitivity_full/01_raw_asr_vs_budget.png`, `02_valid_asr_vs_budget.png`, `03_sp_asr_vs_budget.png` (already generated) |
| 4 | Semantic pass vs budget | `…/04_semantic_pass_vs_budget.png` (generated) |
| 5 | Rate retention vs timing budget | `…/05_rate_retention_vs_timing_budget.png` (generated) |
| 6 | Primitive cost vs ASR | `…/06_primitive_cost_vs_asr.png` (generated) |
| 7 | Timing+padding combined ablation | `…/07_timing_padding_combined.png` (generated) |
| 8 | Semantic failure/testability breakdown (per class) | derive from `budget_sensitivity.csv` + `attack_results.json` (per-class PASS/FAIL/NOT_TESTABLE) — small new bar chart |
| 9 | Val-vs-test macro-F1 gap (distribution-shift illustration) | `classifier_metrics_summary.csv` (both dirs) — small new chart |

Figures 3–7 already exist in `outputs/primattack_budget_sensitivity_full/`. Only Figs 1, 8, 9 need trivial generation.

---

## 16. Thesis Write-Up Guide

### Experimental Setup
- **Cite:** 79 features; 5 classes; totals 2,080,379 (train 1,456,265 / val 312,058 / test 312,056); 15,754 exact duplicates removed; 70/15/15 chronological within-source-label split; RobustScaler + class weights fit on train only; seed 42; RTX 4070 Ti SUPER; torch 2.5.1+cu121, sklearn 1.9.0. **Table §1.6.**
- **Defensible observations:** (1) leakage-guarded (train↔test overlap 0). (2) chronological split induces genuine val/test distribution shift. (3) heavy class imbalance (BruteForce 6,947 vs Benign 1.65M) handled by balanced weighting. (4) BruteForce test = 1,042 rows — small-sample caveat.
- **NOT supported:** global forward-time or cross-campaign generalization (`not_claimed` in manifest); any claim the split is random-stratified.

### Victim Model Performance
- **Reference Table §2.3 (val vs test) + §2.4/§2.5.** Cite test macro-F1 ≈ 0.975–0.978, test acc ≈ 0.9843–0.9846 across all three models.
- **Observations:** (1) MLP, CNN, FT-Transformer are statistically indistinguishable on test despite 15× FT parameter gap — the task is near-saturated. (2) Recon is the weak class (precision ~0.84). (3) DoS Slowhttptest is a shared blind spot (~44–56 % read as Benign). (4) Validation is harder than test everywhere (report both, explain via temporal split).
- **NOT supported:** FT superiority (it isn't); any per-seed variance claim (single seed).

### Baseline Adversarial Results
- **Report honestly:** feature-space PGD/C&W were **not executed** on CICIDS2017 (§4, CRITICAL C1). Present the parameter set as *specified* and either (a) mark baseline results MISSING, or (b) use the archived v2 Input-PGD contrast (99.73 % raw → 0 % valid) explicitly labelled as retired-roster context.
- **NOT supported:** any current-roster baseline ASR number.

### Validity-Aware Evaluation
- **Main finding:** validity gating is decisive in general (archived Input-PGD: 99.73 % raw → 0.00 % valid because gradient perturbations violate integer/definitional rules). For the realizability-constrained PrimAttack, raw==valid (validator never rejects), so the **binding constraint moves to semantics**. **Cite validator = 235 rules (133/79/7/16), 100 % clean acceptance, 0.9998 synthetic detection.**
- **NOT supported:** that the validator is lax (it correctly accepts realizable flows and rejects gradient noise).

### PrimAttack Results
- **Supported comparisons:** across budgets (restricted/intermediate/maximum-evaluated), across primitive modes (timing/padding/joint), across classes and both victims, all on **identical paired source samples** (verified). **Cite:** peak raw ASR = 10/4064 = 0.246 %; SP-ASR = 0 everywhere; only BruteForce ever evades and only at the largest padding budget.
- **Observations:** (1) the two CICFlowMeter-realizable primitives are **insufficient** to evade well-trained MLP/CNN IDS. (2) padding is the only effective primitive; timing adds nothing. (3) even the 10 raw successes fail semantics. (4) this is a **robustness (negative) result** — the defensible thesis contribution is the *evaluation framework* (validity+feasibility+SP gating) and the demonstration that feature-space-looking successes collapse under realizability.
- **NOT supported:** "PrimAttack is a strong/effective attack"; any nonzero SP-ASR; extrapolation to FT-Transformer (not attacked).

### Semantic Preservation
- **Can claim:** flow-level SP proxy with train-calibrated, class-specific rules; 50.34 % of eligible samples are testable; DoS/DDoS fail via rate-retention under timing dilation; Recon/BruteForce are NOT_FULLY_TESTABLE by construction; SP-ASR = 0.
- **Cannot claim:** packet-level functionality, real-world replay, scan/auth semantics (out of scope — §10.3 verbatim). BruteForce raw successes are **not** validated attacks.

### Ablation Study
- **Which components matter:** padding primitive (essential), timing primitive (immaterial to success, only adds cost and rate-retention risk), budget magnitude (raises cost, not success; flips 10 BruteForce rows only at p75). **Reference §7 + §6.**

### Statistical Significance
- **Supported:** padding/joint vs timing at maximum budget (Cochran Q=20, p=4.5e-5; McNemar Holm p=0.0059) — the mode effect on raw success is significant but tiny. Cost metrics differ across budgets/modes (Friedman p→0). **All SP comparisons are null** (no SP difference). Pairing verified.
- **NOT supported:** any effect-size/CI claim (MISSING); practical significance of the 10-row effect; seed-variance significance.

### Limitations
- Single seed (no variance). Feature-space baselines not run (current roster). FT-Transformer trained but not attacked. Flow-level (not packet-level) SP. Attacks executed on CPU/py3.12 vs GPU/py3.11 training. Small BruteForce sample. Two-primitive realizability model is deliberately conservative.

---

## 17. Final Checklist

| Component | Status | Note |
|---|---|---|
| Dataset & preprocessing | **READY** | fully traced; hardware string MISSING (minor) |
| Class distribution tables | **READY** | §1.6 |
| Victim models (mlp/cnn/ft) arch+train | **READY** | single seed |
| Clean performance (val+test, per-class, confusion) | **READY** | §2.3–2.5 |
| Attack eligibility & denominators | **READY** | §3.2 (PrimAttack); VAE/baseline N MISSING |
| Feature-space PGD/C&W baselines | **NEEDS RERUN** | not run on current roster (§4, C1) |
| VAE latent attack (CICIDS2017) | **NEEDS RERUN** | only 2 smoke NPZ, no aggregate |
| PrimAttack config + budget + ablation | **READY** | `_full` dir |
| Budget experiments | **READY** | §6 |
| Primitive ablation | **READY** | §7 |
| Domain validator | **READY** | 235 rules, train-only |
| Semantic preservation | **READY** | flow-level; SP-ASR=0 |
| Statistical tests | **READY (effect sizes/CI MISSING)** | §11 |
| Seed reproducibility | **READY** | single seed 42 |
| Consistency audit | **READY** | §13 |
| Per-victim×class PrimAttack cells | **READY** | per-budget `attack_results.json` |
| Runtime / gradient-eval counts | **MISSING** | not recorded (any attack) |
| Regenerated v2 validity report (mlp/cnn) | **NEEDS MANUAL REVIEW** | only retired-roster copy exists |
| FT-Transformer as attack victim | **NEEDS MANUAL REVIEW** | trained but never attacked — decide scope |

## Remaining CICIDS2017 Work
Only genuinely-required tasks after this audit:
1. **(Optional but recommended for a baseline-vs-PrimAttack claim)** Run the feature-space input-PGD baseline on the current mlp/cnn victims (fix stale mask keys in `run_cicids2017_input_baseline.py`), e.g. `python -m src.attack.run_cicids2017_input_baseline --seeds 42 --steps 50 --epsilon 0.5 --alpha 0.05 --output-dir outputs/cicids2017_input_baseline`. Reason: §4/C1 — no current-roster feature-space baseline exists. Without it, present PrimAttack-only and cite archived context.
2. **(Optional)** Regenerate the v2 attack-artifact validity report for mlp/cnn only via `validation/evaluation/attack_artifact_validity.py` so §9.3 uses current-roster numbers instead of the retired (lstm/serial) archive.
3. **(Optional scope decision)** Either attack the FT-Transformer victim with PrimAttack or explicitly scope it out in the thesis (currently trained-but-unattacked).
4. **(Cosmetic)** Record attack runtime / gradient-evaluation counts if the thesis needs cost-in-seconds.

**If the thesis is framed as "PrimAttack robustness study on MLP/CNN with a validity+semantic evaluation framework," all required experimental evidence is already complete and READY** — items 1–4 are enhancements, not blockers. The core narrative (realizable two-primitive attacks fail against well-trained IDS; successes collapse under validity+semantic gating; padding>timing; budget raises cost not success; results statistically characterized on 4064 paired samples) is fully supported by existing artifacts.

---
*End of audit. Every headline number is traceable to a cited artifact under `data/processed/CICIDS_2017_Distrinet/`, `outputs/cicids2017distrinet*/`, or `outputs/primattack_budget_sensitivity_full/`.*
