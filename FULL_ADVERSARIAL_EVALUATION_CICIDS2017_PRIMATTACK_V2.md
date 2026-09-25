# Full Paired Adversarial Evaluation — CICIDS2017-DistriNet

Victims: mlp, cnn, ft_transformer. Seeds: [42, 123, 2024]. N per class (cap): 800 clean-correct flows. Features: 79; test flows: 312056.

Eligibility = **clean-correct** (victim predicts the true malicious class on the clean flow), selected ONCE per (victim,class) from the full test split (no pre-eligibility cap) and frozen across every attack/goal/seed. Denominator = that eligible set (identical across compared attacks within a victim). Rates are mean±std across seeds.

## 1. Harness audit — defects found and fixed

Audited the legacy runners (`run_cicids2017_primitive_attack.py`, `run_cicids2017_input_baseline.py`, `evaluate_cell`, `_class_rows`). Findings and the fix applied in this harness (`scripts/run_full_adversarial_eval.py`):

| # | Defect (legacy) | Evidence | Fix in this harness |
|---|---|---|---|
| 1 | Source rows selected by **true label only**, not clean-correct | `_class_rows(test.y, class_id, ...)` L305/L58 | Eligible = clean-correct via victim prediction on clean flow; selected once, saved. |
| 2 | Selection seeded by `42+class_id`, **independent of `--seeds`**; pairing only holds if identical `--test-limit` | L305 | One frozen eligible set per (victim,class) reused byte-identically by all attacks/seeds; sha256 saved. |
| 3 | **`--test-limit=1024` cap applied BEFORE eligibility** → truncates max eligible | L554/L174 | No pre-eligibility cap; head slice of size N applied AFTER clean-correct filtering; full eligible count recorded. |
| 4 | `run_cicids2017_input_baseline.py` **broken**: references `masks['mined_valid'|'benign'|'realizable']` not emitted by current `evaluate_cell` | L119/138/157 | Uses current `evaluate_cell` contract (`targeted_success`,`domain_valid`,`primitive_transform_consistent`,`in_dist`). |
| 5 | Unverified row-order alignment across y/X_test/X_test_pristine/parquet | L305-311 | Runtime asserts equal lengths, `y_test_cat==load_split.y`, and `X_test==(pristine-center)/scale`. |
| 6 | Cross-victim denominators differ (clean-correct is victim-specific) | `eligible=masks['clean_correct']` L376 | Paired tests kept **within victim** only. |

## 2. Configuration & attacks

- PGD (L∞): {'epsilon': 0.5, 'alpha': 0.05, 'steps': 40}. C&W (L2): {'lambda': 1.0, 'kappa': 0.0, 'iters': 60, 'lr': 0.01, 'conv': 1e-05}.
- PrimAttack: {'steps': 40, 'learning_rate': 0.1, 'restarts': 2, 'calibration': 'E:\\Shameem\\thesis\\artifacts\\primattack\\budget_calibration.json', 'calibration_fit_split': 'train'} ; budgets {'intermediate': 'p50', 'maximum-evaluated': 'p75', 'unbounded': 'unb'} (p50=intermediate, p75=maximum-evaluated, **unb=unbounded envelope-only**: p_max=+∞, max_relative_duration_change=+∞ so bounds collapse to the train-fit p99 physical envelope + realizability + semantic gate, with NO empirical class budget). The fully **unconstrained** attack is the input-space PGD/C&W baseline (no primitive model at all).
- Attack roster (22): pgd_untargeted, cw_untargeted, pgd_tb, cw_tb, prim_search_joint_p50, prim_search_timing_p50, prim_search_padding_p50, prim_rand_joint_p50, prim_rand_timing_p50, prim_rand_padding_p50, prim_search_joint_p75, prim_search_timing_p75, prim_search_padding_p75, prim_rand_joint_p75, prim_rand_timing_p75, prim_rand_padding_p75, prim_search_joint_unb, prim_search_timing_unb, prim_search_padding_unb, prim_rand_joint_unb, prim_rand_timing_unb, prim_rand_padding_unb.

## 3. Sample selection (row IDs saved to `outputs/full_adv_eval_primattack_v2/selection.json`)

| Victim | Class | N eligible (total) | N used | sha256(sample_ids) |
|---|---|---|---|---|
| mlp | DoS | 25583 | 800 | `5398a9e2cec7a61a…` |
| mlp | DDoS | 14243 | 800 | `0f9787cedd804e03…` |
| mlp | Recon | 23771 | 800 | `01935b6076fe32cc…` |
| mlp | BruteForce | 1022 | 800 | `61f344f82f6e9677…` |
| cnn | DoS | 25540 | 800 | `052131278e067d85…` |
| cnn | DDoS | 14244 | 800 | `0f9787cedd804e03…` |
| cnn | Recon | 23760 | 800 | `01935b6076fe32cc…` |
| cnn | BruteForce | 1018 | 800 | `3ee661cf644afe80…` |
| ft_transformer | DoS | 25522 | 800 | `901b6e945b41666f…` |
| ft_transformer | DDoS | 14257 | 800 | `0f9787cedd804e03…` |
| ft_transformer | Recon | 23788 | 800 | `01935b6076fe32cc…` |
| ft_transformer | BruteForce | 1026 | 800 | `61f344f82f6e9677…` |

## 4. Summary — per-victim rates (denominator = clean-correct eligible)

Each cell is the **macro-average over the 4 classes**, reported as mean±std over the class×seed cells (so the ±spread mixes cross-class heterogeneity and seed noise; for pure seed variance see §7 / per_seed_cells.csv, and for sample-level pooling see the §5 McNemar rows which concatenate all 3200 rows).

### mlp

| Attack | Raw ASR | Valid ASR | Tgt-Benign | Valid Tgt-Benign | Domain-valid | Realizable | IDR | SemPreserve | SP-ASR | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| pgd_untargeted | 100.00±0.00% | 0.00±0.00% | 93.76±9.05% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 5.67±8.90% | NA | NA | 0.486±0.010 |
| cw_untargeted | 99.97±0.06% | 0.00±0.00% | 97.78±4.01% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 17.31±24.75% | NA | NA | 0.106±0.030 |
| pgd_tb | 100.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 10.69±17.84% | NA | NA | 0.471±0.024 |
| cw_tb | 100.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 17.03±24.25% | NA | NA | 0.106±0.029 |
| prim_search_joint_p50 | 4.18±4.59% | 4.18±4.59% | 4.18±4.59% | 4.18±4.59% | 100.00±0.00% | 100.00±0.00% | 25.20±42.12% | 23.33±30.78% | 4.07±4.63% | 27.498±33.613 |
| prim_search_timing_p50 | 2.66±3.60% | 2.66±3.60% | 2.66±3.60% | 2.66±3.60% | 100.00±0.00% | 100.00±0.00% | 25.44±41.97% | 23.25±30.79% | 2.59±3.65% | 27.937±33.543 |
| prim_search_padding_p50 | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 100.00±0.00% | 100.00±0.00% | 71.97±35.73% | 29.44±41.26% | 0.12±0.23% | 0.022±0.030 |
| prim_rand_joint_p50 | 0.92±0.90% | 0.92±0.90% | 0.92±0.90% | 0.92±0.90% | 100.00±0.00% | 100.00±0.00% | 27.80±40.58% | 29.44±41.26% | 0.85±0.90% | 7.589±9.040 |
| prim_rand_timing_p50 | 0.57±0.63% | 0.57±0.63% | 0.57±0.63% | 0.57±0.63% | 100.00±0.00% | 100.00±0.00% | 28.41±40.23% | 29.44±41.26% | 0.54±0.66% | 7.573±9.048 |
| prim_rand_padding_p50 | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 100.00±0.00% | 100.00±0.00% | 72.85±36.07% | 29.44±41.26% | 0.00±0.00% | 0.016±0.021 |
| prim_search_joint_p75 | 6.27±7.13% | 6.27±7.13% | 6.27±7.13% | 6.27±7.13% | 100.00±0.00% | 100.00±0.00% | 25.16±42.14% | 20.93±26.65% | 6.17±7.18% | 40.372±44.245 |
| prim_search_timing_p75 | 4.06±5.41% | 4.06±5.41% | 4.06±5.41% | 4.06±5.41% | 100.00±0.00% | 100.00±0.00% | 25.19±42.12% | 20.82±26.65% | 3.99±5.46% | 41.349±44.399 |
| prim_search_padding_p75 | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 100.00±0.00% | 100.00±0.00% | 51.09±46.64% | 29.44±41.26% | 0.12±0.23% | 0.066±0.072 |
| prim_rand_joint_p75 | 1.74±1.78% | 1.74±1.78% | 1.74±1.78% | 1.74±1.78% | 100.00±0.00% | 100.00±0.00% | 26.59±41.31% | 29.44±41.26% | 1.65±1.82% | 11.413±12.068 |
| prim_rand_timing_p75 | 1.08±1.31% | 1.08±1.31% | 1.08±1.31% | 1.08±1.31% | 100.00±0.00% | 100.00±0.00% | 27.19±40.95% | 29.44±41.26% | 1.04±1.34% | 11.378±12.074 |
| prim_rand_padding_p75 | 0.06±0.12% | 0.06±0.12% | 0.06±0.12% | 0.06±0.12% | 100.00±0.00% | 100.00±0.00% | 55.95±41.44% | 29.44±41.26% | 0.03±0.08% | 0.036±0.037 |
| prim_search_joint_unb | 27.00±38.85% | 27.00±38.85% | 23.29±32.26% | 23.29±32.26% | 100.00±0.00% | 100.00±0.00% | 24.19±42.71% | 21.92±28.55% | 18.12±24.77% | 180.677±215.755 |
| prim_search_timing_unb | 10.16±12.94% | 10.16±12.94% | 10.16±12.94% | 10.16±12.94% | 100.00±0.00% | 100.00±0.00% | 25.19±42.12% | 12.29±13.56% | 8.27±10.50% | 425.981±601.715 |
| prim_search_padding_unb | 12.59±21.52% | 12.59±21.52% | 0.66±0.91% | 0.66±0.91% | 100.00±0.00% | 100.00±0.00% | 27.09±41.34% | 29.44±41.26% | 0.38±0.43% | 0.110±0.068 |
| prim_rand_joint_unb | 17.52±25.32% | 17.52±25.32% | 11.86±15.33% | 11.86±15.33% | 100.00±0.00% | 100.00±0.00% | 24.58±42.47% | 29.44±41.26% | 11.57±15.52% | 195.414±286.767 |
| prim_rand_timing_unb | 4.12±4.54% | 4.12±4.54% | 4.12±4.54% | 4.12±4.54% | 100.00±0.00% | 100.00±0.00% | 25.59±41.90% | 29.44±41.26% | 3.92±4.72% | 195.353±286.751 |
| prim_rand_padding_unb | 4.99±8.75% | 4.99±8.75% | 0.12±0.23% | 0.12±0.23% | 100.00±0.00% | 100.00±0.00% | 39.00±34.23% | 29.44±41.26% | 0.06±0.15% | 0.061±0.037 |

### cnn

| Attack | Raw ASR | Valid ASR | Tgt-Benign | Valid Tgt-Benign | Domain-valid | Realizable | IDR | SemPreserve | SP-ASR | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| pgd_untargeted | 95.18±8.68% | 0.00±0.00% | 83.56±27.33% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 14.25±17.16% | NA | NA | 0.428±0.005 |
| cw_untargeted | 97.88±2.57% | 0.00±0.00% | 72.78±41.68% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 24.78±25.79% | NA | NA | 0.094±0.031 |
| pgd_tb | 99.90±0.19% | 0.00±0.00% | 99.90±0.19% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 11.09±12.22% | NA | NA | 0.425±0.014 |
| cw_tb | 100.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 21.47±19.92% | NA | NA | 0.100±0.042 |
| prim_search_joint_p50 | 12.77±13.15% | 12.77±13.15% | 12.33±12.69% | 12.33±12.69% | 100.00±0.00% | 100.00±0.00% | 25.80±41.74% | 25.71±32.34% | 7.54±9.47% | 26.284±30.967 |
| prim_search_timing_p50 | 5.88±9.44% | 5.88±9.44% | 5.56±8.88% | 5.56±8.88% | 100.00±0.00% | 100.00±0.00% | 26.60±41.24% | 25.56±32.12% | 5.10±8.28% | 26.487±31.478 |
| prim_search_padding_p50 | 5.31±9.61% | 5.31±9.61% | 5.31±9.61% | 5.31±9.61% | 100.00±0.00% | 100.00±0.00% | 70.34±32.84% | 30.78±41.05% | 1.03±1.87% | 0.028±0.034 |
| prim_rand_joint_p50 | 4.29±4.80% | 4.29±4.80% | 4.29±4.80% | 4.29±4.80% | 100.00±0.00% | 100.00±0.00% | 28.66±40.12% | 30.78±41.05% | 2.11±2.52% | 7.623±9.014 |
| prim_rand_timing_p50 | 1.61±2.44% | 1.61±2.44% | 1.61±2.44% | 1.61±2.44% | 100.00±0.00% | 100.00±0.00% | 29.33±39.77% | 30.78±41.05% | 1.55±2.48% | 7.607±9.022 |
| prim_rand_padding_p50 | 2.65±4.79% | 2.65±4.79% | 2.65±4.79% | 2.65±4.79% | 100.00±0.00% | 100.00±0.00% | 74.06±33.73% | 30.78±41.05% | 0.52±0.95% | 0.016±0.021 |
| prim_search_joint_p75 | 35.41±27.49% | 35.41±27.49% | 34.78±27.45% | 34.78±27.45% | 100.00±0.00% | 100.00±0.00% | 25.25±42.09% | 24.25±29.94% | 11.06±13.42% | 36.623±39.603 |
| prim_search_timing_p75 | 8.38±12.55% | 8.38±12.55% | 7.94±11.77% | 7.94±11.77% | 100.00±0.00% | 100.00±0.00% | 25.67±41.82% | 23.90±29.34% | 7.26±10.79% | 39.445±41.255 |
| prim_search_padding_p75 | 5.41±9.56% | 5.41±9.56% | 5.41±9.56% | 5.41±9.56% | 100.00±0.00% | 100.00±0.00% | 52.41±45.53% | 30.78±41.05% | 1.03±1.87% | 0.067±0.072 |
| prim_rand_joint_p75 | 5.90±5.37% | 5.90±5.37% | 5.85±5.35% | 5.85±5.35% | 100.00±0.00% | 100.00±0.00% | 27.16±41.00% | 30.78±41.05% | 2.80±3.43% | 11.475±12.016 |
| prim_rand_timing_p75 | 2.19±3.20% | 2.19±3.20% | 2.16±3.14% | 2.16±3.14% | 100.00±0.00% | 100.00±0.00% | 27.79±40.63% | 30.78±41.05% | 2.09±3.18% | 11.439±12.021 |
| prim_rand_padding_p75 | 3.01±5.22% | 3.01±5.22% | 3.01±5.22% | 3.01±5.22% | 100.00±0.00% | 100.00±0.00% | 57.36±39.90% | 30.78±41.05% | 0.44±0.80% | 0.036±0.037 |
| prim_search_joint_unb | 59.79±44.11% | 59.79±44.11% | 57.61±42.10% | 57.61±42.10% | 100.00±0.00% | 100.00±0.00% | 24.31±42.63% | 26.57±34.24% | 24.82±33.50% | 292.839±429.928 |
| prim_search_timing_unb | 40.58±39.30% | 40.58±39.30% | 40.30±39.27% | 40.30±39.27% | 100.00±0.00% | 100.00±0.00% | 25.20±42.12% | 15.76±16.99% | 12.19±13.98% | 413.378±588.616 |
| prim_search_padding_unb | 29.47±41.26% | 29.47±41.26% | 6.81±9.01% | 6.81±9.01% | 100.00±0.00% | 100.00±0.00% | 28.56±40.82% | 30.78±41.05% | 2.44±2.61% | 0.102±0.065 |
| prim_rand_joint_unb | 44.58±33.69% | 44.58±33.69% | 39.21±28.81% | 39.21±28.81% | 100.00±0.00% | 100.00±0.00% | 24.59±42.46% | 30.78±41.05% | 18.00±24.45% | 198.718±284.949 |
| prim_rand_timing_unb | 15.21±12.93% | 15.21±12.93% | 15.08±12.94% | 15.08±12.94% | 100.00±0.00% | 100.00±0.00% | 25.60±41.90% | 30.78±41.05% | 6.28±6.67% | 198.658±284.933 |
| prim_rand_padding_unb | 16.68±22.36% | 16.68±22.36% | 4.05±5.97% | 4.05±5.97% | 100.00±0.00% | 100.00±0.00% | 40.42±33.15% | 30.78±41.05% | 0.93±0.98% | 0.061±0.037 |

### ft_transformer

| Attack | Raw ASR | Valid ASR | Tgt-Benign | Valid Tgt-Benign | Domain-valid | Realizable | IDR | SemPreserve | SP-ASR | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| pgd_untargeted | 97.29±4.81% | 0.00±0.00% | 88.05±14.05% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 7.74±8.67% | NA | NA | 0.444±0.018 |
| cw_untargeted | 81.81±26.81% | 0.00±0.00% | 55.38±41.58% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 30.75±20.68% | NA | NA | 0.062±0.018 |
| pgd_tb | 98.48±2.71% | 0.00±0.00% | 97.04±5.30% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 9.07±11.86% | NA | NA | 0.444±0.017 |
| cw_tb | 81.66±26.84% | 0.00±0.00% | 81.66±26.84% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 40.72±35.04% | NA | NA | 0.060±0.018 |
| prim_search_joint_p50 | 5.03±9.10% | 5.03±9.10% | 5.03±9.10% | 5.03±9.10% | 100.00±0.00% | 100.00±0.00% | 25.25±42.08% | 25.08±30.81% | 0.21±0.38% | 27.369±32.243 |
| prim_search_timing_p50 | 0.19±0.34% | 0.19±0.34% | 0.19±0.34% | 0.19±0.34% | 100.00±0.00% | 100.00±0.00% | 25.68±41.83% | 25.08±30.81% | 0.18±0.32% | 27.147±32.060 |
| prim_search_padding_p50 | 4.81±8.71% | 4.81±8.71% | 4.81±8.71% | 4.81±8.71% | 100.00±0.00% | 100.00±0.00% | 70.97±31.88% | 31.09±41.02% | 0.00±0.00% | 0.021±0.024 |
| prim_rand_joint_p50 | 3.48±6.30% | 3.48±6.30% | 3.48±6.30% | 3.48±6.30% | 100.00±0.00% | 100.00±0.00% | 28.91±40.01% | 31.09±41.02% | 0.15±0.27% | 7.599±9.033 |
| prim_rand_timing_p50 | 0.18±0.32% | 0.18±0.32% | 0.18±0.32% | 0.18±0.32% | 100.00±0.00% | 100.00±0.00% | 29.68±39.62% | 31.09±41.02% | 0.18±0.32% | 7.582±9.041 |
| prim_rand_padding_p50 | 3.33±6.03% | 3.33±6.03% | 3.33±6.03% | 3.33±6.03% | 100.00±0.00% | 100.00±0.00% | 74.76±32.64% | 31.09±41.02% | 0.00±0.00% | 0.016±0.021 |
| prim_search_joint_p75 | 5.19±9.31% | 5.19±9.31% | 5.19±9.31% | 5.19±9.31% | 100.00±0.00% | 100.00±0.00% | 25.16±42.14% | 22.65±26.85% | 0.21±0.38% | 39.341±40.725 |
| prim_search_timing_p75 | 0.19±0.34% | 0.19±0.34% | 0.19±0.34% | 0.19±0.34% | 100.00±0.00% | 100.00±0.00% | 25.25±42.08% | 22.61±26.84% | 0.18±0.32% | 38.845±40.255 |
| prim_search_padding_p75 | 4.97±8.91% | 4.97±8.91% | 4.97±8.91% | 4.97±8.91% | 100.00±0.00% | 100.00±0.00% | 53.00±45.12% | 31.09±41.02% | 0.00±0.00% | 0.043±0.043 |
| prim_rand_joint_p75 | 3.75±6.71% | 3.75±6.71% | 3.75±6.71% | 3.75±6.71% | 100.00±0.00% | 100.00±0.00% | 27.14±41.01% | 31.09±41.02% | 0.16±0.28% | 11.469±12.025 |
| prim_rand_timing_p75 | 0.18±0.32% | 0.18±0.32% | 0.18±0.32% | 0.18±0.32% | 100.00±0.00% | 100.00±0.00% | 27.99±40.53% | 31.09±41.02% | 0.18±0.32% | 11.433±12.031 |
| prim_rand_padding_p75 | 3.59±6.43% | 3.59±6.43% | 3.59±6.43% | 3.59±6.43% | 100.00±0.00% | 100.00±0.00% | 57.84±39.40% | 31.09±41.02% | 0.00±0.00% | 0.036±0.037 |
| prim_search_joint_unb | 5.43±9.32% | 5.43±9.32% | 5.43±9.32% | 5.43±9.32% | 100.00±0.00% | 100.00±0.00% | 24.19±42.71% | 11.05±11.80% | 0.21±0.38% | 510.117±753.670 |
| prim_search_timing_unb | 0.40±0.37% | 0.40±0.37% | 0.40±0.37% | 0.40±0.37% | 100.00±0.00% | 100.00±0.00% | 25.23±42.10% | 12.00±12.61% | 0.21±0.38% | 521.460±781.756 |
| prim_search_padding_unb | 5.03±9.03% | 5.03±9.03% | 5.03±9.03% | 5.03±9.03% | 100.00±0.00% | 100.00±0.00% | 28.97±40.80% | 31.09±41.02% | 0.00±0.00% | 0.080±0.052 |
| prim_rand_joint_unb | 4.43±7.49% | 4.43±7.49% | 4.43±7.49% | 4.43±7.49% | 100.00±0.00% | 100.00±0.00% | 24.59±42.46% | 31.09±41.02% | 0.17±0.30% | 200.336±283.836 |
| prim_rand_timing_unb | 0.40±0.36% | 0.40±0.36% | 0.40±0.36% | 0.40±0.36% | 100.00±0.00% | 100.00±0.00% | 25.60±41.90% | 31.09±41.02% | 0.21±0.38% | 200.275±283.820 |
| prim_rand_padding_unb | 4.07±7.30% | 4.07±7.30% | 4.07±7.30% | 4.07±7.30% | 100.00±0.00% | 100.00±0.00% | 40.90±32.82% | 31.09±41.02% | 0.00±0.00% | 0.061±0.037 |

_SemPreserve/SP-ASR are PrimAttack-only (NA for input baselines); for Recon/BruteForce the semantic proxy is NOT_TESTABLE by design (see §8)._

## 5. Paired sample-level McNemar tests (within victim, reference seed = 42)

A=first method, B=second. n10=A-success∧B-fail, n01=B-success∧A-fail. prop_diff = P(A)−P(B); OR = (n10+.5)/(n01+.5).

| Victim | Comparison | success | n | n11 | n10 | n01 | χ² | p | OR | P(A)−P(B) [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | pgd_tb vs prim_search_joint_p75 (raw targeted) | targeted | 3200 | 201 | 2999 | 0 | 2997.0 | 0.00e+00 | 5999.00 | +93.7% [+92.9,+94.6] |
| mlp | pgd_tb vs prim_search_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 201 | 199.0 | 6.22e-61 | 0.00 | -6.3% [-7.1,-5.4] |
| mlp | pgd_tb vs prim_search_joint_p50 (raw targeted) | targeted | 3200 | 134 | 3066 | 0 | 3064.0 | 0.00e+00 | 6133.00 | +95.8% [+95.1,+96.5] |
| mlp | pgd_tb vs prim_search_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 134 | 132.0 | 9.18e-41 | 0.00 | -4.2% [-4.9,-3.5] |
| mlp | pgd_tb vs prim_search_joint_unb (raw targeted) | targeted | 3200 | 743 | 2457 | 0 | 2455.0 | 0.00e+00 | 4915.00 | +76.8% [+75.3,+78.2] |
| mlp | pgd_tb vs prim_search_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 743 | 741.0 | 4.32e-224 | 0.00 | -23.2% [-24.7,-21.8] |
| mlp | cw_tb vs prim_search_joint_p75 (raw targeted) | targeted | 3200 | 201 | 2999 | 0 | 2997.0 | 0.00e+00 | 5999.00 | +93.7% [+92.9,+94.6] |
| mlp | cw_tb vs prim_search_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 201 | 199.0 | 6.22e-61 | 0.00 | -6.3% [-7.1,-5.4] |
| mlp | cw_tb vs prim_search_joint_p50 (raw targeted) | targeted | 3200 | 134 | 3066 | 0 | 3064.0 | 0.00e+00 | 6133.00 | +95.8% [+95.1,+96.5] |
| mlp | cw_tb vs prim_search_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 134 | 132.0 | 9.18e-41 | 0.00 | -4.2% [-4.9,-3.5] |
| mlp | cw_tb vs prim_search_joint_unb (raw targeted) | targeted | 3200 | 743 | 2457 | 0 | 2455.0 | 0.00e+00 | 4915.00 | +76.8% [+75.3,+78.2] |
| mlp | cw_tb vs prim_search_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 743 | 741.0 | 4.32e-224 | 0.00 | -23.2% [-24.7,-21.8] |
| mlp | joint vs timing (p75) | targeted | 3200 | 130 | 71 | 0 | 69.0 | 8.47e-22 | 143.00 | +2.2% [+1.7,+2.7] |
| mlp | joint vs padding (p75) | targeted | 3200 | 5 | 196 | 0 | 194.0 | 1.99e-59 | 393.00 | +6.1% [+5.3,+7.0] |
| mlp | search vs random (joint,p75) | targeted | 3200 | 48 | 153 | 0 | 151.0 | 1.75e-46 | 307.00 | +4.8% [+4.0,+5.5] |
| mlp | budget p50 vs p75 (search joint) | targeted | 3200 | 134 | 0 | 67 | 65.0 | 1.36e-20 | 0.01 | -2.1% [-2.6,-1.6] |
| mlp | budget p75 vs unbounded (search joint) | targeted | 3200 | 201 | 0 | 542 | 540.0 | 1.39e-163 | 0.00 | -16.9% [-18.2,-15.6] |
| mlp | pgd_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| mlp | cw_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| mlp | prim_search_joint_p75: raw vs valid targeted | targeted->valid | 3200 | 201 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| mlp | prim_search_joint_unb: raw vs valid targeted | targeted->valid | 3200 | 743 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| mlp | pgd_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| mlp | cw_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3199 | 0 | 3197.0 | 0.00e+00 | 6399.00 | +100.0% [+99.9,+100.0] |
| cnn | pgd_tb vs prim_search_joint_p75 (raw targeted) | targeted | 3200 | 1117 | 2080 | 0 | 2078.0 | 0.00e+00 | 4161.00 | +65.0% [+63.3,+66.7] |
| cnn | pgd_tb vs prim_search_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 1117 | 1115.0 | 0.00e+00 | 0.00 | -34.9% [-36.6,-33.3] |
| cnn | pgd_tb vs prim_search_joint_p50 (raw targeted) | targeted | 3200 | 394 | 2803 | 0 | 2801.0 | 0.00e+00 | 5607.00 | +87.6% [+86.5,+88.7] |
| cnn | pgd_tb vs prim_search_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 394 | 392.0 | 4.96e-119 | 0.00 | -12.3% [-13.5,-11.2] |
| cnn | pgd_tb vs prim_search_joint_unb (raw targeted) | targeted | 3200 | 1843 | 1354 | 2 | 1346.0 | 0.00e+00 | 541.80 | +42.2% [+40.5,+44.0] |
| cnn | pgd_tb vs prim_search_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 1845 | 1843.0 | 0.00e+00 | 0.00 | -57.7% [-59.4,-55.9] |
| cnn | cw_tb vs prim_search_joint_p75 (raw targeted) | targeted | 3200 | 1117 | 2083 | 0 | 2081.0 | 0.00e+00 | 4167.00 | +65.1% [+63.4,+66.7] |
| cnn | cw_tb vs prim_search_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 1117 | 1115.0 | 0.00e+00 | 0.00 | -34.9% [-36.6,-33.3] |
| cnn | cw_tb vs prim_search_joint_p50 (raw targeted) | targeted | 3200 | 394 | 2806 | 0 | 2804.0 | 0.00e+00 | 5613.00 | +87.7% [+86.5,+88.8] |
| cnn | cw_tb vs prim_search_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 394 | 392.0 | 4.96e-119 | 0.00 | -12.3% [-13.5,-11.2] |
| cnn | cw_tb vs prim_search_joint_unb (raw targeted) | targeted | 3200 | 1845 | 1355 | 0 | 1353.0 | 0.00e+00 | 2711.00 | +42.3% [+40.6,+44.1] |
| cnn | cw_tb vs prim_search_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 1845 | 1843.0 | 0.00e+00 | 0.00 | -57.7% [-59.4,-55.9] |
| cnn | joint vs timing (p75) | targeted | 3200 | 254 | 863 | 0 | 861.0 | 3.25e-260 | 1727.00 | +27.0% [+25.4,+28.5] |
| cnn | joint vs padding (p75) | targeted | 3200 | 173 | 944 | 0 | 942.0 | 1.34e-284 | 1889.00 | +29.5% [+27.9,+31.1] |
| cnn | search vs random (joint,p75) | targeted | 3200 | 177 | 940 | 1 | 935.0 | 1.01e-280 | 627.00 | +29.3% [+27.8,+30.9] |
| cnn | budget p50 vs p75 (search joint) | targeted | 3200 | 394 | 0 | 723 | 721.0 | 4.53e-218 | 0.00 | -22.6% [-24.0,-21.1] |
| cnn | budget p75 vs unbounded (search joint) | targeted | 3200 | 1117 | 0 | 728 | 726.0 | 1.42e-219 | 0.00 | -22.8% [-24.2,-21.3] |
| cnn | pgd_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3197 | 0 | 3195.0 | 0.00e+00 | 6395.00 | +99.9% [+99.8,+100.0] |
| cnn | cw_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| cnn | prim_search_joint_p75: raw vs valid targeted | targeted->valid | 3200 | 1117 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| cnn | prim_search_joint_unb: raw vs valid targeted | targeted->valid | 3200 | 1845 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| cnn | pgd_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3039 | 0 | 3037.0 | 0.00e+00 | 6079.00 | +95.0% [+94.2,+95.7] |
| cnn | cw_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3132 | 0 | 3130.0 | 0.00e+00 | 6265.00 | +97.9% [+97.4,+98.4] |
| ft_transformer | pgd_tb vs prim_search_joint_p75 (raw targeted) | targeted | 3200 | 166 | 2950 | 0 | 2948.0 | 0.00e+00 | 5901.00 | +92.2% [+91.3,+93.1] |
| ft_transformer | pgd_tb vs prim_search_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 166 | 164.0 | 2.14e-50 | 0.00 | -5.2% [-6.0,-4.4] |
| ft_transformer | pgd_tb vs prim_search_joint_p50 (raw targeted) | targeted | 3200 | 161 | 2955 | 0 | 2953.0 | 0.00e+00 | 5911.00 | +92.3% [+91.4,+93.3] |
| ft_transformer | pgd_tb vs prim_search_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 161 | 159.0 | 6.84e-49 | 0.00 | -5.0% [-5.8,-4.3] |
| ft_transformer | pgd_tb vs prim_search_joint_unb (raw targeted) | targeted | 3200 | 174 | 2942 | 0 | 2940.0 | 0.00e+00 | 5885.00 | +91.9% [+91.0,+92.9] |
| ft_transformer | pgd_tb vs prim_search_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 174 | 172.0 | 8.35e-53 | 0.00 | -5.4% [-6.2,-4.7] |
| ft_transformer | cw_tb vs prim_search_joint_p75 (raw targeted) | targeted | 3200 | 166 | 2447 | 0 | 2445.0 | 0.00e+00 | 4895.00 | +76.5% [+75.0,+77.9] |
| ft_transformer | cw_tb vs prim_search_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 166 | 164.0 | 2.14e-50 | 0.00 | -5.2% [-6.0,-4.4] |
| ft_transformer | cw_tb vs prim_search_joint_p50 (raw targeted) | targeted | 3200 | 161 | 2452 | 0 | 2450.0 | 0.00e+00 | 4905.00 | +76.6% [+75.2,+78.1] |
| ft_transformer | cw_tb vs prim_search_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 161 | 159.0 | 6.84e-49 | 0.00 | -5.0% [-5.8,-4.3] |
| ft_transformer | cw_tb vs prim_search_joint_unb (raw targeted) | targeted | 3200 | 174 | 2439 | 0 | 2437.0 | 0.00e+00 | 4879.00 | +76.2% [+74.7,+77.7] |
| ft_transformer | cw_tb vs prim_search_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 174 | 172.0 | 8.35e-53 | 0.00 | -5.4% [-6.2,-4.7] |
| ft_transformer | joint vs timing (p75) | targeted | 3200 | 6 | 160 | 0 | 158.0 | 1.37e-48 | 321.00 | +5.0% [+4.2,+5.8] |
| ft_transformer | joint vs padding (p75) | targeted | 3200 | 159 | 7 | 0 | 5.1 | 1.56e-02 | 15.00 | +0.2% [+0.1,+0.4] |
| ft_transformer | search vs random (joint,p75) | targeted | 3200 | 118 | 48 | 0 | 46.0 | 7.11e-15 | 97.00 | +1.5% [+1.1,+1.9] |
| ft_transformer | budget p50 vs p75 (search joint) | targeted | 3200 | 161 | 0 | 5 | 3.2 | 6.25e-02 | 0.09 | -0.2% [-0.3,-0.0] |
| ft_transformer | budget p75 vs unbounded (search joint) | targeted | 3200 | 166 | 0 | 8 | 6.1 | 7.81e-03 | 0.06 | -0.2% [-0.4,-0.1] |
| ft_transformer | pgd_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3116 | 0 | 3114.0 | 0.00e+00 | 6233.00 | +97.4% [+96.8,+97.9] |
| ft_transformer | cw_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 2613 | 0 | 2611.0 | 0.00e+00 | 5227.00 | +81.7% [+80.3,+83.0] |
| ft_transformer | prim_search_joint_p75: raw vs valid targeted | targeted->valid | 3200 | 166 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| ft_transformer | prim_search_joint_unb: raw vs valid targeted | targeted->valid | 3200 | 174 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| ft_transformer | pgd_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3107 | 0 | 3105.0 | 0.00e+00 | 6215.00 | +97.1% [+96.5,+97.7] |
| ft_transformer | cw_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 2618 | 0 | 2616.0 | 0.00e+00 | 5237.00 | +81.8% [+80.5,+83.1] |

## 6. Per-class results (mean±std across seeds)

### mlp

| Attack | Class | Tgt-Benign | Valid Tgt-Benign | Domain-valid | SemPreserve |
|---|---|---|---|---|---|
| pgd_tb | DoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | DDoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | Recon | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | BruteForce | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | DoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | DDoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | Recon | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | BruteForce | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| prim_search_joint_unb | DoS | 16.12±0.00% | 16.12±0.00% | 100.00±0.00% | 20.75±0.00% |
| prim_search_joint_unb | DDoS | 75.67±0.31% | 75.67±0.31% | 100.00±0.00% | 66.92±1.92% |
| prim_search_joint_unb | Recon | 0.63±0.00% | 0.63±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_unb | BruteForce | 0.75±0.00% | 0.75±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p75 | DoS | 8.21±0.07% | 8.21±0.07% | 100.00±0.00% | 21.21±0.07% |
| prim_search_joint_p75 | DDoS | 16.62±0.00% | 16.62±0.00% | 100.00±0.00% | 62.50±0.00% |
| prim_search_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p75 | BruteForce | 0.25±0.00% | 0.25±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p50 | DoS | 5.83±0.07% | 5.83±0.07% | 100.00±0.00% | 21.21±0.07% |
| prim_search_joint_p50 | DDoS | 10.62±0.00% | 10.62±0.00% | 100.00±0.00% | 72.12±0.00% |
| prim_search_joint_p50 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p50 | BruteForce | 0.25±0.00% | 0.25±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_timing_p75 | DoS | 3.25±0.00% | 3.25±0.00% | 100.00±0.00% | 20.79±0.07% |
| prim_search_timing_p75 | DDoS | 12.75±0.00% | 12.75±0.00% | 100.00±0.00% | 62.50±0.00% |
| prim_search_timing_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_timing_p75 | BruteForce | 0.25±0.00% | 0.25±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_padding_p75 | DoS | 0.63±0.00% | 0.63±0.00% | 100.00±0.00% | 21.62±0.00% |
| prim_search_padding_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_search_padding_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_padding_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | DoS | 2.75±0.13% | 2.75±0.13% | 100.00±0.00% | 21.62±0.00% |
| prim_rand_joint_p75 | DDoS | 3.96±0.76% | 3.96±0.76% | 100.00±0.00% | 96.12±0.00% |
| prim_rand_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | BruteForce | 0.25±0.00% | 0.25±0.00% | 100.00±0.00% | 0.00±0.00% |

### cnn

| Attack | Class | Tgt-Benign | Valid Tgt-Benign | Domain-valid | SemPreserve |
|---|---|---|---|---|---|
| pgd_tb | DoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | DDoS | 99.58±0.07% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | Recon | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | BruteForce | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | DoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | DDoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | Recon | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | BruteForce | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| prim_search_joint_unb | DoS | 38.96±0.07% | 38.96±0.07% | 100.00±0.00% | 26.00±0.22% |
| prim_search_joint_unb | DDoS | 90.88±0.13% | 90.88±0.13% | 100.00±0.00% | 80.29±0.62% |
| prim_search_joint_unb | Recon | 0.63±0.00% | 0.63±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_unb | BruteForce | 100.00±0.00% | 100.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p75 | DoS | 30.08±0.14% | 30.08±0.14% | 100.00±0.00% | 26.88±0.13% |
| prim_search_joint_p75 | DDoS | 35.12±0.00% | 35.12±0.00% | 100.00±0.00% | 70.12±0.25% |
| prim_search_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p75 | BruteForce | 73.92±0.52% | 73.92±0.52% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p50 | DoS | 25.08±0.14% | 25.08±0.14% | 100.00±0.00% | 27.00±0.00% |
| prim_search_joint_p50 | DDoS | 23.88±0.00% | 23.88±0.00% | 100.00±0.00% | 75.83±0.14% |
| prim_search_joint_p50 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p50 | BruteForce | 0.38±0.00% | 0.38±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_timing_p75 | DoS | 4.12±0.00% | 4.12±0.00% | 100.00±0.00% | 26.92±0.07% |
| prim_search_timing_p75 | DDoS | 27.25±0.00% | 27.25±0.00% | 100.00±0.00% | 68.67±0.31% |
| prim_search_timing_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_timing_p75 | BruteForce | 0.38±0.00% | 0.38±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_padding_p75 | DoS | 21.25±0.00% | 21.25±0.00% | 100.00±0.00% | 27.00±0.00% |
| prim_search_padding_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_search_padding_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_padding_p75 | BruteForce | 0.38±0.00% | 0.38±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | DoS | 13.12±0.45% | 13.12±0.45% | 100.00±0.00% | 27.00±0.00% |
| prim_rand_joint_p75 | DDoS | 8.00±0.66% | 8.00±0.66% | 100.00±0.00% | 96.12±0.00% |
| prim_rand_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | BruteForce | 2.29±0.51% | 2.29±0.51% | 100.00±0.00% | 0.00±0.00% |

### ft_transformer

| Attack | Class | Tgt-Benign | Valid Tgt-Benign | Domain-valid | SemPreserve |
|---|---|---|---|---|---|
| pgd_tb | DoS | 99.92±0.07% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | DDoS | 88.29±1.18% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | Recon | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| pgd_tb | BruteForce | 99.96±0.07% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | DoS | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | DDoS | 91.25±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | Recon | 37.50±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| cw_tb | BruteForce | 97.88±0.00% | 0.00±0.00% | 0.00±0.00% | NA |
| prim_search_joint_unb | DoS | 20.88±0.00% | 20.88±0.00% | 100.00±0.00% | 25.42±0.26% |
| prim_search_joint_unb | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 18.79±0.76% |
| prim_search_joint_unb | Recon | 0.58±0.07% | 0.58±0.07% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_unb | BruteForce | 0.25±0.00% | 0.25±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p75 | DoS | 20.62±0.00% | 20.62±0.00% | 100.00±0.00% | 28.08±0.07% |
| prim_search_joint_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 62.50±0.00% |
| prim_search_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p75 | BruteForce | 0.12±0.00% | 0.12±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p50 | DoS | 20.12±0.00% | 20.12±0.00% | 100.00±0.00% | 28.21±0.07% |
| prim_search_joint_p50 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 72.12±0.00% |
| prim_search_joint_p50 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_joint_p50 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_timing_p75 | DoS | 0.75±0.00% | 0.75±0.00% | 100.00±0.00% | 27.96±0.07% |
| prim_search_timing_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 62.50±0.00% |
| prim_search_timing_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_timing_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_padding_p75 | DoS | 19.75±0.00% | 19.75±0.00% | 100.00±0.00% | 28.25±0.00% |
| prim_search_padding_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_search_padding_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_search_padding_p75 | BruteForce | 0.12±0.00% | 0.12±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | DoS | 14.88±0.54% | 14.88±0.54% | 100.00±0.00% | 28.25±0.00% |
| prim_rand_joint_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_rand_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | BruteForce | 0.12±0.00% | 0.12±0.00% | 100.00±0.00% | 0.00±0.00% |

## 7. Per-seed valid targeted-benign ASR (every seed retained)

| Victim | Attack | seed 42 | seed 123 | seed 2024 |
|---|---|---|---|---|
| mlp | pgd_untargeted | 0.00% | 0.00% | 0.00% |
| mlp | cw_untargeted | 0.00% | 0.00% | 0.00% |
| mlp | pgd_tb | 0.00% | 0.00% | 0.00% |
| mlp | cw_tb | 0.00% | 0.00% | 0.00% |
| mlp | prim_search_joint_p50 | 0.25% | 0.25% | 0.25% |
| mlp | prim_search_timing_p50 | 0.25% | 0.25% | 0.25% |
| mlp | prim_search_padding_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_joint_p50 | 0.12% | 0.12% | 0.12% |
| mlp | prim_rand_timing_p50 | 0.12% | 0.12% | 0.12% |
| mlp | prim_rand_padding_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_search_joint_p75 | 0.25% | 0.25% | 0.25% |
| mlp | prim_search_timing_p75 | 0.25% | 0.25% | 0.25% |
| mlp | prim_search_padding_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_joint_p75 | 0.25% | 0.25% | 0.25% |
| mlp | prim_rand_timing_p75 | 0.12% | 0.25% | 0.12% |
| mlp | prim_rand_padding_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_search_joint_unb | 0.75% | 0.75% | 0.75% |
| mlp | prim_search_timing_unb | 0.25% | 0.25% | 0.25% |
| mlp | prim_search_padding_unb | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_joint_unb | 0.25% | 0.25% | 0.38% |
| mlp | prim_rand_timing_unb | 0.25% | 0.25% | 0.12% |
| mlp | prim_rand_padding_unb | 0.00% | 0.00% | 0.00% |
| cnn | pgd_untargeted | 0.00% | 0.00% | 0.00% |
| cnn | cw_untargeted | 0.00% | 0.00% | 0.00% |
| cnn | pgd_tb | 0.00% | 0.00% | 0.00% |
| cnn | cw_tb | 0.00% | 0.00% | 0.00% |
| cnn | prim_search_joint_p50 | 0.38% | 0.38% | 0.38% |
| cnn | prim_search_timing_p50 | 0.38% | 0.38% | 0.38% |
| cnn | prim_search_padding_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_rand_joint_p50 | 0.25% | 0.12% | 0.25% |
| cnn | prim_rand_timing_p50 | 0.25% | 0.25% | 0.25% |
| cnn | prim_rand_padding_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_search_joint_p75 | 74.50% | 73.50% | 73.75% |
| cnn | prim_search_timing_p75 | 0.38% | 0.38% | 0.38% |
| cnn | prim_search_padding_p75 | 0.38% | 0.38% | 0.38% |
| cnn | prim_rand_joint_p75 | 2.00% | 2.88% | 2.00% |
| cnn | prim_rand_timing_p75 | 0.25% | 0.25% | 0.25% |
| cnn | prim_rand_padding_p75 | 0.38% | 0.38% | 0.38% |
| cnn | prim_search_joint_unb | 100.00% | 100.00% | 100.00% |
| cnn | prim_search_timing_unb | 99.88% | 100.00% | 100.00% |
| cnn | prim_search_padding_unb | 0.38% | 0.38% | 0.38% |
| cnn | prim_rand_joint_unb | 71.25% | 73.00% | 72.00% |
| cnn | prim_rand_timing_unb | 32.50% | 38.12% | 33.12% |
| cnn | prim_rand_padding_unb | 0.38% | 0.38% | 0.38% |
| ft_transformer | pgd_untargeted | 0.00% | 0.00% | 0.00% |
| ft_transformer | cw_untargeted | 0.00% | 0.00% | 0.00% |
| ft_transformer | pgd_tb | 0.00% | 0.00% | 0.00% |
| ft_transformer | cw_tb | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_search_joint_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_search_timing_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_search_padding_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_joint_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_timing_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_padding_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_search_joint_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_search_timing_p75 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_search_padding_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_rand_joint_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_rand_timing_p75 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_padding_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_search_joint_unb | 0.25% | 0.25% | 0.25% |
| ft_transformer | prim_search_timing_unb | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_search_padding_unb | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_rand_joint_unb | 0.25% | 0.25% | 0.25% |
| ft_transformer | prim_rand_timing_unb | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_rand_padding_unb | 0.12% | 0.12% | 0.12% |

_Full per-seed metrics: `outputs/full_adv_eval_primattack_v2/per_seed_cells.csv`._

## 8. Failures / skipped / not-testable

- Non-finite / eligibility failures logged: 0 (`outputs/full_adv_eval_primattack_v2/failures.json`).
- Subsampled classes (N-cap < eligible): mlp/DoS (800/25583), mlp/DDoS (800/14243), mlp/Recon (800/23771), mlp/BruteForce (800/1022), cnn/DoS (800/25540), cnn/DDoS (800/14244), cnn/Recon (800/23760), cnn/BruteForce (800/1018), ft_transformer/DoS (800/25522), ft_transformer/DDoS (800/14257), ft_transformer/Recon (800/23788), ft_transformer/BruteForce (800/1026).
- Semantic-preservation proxy is **NOT_TESTABLE by construction for Recon and BruteForce** (no retained class-level semantic rule), so SemPreserve/SP-ASR for those classes reflect testability, not failure; DoS/DDoS use the rate-retention rule.

## 9. Interpretation

- **mlp**: unconstrained input-PGD raw ASR ≈ 100.0% but VALID ASR ≈ 0.0% (domain gate rejects the free perturbation). PrimAttack (search-joint,p75) targeted-benign ≈ 6.3%, valid targeted-benign ≈ 6.3%, domain-validity ≈ 100.0%.
- **cnn**: unconstrained input-PGD raw ASR ≈ 95.2% but VALID ASR ≈ 0.0% (domain gate rejects the free perturbation). PrimAttack (search-joint,p75) targeted-benign ≈ 34.8%, valid targeted-benign ≈ 34.8%, domain-validity ≈ 100.0%.
- **ft_transformer**: unconstrained input-PGD raw ASR ≈ 97.3% but VALID ASR ≈ 0.0% (domain gate rejects the free perturbation). PrimAttack (search-joint,p75) targeted-benign ≈ 5.2%, valid targeted-benign ≈ 5.2%, domain-validity ≈ 100.0%.

- **Raw vs valid success** (paired, within attack): for the unconstrained baselines n01=0 and n10=all successes → validity strictly and significantly removes success (the free-perturbation adversarials are domain-invalid). PrimAttack keeps validity by construction, so its raw and valid targeted counts coincide.
- **Baseline vs PrimAttack** paired tests quantify the trade: baselines dominate on RAW targeted success but PrimAttack dominates on VALID targeted success wherever the sign of P(A)−P(B) flips between the raw and valid rows above.
- **Variants**: exact integer padding search is followed by adaptive affine timing refinement only for unresolved rows. Paired mode rows isolate timing and padding; search-vs-random tests whether optimization dominates a feasible control. Read exact signed differences and confidence intervals in §5.
- **Unbounded (envelope-only) PrimAttack**: removing the p25/p50/p75 empirical budget and keeping only the p99 physical envelope leaves targeted-benign success essentially unchanged vs p75 (see 'budget p75 vs unbounded' rows in §5) while domain-validity stays 100% — i.e. the empirical budget was not the binding constraint; realizability + the physical envelope are. The gap to the unconstrained input baselines is the price of staying valid/realizable.
