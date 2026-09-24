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
- PrimAttack: {'steps': 40, 'lr': 0.1, 'cost_weight': 0.01, 'init_noise': 0.5, 'calibration': 'E:\\Shameem\\thesis\\artifacts\\primattack\\budget_calibration.json', 'calibration_fit_split': 'train'} ; budgets {'intermediate': 'p50', 'maximum-evaluated': 'p75', 'unbounded': 'unb'} (p50=intermediate, p75=maximum-evaluated, **unb=unbounded envelope-only**: p_max=+∞, max_relative_duration_change=+∞ so bounds collapse to the train-fit p99 physical envelope + realizability + semantic gate, with NO empirical class budget). The fully **unconstrained** attack is the input-space PGD/C&W baseline (no primitive model at all).
- Attack roster (22): pgd_untargeted, cw_untargeted, pgd_tb, cw_tb, prim_opt_joint_p50, prim_opt_timing_p50, prim_opt_padding_p50, prim_rand_joint_p50, prim_rand_timing_p50, prim_rand_padding_p50, prim_opt_joint_p75, prim_opt_timing_p75, prim_opt_padding_p75, prim_rand_joint_p75, prim_rand_timing_p75, prim_rand_padding_p75, prim_opt_joint_unb, prim_opt_timing_unb, prim_opt_padding_unb, prim_rand_joint_unb, prim_rand_timing_unb, prim_rand_padding_unb.

## 3. Sample selection (row IDs saved to `outputs/full_adv_eval/selection.json`)

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
| prim_opt_joint_p50 | 0.09±0.18% | 0.09±0.18% | 0.09±0.18% | 0.09±0.18% | 100.00±0.00% | 100.00±0.00% | 72.94±35.99% | 29.44±41.26% | 0.06±0.12% | 0.028±0.026 |
| prim_opt_timing_p50 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 73.41±36.10% | 29.44±41.26% | 0.00±0.00% | 0.011±0.018 |
| prim_opt_padding_p50 | 0.09±0.18% | 0.09±0.18% | 0.09±0.18% | 0.09±0.18% | 100.00±0.00% | 100.00±0.00% | 73.47±36.34% | 29.44±41.26% | 0.06±0.12% | 0.017±0.024 |
| prim_rand_joint_p50 | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 100.00±0.00% | 100.00±0.00% | 72.53±35.83% | 29.44±41.26% | 0.00±0.00% | 0.025±0.019 |
| prim_rand_timing_p50 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 73.42±36.14% | 29.44±41.26% | 0.00±0.00% | 0.009±0.010 |
| prim_rand_padding_p50 | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 100.00±0.00% | 100.00±0.00% | 72.85±36.07% | 29.44±41.26% | 0.00±0.00% | 0.016±0.021 |
| prim_opt_joint_p75 | 0.15±0.27% | 0.15±0.27% | 0.15±0.27% | 0.15±0.27% | 100.00±0.00% | 100.00±0.00% | 62.35±35.26% | 29.44±41.26% | 0.11±0.21% | 0.052±0.032 |
| prim_opt_timing_p75 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 72.92±35.92% | 29.44±41.26% | 0.00±0.00% | 0.015±0.023 |
| prim_opt_padding_p75 | 0.15±0.27% | 0.15±0.27% | 0.15±0.27% | 0.15±0.27% | 100.00±0.00% | 100.00±0.00% | 63.44±36.04% | 29.44±41.26% | 0.11±0.21% | 0.037±0.035 |
| prim_rand_joint_p75 | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 0.03±0.06% | 100.00±0.00% | 100.00±0.00% | 55.07±41.28% | 29.44±41.26% | 0.00±0.00% | 0.050±0.040 |
| prim_rand_timing_p75 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 70.66±35.06% | 29.44±41.26% | 0.00±0.00% | 0.014±0.014 |
| prim_rand_padding_p75 | 0.06±0.12% | 0.06±0.12% | 0.06±0.12% | 0.06±0.12% | 100.00±0.00% | 100.00±0.00% | 55.95±41.44% | 29.44±41.26% | 0.03±0.08% | 0.036±0.037 |
| prim_opt_joint_unb | 16.68±29.07% | 16.68±29.07% | 0.51±0.48% | 0.51±0.48% | 100.00±0.00% | 100.00±0.00% | 31.82±40.13% | 29.44±41.26% | 0.24±0.32% | 0.169±0.047 |
| prim_opt_timing_unb | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 100.00±0.00% | 100.00±0.00% | 65.08±39.00% | 29.44±41.26% | 0.00±0.00% | 0.088±0.065 |
| prim_opt_padding_unb | 15.11±26.62% | 15.11±26.62% | 0.40±0.52% | 0.40±0.52% | 100.00±0.00% | 100.00±0.00% | 39.44±38.45% | 29.44±41.26% | 0.28±0.32% | 0.073±0.048 |
| prim_rand_joint_unb | 4.69±7.82% | 4.69±7.82% | 0.28±0.30% | 0.28±0.30% | 100.00±0.00% | 100.00±0.00% | 30.48±39.89% | 29.44±41.26% | 0.06±0.15% | 2.717±4.194 |
| prim_rand_timing_unb | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 100.00±0.00% | 100.00±0.00% | 46.29±45.36% | 29.44±41.26% | 0.00±0.00% | 2.657±4.229 |
| prim_rand_padding_unb | 4.99±8.75% | 4.99±8.75% | 0.12±0.23% | 0.12±0.23% | 100.00±0.00% | 100.00±0.00% | 39.00±34.23% | 29.44±41.26% | 0.06±0.15% | 0.061±0.037 |

### cnn

| Attack | Raw ASR | Valid ASR | Tgt-Benign | Valid Tgt-Benign | Domain-valid | Realizable | IDR | SemPreserve | SP-ASR | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| pgd_untargeted | 95.18±8.68% | 0.00±0.00% | 83.56±27.33% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 14.25±17.16% | NA | NA | 0.428±0.005 |
| cw_untargeted | 97.88±2.57% | 0.00±0.00% | 72.78±41.68% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 24.78±25.79% | NA | NA | 0.094±0.031 |
| pgd_tb | 99.90±0.19% | 0.00±0.00% | 99.90±0.19% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 11.09±12.22% | NA | NA | 0.425±0.014 |
| cw_tb | 100.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 21.47±19.92% | NA | NA | 0.100±0.042 |
| prim_opt_joint_p50 | 1.04±1.88% | 1.04±1.88% | 1.04±1.88% | 1.04±1.88% | 100.00±0.00% | 100.00±0.00% | 70.52±32.76% | 30.78±41.05% | 1.03±1.87% | 0.024±0.019 |
| prim_opt_timing_p50 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 74.69±33.83% | 30.78±41.05% | 0.00±0.00% | 0.006±0.006 |
| prim_opt_padding_p50 | 1.03±1.87% | 1.03±1.87% | 1.03±1.87% | 1.03±1.87% | 100.00±0.00% | 100.00±0.00% | 71.15±32.91% | 30.78±41.05% | 1.02±1.85% | 0.018±0.019 |
| prim_rand_joint_p50 | 2.64±4.77% | 2.64±4.77% | 2.64±4.77% | 2.64±4.77% | 100.00±0.00% | 100.00±0.00% | 73.72±33.51% | 30.78±41.05% | 0.51±0.94% | 0.026±0.020 |
| prim_rand_timing_p50 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 74.62±33.79% | 30.78±41.05% | 0.00±0.00% | 0.009±0.010 |
| prim_rand_padding_p50 | 2.65±4.79% | 2.65±4.79% | 2.65±4.79% | 2.65±4.79% | 100.00±0.00% | 100.00±0.00% | 74.06±33.73% | 30.78±41.05% | 0.52±0.95% | 0.016±0.021 |
| prim_opt_joint_p75 | 1.14±1.91% | 1.14±1.91% | 1.14±1.91% | 1.14±1.91% | 100.00±0.00% | 100.00±0.00% | 52.29±45.41% | 30.78±41.05% | 1.03±1.87% | 0.047±0.046 |
| prim_opt_timing_p75 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 70.39±32.66% | 30.78±41.05% | 0.00±0.00% | 0.010±0.010 |
| prim_opt_padding_p75 | 1.14±1.91% | 1.14±1.91% | 1.14±1.91% | 1.14±1.91% | 100.00±0.00% | 100.00±0.00% | 52.41±45.53% | 30.78±41.05% | 1.03±1.87% | 0.035±0.035 |
| prim_rand_joint_p75 | 3.01±5.22% | 3.01±5.22% | 3.01±5.22% | 3.01±5.22% | 100.00±0.00% | 100.00±0.00% | 56.40±39.87% | 30.78±41.05% | 0.44±0.80% | 0.051±0.040 |
| prim_rand_timing_p75 | 0.01±0.04% | 0.01±0.04% | 0.01±0.04% | 0.01±0.04% | 100.00±0.00% | 100.00±0.00% | 71.88±32.72% | 30.78±41.05% | 0.01±0.04% | 0.015±0.013 |
| prim_rand_padding_p75 | 3.01±5.22% | 3.01±5.22% | 3.01±5.22% | 3.01±5.22% | 100.00±0.00% | 100.00±0.00% | 57.36±39.90% | 30.78±41.05% | 0.44±0.80% | 0.036±0.037 |
| prim_opt_joint_unb | 26.74±44.20% | 26.74±44.20% | 2.59±2.16% | 2.59±2.16% | 100.00±0.00% | 100.00±0.00% | 26.98±41.38% | 30.78±41.05% | 1.88±1.98% | 0.447±0.537 |
| prim_opt_timing_unb | 0.27±0.26% | 0.27±0.26% | 0.27±0.26% | 0.27±0.26% | 100.00±0.00% | 100.00±0.00% | 50.54±46.63% | 30.78±41.05% | 0.02±0.05% | 0.390±0.551 |
| prim_opt_padding_unb | 26.44±44.35% | 26.44±44.35% | 2.37±2.47% | 2.37±2.47% | 100.00±0.00% | 100.00±0.00% | 28.42±40.93% | 30.78±41.05% | 1.94±2.03% | 0.068±0.043 |
| prim_rand_joint_unb | 16.83±22.07% | 16.83±22.07% | 4.22±5.84% | 4.22±5.84% | 100.00±0.00% | 100.00±0.00% | 30.77±39.68% | 30.78±41.05% | 0.88±0.93% | 2.723±4.190 |
| prim_rand_timing_unb | 0.25±0.24% | 0.25±0.24% | 0.25±0.24% | 0.25±0.24% | 100.00±0.00% | 100.00±0.00% | 46.51±45.14% | 30.78±41.05% | 0.03±0.06% | 2.662±4.226 |
| prim_rand_padding_unb | 16.68±22.36% | 16.68±22.36% | 4.05±5.97% | 4.05±5.97% | 100.00±0.00% | 100.00±0.00% | 40.42±33.15% | 30.78±41.05% | 0.93±0.98% | 0.061±0.037 |

### ft_transformer

| Attack | Raw ASR | Valid ASR | Tgt-Benign | Valid Tgt-Benign | Domain-valid | Realizable | IDR | SemPreserve | SP-ASR | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| pgd_untargeted | 97.29±4.81% | 0.00±0.00% | 88.05±14.05% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 7.74±8.67% | NA | NA | 0.444±0.018 |
| cw_untargeted | 81.81±26.81% | 0.00±0.00% | 55.38±41.58% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 30.75±20.68% | NA | NA | 0.062±0.018 |
| pgd_tb | 98.48±2.71% | 0.00±0.00% | 97.04±5.30% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 9.07±11.86% | NA | NA | 0.444±0.017 |
| cw_tb | 81.66±26.84% | 0.00±0.00% | 81.66±26.84% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 40.72±35.04% | NA | NA | 0.060±0.018 |
| prim_opt_joint_p50 | 4.47±8.08% | 4.47±8.08% | 4.47±8.08% | 4.47±8.08% | 100.00±0.00% | 100.00±0.00% | 71.81±31.88% | 31.09±41.02% | 0.00±0.00% | 0.014±0.017 |
| prim_opt_timing_p50 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 75.69±32.98% | 31.09±41.02% | 0.00±0.00% | 0.000±0.000 |
| prim_opt_padding_p50 | 4.47±8.08% | 4.47±8.08% | 4.47±8.08% | 4.47±8.08% | 100.00±0.00% | 100.00±0.00% | 71.82±31.88% | 31.09±41.02% | 0.00±0.00% | 0.014±0.016 |
| prim_rand_joint_p50 | 3.33±6.03% | 3.33±6.03% | 3.33±6.03% | 3.33±6.03% | 100.00±0.00% | 100.00±0.00% | 74.42±32.44% | 31.09±41.02% | 0.00±0.00% | 0.026±0.019 |
| prim_rand_timing_p50 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 75.29±32.75% | 31.09±41.02% | 0.00±0.00% | 0.009±0.010 |
| prim_rand_padding_p50 | 3.33±6.03% | 3.33±6.03% | 3.33±6.03% | 3.33±6.03% | 100.00±0.00% | 100.00±0.00% | 74.76±32.64% | 31.09±41.02% | 0.00±0.00% | 0.016±0.021 |
| prim_opt_joint_p75 | 4.67±8.37% | 4.67±8.37% | 4.67±8.37% | 4.67±8.37% | 100.00±0.00% | 100.00±0.00% | 53.04±45.14% | 31.09±41.02% | 0.00±0.00% | 0.033±0.037 |
| prim_opt_timing_p75 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 75.55±32.99% | 31.09±41.02% | 0.00±0.00% | 0.001±0.001 |
| prim_opt_padding_p75 | 4.67±8.37% | 4.67±8.37% | 4.67±8.37% | 4.67±8.37% | 100.00±0.00% | 100.00±0.00% | 53.07±45.12% | 31.09±41.02% | 0.00±0.00% | 0.032±0.037 |
| prim_rand_joint_p75 | 3.59±6.43% | 3.59±6.43% | 3.59±6.43% | 3.59±6.43% | 100.00±0.00% | 100.00±0.00% | 56.95±39.31% | 31.09±41.02% | 0.00±0.00% | 0.051±0.040 |
| prim_rand_timing_p75 | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 100.00±0.00% | 72.55±31.69% | 31.09±41.02% | 0.00±0.00% | 0.015±0.013 |
| prim_rand_padding_p75 | 3.59±6.43% | 3.59±6.43% | 3.59±6.43% | 3.59±6.43% | 100.00±0.00% | 100.00±0.00% | 57.84±39.40% | 31.09±41.02% | 0.00±0.00% | 0.036±0.037 |
| prim_opt_joint_unb | 5.19±8.96% | 5.19±8.96% | 5.19±8.96% | 5.19±8.96% | 100.00±0.00% | 100.00±0.00% | 26.97±41.17% | 31.09±41.02% | 0.03±0.06% | 0.514±0.683 |
| prim_opt_timing_unb | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 100.00±0.00% | 100.00±0.00% | 49.95±47.81% | 31.09±41.02% | 0.00±0.00% | 0.422±0.622 |
| prim_opt_padding_unb | 5.01±8.99% | 5.01±8.99% | 5.01±8.99% | 5.01±8.99% | 100.00±0.00% | 100.00±0.00% | 30.77±39.53% | 31.09±41.02% | 0.00±0.00% | 0.062±0.041 |
| prim_rand_joint_unb | 4.21±7.24% | 4.21±7.24% | 4.21±7.24% | 4.21±7.24% | 100.00±0.00% | 100.00±0.00% | 30.90±39.59% | 31.09±41.02% | 0.01±0.04% | 2.723±4.190 |
| prim_rand_timing_unb | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 0.16±0.28% | 100.00±0.00% | 100.00±0.00% | 46.70±44.95% | 31.09±41.02% | 0.00±0.00% | 2.662±4.226 |
| prim_rand_padding_unb | 4.07±7.30% | 4.07±7.30% | 4.07±7.30% | 4.07±7.30% | 100.00±0.00% | 100.00±0.00% | 40.90±32.82% | 31.09±41.02% | 0.00±0.00% | 0.061±0.037 |

_SemPreserve/SP-ASR are PrimAttack-only (NA for input baselines); for Recon/BruteForce the semantic proxy is NOT_TESTABLE by design (see §8)._

## 5. Paired sample-level McNemar tests (within victim, reference seed = 42)

A=first method, B=second. n10=A-success∧B-fail, n01=B-success∧A-fail. prop_diff = P(A)−P(B); OR = (n10+.5)/(n01+.5).

| Victim | Comparison | success | n | n11 | n10 | n01 | χ² | p | OR | P(A)−P(B) [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | pgd_tb vs prim_opt_joint_p75 (raw targeted) | targeted | 3200 | 5 | 3195 | 0 | 3193.0 | 0.00e+00 | 6391.00 | +99.8% [+99.7,+100.0] |
| mlp | pgd_tb vs prim_opt_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 5 | 3.2 | 6.25e-02 | 0.09 | -0.2% [-0.3,-0.0] |
| mlp | pgd_tb vs prim_opt_joint_p50 (raw targeted) | targeted | 3200 | 2 | 3198 | 0 | 3196.0 | 0.00e+00 | 6397.00 | +99.9% [+99.9,+100.0] |
| mlp | pgd_tb vs prim_opt_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 2 | 0.5 | 5.00e-01 | 0.20 | -0.1% [-0.1,+0.0] |
| mlp | pgd_tb vs prim_opt_joint_unb (raw targeted) | targeted | 3200 | 16 | 3184 | 0 | 3182.0 | 0.00e+00 | 6369.00 | +99.5% [+99.3,+99.7] |
| mlp | pgd_tb vs prim_opt_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 16 | 14.1 | 3.05e-05 | 0.03 | -0.5% [-0.7,-0.3] |
| mlp | cw_tb vs prim_opt_joint_p75 (raw targeted) | targeted | 3200 | 5 | 3195 | 0 | 3193.0 | 0.00e+00 | 6391.00 | +99.8% [+99.7,+100.0] |
| mlp | cw_tb vs prim_opt_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 5 | 3.2 | 6.25e-02 | 0.09 | -0.2% [-0.3,-0.0] |
| mlp | cw_tb vs prim_opt_joint_p50 (raw targeted) | targeted | 3200 | 2 | 3198 | 0 | 3196.0 | 0.00e+00 | 6397.00 | +99.9% [+99.9,+100.0] |
| mlp | cw_tb vs prim_opt_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 2 | 0.5 | 5.00e-01 | 0.20 | -0.1% [-0.1,+0.0] |
| mlp | cw_tb vs prim_opt_joint_unb (raw targeted) | targeted | 3200 | 16 | 3184 | 0 | 3182.0 | 0.00e+00 | 6369.00 | +99.5% [+99.3,+99.7] |
| mlp | cw_tb vs prim_opt_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 16 | 14.1 | 3.05e-05 | 0.03 | -0.5% [-0.7,-0.3] |
| mlp | joint vs timing (p75) | targeted | 3200 | 0 | 5 | 0 | 3.2 | 6.25e-02 | 11.00 | +0.2% [+0.0,+0.3] |
| mlp | joint vs padding (p75) | targeted | 3200 | 5 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| mlp | optimized vs random (joint,p75) | targeted | 3200 | 1 | 4 | 0 | 2.2 | 1.25e-01 | 9.00 | +0.1% [+0.0,+0.2] |
| mlp | budget p50 vs p75 (opt joint) | targeted | 3200 | 2 | 0 | 3 | 1.3 | 2.50e-01 | 0.14 | -0.1% [-0.2,+0.0] |
| mlp | budget p75 vs unbounded (opt joint) | targeted | 3200 | 5 | 0 | 11 | 9.1 | 9.77e-04 | 0.04 | -0.3% [-0.5,-0.1] |
| mlp | pgd_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| mlp | cw_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| mlp | prim_opt_joint_p75: raw vs valid targeted | targeted->valid | 3200 | 5 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| mlp | prim_opt_joint_unb: raw vs valid targeted | targeted->valid | 3200 | 16 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| mlp | pgd_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| mlp | cw_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3199 | 0 | 3197.0 | 0.00e+00 | 6399.00 | +100.0% [+99.9,+100.0] |
| cnn | pgd_tb vs prim_opt_joint_p75 (raw targeted) | targeted | 3200 | 38 | 3159 | 0 | 3157.0 | 0.00e+00 | 6319.00 | +98.7% [+98.3,+99.1] |
| cnn | pgd_tb vs prim_opt_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 38 | 36.0 | 7.28e-12 | 0.01 | -1.2% [-1.6,-0.8] |
| cnn | pgd_tb vs prim_opt_joint_p50 (raw targeted) | targeted | 3200 | 33 | 3164 | 0 | 3162.0 | 0.00e+00 | 6329.00 | +98.9% [+98.5,+99.2] |
| cnn | pgd_tb vs prim_opt_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 33 | 31.0 | 2.33e-10 | 0.01 | -1.0% [-1.4,-0.7] |
| cnn | pgd_tb vs prim_opt_joint_unb (raw targeted) | targeted | 3200 | 85 | 3112 | 0 | 3110.0 | 0.00e+00 | 6225.00 | +97.2% [+96.7,+97.8] |
| cnn | pgd_tb vs prim_opt_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 85 | 83.0 | 5.17e-26 | 0.01 | -2.7% [-3.2,-2.1] |
| cnn | cw_tb vs prim_opt_joint_p75 (raw targeted) | targeted | 3200 | 38 | 3162 | 0 | 3160.0 | 0.00e+00 | 6325.00 | +98.8% [+98.4,+99.2] |
| cnn | cw_tb vs prim_opt_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 38 | 36.0 | 7.28e-12 | 0.01 | -1.2% [-1.6,-0.8] |
| cnn | cw_tb vs prim_opt_joint_p50 (raw targeted) | targeted | 3200 | 33 | 3167 | 0 | 3165.0 | 0.00e+00 | 6335.00 | +99.0% [+98.6,+99.3] |
| cnn | cw_tb vs prim_opt_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 33 | 31.0 | 2.33e-10 | 0.01 | -1.0% [-1.4,-0.7] |
| cnn | cw_tb vs prim_opt_joint_unb (raw targeted) | targeted | 3200 | 85 | 3115 | 0 | 3113.0 | 0.00e+00 | 6231.00 | +97.3% [+96.8,+97.9] |
| cnn | cw_tb vs prim_opt_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 85 | 83.0 | 5.17e-26 | 0.01 | -2.7% [-3.2,-2.1] |
| cnn | joint vs timing (p75) | targeted | 3200 | 0 | 38 | 0 | 36.0 | 7.28e-12 | 77.00 | +1.2% [+0.8,+1.6] |
| cnn | joint vs padding (p75) | targeted | 3200 | 38 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| cnn | optimized vs random (joint,p75) | targeted | 3200 | 19 | 19 | 77 | 33.8 | 1.86e-09 | 0.25 | -1.8% [-2.4,-1.2] |
| cnn | budget p50 vs p75 (opt joint) | targeted | 3200 | 33 | 0 | 5 | 3.2 | 6.25e-02 | 0.09 | -0.2% [-0.3,-0.0] |
| cnn | budget p75 vs unbounded (opt joint) | targeted | 3200 | 38 | 0 | 47 | 45.0 | 1.42e-14 | 0.01 | -1.5% [-1.9,-1.1] |
| cnn | pgd_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3197 | 0 | 3195.0 | 0.00e+00 | 6395.00 | +99.9% [+99.8,+100.0] |
| cnn | cw_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3200 | 0 | 3198.0 | 0.00e+00 | 6401.00 | +100.0% [+100.0,+100.0] |
| cnn | prim_opt_joint_p75: raw vs valid targeted | targeted->valid | 3200 | 38 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| cnn | prim_opt_joint_unb: raw vs valid targeted | targeted->valid | 3200 | 85 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| cnn | pgd_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3039 | 0 | 3037.0 | 0.00e+00 | 6079.00 | +95.0% [+94.2,+95.7] |
| cnn | cw_untargeted: raw vs valid evasion | evasion->valid | 3200 | 0 | 3132 | 0 | 3130.0 | 0.00e+00 | 6265.00 | +97.9% [+97.4,+98.4] |
| ft_transformer | pgd_tb vs prim_opt_joint_p75 (raw targeted) | targeted | 3200 | 149 | 2967 | 0 | 2965.0 | 0.00e+00 | 5935.00 | +92.7% [+91.8,+93.6] |
| ft_transformer | pgd_tb vs prim_opt_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 149 | 147.0 | 2.80e-45 | 0.00 | -4.7% [-5.4,-3.9] |
| ft_transformer | pgd_tb vs prim_opt_joint_p50 (raw targeted) | targeted | 3200 | 142 | 2974 | 0 | 2972.0 | 0.00e+00 | 5949.00 | +92.9% [+92.0,+93.8] |
| ft_transformer | pgd_tb vs prim_opt_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 142 | 140.0 | 3.59e-43 | 0.00 | -4.4% [-5.2,-3.7] |
| ft_transformer | pgd_tb vs prim_opt_joint_unb (raw targeted) | targeted | 3200 | 165 | 2951 | 0 | 2949.0 | 0.00e+00 | 5903.00 | +92.2% [+91.3,+93.1] |
| ft_transformer | pgd_tb vs prim_opt_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 165 | 163.0 | 4.28e-50 | 0.00 | -5.2% [-5.9,-4.4] |
| ft_transformer | cw_tb vs prim_opt_joint_p75 (raw targeted) | targeted | 3200 | 149 | 2464 | 0 | 2462.0 | 0.00e+00 | 4929.00 | +77.0% [+75.5,+78.5] |
| ft_transformer | cw_tb vs prim_opt_joint_p75 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 149 | 147.0 | 2.80e-45 | 0.00 | -4.7% [-5.4,-3.9] |
| ft_transformer | cw_tb vs prim_opt_joint_p50 (raw targeted) | targeted | 3200 | 142 | 2471 | 0 | 2469.0 | 0.00e+00 | 4943.00 | +77.2% [+75.8,+78.7] |
| ft_transformer | cw_tb vs prim_opt_joint_p50 (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 142 | 140.0 | 3.59e-43 | 0.00 | -4.4% [-5.2,-3.7] |
| ft_transformer | cw_tb vs prim_opt_joint_unb (raw targeted) | targeted | 3200 | 165 | 2448 | 0 | 2446.0 | 0.00e+00 | 4897.00 | +76.5% [+75.0,+78.0] |
| ft_transformer | cw_tb vs prim_opt_joint_unb (VALID targeted) | targeted_valid | 3200 | 0 | 0 | 165 | 163.0 | 4.28e-50 | 0.00 | -5.2% [-5.9,-4.4] |
| ft_transformer | joint vs timing (p75) | targeted | 3200 | 0 | 149 | 0 | 147.0 | 2.80e-45 | 299.00 | +4.7% [+3.9,+5.4] |
| ft_transformer | joint vs padding (p75) | targeted | 3200 | 149 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| ft_transformer | optimized vs random (joint,p75) | targeted | 3200 | 110 | 39 | 3 | 29.2 | 5.63e-09 | 11.29 | +1.1% [+0.7,+1.5] |
| ft_transformer | budget p50 vs p75 (opt joint) | targeted | 3200 | 141 | 1 | 8 | 4.0 | 3.91e-02 | 0.18 | -0.2% [-0.4,-0.0] |
| ft_transformer | budget p75 vs unbounded (opt joint) | targeted | 3200 | 149 | 0 | 16 | 14.1 | 3.05e-05 | 0.03 | -0.5% [-0.7,-0.3] |
| ft_transformer | pgd_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 3116 | 0 | 3114.0 | 0.00e+00 | 6233.00 | +97.4% [+96.8,+97.9] |
| ft_transformer | cw_tb: raw vs valid targeted | targeted->valid | 3200 | 0 | 2613 | 0 | 2611.0 | 0.00e+00 | 5227.00 | +81.7% [+80.3,+83.0] |
| ft_transformer | prim_opt_joint_p75: raw vs valid targeted | targeted->valid | 3200 | 149 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
| ft_transformer | prim_opt_joint_unb: raw vs valid targeted | targeted->valid | 3200 | 165 | 0 | 0 | 0.0 | 1.00e+00 | 1.00 | +0.0% [+0.0,+0.0] |
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
| prim_opt_joint_unb | DoS | 1.21±0.07% | 1.21±0.07% | 100.00±0.00% | 21.62±0.00% |
| prim_opt_joint_unb | DDoS | 0.21±0.07% | 0.21±0.07% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_unb | Recon | 0.63±0.00% | 0.63±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_unb | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p75 | DoS | 0.58±0.07% | 0.58±0.07% | 100.00±0.00% | 21.62±0.00% |
| prim_opt_joint_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p50 | DoS | 0.38±0.12% | 0.38±0.12% | 100.00±0.00% | 21.62±0.00% |
| prim_opt_joint_p50 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_p50 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p50 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_timing_p75 | DoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 21.62±0.00% |
| prim_opt_timing_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_timing_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_timing_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_padding_p75 | DoS | 0.58±0.07% | 0.58±0.07% | 100.00±0.00% | 21.62±0.00% |
| prim_opt_padding_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_padding_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_padding_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | DoS | 0.12±0.00% | 0.12±0.00% | 100.00±0.00% | 21.62±0.00% |
| prim_rand_joint_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_rand_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |

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
| prim_opt_joint_unb | DoS | 5.62±0.25% | 5.62±0.25% | 100.00±0.00% | 27.00±0.00% |
| prim_opt_joint_unb | DDoS | 3.38±0.25% | 3.38±0.25% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_unb | Recon | 0.63±0.00% | 0.63±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_unb | BruteForce | 0.75±0.00% | 0.75±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p75 | DoS | 4.29±0.07% | 4.29±0.07% | 100.00±0.00% | 27.00±0.00% |
| prim_opt_joint_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p75 | BruteForce | 0.25±0.12% | 0.25±0.12% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p50 | DoS | 4.17±0.07% | 4.17±0.07% | 100.00±0.00% | 27.00±0.00% |
| prim_opt_joint_p50 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_p50 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p50 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_timing_p75 | DoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 27.00±0.00% |
| prim_opt_timing_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_timing_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_timing_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_padding_p75 | DoS | 4.29±0.07% | 4.29±0.07% | 100.00±0.00% | 27.00±0.00% |
| prim_opt_padding_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_padding_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_padding_p75 | BruteForce | 0.25±0.12% | 0.25±0.12% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | DoS | 11.67±0.19% | 11.67±0.19% | 100.00±0.00% | 27.00±0.00% |
| prim_rand_joint_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_rand_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | BruteForce | 0.38±0.00% | 0.38±0.00% | 100.00±0.00% | 0.00±0.00% |

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
| prim_opt_joint_unb | DoS | 20.04±0.14% | 20.04±0.14% | 100.00±0.00% | 28.25±0.00% |
| prim_opt_joint_unb | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_unb | Recon | 0.63±0.00% | 0.63±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_unb | BruteForce | 0.08±0.07% | 0.08±0.07% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p75 | DoS | 18.54±0.07% | 18.54±0.07% | 100.00±0.00% | 28.25±0.00% |
| prim_opt_joint_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p75 | BruteForce | 0.12±0.00% | 0.12±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p50 | DoS | 17.87±0.13% | 17.87±0.13% | 100.00±0.00% | 28.25±0.00% |
| prim_opt_joint_p50 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_joint_p50 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_joint_p50 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_timing_p75 | DoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 28.25±0.00% |
| prim_opt_timing_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_timing_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_timing_p75 | BruteForce | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_padding_p75 | DoS | 18.54±0.07% | 18.54±0.07% | 100.00±0.00% | 28.25±0.00% |
| prim_opt_padding_p75 | DDoS | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 96.12±0.00% |
| prim_opt_padding_p75 | Recon | 0.00±0.00% | 0.00±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_opt_padding_p75 | BruteForce | 0.12±0.00% | 0.12±0.00% | 100.00±0.00% | 0.00±0.00% |
| prim_rand_joint_p75 | DoS | 14.25±0.54% | 14.25±0.54% | 100.00±0.00% | 28.25±0.00% |
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
| mlp | prim_opt_joint_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_timing_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_padding_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_joint_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_timing_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_padding_p50 | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_joint_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_timing_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_padding_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_joint_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_timing_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_padding_p75 | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_joint_unb | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_timing_unb | 0.00% | 0.00% | 0.00% |
| mlp | prim_opt_padding_unb | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_joint_unb | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_timing_unb | 0.00% | 0.00% | 0.00% |
| mlp | prim_rand_padding_unb | 0.00% | 0.00% | 0.00% |
| cnn | pgd_untargeted | 0.00% | 0.00% | 0.00% |
| cnn | cw_untargeted | 0.00% | 0.00% | 0.00% |
| cnn | pgd_tb | 0.00% | 0.00% | 0.00% |
| cnn | cw_tb | 0.00% | 0.00% | 0.00% |
| cnn | prim_opt_joint_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_opt_timing_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_opt_padding_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_rand_joint_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_rand_timing_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_rand_padding_p50 | 0.00% | 0.00% | 0.00% |
| cnn | prim_opt_joint_p75 | 0.38% | 0.25% | 0.12% |
| cnn | prim_opt_timing_p75 | 0.00% | 0.00% | 0.00% |
| cnn | prim_opt_padding_p75 | 0.38% | 0.25% | 0.12% |
| cnn | prim_rand_joint_p75 | 0.38% | 0.38% | 0.38% |
| cnn | prim_rand_timing_p75 | 0.00% | 0.00% | 0.00% |
| cnn | prim_rand_padding_p75 | 0.38% | 0.38% | 0.38% |
| cnn | prim_opt_joint_unb | 0.75% | 0.75% | 0.75% |
| cnn | prim_opt_timing_unb | 0.38% | 0.38% | 0.38% |
| cnn | prim_opt_padding_unb | 0.38% | 0.25% | 0.12% |
| cnn | prim_rand_joint_unb | 0.50% | 0.62% | 0.75% |
| cnn | prim_rand_timing_unb | 0.25% | 0.25% | 0.25% |
| cnn | prim_rand_padding_unb | 0.38% | 0.38% | 0.38% |
| ft_transformer | pgd_untargeted | 0.00% | 0.00% | 0.00% |
| ft_transformer | cw_untargeted | 0.00% | 0.00% | 0.00% |
| ft_transformer | pgd_tb | 0.00% | 0.00% | 0.00% |
| ft_transformer | cw_tb | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_opt_joint_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_opt_timing_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_opt_padding_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_joint_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_timing_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_padding_p50 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_opt_joint_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_opt_timing_p75 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_opt_padding_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_rand_joint_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_rand_timing_p75 | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_padding_p75 | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_opt_joint_unb | 0.12% | 0.12% | 0.00% |
| ft_transformer | prim_opt_timing_unb | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_opt_padding_unb | 0.12% | 0.12% | 0.12% |
| ft_transformer | prim_rand_joint_unb | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_timing_unb | 0.00% | 0.00% | 0.00% |
| ft_transformer | prim_rand_padding_unb | 0.12% | 0.12% | 0.12% |

_Full per-seed metrics: `outputs/full_adv_eval/per_seed_cells.csv`._

## 8. Failures / skipped / not-testable

- Non-finite / eligibility failures logged: 0 (`outputs/full_adv_eval/failures.json`).
- Subsampled classes (N-cap < eligible): mlp/DoS (800/25583), mlp/DDoS (800/14243), mlp/Recon (800/23771), mlp/BruteForce (800/1022), cnn/DoS (800/25540), cnn/DDoS (800/14244), cnn/Recon (800/23760), cnn/BruteForce (800/1018), ft_transformer/DoS (800/25522), ft_transformer/DDoS (800/14257), ft_transformer/Recon (800/23788), ft_transformer/BruteForce (800/1026).
- Semantic-preservation proxy is **NOT_TESTABLE by construction for Recon and BruteForce** (no retained class-level semantic rule), so SemPreserve/SP-ASR for those classes reflect testability, not failure; DoS/DDoS use the rate-retention rule.

## 9. Interpretation

- **mlp**: unconstrained input-PGD raw ASR ≈ 100.0% but VALID ASR ≈ 0.0% (domain gate rejects the free perturbation). PrimAttack (opt-joint,p75) targeted-benign ≈ 0.1%, valid targeted-benign ≈ 0.1%, domain-validity ≈ 100.0%.
- **cnn**: unconstrained input-PGD raw ASR ≈ 95.2% but VALID ASR ≈ 0.0% (domain gate rejects the free perturbation). PrimAttack (opt-joint,p75) targeted-benign ≈ 1.1%, valid targeted-benign ≈ 1.1%, domain-validity ≈ 100.0%.
- **ft_transformer**: unconstrained input-PGD raw ASR ≈ 97.3% but VALID ASR ≈ 0.0% (domain gate rejects the free perturbation). PrimAttack (opt-joint,p75) targeted-benign ≈ 4.7%, valid targeted-benign ≈ 4.7%, domain-validity ≈ 100.0%.

- **Raw vs valid success** (paired, within attack): for the unconstrained baselines n01=0 and n10=all successes → validity strictly and significantly removes success (the free-perturbation adversarials are domain-invalid). PrimAttack keeps validity by construction, so its raw and valid targeted counts coincide.
- **Baseline vs PrimAttack** paired tests quantify the trade: baselines dominate on RAW targeted success but PrimAttack dominates on VALID targeted success wherever the sign of P(A)−P(B) flips between the raw and valid rows above.
- **Variants**: joint ≥ timing/padding (timing-only alone flips ~nothing; padding drives joint); optimized vs random-feasible is **victim-dependent** (optimized > random on mlp/ft_transformer, but random > optimized on cnn — see the signed prop-diff/OR in §5); budget p75 ≥ p50 (looser budget → marginally more success). Read the exact signs/CI from §5.
- **Unbounded (envelope-only) PrimAttack**: removing the p25/p50/p75 empirical budget and keeping only the p99 physical envelope leaves targeted-benign success essentially unchanged vs p75 (see 'budget p75 vs unbounded' rows in §5) while domain-validity stays 100% — i.e. the empirical budget was not the binding constraint; realizability + the physical envelope are. The gap to the unconstrained input baselines is the price of staying valid/realizable.
