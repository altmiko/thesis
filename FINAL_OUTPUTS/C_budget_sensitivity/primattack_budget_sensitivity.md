# Final Experiment C — PrimAttack budget sensitivity (targeted → Benign)

Budgets are train-only per-class calibrations (`artifacts/primattack/budget_calibration*.json`, `fit_split = train`): **p50** (intermediate) and **p75** (maximum-evaluated) cap padding bytes and relative duration change at the class's train percentiles. **unbounded** removes those caps and keeps only the train-p99 feature envelope and the DoS/DDoS min-rate floor. All other settings are identical (joint mode, 256 evaluations/flow, same flows, seeds, victims, validator, success predicate). Optimizers: the top two of the Exp B ranking (Hybrid Search, Prim-PGD). Their p75 cells are the Exp B cells.

## Valid / Raw ASR by budget (side by side per optimizer)

| Dataset | Victim | Budget | Hybrid Search Raw | Hybrid Search Valid | Hybrid Search Gap | Prim-PGD Raw | Prim-PGD Valid | Prim-PGD Gap |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | p50 | 6.31% ± 0.00% | 6.31% ± 0.00% | 0.00 ± 0.00 pp | 6.31% ± 0.00% | 6.31% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | p75 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | unbounded | 41.90% ± 0.11% | 41.90% ± 0.11% | 0.00 ± 0.00 pp | 42.01% ± 0.04% | 42.01% ± 0.04% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | p50 | 13.10% ± 0.04% | 13.10% ± 0.04% | 0.00 ± 0.00 pp | 13.06% ± 0.00% | 13.06% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | p75 | 36.15% ± 0.02% | 36.15% ± 0.02% | 0.00 ± 0.00 pp | 36.10% ± 0.13% | 36.10% ± 0.13% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | unbounded | 70.48% ± 0.04% | 70.48% ± 0.04% | 0.00 ± 0.00 pp | 70.51% ± 0.02% | 70.51% ± 0.02% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | p50 | 0.16% ± 0.00% | 0.16% ± 0.00% | 0.00 ± 0.00 pp | 0.16% ± 0.00% | 0.16% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | p75 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | unbounded | 0.88% ± 0.00% | 0.88% ± 0.00% | 0.00 ± 0.00 pp | 0.88% ± 0.00% | 0.88% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | p50 | 1.09% ± 0.00% | 0.17% ± 0.02% | 0.93 ± 0.02 pp | 1.09% ± 0.00% | 0.09% ± 0.00% | 1.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | p75 | 1.28% ± 0.03% | 0.29% ± 0.04% | 0.99 ± 0.02 pp | 1.31% ± 0.00% | 0.26% ± 0.02% | 1.05 ± 0.02 pp |
| CICIDS2018 | mlp-s42 | unbounded | 24.60% ± 0.05% | 24.05% ± 0.08% | 0.55 ± 0.04 pp | 24.75% ± 0.00% | 24.17% ± 0.05% | 0.58 ± 0.05 pp |
| CICIDS2018 | cnn-s42 | p50 | 14.05% ± 0.02% | 0.00% ± 0.00% | 14.05 ± 0.02 pp | 13.94% ± 0.05% | 0.00% ± 0.00% | 13.94 ± 0.05 pp |
| CICIDS2018 | cnn-s42 | p75 | 15.00% ± 0.00% | 0.00% ± 0.00% | 15.00 ± 0.00 pp | 15.01% ± 0.02% | 0.00% ± 0.00% | 15.01 ± 0.02 pp |
| CICIDS2018 | cnn-s42 | unbounded | 45.85% ± 0.02% | 22.68% ± 0.13% | 23.18 ± 0.13 pp | 45.85% ± 0.02% | 25.10% ± 0.05% | 20.75 ± 0.03 pp |
| CICIDS2018 | ft_transformer-s42 | p50 | 0.34% ± 0.00% | 0.00% ± 0.00% | 0.34 ± 0.00 pp | 0.34% ± 0.00% | 0.00% ± 0.00% | 0.34 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | p75 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 0.32% ± 0.02% | 0.00% ± 0.00% | 0.32 ± 0.02 pp |
| CICIDS2018 | ft_transformer-s42 | unbounded | 0.78% ± 0.00% | 0.09% ± 0.00% | 0.69 ± 0.00 pp | 0.77% ± 0.02% | 0.09% ± 0.00% | 0.68 ± 0.02 pp |

## Per-seed values

| Dataset | Victim | Condition | Raw per seed (42/2024/2026, %) | Valid per seed (%) | Median primitive cost (valid) |
|---|---|---|---|---|---|
| CICIDS2017 | mlp | Hybrid Search p50 | 6.31 / 6.31 / 6.31 | 6.31 / 6.31 / 6.31 | 1.500 |
| CICIDS2017 | cnn | Hybrid Search p50 | 13.06 / 13.12 / 13.12 | 13.06 / 13.12 / 13.12 | 1.405 |
| CICIDS2017 | ft_transformer | Hybrid Search p50 | 0.16 / 0.16 / 0.16 | 0.16 / 0.16 / 0.16 | 0.200 |
| CICIDS2018 | mlp-s42 | Hybrid Search p50 | 1.09 / 1.09 / 1.09 | 0.16 / 0.19 / 0.16 | 1.000 |
| CICIDS2018 | cnn-s42 | Hybrid Search p50 | 14.03 / 14.06 / 14.06 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search p50 | 0.34 / 0.34 / 0.34 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Hybrid Search p75 | 11.06 / 11.06 / 11.06 | 11.06 / 11.06 / 11.06 | 1.400 |
| CICIDS2017 | cnn | Hybrid Search p75 | 36.16 / 36.12 / 36.16 | 36.16 / 36.12 / 36.16 | 1.608 |
| CICIDS2017 | ft_transformer | Hybrid Search p75 | 0.44 / 0.44 / 0.44 | 0.44 / 0.44 / 0.44 | 0.196 |
| CICIDS2018 | mlp-s42 | Hybrid Search p75 | 1.31 / 1.28 / 1.25 | 0.31 / 0.31 / 0.25 | 1.000 |
| CICIDS2018 | cnn-s42 | Hybrid Search p75 | 15.00 / 15.00 / 15.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search p75 | 0.34 / 0.34 / 0.31 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Hybrid Search unbounded | 42.00 / 41.78 / 41.91 | 42.00 / 41.78 / 41.91 | 0.789 |
| CICIDS2017 | cnn | Hybrid Search unbounded | 70.50 / 70.44 / 70.50 | 70.50 / 70.44 / 70.50 | 0.980 |
| CICIDS2017 | ft_transformer | Hybrid Search unbounded | 0.88 / 0.88 / 0.88 | 0.88 / 0.88 / 0.88 | 0.215 |
| CICIDS2018 | mlp-s42 | Hybrid Search unbounded | 24.66 / 24.59 / 24.56 | 24.12 / 24.06 / 23.97 | 0.600 |
| CICIDS2018 | cnn-s42 | Hybrid Search unbounded | 45.88 / 45.84 / 45.84 | 22.66 / 22.81 / 22.56 | 0.401 |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search unbounded | 0.78 / 0.78 / 0.78 | 0.09 / 0.09 / 0.09 | 0.300 |
| CICIDS2017 | mlp | Prim-PGD p50 | 6.31 / 6.31 / 6.31 | 6.31 / 6.31 / 6.31 | 1.493 |
| CICIDS2017 | cnn | Prim-PGD p50 | 13.06 / 13.06 / 13.06 | 13.06 / 13.06 / 13.06 | 1.361 |
| CICIDS2017 | ft_transformer | Prim-PGD p50 | 0.16 / 0.16 / 0.16 | 0.16 / 0.16 / 0.16 | 0.316 |
| CICIDS2018 | mlp-s42 | Prim-PGD p50 | 1.09 / 1.09 / 1.09 | 0.09 / 0.09 / 0.09 | 0.802 |
| CICIDS2018 | cnn-s42 | Prim-PGD p50 | 13.91 / 14.00 / 13.91 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD p50 | 0.34 / 0.34 / 0.34 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Prim-PGD p75 | 11.06 / 11.06 / 11.06 | 11.06 / 11.06 / 11.06 | 1.330 |
| CICIDS2017 | cnn | Prim-PGD p75 | 36.03 / 36.25 / 36.03 | 36.03 / 36.25 / 36.03 | 1.576 |
| CICIDS2017 | ft_transformer | Prim-PGD p75 | 0.44 / 0.44 / 0.44 | 0.44 / 0.44 / 0.44 | 0.205 |
| CICIDS2018 | mlp-s42 | Prim-PGD p75 | 1.31 / 1.31 / 1.31 | 0.28 / 0.25 / 0.25 | 0.927 |
| CICIDS2018 | cnn-s42 | Prim-PGD p75 | 15.00 / 15.03 / 15.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD p75 | 0.31 / 0.31 / 0.34 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Prim-PGD unbounded | 42.03 / 42.03 / 41.97 | 42.03 / 42.03 / 41.97 | 0.595 |
| CICIDS2017 | cnn | Prim-PGD unbounded | 70.50 / 70.53 / 70.50 | 70.50 / 70.53 / 70.50 | 0.595 |
| CICIDS2017 | ft_transformer | Prim-PGD unbounded | 0.88 / 0.88 / 0.88 | 0.88 / 0.88 / 0.88 | 0.204 |
| CICIDS2018 | mlp-s42 | Prim-PGD unbounded | 24.75 / 24.75 / 24.75 | 24.12 / 24.16 / 24.22 | 0.600 |
| CICIDS2018 | cnn-s42 | Prim-PGD unbounded | 45.84 / 45.88 / 45.84 | 25.09 / 25.16 / 25.06 | 0.100 |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD unbounded | 0.75 / 0.78 / 0.78 | 0.09 / 0.09 / 0.09 | 0.050 |

## Statistical analysis (Valid Targeted Success)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

Per optimizer and (dataset, victim): Cochran's Q across p50 / p75 / unbounded. If significant: the adjacent comparisons p75 vs p50 and unbounded vs p75, Holm over the two. Direct optimizer-vs-optimizer tests per budget are not part of the planned suite.

| Dataset | Victim | Family | Test | Comparison | n | A-only | B-only | Δ (pp) | Variant | Statistic | p | Holm p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 2020.46 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 2020.5, p = <1e-300); planned McNemar tests follow. |
| CICIDS2017 | mlp | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 152 | 0 | +4.75 | χ² (cc) | 150.01 | 1.73e-34 | 1.73e-34 | Hybrid Search p75 has higher Valid Targeted ASR than Hybrid Search p50 by 4.75 pp (152 vs 0 discordant flows; Holm-adjusted p = 1.73e-34). |
| CICIDS2017 | mlp | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 990 | 0 | +30.94 | χ² (cc) | 988.00 | 7.28e-217 | 1.46e-216 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 30.94 pp (990 vs 0 discordant flows; Holm-adjusted p = 1.46e-216). |
| CICIDS2017 | cnn | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 2792.26 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 2792.3, p = <1e-300); planned McNemar tests follow. |
| CICIDS2017 | cnn | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 739 | 0 | +23.09 | χ² (cc) | 737.00 | 2.69e-162 | 2.69e-162 | Hybrid Search p75 has higher Valid Targeted ASR than Hybrid Search p50 by 23.09 pp (739 vs 0 discordant flows; Holm-adjusted p = 2.69e-162). |
| CICIDS2017 | cnn | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 1099 | 0 | +34.34 | χ² (cc) | 1097.00 | 1.48e-240 | 2.96e-240 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 34.34 pp (1099 vs 0 discordant flows; Holm-adjusted p = 2.96e-240). |
| CICIDS2017 | ft_transformer | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 35.04 | 2.46e-08 | — | Valid success differs among the 3 paired conditions (Q = 35.0, p = 2.46e-08); planned McNemar tests follow. |
| CICIDS2017 | ft_transformer | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 9 | 0 | +0.28 | exact binomial |  | 0.00391 | 0.00391 | Hybrid Search p75 has higher Valid Targeted ASR than Hybrid Search p50 by 0.28 pp (9 vs 0 discordant flows; Holm-adjusted p = 0.00391). |
| CICIDS2017 | ft_transformer | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 14 | 0 | +0.44 | exact binomial |  | 0.000122 | 0.000244 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 0.44 pp (14 vs 0 discordant flows; Holm-adjusted p = 0.000244). |
| CICIDS2018 | mlp-s42 | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 1524.07 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1524.1, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | mlp-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 5 | 0 | +0.16 | exact binomial |  | 0.0625 | 0.0625 | No significant difference (Holm-adjusted p = 0.0625; Δ = +0.16 pp, 5 vs 0 discordant flows). |
| CICIDS2018 | mlp-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 762 | 0 | +23.81 | χ² (cc) | 760.00 | 2.68e-167 | 5.37e-167 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 23.81 pp (762 vs 0 discordant flows; Holm-adjusted p = 5.37e-167). |
| CICIDS2018 | cnn-s42 | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 1450.00 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1450.0, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | cnn-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 725 | 0 | +22.66 | χ² (cc) | 723.00 | 2.98e-159 | 5.96e-159 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 22.66 pp (725 vs 0 discordant flows; Holm-adjusted p = 5.96e-159). |
| CICIDS2018 | ft_transformer-s42 | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 6.00 | 0.0498 | — | Valid success differs among the 3 paired conditions (Q = 6.0, p = 0.0498); planned McNemar tests follow. |
| CICIDS2018 | ft_transformer-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | ft_transformer-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 3 | 0 | +0.09 | exact binomial |  | 0.25 | 0.5 | No significant difference (Holm-adjusted p = 0.5; Δ = +0.09 pp, 3 vs 0 discordant flows). |
| CICIDS2017 | mlp | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 2022.43 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 2022.4, p = <1e-300); planned McNemar tests follow. |
| CICIDS2017 | mlp | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 152 | 0 | +4.75 | χ² (cc) | 150.01 | 1.73e-34 | 1.73e-34 | Prim-PGD p75 has higher Valid Targeted ASR than Prim-PGD p50 by 4.75 pp (152 vs 0 discordant flows; Holm-adjusted p = 1.73e-34). |
| CICIDS2017 | mlp | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 991 | 0 | +30.97 | χ² (cc) | 989.00 | 4.42e-217 | 8.83e-217 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 30.97 pp (991 vs 0 discordant flows; Holm-adjusted p = 8.83e-217). |
| CICIDS2017 | cnn | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 2793.84 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 2793.8, p = <1e-300); planned McNemar tests follow. |
| CICIDS2017 | cnn | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 735 | 0 | +22.97 | χ² (cc) | 733.00 | 1.99e-161 | 1.99e-161 | Prim-PGD p75 has higher Valid Targeted ASR than Prim-PGD p50 by 22.97 pp (735 vs 0 discordant flows; Holm-adjusted p = 1.99e-161). |
| CICIDS2017 | cnn | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 1103 | 0 | +34.47 | χ² (cc) | 1101.00 | 2e-241 | 4e-241 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 34.47 pp (1103 vs 0 discordant flows; Holm-adjusted p = 4e-241). |
| CICIDS2017 | ft_transformer | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 35.04 | 2.46e-08 | — | Valid success differs among the 3 paired conditions (Q = 35.0, p = 2.46e-08); planned McNemar tests follow. |
| CICIDS2017 | ft_transformer | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 9 | 0 | +0.28 | exact binomial |  | 0.00391 | 0.00391 | Prim-PGD p75 has higher Valid Targeted ASR than Prim-PGD p50 by 0.28 pp (9 vs 0 discordant flows; Holm-adjusted p = 0.00391). |
| CICIDS2017 | ft_transformer | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 14 | 0 | +0.44 | exact binomial |  | 0.000122 | 0.000244 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 0.44 pp (14 vs 0 discordant flows; Holm-adjusted p = 0.000244). |
| CICIDS2018 | mlp-s42 | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 1524.11 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1524.1, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | mlp-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 6 | 0 | +0.19 | exact binomial |  | 0.0312 | 0.0312 | Prim-PGD p75 has higher Valid Targeted ASR than Prim-PGD p50 by 0.19 pp (6 vs 0 discordant flows; Holm-adjusted p = 0.0312). |
| CICIDS2018 | mlp-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 764 | 1 | +23.84 | χ² (cc) | 759.01 | 4.4e-167 | 8.81e-167 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 23.84 pp (764 vs 1 discordant flows; Holm-adjusted p = 8.81e-167). |
| CICIDS2018 | cnn-s42 | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 1606.00 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1606.0, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | cnn-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 803 | 0 | +25.09 | χ² (cc) | 801.00 | 3.27e-176 | 6.54e-176 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 25.09 pp (803 vs 0 discordant flows; Holm-adjusted p = 6.54e-176). |
| CICIDS2018 | ft_transformer-s42 | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 6.00 | 0.0498 | — | Valid success differs among the 3 paired conditions (Q = 6.0, p = 0.0498); planned McNemar tests follow. |
| CICIDS2018 | ft_transformer-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | ft_transformer-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 3 | 0 | +0.09 | exact binomial |  | 0.25 | 0.5 | No significant difference (Holm-adjusted p = 0.5; Δ = +0.09 pp, 3 vs 0 discordant flows). |

## Class-wise results

| Dataset | Victim | Class | Condition | Raw | Valid |
|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | Hybrid Search p50 | 14.50% ± 0.00% | 14.50% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search p50 | 9.88% ± 0.00% | 9.88% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search p50 | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search p50 | 23.42% ± 0.14% | 23.42% ± 0.14% |
| CICIDS2017 | cnn | DDoS | Hybrid Search p50 | 28.12% ± 0.00% | 28.12% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search p50 | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search p50 | 0.25% ± 0.00% | 0.25% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search p50 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search p50 | 0.12% ± 0.00% | 0.04% ± 0.07% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search p50 | 4.25% ± 0.00% | 0.63% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search p50 | 52.21% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search p50 | 4.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search p50 | 0.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search p50 | 1.25% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Hybrid Search p75 | 28.88% ± 0.00% | 28.88% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search p75 | 13.25% ± 0.00% | 13.25% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search p75 | 2.12% ± 0.00% | 2.12% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search p75 | 49.33% ± 0.07% | 49.33% ± 0.07% |
| CICIDS2017 | cnn | DDoS | Hybrid Search p75 | 35.87% ± 0.00% | 35.87% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search p75 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search p75 | 59.25% ± 0.00% | 59.25% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search p75 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search p75 | 1.38% ± 0.00% | 1.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search p75 | 0.38% ± 0.12% | 0.29% ± 0.14% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search p75 | 4.75% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search p75 | 55.88% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search p75 | 4.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search p75 | 0.08% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search p75 | 1.25% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Hybrid Search unbounded | 89.12% ± 0.00% | 89.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search unbounded | 74.96% ± 0.44% | 74.96% ± 0.44% |
| CICIDS2017 | mlp | Recon | Hybrid Search unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search unbounded | 2.50% ± 0.00% | 2.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search unbounded | 92.62% ± 0.00% | 92.62% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Hybrid Search unbounded | 88.29% ± 0.14% | 88.29% ± 0.14% |
| CICIDS2017 | cnn | Recon | Hybrid Search unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search unbounded | 100.00% ± 0.00% | 100.00% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search unbounded | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search unbounded | 0.75% ± 0.00% | 0.75% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search unbounded | 2.25% ± 0.00% | 2.25% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search unbounded | 1.42% ± 0.19% | 1.25% ± 0.25% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search unbounded | 6.38% ± 0.00% | 4.33% ± 0.07% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search unbounded | 90.62% ± 0.00% | 90.62% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search unbounded | 77.04% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search unbounded | 6.38% ± 0.00% | 1.75% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search unbounded | 100.00% ± 0.00% | 88.96% ± 0.51% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search unbounded | 3.12% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD p50 | 14.50% ± 0.00% | 14.50% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD p50 | 9.88% ± 0.00% | 9.88% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD p50 | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD p50 | 23.25% ± 0.00% | 23.25% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD p50 | 28.12% ± 0.00% | 28.12% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD p50 | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD p50 | 0.25% ± 0.00% | 0.25% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD p50 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD p50 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD p50 | 4.25% ± 0.00% | 0.25% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD p50 | 51.75% ± 0.22% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD p50 | 4.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD p50 | 0.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD p50 | 1.25% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD p75 | 28.88% ± 0.00% | 28.88% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD p75 | 13.25% ± 0.00% | 13.25% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD p75 | 2.12% ± 0.00% | 2.12% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD p75 | 49.29% ± 0.40% | 49.29% ± 0.40% |
| CICIDS2017 | cnn | DDoS | Prim-PGD p75 | 35.87% ± 0.00% | 35.87% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD p75 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD p75 | 59.13% ± 0.13% | 59.13% ± 0.13% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD p75 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD p75 | 1.38% ± 0.00% | 1.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD p75 | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD p75 | 4.75% ± 0.00% | 0.54% ± 0.07% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD p75 | 55.92% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD p75 | 4.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD p75 | 0.08% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD p75 | 1.21% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD unbounded | 89.12% ± 0.00% | 89.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD unbounded | 75.42% ± 0.14% | 75.42% ± 0.14% |
| CICIDS2017 | mlp | Recon | Prim-PGD unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD unbounded | 2.50% ± 0.00% | 2.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD unbounded | 92.62% ± 0.00% | 92.62% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD unbounded | 88.42% ± 0.07% | 88.42% ± 0.07% |
| CICIDS2017 | cnn | Recon | Prim-PGD unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD unbounded | 100.00% ± 0.00% | 100.00% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD unbounded | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD unbounded | 0.75% ± 0.00% | 0.75% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD unbounded | 2.25% ± 0.00% | 2.25% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD unbounded | 2.00% ± 0.00% | 2.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD unbounded | 6.38% ± 0.00% | 4.04% ± 0.19% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD unbounded | 90.62% ± 0.00% | 90.62% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD unbounded | 77.04% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD unbounded | 6.38% ± 0.00% | 2.50% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD unbounded | 100.00% ± 0.00% | 97.92% ± 0.19% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD unbounded | 3.08% ± 0.07% | 0.38% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |

## Plots

- `plots/C1_raw_asr_vs_budget.png`
- `plots/C2_valid_asr_vs_budget.png`
- `plots/C3_validity_gap_vs_budget.png`

## Interpretation

**Valid Targeted ASR grows with attacker capability, but the budget that matters differs by
dataset.** Values are for Hybrid Search. Prim-PGD differs by ≤ 0.12 pp everywhere except CICIDS2018
CNN unbounded, where it reaches 25.10% vs 22.68%.

| Victim | p50 | p75 | unbounded |
|---|---|---|---|
| CICIDS2017 MLP | 6.31% | 11.06% | 41.90% |
| CICIDS2017 CNN | 13.10% | 36.15% | 70.48% |
| CICIDS2017 FT-Transformer | 0.16% | 0.44% | 0.88% |
| CICIDS2018 MLP | 0.17% | 0.29% | 24.05% |
| CICIDS2018 CNN | 0.00% | 0.00% | 22.68% |
| CICIDS2018 FT-Transformer | 0.00% | 0.00% | 0.09% |

**Planned tests (seed 42, Holm over the two adjacent comparisons).** On CICIDS2017 both steps
are significant for every victim and both optimizers. p50 → p75 adds +4.75 pp (MLP), +23.09 pp
(CNN) and +0.28 pp (FT). p75 → unbounded adds +30.94 / +34.34 / +0.44 pp. On CICIDS2018
p50 → p75 changes little. It is significant only for Prim-PGD on MLP (+0.19 pp, Holm p = 0.031).
Hybrid MLP (+0.16 pp, p = 0.063), CNN (0 discordant flows) and FT are not significant.
p75 → unbounded is large and significant for MLP (+23.81 pp) and CNN (+22.66 pp; Prim-PGD
+25.09 pp). It is not significant for FT (+0.09 pp, 3 vs 0 flows; Holm p = 0.5), although its
Cochran's Q is borderline (p = 0.0498). Across all 24 adjacent comparisons, every discordant
flow favors the larger budget except one (Prim-PGD, CICIDS2018 MLP, 764 vs 1). A larger box
almost never loses a success.

**Validity gap vs budget.** On CICIDS2017 the gap is 0.00 pp at every budget: every raw
success is valid. On CICIDS2018 CNN, raw targeted success is already 14.05% at p50 and 15.00% at
p75, but none of it is valid (gap 14.05 / 15.00 pp). At p75 every such example uses padding and
breaks the train-mined `MINED_0001` invariant (Exp E). Unbounded raises raw success to 45.85% and valid
success to 22.68%. The gap widens to 23.18 pp, so validity absorbs a large share of the extra
raw success.

**Reading.** The train-calibrated p50/p75 budgets on CICIDS2018 allow little timing freedom.
The median per-flow delay cap at p75 is about 31 ms, against about 0.9 s on CICIDS2017 (Exp A
primitive-cost table). Valid evasion there needs larger timing changes, which only the
envelope-only unbounded budget allows. On CICIDS2017 the same percentile budgets already allow
sizable valid evasion of MLP and CNN.
FT-Transformer stays at or below 0.88% under every budget on both datasets. Budget is thus a
first-order control of valid attack success (Contribution 5). The p75 headline figures are
budget-conditional and should always be reported with their budget. The unbounded budget is
a stress test bounded only by the train-p99 feature envelope and the DoS/DDoS min-rate floor.
It is not a realistic attacker budget.
