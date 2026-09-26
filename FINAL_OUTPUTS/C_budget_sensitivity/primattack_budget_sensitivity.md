# Final Experiment C — PrimAttack budget sensitivity (targeted → Benign)

Budgets are train-only per-class calibrations (`artifacts/primattack/budget_calibration*.json`, `fit_split = train`): **p50** (intermediate) and **p75** (maximum-evaluated) cap padding bytes and relative duration change at the class's train percentiles. **unbounded** removes those caps and keeps only the train-p99 feature envelope and the DoS/DDoS min-rate floor. All other settings are identical (joint mode, 256 evaluations/flow, same flows, seeds, victims, validator, success predicate). Optimizers: the top two of the Exp B ranking (Prim-PGD, Hybrid Search). Their p75 cells are the Exp B cells.

## Valid / Raw ASR by budget (side by side per optimizer)

| Dataset | Victim | Budget | Prim-PGD Raw | Prim-PGD Valid | Prim-PGD Gap | Hybrid Search Raw | Hybrid Search Valid | Hybrid Search Gap |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | p50 | 2.31% ± 0.00% | 2.31% ± 0.00% | 0.00 ± 0.00 pp | 2.31% ± 0.00% | 2.31% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | p75 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | unbounded | 22.94% ± 0.00% | 22.94% ± 0.00% | 0.00 ± 0.00 pp | 22.94% ± 0.00% | 22.94% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | p50 | 9.19% ± 0.00% | 9.19% ± 0.00% | 0.00 ± 0.00 pp | 9.19% ± 0.00% | 9.19% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | p75 | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | unbounded | 59.69% ± 0.00% | 59.69% ± 0.00% | 0.00 ± 0.00 pp | 59.69% ± 0.00% | 59.69% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | p50 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | p75 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | unbounded | 0.59% ± 0.00% | 0.59% ± 0.00% | 0.00 ± 0.00 pp | 0.55% ± 0.02% | 0.55% ± 0.02% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | p50 | 0.69% ± 0.00% | 0.69% ± 0.00% | 0.00 ± 0.00 pp | 0.69% ± 0.00% | 0.69% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | p75 | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 ± 0.00 pp | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | unbounded | 24.76% ± 0.02% | 24.76% ± 0.02% | 0.00 ± 0.00 pp | 24.80% ± 0.02% | 24.80% ± 0.02% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | p50 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | p75 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | unbounded | 26.09% ± 0.00% | 26.09% ± 0.00% | 0.00 ± 0.00 pp | 26.09% ± 0.00% | 26.09% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | p50 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | p75 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | unbounded | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp |

## Per-seed values

| Dataset | Victim | Condition | Raw per seed (42/2024/2026, %) | Valid per seed (%) | Median primitive cost (valid) |
|---|---|---|---|---|---|
| CICIDS2017 | mlp | Prim-PGD p50 | 2.31 / 2.31 / 2.31 | 2.31 / 2.31 / 2.31 | 0.689 |
| CICIDS2017 | cnn | Prim-PGD p50 | 9.19 / 9.19 / 9.19 | 9.19 / 9.19 / 9.19 | 0.629 |
| CICIDS2017 | ft_transformer | Prim-PGD p50 | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | 0.370 |
| CICIDS2018 | mlp-s42 | Prim-PGD p50 | 0.69 / 0.69 / 0.69 | 0.69 / 0.69 / 0.69 | 0.433 |
| CICIDS2018 | cnn-s42 | Prim-PGD p50 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD p50 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Prim-PGD p75 | 4.09 / 4.09 / 4.09 | 4.09 / 4.09 / 4.09 | 0.617 |
| CICIDS2017 | cnn | Prim-PGD p75 | 13.25 / 13.25 / 13.25 | 13.25 / 13.25 / 13.25 | 0.646 |
| CICIDS2017 | ft_transformer | Prim-PGD p75 | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | 0.265 |
| CICIDS2018 | mlp-s42 | Prim-PGD p75 | 0.78 / 0.78 / 0.78 | 0.78 / 0.78 / 0.78 | 0.453 |
| CICIDS2018 | cnn-s42 | Prim-PGD p75 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD p75 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Prim-PGD unbounded | 22.94 / 22.94 / 22.94 | 22.94 / 22.94 / 22.94 | 0.300 |
| CICIDS2017 | cnn | Prim-PGD unbounded | 59.69 / 59.69 / 59.69 | 59.69 / 59.69 / 59.69 | 0.400 |
| CICIDS2017 | ft_transformer | Prim-PGD unbounded | 0.59 / 0.59 / 0.59 | 0.59 / 0.59 / 0.59 | 0.050 |
| CICIDS2018 | mlp-s42 | Prim-PGD unbounded | 24.75 / 24.75 / 24.78 | 24.75 / 24.75 / 24.78 | 0.600 |
| CICIDS2018 | cnn-s42 | Prim-PGD unbounded | 26.09 / 26.09 / 26.09 | 26.09 / 26.09 / 26.09 | 0.100 |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD unbounded | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | 0.075 |
| CICIDS2017 | mlp | Hybrid Search p50 | 2.31 / 2.31 / 2.31 | 2.31 / 2.31 / 2.31 | 0.677 |
| CICIDS2017 | cnn | Hybrid Search p50 | 9.19 / 9.19 / 9.19 | 9.19 / 9.19 / 9.19 | 0.594 |
| CICIDS2017 | ft_transformer | Hybrid Search p50 | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | 0.356 |
| CICIDS2018 | mlp-s42 | Hybrid Search p50 | 0.69 / 0.69 / 0.69 | 0.69 / 0.69 / 0.69 | 0.458 |
| CICIDS2018 | cnn-s42 | Hybrid Search p50 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search p50 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Hybrid Search p75 | 4.09 / 4.09 / 4.09 | 4.09 / 4.09 / 4.09 | 0.635 |
| CICIDS2017 | cnn | Hybrid Search p75 | 13.25 / 13.25 / 13.25 | 13.25 / 13.25 / 13.25 | 0.596 |
| CICIDS2017 | ft_transformer | Hybrid Search p75 | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | 0.285 |
| CICIDS2018 | mlp-s42 | Hybrid Search p75 | 0.78 / 0.78 / 0.78 | 0.78 / 0.78 / 0.78 | 0.475 |
| CICIDS2018 | cnn-s42 | Hybrid Search p75 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search p75 | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | Hybrid Search unbounded | 22.94 / 22.94 / 22.94 | 22.94 / 22.94 / 22.94 | 0.300 |
| CICIDS2017 | cnn | Hybrid Search unbounded | 59.69 / 59.69 / 59.69 | 59.69 / 59.69 / 59.69 | 0.385 |
| CICIDS2017 | ft_transformer | Hybrid Search unbounded | 0.56 / 0.53 / 0.56 | 0.56 / 0.53 / 0.56 | 0.049 |
| CICIDS2018 | mlp-s42 | Hybrid Search unbounded | 24.78 / 24.81 / 24.81 | 24.78 / 24.81 / 24.81 | 0.600 |
| CICIDS2018 | cnn-s42 | Hybrid Search unbounded | 26.09 / 26.09 / 26.09 | 26.09 / 26.09 / 26.09 | 0.163 |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search unbounded | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | 0.100 |

## Statistical analysis (Valid Targeted Success)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

Per optimizer and (dataset, victim): Cochran's Q across p50 / p75 / unbounded. If significant: the adjacent comparisons p75 vs p50 and unbounded vs p75, Holm over the two. Direct optimizer-vs-optimizer tests per budget are not part of the planned suite.

| Dataset | Victim | Family | Test | Comparison | n | A-only | B-only | Δ (pp) | Variant | Statistic | p | Holm p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 1215.85 | 9.61e-265 | — | Valid success differs among the 3 paired conditions (Q = 1215.8, p = 9.61e-265); planned McNemar tests follow. |
| CICIDS2017 | mlp | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 57 | 0 | +1.78 | χ² (cc) | 55.02 | 1.19e-13 | 1.19e-13 | Prim-PGD p75 has higher Valid Targeted ASR than Prim-PGD p50 by 1.78 pp (57 vs 0 discordant flows; Holm-adjusted p = 1.19e-13). |
| CICIDS2017 | mlp | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 603 | 0 | +18.84 | χ² (cc) | 601.00 | 1.01e-132 | 2.03e-132 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 18.84 pp (603 vs 0 discordant flows; Holm-adjusted p = 2.03e-132). |
| CICIDS2017 | cnn | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 2992.92 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 2992.9, p = <1e-300); planned McNemar tests follow. |
| CICIDS2017 | cnn | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 130 | 0 | +4.06 | χ² (cc) | 128.01 | 1.12e-29 | 1.12e-29 | Prim-PGD p75 has higher Valid Targeted ASR than Prim-PGD p50 by 4.06 pp (130 vs 0 discordant flows; Holm-adjusted p = 1.12e-29). |
| CICIDS2017 | cnn | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 1486 | 0 | +46.44 | χ² (cc) | 1484.00 | <1e-300 | <1e-300 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 46.44 pp (1486 vs 0 discordant flows; Holm-adjusted p = <1e-300). |
| CICIDS2017 | ft_transformer | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 30.00 | 3.06e-07 | — | Valid success differs among the 3 paired conditions (Q = 30.0, p = 3.06e-07); planned McNemar tests follow. |
| CICIDS2017 | ft_transformer | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2017 | ft_transformer | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 15 | 0 | +0.47 | exact binomial |  | 6.1e-05 | 0.000122 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 0.47 pp (15 vs 0 discordant flows; Holm-adjusted p = 0.000122). |
| CICIDS2018 | mlp-s42 | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 1534.02 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1534.0, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | mlp-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 3 | 0 | +0.09 | exact binomial |  | 0.25 | 0.25 | No significant difference (Holm-adjusted p = 0.25; Δ = +0.09 pp, 3 vs 0 discordant flows). |
| CICIDS2018 | mlp-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 767 | 0 | +23.97 | χ² (cc) | 765.00 | 2.2e-168 | 4.39e-168 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 23.97 pp (767 vs 0 discordant flows; Holm-adjusted p = 4.39e-168). |
| CICIDS2018 | cnn-s42 | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 1670.00 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1670.0, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | cnn-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 835 | 0 | +26.09 | χ² (cc) | 833.00 | 3.61e-183 | 7.21e-183 | Prim-PGD unbounded has higher Valid Targeted ASR than Prim-PGD p75 by 26.09 pp (835 vs 0 discordant flows; Holm-adjusted p = 7.21e-183). |
| CICIDS2018 | ft_transformer-s42 | C: Prim-PGD 3 budgets | Cochran's Q | Prim-PGD p50 / Prim-PGD p75 / Prim-PGD unbounded | 3200 |  |  |  | Cochran χ² | 8.00 | 0.0183 | — | Valid success differs among the 3 paired conditions (Q = 8.0, p = 0.0183); planned McNemar tests follow. |
| CICIDS2018 | ft_transformer-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD p75 vs Prim-PGD p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | ft_transformer-s42 | C: Prim-PGD adjacent budgets (Holm over 2) | McNemar | Prim-PGD unbounded vs Prim-PGD p75 | 3200 | 4 | 0 | +0.12 | exact binomial |  | 0.125 | 0.25 | No significant difference (Holm-adjusted p = 0.25; Δ = +0.12 pp, 4 vs 0 discordant flows). |
| CICIDS2017 | mlp | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 1215.85 | 9.61e-265 | — | Valid success differs among the 3 paired conditions (Q = 1215.8, p = 9.61e-265); planned McNemar tests follow. |
| CICIDS2017 | mlp | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 57 | 0 | +1.78 | χ² (cc) | 55.02 | 1.19e-13 | 1.19e-13 | Hybrid Search p75 has higher Valid Targeted ASR than Hybrid Search p50 by 1.78 pp (57 vs 0 discordant flows; Holm-adjusted p = 1.19e-13). |
| CICIDS2017 | mlp | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 603 | 0 | +18.84 | χ² (cc) | 601.00 | 1.01e-132 | 2.03e-132 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 18.84 pp (603 vs 0 discordant flows; Holm-adjusted p = 2.03e-132). |
| CICIDS2017 | cnn | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 2992.92 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 2992.9, p = <1e-300); planned McNemar tests follow. |
| CICIDS2017 | cnn | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 130 | 0 | +4.06 | χ² (cc) | 128.01 | 1.12e-29 | 1.12e-29 | Hybrid Search p75 has higher Valid Targeted ASR than Hybrid Search p50 by 4.06 pp (130 vs 0 discordant flows; Holm-adjusted p = 1.12e-29). |
| CICIDS2017 | cnn | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 1486 | 0 | +46.44 | χ² (cc) | 1484.00 | <1e-300 | <1e-300 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 46.44 pp (1486 vs 0 discordant flows; Holm-adjusted p = <1e-300). |
| CICIDS2017 | ft_transformer | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 28.00 | 8.32e-07 | — | Valid success differs among the 3 paired conditions (Q = 28.0, p = 8.32e-07); planned McNemar tests follow. |
| CICIDS2017 | ft_transformer | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2017 | ft_transformer | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 14 | 0 | +0.44 | exact binomial |  | 0.000122 | 0.000244 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 0.44 pp (14 vs 0 discordant flows; Holm-adjusted p = 0.000244). |
| CICIDS2018 | mlp-s42 | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 1536.02 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1536.0, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | mlp-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 3 | 0 | +0.09 | exact binomial |  | 0.25 | 0.25 | No significant difference (Holm-adjusted p = 0.25; Δ = +0.09 pp, 3 vs 0 discordant flows). |
| CICIDS2018 | mlp-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 768 | 0 | +24.00 | χ² (cc) | 766.00 | 1.33e-168 | 2.66e-168 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 24.00 pp (768 vs 0 discordant flows; Holm-adjusted p = 2.66e-168). |
| CICIDS2018 | cnn-s42 | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 1670.00 | <1e-300 | — | Valid success differs among the 3 paired conditions (Q = 1670.0, p = <1e-300); planned McNemar tests follow. |
| CICIDS2018 | cnn-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 835 | 0 | +26.09 | χ² (cc) | 833.00 | 3.61e-183 | 7.21e-183 | Hybrid Search unbounded has higher Valid Targeted ASR than Hybrid Search p75 by 26.09 pp (835 vs 0 discordant flows; Holm-adjusted p = 7.21e-183). |
| CICIDS2018 | ft_transformer-s42 | C: Hybrid Search 3 budgets | Cochran's Q | Hybrid Search p50 / Hybrid Search p75 / Hybrid Search unbounded | 3200 |  |  |  | Cochran χ² | 8.00 | 0.0183 | — | Valid success differs among the 3 paired conditions (Q = 8.0, p = 0.0183); planned McNemar tests follow. |
| CICIDS2018 | ft_transformer-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search p75 vs Hybrid Search p50 | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | ft_transformer-s42 | C: Hybrid Search adjacent budgets (Holm over 2) | McNemar | Hybrid Search unbounded vs Hybrid Search p75 | 3200 | 4 | 0 | +0.12 | exact binomial |  | 0.125 | 0.25 | No significant difference (Holm-adjusted p = 0.25; Δ = +0.12 pp, 4 vs 0 discordant flows). |

## Class-wise results

| Dataset | Victim | Class | Condition | Raw | Valid |
|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | Prim-PGD p50 | 1.25% ± 0.00% | 1.25% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD p50 | 7.62% ± 0.00% | 7.62% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD p50 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD p50 | 9.50% ± 0.00% | 9.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD p50 | 26.62% ± 0.00% | 26.62% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD p50 | 0.63% ± 0.00% | 0.63% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD p50 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD p50 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD p50 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD p50 | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD p75 | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD p75 | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD p75 | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD p75 | 20.50% ± 0.00% | 20.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD p75 | 31.50% ± 0.00% | 31.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD p75 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD p75 | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD p75 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD p75 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD p75 | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD p75 | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD unbounded | 62.50% ± 0.00% | 62.50% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD unbounded | 26.50% ± 0.00% | 26.50% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD unbounded | 1.75% ± 0.00% | 1.75% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD unbounded | 91.63% ± 0.00% | 91.63% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD unbounded | 46.12% ± 0.00% | 46.12% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD unbounded | 100.00% ± 0.00% | 100.00% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD unbounded | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD unbounded | 0.75% ± 0.00% | 0.75% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD unbounded | 1.50% ± 0.00% | 1.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD unbounded | 2.04% ± 0.07% | 2.04% ± 0.07% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD unbounded | 6.38% ± 0.00% | 6.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD unbounded | 90.62% ± 0.00% | 90.62% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD unbounded | 0.25% ± 0.00% | 0.25% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD unbounded | 4.12% ± 0.00% | 4.12% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD unbounded | 100.00% ± 0.00% | 100.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD unbounded | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Hybrid Search p50 | 1.25% ± 0.00% | 1.25% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search p50 | 7.62% ± 0.00% | 7.62% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search p50 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search p50 | 9.50% ± 0.00% | 9.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Hybrid Search p50 | 26.62% ± 0.00% | 26.62% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search p50 | 0.63% ± 0.00% | 0.63% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search p50 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search p50 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search p50 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search p50 | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search p50 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Hybrid Search p75 | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search p75 | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search p75 | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search p75 | 20.50% ± 0.00% | 20.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Hybrid Search p75 | 31.50% ± 0.00% | 31.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search p75 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search p75 | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search p75 | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search p75 | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search p75 | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search p75 | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search p75 | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Hybrid Search unbounded | 62.50% ± 0.00% | 62.50% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search unbounded | 26.50% ± 0.00% | 26.50% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search unbounded | 1.75% ± 0.00% | 1.75% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search unbounded | 91.63% ± 0.00% | 91.63% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Hybrid Search unbounded | 46.12% ± 0.00% | 46.12% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search unbounded | 1.00% ± 0.00% | 1.00% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search unbounded | 100.00% ± 0.00% | 100.00% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search unbounded | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search unbounded | 0.75% ± 0.00% | 0.75% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search unbounded | 1.33% ± 0.07% | 1.33% ± 0.07% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search unbounded | 2.21% ± 0.07% | 2.21% ± 0.07% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search unbounded | 6.38% ± 0.00% | 6.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search unbounded | 90.62% ± 0.00% | 90.62% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search unbounded | 0.25% ± 0.00% | 0.25% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search unbounded | 4.12% ± 0.00% | 4.12% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search unbounded | 100.00% ± 0.00% | 100.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search unbounded | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search unbounded | 0.00% ± 0.00% | 0.00% ± 0.00% |

## Plots

- `plots/C1_raw_asr_vs_budget.png`
- `plots/C2_valid_asr_vs_budget.png`
- `plots/C3_validity_gap_vs_budget.png`

## Interpretation

**Budget is the dominant factor.** With capability-aware PrimAttack almost every attacked flow is
timing-only, so the budget is essentially the added-delay cap. Valid Targeted ASR (Prim-PGD;
Hybrid is identical except where noted) grows monotonically with the budget:

- CICIDS2017 MLP 2.31% (p50) → 4.09% (p75) → 22.94% (unbounded); CNN 9.19% → 13.25% → 59.69%;
  FT-Transformer 0.12% → 0.12% → 0.59% (Hybrid 0.55%).
- CICIDS2018 MLP 0.69% → 0.78% → 24.76% (Hybrid 24.80%); CNN 0.00% → 0.00% → 26.09%;
  FT-Transformer 0.00% → 0.00% → 0.12%.

**Inference (seed 42, Holm over the two adjacent comparisons, per optimizer).** Cochran's Q is
significant on every victim for both optimizers. p75 beats p50 only on the CICIDS2017 MLP and CNN
(57 and 130 flows gained, none lost; Holm p = 1.2e-13 and 1.1e-29); elsewhere the p50→p75 step
adds at most 3 flows (n.s.). Unbounded beats p75 on every victim (Holm p ≤ 2.4e-4) except
CICIDS2018 FT-Transformer (4 flows, Holm p = 0.25). No flow is ever lost by a larger budget
(B-only = 0 in every test), as expected from nested boxes.

**Why CICIDS2018 needs the unbounded box.** The calibrated p75 delay cap is small on CICIDS2018
(median per-flow cap about 31 ms vs about 0.9 s on CICIDS2017), and at p50/p75 the CICIDS2018 CNN
and FT-Transformer have no valid targeted success at all. Without the calibrated class budget
(envelope-only box) the same timing search reaches about a quarter of the MLP and CNN flows. The
p75 results therefore measure a deliberately conservative budget, not the limit of timing
manipulation.

**Validity at every budget.** The Validity Gap is 0.00 pp in all 36 cells: larger delays stay
inside the extractor identities and mined invariants, and no flow can be padded into an empty
packet. Valid success grows because the victims respond to larger delays, not because the
validator is relaxed. FT-Transformer remains the most robust victim at every budget (≤ 0.59%).
