# Final Experiment E — Paired validity-gap analysis

The research question: how much classifier-level attack success disappears when domain validity is required, and is that paired loss systematic? For every main condition, Raw Success and Valid Success are two binary outcomes of the **same** final adversarial example. Valid ⊆ Raw, so the only possible discordant cell is raw success = 1, valid success = 0 (fools the classifier, fails validator_v2). McNemar's test on that paired table is the one test where Raw Success is tested directly. There is no Cochran's Q. Each (condition, dataset, victim) test stands alone, organized by the experiment it belongs to, with no cross-thesis correction.

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

## Paired validity gap per condition

| Group | Dataset | Victim | Condition | n (seed 42) | Raw ASR | Valid ASR | Validity Gap | Raw-success-but-invalid (seed 42) | Valid successes (seed 42) | Gap seed 42 (pp) | Test | Statistic | McNemar p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (untargeted) | CICIDS2017 | mlp | PrimAttack (Hybrid Search, p75) | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 0 | 354 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2017 | cnn | PrimAttack (Hybrid Search, p75) | 3200 | 36.67% ± 0.04% | 36.67% ± 0.04% | 0.00 ± 0.00 pp | 0 | 1174 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2017 | ft_transformer | PrimAttack (Hybrid Search, p75) | 3200 | 0.50% ± 0.00% | 0.50% ± 0.00% | 0.00 ± 0.00 pp | 0 | 16 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2018 | mlp-s42 | PrimAttack (Hybrid Search, p75) | 3200 | 12.00% ± 0.03% | 0.19% ± 0.03% | 11.81 ± 0.03 pp | 379 | 6 | 11.84 | asymptotic chi-square (continuity corrected) | 377.0 | 5.59e-84 | 379 of 385 raw successes fail the validator (11.84 pp lost); the paired loss is systematic (p = 5.59e-84). |
| A (untargeted) | CICIDS2018 | cnn-s42 | PrimAttack (Hybrid Search, p75) | 3200 | 15.31% ± 0.00% | 0.03% ± 0.00% | 15.28 ± 0.00 pp | 489 | 1 | 15.28 | asymptotic chi-square (continuity corrected) | 487.0 | 6.4e-108 | 489 of 490 raw successes fail the validator (15.28 pp lost); the paired loss is systematic (p = 6.4e-108). |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | PrimAttack (Hybrid Search, p75) | 3200 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 11 | 0 | 0.34 | exact binomial McNemar |  | 0.000977 | 11 of 11 raw successes fail the validator (0.34 pp lost); the paired loss is systematic (p = 0.000977). |
| A (untargeted) | CICIDS2017 | mlp | PGD | 3200 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp | 3200 | 0 | 100.00 | asymptotic chi-square (continuity corrected) | 3198.0 | <1e-300 | 3200 of 3200 raw successes fail the validator (100.00 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | cnn | PGD | 3200 | 96.12% ± 0.09% | 0.00% ± 0.00% | 96.12 ± 0.09 pp | 3079 | 0 | 96.22 | asymptotic chi-square (continuity corrected) | 3077.0 | <1e-300 | 3079 of 3079 raw successes fail the validator (96.22 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | ft_transformer | PGD | 3200 | 97.36% ± 0.28% | 0.00% ± 0.00% | 97.36 ± 0.28 pp | 3115 | 0 | 97.34 | asymptotic chi-square (continuity corrected) | 3113.0 | <1e-300 | 3115 of 3115 raw successes fail the validator (97.34 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | mlp-s42 | PGD | 3200 | 94.34% ± 0.25% | 0.00% ± 0.00% | 94.34 ± 0.25 pp | 3019 | 0 | 94.34 | asymptotic chi-square (continuity corrected) | 3017.0 | <1e-300 | 3019 of 3019 raw successes fail the validator (94.34 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | cnn-s42 | PGD | 3200 | 99.70% ± 0.02% | 0.00% ± 0.00% | 99.70 ± 0.02 pp | 3191 | 0 | 99.72 | asymptotic chi-square (continuity corrected) | 3189.0 | <1e-300 | 3191 of 3191 raw successes fail the validator (99.72 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | PGD | 3200 | 91.70% ± 0.18% | 0.00% ± 0.00% | 91.70 ± 0.18 pp | 2941 | 0 | 91.91 | asymptotic chi-square (continuity corrected) | 2939.0 | <1e-300 | 2941 of 2941 raw successes fail the validator (91.91 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | mlp | C&W | 3200 | 99.94% ± 0.00% | 0.00% ± 0.00% | 99.94 ± 0.00 pp | 3198 | 0 | 99.94 | asymptotic chi-square (continuity corrected) | 3196.0 | <1e-300 | 3198 of 3198 raw successes fail the validator (99.94 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | cnn | C&W | 3200 | 95.53% ± 0.00% | 0.00% ± 0.00% | 95.53 ± 0.00 pp | 3057 | 0 | 95.53 | asymptotic chi-square (continuity corrected) | 3055.0 | <1e-300 | 3057 of 3057 raw successes fail the validator (95.53 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | ft_transformer | C&W | 3200 | 77.16% ± 0.00% | 0.00% ± 0.00% | 77.16 ± 0.00 pp | 2469 | 0 | 77.16 | asymptotic chi-square (continuity corrected) | 2467.0 | <1e-300 | 2469 of 2469 raw successes fail the validator (77.16 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | mlp-s42 | C&W | 3200 | 87.63% ± 0.00% | 0.00% ± 0.00% | 87.62 ± 0.00 pp | 2804 | 0 | 87.62 | asymptotic chi-square (continuity corrected) | 2802.0 | <1e-300 | 2804 of 2804 raw successes fail the validator (87.62 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | cnn-s42 | C&W | 3200 | 99.16% ± 0.00% | 0.00% ± 0.00% | 99.16 ± 0.00 pp | 3173 | 0 | 99.16 | asymptotic chi-square (continuity corrected) | 3171.0 | <1e-300 | 3173 of 3173 raw successes fail the validator (99.16 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | C&W | 3200 | 53.59% ± 0.00% | 0.00% ± 0.00% | 53.59 ± 0.00 pp | 1715 | 0 | 53.59 | asymptotic chi-square (continuity corrected) | 1713.0 | <1e-300 | 1715 of 1715 raw successes fail the validator (53.59 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | mlp | CAPGD-PrimSupport | 3200 | 94.41% ± 0.71% | 2.18% ± 0.08% | 92.23 ± 0.78 pp | 2957 | 70 | 92.41 | asymptotic chi-square (continuity corrected) | 2955.0 | <1e-300 | 2957 of 3027 raw successes fail the validator (92.41 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | cnn | CAPGD-PrimSupport | 3200 | 96.53% ± 1.07% | 5.21% ± 0.07% | 91.32 ± 1.00 pp | 2953 | 168 | 92.28 | asymptotic chi-square (continuity corrected) | 2951.0 | <1e-300 | 2953 of 3121 raw successes fail the validator (92.28 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | ft_transformer | CAPGD-PrimSupport | 3200 | 52.21% ± 4.33% | 0.18% ± 0.02% | 52.03 ± 4.32 pp | 1653 | 5 | 51.66 | asymptotic chi-square (continuity corrected) | 1651.0 | <1e-300 | 1653 of 1658 raw successes fail the validator (51.66 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | mlp-s42 | CAPGD-PrimSupport | 3200 | 91.57% ± 0.84% | 0.14% ± 0.02% | 91.44 ± 0.85 pp | 2957 | 4 | 92.41 | asymptotic chi-square (continuity corrected) | 2955.0 | <1e-300 | 2957 of 2961 raw successes fail the validator (92.41 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | cnn-s42 | CAPGD-PrimSupport | 3200 | 76.01% ± 1.68% | 0.29% ± 0.07% | 75.72 ± 1.69 pp | 2448 | 7 | 76.50 | asymptotic chi-square (continuity corrected) | 2446.0 | <1e-300 | 2448 of 2455 raw successes fail the validator (76.50 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | CAPGD-PrimSupport | 3200 | 9.74% ± 0.70% | 0.00% ± 0.00% | 9.74 ± 0.70 pp | 335 | 0 | 10.47 | asymptotic chi-square (continuity corrected) | 333.0 | 2.13e-74 | 335 of 335 raw successes fail the validator (10.47 pp lost); the paired loss is systematic (p = 2.13e-74). |
| A (untargeted) | CICIDS2017 | mlp | C-PGD-PrimSupport | 3200 | 50.80% ± 2.04% | 0.00% ± 0.00% | 50.80 ± 2.04 pp | 1645 | 0 | 51.41 | asymptotic chi-square (continuity corrected) | 1643.0 | <1e-300 | 1645 of 1645 raw successes fail the validator (51.41 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | cnn | C-PGD-PrimSupport | 3200 | 60.42% ± 2.98% | 0.00% ± 0.00% | 60.42 ± 2.98 pp | 2032 | 0 | 63.50 | asymptotic chi-square (continuity corrected) | 2030.0 | <1e-300 | 2032 of 2032 raw successes fail the validator (63.50 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | ft_transformer | C-PGD-PrimSupport | 3200 | 21.61% ± 0.31% | 0.00% ± 0.00% | 21.61 ± 0.31 pp | 686 | 0 | 21.44 | asymptotic chi-square (continuity corrected) | 684.0 | 9.01e-151 | 686 of 686 raw successes fail the validator (21.44 pp lost); the paired loss is systematic (p = 9.01e-151). |
| A (untargeted) | CICIDS2018 | mlp-s42 | C-PGD-PrimSupport | 3200 | 28.25% ± 0.51% | 0.00% ± 0.00% | 28.25 ± 0.51 pp | 885 | 0 | 27.66 | asymptotic chi-square (continuity corrected) | 883.0 | 4.87e-194 | 885 of 885 raw successes fail the validator (27.66 pp lost); the paired loss is systematic (p = 4.87e-194). |
| A (untargeted) | CICIDS2018 | cnn-s42 | C-PGD-PrimSupport | 3200 | 50.54% ± 3.30% | 0.00% ± 0.00% | 50.54 ± 3.30 pp | 1531 | 0 | 47.84 | asymptotic chi-square (continuity corrected) | 1529.0 | <1e-300 | 1531 of 1531 raw successes fail the validator (47.84 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | C-PGD-PrimSupport | 3200 | 1.65% ± 0.42% | 0.00% ± 0.00% | 1.65 ± 0.42 pp | 39 | 0 | 1.22 | asymptotic chi-square (continuity corrected) | 37.0 | 1.17e-09 | 39 of 39 raw successes fail the validator (1.22 pp lost); the paired loss is systematic (p = 1.17e-09). |
| B (targeted→Benign) | CICIDS2017 | mlp | Hybrid Search (targeted, p75) | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 0 | 354 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | cnn | Hybrid Search (targeted, p75) | 3200 | 36.15% ± 0.02% | 36.15% ± 0.02% | 0.00 ± 0.00 pp | 0 | 1157 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | ft_transformer | Hybrid Search (targeted, p75) | 3200 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0 | 14 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Hybrid Search (targeted, p75) | 3200 | 1.28% ± 0.03% | 0.29% ± 0.04% | 0.99 ± 0.02 pp | 32 | 10 | 1.00 | asymptotic chi-square (continuity corrected) | 30.0 | 4.25e-08 | 32 of 42 raw successes fail the validator (1.00 pp lost); the paired loss is systematic (p = 4.25e-08). |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Hybrid Search (targeted, p75) | 3200 | 15.00% ± 0.00% | 0.00% ± 0.00% | 15.00 ± 0.00 pp | 480 | 0 | 15.00 | asymptotic chi-square (continuity corrected) | 478.0 | 5.81e-106 | 480 of 480 raw successes fail the validator (15.00 pp lost); the paired loss is systematic (p = 5.81e-106). |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Hybrid Search (targeted, p75) | 3200 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 11 | 0 | 0.34 | exact binomial McNemar |  | 0.000977 | 11 of 11 raw successes fail the validator (0.34 pp lost); the paired loss is systematic (p = 0.000977). |
| B (targeted→Benign) | CICIDS2017 | mlp | Prim-PGD (targeted, p75) | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 0 | 354 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | cnn | Prim-PGD (targeted, p75) | 3200 | 36.10% ± 0.13% | 36.10% ± 0.13% | 0.00 ± 0.00 pp | 0 | 1153 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | ft_transformer | Prim-PGD (targeted, p75) | 3200 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0 | 14 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Prim-PGD (targeted, p75) | 3200 | 1.31% ± 0.00% | 0.26% ± 0.02% | 1.05 ± 0.02 pp | 33 | 9 | 1.03 | asymptotic chi-square (continuity corrected) | 31.0 | 2.54e-08 | 33 of 42 raw successes fail the validator (1.03 pp lost); the paired loss is systematic (p = 2.54e-08). |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Prim-PGD (targeted, p75) | 3200 | 15.01% ± 0.02% | 0.00% ± 0.00% | 15.01 ± 0.02 pp | 480 | 0 | 15.00 | asymptotic chi-square (continuity corrected) | 478.0 | 5.81e-106 | 480 of 480 raw successes fail the validator (15.00 pp lost); the paired loss is systematic (p = 5.81e-106). |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Prim-PGD (targeted, p75) | 3200 | 0.32% ± 0.02% | 0.00% ± 0.00% | 0.32 ± 0.02 pp | 10 | 0 | 0.31 | exact binomial McNemar |  | 0.00195 | 10 of 10 raw successes fail the validator (0.31 pp lost); the paired loss is systematic (p = 0.00195). |
| B (targeted→Benign) | CICIDS2017 | mlp | Prim-C&W (targeted, p75) | 3200 | 11.00% ± 0.00% | 11.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 352 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | cnn | Prim-C&W (targeted, p75) | 3200 | 14.31% ± 0.00% | 14.31% ± 0.00% | 0.00 ± 0.00 pp | 0 | 458 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | ft_transformer | Prim-C&W (targeted, p75) | 3200 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0 | 14 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Prim-C&W (targeted, p75) | 3200 | 1.28% ± 0.00% | 0.72% ± 0.00% | 0.56 ± 0.00 pp | 18 | 23 | 0.56 | exact binomial McNemar |  | 7.63e-06 | 18 of 41 raw successes fail the validator (0.56 pp lost); the paired loss is systematic (p = 7.63e-06). |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Prim-C&W (targeted, p75) | 3200 | 13.81% ± 0.00% | 0.00% ± 0.00% | 13.81 ± 0.00 pp | 442 | 0 | 13.81 | asymptotic chi-square (continuity corrected) | 440.0 | 1.08e-97 | 442 of 442 raw successes fail the validator (13.81 pp lost); the paired loss is systematic (p = 1.08e-97). |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Prim-C&W (targeted, p75) | 3200 | 0.19% ± 0.00% | 0.00% ± 0.00% | 0.19 ± 0.00 pp | 6 | 0 | 0.19 | exact binomial McNemar |  | 0.0312 | 6 of 6 raw successes fail the validator (0.19 pp lost); the paired loss is systematic (p = 0.0312). |

## Why raw successes are rejected (descriptive)

For every raw-success-but-invalid example at seed 42: the share that violates at least one rule of each validator_v2 category. An example can fail several categories. SCHEMA, EXTRACTOR and PROTOCOL are general flow-consistency rules. MINED rules are train-mined, dataset-specific invariants. Also in `rejection_categories_of_invalid_successes.csv`.

| Group | Dataset | Victim | Condition | Raw-success-but-invalid (seed 42) | Failing SCHEMA | Failing EXTRACTOR | Failing PROTOCOL | Failing MINED |
|---|---|---|---|---|---|---|---|---|
| A (untargeted) | CICIDS2018 | mlp-s42 | PrimAttack (Hybrid Search, p75) | 379 | 0.0% | 0.0% | 0.0% | 100.0% |
| A (untargeted) | CICIDS2018 | cnn-s42 | PrimAttack (Hybrid Search, p75) | 489 | 0.0% | 0.0% | 0.0% | 100.0% |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | PrimAttack (Hybrid Search, p75) | 11 | 0.0% | 0.0% | 0.0% | 100.0% |
| A (untargeted) | CICIDS2017 | mlp | PGD | 3200 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2017 | cnn | PGD | 3079 | 100.0% | 100.0% | 100.0% | 99.9% |
| A (untargeted) | CICIDS2017 | ft_transformer | PGD | 3115 | 100.0% | 100.0% | 100.0% | 99.7% |
| A (untargeted) | CICIDS2018 | mlp-s42 | PGD | 3019 | 100.0% | 100.0% | 100.0% | 98.5% |
| A (untargeted) | CICIDS2018 | cnn-s42 | PGD | 3191 | 100.0% | 100.0% | 100.0% | 99.9% |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | PGD | 2941 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2017 | mlp | C&W | 3198 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2017 | cnn | C&W | 3057 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2017 | ft_transformer | C&W | 2469 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2018 | mlp-s42 | C&W | 2804 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2018 | cnn-s42 | C&W | 3173 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | C&W | 1715 | 100.0% | 100.0% | 100.0% | 100.0% |
| A (untargeted) | CICIDS2017 | mlp | CAPGD-PrimSupport | 2957 | 0.0% | 67.4% | 0.0% | 92.9% |
| A (untargeted) | CICIDS2017 | cnn | CAPGD-PrimSupport | 2953 | 0.0% | 84.7% | 0.0% | 86.3% |
| A (untargeted) | CICIDS2017 | ft_transformer | CAPGD-PrimSupport | 1653 | 0.0% | 83.2% | 0.0% | 96.2% |
| A (untargeted) | CICIDS2018 | mlp-s42 | CAPGD-PrimSupport | 2957 | 0.0% | 82.7% | 0.0% | 92.5% |
| A (untargeted) | CICIDS2018 | cnn-s42 | CAPGD-PrimSupport | 2448 | 0.0% | 95.8% | 0.0% | 92.3% |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | CAPGD-PrimSupport | 335 | 0.0% | 100.0% | 0.0% | 99.1% |
| A (untargeted) | CICIDS2017 | mlp | C-PGD-PrimSupport | 1645 | 0.0% | 100.0% | 0.0% | 89.5% |
| A (untargeted) | CICIDS2017 | cnn | C-PGD-PrimSupport | 2032 | 0.0% | 100.0% | 0.0% | 84.4% |
| A (untargeted) | CICIDS2017 | ft_transformer | C-PGD-PrimSupport | 686 | 0.0% | 100.0% | 0.0% | 95.2% |
| A (untargeted) | CICIDS2018 | mlp-s42 | C-PGD-PrimSupport | 885 | 0.0% | 100.0% | 0.0% | 95.7% |
| A (untargeted) | CICIDS2018 | cnn-s42 | C-PGD-PrimSupport | 1531 | 0.0% | 100.0% | 0.0% | 95.2% |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | C-PGD-PrimSupport | 39 | 0.0% | 100.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Hybrid Search (targeted, p75) | 32 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Hybrid Search (targeted, p75) | 480 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Hybrid Search (targeted, p75) | 11 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Prim-PGD (targeted, p75) | 33 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Prim-PGD (targeted, p75) | 480 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Prim-PGD (targeted, p75) | 10 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Prim-C&W (targeted, p75) | 18 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Prim-C&W (targeted, p75) | 442 | 0.0% | 0.0% | 0.0% | 100.0% |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Prim-C&W (targeted, p75) | 6 | 0.0% | 0.0% | 0.0% | 100.0% |

## Plots

- `plots/E1_raw_vs_valid_asr.png` — Raw ASR vs Valid ASR (points below the diagonal = gap)
- `plots/E2_validity_gap_by_method.png` — Validity gap by attack / method

## Interpretation

**How much classifier-level success disappears when validity is required.** On identical
adversarial examples (seed-42 McNemar, raw vs valid):

- **Unconstrained feature-space attacks (PGD, C&W):** all of it. 1,715–3,200 raw successes per
  victim, 0 valid, gap 53.59–100.00 pp. Every test is significant (p < 1e-300).
- **Matched-support constrained attacks:** almost all of it. CAPGD-PrimSupport keeps 0–168 valid
  of 335–3,121 raw successes (gap 9.74–92.23 pp). C-PGD-PrimSupport keeps none of 39–2,032
  (gap 1.65–60.42 pp). Every test is significant (p ≤ 1.2e-9). The invalid examples fail
  EXTRACTOR identities (67–100%) and MINED invariants (84–100%). They never fail SCHEMA or
  PROTOCOL: the attacks' train-range box and type repair keep single features in-domain, but not
  their mutual consistency.
- **PrimAttack (untargeted and all three targeted optimizers, p75):** none of it on CICIDS2017.
  There is no raw-success-but-invalid example on any victim (p = 1, gap 0.00 pp). On
  CICIDS2018 a systematic gap appears. Untargeted: 379 of 385 (MLP), 489 of 490 (CNN) and 11 of 11 (FT)
  raw successes are invalid (gap 11.81 / 15.28 / 0.33 pp; p ≤ 0.001). Targeted: gaps of
  0.56–1.05 pp (MLP), 13.81–15.01 pp (CNN) and 0.19–0.33 pp (FT) across the three optimizers.
  Every such example violates only the MINED category, specifically `MINED_0001`
  (`Fwd Packet Length Min ≈ Packet Length Min`), and uses padding.

**Is the loss systematic?** Yes, wherever it exists. Because Valid ⊆ Raw, the discordant cell
"raw success, invalid" is one-sided by construction, and every non-zero count is significant.
The seed replicates show the same picture: the Validity Gap SD across seeds is ≤ 0.03 pp for
every PrimAttack condition and ≤ 4.32 pp for every baseline. What matters is the size of the
gap, not the p-value. The feature-space baselines lose essentially their whole Raw ASR, and
PrimAttack loses nothing on CICIDS2017 and 0.2–15.3 pp on CICIDS2018.

**What the gap measures.** Raw ASR alone would rank PGD (≥ 91.70%) far above PrimAttack
(≤ 36.67% at p75). Valid ASR reverses that ranking on CICIDS2017 and flattens it to near zero on
CICIDS2018. The paired validity gap separates two failure modes: classifier robustness (low
raw success) and domain invalidity (high raw, low valid). It also shows where the independent
validator adds information that the attack's own constraints do not encode. For C-PGD, the
differentiable relation penalty did not prevent EXTRACTOR failures. For PrimAttack on
CICIDS2018, the train-mined regularity `Fwd Packet Length Min ≈ Packet Length Min` is not one of
the constraints PrimAttack's recomputation φ preserves. Padding raises the forward minimum, and
the validator rejects the resulting flow (Contribution 3).
