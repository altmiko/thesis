# Final Experiment E — Paired validity-gap analysis

The research question: how much classifier-level attack success disappears when domain validity is required, and is that paired loss systematic? For every main condition, Raw Success and Valid Success are two binary outcomes of the **same** final adversarial example. Valid ⊆ Raw, so the only possible discordant cell is raw success = 1, valid success = 0 (fools the classifier, fails validator_v2). McNemar's test on that paired table is the one test where Raw Success is tested directly. There is no Cochran's Q. Each (condition, dataset, victim) test stands alone, organized by the experiment it belongs to, with no cross-thesis correction.

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

## Paired validity gap per condition

| Group | Dataset | Victim | Condition | n (seed 42) | Raw ASR | Valid ASR | Validity Gap | Raw-success-but-invalid (seed 42) | Valid successes (seed 42) | Gap seed 42 (pp) | Test | Statistic | McNemar p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A (untargeted) | CICIDS2017 | mlp | PrimAttack (Prim-PGD, p75) | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 0 | 131 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2017 | cnn | PrimAttack (Prim-PGD, p75) | 3200 | 13.47% ± 0.00% | 13.47% ± 0.00% | 0.00 ± 0.00 pp | 0 | 431 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2017 | ft_transformer | PrimAttack (Prim-PGD, p75) | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0 | 4 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2018 | mlp-s42 | PrimAttack (Prim-PGD, p75) | 3200 | 2.53% ± 0.00% | 2.53% ± 0.00% | 0.00 ± 0.00 pp | 0 | 81 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2018 | cnn-s42 | PrimAttack (Prim-PGD, p75) | 3200 | 1.16% ± 0.00% | 1.16% ± 0.00% | 0.00 ± 0.00 pp | 0 | 37 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | PrimAttack (Prim-PGD, p75) | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 0 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
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
| A (untargeted) | CICIDS2017 | mlp | CAPGD-PrimSupport | 3200 | 94.41% ± 0.71% | 2.01% ± 0.10% | 92.40 ± 0.79 pp | 2962 | 65 | 92.56 | asymptotic chi-square (continuity corrected) | 2960.0 | <1e-300 | 2962 of 3027 raw successes fail the validator (92.56 pp lost); the paired loss is systematic (p = <1e-300). |
| A (untargeted) | CICIDS2017 | cnn | CAPGD-PrimSupport | 3200 | 96.53% ± 1.07% | 5.15% ± 0.07% | 91.39 ± 1.00 pp | 2955 | 166 | 92.34 | asymptotic chi-square (continuity corrected) | 2953.0 | <1e-300 | 2955 of 3121 raw successes fail the validator (92.34 pp lost); the paired loss is systematic (p = <1e-300). |
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
| B (targeted→Benign) | CICIDS2017 | mlp | Hybrid Search (targeted, p75) | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 0 | 131 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | cnn | Hybrid Search (targeted, p75) | 3200 | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp | 0 | 424 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | ft_transformer | Hybrid Search (targeted, p75) | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0 | 4 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Hybrid Search (targeted, p75) | 3200 | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 ± 0.00 pp | 0 | 25 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Hybrid Search (targeted, p75) | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 0 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Hybrid Search (targeted, p75) | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 0 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | mlp | Prim-PGD (targeted, p75) | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 0 | 131 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | cnn | Prim-PGD (targeted, p75) | 3200 | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp | 0 | 424 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | ft_transformer | Prim-PGD (targeted, p75) | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0 | 4 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Prim-PGD (targeted, p75) | 3200 | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 ± 0.00 pp | 0 | 25 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Prim-PGD (targeted, p75) | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 0 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Prim-PGD (targeted, p75) | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 0 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | mlp | Prim-C&W (targeted, p75) | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 0 | 131 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | cnn | Prim-C&W (targeted, p75) | 3200 | 4.00% ± 0.00% | 4.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 128 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2017 | ft_transformer | Prim-C&W (targeted, p75) | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0 | 4 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | mlp-s42 | Prim-C&W (targeted, p75) | 3200 | 0.75% ± 0.00% | 0.75% ± 0.00% | 0.00 ± 0.00 pp | 0 | 24 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | cnn-s42 | Prim-C&W (targeted, p75) | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 0 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |
| B (targeted→Benign) | CICIDS2018 | ft_transformer-s42 | Prim-C&W (targeted, p75) | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0 | 0 | 0.00 | exact binomial McNemar |  | 1 | No raw-success-but-invalid example: no validity gap at seed 42. |

## Why raw successes are rejected (descriptive)

For every raw-success-but-invalid example at seed 42: the share that violates at least one rule of each validator_v2 category. An example can fail several categories. SCHEMA, EXTRACTOR and PROTOCOL are general flow-consistency rules. MINED rules are train-mined, dataset-specific invariants. Also in `rejection_categories_of_invalid_successes.csv`.

| Group | Dataset | Victim | Condition | Raw-success-but-invalid (seed 42) | Failing SCHEMA | Failing EXTRACTOR | Failing PROTOCOL | Failing MINED |
|---|---|---|---|---|---|---|---|---|
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
| A (untargeted) | CICIDS2017 | mlp | CAPGD-PrimSupport | 2962 | 0.0% | 67.3% | 16.3% | 92.7% |
| A (untargeted) | CICIDS2017 | cnn | CAPGD-PrimSupport | 2955 | 0.0% | 84.6% | 11.4% | 86.2% |
| A (untargeted) | CICIDS2017 | ft_transformer | CAPGD-PrimSupport | 1653 | 0.0% | 83.2% | 16.6% | 96.2% |
| A (untargeted) | CICIDS2018 | mlp-s42 | CAPGD-PrimSupport | 2957 | 0.0% | 82.7% | 15.5% | 92.5% |
| A (untargeted) | CICIDS2018 | cnn-s42 | CAPGD-PrimSupport | 2448 | 0.0% | 95.8% | 22.2% | 92.3% |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | CAPGD-PrimSupport | 335 | 0.0% | 100.0% | 61.8% | 99.1% |
| A (untargeted) | CICIDS2017 | mlp | C-PGD-PrimSupport | 1645 | 0.0% | 100.0% | 53.3% | 89.5% |
| A (untargeted) | CICIDS2017 | cnn | C-PGD-PrimSupport | 2032 | 0.0% | 100.0% | 54.9% | 84.4% |
| A (untargeted) | CICIDS2017 | ft_transformer | C-PGD-PrimSupport | 686 | 0.0% | 100.0% | 51.6% | 95.2% |
| A (untargeted) | CICIDS2018 | mlp-s42 | C-PGD-PrimSupport | 885 | 0.0% | 100.0% | 47.9% | 95.7% |
| A (untargeted) | CICIDS2018 | cnn-s42 | C-PGD-PrimSupport | 1531 | 0.0% | 100.0% | 57.3% | 95.2% |
| A (untargeted) | CICIDS2018 | ft_transformer-s42 | C-PGD-PrimSupport | 39 | 0.0% | 100.0% | 82.1% | 100.0% |

## Plots

- `plots/E1_raw_vs_valid_asr.png` — Raw ASR vs Valid ASR (points below the diagonal = gap)
- `plots/E2_validity_gap_by_method.png` — Validity gap by attack / method

## Interpretation

**Size of the gap.** On identical source flows, the share of raw successes that validator_v2
rejects depends almost entirely on the attack's threat model:

- PGD and C&W: every raw success is invalid on every victim (Validity Gap = Raw ASR,
  53.59–100.00 pp; McNemar p < 1e-300 on all 12 cells).
- CAPGD-PrimSupport: gaps of 9.74–92.40 pp; at seed 42 between 5 (CICIDS2017 FT-Transformer)
  and 166 (CICIDS2017 CNN) of up to 3,121 raw successes survive.
- C-PGD-PrimSupport: gaps of 1.65–60.42 pp; no raw success survives on any victim.
- Capability-aware PrimAttack (Exp A untargeted and all three Exp B optimizers, targeted): gap
  0.00 pp in all 24 cells; every raw success is valid (McNemar p = 1, no discordant flow).

**Optimizer duplication is deliberate, not independent replication.** In Exp B, Hybrid and
Prim-PGD have identical targeted success masks: 1,752 valid successes each and no discordant
flow. Nearly all rows are timing-only, so Hybrid skips exact padding enumeration and both reduce
to closely related sign-momentum timing searches. Their two zero-gap rows are retained because
the optimizer comparison was pre-registered; they must not be treated as two independent pieces
of evidence for validator performance. Prim-C&W follows the same validity gate but finds fewer
CICIDS2017 CNN successes because its cost-penalized trajectory differs.

**Why raw successes are rejected.** PGD and C&W examples fail every validator category (SCHEMA,
EXTRACTOR and PROTOCOL for 100% of them; MINED for ≥ 98.5%). The matched-support attacks never
fail SCHEMA (their type repair works) but mostly break CICFlowMeter identities (EXTRACTOR:
67–100% of CAPGD, 100% of C-PGD invalid examples) and mined invariants (MINED: 84–100%).
PROTOCOL failures (11–82%) all come from the empty-forward-packet rule `PROTO_0080` (0% before
amendment A2 on the identical adversarial flows): both attacks raise `Fwd Packet Length Min` of
flows that had a zero-length forward packet (CAPGD-PrimSupport in about 454 of 3,021 raw
successes per seed on CICIDS2017 MLP). The rule changes almost no verdict on its own: those flows
usually also violate EXTRACTOR or MINED rules, and only 16 / 6 CAPGD-PrimSupport valid successes
(CICIDS2017 MLP / CNN, three seeds) were lost to it alone.

**Why PrimAttack has no gap.** It constructs every candidate through the canonical recomputation
φ (identities hold by construction), keeps only validator-accepted incumbents in its search, and
under the capability rule never pads a flow with an empty forward packet. Before amendment A2 the
relaxed PrimAttack had a 0.33–15.28 pp gap on CICIDS2018, entirely from `MINED_0001` rejecting
padded flows; that source of invalidity is now excluded before optimization.

**Reading (Contribution 3).** A high Raw ASR says little about constrained evasion: the attacks
with the highest raw success produce no valid flow, and matched feature support alone does not
keep flows consistent. The validity gap is a property of the attack parameterization, measured on
the same flows, not an artefact of different samples.
