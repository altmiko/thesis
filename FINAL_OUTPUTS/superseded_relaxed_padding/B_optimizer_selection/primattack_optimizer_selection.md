# Final Experiment B — PrimAttack optimizer selection (targeted → Benign, p75)

Three optimizers over the identical PrimAttack attack space (`RealizedSearch`: the same primitive parameterization, joint mode, per-class p75 box, recomputation φ, integer rounding, victim, validator_v2 gate and success predicate; every reported candidate is a realized, quantized, recomputed flow; per-flow incumbent: success > failure, cheapest success, best-margin failure). Matched per-flow cap of 256 victim evaluations. Success = prediction == Benign. Valid success additionally requires validator_v2 `hybrid_valid`. Mean ± SD over seeds 42/2024/2026.

| Optimizer | Hyperparameters (frozen; Prim-PGD/C&W tuned on the CICIDS2017 validation split) |
|---|---|
| Hybrid Search | exact integer padding enumeration, then adaptive projected sign-momentum refinement (40 steps/restart, lr 0.1, momentum 0.75, stall halving), restarts until the 256-evaluation budget |
| Prim-PGD | 3 restarts (clean + 2 uniform) × 42 steps, α = 0.05, momentum 0.75 |
| Prim-C&W | 3 binary-search stages × 42 Adam steps, lr 0.5, c₀ = 1, κ = 0 |

## Optimizer selection (pre-registered criterion)

Rule: highest aggregate Valid Targeted ASR at p75 pooled over both datasets, all victims, classes and seeds (sum successes / sum attempts); ties -> fewer mean victim evaluations per flow -> order hybrid, pgd, cw. **Selected: Hybrid Search.** Budget-sensitivity optimizers (top 2): Hybrid Search, Prim-PGD. The selection is based on the aggregate Valid Targeted ASR, not on p-values. The tests below are supporting evidence.

| Rank | Optimizer | Valid targeted successes | Attempts | Aggregate Valid Targeted ASR | Mean evals / flow |
|---|---|---|---|---|---|
| 1 | Hybrid Search | 4602 | 57600 | 7.990% | 190.6 |
| 2 | Prim-PGD | 4595 | 57600 | 7.977% | 194.0 |
| 3 | Prim-C&W | 2541 | 57600 | 4.411% | 194.0 |

## Results per dataset and victim

| Dataset | Victim | Optimizer | n/seed | Raw Targeted ASR | Valid Targeted ASR | Validity Gap | Valid per seed (42/2024/2026, %) | Evals / flow | ms / flow | Median primitive cost (valid) |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | Hybrid Search | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 11.06 / 11.06 / 11.06 | 188.31 ± 0.00 | 2.84 ± 0.03 | 1.400 |
| CICIDS2017 | cnn | Hybrid Search | 3200 | 36.15% ± 0.02% | 36.15% ± 0.02% | 0.00 ± 0.00 pp | 36.16 / 36.12 / 36.16 | 188.37 ± 0.00 | 2.96 ± 0.11 | 1.608 |
| CICIDS2017 | ft_transformer | Hybrid Search | 3200 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0.44 / 0.44 / 0.44 | 188.21 ± 0.00 | 12.22 ± 0.12 | 0.196 |
| CICIDS2018 | mlp-s42 | Hybrid Search | 3200 | 1.28% ± 0.03% | 0.29% ± 0.04% | 0.99 ± 0.02 pp | 0.31 / 0.31 / 0.25 | 192.85 ± 0.00 | 3.07 ± 0.02 | 1.000 |
| CICIDS2018 | cnn-s42 | Hybrid Search | 3200 | 15.00% ± 0.00% | 0.00% ± 0.00% | 15.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 192.79 ± 0.00 | 3.00 ± 0.09 | — |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search | 3200 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 0.00 / 0.00 / 0.00 | 192.85 ± 0.00 | 12.75 ± 0.09 | — |
| CICIDS2017 | mlp | Prim-PGD | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 11.06 / 11.06 / 11.06 | 190.63 ± 0.00 | 3.02 ± 0.03 | 1.330 |
| CICIDS2017 | cnn | Prim-PGD | 3200 | 36.10% ± 0.13% | 36.10% ± 0.13% | 0.00 ± 0.00 pp | 36.03 / 36.25 / 36.03 | 190.55 ± 0.00 | 3.14 ± 0.11 | 1.576 |
| CICIDS2017 | ft_transformer | Prim-PGD | 3200 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0.44 / 0.44 / 0.44 | 190.63 ± 0.00 | 13.71 ± 0.06 | 0.205 |
| CICIDS2018 | mlp-s42 | Prim-PGD | 3200 | 1.31% ± 0.00% | 0.26% ± 0.02% | 1.05 ± 0.02 pp | 0.28 / 0.25 / 0.25 | 197.40 ± 0.00 | 3.01 ± 0.03 | 0.927 |
| CICIDS2018 | cnn-s42 | Prim-PGD | 3200 | 15.01% ± 0.02% | 0.00% ± 0.00% | 15.01 ± 0.02 pp | 0.00 / 0.00 / 0.00 | 197.40 ± 0.00 | 2.97 ± 0.04 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD | 3200 | 0.32% ± 0.02% | 0.00% ± 0.00% | 0.32 ± 0.02 pp | 0.00 / 0.00 / 0.00 | 197.40 ± 0.00 | 13.86 ± 0.02 | — |
| CICIDS2017 | mlp | Prim-C&W | 3200 | 11.00% ± 0.00% | 11.00% ± 0.00% | 0.00 ± 0.00 pp | 11.00 / 11.00 / 11.00 | 190.63 ± 0.00 | 3.21 ± 0.04 | 1.056 |
| CICIDS2017 | cnn | Prim-C&W | 3200 | 14.31% ± 0.00% | 14.31% ± 0.00% | 0.00 ± 0.00 pp | 14.31 / 14.31 / 14.31 | 190.55 ± 0.00 | 3.36 ± 0.12 | 1.482 |
| CICIDS2017 | ft_transformer | Prim-C&W | 3200 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0.44 / 0.44 / 0.44 | 190.63 ± 0.00 | 13.86 ± 0.11 | 0.199 |
| CICIDS2018 | mlp-s42 | Prim-C&W | 3200 | 1.28% ± 0.00% | 0.72% ± 0.00% | 0.56 ± 0.00 pp | 0.72 / 0.72 / 0.72 | 197.40 ± 0.00 | 3.17 ± 0.03 | 0.712 |
| CICIDS2018 | cnn-s42 | Prim-C&W | 3200 | 13.81% ± 0.00% | 0.00% ± 0.00% | 13.81 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 197.40 ± 0.00 | 3.02 ± 0.03 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-C&W | 3200 | 0.19% ± 0.00% | 0.00% ± 0.00% | 0.19 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 197.40 ± 0.00 | 13.95 ± 0.08 | — |

## Statistical analysis (Valid Targeted Success)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

Cochran's Q across the three optimizers per (dataset, victim). If significant: Hybrid vs Prim-PGD, Hybrid vs Prim-C&W and Prim-PGD vs Prim-C&W, Holm over the three.

| Dataset | Victim | Family | Test | Comparison | n | A-only | B-only | Δ (pp) | Variant | Statistic | p | Holm p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 4.00 | 0.135 | — | No evidence that valid success differs among the 3 conditions (Q = 4.00, p = 0.135); planned McNemar tests not performed. |
| CICIDS2017 | mlp | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | mlp | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | mlp | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | cnn | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 1380.17 | 1.99e-300 | — | Valid success differs among the 3 paired conditions (Q = 1380.2, p = 1.99e-300); planned McNemar tests follow. |
| CICIDS2017 | cnn | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD | 3200 | 8 | 4 | +0.13 | exact binomial |  | 0.388 | 0.388 | No significant difference (Holm-adjusted p = 0.388; Δ = +0.13 pp, 8 vs 4 discordant flows). |
| CICIDS2017 | cnn | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W | 3200 | 699 | 0 | +21.84 | χ² (cc) | 697.00 | 1.34e-153 | 4.03e-153 | Hybrid Search has higher Valid Targeted ASR than Prim-C&W by 21.84 pp (699 vs 0 discordant flows; Holm-adjusted p = 4.03e-153). |
| CICIDS2017 | cnn | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W | 3200 | 696 | 1 | +21.72 | χ² (cc) | 691.01 | 2.69e-152 | 5.38e-152 | Prim-PGD has higher Valid Targeted ASR than Prim-C&W by 21.72 pp (696 vs 1 discordant flows; Holm-adjusted p = 5.38e-152). |
| CICIDS2017 | ft_transformer | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 0.00 | 1 | — | No evidence that valid success differs among the 3 conditions (Q = 0.00, p = 1); planned McNemar tests not performed. |
| CICIDS2017 | ft_transformer | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | ft_transformer | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | ft_transformer | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | mlp-s42 | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 24.40 | 5.03e-06 | — | Valid success differs among the 3 paired conditions (Q = 24.4, p = 5.03e-06); planned McNemar tests follow. |
| CICIDS2018 | mlp-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD | 3200 | 2 | 1 | +0.03 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.03 pp, 2 vs 1 discordant flows). |
| CICIDS2018 | mlp-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W | 3200 | 0 | 13 | -0.41 | exact binomial |  | 0.000244 | 0.000488 | Hybrid Search has lower Valid Targeted ASR than Prim-C&W by 0.41 pp (0 vs 13 discordant flows; Holm-adjusted p = 0.000488). |
| CICIDS2018 | mlp-s42 | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W | 3200 | 0 | 14 | -0.44 | exact binomial |  | 0.000122 | 0.000366 | Prim-PGD has lower Valid Targeted ASR than Prim-C&W by 0.44 pp (0 vs 14 discordant flows; Holm-adjusted p = 0.000366). |
| CICIDS2018 | cnn-s42 | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 0.00 | 1 | — | No evidence that valid success differs among the 3 conditions (Q = 0.00, p = 1); planned McNemar tests not performed. |
| CICIDS2018 | cnn-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | cnn-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | cnn-s42 | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 0.00 | 1 | — | No evidence that valid success differs among the 3 conditions (Q = 0.00, p = 1); planned McNemar tests not performed. |
| CICIDS2018 | ft_transformer-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |

## Class-wise results

| Dataset | Victim | Class | Optimizer | Raw | Valid |
|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | Hybrid Search | 28.88% ± 0.00% | 28.88% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search | 13.25% ± 0.00% | 13.25% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search | 2.12% ± 0.00% | 2.12% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search | 49.33% ± 0.07% | 49.33% ± 0.07% |
| CICIDS2017 | cnn | DDoS | Hybrid Search | 35.87% ± 0.00% | 35.87% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search | 59.25% ± 0.00% | 59.25% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search | 1.38% ± 0.00% | 1.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search | 0.38% ± 0.12% | 0.29% ± 0.14% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search | 4.75% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search | 55.88% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search | 4.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search | 0.08% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search | 1.25% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD | 28.88% ± 0.00% | 28.88% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD | 13.25% ± 0.00% | 13.25% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD | 2.12% ± 0.00% | 2.12% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD | 49.29% ± 0.40% | 49.29% ± 0.40% |
| CICIDS2017 | cnn | DDoS | Prim-PGD | 35.87% ± 0.00% | 35.87% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD | 59.13% ± 0.13% | 59.13% ± 0.13% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD | 1.38% ± 0.00% | 1.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD | 4.75% ± 0.00% | 0.54% ± 0.07% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD | 55.92% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD | 4.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD | 0.08% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD | 1.21% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-C&W | 28.88% ± 0.00% | 28.88% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-C&W | 13.25% ± 0.00% | 13.25% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-C&W | 1.88% ± 0.00% | 1.88% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-C&W | 10.00% ± 0.00% | 10.00% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-C&W | 9.50% ± 0.00% | 9.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-C&W | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-C&W | 37.62% ± 0.00% | 37.62% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-C&W | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-C&W | 1.38% ± 0.00% | 1.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-C&W | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-C&W | 4.62% ± 0.00% | 2.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-C&W | 53.37% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-C&W | 1.88% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-C&W | 0.75% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |

## Plots

- `plots/B1_raw_asr_by_optimizer.png`
- `plots/B2_valid_asr_by_optimizer.png`
- `plots/B3_runtime_and_evaluations.png`

## Interpretation

**Selection (locked criterion).** Hybrid Search has the highest aggregate Valid Targeted ASR
at p75 over both datasets, all victims, classes and seeds: 4,602 / 57,600 = 7.990%. Prim-PGD
follows with 4,595 = 7.977% and Prim-C&W with 2,541 = 4.411%. Hybrid Search is therefore the
PrimAttack optimizer for Exp A and D, and Hybrid and Prim-PGD are the Exp C optimizers. The
margin between Hybrid and Prim-PGD is 7 flow-instances (0.012 pp) across 57,600 attacked
instances. The selection follows the pre-registered rule. It is not evidence that Hybrid is
the more effective search.

**Supporting paired evidence (seed 42, per dataset × victim).** Cochran's Q is significant for
CICIDS2017 CNN (Q = 1380.2) and CICIDS2018 MLP (Q = 24.4). It is not significant for the other
four victims. On CICIDS2017 FT and CICIDS2018 CNN/FT all three optimizers solve identical flow
sets (Q = 0). On CICIDS2017 MLP they differ by 2 flows (Q = 4.0, p = 0.135).
- CICIDS2017 CNN: Hybrid vs Prim-PGD is not significant (8 vs 4 discordant flows, +0.13 pp,
  Holm p = 0.39). Both beat Prim-C&W by about 22 pp (Hybrid +21.84 pp, 699 vs 0; Prim-PGD
  +21.72 pp, 696 vs 1; Holm p < 1e-150).
- CICIDS2018 MLP: Prim-C&W is higher than Hybrid (+0.41 pp, 13 vs 0, Holm p = 0.0005) and
  than Prim-PGD (+0.44 pp, 14 vs 0, Holm p = 0.0004). The effect is statistically clear but
  practically small (under 0.5 pp of 3,200 flows).

**Why.** Hybrid and Prim-PGD share the same projected sign-momentum update and differ mainly in
Hybrid's exact padding enumeration and step adaptation. They find essentially the same flows.
Prim-C&W restarts every binary-search stage from the clean flow and minimizes primitive cost
jointly with the margin. It is markedly weaker on CICIDS2017 CNN (−21.8 pp). On CICIDS2018 MLP
every valid success of every optimizer is timing-only (p = 0) and mostly Recon, while every
invalid raw success uses padding and fails a MINED rule (`MINED_0001` in every Hybrid case
checked). Prim-C&W's cost term penalizes padding. At seed 42 it finds 23 timing-only valid
successes, against 10 for Hybrid and 9 for Prim-PGD. Validity gaps are similar across
optimizers: there is none on CICIDS2017, and on CICIDS2018 gaps reach 15.01 pp on CNN. Every
invalid example there fails only the MINED category (Exp E).

**Cost.** Under the matched cap of 256 victim evaluations per flow, Hybrid uses the fewest
evaluations (188.2–188.4 on CICIDS2017, 192.8–192.9 on CICIDS2018, vs 190.6 / 197.4 for Prim-PGD
and Prim-C&W). It is also the fastest on the FT-Transformer (12.2–12.8 vs 13.7–14.0 ms/flow);
on MLP/CNN all three take about 3 ms/flow. Seed variability is negligible (SD ≤ 0.13 pp in Valid
Targeted ASR). Prim-C&W is deterministic (SD 0).

**Thesis reading (Contribution 1).** Given this primitive parameterization, the valid-success
ceiling is set mainly by the attack space (budget, capabilities, validator) and by the victim,
not by the optimizer. Two of the three gradient searches reach the same flows. The selected
Hybrid Search is at least as effective as the strongest alternative, at the lowest query cost.
