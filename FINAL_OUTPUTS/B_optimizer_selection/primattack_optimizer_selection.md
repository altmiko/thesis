# Final Experiment B — PrimAttack optimizer selection (targeted → Benign, p75)

Three optimizers over the identical PrimAttack attack space (`RealizedSearch`: the same primitive parameterization, joint mode, per-class p75 box, recomputation φ, integer rounding, victim, validator_v2 gate and success predicate; every reported candidate is a realized, quantized, recomputed flow; per-flow incumbent: success > failure, cheapest success, best-margin failure). Matched per-flow cap of 256 victim evaluations. Success = prediction == Benign. Valid success additionally requires validator_v2 `hybrid_valid`. Mean ± SD over seeds 42/2024/2026.

| Optimizer | Hyperparameters (frozen; Prim-PGD/C&W tuned on the CICIDS2017 validation split) |
|---|---|
| Hybrid Search | exact integer padding enumeration, then adaptive projected sign-momentum refinement (40 steps/restart, lr 0.1, momentum 0.75, stall halving), restarts until the 256-evaluation budget |
| Prim-PGD | 3 restarts (clean + 2 uniform) × 42 steps, α = 0.05, momentum 0.75 |
| Prim-C&W | 3 binary-search stages × 42 Adam steps, lr 0.5, c₀ = 1, κ = 0 |

## Optimizer selection (pre-registered criterion)

Rule: highest aggregate Valid Targeted ASR at p75 pooled over both datasets, all victims, classes and seeds (sum successes / sum attempts); ties -> fewer mean victim evaluations per flow -> order hybrid, pgd, cw. **Selected: Prim-PGD.** Budget-sensitivity optimizers (top 2): Prim-PGD, Hybrid Search. The selection is based on the aggregate Valid Targeted ASR, not on p-values. The tests below are supporting evidence.

| Rank | Optimizer | Valid targeted successes | Attempts | Aggregate Valid Targeted ASR | Mean evals / flow |
|---|---|---|---|---|---|
| 1 | Prim-PGD | 1752 | 57600 | 3.042% | 188.5 |
| 2 | Hybrid Search | 1752 | 57600 | 3.042% | 189.6 |
| 3 | Prim-C&W | 861 | 57600 | 1.495% | 188.5 |

## Results per dataset and victim

| Dataset | Victim | Optimizer | n/seed | Raw Targeted ASR | Valid Targeted ASR | Validity Gap | Valid per seed (42/2024/2026, %) | Evals / flow | ms / flow | Median primitive cost (valid) |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | Hybrid Search | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 4.09 / 4.09 / 4.09 | 188.50 ± 0.00 | 3.09 ± 0.02 | 0.635 |
| CICIDS2017 | cnn | Hybrid Search | 3200 | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp | 13.25 / 13.25 / 13.25 | 188.50 ± 0.00 | 3.32 ± 0.07 | 0.596 |
| CICIDS2017 | ft_transformer | Hybrid Search | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12 / 0.12 / 0.12 | 188.58 ± 0.00 | 13.58 ± 0.11 | 0.285 |
| CICIDS2018 | mlp-s42 | Hybrid Search | 3200 | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 ± 0.00 pp | 0.78 / 0.78 / 0.78 | 190.79 ± 0.00 | 3.18 ± 0.07 | 0.475 |
| CICIDS2018 | cnn-s42 | Hybrid Search | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 190.71 ± 0.00 | 3.02 ± 0.03 | — |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 190.79 ± 0.00 | 13.54 ± 0.10 | — |
| CICIDS2017 | mlp | Prim-PGD | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 4.09 / 4.09 / 4.09 | 187.09 ± 0.00 | 2.98 ± 0.03 | 0.617 |
| CICIDS2017 | cnn | Prim-PGD | 3200 | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp | 13.25 / 13.25 / 13.25 | 187.09 ± 0.00 | 3.22 ± 0.02 | 0.646 |
| CICIDS2017 | ft_transformer | Prim-PGD | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12 / 0.12 / 0.12 | 187.16 ± 0.00 | 13.50 ± 0.11 | 0.265 |
| CICIDS2018 | mlp-s42 | Prim-PGD | 3200 | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 ± 0.00 pp | 0.78 / 0.78 / 0.78 | 189.92 ± 0.00 | 3.07 ± 0.06 | 0.453 |
| CICIDS2018 | cnn-s42 | Prim-PGD | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 189.84 ± 0.00 | 2.93 ± 0.04 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 189.92 ± 0.00 | 13.39 ± 0.02 | — |
| CICIDS2017 | mlp | Prim-C&W | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 4.09 / 4.09 / 4.09 | 187.09 ± 0.00 | 3.14 ± 0.02 | 0.504 |
| CICIDS2017 | cnn | Prim-C&W | 3200 | 4.00% ± 0.00% | 4.00% ± 0.00% | 0.00 ± 0.00 pp | 4.00 / 4.00 / 4.00 | 187.09 ± 0.00 | 3.33 ± 0.03 | 0.524 |
| CICIDS2017 | ft_transformer | Prim-C&W | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12 / 0.12 / 0.12 | 187.16 ± 0.00 | 13.59 ± 0.03 | 0.085 |
| CICIDS2018 | mlp-s42 | Prim-C&W | 3200 | 0.75% ± 0.00% | 0.75% ± 0.00% | 0.00 ± 0.00 pp | 0.75 / 0.75 / 0.75 | 189.92 ± 0.00 | 3.17 ± 0.05 | 0.319 |
| CICIDS2018 | cnn-s42 | Prim-C&W | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 189.84 ± 0.00 | 3.05 ± 0.02 | — |
| CICIDS2018 | ft_transformer-s42 | Prim-C&W | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 189.92 ± 0.00 | 13.44 ± 0.08 | — |

## Statistical analysis (Valid Targeted Success)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

Cochran's Q across the three optimizers per (dataset, victim). If significant: Hybrid vs Prim-PGD, Hybrid vs Prim-C&W and Prim-PGD vs Prim-C&W, Holm over the three.

| Dataset | Victim | Family | Test | Comparison | n | A-only | B-only | Δ (pp) | Variant | Statistic | p | Holm p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 0.00 | 1 | — | No evidence that valid success differs among the 3 conditions (Q = 0.00, p = 1); planned McNemar tests not performed. |
| CICIDS2017 | mlp | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | mlp | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | mlp | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | cnn | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 592.00 | 2.81e-129 | — | Valid success differs among the 3 paired conditions (Q = 592.0, p = 2.81e-129); planned McNemar tests follow. |
| CICIDS2017 | cnn | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD | 3200 | 0 | 0 | +0.00 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2017 | cnn | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W | 3200 | 296 | 0 | +9.25 | χ² (cc) | 294.00 | 6.67e-66 | 2e-65 | Hybrid Search has higher Valid Targeted ASR than Prim-C&W by 9.25 pp (296 vs 0 discordant flows; Holm-adjusted p = 2e-65). |
| CICIDS2017 | cnn | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W | 3200 | 296 | 0 | +9.25 | χ² (cc) | 294.00 | 6.67e-66 | 2e-65 | Prim-PGD has higher Valid Targeted ASR than Prim-C&W by 9.25 pp (296 vs 0 discordant flows; Holm-adjusted p = 2e-65). |
| CICIDS2017 | ft_transformer | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 0.00 | 1 | — | No evidence that valid success differs among the 3 conditions (Q = 0.00, p = 1); planned McNemar tests not performed. |
| CICIDS2017 | ft_transformer | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | ft_transformer | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2017 | ft_transformer | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | mlp-s42 | B: 3 optimizers | Cochran's Q | Hybrid Search / Prim-PGD / Prim-C&W | 3200 |  |  |  | Cochran χ² | 2.00 | 0.368 | — | No evidence that valid success differs among the 3 conditions (Q = 2.00, p = 0.368); planned McNemar tests not performed. |
| CICIDS2018 | mlp-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | mlp-s42 | B: optimizer pairs (Holm over 3) | McNemar | Hybrid Search vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | mlp-s42 | B: optimizer pairs (Holm over 3) | McNemar | Prim-PGD vs Prim-C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
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
| CICIDS2017 | mlp | DoS | Hybrid Search | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search | 20.50% ± 0.00% | 20.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Hybrid Search | 31.50% ± 0.00% | 31.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD | 20.50% ± 0.00% | 20.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD | 31.50% ± 0.00% | 31.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-C&W | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-C&W | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-C&W | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-C&W | 7.88% ± 0.00% | 7.88% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-C&W | 7.50% ± 0.00% | 7.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-C&W | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-C&W | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-C&W | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-C&W | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-C&W | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-C&W | 2.50% ± 0.00% | 2.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-C&W | 0.00% ± 0.00% | 0.00% ± 0.00% |

## Plots

- `plots/B1_raw_asr_by_optimizer.png`
- `plots/B2_valid_asr_by_optimizer.png`
- `plots/B3_runtime_and_evaluations.png`

## Interpretation

**Selection.** Under the pre-registered rule, Hybrid Search and Prim-PGD tie exactly: 1,752 valid
targeted successes each out of 57,600 attempts (3.042%). The tie-break (fewer mean victim
evaluations per flow) selects **Prim-PGD** (188.5 vs 189.6). Prim-C&W is third with 861 (1.495%).
Budget sensitivity (Exp C) therefore uses Prim-PGD and Hybrid Search. Before amendment A2 the
same rule had selected Hybrid (7.990% vs 7.977%); the capability fix, not the rule, changed the
outcome.

**Why Hybrid and Prim-PGD coincide.** With the capability-aware padding rule, only 1 (CICIDS2017)
or 12 (CICIDS2018) of 3,200 attacked flows per victim may be padded, so Hybrid's distinguishing
component, the exhaustive integer padding enumeration, has almost nothing to enumerate. What
remains for both is projected sign-momentum descent on the timing controls (delay, shape) from
the clean flow plus restarts, with the same 256-evaluation cap. The two reach identical success
sets: on every victim their Valid Targeted ASR is equal (4.09% / 13.25% / 0.12% on CICIDS2017,
0.78% / 0.00% / 0.00% on CICIDS2018), and on the CICIDS2017 CNN, where the planned McNemar tests
run, Hybrid vs Prim-PGD has 0 discordant flows (Holm p = 1).

The tie is in the binary success outcome, not in the optimization path. Hybrid and Prim-PGD can
return different delay/shape values and different costs on the same successful flow. For example,
their median normalized valid-success costs are 0.635 vs 0.617 on the CICIDS2017 MLP and
0.596 vs 0.646 on the CNN. The result therefore supports equivalence of the observed success
sets under this protocol, not algorithmic identity.

**Prim-C&W.** Its cost-penalized objective (normalized primitive cost + c · margin) finds fewer
successes where the needed delay is large: on the CICIDS2017 CNN it reaches 4.00% vs 13.25% for
the other two (296 vs 0 discordant flows; Holm p = 2e-65 for both comparisons). Elsewhere it ties
within 0.03 pp (CICIDS2018 MLP 0.75% vs 0.78%; Cochran's Q p = 0.368). Cochran's Q is not
significant, or not computable because all three are identical, on the other five victims.

**Validity.** Every targeted raw success of every optimizer is valid (Validity Gap 0.00 pp in all
18 cells): the realized-flow search keeps only validator-accepted incumbents, and the timing-only
flows cannot trigger the empty-packet rule. The optimizers spend a similar per-flow budget
(187–191 victim evaluations) because flows without primitive headroom stop at the identity.

**Reading.** After the fix, the optimizer choice matters little: timing is a low-dimensional
search that simple projected descent already saturates within the budget. The victim and the
timing budget determine success (Exp C), not the optimizer.
