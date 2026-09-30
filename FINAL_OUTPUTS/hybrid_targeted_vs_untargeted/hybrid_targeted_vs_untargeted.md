# PrimAttack Hybrid — targeted vs untargeted Valid ASR

Hybrid Search, p75 budget, joint mode, capability-aware padding, 256 victim evaluations per flow, validator_v2 `hybrid_valid`; the locked final-suite configuration (`runs/final_suite_config.json`, `00_PROTOCOL.md`). Both objectives attack the same canonical clean-correct source flows (`runs/<dataset>/baselines_untargeted/selection.json`, 800 per class, N = 3,200 per victim) with attack seeds 42, 2024, 2026.

- Untargeted success: final prediction ≠ original malicious class.
- Targeted success: final prediction = Benign.
- Valid ASR = (success ∧ validator-valid final flow) / attempted clean-correct malicious flows. Mean ± SD (ddof = 1) over the three seed-level rates; classes pooled within a victim; datasets and victims never pooled.
- Test: paired McNemar on reference seed 42 (exact binomial if discordant pairs < 25, else continuity-corrected χ²), Holm over the six dataset-victim comparisons. Seeds 2024 / 2026 are descriptive only.
- Each objective optimizes its own margin, so the two conditions are two separate searches on the same flows, not one search scored two ways.

## Final table

| Dataset | Victim | Untargeted Valid ASR | Targeted→Benign Valid ASR | Difference (Untargeted−Targeted) | Seed-42 discordance Untargeted-only/Targeted-only | Holm p |
|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 4.09% ± 0.00% | 4.09% ± 0.00% | +0.00 ± 0.00 pp | 0 / 0 | 1 |
| CICIDS2017 | cnn | 13.47% ± 0.00% | 13.25% ± 0.00% | +0.22 ± 0.00 pp | 7 / 0 | 0.0625 |
| CICIDS2017 | ft_transformer | 0.12% ± 0.00% | 0.12% ± 0.00% | +0.00 ± 0.00 pp | 0 / 0 | 1 |
| CICIDS2018 | mlp | 2.53% ± 0.00% | 0.78% ± 0.00% | +1.75 ± 0.00 pp | 56 / 0 | 1.19e-12 |
| CICIDS2018 | cnn | 1.16% ± 0.00% | 0.00% ± 0.00% | +1.16 ± 0.00 pp | 37 / 0 | 1.63e-08 |
| CICIDS2018 | ft_transformer | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 ± 0.00 pp | 0 / 0 | 1 |

## Per-seed values (Raw ASR kept for reference)

| Dataset | Victim | Objective | Valid ASR 42 / 2024 / 2026 (%) | Valid ASR mean ± SD | Raw ASR 42 / 2024 / 2026 (%) | Raw ASR mean ± SD | Validity gap (pp) |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | untargeted | 4.09 / 4.09 / 4.09 | 4.09% ± 0.00% | 4.09 / 4.09 / 4.09 | 4.09% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2017 | mlp | targeted→Benign | 4.09 / 4.09 / 4.09 | 4.09% ± 0.00% | 4.09 / 4.09 / 4.09 | 4.09% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2017 | cnn | untargeted | 13.47 / 13.47 / 13.47 | 13.47% ± 0.00% | 13.47 / 13.47 / 13.47 | 13.47% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2017 | cnn | targeted→Benign | 13.25 / 13.25 / 13.25 | 13.25% ± 0.00% | 13.25 / 13.25 / 13.25 | 13.25% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2017 | ft_transformer | untargeted | 0.12 / 0.12 / 0.12 | 0.12% ± 0.00% | 0.12 / 0.12 / 0.12 | 0.12% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2017 | ft_transformer | targeted→Benign | 0.12 / 0.12 / 0.12 | 0.12% ± 0.00% | 0.12 / 0.12 / 0.12 | 0.12% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2018 | mlp | untargeted | 2.53 / 2.53 / 2.53 | 2.53% ± 0.00% | 2.53 / 2.53 / 2.53 | 2.53% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2018 | mlp | targeted→Benign | 0.78 / 0.78 / 0.78 | 0.78% ± 0.00% | 0.78 / 0.78 / 0.78 | 0.78% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2018 | cnn | untargeted | 1.16 / 1.16 / 1.16 | 1.16% ± 0.00% | 1.16 / 1.16 / 1.16 | 1.16% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2018 | cnn | targeted→Benign | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2018 | ft_transformer | untargeted | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% | 0.00 ± 0.00 |
| CICIDS2018 | ft_transformer | targeted→Benign | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% | 0.00 ± 0.00 |

## Paired tests (Valid success, seed 42)

| Dataset | Victim | N | Untargeted / targeted valid successes (seed 42) | Both / neither | Untargeted-only / targeted-only (42; 2024; 2026) | Δ seed 42 [Newcombe 95% CI] | Test | Statistic | p | Holm p | Significant (Holm, α = 0.05) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 3200 | 131 / 131 | 131 / 3069 | 0/0; 0/0; 0/0 | +0.00 pp [-0.13, +0.13] | exact binomial McNemar | — | 1 | 1 | no |
| CICIDS2017 | cnn | 3200 | 431 / 424 | 424 / 2769 | 7/0; 7/0; 7/0 | +0.22 pp [+0.03, +0.42] | exact binomial McNemar | — | 0.0156 | 0.0625 | no |
| CICIDS2017 | ft_transformer | 3200 | 4 / 4 | 4 / 3196 | 0/0; 0/0; 0/0 | +0.00 pp [-0.13, +0.13] | exact binomial McNemar | — | 1 | 1 | no |
| CICIDS2018 | mlp | 3200 | 81 / 25 | 25 / 3119 | 56/0; 56/0; 56/0 | +1.75 pp [+1.32, +2.26] | asymptotic McNemar chi-square (continuity corrected) | 54.018 | 1.99e-13 | 1.19e-12 | yes |
| CICIDS2018 | cnn | 3200 | 37 / 0 | 0 / 3163 | 37/0; 37/0; 37/0 | +1.16 pp [+0.82, +1.59] | asymptotic McNemar chi-square (continuity corrected) | 35.027 | 3.25e-09 | 1.63e-08 | yes |
| CICIDS2018 | ft_transformer | 3200 | 0 / 0 | 0 / 3200 | 0/0; 0/0; 0/0 | +0.00 pp [-0.12, +0.12] | exact binomial McNemar | — | 1 | 1 | no |

## Per class (Valid ASR, mean ± SD over seeds)

| Dataset | Victim | Class | Untargeted Valid ASR | Targeted→Benign Valid ASR |
|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | 20.50% ± 0.00% | 20.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | 32.12% ± 0.00% | 31.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | 0.38% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp | DoS | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp | DDoS | 7.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp | Recon | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp | BruteForce | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn | DoS | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn | DDoS | 4.62% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn | Recon | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn | BruteForce | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer | DoS | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer | DDoS | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer | Recon | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer | BruteForce | 0.00% ± 0.00% | 0.00% ± 0.00% |

## Provenance and checks

- Integrity audit: 144 artifacts, 115,200 rows; validator_v2 re-run on 115,200 stored final flows; victim re-prediction on 115,200 flows with 0 mismatches; identical sample IDs, order, clean-input and checkpoint hashes across both objectives and all seeds.
- Targeted re-run vs the existing Exp B Hybrid p75 cell (`primattack_targeted_optimizers`): flows whose valid / raw success differs = 0 / 0 (reproduced exactly).
- Run stages: `runs/<dataset>/primattack_hybrid_objective_untargeted`, `runs/<dataset>/primattack_hybrid_objective_targeted`. This comparison is an added analysis (not part of the locked Exp A-F families); Exp D remains the pre-registered targeted-vs-untargeted test for the selected optimizer (Prim-PGD).
- Feature-space proxy on CICFlowMeter aggregates; no PCAP edited or replayed.

Files: `final_table.{md,csv}`, `paired_tests.csv`, `seed_level.csv`, `table_level.csv`, `per_sample.parquet`, `targeted_rerun_vs_exp_b.csv`, `audit.json`.
