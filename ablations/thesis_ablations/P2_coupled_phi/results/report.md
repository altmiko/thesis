# P2 - Coupled feature-recomputation ablation: tables

Targeted → Benign. Per (dataset, victim, budget) the four classes are pooled (3,200 frozen flows per seed). Mean ± SD over attack seeds 42/2024/2026. `full_phi` = reference (canonical φ). `direct_only` = P2: the search sees φ's direct writes only; its primitives are then realized through canonical φ and judged by validator_v2. Definition of the reduced path: `phi_mapping.md`.

## Reference vs P2 (Valid = full-φ flow ∧ validator_v2)

| dataset | victim | budget | Ref Raw | Ref Valid | P2 reduced-space Raw | P2 full-φ Raw | P2 full-φ Valid | reduced→full-φ Valid loss (pp) | seed-42 P2-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 4.09 ± 0.00% | 4.09 ± 0.00% | 4.16 ± 0.00% | 3.08 ± 0.09% | 3.08 ± 0.09% | 1.07 ± 0.09 | 0 / 34 | 1.52e-07 |
| cicids2017_distrinet | mlp | unb | 22.94 ± 0.00% | 22.94 ± 0.00% | 28.91 ± 0.00% | 14.90 ± 0.02% | 14.90 ± 0.02% | 14.01 ± 0.02 | 0 / 257 | 2.53e-56 |
| cicids2017_distrinet | cnn | p75 | 13.25 ± 0.00% | 13.25 ± 0.00% | 13.41 ± 0.00% | 12.07 ± 0.15% | 12.07 ± 0.15% | 1.33 ± 0.15 | 0 / 34 | 1.52e-07 |
| cicids2017_distrinet | cnn | unb | 59.69 ± 0.00% | 59.69 ± 0.00% | 60.81 ± 0.00% | 53.75 ± 0.29% | 53.75 ± 0.29% | 7.06 ± 0.29 | 0 / 188 | 2.6e-41 |
| cicids2017_distrinet | ft_transformer | p75 | 0.12 ± 0.00% | 0.12 ± 0.00% | 0.12 ± 0.00% | 0.12 ± 0.00% | 0.12 ± 0.00% | 0.00 ± 0.00 | 0 / 0 | 1 |
| cicids2017_distrinet | ft_transformer | unb | 0.55 ± 0.02% | 0.55 ± 0.02% | 0.41 ± 0.00% | 0.45 ± 0.02% | 0.45 ± 0.02% | -0.04 ± 0.02 | 0 / 4 | 0.75 |
| cicids2018_distrinet | mlp-s42 | p75 | 0.78 ± 0.00% | 0.78 ± 0.00% | 0.69 ± 0.00% | 0.69 ± 0.00% | 0.69 ± 0.00% | 0.00 ± 0.00 | 0 / 3 | 1 |
| cicids2018_distrinet | mlp-s42 | unb | 24.80 ± 0.02% | 24.80 ± 0.02% | 1.61 ± 0.02% | 24.02 ± 0.04% | 24.02 ± 0.04% | -22.41 ± 0.03 | 0 / 23 | 1.91e-06 |
| cicids2018_distrinet | cnn-s42 | p75 | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00 | 0 / 0 | 1 |
| cicids2018_distrinet | cnn-s42 | unb | 26.09 ± 0.00% | 26.09 ± 0.00% | 20.03 ± 0.00% | 25.64 ± 0.02% | 25.64 ± 0.02% | -5.60 ± 0.02 | 0 / 14 | 0.000854 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00% | 0.00 ± 0.00 | 0 / 0 | 1 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 0.12 ± 0.00% | 0.12 ± 0.00% | 0.00 ± 0.00% | 0.12 ± 0.00% | 0.12 ± 0.00% | -0.12 ± 0.00 | 0 / 0 | 1 |

## Apparent (reduced-space) successes after full φ, all seeds pooled

| dataset | victim | budget | apparent | invalidated (%) | lost: prediction | lost: validator | prediction changed among apparent (%) | full-φ hits not apparent | median cost of valid successes P2 / ref |
|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 399 | 121 (30.3%) | 121 | 0 | 121 (30.3%) | 18 | 0.59 / 0.635 |
| cicids2017_distrinet | mlp | unb | 2775 | 1348 (48.6%) | 1348 | 0 | 1348 (48.6%) | 3 | 0.269 / 0.3 |
| cicids2017_distrinet | cnn | p75 | 1287 | 164 (12.7%) | 164 | 0 | 164 (12.7%) | 36 | 0.568 / 0.596 |
| cicids2017_distrinet | cnn | unb | 5838 | 774 (13.3%) | 774 | 0 | 774 (13.3%) | 96 | 0.537 / 0.385 |
| cicids2017_distrinet | ft_transformer | p75 | 12 | 0 (0.0%) | 0 | 0 | 0 (0.0%) | 0 | 0.302 / 0.285 |
| cicids2017_distrinet | ft_transformer | unb | 39 | 14 (35.9%) | 14 | 0 | 14 (35.9%) | 18 | 0.351 / 0.0492 |
| cicids2018_distrinet | mlp-s42 | p75 | 66 | 12 (18.2%) | 12 | 0 | 12 (18.2%) | 12 | 0.517 / 0.475 |
| cicids2018_distrinet | mlp-s42 | unb | 155 | 76 (49.0%) | 76 | 0 | 76 (49.0%) | 2227 | 1 / 0.6 |
| cicids2018_distrinet | cnn-s42 | p75 | 0 | 0 (0.0%) | 0 | 0 | 0 (0.0%) | 0 | nan / nan |
| cicids2018_distrinet | cnn-s42 | unb | 1923 | 44 (2.3%) | 44 | 0 | 44 (2.3%) | 582 | 0.8 / 0.163 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 0 | 0 (0.0%) | 0 | 0 | 0 (0.0%) | 0 | nan / nan |
| cicids2018_distrinet | ft_transformer-s42 | unb | 0 | 0 (0.0%) | 0 | 0 | 0 (0.0%) | 12 | 1 / 0.1 |

## McNemar (P2 full-φ Valid vs reference Valid), all seeds

Holm within each seed's family of 12 cells; seed 42 is the primary family.

| dataset | victim | budget | seed | P2 | ref | diff (pp) [95% CI] | P2-only / ref-only | test | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 42 | 3.03% | 4.09% | -1.06 [-1.46, -0.71] | 0 / 34 | asymptotic McNemar chi-square (continuity corrected) | 1.52e-08 | 1.52e-07 |
| cicids2017_distrinet | mlp | p75 | 2024 | 3.19% | 4.09% | -0.91 [-1.28, -0.58] | 0 / 29 | asymptotic McNemar chi-square (continuity corrected) | 2e-07 | 1.8e-06 |
| cicids2017_distrinet | mlp | p75 | 2026 | 3.03% | 4.09% | -1.06 [-1.46, -0.71] | 0 / 34 | asymptotic McNemar chi-square (continuity corrected) | 1.52e-08 | 1.37e-07 |
| cicids2017_distrinet | mlp | unb | 42 | 14.91% | 22.94% | -8.03 [-8.99, -7.10] | 0 / 257 | asymptotic McNemar chi-square (continuity corrected) | 2.11e-57 | 2.53e-56 |
| cicids2017_distrinet | mlp | unb | 2024 | 14.91% | 22.94% | -8.03 [-8.99, -7.10] | 0 / 257 | asymptotic McNemar chi-square (continuity corrected) | 2.11e-57 | 2.53e-56 |
| cicids2017_distrinet | mlp | unb | 2026 | 14.88% | 22.94% | -8.06 [-9.02, -7.13] | 0 / 258 | asymptotic McNemar chi-square (continuity corrected) | 1.28e-57 | 1.53e-56 |
| cicids2017_distrinet | cnn | p75 | 42 | 12.19% | 13.25% | -1.06 [-1.44, -0.70] | 0 / 34 | asymptotic McNemar chi-square (continuity corrected) | 1.52e-08 | 1.52e-07 |
| cicids2017_distrinet | cnn | p75 | 2024 | 12.12% | 13.25% | -1.13 [-1.52, -0.75] | 0 / 36 | asymptotic McNemar chi-square (continuity corrected) | 5.43e-09 | 5.43e-08 |
| cicids2017_distrinet | cnn | p75 | 2026 | 11.91% | 13.25% | -1.34 [-1.77, -0.94] | 0 / 43 | asymptotic McNemar chi-square (continuity corrected) | 1.5e-10 | 1.5e-09 |
| cicids2017_distrinet | cnn | unb | 42 | 53.81% | 59.69% | -5.88 [-6.69, -5.06] | 0 / 188 | asymptotic McNemar chi-square (continuity corrected) | 2.37e-42 | 2.6e-41 |
| cicids2017_distrinet | cnn | unb | 2024 | 54.00% | 59.69% | -5.69 [-6.49, -4.88] | 0 / 182 | asymptotic McNemar chi-square (continuity corrected) | 4.83e-41 | 5.32e-40 |
| cicids2017_distrinet | cnn | unb | 2026 | 53.44% | 59.69% | -6.25 [-7.09, -5.41] | 0 / 200 | asymptotic McNemar chi-square (continuity corrected) | 5.69e-45 | 6.26e-44 |
| cicids2017_distrinet | ft_transformer | p75 | 42 | 0.12% | 0.12% | +0.00 [-0.13, +0.13] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2017_distrinet | ft_transformer | p75 | 2024 | 0.12% | 0.12% | +0.00 [-0.13, +0.13] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2017_distrinet | ft_transformer | p75 | 2026 | 0.12% | 0.12% | +0.00 [-0.13, +0.13] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2017_distrinet | ft_transformer | unb | 42 | 0.44% | 0.56% | -0.12 [-0.32, +0.04] | 0 / 4 | exact binomial McNemar | 0.125 | 0.75 |
| cicids2017_distrinet | ft_transformer | unb | 2024 | 0.44% | 0.53% | -0.09 [-0.28, +0.06] | 0 / 3 | exact binomial McNemar | 0.25 | 1 |
| cicids2017_distrinet | ft_transformer | unb | 2026 | 0.47% | 0.56% | -0.09 [-0.28, +0.06] | 0 / 3 | exact binomial McNemar | 0.25 | 1 |
| cicids2018_distrinet | mlp-s42 | p75 | 42 | 0.69% | 0.78% | -0.09 [-0.28, +0.06] | 0 / 3 | exact binomial McNemar | 0.25 | 1 |
| cicids2018_distrinet | mlp-s42 | p75 | 2024 | 0.69% | 0.78% | -0.09 [-0.28, +0.06] | 0 / 3 | exact binomial McNemar | 0.25 | 1 |
| cicids2018_distrinet | mlp-s42 | p75 | 2026 | 0.69% | 0.78% | -0.09 [-0.28, +0.06] | 0 / 3 | exact binomial McNemar | 0.25 | 1 |
| cicids2018_distrinet | mlp-s42 | unb | 42 | 24.06% | 24.78% | -0.72 [-1.03, -0.42] | 0 / 23 | exact binomial McNemar | 2.38e-07 | 1.91e-06 |
| cicids2018_distrinet | mlp-s42 | unb | 2024 | 24.00% | 24.81% | -0.81 [-1.14, -0.49] | 0 / 26 | asymptotic McNemar chi-square (continuity corrected) | 9.44e-07 | 7.55e-06 |
| cicids2018_distrinet | mlp-s42 | unb | 2026 | 24.00% | 24.81% | -0.81 [-1.14, -0.49] | 0 / 26 | asymptotic McNemar chi-square (continuity corrected) | 9.44e-07 | 7.55e-06 |
| cicids2018_distrinet | cnn-s42 | p75 | 42 | 0.00% | 0.00% | +0.00 [-0.12, +0.12] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | cnn-s42 | p75 | 2024 | 0.00% | 0.00% | +0.00 [-0.12, +0.12] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | cnn-s42 | p75 | 2026 | 0.00% | 0.00% | +0.00 [-0.12, +0.12] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | cnn-s42 | unb | 42 | 25.66% | 26.09% | -0.44 [-0.68, -0.20] | 0 / 14 | exact binomial McNemar | 0.000122 | 0.000854 |
| cicids2018_distrinet | cnn-s42 | unb | 2024 | 25.62% | 26.09% | -0.47 [-0.72, -0.22] | 0 / 15 | exact binomial McNemar | 6.1e-05 | 0.000427 |
| cicids2018_distrinet | cnn-s42 | unb | 2026 | 25.62% | 26.09% | -0.47 [-0.72, -0.22] | 0 / 15 | exact binomial McNemar | 6.1e-05 | 0.000427 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 42 | 0.00% | 0.00% | +0.00 [-0.12, +0.12] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 2024 | 0.00% | 0.00% | +0.00 [-0.12, +0.12] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 2026 | 0.00% | 0.00% | +0.00 [-0.12, +0.12] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 42 | 0.12% | 0.12% | +0.00 [-0.13, +0.13] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 2024 | 0.12% | 0.12% | +0.00 [-0.13, +0.13] | 0 / 0 | exact binomial McNemar | 1 | 1 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 2026 | 0.12% | 0.12% | +0.00 [-0.13, +0.13] | 0 / 0 | exact binomial McNemar | 1 | 1 |

## Reduced-space hit vs full-φ Valid within P2 (same primitives), seed 42

| dataset | victim | budget | reduced hit | full-φ Valid | reduced-only / full-only | p (Holm) |
|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 4.16% | 3.03% | 42 / 6 | 3.5e-06 |
| cicids2017_distrinet | mlp | unb | 28.91% | 14.91% | 449 / 1 | 1.59e-97 |
| cicids2017_distrinet | cnn | p75 | 13.41% | 12.19% | 51 / 12 | 1.18e-05 |
| cicids2017_distrinet | cnn | unb | 60.81% | 53.81% | 256 / 32 | 1.93e-38 |
| cicids2017_distrinet | ft_transformer | p75 | 0.12% | 0.12% | 0 / 0 | 1 |
| cicids2017_distrinet | ft_transformer | unb | 0.41% | 0.44% | 5 / 6 | 1 |
| cicids2018_distrinet | mlp-s42 | p75 | 0.69% | 0.69% | 4 / 4 | 1 |
| cicids2018_distrinet | mlp-s42 | unb | 1.62% | 24.06% | 24 / 742 | 6.79e-147 |
| cicids2018_distrinet | cnn-s42 | p75 | 0.00% | 0.00% | 0 / 0 | 1 |
| cicids2018_distrinet | cnn-s42 | unb | 20.03% | 25.66% | 14 / 194 | 2.04e-34 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 0.00% | 0.00% | 0 / 0 | 1 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 0.00% | 0.12% | 0 / 4 | 0.75 |

## Consistency of the reduced flows (rows with a non-zero primitive, all seeds)

validator_v2 = the final validator (reduced flow judged given its source flow). Level B = `attack/realizability/validator.py`: φ's exact identities (`algebraic_identities`) and timing order checks (e.g. Fwd IAT Total ≤ Flow Duration). Diagnostic only; no P2 result is counted on a reduced flow.

| dataset | victim | budget | moved rows | reduced: v2 accepts | reduced: identity violated | v2-accepted with identity violated | v2-accepted with timing order violated | full-φ: Level-B fail | full-φ: v2 rejects |
|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 7071 | 69.2% | 100.0% | 4894 | 4894 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | 7089 | 22.8% | 100.0% | 1615 | 1615 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | 7084 | 72.4% | 100.0% | 5126 | 5126 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | 7089 | 23.2% | 100.0% | 1643 | 1643 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | 7081 | 69.8% | 100.0% | 4942 | 4941 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | 7089 | 17.7% | 100.0% | 1256 | 1256 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | 4209 | 78.5% | 99.7% | 3291 | 3270 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | 5107 | 8.3% | 100.0% | 422 | 410 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | 5304 | 59.5% | 99.9% | 3152 | 3139 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | 5333 | 12.4% | 100.0% | 661 | 654 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 5261 | 69.0% | 99.9% | 3624 | 3626 | 0 | 5 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 6638 | 23.1% | 100.0% | 1532 | 1532 | 0 | 5 |

Per-identity violations on the reduced flows (all victims and seeds):

| dataset | budget | identity target | violated | of which accepted by validator_v2 |
|---|---|---|---|---|
| cicids2017_distrinet | p75 | Fwd Packet Length Mean | 9 | 0 |
| cicids2017_distrinet | p75 | Packet Length Mean | 9 | 0 |
| cicids2017_distrinet | p75 | Packet Length Max | 9 | 0 |
| cicids2017_distrinet | p75 | Fwd IAT Mean | 21222 | 14962 |
| cicids2017_distrinet | p75 | Flow Bytes/s | 9 | 0 |
| cicids2017_distrinet | unb | Fwd Packet Length Mean | 9 | 0 |
| cicids2017_distrinet | unb | Packet Length Mean | 9 | 0 |
| cicids2017_distrinet | unb | Packet Length Max | 9 | 0 |
| cicids2017_distrinet | unb | Fwd IAT Mean | 21258 | 4514 |
| cicids2017_distrinet | unb | Flow Bytes/s | 9 | 0 |
| cicids2018_distrinet | p75 | Fwd Packet Length Mean | 48 | 0 |
| cicids2018_distrinet | p75 | Packet Length Mean | 48 | 0 |
| cicids2018_distrinet | p75 | Packet Length Min | 48 | 0 |
| cicids2018_distrinet | p75 | Fwd IAT Mean | 14704 | 10067 |
| cicids2018_distrinet | p75 | Flow Bytes/s | 48 | 0 |
| cicids2018_distrinet | unb | Fwd Packet Length Mean | 48 | 0 |
| cicids2018_distrinet | unb | Packet Length Mean | 48 | 0 |
| cicids2018_distrinet | unb | Packet Length Min | 48 | 0 |
| cicids2018_distrinet | unb | Fwd IAT Mean | 17029 | 2615 |
| cicids2018_distrinet | unb | Flow Bytes/s | 48 | 0 |

## Which recomputed groups account for the lost successes (lost by prediction)

Single substitutions between the reduced and the full-φ flow of the same primitives. *alone breaks*: reduced flow + this group's full-φ values → no longer Benign. *reverting restores*: full-φ flow with this group reset to source values → Benign again. Descriptive of the victims' decisions on these flows; interactions between groups are not separated.

| dataset | φ code block (derived group) | lost rows | rows where it changed | alone breaks | reverting restores |
|---|---|---|---|---|---|
| cicids2017_distrinet | affine allocation of total forward delay | 2416 | 2416 | 2340 (96.9%) | 2234 (92.5%) |
| cicids2017_distrinet | rates | 2416 | 2416 | 182 (7.5%) | 76 (3.1%) |
| cicids2017_distrinet | forward packet-length augmentation | 2416 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | combined packet-length statistics | 2416 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | affine allocation of total forward delay | 132 | 132 | 132 (100.0%) | 129 (97.7%) |
| cicids2018_distrinet | rates | 132 | 132 | 3 (2.3%) | 0 (0.0%) |
| cicids2018_distrinet | forward packet-length augmentation | 132 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | combined packet-length statistics | 132 | 0 | 0 (0.0%) | 0 (0.0%) |

| dataset | feature | lost rows | changed | median abs. change (scaled) | alone breaks | reverting restores |
|---|---|---|---|---|---|---|
| cicids2017_distrinet | Fwd IAT Mean | 2416 | 2416 | 1.39 | 1901 (78.7%) | 1419 (58.7%) |
| cicids2017_distrinet | Flow IAT Mean | 2416 | 2416 | 1.45 | 1466 (60.7%) | 532 (22.0%) |
| cicids2017_distrinet | Flow Duration | 2416 | 2416 | 1.17 | 749 (31.0%) | 367 (15.2%) |
| cicids2017_distrinet | Flow Bytes/s | 2416 | 2416 | 0.025 | 186 (7.7%) | 79 (3.3%) |
| cicids2017_distrinet | Flow IAT Max | 2416 | 2416 | 1.37 | 112 (4.6%) | 82 (3.4%) |
| cicids2017_distrinet | Bwd Packets/s | 2416 | 2416 | 0.00136 | 19 (0.8%) | 20 (0.8%) |
| cicids2017_distrinet | Flow Packets/s | 2416 | 2416 | 0.00131 | 2 (0.1%) | 10 (0.4%) |
| cicids2017_distrinet | Fwd Packets/s | 2416 | 2416 | 0.00125 | 2 (0.1%) | 8 (0.3%) |
| cicids2017_distrinet | Fwd Packet Length Mean | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | Fwd Segment Size Avg | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | Packet Length Max | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | Packet Length Min | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | Packet Length Mean | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | Average Packet Size | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | Packet Length Variance | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2017_distrinet | Packet Length Std | 2416 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Flow Duration | 132 | 132 | 31.6 | 123 (93.2%) | 66 (50.0%) |
| cicids2018_distrinet | Fwd IAT Mean | 132 | 132 | 60.9 | 91 (68.9%) | 85 (64.4%) |
| cicids2018_distrinet | Flow IAT Mean | 132 | 132 | 50.1 | 37 (28.0%) | 17 (12.9%) |
| cicids2018_distrinet | Flow IAT Max | 132 | 132 | 79.9 | 8 (6.1%) | 31 (23.5%) |
| cicids2018_distrinet | Flow Packets/s | 132 | 132 | 0.00677 | 4 (3.0%) | 3 (2.3%) |
| cicids2018_distrinet | Flow Bytes/s | 132 | 132 | 0.0239 | 3 (2.3%) | 0 (0.0%) |
| cicids2018_distrinet | Fwd Packet Length Mean | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Fwd Segment Size Avg | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Packet Length Max | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Packet Length Min | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Packet Length Mean | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Average Packet Size | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Packet Length Variance | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Packet Length Std | 132 | 0 | 0 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Fwd Packets/s | 132 | 132 | 0.00675 | 0 (0.0%) | 0 (0.0%) |
| cicids2018_distrinet | Bwd Packets/s | 132 | 132 | 0.00675 | 0 (0.0%) | 0 (0.0%) |

## Sanity checks

* full_phi reproduces FINAL targeted Hybrid: 115,200 / 115,200 realized flows identical, 115,200 predictions, 115,200 validator verdicts, 115,200 valid-success outcomes (p75 vs `primattack_hybrid_objective_targeted`, unbounded vs `primattack_targeted_budgets`).
* pairing: {'cells': 36, 'sample_ids_identical_cells': 36, 'budget_boxes_identical_cells': 36}
* primitive identity (stored primitives → φ / direct-only map): {'rows': 115200, 'full_phi_recomputed_identical': 115200, 'reduced_recomputed_identical': 115200, 'reprojection_identical': 115200, 'reduced_full_differ_only_on_derived': 115200}
* capability inference: {'ref_violations': 0, 'p2_violations': 0, 'cap_mask_matches_inference': 115200, 'rows': 115200}
* validator_v2 recomputed on stored flows: {'rows': 115200, 'p2_validator_recomputed_identical': 115200, 'ref_validator_recomputed_identical': 115200}
* final counting (must be all 0): {'p2_valid_not_full_hit': 0, 'p2_valid_not_full_validator': 0, 'p2_raw_success_not_full_benign': 0, 'ref_valid_not_hit_and_valid': 0}
* lost-success verdicts re-scored from stored vectors: {'rows': 2553, 'reduced_hit_reproduced': 2548, 'full_verdict_reproduced': 2553}
* all passed: **True**

