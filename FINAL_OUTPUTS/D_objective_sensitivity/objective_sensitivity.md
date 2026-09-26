# Final Experiment D — Objective sensitivity (targeted → Benign vs untargeted)

PrimAttack with the selected optimizer (Hybrid Search), joint mode, p75 budget, on identical flows and seeds. Targeted success = prediction == Benign. Untargeted success = prediction ≠ source class. Both arms use the same validator gate and the same incumbent rule. Only the objective margin differs. The targeted arm is the Exp B cell and the untargeted arm is the Exp A PrimAttack cell.

## Results

| Dataset | Victim | Objective | n/seed | Raw ASR | Valid ASR | Validity Gap | Valid per seed (42/2024/2026, %) |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | Hybrid Search targeted→Benign | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 11.06 / 11.06 / 11.06 |
| CICIDS2017 | cnn | Hybrid Search targeted→Benign | 3200 | 36.15% ± 0.02% | 36.15% ± 0.02% | 0.00 ± 0.00 pp | 36.16 / 36.12 / 36.16 |
| CICIDS2017 | ft_transformer | Hybrid Search targeted→Benign | 3200 | 0.44% ± 0.00% | 0.44% ± 0.00% | 0.00 ± 0.00 pp | 0.44 / 0.44 / 0.44 |
| CICIDS2018 | mlp-s42 | Hybrid Search targeted→Benign | 3200 | 1.28% ± 0.03% | 0.29% ± 0.04% | 0.99 ± 0.02 pp | 0.31 / 0.31 / 0.25 |
| CICIDS2018 | cnn-s42 | Hybrid Search targeted→Benign | 3200 | 15.00% ± 0.00% | 0.00% ± 0.00% | 15.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search targeted→Benign | 3200 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 0.00 / 0.00 / 0.00 |
| CICIDS2017 | mlp | Hybrid Search untargeted | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 11.06 / 11.06 / 11.06 |
| CICIDS2017 | cnn | Hybrid Search untargeted | 3200 | 36.67% ± 0.04% | 36.67% ± 0.04% | 0.00 ± 0.00 pp | 36.69 / 36.62 / 36.69 |
| CICIDS2017 | ft_transformer | Hybrid Search untargeted | 3200 | 0.50% ± 0.00% | 0.50% ± 0.00% | 0.00 ± 0.00 pp | 0.50 / 0.50 / 0.50 |
| CICIDS2018 | mlp-s42 | Hybrid Search untargeted | 3200 | 12.00% ± 0.03% | 0.19% ± 0.03% | 11.81 ± 0.03 pp | 0.19 / 0.22 / 0.16 |
| CICIDS2018 | cnn-s42 | Hybrid Search untargeted | 3200 | 15.31% ± 0.00% | 0.03% ± 0.00% | 15.28 ± 0.00 pp | 0.03 / 0.03 / 0.03 |
| CICIDS2018 | ft_transformer-s42 | Hybrid Search untargeted | 3200 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 0.00 / 0.00 / 0.00 |

## Statistical analysis (one planned McNemar per (dataset, victim), no Holm)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

| Dataset | Victim | n | Targeted-only valid | Untargeted-only valid | Δ Valid ASR (targeted − untargeted, pp) | Test | Statistic | McNemar p | Δ seed 2024 / 2026 (pp) | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 3200 | 0 | 0 | +0.00 | exact binomial McNemar |  | 1 | +0.00 / +0.00 | No significant difference (McNemar p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2017 | cnn | 3200 | 0 | 17 | -0.53 | exact binomial McNemar |  | 1.53e-05 | -0.50 / -0.53 | Hybrid Search targeted→Benign has lower Valid ASR than Hybrid Search untargeted by 0.53 pp (0 vs 17 discordant flows; McNemar p = 1.53e-05). |
| CICIDS2017 | ft_transformer | 3200 | 0 | 2 | -0.06 | exact binomial McNemar |  | 0.5 | -0.06 / -0.06 | No significant difference (McNemar p = 0.5; Δ = -0.06 pp, 0 vs 2 discordant flows). |
| CICIDS2018 | mlp-s42 | 3200 | 4 | 0 | +0.13 | exact binomial McNemar |  | 0.125 | +0.09 / +0.09 | No significant difference (McNemar p = 0.125; Δ = +0.13 pp, 4 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | 3200 | 0 | 1 | -0.03 | exact binomial McNemar |  | 1 | -0.03 / -0.03 | No significant difference (McNemar p = 1; Δ = -0.03 pp, 0 vs 1 discordant flows). |
| CICIDS2018 | ft_transformer-s42 | 3200 | 0 | 0 | +0.00 | exact binomial McNemar |  | 1 | +0.00 / +0.00 | No significant difference (McNemar p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |

Raw targeted and raw untargeted ASR are descriptive only. No Raw-Success test is run.

## Class-wise results

| Dataset | Victim | Class | Objective | Raw | Valid |
|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | Hybrid Search targeted→Benign | 28.88% ± 0.00% | 28.88% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search targeted→Benign | 13.25% ± 0.00% | 13.25% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search targeted→Benign | 2.12% ± 0.00% | 2.12% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search targeted→Benign | 49.33% ± 0.07% | 49.33% ± 0.07% |
| CICIDS2017 | cnn | DDoS | Hybrid Search targeted→Benign | 35.87% ± 0.00% | 35.87% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search targeted→Benign | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search targeted→Benign | 59.25% ± 0.00% | 59.25% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search targeted→Benign | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search targeted→Benign | 1.38% ± 0.00% | 1.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search targeted→Benign | 0.38% ± 0.12% | 0.29% ± 0.14% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search targeted→Benign | 4.75% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search targeted→Benign | 55.88% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search targeted→Benign | 4.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search targeted→Benign | 0.08% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search targeted→Benign | 1.25% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Hybrid Search untargeted | 28.88% ± 0.00% | 28.88% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Hybrid Search untargeted | 13.25% ± 0.00% | 13.25% ± 0.00% |
| CICIDS2017 | mlp | Recon | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Hybrid Search untargeted | 2.12% ± 0.00% | 2.12% ± 0.00% |
| CICIDS2017 | cnn | DoS | Hybrid Search untargeted | 49.29% ± 0.14% | 49.29% ± 0.14% |
| CICIDS2017 | cnn | DDoS | Hybrid Search untargeted | 37.75% ± 0.00% | 37.75% ± 0.00% |
| CICIDS2017 | cnn | Recon | Hybrid Search untargeted | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Hybrid Search untargeted | 59.25% ± 0.00% | 59.25% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Hybrid Search untargeted | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Hybrid Search untargeted | 1.62% ± 0.00% | 1.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Hybrid Search untargeted | 43.25% ± 0.13% | 0.29% ± 0.14% |
| CICIDS2018 | mlp-s42 | Recon | Hybrid Search untargeted | 4.75% ± 0.00% | 0.46% ± 0.07% |
| CICIDS2018 | mlp-s42 | BruteForce | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Hybrid Search untargeted | 57.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Hybrid Search untargeted | 4.12% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Hybrid Search untargeted | 0.08% ± 0.07% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Hybrid Search untargeted | 1.25% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Hybrid Search untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |

## Plots

- `plots/D1_raw_asr_targeted_vs_untargeted.png`
- `plots/D2_valid_asr_targeted_vs_untargeted.png`

## Interpretation

**Valid success hardly depends on the objective.** With Hybrid Search at p75, Valid ASR is:

| Victim | Targeted → Benign | Untargeted | Δ | Planned McNemar test (seed 42) |
|---|---|---|---|---|
| CICIDS2017 MLP | 11.06% | 11.06% | 0.00 pp | 0 vs 0 discordant, p = 1 |
| CICIDS2017 CNN | 36.15% | 36.67% | −0.53 pp | 0 vs 17, p = 1.5e-5 |
| CICIDS2017 FT-Transformer | 0.44% | 0.50% | −0.06 pp | 0 vs 2, p = 0.5 |
| CICIDS2018 MLP | 0.29% | 0.19% | +0.13 pp | 4 vs 0, p = 0.125 |
| CICIDS2018 CNN | 0.00% | 0.03% | −0.03 pp | 0 vs 1, p = 1 |
| CICIDS2018 FT-Transformer | 0.00% | 0.00% | 0.00 pp | 0 vs 0, p = 1 |

Only CICIDS2017 CNN shows a significant difference: 17 flows that the untargeted search
evades validly but the targeted search does not. That is a small effect (0.53 pp). Its sign
agrees on seeds 2024 (−0.50 pp) and 2026 (−0.53 pp). A valid evasion that PrimAttack finds for a
malicious flow almost always lands in the Benign class. On five of six victims, requiring
"→ Benign" instead of "any other class" costs nothing measurable.

**Raw success depends on the objective.** Raw untargeted ASR is much higher than raw targeted
ASR on CICIDS2018 MLP (12.00% vs 1.28%). The two are similar elsewhere (e.g. CICIDS2018 CNN
15.31% vs 15.00%; CICIDS2017 identical to within 0.52 pp). At seed 42, 343 of the 379 invalid raw
untargeted successes on CICIDS2018 MLP are DDoS flows pushed into the DoS class, all by padding,
and all fail the validator (`MINED_0001`). The Validity Gap is therefore 11.81 pp untargeted vs
0.99 pp targeted. Without the validator, the untargeted objective would look far stronger. With
it, the difference disappears. This is a direct example of why objective and validity must be
reported separately (Contribution 2).

**Reading.** The targeted-to-Benign setting is the operationally relevant evasion goal and
PrimAttack's primary contribution. On these victims it costs almost no valid success compared
with the easier untargeted goal. The weak valid results on CICIDS2018 at p75 and on
FT-Transformer are therefore not an artefact of choosing the harder targeted objective.
