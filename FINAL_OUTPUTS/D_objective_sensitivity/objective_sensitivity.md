# Final Experiment D — Objective sensitivity (targeted → Benign vs untargeted)

PrimAttack with the selected optimizer (Prim-PGD), joint mode, p75 budget, on identical flows and seeds. Targeted success = prediction == Benign. Untargeted success = prediction ≠ source class. Both arms use the same validator gate and the same incumbent rule. Only the objective margin differs. The targeted arm is the Exp B cell and the untargeted arm is the Exp A PrimAttack cell.

## Results

| Dataset | Victim | Objective | n/seed | Raw ASR | Valid ASR | Validity Gap | Valid per seed (42/2024/2026, %) |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | Prim-PGD targeted→Benign | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 4.09 / 4.09 / 4.09 |
| CICIDS2017 | cnn | Prim-PGD targeted→Benign | 3200 | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp | 13.25 / 13.25 / 13.25 |
| CICIDS2017 | ft_transformer | Prim-PGD targeted→Benign | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12 / 0.12 / 0.12 |
| CICIDS2018 | mlp-s42 | Prim-PGD targeted→Benign | 3200 | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 ± 0.00 pp | 0.78 / 0.78 / 0.78 |
| CICIDS2018 | cnn-s42 | Prim-PGD targeted→Benign | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD targeted→Benign | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 |
| CICIDS2017 | mlp | Prim-PGD untargeted | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 4.09 / 4.09 / 4.09 |
| CICIDS2017 | cnn | Prim-PGD untargeted | 3200 | 13.47% ± 0.00% | 13.47% ± 0.00% | 0.00 ± 0.00 pp | 13.47 / 13.47 / 13.47 |
| CICIDS2017 | ft_transformer | Prim-PGD untargeted | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12 / 0.12 / 0.12 |
| CICIDS2018 | mlp-s42 | Prim-PGD untargeted | 3200 | 2.53% ± 0.00% | 2.53% ± 0.00% | 0.00 ± 0.00 pp | 2.53 / 2.53 / 2.53 |
| CICIDS2018 | cnn-s42 | Prim-PGD untargeted | 3200 | 1.16% ± 0.00% | 1.16% ± 0.00% | 0.00 ± 0.00 pp | 1.16 / 1.16 / 1.16 |
| CICIDS2018 | ft_transformer-s42 | Prim-PGD untargeted | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 |

## Statistical analysis (one planned McNemar per (dataset, victim), no Holm)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

| Dataset | Victim | n | Targeted-only valid | Untargeted-only valid | Δ Valid ASR (targeted − untargeted, pp) | Test | Statistic | McNemar p | Δ seed 2024 / 2026 (pp) | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 3200 | 0 | 0 | +0.00 | exact binomial McNemar |  | 1 | +0.00 / +0.00 | No significant difference (McNemar p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2017 | cnn | 3200 | 0 | 7 | -0.22 | exact binomial McNemar |  | 0.0156 | -0.22 / -0.22 | Prim-PGD targeted→Benign has lower Valid ASR than Prim-PGD untargeted by 0.22 pp (0 vs 7 discordant flows; McNemar p = 0.0156). |
| CICIDS2017 | ft_transformer | 3200 | 0 | 0 | +0.00 | exact binomial McNemar |  | 1 | +0.00 / +0.00 | No significant difference (McNemar p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |
| CICIDS2018 | mlp-s42 | 3200 | 0 | 56 | -1.75 | asymptotic McNemar chi-square (continuity corrected) | 54.02 | 1.99e-13 | -1.75 / -1.75 | Prim-PGD targeted→Benign has lower Valid ASR than Prim-PGD untargeted by 1.75 pp (0 vs 56 discordant flows; McNemar p = 1.99e-13). |
| CICIDS2018 | cnn-s42 | 3200 | 0 | 37 | -1.16 | asymptotic McNemar chi-square (continuity corrected) | 35.03 | 3.25e-09 | -1.16 / -1.16 | Prim-PGD targeted→Benign has lower Valid ASR than Prim-PGD untargeted by 1.16 pp (0 vs 37 discordant flows; McNemar p = 3.25e-09). |
| CICIDS2018 | ft_transformer-s42 | 3200 | 0 | 0 | +0.00 | exact binomial McNemar |  | 1 | +0.00 / +0.00 | No significant difference (McNemar p = 1; Δ = +0.00 pp, 0 vs 0 discordant flows). |

Raw targeted and raw untargeted ASR are descriptive only. No Raw-Success test is run.

## Class-wise results

| Dataset | Victim | Class | Objective | Raw | Valid |
|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | Prim-PGD targeted→Benign | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD targeted→Benign | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD targeted→Benign | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD targeted→Benign | 20.50% ± 0.00% | 20.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD targeted→Benign | 31.50% ± 0.00% | 31.50% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD targeted→Benign | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD targeted→Benign | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD targeted→Benign | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD targeted→Benign | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD targeted→Benign | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD targeted→Benign | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD targeted→Benign | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | DoS | Prim-PGD untargeted | 5.12% ± 0.00% | 5.12% ± 0.00% |
| CICIDS2017 | mlp | DDoS | Prim-PGD untargeted | 10.75% ± 0.00% | 10.75% ± 0.00% |
| CICIDS2017 | mlp | Recon | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | mlp | BruteForce | Prim-PGD untargeted | 0.50% ± 0.00% | 0.50% ± 0.00% |
| CICIDS2017 | cnn | DoS | Prim-PGD untargeted | 20.50% ± 0.00% | 20.50% ± 0.00% |
| CICIDS2017 | cnn | DDoS | Prim-PGD untargeted | 32.12% ± 0.00% | 32.12% ± 0.00% |
| CICIDS2017 | cnn | Recon | Prim-PGD untargeted | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2017 | cnn | BruteForce | Prim-PGD untargeted | 0.88% ± 0.00% | 0.88% ± 0.00% |
| CICIDS2017 | ft_transformer | DoS | Prim-PGD untargeted | 0.12% ± 0.00% | 0.12% ± 0.00% |
| CICIDS2017 | ft_transformer | DDoS | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | Recon | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | BruteForce | Prim-PGD untargeted | 0.38% ± 0.00% | 0.38% ± 0.00% |
| CICIDS2018 | mlp-s42 | DoS | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | DDoS | Prim-PGD untargeted | 7.50% ± 0.00% | 7.50% ± 0.00% |
| CICIDS2018 | mlp-s42 | Recon | Prim-PGD untargeted | 2.62% ± 0.00% | 2.62% ± 0.00% |
| CICIDS2018 | mlp-s42 | BruteForce | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DoS | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | DDoS | Prim-PGD untargeted | 4.62% ± 0.00% | 4.62% ± 0.00% |
| CICIDS2018 | cnn-s42 | Recon | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | BruteForce | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DoS | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | DDoS | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | Recon | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | Prim-PGD untargeted | 0.00% ± 0.00% | 0.00% ± 0.00% |

## Plots

- `plots/D1_raw_asr_targeted_vs_untargeted.png`
- `plots/D2_valid_asr_targeted_vs_untargeted.png`

## Interpretation

**Targeted→Benign vs untargeted (Prim-PGD, p75, same flows and seeds).** On CICIDS2017 the two
objectives give almost the same Valid ASR: MLP 4.09% vs 4.09% and FT-Transformer 0.12% vs 0.12%
(identical success sets, 0 discordant flows), CNN 13.25% vs 13.47% (7 flows succeed only
untargeted; McNemar p = 0.016). There, the flows that timing moves out of their class almost all
move to Benign (seed 42: 423 of 431 untargeted CNN successes; the rest are DDoS↔Recon). On
CICIDS2018 the untargeted objective is clearly easier: MLP 0.78% vs 2.53% (56 vs 0 discordant
flows, p = 2.0e-13) and CNN 0.00% vs 1.16% (37 vs 0, p = 3.3e-9). These extra untargeted
successes are DDoS flows pushed into DoS, i.e. into another attack class, not into Benign; the 25
targeted successes on the MLP (4 DDoS, 21 Recon) reach Benign under both objectives.
FT-Transformer has no valid success under either objective on CICIDS2018.

**Optimizer scope.** Prim-PGD was selected only after it tied Hybrid exactly on Exp B's targeted
success outcome and won the evaluation-count tie-break. That tie is expected because almost all
rows are timing-only: Hybrid's exact-padding phase is skipped and both optimizers use closely
related sign-momentum timing updates. Exp D's targeted arm would therefore have the same
flow-level outcome under Hybrid in the observed Exp B artifacts. The untargeted arm was rerun
only with the selected Prim-PGD, so the targeted/untargeted conclusion is formally a Prim-PGD
result rather than an optimizer-general claim.

**Direction of the effect.** No flow is valid-targeted-only on any victim (targeted-only = 0
everywhere), consistent with Benign being one of the classes an untargeted success may reach.

**Validity.** Both objectives have a Validity Gap of 0.00 pp on every victim. The objective
changes which flows succeed, not whether the successes are valid.

**Reading.** Reporting untargeted evasion (Exp A) and targeted evasion (Exp B/C) separately
matters only on CICIDS2018, where up to 1.75 pp of the untargeted Valid ASR is class-to-class
confusion between attack categories rather than evasion to Benign.
