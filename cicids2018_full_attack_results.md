# CSE-CIC-IDS-2018-DistriNet — full adversarial attack suite (no VAE, no IDR)

Same test as `cicids2017_full_attack_results.md`: canonical driver `scripts/run_full_adversarial_eval.py`, run 2026-09-26 under the CLAUDE.md **RUN POLICY** (no VAE latent attacks, no VAE IDR / True-IDSR). Raw per-row artifacts: `outputs/adv_campaign_noidr/cicids2018_distrinet/` (`cells.json`, `config.json`, `selection.json`, `artifacts/*.npz`); log `outputs/adv_campaign_noidr/run_cicids2018.log`. Paired statistical tables for both datasets (Wilson CIs, McNemar/Newcombe, Holm): `outputs/adv_campaign_noidr/analysis_tables.md` / `analysis.json` (`scripts/analyze_adversarial_campaign.py --root outputs/adv_campaign_noidr`).

**Run status:** 900 cells (9 victims × 4 classes × 25 attacks × 1 seed), 0 failures, pairing assertions PASSED. Wall time ≈1 h 55 min on CUDA (FT-Transformer victims ≈85% of it).

## Setup

| Item | Value |
|---|---|
| Dataset | CSE-CIC-IDS-2018-DistriNet, test split (chronological within source label; Benign/DoS/DDoS time-stratified down, Recon/BruteForce full — controlled, not natural prevalence) |
| Victims | 9 = {`mlp`, `cnn`, `ft_transformer`} × training seeds {42, 123, 2024} (`outputs/cicids2018distrinet/classifiers_multiseed/`) |
| Attack seed | `--match-victim-seed`: each replicate `<arch>-s<seed>` attacked once with its own training seed (same design as the earlier 2018 campaign) |
| Attack classes | DoS, DDoS, Recon, BruteForce (target for targeted attacks: Benign) |
| Eligibility / denominator | clean-correct rows; seeded uniform random sample (selection seed 42) of ≤800 per (victim, class) — all 800 used everywhere |
| PGD | L∞ ε=0.5, α=0.05, 40 steps |
| C&W | λ=1.0, κ=0.0, 60 iters, lr=0.01 |
| PrimAttack | as 2017 (40 steps, lr=0.1, 2 restarts); budgets from train-only `artifacts/primattack/budget_calibration_cicids2018.json` (p50/p75) + unbounded |
| CAPGD | native L2 ε=0.5, 10 steps; primitive: PrimAttack p75 joint box, 40 steps |
| FAB | AutoAttack `FABAttack_PT` untargeted, L2 ε=0.5, 100 iters, train min-max [0,1] box |
| Validity gate | validator_v2 `hybrid_valid`, `cicids2018_distrinet` profile; clean eligible rows are 100% hybrid-valid |

Eligible clean-correct pool per victim (before the 800 cap):

| Victim | DoS | DDoS | Recon | BruteForce |
|---|---|---|---|---|
| mlp-s42 | 29,967 | 29,828 | 13,383 | 14,129 |
| mlp-s123 | 29,966 | 29,877 | 13,389 | 14,129 |
| mlp-s2024 | 29,972 | 29,858 | 13,378 | 14,129 |
| cnn-s42 | 29,978 | 29,949 | 13,377 | 14,129 |
| cnn-s123 | 29,974 | 29,785 | 13,394 | 14,129 |
| cnn-s2024 | 29,952 | 29,807 | 13,391 | 14,129 |
| ft_transformer-s42 | 29,963 | 29,995 | 13,403 | 14,129 |
| ft_transformer-s123 | 29,961 | 30,000 | 13,402 | 14,129 |
| ft_transformer-s2024 | 29,969 | 30,000 | 13,400 | 14,129 |

## Metrics

Identical to the 2017 report. All rates are % of the clean-correct eligible set (n=3,200 rows per victim). Nested raw ⊇ valid ⊇ primitive-feasible ⊇ SP.

- **raw U / valid U** — untargeted evasion, without / with validator_v2 `hybrid_valid`.
- **raw TB / valid TB** — targeted-to-Benign success, without / with `hybrid_valid`.
- **SP** — valid TB ∧ primitive-feasible ∧ flow-semantics PASS (primitive attacks only; “–” = undefined).
- **dom.valid** — share of adversarial rows passing `hybrid_valid`; **feas.** — primitive-feasible share.

Classes are pooled within a victim. Architecture tables report the mean over the 3 training replicates; `±` is the across-replicate SD (training-seed + attack-seed variability, descriptive only — replicates are never pooled into one test).

## Headline — valid TB (%) per replicate

| Attack | mlp-s42 | mlp-s123 | mlp-s2024 | cnn-s42 | cnn-s123 | cnn-s2024 | ft_transformer-s42 | ft_transformer-s123 | ft_transformer-s2024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `cw_untargeted` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `pgd_tb` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `cw_tb` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `fab_untargeted` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `capgd_native` | 28.1 | 18.8 | 27.9 | 11.8 | 5.8 | 8.2 | 0.2 | 0.9 | 0.2 |
| `capgd_prim_p75` | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_p50` | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_p75` | 0.1 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_unb` | 22.7 | 25.2 | 25.3 | 10.4 | 21.2 | 2.5 | 0.1 | 0.1 | 0.1 |
| `prim_rand_joint_p75` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Headline — SP (%) per replicate

| Attack | mlp-s42 | mlp-s123 | mlp-s2024 | cnn-s42 | cnn-s123 | cnn-s2024 | ft_transformer-s42 | ft_transformer-s123 | ft_transformer-s2024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `capgd_prim_p75` | 0.0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_p50` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_p75` | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_unb` | 0.1 | 0.2 | 0.7 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_rand_joint_p75` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Headline — valid U (%) per replicate

| Attack | mlp-s42 | mlp-s123 | mlp-s2024 | cnn-s42 | cnn-s123 | cnn-s2024 | ft_transformer-s42 | ft_transformer-s123 | ft_transformer-s2024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `cw_untargeted` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `pgd_tb` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `cw_tb` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `fab_untargeted` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `capgd_native` | 28.2 | 21.2 | 28.0 | 15.2 | 8.2 | 10.5 | 0.2 | 0.9 | 0.2 |
| `capgd_prim_p75` | 0.1 | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_p50` | 0.2 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_p75` | 0.3 | 0.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_unb` | 32.4 | 41.0 | 37.8 | 10.4 | 21.2 | 2.5 | 0.1 | 0.1 | 0.1 |
| `prim_rand_joint_p75` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Full results per architecture (mean ± SD over 3 replicates)

### mlp

| Attack | family | raw U | valid U (± sd) | raw TB | valid TB (± sd) | SP | dom.valid | feas. | mean L2 (scaled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | input | 91.8 | 0.0 ± 0.0 | 69.5 | 0.0 ± 0.0 | – | 0.0 | – | 4.36 |
| `cw_untargeted` | input | 83.9 | 0.0 ± 0.0 | 64.3 | 0.0 ± 0.0 | – | 0.0 | – | 1.6 |
| `pgd_tb` | input | 85.3 | 0.0 ± 0.0 | 85.3 | 0.0 ± 0.0 | – | 0.0 | – | 4.32 |
| `cw_tb` | input | 80.1 | 0.0 ± 0.0 | 79.5 | 0.0 ± 0.0 | – | 0.0 | – | 1.64 |
| `fab_untargeted` | fab | 99.6 | 0.0 ± 0.0 | 99.3 | 0.0 ± 0.0 | – | 0.3 | – | 1.5e+03 |
| `capgd_native` | capgd | 78.6 | 25.8 ± 4.0 | 69.1 | 24.9 ± 5.3 | – | 26.9 | – | 1.61e+06 |
| `capgd_prim_p75` | capgd | 5.3 | 0.3 ± 0.5 | 0.5 | 0.0 ± 0.0 | 0.0 | 85.9 | 100.0 | 96.9 |
| `prim_search_joint_p50` | primattack | 5.0 | 0.1 ± 0.1 | 1.1 | 0.1 ± 0.0 | 0.0 | 65.9 | 100.0 | 301 |
| `prim_search_joint_p75` | primattack | 5.9 | 0.2 ± 0.2 | 1.2 | 0.1 ± 0.1 | 0.0 | 66.1 | 100.0 | 370 |
| `prim_search_joint_unb` | primattack | 41.2 | 37.1 ± 4.4 | 26.4 | 24.4 ± 1.5 | 0.4 | 89.5 | 100.0 | 2.33e+03 |
| `prim_search_timing_p50` | primattack | 1.8 | 1.8 ± 1.2 | 0.8 | 0.8 ± 0.1 | 0.1 | 100.0 | 100.0 | 302 |
| `prim_search_timing_p75` | primattack | 2.3 | 2.3 ± 1.7 | 0.9 | 0.9 ± 0.1 | 0.1 | 100.0 | 100.0 | 378 |
| `prim_search_timing_unb` | primattack | 41.3 | 41.3 ± 2.2 | 26.5 | 26.5 ± 1.9 | 0.6 | 99.9 | 100.0 | 2.62e+03 |
| `prim_search_padding_p50` | primattack | 2.1 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 65.4 | 100.0 | 12.7 |
| `prim_search_padding_p75` | primattack | 2.2 | 0.0 ± 0.0 | 0.1 | 0.0 ± 0.0 | 0.0 | 65.4 | 100.0 | 13.1 |
| `prim_search_padding_unb` | primattack | 2.3 | 0.0 ± 0.0 | 0.1 | 0.0 ± 0.0 | 0.0 | 65.4 | 100.0 | 13.1 |
| `prim_rand_joint_p50` | primattack | 5.2 | 0.0 ± 0.0 | 0.6 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 151 |
| `prim_rand_joint_p75` | primattack | 6.4 | 0.0 ± 0.0 | 0.6 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 172 |
| `prim_rand_joint_unb` | primattack | 26.0 | 1.6 ± 0.4 | 6.8 | 1.5 ± 0.4 | 0.0 | 27.6 | 100.0 | 2.8e+03 |
| `prim_rand_timing_p50` | primattack | 0.8 | 0.8 ± 0.8 | 0.4 | 0.4 ± 0.1 | 0.0 | 100.0 | 100.0 | 141 |
| `prim_rand_timing_p75` | primattack | 1.3 | 1.3 ± 1.3 | 0.4 | 0.4 ± 0.1 | 0.0 | 100.0 | 100.0 | 164 |
| `prim_rand_timing_unb` | primattack | 25.6 | 25.6 ± 3.8 | 9.9 | 9.9 ± 1.7 | 0.2 | 100.0 | 100.0 | 2.8e+03 |
| `prim_rand_padding_p50` | primattack | 1.1 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 27.7 | 100.0 | 13.2 |
| `prim_rand_padding_p75` | primattack | 1.1 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 27.7 | 100.0 | 13.4 |
| `prim_rand_padding_unb` | primattack | 1.1 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 27.7 | 100.0 | 13.4 |

### cnn

| Attack | family | raw U | valid U (± sd) | raw TB | valid TB (± sd) | SP | dom.valid | feas. | mean L2 (scaled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | input | 99.9 | 0.0 ± 0.0 | 95.0 | 0.0 ± 0.0 | – | 0.0 | – | 3.92 |
| `cw_untargeted` | input | 99.1 | 0.0 ± 0.0 | 83.4 | 0.0 ± 0.0 | – | 0.0 | – | 1.22 |
| `pgd_tb` | input | 99.9 | 0.0 ± 0.0 | 99.9 | 0.0 ± 0.0 | – | 0.0 | – | 3.76 |
| `cw_tb` | input | 99.1 | 0.0 ± 0.0 | 99.1 | 0.0 ± 0.0 | – | 0.0 | – | 1.23 |
| `fab_untargeted` | fab | 91.9 | 0.0 ± 0.0 | 74.4 | 0.0 ± 0.0 | – | 8.1 | – | 3.21e+03 |
| `capgd_native` | capgd | 64.9 | 11.3 ± 3.6 | 39.2 | 8.6 ± 3.0 | – | 13.0 | – | 1.55e+06 |
| `capgd_prim_p75` | capgd | 7.8 | 0.0 ± 0.0 | 7.4 | 0.0 ± 0.0 | 0.0 | 61.3 | 100.0 | 86.3 |
| `prim_search_joint_p50` | primattack | 15.5 | 0.0 ± 0.0 | 15.4 | 0.0 ± 0.0 | 0.0 | 23.8 | 100.0 | 210 |
| `prim_search_joint_p75` | primattack | 16.2 | 0.0 ± 0.0 | 16.2 | 0.0 ± 0.0 | 0.0 | 23.8 | 100.0 | 221 |
| `prim_search_joint_unb` | primattack | 41.5 | 11.4 ± 9.4 | 41.5 | 11.4 ± 9.4 | 0.0 | 34.7 | 100.0 | 2.01e+03 |
| `prim_search_timing_p50` | primattack | 0.1 | 0.1 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 99.9 | 100.0 | 217 |
| `prim_search_timing_p75` | primattack | 0.1 | 0.1 ± 0.1 | 0.0 | 0.0 ± 0.0 | 0.0 | 99.9 | 100.0 | 264 |
| `prim_search_timing_unb` | primattack | 25.9 | 25.9 ± 0.3 | 25.9 | 25.9 ± 0.3 | 0.0 | 99.9 | 100.0 | 5.95e+03 |
| `prim_search_padding_p50` | primattack | 6.6 | 0.0 ± 0.0 | 6.6 | 0.0 ± 0.0 | 0.0 | 23.8 | 100.0 | 24.4 |
| `prim_search_padding_p75` | primattack | 6.7 | 0.0 ± 0.0 | 6.7 | 0.0 ± 0.0 | 0.0 | 23.8 | 100.0 | 24.6 |
| `prim_search_padding_unb` | primattack | 6.7 | 0.0 ± 0.0 | 6.7 | 0.0 ± 0.0 | 0.0 | 23.8 | 100.0 | 24.6 |
| `prim_rand_joint_p50` | primattack | 6.2 | 0.0 ± 0.0 | 6.1 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 151 |
| `prim_rand_joint_p75` | primattack | 7.4 | 0.0 ± 0.0 | 7.1 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 172 |
| `prim_rand_joint_unb` | primattack | 25.8 | 3.5 ± 0.1 | 25.7 | 3.5 ± 0.1 | 0.0 | 27.6 | 100.0 | 2.8e+03 |
| `prim_rand_timing_p50` | primattack | 0.0 | 0.0 ± 0.1 | 0.0 | 0.0 ± 0.0 | 0.0 | 100.0 | 100.0 | 141 |
| `prim_rand_timing_p75` | primattack | 0.1 | 0.1 ± 0.1 | 0.0 | 0.0 ± 0.0 | 0.0 | 100.0 | 100.0 | 164 |
| `prim_rand_timing_unb` | primattack | 24.1 | 24.1 ± 0.5 | 24.1 | 24.1 ± 0.5 | 0.0 | 100.0 | 100.0 | 2.79e+03 |
| `prim_rand_padding_p50` | primattack | 3.0 | 0.0 ± 0.0 | 3.0 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 13.2 |
| `prim_rand_padding_p75` | primattack | 3.1 | 0.0 ± 0.0 | 3.1 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 13.4 |
| `prim_rand_padding_unb` | primattack | 3.1 | 0.0 ± 0.0 | 3.1 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 13.4 |

### ft_transformer

| Attack | family | raw U | valid U (± sd) | raw TB | valid TB (± sd) | SP | dom.valid | feas. | mean L2 (scaled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | input | 96.7 | 0.0 ± 0.0 | 69.9 | 0.0 ± 0.0 | – | 0.0 | – | 3.98 |
| `cw_untargeted` | input | 68.3 | 0.0 ± 0.0 | 58.2 | 0.0 ± 0.0 | – | 0.0 | – | 0.646 |
| `pgd_tb` | input | 94.5 | 0.0 ± 0.0 | 92.7 | 0.0 ± 0.0 | – | 0.0 | – | 3.97 |
| `cw_tb` | input | 79.7 | 0.0 ± 0.0 | 79.6 | 0.0 ± 0.0 | – | 0.0 | – | 0.785 |
| `fab_untargeted` | fab | 34.1 | 0.0 ± 0.0 | 25.0 | 0.0 ± 0.0 | – | 65.8 | – | 7.63e+05 |
| `capgd_native` | capgd | 6.1 | 0.4 ± 0.4 | 4.8 | 0.4 ± 0.4 | – | 4.5 | – | 2.41e+06 |
| `capgd_prim_p75` | capgd | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 65.5 | 100.0 | 102 |
| `prim_search_joint_p50` | primattack | 0.1 | 0.0 ± 0.0 | 0.1 | 0.0 ± 0.0 | 0.0 | 25.1 | 100.0 | 287 |
| `prim_search_joint_p75` | primattack | 0.1 | 0.0 ± 0.0 | 0.1 | 0.0 ± 0.0 | 0.0 | 24.9 | 100.0 | 305 |
| `prim_search_joint_unb` | primattack | 0.8 | 0.1 ± 0.0 | 0.8 | 0.1 ± 0.0 | 0.0 | 25.1 | 100.0 | 2.16e+03 |
| `prim_search_timing_p50` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 99.9 | 100.0 | 256 |
| `prim_search_timing_p75` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 99.9 | 100.0 | 277 |
| `prim_search_timing_unb` | primattack | 0.3 | 0.3 ± 0.2 | 0.3 | 0.3 ± 0.2 | 0.0 | 99.9 | 100.0 | 2.47e+03 |
| `prim_search_padding_p50` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 25.5 | 100.0 | 25.3 |
| `prim_search_padding_p75` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 25.5 | 100.0 | 25.6 |
| `prim_search_padding_unb` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 25.5 | 100.0 | 25.6 |
| `prim_rand_joint_p50` | primattack | 0.1 | 0.0 ± 0.0 | 0.1 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 150 |
| `prim_rand_joint_p75` | primattack | 0.1 | 0.0 ± 0.0 | 0.1 | 0.0 ± 0.0 | 0.0 | 27.6 | 100.0 | 172 |
| `prim_rand_joint_unb` | primattack | 0.4 | 0.1 ± 0.0 | 0.4 | 0.1 ± 0.0 | 0.0 | 27.6 | 100.0 | 2.79e+03 |
| `prim_rand_timing_p50` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 100.0 | 100.0 | 141 |
| `prim_rand_timing_p75` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 100.0 | 100.0 | 163 |
| `prim_rand_timing_unb` | primattack | 0.2 | 0.2 ± 0.1 | 0.2 | 0.2 ± 0.1 | 0.0 | 100.0 | 100.0 | 2.79e+03 |
| `prim_rand_padding_p50` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 27.7 | 100.0 | 13.2 |
| `prim_rand_padding_p75` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 27.7 | 100.0 | 13.4 |
| `prim_rand_padding_unb` | primattack | 0.0 | 0.0 ± 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 27.7 | 100.0 | 13.4 |

## Per-class breakdown (valid U, mean over replicates)

**`capgd_native`** — valid U / valid TB

| Arch | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|
| mlp | 49.5 / 46.4 | 31.2 / 30.9 | 19.2 / 19.2 | 3.2 / 3.2 |
| cnn | 1.6 / 1.6 | 11.7 / 4.1 | 12.1 / 12.0 | 19.6 / 16.8 |
| ft_transformer | 0.1 / 0.0 | 0.1 / 0.1 | 0.5 / 0.5 | 1.1 / 1.1 |

**`prim_search_joint_p75`** — valid U / valid TB

| Arch | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|
| mlp | 0.0 / 0.0 | 0.5 / 0.1 | 0.3 / 0.3 | 0.0 / 0.0 |
| cnn | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| ft_transformer | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

**`prim_search_joint_unb`** — valid U / valid TB

| Arch | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|
| mlp | 0.0 / 0.0 | 52.2 / 1.5 | 2.6 / 2.6 | 93.5 / 93.5 |
| cnn | 0.0 / 0.0 | 0.0 / 0.0 | 1.5 / 1.5 | 43.9 / 43.9 |
| ft_transformer | 0.0 / 0.0 | 0.0 / 0.0 | 0.4 / 0.4 | 0.0 / 0.0 |

## Cross-dataset comparison vs CICIDS2017 (from the analyzer)

Independent samples; 2017 victim vs 2018 replicate `-s42`, seed 42; metric = valid untargeted evasion; Newcombe CI + Fisher exact, Holm over attacks. FAB is not in the analyzer's cross-dataset roster.

| arch | attack | metric | 2017 % | 2018 % | RD pp [Newcombe 95% CI] | Fisher p | p_Holm |
|---|---|---|---|---|---|---|---|
| MLP | `pgd_untargeted` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| MLP | `cw_untargeted` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| MLP | `pgd_tb` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| MLP | `cw_tb` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| MLP | `prim_search_joint_p50` | valid_evasion | 6.22 | 0.19 | +6.03 [+5.22, +6.93] | 1.93e-52 | 1.54e-51 |
| MLP | `prim_search_joint_p75` | valid_evasion | 11.03 | 0.34 | +10.69 [+9.61, +11.83] | 1.11e-93 | 1.11e-92 |
| MLP | `prim_search_joint_unb` | valid_evasion | 45.88 | 32.38 | +13.50 [+11.12, +15.86] | 1.95e-28 | 1.17e-27 |
| MLP | `prim_rand_joint_p75` | valid_evasion | 2.25 | 0.00 | +2.25 [+1.78, +2.82] | 2.83e-22 | 1.41e-21 |
| MLP | `capgd_native` | valid_evasion | 11.00 | 28.16 | -17.16 [-19.05, -15.25] | 1.34e-68 | 1.20e-67 |
| MLP | `capgd_prim_p75` | valid_evasion | 3.72 | 0.09 | +3.62 [+3.00, +4.34] | 3.93e-32 | 2.75e-31 |
| CNN | `pgd_untargeted` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| CNN | `cw_untargeted` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| CNN | `pgd_tb` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| CNN | `cw_tb` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| CNN | `prim_search_joint_p50` | valid_evasion | 12.81 | 0.00 | +12.81 [+11.69, +14.02] | 6.24e-130 | 5.00e-129 |
| CNN | `prim_search_joint_p75` | valid_evasion | 35.75 | 0.00 | +35.75 [+34.10, +37.43] | <1e-300 | <1e-300 |
| CNN | `prim_search_joint_unb` | valid_evasion | 73.19 | 10.41 | +62.78 [+60.87, +64.60] | <1e-300 | <1e-300 |
| CNN | `prim_rand_joint_p75` | valid_evasion | 5.38 | 0.00 | +5.38 [+4.64, +6.21] | 3.15e-53 | 1.89e-52 |
| CNN | `capgd_native` | valid_evasion | 21.97 | 15.16 | +6.81 [+4.91, +8.71] | 2.70e-12 | 1.35e-11 |
| CNN | `capgd_prim_p75` | valid_evasion | 11.59 | 0.00 | +11.59 [+10.52, +12.75] | 4.70e-117 | 3.29e-116 |
| FT-Transformer | `pgd_untargeted` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| FT-Transformer | `cw_untargeted` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| FT-Transformer | `pgd_tb` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| FT-Transformer | `cw_tb` | valid_evasion | 0.00 | 0.00 | +0.00 [-0.12, +0.12] | 1.000 | 1.000 |
| FT-Transformer | `prim_search_joint_p50` | valid_evasion | 0.22 | 0.00 | +0.22 [+0.05, +0.45] | 0.016 | 0.078 |
| FT-Transformer | `prim_search_joint_p75` | valid_evasion | 0.47 | 0.00 | +0.47 [+0.25, +0.77] | 6.00e-05 | 4.80e-04 |
| FT-Transformer | `prim_search_joint_unb` | valid_evasion | 0.84 | 0.09 | +0.75 [+0.43, +1.14] | 8.07e-06 | 7.26e-05 |
| FT-Transformer | `prim_rand_joint_p75` | valid_evasion | 0.38 | 0.00 | +0.38 [+0.17, +0.65] | 4.83e-04 | 0.003 |
| FT-Transformer | `capgd_native` | valid_evasion | 4.97 | 0.22 | +4.75 [+4.01, +5.57] | 2.15e-39 | 2.15e-38 |
| FT-Transformer | `capgd_prim_p75` | valid_evasion | 0.25 | 0.00 | +0.25 [+0.08, +0.49] | 0.008 | 0.047 |

## Diagnostic — why PrimAttack validity differs from 2017

Per-class shares (%) over all 9 replicates, from the per-row artifacts. `hard` = validator_v2 `hard_structural_valid`; `dom` = full `hybrid_valid`; `feas` = primitive-feasible; `sem` = flow-semantics PASS.

| Attack | Class | dom | hard | feas | sem | raw TB | valid TB |
|---|---|---:|---:|---:|---:|---:|---:|
| `prim_rand_joint_p75` | BruteForce | 13.9 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| `prim_rand_joint_p75` | DDoS | 0.8 | 100.0 | 100.0 | 95.0 | 8.8 | 0.0 |
| `prim_rand_joint_p75` | DoS | 1.5 | 100.0 | 100.0 | 92.8 | 0.0 | 0.0 |
| `prim_rand_joint_p75` | Recon | 94.3 | 100.0 | 100.0 | 0.0 | 1.6 | 0.0 |
| `prim_search_joint_p75` | BruteForce | 22.3 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_joint_p75` | DDoS | 1.3 | 100.0 | 100.0 | 89.0 | 20.5 | 0.0 |
| `prim_search_joint_p75` | DoS | 35.1 | 100.0 | 100.0 | 92.8 | 0.1 | 0.0 |
| `prim_search_joint_p75` | Recon | 94.4 | 100.0 | 100.0 | 0.0 | 2.8 | 0.1 |
| `prim_search_padding_p75` | BruteForce | 22.4 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_padding_p75` | DDoS | 0.7 | 100.0 | 100.0 | 95.0 | 8.2 | 0.0 |
| `prim_search_padding_p75` | DoS | 35.6 | 100.0 | 100.0 | 92.8 | 0.1 | 0.0 |
| `prim_search_padding_p75` | Recon | 94.4 | 100.0 | 100.0 | 0.0 | 0.8 | 0.0 |
| `prim_search_timing_p75` | BruteForce | 100.0 | 100.0 | 100.0 | 0.0 | 0.0 | 0.0 |
| `prim_search_timing_p75` | DDoS | 100.0 | 100.0 | 100.0 | 89.8 | 0.2 | 0.2 |
| `prim_search_timing_p75` | DoS | 100.0 | 100.0 | 100.0 | 92.8 | 0.0 | 0.0 |
| `prim_search_timing_p75` | Recon | 99.8 | 100.0 | 100.0 | 0.0 | 1.0 | 1.0 |
| `prim_search_timing_unb` | BruteForce | 100.0 | 100.0 | 100.0 | 0.0 | 65.4 | 65.4 |
| `prim_search_timing_unb` | DDoS | 100.0 | 100.0 | 100.0 | 85.8 | 1.1 | 1.1 |
| `prim_search_timing_unb` | DoS | 100.0 | 100.0 | 100.0 | 92.8 | 0.0 | 0.0 |
| `prim_search_timing_unb` | Recon | 99.8 | 100.0 | 100.0 | 0.0 | 3.8 | 3.8 |

## Observations

- **Unconstrained PGD / C&W / FAB**: 22–100% raw untargeted evasion per replicate (FAB on FT-Transformer is the weakest, 34.1% mean) but **0.0% valid** evasion on every replicate — same as 2017.
- **CAPGD native** is the strongest valid attack on 2018 at the calibrated budgets: valid U mlp 25.8, cnn 11.3, FT 0.5 (mean over replicates).
- **PrimAttack at the calibrated budgets (p50/p75) is ≈0 on every 2018 victim** (joint p75 valid TB: mlp 0.1, cnn 0.0, FT 0.0), versus 11.0 / 35.2 / 0.4 on 2017. Only the unbounded budget gets through (joint unb valid TB: mlp 24.4, cnn 11.4, FT 0.1).
- **Padding is what breaks validity on 2018.** Every primitive output passes `hard_structural_valid` and is primitive-feasible (100%), but any configuration with padding fails full `hybrid_valid` for most DoS/DDoS/BruteForce rows (joint p75 dom: DoS 35.1, DDoS 1.3, BruteForce 22.3, Recon 94.4). Timing-only outputs stay ≥99.8% valid in every class. The rejection therefore comes from the protocol/mined-rule part of the 2018 validator_v2 profile, not the hard structural checks [INFERENCE: hybrid = hard ∧ protocol ∧ mined; which rule fires was not logged by the driver and was not investigated].
- **Timing-only unbounded** is the best PrimAttack configuration on 2018 (valid TB mlp 26.5, cnn 25.9, FT 0.3), driven almost entirely by BruteForce (65.4% valid TB pooled over replicates).
- **SP ≈0 everywhere on 2018** (max 0.65% mean, timing-only unbounded on mlp). The flow-semantics check passes 0% of Recon and BruteForce rows for every primitive attack — also true on 2017 — and on 2018 the successful valid rows are concentrated in exactly those classes, so they fail SP.
- **Search vs random-feasible**: random-feasible joint p75 is 0.0% valid TB on every replicate; at the calibrated budgets search is not better in practice (both ≈0).
- **FT-Transformer** stays near-immune to every validity-constrained attack (valid TB ≤0.9% on every replicate; max = CAPGD native), as on 2017.
- **Replicate variability is large where attacks succeed** (up to 9.4 pp SD in valid TB across the 3 training replicates for PrimAttack), so per-replicate numbers matter; see the per-replicate headline tables.
- **Cross-dataset (s42, seed 42)**: PrimAttack valid evasion is lower on 2018 than 2017 for every architecture and budget (Holm-adjusted p <0.01 for every PrimAttack row except FT search p50, p=0.078); CAPGD native is significantly *higher* on 2018 for MLP (28.2 vs 11.0) and lower for CNN and FT.

## Claim boundary

Feature-space proxies on aggregate CICFlowMeter statistics; no PCAP edited or replayed; no packet-level realizability, malicious-functionality preservation, or in-distribution realism is claimed (IDR not computed). 2018 splits are controlled (time-stratified down-sampling of Benign/DoS/DDoS), not natural prevalence; header-length features are int16-wrapped (kept and flagged in preprocessing). One attack seed per replicate; statistics are reported per victim replicate.
