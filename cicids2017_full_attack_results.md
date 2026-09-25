# CICIDS2017-DistriNet — full adversarial attack suite (no VAE, no IDR)

Canonical driver `scripts/run_full_adversarial_eval.py`, run 2026-09-26 under the CLAUDE.md **RUN POLICY**: no VAE latent attacks were run and no VAE in-distribution (IDR) / True-IDSR metric was computed. Raw per-row artifacts: `outputs/adv_campaign_noidr/cicids2017_distrinet/` (`cells.json`, `config.json`, `selection.json`, `artifacts/*.npz`); log `outputs/adv_campaign_noidr/run_cicids2017.log`.

**Run status:** 900 cells (3 victims × 4 classes × 25 attacks × 3 seeds), 0 failures, pairing assertions PASSED (every attack of a victim hit the identical rows in identical order).

## Setup

| Item | Value |
|---|---|
| Dataset | CICIDS2017-DistriNet, test split (chronological within source label, 70/15/15) |
| Victims | `mlp`, `cnn`, `ft_transformer` category heads (`outputs/cicids2017distrinet/models/*_category.pt`) |
| Attack classes | DoS, DDoS, Recon, BruteForce (target for targeted attacks: Benign) |
| Eligibility / denominator | clean-correct rows; seeded uniform random sample (selection seed 42) of ≤800 per (victim, class) — all 800 used everywhere |
| Attack seeds | 42, 123, 2024 |
| PGD | L∞ ε=0.5, α=0.05, 40 steps (scaled feature space, all 79 features) |
| C&W | λ=1.0, κ=0.0, 60 iters, lr=0.01 |
| PrimAttack | search (exact padding enumeration + projected timing refinement), 40 steps, lr=0.1, 2 restarts; random-feasible control; modes joint/timing/padding; budgets p50/p75 (train-calibrated) and unbounded |
| CAPGD | native: L2 ε=0.5, 10 steps; primitive: restricted to PrimAttack p75 joint box, 40 steps (TabularBench, vendored) |
| FAB | AutoAttack `FABAttack_PT` untargeted, L2 ε=0.5, 100 iters, train min-max [0,1] box |
| Validity gate | validator_v2 `hybrid_valid` (SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED); clean eligible rows are 100% hybrid-valid |
| Device | CUDA (thesis env, Python 3.11, torch 2.5.1+cu121) |

Eligible clean-correct pool per victim (before the 800 cap):

| Victim | DoS | DDoS | Recon | BruteForce |
|---|---|---|---|---|
| mlp | 25,583 | 14,243 | 23,771 | 1,022 |
| cnn | 25,540 | 14,244 | 23,760 | 1,018 |
| ft_transformer | 25,522 | 14,257 | 23,788 | 1,026 |

## Metrics

All rates are % of the clean-correct eligible set (n=3,200 rows per victim = 4 classes × 800). Nested: raw ⊇ valid ⊇ primitive-feasible ⊇ SP.

- **raw U** — untargeted evasion (prediction ≠ true class).
- **valid U** — raw U ∧ validator_v2 `hybrid_valid`.
- **raw TB** — targeted success (prediction = Benign).
- **valid TB** — raw TB ∧ `hybrid_valid`.
- **SP** — valid TB ∧ primitive-feasible ∧ flow-semantics PASS (PrimAttack / primitive-CAPGD only; “–” = undefined).
- **dom.valid** — share of adversarial rows passing `hybrid_valid` regardless of success.
- **feas.** — primitive-feasible share (primitive attacks only).

Classes are pooled within a victim (equal n per class); victims and seeds are never pooled. Values are the mean over the 3 attack seeds; `±` is the across-seed SD (run-to-run variability, not a CI).

## Headline (mean over seeds)

| Attack | mlp valid TB | cnn valid TB | ft_transformer valid TB | mlp SP | cnn SP | ft_transformer SP |
|---|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | 0.0 | 0.0 | 0.0 | – | – | – |
| `cw_untargeted` | 0.0 | 0.0 | 0.0 | – | – | – |
| `pgd_tb` | 0.0 | 0.0 | 0.0 | – | – | – |
| `cw_tb` | 0.0 | 0.0 | 0.0 | – | – | – |
| `fab_untargeted` | 0.0 | 0.0 | 0.0 | – | – | – |
| `capgd_native` | 1.8 | 13.4 | 1.8 | – | – | – |
| `capgd_prim_p75` | 3.4 | 10.7 | 0.3 | 3.1 | 4.5 | 0.0 |
| `prim_search_joint_p50` | 6.3 | 12.7 | 0.2 | 6.0 | 12.0 | 0.0 |
| `prim_search_joint_p75` | 11.0 | 35.2 | 0.4 | 10.5 | 19.6 | 0.0 |
| `prim_search_joint_unb` | 41.5 | 70.4 | 0.8 | 35.5 | 40.9 | 0.1 |
| `prim_rand_joint_p75` | 2.2 | 5.5 | 0.3 | 2.0 | 4.9 | 0.0 |

## Full results per victim

### mlp

| Attack | family | raw U | valid U | raw TB | valid TB (± sd) | SP | dom.valid | feas. | mean L2 (scaled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | input | 100.0 | 0.0 | 93.1 | 0.0 ± 0.0 | – | 0.0 | – | 4.38 |
| `cw_untargeted` | input | 99.9 | 0.0 | 97.0 | 0.0 ± 0.0 | – | 0.0 | – | 1.32 |
| `pgd_tb` | input | 100.0 | 0.0 | 100.0 | 0.0 ± 0.0 | – | 0.0 | – | 4.28 |
| `cw_tb` | input | 100.0 | 0.0 | 100.0 | 0.0 ± 0.0 | – | 0.0 | – | 1.32 |
| `fab_untargeted` | fab | 89.4 | 0.0 | 76.2 | 0.0 ± 0.0 | – | 10.6 | – | 2.26e+03 |
| `capgd_native` | capgd | 94.7 | 10.8 | 73.1 | 1.8 ± 0.2 | – | 11.1 | – | 8.75e+04 |
| `capgd_prim_p75` | capgd | 3.4 | 3.4 | 3.4 | 3.4 ± 0.4 | 3.1 | 100.0 | 100.0 | 452 |
| `prim_search_joint_p50` | primattack | 6.3 | 6.3 | 6.3 | 6.3 ± 0.0 | 6.0 | 100.0 | 100.0 | 2.32e+03 |
| `prim_search_joint_p75` | primattack | 11.0 | 11.0 | 11.0 | 11.0 ± 0.0 | 10.5 | 100.0 | 100.0 | 3.45e+03 |
| `prim_search_joint_unb` | primattack | 45.8 | 45.8 | 41.5 | 41.5 ± 0.1 | 35.5 | 100.0 | 100.0 | 1.8e+04 |
| `prim_search_timing_p50` | primattack | 2.3 | 2.3 | 2.3 | 2.3 ± 0.0 | 2.2 | 100.0 | 100.0 | 2.31e+03 |
| `prim_search_timing_p75` | primattack | 4.1 | 4.1 | 4.1 | 4.1 ± 0.0 | 4.0 | 100.0 | 100.0 | 3.47e+03 |
| `prim_search_timing_unb` | primattack | 23.0 | 22.9 | 22.9 | 22.9 ± 0.0 | 20.1 | 100.0 | 100.0 | 3.75e+04 |
| `prim_search_padding_p50` | primattack | 0.1 | 0.1 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 0.438 |
| `prim_search_padding_p75` | primattack | 0.3 | 0.3 | 0.3 | 0.3 ± 0.0 | 0.0 | 100.0 | 100.0 | 2.03 |
| `prim_search_padding_unb` | primattack | 12.7 | 12.7 | 0.6 | 0.6 ± 0.0 | 0.2 | 100.0 | 100.0 | 3.42 |
| `prim_rand_joint_p50` | primattack | 0.7 | 0.7 | 0.7 | 0.7 ± 0.0 | 0.7 | 100.0 | 100.0 | 627 |
| `prim_rand_joint_p75` | primattack | 2.2 | 2.2 | 2.2 | 2.2 ± 0.0 | 2.0 | 100.0 | 100.0 | 954 |
| `prim_rand_joint_unb` | primattack | 30.9 | 30.9 | 24.8 | 24.8 ± 0.5 | 24.2 | 100.0 | 100.0 | 1.91e+04 |
| `prim_rand_timing_p50` | primattack | 0.4 | 0.4 | 0.4 | 0.4 ± 0.0 | 0.4 | 100.0 | 100.0 | 627 |
| `prim_rand_timing_p75` | primattack | 0.9 | 0.9 | 0.9 | 0.9 ± 0.0 | 0.8 | 100.0 | 100.0 | 954 |
| `prim_rand_timing_unb` | primattack | 13.2 | 13.2 | 13.2 | 13.1 ± 0.6 | 12.5 | 100.0 | 100.0 | 1.91e+04 |
| `prim_rand_padding_p50` | primattack | 0.0 | 0.0 | 0.0 | 0.0 ± 0.0 | 0.0 | 100.0 | 100.0 | 0.592 |
| `prim_rand_padding_p75` | primattack | 0.1 | 0.1 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 1.39 |
| `prim_rand_padding_unb` | primattack | 5.0 | 5.0 | 0.2 | 0.2 ± 0.0 | 0.0 | 100.0 | 100.0 | 2.32 |

### cnn

| Attack | family | raw U | valid U | raw TB | valid TB (± sd) | SP | dom.valid | feas. | mean L2 (scaled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | input | 96.1 | 0.0 | 81.9 | 0.0 ± 0.0 | – | 0.0 | – | 4.01 |
| `cw_untargeted` | input | 95.5 | 0.0 | 64.2 | 0.0 ± 0.0 | – | 0.0 | – | 1.41 |
| `pgd_tb` | input | 99.9 | 0.0 | 99.9 | 0.0 ± 0.0 | – | 0.0 | – | 3.99 |
| `cw_tb` | input | 99.9 | 0.0 | 99.9 | 0.0 ± 0.0 | – | 0.0 | – | 1.4 |
| `fab_untargeted` | fab | 85.7 | 0.0 | 83.4 | 0.0 ± 0.0 | – | 14.3 | – | 3.23e+03 |
| `capgd_native` | capgd | 96.7 | 21.5 | 83.2 | 13.4 ± 0.8 | – | 21.8 | – | 6.87e+04 |
| `capgd_prim_p75` | capgd | 10.9 | 10.9 | 10.7 | 10.7 ± 0.8 | 4.5 | 100.0 | 100.0 | 753 |
| `prim_search_joint_p50` | primattack | 12.9 | 12.9 | 12.7 | 12.7 ± 0.1 | 12.0 | 100.0 | 100.0 | 2.12e+03 |
| `prim_search_joint_p75` | primattack | 35.7 | 35.7 | 35.2 | 35.2 ± 0.2 | 19.6 | 100.0 | 100.0 | 3.02e+03 |
| `prim_search_joint_unb` | primattack | 73.1 | 73.1 | 70.4 | 70.4 ± 0.1 | 40.9 | 100.0 | 100.0 | 2.68e+04 |
| `prim_search_timing_p50` | primattack | 9.2 | 9.2 | 9.2 | 9.2 ± 0.0 | 8.6 | 100.0 | 100.0 | 2.15e+03 |
| `prim_search_timing_p75` | primattack | 13.4 | 13.4 | 13.2 | 13.2 ± 0.0 | 12.2 | 100.0 | 100.0 | 3.26e+03 |
| `prim_search_timing_unb` | primattack | 59.9 | 59.9 | 59.7 | 59.7 ± 0.0 | 30.6 | 100.0 | 100.0 | 3.59e+04 |
| `prim_search_padding_p50` | primattack | 0.2 | 0.2 | 0.2 | 0.2 ± 0.0 | 0.0 | 100.0 | 100.0 | 1.17 |
| `prim_search_padding_p75` | primattack | 0.3 | 0.3 | 0.3 | 0.3 ± 0.0 | 0.0 | 100.0 | 100.0 | 2.75 |
| `prim_search_padding_unb` | primattack | 24.2 | 24.2 | 2.1 | 2.1 ± 0.0 | 1.7 | 100.0 | 100.0 | 4.17 |
| `prim_rand_joint_p50` | primattack | 3.3 | 3.3 | 3.3 | 3.3 ± 0.1 | 3.1 | 100.0 | 100.0 | 628 |
| `prim_rand_joint_p75` | primattack | 5.6 | 5.6 | 5.5 | 5.5 ± 0.2 | 4.9 | 100.0 | 100.0 | 955 |
| `prim_rand_joint_unb` | primattack | 55.9 | 55.9 | 50.4 | 50.4 ± 0.6 | 34.2 | 100.0 | 100.0 | 1.88e+04 |
| `prim_rand_timing_p50` | primattack | 2.9 | 2.9 | 2.9 | 2.9 ± 0.1 | 2.8 | 100.0 | 100.0 | 628 |
| `prim_rand_timing_p75` | primattack | 4.1 | 4.1 | 4.1 | 4.1 ± 0.0 | 4.0 | 100.0 | 100.0 | 954 |
| `prim_rand_timing_unb` | primattack | 32.0 | 32.0 | 31.9 | 31.9 ± 0.7 | 23.1 | 100.0 | 100.0 | 1.88e+04 |
| `prim_rand_padding_p50` | primattack | 0.1 | 0.1 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 0.592 |
| `prim_rand_padding_p75` | primattack | 0.2 | 0.2 | 0.2 | 0.2 ± 0.0 | 0.0 | 100.0 | 100.0 | 1.39 |
| `prim_rand_padding_unb` | primattack | 13.7 | 13.7 | 1.0 | 1.0 ± 0.0 | 0.7 | 100.0 | 100.0 | 2.32 |

### ft_transformer

| Attack | family | raw U | valid U | raw TB | valid TB (± sd) | SP | dom.valid | feas. | mean L2 (scaled) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pgd_untargeted` | input | 97.5 | 0.0 | 87.9 | 0.0 ± 0.0 | – | 0.0 | – | 4.14 |
| `cw_untargeted` | input | 77.2 | 0.0 | 51.9 | 0.0 ± 0.0 | – | 0.0 | – | 0.845 |
| `pgd_tb` | input | 98.5 | 0.0 | 96.9 | 0.0 ± 0.0 | – | 0.0 | – | 4.12 |
| `cw_tb` | input | 79.9 | 0.0 | 79.9 | 0.0 ± 0.0 | – | 0.0 | – | 0.809 |
| `fab_untargeted` | fab | 91.8 | 0.0 | 91.6 | 0.0 ± 0.0 | – | 8.2 | – | 2.63e+04 |
| `capgd_native` | capgd | 51.0 | 5.6 | 41.7 | 1.8 ± 0.4 | – | 8.2 | – | 2.41e+05 |
| `capgd_prim_p75` | capgd | 0.3 | 0.3 | 0.3 | 0.3 ± 0.0 | 0.0 | 100.0 | 100.0 | 372 |
| `prim_search_joint_p50` | primattack | 0.2 | 0.2 | 0.2 | 0.2 ± 0.0 | 0.0 | 100.0 | 100.0 | 2.28e+03 |
| `prim_search_joint_p75` | primattack | 0.5 | 0.5 | 0.4 | 0.4 ± 0.0 | 0.0 | 100.0 | 100.0 | 3.31e+03 |
| `prim_search_joint_unb` | primattack | 0.9 | 0.9 | 0.9 | 0.8 ± 0.0 | 0.1 | 100.0 | 100.0 | 4.25e+04 |
| `prim_search_timing_p50` | primattack | 0.1 | 0.1 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 2.27e+03 |
| `prim_search_timing_p75` | primattack | 0.1 | 0.1 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 3.28e+03 |
| `prim_search_timing_unb` | primattack | 0.4 | 0.4 | 0.4 | 0.4 ± 0.0 | 0.0 | 100.0 | 100.0 | 4.51e+04 |
| `prim_search_padding_p50` | primattack | 0.2 | 0.2 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 1.16 |
| `prim_search_padding_p75` | primattack | 0.4 | 0.4 | 0.4 | 0.4 ± 0.0 | 0.0 | 100.0 | 100.0 | 2.18 |
| `prim_search_padding_unb` | primattack | 0.5 | 0.5 | 0.4 | 0.4 ± 0.0 | 0.0 | 100.0 | 100.0 | 3.86 |
| `prim_rand_joint_p50` | primattack | 0.2 | 0.2 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 627 |
| `prim_rand_joint_p75` | primattack | 0.4 | 0.4 | 0.3 | 0.3 ± 0.0 | 0.0 | 100.0 | 100.0 | 953 |
| `prim_rand_joint_unb` | primattack | 0.7 | 0.7 | 0.6 | 0.6 ± 0.0 | 0.1 | 100.0 | 100.0 | 1.9e+04 |
| `prim_rand_timing_p50` | primattack | 0.1 | 0.1 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 627 |
| `prim_rand_timing_p75` | primattack | 0.1 | 0.1 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 952 |
| `prim_rand_timing_unb` | primattack | 0.3 | 0.3 | 0.3 | 0.3 ± 0.0 | 0.0 | 100.0 | 100.0 | 1.9e+04 |
| `prim_rand_padding_p50` | primattack | 0.2 | 0.2 | 0.1 | 0.1 ± 0.0 | 0.0 | 100.0 | 100.0 | 0.592 |
| `prim_rand_padding_p75` | primattack | 0.4 | 0.4 | 0.3 | 0.3 ± 0.0 | 0.0 | 100.0 | 100.0 | 1.39 |
| `prim_rand_padding_unb` | primattack | 0.4 | 0.4 | 0.3 | 0.3 ± 0.0 | 0.0 | 100.0 | 100.0 | 2.32 |

## Per-class breakdown (valid TB, mean over seeds)

**`capgd_native`**

| Victim | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|
| mlp | 2.3 | 1.0 | 3.1 | 0.7 |
| cnn | 17.2 | 14.0 | 11.2 | 11.3 |
| ft_transformer | 0.3 | 2.2 | 4.5 | 0.1 |

**`prim_search_joint_p50`**

| Victim | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|
| mlp | 14.3 | 9.9 | 0.0 | 0.9 |
| cnn | 21.8 | 28.1 | 0.0 | 0.9 |
| ft_transformer | 0.2 | 0.0 | 0.0 | 0.4 |

**`prim_search_joint_p75`**

| Victim | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|
| mlp | 28.8 | 13.2 | 0.0 | 2.1 |
| cnn | 46.4 | 35.8 | 0.1 | 58.5 |
| ft_transformer | 0.4 | 0.0 | 0.0 | 1.4 |

**`prim_search_joint_unb`**

| Victim | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|
| mlp | 89.0 | 73.4 | 0.9 | 2.5 |
| cnn | 92.5 | 88.1 | 1.0 | 100.0 |
| ft_transformer | 0.5 | 0.0 | 0.6 | 2.2 |

## Seed variability — `prim_search_joint_p75` valid TB (%)

| Victim | seed 42 | seed 123 | seed 2024 |
|---|---:|---:|---:|
| mlp | 11.0 | 11.0 | 11.1 |
| cnn | 35.3 | 35.3 | 35.0 |
| ft_transformer | 0.4 | 0.4 | 0.4 |

## Observations

- **Unconstrained PGD / C&W / FAB** reach 77–100% raw untargeted evasion (52–100% raw targeted-to-Benign) on every victim but **0.0% valid** evasion: no successful adversarial row passes validator_v2 `hybrid_valid`.
- **CAPGD native** keeps a minority of valid rows (valid U: mlp 10.8, cnn 21.5, FT 5.6) — its targeted-to-Benign valid rate is lower (1.8 / 13.4 / 1.8).
- **PrimAttack** outputs are ≥99.98% domain-valid (raw ≈ valid). Valid TB grows with budget and is strongly **victim-dependent**: cnn is most vulnerable, FT-Transformer is near-immune at every budget (≤0.8% even unbounded).
- **Search vs random-feasible** at p75 joint: search is 5× (mlp) to 6× (cnn) the random baseline; on FT both are ≈0.
- **CAPGD restricted to the PrimAttack p75 box** is below PrimAttack search at the same box on every victim.
- Run-to-run (seed) SD of PrimAttack valid TB is ≤0.2 pp for joint p75 and ≤0.7 pp across all PrimAttack configurations.
- FT-Transformer p75 joint valid TB here (0.4%) is far below the 5.2% FT figure recorded in CLAUDE.md for an earlier seed-42 run (targeted∧valid∧feasible); the eligible-row selection and harness differ, and this discrepancy has not been investigated.

## Claim boundary

Feature-space proxies on aggregate CICFlowMeter statistics; no PCAP edited or replayed; no packet-level realizability, malicious-functionality preservation, or in-distribution realism is claimed (IDR not computed). Single test split; statistics reported per victim. Paired tests (Wilson CIs, McNemar/Newcombe, Holm) and the 2017-vs-2018 comparison are in `outputs/adv_campaign_noidr/analysis_tables.md` (`scripts/analyze_adversarial_campaign.py --root outputs/adv_campaign_noidr`); see also `cicids2018_full_attack_results.md`.
