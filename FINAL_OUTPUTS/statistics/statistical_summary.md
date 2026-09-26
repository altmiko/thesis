# Minimum paired statistical tests for the final experiments

## Protocol

- Paired unit: one canonical clean-correct source flow; four source classes are pooled within each victim (800 per class, N = 3,200). Datasets and victims are never pooled.
- Inference uses reference attack seed 42 only. Seeds 42, 2024, and 2026 are used only for descriptive mean ± sample SD of rates; seed-level means are not inferential units.
- Primary inferential outcome: `valid_success`. The only exception is the dedicated paired `raw_success` versus `valid_success` comparison.
- α = 0.05. Cochran’s Q gates the two planned McNemar comparisons in each three-condition family. Holm correction is applied only across those two comparisons.
- McNemar uses an exact two-sided binomial test for fewer than 25 discordant pairs; otherwise it uses the continuity-corrected χ² statistic.
- Selected canonical PrimAttack optimizer: `pgd`.

## Pairing and provenance audit

Before any statistic was computed, 936 stored NPZ artifacts (748,800 per-sample rows) were checked against the canonical `selection.json` records. Checks covered exact sample-ID order, duplicate IDs, positional indices, clean-input SHA-256, seed, objective, attack method and, where applicable, budget and primitive mode, outcome lengths, and `valid_success ⊆ raw_success`. Every multi-condition family was checked for identical paired IDs for all three seeds. Any mismatch aborts before output writing.

## 1. Primitive-mode ablation

This test asks whether valid untargeted success differs across timing-only, padding-only, and joint primitive control at the maximum-evaluated budget. A significant Q establishes a condition effect; the gated joint-vs-timing and joint-vs-padding McNemar tests identify which planned contrasts differ.

| Dataset | Victim | Mode | N | Successes (seed 42) | Rate (seed 42) | Mean ± SD (3 seeds) |
|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | timing-only | 3200 | 131 | 4.09% | 4.09% ± 0.00% |
| cicids2017_distrinet | mlp | padding-only | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2017_distrinet | mlp | joint | 3200 | 131 | 4.09% | 4.09% ± 0.00% |
| cicids2017_distrinet | cnn | timing-only | 3200 | 431 | 13.47% | 13.47% ± 0.00% |
| cicids2017_distrinet | cnn | padding-only | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2017_distrinet | cnn | joint | 3200 | 431 | 13.47% | 13.47% ± 0.00% |
| cicids2017_distrinet | ft_transformer | timing-only | 3200 | 4 | 0.12% | 0.12% ± 0.00% |
| cicids2017_distrinet | ft_transformer | padding-only | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2017_distrinet | ft_transformer | joint | 3200 | 4 | 0.12% | 0.12% ± 0.00% |
| cicids2018_distrinet | mlp-s42 | timing-only | 3200 | 81 | 2.53% | 2.53% ± 0.00% |
| cicids2018_distrinet | mlp-s42 | padding-only | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2018_distrinet | mlp-s42 | joint | 3200 | 81 | 2.53% | 2.53% ± 0.00% |
| cicids2018_distrinet | cnn-s42 | timing-only | 3200 | 37 | 1.16% | 1.16% ± 0.00% |
| cicids2018_distrinet | cnn-s42 | padding-only | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2018_distrinet | cnn-s42 | joint | 3200 | 37 | 1.16% | 1.16% ± 0.00% |
| cicids2018_distrinet | ft_transformer-s42 | timing-only | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2018_distrinet | ft_transformer-s42 | padding-only | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2018_distrinet | ft_transformer-s42 | joint | 3200 | 0 | 0.00% | 0.00% ± 0.00% |

## 2. Budget sensitivity

**Not computed.** The required restricted (train-p25) condition has no stored per-sample artifacts (72 expected files are absent). The stored final suite contains intermediate (p50), maximum-evaluated (p75), and an unbounded sensitivity condition. Unbounded was not substituted for restricted because these are different perturbation budgets. No attack was rerun.

| Dataset | Victim | Budget | N | Successes (seed 42) | Rate (seed 42) | Mean ± SD (3 seeds) |
|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | intermediate | 3200 | 74 | 2.31% | 2.31% ± 0.00% |
| cicids2017_distrinet | mlp | maximum-evaluated | 3200 | 131 | 4.09% | 4.09% ± 0.00% |
| cicids2017_distrinet | cnn | intermediate | 3200 | 294 | 9.19% | 9.19% ± 0.00% |
| cicids2017_distrinet | cnn | maximum-evaluated | 3200 | 424 | 13.25% | 13.25% ± 0.00% |
| cicids2017_distrinet | ft_transformer | intermediate | 3200 | 4 | 0.12% | 0.12% ± 0.00% |
| cicids2017_distrinet | ft_transformer | maximum-evaluated | 3200 | 4 | 0.12% | 0.12% ± 0.00% |
| cicids2018_distrinet | mlp-s42 | intermediate | 3200 | 22 | 0.69% | 0.69% ± 0.00% |
| cicids2018_distrinet | mlp-s42 | maximum-evaluated | 3200 | 25 | 0.78% | 0.78% ± 0.00% |
| cicids2018_distrinet | cnn-s42 | intermediate | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2018_distrinet | cnn-s42 | maximum-evaluated | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2018_distrinet | ft_transformer-s42 | intermediate | 3200 | 0 | 0.00% | 0.00% ± 0.00% |
| cicids2018_distrinet | ft_transformer-s42 | maximum-evaluated | 3200 | 0 | 0.00% | 0.00% ± 0.00% |

## 3. Raw-vs-Valid validity gap

For each canonical headline attack condition (the five inferential Exp-A methods and the three maximum-evaluated targeted PrimAttack optimizers), McNemar compares `raw_success` with `valid_success` on the same seed-42 flows. Because valid success is a subset of raw success, the directional count `Raw succeeds / Valid fails` is the number of apparent successes removed by validity enforcement; the reverse discordance must be zero.

| Dataset | Victim | Condition | N | Raw seed 42 | Valid seed 42 | Raw mean ± SD | Valid mean ± SD | Mean validity gap |
|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | PrimAttack-pgd untargeted maximum-evaluated | 3200 | 131 (4.09%) | 131 (4.09%) | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | mlp | Input PGD untargeted | 3200 | 3200 (100.00%) | 0 (0.00%) | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 pp |
| cicids2017_distrinet | mlp | Input C&W untargeted | 3200 | 3198 (99.94%) | 0 (0.00%) | 99.94% ± 0.00% | 0.00% ± 0.00% | 99.94 pp |
| cicids2017_distrinet | mlp | CAPGD-PrimSupport untargeted | 3200 | 3027 (94.59%) | 65 (2.03%) | 94.41% ± 0.71% | 2.01% ± 0.10% | 92.40 pp |
| cicids2017_distrinet | mlp | C-PGD-PrimSupport untargeted | 3200 | 1645 (51.41%) | 0 (0.00%) | 50.80% ± 2.04% | 0.00% ± 0.00% | 50.80 pp |
| cicids2017_distrinet | mlp | PrimAttack-hybrid targeted maximum-evaluated | 3200 | 131 (4.09%) | 131 (4.09%) | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | mlp | PrimAttack-pgd targeted maximum-evaluated | 3200 | 131 (4.09%) | 131 (4.09%) | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | mlp | PrimAttack-cw targeted maximum-evaluated | 3200 | 131 (4.09%) | 131 (4.09%) | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | cnn | PrimAttack-pgd untargeted maximum-evaluated | 3200 | 431 (13.47%) | 431 (13.47%) | 13.47% ± 0.00% | 13.47% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | cnn | Input PGD untargeted | 3200 | 3079 (96.22%) | 0 (0.00%) | 96.12% ± 0.09% | 0.00% ± 0.00% | 96.12 pp |
| cicids2017_distrinet | cnn | Input C&W untargeted | 3200 | 3057 (95.53%) | 0 (0.00%) | 95.53% ± 0.00% | 0.00% ± 0.00% | 95.53 pp |
| cicids2017_distrinet | cnn | CAPGD-PrimSupport untargeted | 3200 | 3121 (97.53%) | 166 (5.19%) | 96.53% ± 1.07% | 5.15% ± 0.07% | 91.39 pp |
| cicids2017_distrinet | cnn | C-PGD-PrimSupport untargeted | 3200 | 2032 (63.50%) | 0 (0.00%) | 60.42% ± 2.98% | 0.00% ± 0.00% | 60.42 pp |
| cicids2017_distrinet | cnn | PrimAttack-hybrid targeted maximum-evaluated | 3200 | 424 (13.25%) | 424 (13.25%) | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | cnn | PrimAttack-pgd targeted maximum-evaluated | 3200 | 424 (13.25%) | 424 (13.25%) | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | cnn | PrimAttack-cw targeted maximum-evaluated | 3200 | 128 (4.00%) | 128 (4.00%) | 4.00% ± 0.00% | 4.00% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | ft_transformer | PrimAttack-pgd untargeted maximum-evaluated | 3200 | 4 (0.12%) | 4 (0.12%) | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | ft_transformer | Input PGD untargeted | 3200 | 3115 (97.34%) | 0 (0.00%) | 97.36% ± 0.28% | 0.00% ± 0.00% | 97.36 pp |
| cicids2017_distrinet | ft_transformer | Input C&W untargeted | 3200 | 2469 (77.16%) | 0 (0.00%) | 77.16% ± 0.00% | 0.00% ± 0.00% | 77.16 pp |
| cicids2017_distrinet | ft_transformer | CAPGD-PrimSupport untargeted | 3200 | 1658 (51.81%) | 5 (0.16%) | 52.21% ± 4.33% | 0.18% ± 0.02% | 52.03 pp |
| cicids2017_distrinet | ft_transformer | C-PGD-PrimSupport untargeted | 3200 | 686 (21.44%) | 0 (0.00%) | 21.61% ± 0.31% | 0.00% ± 0.00% | 21.61 pp |
| cicids2017_distrinet | ft_transformer | PrimAttack-hybrid targeted maximum-evaluated | 3200 | 4 (0.12%) | 4 (0.12%) | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | ft_transformer | PrimAttack-pgd targeted maximum-evaluated | 3200 | 4 (0.12%) | 4 (0.12%) | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 pp |
| cicids2017_distrinet | ft_transformer | PrimAttack-cw targeted maximum-evaluated | 3200 | 4 (0.12%) | 4 (0.12%) | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | mlp-s42 | PrimAttack-pgd untargeted maximum-evaluated | 3200 | 81 (2.53%) | 81 (2.53%) | 2.53% ± 0.00% | 2.53% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | mlp-s42 | Input PGD untargeted | 3200 | 3019 (94.34%) | 0 (0.00%) | 94.34% ± 0.25% | 0.00% ± 0.00% | 94.34 pp |
| cicids2018_distrinet | mlp-s42 | Input C&W untargeted | 3200 | 2804 (87.62%) | 0 (0.00%) | 87.63% ± 0.00% | 0.00% ± 0.00% | 87.63 pp |
| cicids2018_distrinet | mlp-s42 | CAPGD-PrimSupport untargeted | 3200 | 2961 (92.53%) | 4 (0.12%) | 91.57% ± 0.84% | 0.14% ± 0.02% | 91.44 pp |
| cicids2018_distrinet | mlp-s42 | C-PGD-PrimSupport untargeted | 3200 | 885 (27.66%) | 0 (0.00%) | 28.25% ± 0.51% | 0.00% ± 0.00% | 28.25 pp |
| cicids2018_distrinet | mlp-s42 | PrimAttack-hybrid targeted maximum-evaluated | 3200 | 25 (0.78%) | 25 (0.78%) | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | mlp-s42 | PrimAttack-pgd targeted maximum-evaluated | 3200 | 25 (0.78%) | 25 (0.78%) | 0.78% ± 0.00% | 0.78% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | mlp-s42 | PrimAttack-cw targeted maximum-evaluated | 3200 | 24 (0.75%) | 24 (0.75%) | 0.75% ± 0.00% | 0.75% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | cnn-s42 | PrimAttack-pgd untargeted maximum-evaluated | 3200 | 37 (1.16%) | 37 (1.16%) | 1.16% ± 0.00% | 1.16% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | cnn-s42 | Input PGD untargeted | 3200 | 3191 (99.72%) | 0 (0.00%) | 99.70% ± 0.02% | 0.00% ± 0.00% | 99.70 pp |
| cicids2018_distrinet | cnn-s42 | Input C&W untargeted | 3200 | 3173 (99.16%) | 0 (0.00%) | 99.16% ± 0.00% | 0.00% ± 0.00% | 99.16 pp |
| cicids2018_distrinet | cnn-s42 | CAPGD-PrimSupport untargeted | 3200 | 2455 (76.72%) | 7 (0.22%) | 76.01% ± 1.68% | 0.29% ± 0.07% | 75.72 pp |
| cicids2018_distrinet | cnn-s42 | C-PGD-PrimSupport untargeted | 3200 | 1531 (47.84%) | 0 (0.00%) | 50.54% ± 3.30% | 0.00% ± 0.00% | 50.54 pp |
| cicids2018_distrinet | cnn-s42 | PrimAttack-hybrid targeted maximum-evaluated | 3200 | 0 (0.00%) | 0 (0.00%) | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | cnn-s42 | PrimAttack-pgd targeted maximum-evaluated | 3200 | 0 (0.00%) | 0 (0.00%) | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | cnn-s42 | PrimAttack-cw targeted maximum-evaluated | 3200 | 0 (0.00%) | 0 (0.00%) | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | ft_transformer-s42 | PrimAttack-pgd untargeted maximum-evaluated | 3200 | 0 (0.00%) | 0 (0.00%) | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | ft_transformer-s42 | Input PGD untargeted | 3200 | 2941 (91.91%) | 0 (0.00%) | 91.70% ± 0.18% | 0.00% ± 0.00% | 91.70 pp |
| cicids2018_distrinet | ft_transformer-s42 | Input C&W untargeted | 3200 | 1715 (53.59%) | 0 (0.00%) | 53.59% ± 0.00% | 0.00% ± 0.00% | 53.59 pp |
| cicids2018_distrinet | ft_transformer-s42 | CAPGD-PrimSupport untargeted | 3200 | 335 (10.47%) | 0 (0.00%) | 9.74% ± 0.70% | 0.00% ± 0.00% | 9.74 pp |
| cicids2018_distrinet | ft_transformer-s42 | C-PGD-PrimSupport untargeted | 3200 | 39 (1.22%) | 0 (0.00%) | 1.65% ± 0.42% | 0.00% ± 0.00% | 1.65 pp |
| cicids2018_distrinet | ft_transformer-s42 | PrimAttack-hybrid targeted maximum-evaluated | 3200 | 0 (0.00%) | 0 (0.00%) | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | ft_transformer-s42 | PrimAttack-pgd targeted maximum-evaluated | 3200 | 0 (0.00%) | 0 (0.00%) | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |
| cicids2018_distrinet | ft_transformer-s42 | PrimAttack-cw targeted maximum-evaluated | 3200 | 0 (0.00%) | 0 (0.00%) | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |

## Inferential results

| Analysis | Dataset | Victim | Test | Conditions | N | Success counts | Success rates | Statistic | p | Holm p | Significant | A-only / B-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Primitive-mode ablation | cicids2017_distrinet | mlp | Cochran's Q | timing-only / padding-only / joint | 3200 | 131 / 0 / 131 | 4.09% / 0.00% / 4.09% | 262.000 | 1.28e-57 | — | yes | — |
| Primitive-mode ablation | cicids2017_distrinet | cnn | Cochran's Q | timing-only / padding-only / joint | 3200 | 431 / 0 / 431 | 13.47% / 0.00% / 13.47% | 862.000 | 6.59e-188 | — | yes | — |
| Primitive-mode ablation | cicids2017_distrinet | ft_transformer | Cochran's Q | timing-only / padding-only / joint | 3200 | 4 / 0 / 4 | 0.12% / 0.00% / 0.12% | 8.000 | 0.0183 | — | yes | — |
| Primitive-mode ablation | cicids2018_distrinet | mlp-s42 | Cochran's Q | timing-only / padding-only / joint | 3200 | 81 / 0 / 81 | 2.53% / 0.00% / 2.53% | 162.000 | 6.64e-36 | — | yes | — |
| Primitive-mode ablation | cicids2018_distrinet | cnn-s42 | Cochran's Q | timing-only / padding-only / joint | 3200 | 37 / 0 / 37 | 1.16% / 0.00% / 1.16% | 74.000 | 8.53e-17 | — | yes | — |
| Primitive-mode ablation | cicids2018_distrinet | ft_transformer-s42 | Cochran's Q | timing-only / padding-only / joint | 3200 | 0 / 0 / 0 | 0.00% / 0.00% / 0.00% | 0.000 | 1 | — | no | — |
| Primitive-mode ablation | cicids2017_distrinet | mlp | McNemar | joint vs timing-only | 3200 | 131 / 131 | 4.09% / 4.09% | — | 1 | 1 | no | 0 / 0 |
| Primitive-mode ablation | cicids2017_distrinet | mlp | McNemar | joint vs padding-only | 3200 | 131 / 0 | 4.09% / 0.00% | 129.008 | 6.76e-30 | 1.35e-29 | yes | 131 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | PrimAttack-pgd untargeted maximum-evaluated:raw_success vs PrimAttack-pgd untargeted maximum-evaluated:valid_success | 3200 | 131 / 131 | 4.09% / 4.09% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | Input PGD untargeted:raw_success vs Input PGD untargeted:valid_success | 3200 | 3200 / 0 | 100.00% / 0.00% | 3198.000 | <1×10⁻³⁰⁰ | — | yes | 3200 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | Input C&W untargeted:raw_success vs Input C&W untargeted:valid_success | 3200 | 3198 / 0 | 99.94% / 0.00% | 3196.000 | <1×10⁻³⁰⁰ | — | yes | 3198 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | CAPGD-PrimSupport untargeted:raw_success vs CAPGD-PrimSupport untargeted:valid_success | 3200 | 3027 / 65 | 94.59% / 2.03% | 2960.000 | <1×10⁻³⁰⁰ | — | yes | 2962 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | C-PGD-PrimSupport untargeted:raw_success vs C-PGD-PrimSupport untargeted:valid_success | 3200 | 1645 / 0 | 51.41% / 0.00% | 1643.001 | <1×10⁻³⁰⁰ | — | yes | 1645 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | PrimAttack-hybrid targeted maximum-evaluated:raw_success vs PrimAttack-hybrid targeted maximum-evaluated:valid_success | 3200 | 131 / 131 | 4.09% / 4.09% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | PrimAttack-pgd targeted maximum-evaluated:raw_success vs PrimAttack-pgd targeted maximum-evaluated:valid_success | 3200 | 131 / 131 | 4.09% / 4.09% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | mlp | McNemar | PrimAttack-cw targeted maximum-evaluated:raw_success vs PrimAttack-cw targeted maximum-evaluated:valid_success | 3200 | 131 / 131 | 4.09% / 4.09% | — | 1 | — | no | 0 / 0 |
| Primitive-mode ablation | cicids2017_distrinet | cnn | McNemar | joint vs timing-only | 3200 | 431 / 431 | 13.47% / 13.47% | — | 1 | 1 | no | 0 / 0 |
| Primitive-mode ablation | cicids2017_distrinet | cnn | McNemar | joint vs padding-only | 3200 | 431 / 0 | 13.47% / 0.00% | 429.002 | 2.68e-95 | 5.36e-95 | yes | 431 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | PrimAttack-pgd untargeted maximum-evaluated:raw_success vs PrimAttack-pgd untargeted maximum-evaluated:valid_success | 3200 | 431 / 431 | 13.47% / 13.47% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | Input PGD untargeted:raw_success vs Input PGD untargeted:valid_success | 3200 | 3079 / 0 | 96.22% / 0.00% | 3077.000 | <1×10⁻³⁰⁰ | — | yes | 3079 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | Input C&W untargeted:raw_success vs Input C&W untargeted:valid_success | 3200 | 3057 / 0 | 95.53% / 0.00% | 3055.000 | <1×10⁻³⁰⁰ | — | yes | 3057 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | CAPGD-PrimSupport untargeted:raw_success vs CAPGD-PrimSupport untargeted:valid_success | 3200 | 3121 / 166 | 97.53% / 5.19% | 2953.000 | <1×10⁻³⁰⁰ | — | yes | 2955 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | C-PGD-PrimSupport untargeted:raw_success vs C-PGD-PrimSupport untargeted:valid_success | 3200 | 2032 / 0 | 63.50% / 0.00% | 2030.000 | <1×10⁻³⁰⁰ | — | yes | 2032 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | PrimAttack-hybrid targeted maximum-evaluated:raw_success vs PrimAttack-hybrid targeted maximum-evaluated:valid_success | 3200 | 424 / 424 | 13.25% / 13.25% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | PrimAttack-pgd targeted maximum-evaluated:raw_success vs PrimAttack-pgd targeted maximum-evaluated:valid_success | 3200 | 424 / 424 | 13.25% / 13.25% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | cnn | McNemar | PrimAttack-cw targeted maximum-evaluated:raw_success vs PrimAttack-cw targeted maximum-evaluated:valid_success | 3200 | 128 / 128 | 4.00% / 4.00% | — | 1 | — | no | 0 / 0 |
| Primitive-mode ablation | cicids2017_distrinet | ft_transformer | McNemar | joint vs timing-only | 3200 | 4 / 4 | 0.12% / 0.12% | — | 1 | 1 | no | 0 / 0 |
| Primitive-mode ablation | cicids2017_distrinet | ft_transformer | McNemar | joint vs padding-only | 3200 | 4 / 0 | 0.12% / 0.00% | — | 0.125 | 0.25 | no | 4 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | PrimAttack-pgd untargeted maximum-evaluated:raw_success vs PrimAttack-pgd untargeted maximum-evaluated:valid_success | 3200 | 4 / 4 | 0.12% / 0.12% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | Input PGD untargeted:raw_success vs Input PGD untargeted:valid_success | 3200 | 3115 / 0 | 97.34% / 0.00% | 3113.000 | <1×10⁻³⁰⁰ | — | yes | 3115 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | Input C&W untargeted:raw_success vs Input C&W untargeted:valid_success | 3200 | 2469 / 0 | 77.16% / 0.00% | 2467.000 | <1×10⁻³⁰⁰ | — | yes | 2469 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | CAPGD-PrimSupport untargeted:raw_success vs CAPGD-PrimSupport untargeted:valid_success | 3200 | 1658 / 5 | 51.81% / 0.16% | 1651.001 | <1×10⁻³⁰⁰ | — | yes | 1653 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | C-PGD-PrimSupport untargeted:raw_success vs C-PGD-PrimSupport untargeted:valid_success | 3200 | 686 / 0 | 21.44% / 0.00% | 684.001 | 9.01e-151 | — | yes | 686 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | PrimAttack-hybrid targeted maximum-evaluated:raw_success vs PrimAttack-hybrid targeted maximum-evaluated:valid_success | 3200 | 4 / 4 | 0.12% / 0.12% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | PrimAttack-pgd targeted maximum-evaluated:raw_success vs PrimAttack-pgd targeted maximum-evaluated:valid_success | 3200 | 4 / 4 | 0.12% / 0.12% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2017_distrinet | ft_transformer | McNemar | PrimAttack-cw targeted maximum-evaluated:raw_success vs PrimAttack-cw targeted maximum-evaluated:valid_success | 3200 | 4 / 4 | 0.12% / 0.12% | — | 1 | — | no | 0 / 0 |
| Primitive-mode ablation | cicids2018_distrinet | mlp-s42 | McNemar | joint vs timing-only | 3200 | 81 / 81 | 2.53% / 2.53% | — | 1 | 1 | no | 0 / 0 |
| Primitive-mode ablation | cicids2018_distrinet | mlp-s42 | McNemar | joint vs padding-only | 3200 | 81 / 0 | 2.53% / 0.00% | 79.012 | 6.17e-19 | 1.23e-18 | yes | 81 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | PrimAttack-pgd untargeted maximum-evaluated:raw_success vs PrimAttack-pgd untargeted maximum-evaluated:valid_success | 3200 | 81 / 81 | 2.53% / 2.53% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | Input PGD untargeted:raw_success vs Input PGD untargeted:valid_success | 3200 | 3019 / 0 | 94.34% / 0.00% | 3017.000 | <1×10⁻³⁰⁰ | — | yes | 3019 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | Input C&W untargeted:raw_success vs Input C&W untargeted:valid_success | 3200 | 2804 / 0 | 87.62% / 0.00% | 2802.000 | <1×10⁻³⁰⁰ | — | yes | 2804 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | CAPGD-PrimSupport untargeted:raw_success vs CAPGD-PrimSupport untargeted:valid_success | 3200 | 2961 / 4 | 92.53% / 0.12% | 2955.000 | <1×10⁻³⁰⁰ | — | yes | 2957 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | C-PGD-PrimSupport untargeted:raw_success vs C-PGD-PrimSupport untargeted:valid_success | 3200 | 885 / 0 | 27.66% / 0.00% | 883.001 | 4.87e-194 | — | yes | 885 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | PrimAttack-hybrid targeted maximum-evaluated:raw_success vs PrimAttack-hybrid targeted maximum-evaluated:valid_success | 3200 | 25 / 25 | 0.78% / 0.78% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | PrimAttack-pgd targeted maximum-evaluated:raw_success vs PrimAttack-pgd targeted maximum-evaluated:valid_success | 3200 | 25 / 25 | 0.78% / 0.78% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | mlp-s42 | McNemar | PrimAttack-cw targeted maximum-evaluated:raw_success vs PrimAttack-cw targeted maximum-evaluated:valid_success | 3200 | 24 / 24 | 0.75% / 0.75% | — | 1 | — | no | 0 / 0 |
| Primitive-mode ablation | cicids2018_distrinet | cnn-s42 | McNemar | joint vs timing-only | 3200 | 37 / 37 | 1.16% / 1.16% | — | 1 | 1 | no | 0 / 0 |
| Primitive-mode ablation | cicids2018_distrinet | cnn-s42 | McNemar | joint vs padding-only | 3200 | 37 / 0 | 1.16% / 0.00% | 35.027 | 3.25e-09 | 6.5e-09 | yes | 37 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | PrimAttack-pgd untargeted maximum-evaluated:raw_success vs PrimAttack-pgd untargeted maximum-evaluated:valid_success | 3200 | 37 / 37 | 1.16% / 1.16% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | Input PGD untargeted:raw_success vs Input PGD untargeted:valid_success | 3200 | 3191 / 0 | 99.72% / 0.00% | 3189.000 | <1×10⁻³⁰⁰ | — | yes | 3191 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | Input C&W untargeted:raw_success vs Input C&W untargeted:valid_success | 3200 | 3173 / 0 | 99.16% / 0.00% | 3171.000 | <1×10⁻³⁰⁰ | — | yes | 3173 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | CAPGD-PrimSupport untargeted:raw_success vs CAPGD-PrimSupport untargeted:valid_success | 3200 | 2455 / 7 | 76.72% / 0.22% | 2446.000 | <1×10⁻³⁰⁰ | — | yes | 2448 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | C-PGD-PrimSupport untargeted:raw_success vs C-PGD-PrimSupport untargeted:valid_success | 3200 | 1531 / 0 | 47.84% / 0.00% | 1529.001 | <1×10⁻³⁰⁰ | — | yes | 1531 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | PrimAttack-hybrid targeted maximum-evaluated:raw_success vs PrimAttack-hybrid targeted maximum-evaluated:valid_success | 3200 | 0 / 0 | 0.00% / 0.00% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | PrimAttack-pgd targeted maximum-evaluated:raw_success vs PrimAttack-pgd targeted maximum-evaluated:valid_success | 3200 | 0 / 0 | 0.00% / 0.00% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | cnn-s42 | McNemar | PrimAttack-cw targeted maximum-evaluated:raw_success vs PrimAttack-cw targeted maximum-evaluated:valid_success | 3200 | 0 / 0 | 0.00% / 0.00% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | PrimAttack-pgd untargeted maximum-evaluated:raw_success vs PrimAttack-pgd untargeted maximum-evaluated:valid_success | 3200 | 0 / 0 | 0.00% / 0.00% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | Input PGD untargeted:raw_success vs Input PGD untargeted:valid_success | 3200 | 2941 / 0 | 91.91% / 0.00% | 2939.000 | <1×10⁻³⁰⁰ | — | yes | 2941 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | Input C&W untargeted:raw_success vs Input C&W untargeted:valid_success | 3200 | 1715 / 0 | 53.59% / 0.00% | 1713.001 | <1×10⁻³⁰⁰ | — | yes | 1715 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | CAPGD-PrimSupport untargeted:raw_success vs CAPGD-PrimSupport untargeted:valid_success | 3200 | 335 / 0 | 10.47% / 0.00% | 333.003 | 2.13e-74 | — | yes | 335 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | C-PGD-PrimSupport untargeted:raw_success vs C-PGD-PrimSupport untargeted:valid_success | 3200 | 39 / 0 | 1.22% / 0.00% | 37.026 | 1.17e-09 | — | yes | 39 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | PrimAttack-hybrid targeted maximum-evaluated:raw_success vs PrimAttack-hybrid targeted maximum-evaluated:valid_success | 3200 | 0 / 0 | 0.00% / 0.00% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | PrimAttack-pgd targeted maximum-evaluated:raw_success vs PrimAttack-pgd targeted maximum-evaluated:valid_success | 3200 | 0 / 0 | 0.00% / 0.00% | — | 1 | — | no | 0 / 0 |
| Raw-vs-Valid validity gap | cicids2018_distrinet | ft_transformer-s42 | McNemar | PrimAttack-cw targeted maximum-evaluated:raw_success vs PrimAttack-cw targeted maximum-evaluated:valid_success | 3200 | 0 / 0 | 0.00% / 0.00% | — | 1 | — | no | 0 / 0 |

The machine-readable version is `statistical_tests.csv`. Rows whose planned McNemar tests were gated off by a non-significant Cochran’s Q remain in the CSV with status `not performed`; they are not included in the table above.

## Interpretation: statistical versus practical significance

Among 48 performed Raw-vs-Valid comparisons, 24 show a statistically significant loss after validity enforcement and 24 have no seed-42 raw-success-but-invalid discordance.

Statistical significance addresses whether the paired outcome difference is unlikely under the null of equal marginal success probabilities. It does not measure the size or thesis importance of that difference. Practical significance must be read from the success-rate difference and directional discordant count. With N = 3,200, a small rate difference can be statistically significant; conversely, a non-significant result does not establish equivalence. Mean ± SD across attack seeds describes run-to-run variability only and is not uncertainty for the paired hypothesis tests.

The primitive-mode inferential analysis was not part of the originally locked final-suite plan, where that ablation was descriptive. It must therefore be identified as an added post-run analysis in the thesis rather than described as pre-registered.
