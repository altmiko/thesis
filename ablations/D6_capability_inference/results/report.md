# D6 - Capability inference removed (u = (p, D, s) vs u = M(x) ⊙ (p, D, s))

Outcome: **Valid ASR** over the frozen clean-correct flows (4 classes x 800 per attack seed). Mean over attack seeds 42/2024/2026 with the seed range. Paired McNemar vs the reference arm at seed 42, Holm over this table's comparisons.

### cicids2017_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn | p75 | reference | 13.47% (13.47%-13.47%) | - | 13.47% | - | - |
| cnn | p75 | no_capability | 2.70% (2.53%-2.84%) | -10.77 | 49.07% | 0 / 344 | 2.1e-75 |
| cnn | unb | reference | 59.94% (59.94%-59.94%) | - | 59.94% | - | - |
| cnn | unb | no_capability | 7.25% (6.91%-7.50%) | -52.69 | 98.16% | 0 / 1683 | 0 |
| ft_transformer | p75 | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer | p75 | no_capability | 0.03% (0.03%-0.03%) | -0.09 | 25.25% | 0 / 3 | 0.5 |
| ft_transformer | unb | reference | 0.55% (0.53%-0.56%) | - | 0.55% | - | - |
| ft_transformer | unb | no_capability | 0.03% (0.03%-0.03%) | -0.52 | 25.69% | 0 / 17 | 6.1e-05 |
| mlp | p75 | reference | 4.09% (4.09%-4.09%) | - | 4.09% | - | - |
| mlp | p75 | no_capability | 0.19% (0.16%-0.22%) | -3.91 | 13.97% | 0 / 126 | 6.71e-28 |
| mlp | unb | reference | 22.97% (22.97%-22.97%) | - | 22.97% | - | - |
| mlp | unb | no_capability | 0.71% (0.69%-0.75%) | -22.26 | 72.62% | 0 / 711 | 3.63e-155 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | reference | 1.16% (1.16%-1.16%) | - | 1.16% | - | - |
| cnn-s42 | p75 | no_capability | 0.03% (0.03%-0.03%) | -1.12 | 17.72% | 0 / 36 | 3.26e-08 |
| cnn-s42 | unb | reference | 26.19% (26.16%-26.22%) | - | 26.19% | - | - |
| cnn-s42 | unb | no_capability | 4.87% (4.66%-5.00%) | -21.31 | 68.11% | 9 / 686 | 5.17e-144 |
| ft_transformer-s42 | p75 | reference | 0.00% (0.00%-0.00%) | - | 0.00% | - | - |
| ft_transformer-s42 | p75 | no_capability | 0.00% (0.00%-0.00%) | +0.00 | 0.36% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer-s42 | unb | no_capability | 0.34% (0.34%-0.34%) | +0.22 | 1.79% | 8 / 1 | 0.117 |
| mlp-s42 | p75 | reference | 2.53% (2.53%-2.53%) | - | 2.53% | - | - |
| mlp-s42 | p75 | no_capability | 0.22% (0.19%-0.25%) | -2.31 | 31.69% | 0 / 75 | 9.01e-17 |
| mlp-s42 | unb | reference | 44.36% (44.34%-44.38%) | - | 44.36% | - | - |
| mlp-s42 | unb | no_capability | 38.52% (38.16%-39.06%) | -5.84 | 66.61% | 500 / 669 | 4.47e-06 |

## Ablated arm: valid successes that a capability-aware attack could not produce (all seeds pooled)

`violating` = the realized controls use padding on a flow without padding capability or delay on a flow without timing capability. `feasible` = valid success that also passes the primitive-feasibility/realizability check.

| dataset | victim | budget | flows | valid successes | violating | pad viol. | timing viol. | empty fwd packet filled | within capability | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | 9600 | 259 | 0 | 0 | 0 | 0 | 259 | 259 |
| cicids2017_distrinet | cnn | unb | 9600 | 696 | 0 | 0 | 0 | 0 | 696 | 696 |
| cicids2017_distrinet | ft_transformer | p75 | 9600 | 3 | 0 | 0 | 0 | 0 | 3 | 3 |
| cicids2017_distrinet | ft_transformer | unb | 9600 | 3 | 0 | 0 | 0 | 0 | 3 | 3 |
| cicids2017_distrinet | mlp | p75 | 9600 | 18 | 0 | 0 | 0 | 0 | 18 | 18 |
| cicids2017_distrinet | mlp | unb | 9600 | 68 | 0 | 0 | 0 | 0 | 68 | 68 |
| cicids2018_distrinet | cnn-s42 | p75 | 9600 | 3 | 0 | 0 | 0 | 0 | 3 | 3 |
| cicids2018_distrinet | cnn-s42 | unb | 9600 | 468 | 25 | 0 | 25 | 0 | 443 | 468 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 9600 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 9600 | 33 | 24 | 0 | 24 | 0 | 9 | 33 |
| cicids2018_distrinet | mlp-s42 | p75 | 9600 | 21 | 2 | 0 | 2 | 0 | 19 | 21 |
| cicids2018_distrinet | mlp-s42 | unb | 9600 | 3698 | 1461 | 0 | 1461 | 0 | 2237 | 3698 |

### Which validator layer rejects capability-violating objective hits

| dataset | victim | budget | violating hits | fail schema | fail extractor | fail protocol | fail mined |
|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | 4452 | 0 | 0 | 4452 | 0 |
| cicids2017_distrinet | cnn | unb | 8727 | 0 | 0 | 8727 | 20 |
| cicids2017_distrinet | ft_transformer | p75 | 2421 | 0 | 0 | 2421 | 21 |
| cicids2017_distrinet | ft_transformer | unb | 2463 | 0 | 0 | 2463 | 135 |
| cicids2017_distrinet | mlp | p75 | 1323 | 0 | 0 | 1323 | 0 |
| cicids2017_distrinet | mlp | unb | 6904 | 0 | 0 | 6904 | 30 |
| cicids2018_distrinet | cnn-s42 | p75 | 1698 | 0 | 0 | 1698 | 1650 |
| cicids2018_distrinet | cnn-s42 | unb | 6096 | 0 | 0 | 6071 | 6015 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 35 | 0 | 0 | 35 | 35 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 163 | 0 | 0 | 139 | 139 |
| cicids2018_distrinet | mlp-s42 | p75 | 3023 | 0 | 0 | 3021 | 3021 |
| cicids2018_distrinet | mlp-s42 | unb | 4158 | 0 | 0 | 2532 | 2669 |

### Source capability reason of the violating valid successes

| dataset | victim | budget | primitive | source reason | valid successes |
|---|---|---|---|---|---|
| cicids2018_distrinet | cnn-s42 | unb | timing | SINGLE_FWD_PACKET | 25 |
| cicids2018_distrinet | ft_transformer-s42 | unb | timing | SINGLE_FWD_PACKET | 24 |
| cicids2018_distrinet | mlp-s42 | p75 | timing | SINGLE_FWD_PACKET | 2 |
| cicids2018_distrinet | mlp-s42 | unb | timing | SINGLE_FWD_PACKET | 1461 |

## Where the evaluation budget went (flows whose source permits timing)

| dataset | victim | budget | arm | timing-capable flows | with padding headroom | never refined | Valid ASR on these flows |
|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 7086 | 0 | 0 | 18.25% |
| cicids2017_distrinet | cnn | p75 | no_capability | 7086 | 7086 | 0 | 3.66% |
| cicids2017_distrinet | cnn | unb | reference | 7086 | 0 | 0 | 81.20% |
| cicids2017_distrinet | cnn | unb | no_capability | 7086 | 7086 | 0 | 9.82% |
| cicids2017_distrinet | ft_transformer | p75 | reference | 7089 | 0 | 0 | 0.17% |
| cicids2017_distrinet | ft_transformer | p75 | no_capability | 7089 | 7089 | 0 | 0.04% |
| cicids2017_distrinet | ft_transformer | unb | reference | 7089 | 0 | 0 | 0.75% |
| cicids2017_distrinet | ft_transformer | unb | no_capability | 7089 | 7089 | 0 | 0.04% |
| cicids2017_distrinet | mlp | p75 | reference | 7086 | 0 | 0 | 5.55% |
| cicids2017_distrinet | mlp | p75 | no_capability | 7086 | 7086 | 0 | 0.25% |
| cicids2017_distrinet | mlp | unb | reference | 7086 | 0 | 0 | 31.12% |
| cicids2017_distrinet | mlp | unb | no_capability | 7086 | 7086 | 0 | 0.96% |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 7170 | 0 | 0 | 1.55% |
| cicids2018_distrinet | cnn-s42 | p75 | no_capability | 7170 | 7149 | 0 | 0.04% |
| cicids2018_distrinet | cnn-s42 | unb | reference | 7170 | 0 | 0 | 35.06% |
| cicids2018_distrinet | cnn-s42 | unb | no_capability | 7170 | 7149 | 0 | 6.18% |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 7173 | 0 | 0 | 0.00% |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_capability | 7173 | 7149 | 0 | 0.00% |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 7173 | 0 | 0 | 0.17% |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_capability | 7173 | 7149 | 0 | 0.13% |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 7173 | 0 | 0 | 3.39% |
| cicids2018_distrinet | mlp-s42 | p75 | no_capability | 7173 | 7149 | 0 | 0.26% |
| cicids2018_distrinet | mlp-s42 | unb | reference | 7173 | 0 | 0 | 59.38% |
| cicids2018_distrinet | mlp-s42 | unb | no_capability | 7173 | 7149 | 0 | 31.19% |

`never refined` counts flows that reached no refinement gradient step: either already solved before refinement (identity / padding enumeration) or their evaluation budget was used up by the padding enumeration. Without capability inference every flow gets padding headroom, so the enumeration runs on every flow before refinement (all seeds pooled).

## Non-finite gradients of φ in the ablated arm (all seeds pooled)

Outside the capability-admissible set φ is not differentiable everywhere (e.g. padding a flow whose packet-length variance is 0: d sqrt(var)/dp = inf · 0). The ablated arm zeroes such gradient coordinates; the reference never meets one.

| dataset | victim | budget | flows | flows with a non-finite gradient step |
|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | 9600 | 0 |
| cicids2017_distrinet | cnn | unb | 9600 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | 9600 | 0 |
| cicids2017_distrinet | ft_transformer | unb | 9600 | 0 |
| cicids2017_distrinet | mlp | p75 | 9600 | 0 |
| cicids2017_distrinet | mlp | unb | 9600 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | 9600 | 69 |
| cicids2018_distrinet | cnn-s42 | unb | 9600 | 69 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 9600 | 69 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 9600 | 69 |
| cicids2018_distrinet | mlp-s42 | p75 | 9600 | 69 |
| cicids2018_distrinet | mlp-s42 | unb | 9600 | 69 |

## Cost profile of the successes (`valid_success`, all seeds pooled)

| dataset | victim | budget | condition | successes | median norm. cost | median rel. duration change | median delay (µs) | median shape | frac. using padding | median added bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.939 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | no_capability | 259 | 0.871 | 0.906 | 5.27e+05 | 0.953 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | reference | 5754 | 0.379 | 6.01 | 7.76e+06 | 0.691 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | no_capability | 696 | 0.468 | 59.4 | 9.94e+06 | 0.506 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | no_capability | 3 | 0.2 | 0.000939 | 2.69e+04 | 0.3 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | reference | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | no_capability | 3 | 0.2 | 0.000939 | 2.69e+04 | 0.3 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | reference | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | no_capability | 18 | 0.797 | 0.64 | 2.13e+06 | 0.6 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | reference | 2205 | 0.3 | 25.7 | 5.2e+06 | 0.664 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | no_capability | 68 | 0.1 | 111 | 2.54e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | no_capability | 3 | 0.432 | 0.617 | 1.27e+05 | 0.931 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 2514 | 0.166 | 51.2 | 1.86e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | no_capability | 468 | 0.498 | 169 | 5.47e+07 | 0.654 | 0.0299 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_capability | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_capability | 33 | 0.725 | 2.66e+04 | 9.61e+06 | 1 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | no_capability | 21 | 0.962 | 0.991 | 5.08e+06 | 1 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | no_capability | 3698 | 0.7 | 279 | 5.84e+07 | 0 | 0.000541 | 0 |

