# V1 - Toggle-able validator rule: no forward IAT on single-forward-packet flows

Primary outcome: **Valid ASR under validator_v2 + rule** (`extended_valid_success`). Mean over attack seeds 42/2024/2026 (seed range); paired McNemar rule-on vs rule-off at seed 42, Holm within each PrimAttack variant.

## PrimAttack variant: capability-aware (capaware_rule_off vs capaware_rule_on)

### cicids2017_distrinet

| victim | budget | condition | Valid ASR (validator_v2 + rule) mean (min-max) | Δ vs ref (pp) | Valid ASR (validator_v2) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| cnn | p75 | capaware_rule_off | 13.47% (13.47%-13.47%) | - | 13.47% | 13.47% | - | - |
| cnn | p75 | capaware_rule_on | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 13.47% | 0 / 0 | 1 |
| cnn | unb | capaware_rule_off | 59.94% (59.94%-59.94%) | - | 59.94% | 59.94% | - | - |
| cnn | unb | capaware_rule_on | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 59.94% | 0 / 0 | 1 |
| ft_transformer | p75 | capaware_rule_off | 0.12% (0.12%-0.12%) | - | 0.12% | 0.12% | - | - |
| ft_transformer | p75 | capaware_rule_on | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0.12% | 0 / 0 | 1 |
| ft_transformer | unb | capaware_rule_off | 0.55% (0.53%-0.56%) | - | 0.55% | 0.55% | - | - |
| ft_transformer | unb | capaware_rule_on | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 0.55% | 0 / 0 | 1 |
| mlp | p75 | capaware_rule_off | 4.09% (4.09%-4.09%) | - | 4.09% | 4.09% | - | - |
| mlp | p75 | capaware_rule_on | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 4.09% | 0 / 0 | 1 |
| mlp | unb | capaware_rule_off | 22.97% (22.97%-22.97%) | - | 22.97% | 22.97% | - | - |
| mlp | unb | capaware_rule_on | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 22.97% | 0 / 0 | 1 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR (validator_v2 + rule) mean (min-max) | Δ vs ref (pp) | Valid ASR (validator_v2) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | capaware_rule_off | 1.16% (1.16%-1.16%) | - | 1.16% | 1.16% | - | - |
| cnn-s42 | p75 | capaware_rule_on | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 1.16% | 0 / 0 | 1 |
| cnn-s42 | unb | capaware_rule_off | 26.19% (26.16%-26.22%) | - | 26.19% | 26.19% | - | - |
| cnn-s42 | unb | capaware_rule_on | 26.19% (26.16%-26.22%) | +0.00 | 26.19% | 26.19% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | capaware_rule_off | 0.00% (0.00%-0.00%) | - | 0.00% | 0.00% | - | - |
| ft_transformer-s42 | p75 | capaware_rule_on | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | capaware_rule_off | 0.12% (0.12%-0.12%) | - | 0.12% | 0.12% | - | - |
| ft_transformer-s42 | unb | capaware_rule_on | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0.12% | 0 / 0 | 1 |
| mlp-s42 | p75 | capaware_rule_off | 2.53% (2.53%-2.53%) | - | 2.53% | 2.53% | - | - |
| mlp-s42 | p75 | capaware_rule_on | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 2.53% | 0 / 0 | 1 |
| mlp-s42 | unb | capaware_rule_off | 44.36% (44.34%-44.38%) | - | 44.36% | 44.36% | - | - |
| mlp-s42 | unb | capaware_rule_on | 44.36% (44.34%-44.38%) | +0.00 | 44.36% | 44.36% | 0 / 0 | 1 |

## PrimAttack variant: capability-ablated (nocap_rule_off vs nocap_rule_on)

### cicids2017_distrinet

| victim | budget | condition | Valid ASR (validator_v2 + rule) mean (min-max) | Δ vs ref (pp) | Valid ASR (validator_v2) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| cnn | p75 | nocap_rule_off | 2.70% (2.53%-2.84%) | - | 2.70% | 49.07% | - | - |
| cnn | p75 | nocap_rule_on | 2.70% (2.53%-2.84%) | +0.00 | 2.70% | 49.07% | 0 / 0 | 1 |
| cnn | unb | nocap_rule_off | 7.25% (6.91%-7.50%) | - | 7.25% | 98.16% | - | - |
| cnn | unb | nocap_rule_on | 7.25% (6.91%-7.50%) | +0.00 | 7.25% | 98.16% | 0 / 0 | 1 |
| ft_transformer | p75 | nocap_rule_off | 0.03% (0.03%-0.03%) | - | 0.03% | 25.25% | - | - |
| ft_transformer | p75 | nocap_rule_on | 0.03% (0.03%-0.03%) | +0.00 | 0.03% | 25.25% | 0 / 0 | 1 |
| ft_transformer | unb | nocap_rule_off | 0.03% (0.03%-0.03%) | - | 0.03% | 25.69% | - | - |
| ft_transformer | unb | nocap_rule_on | 0.03% (0.03%-0.03%) | +0.00 | 0.03% | 25.69% | 0 / 0 | 1 |
| mlp | p75 | nocap_rule_off | 0.19% (0.16%-0.22%) | - | 0.19% | 13.97% | - | - |
| mlp | p75 | nocap_rule_on | 0.19% (0.16%-0.22%) | +0.00 | 0.19% | 13.97% | 0 / 0 | 1 |
| mlp | unb | nocap_rule_off | 0.71% (0.69%-0.75%) | - | 0.71% | 72.62% | - | - |
| mlp | unb | nocap_rule_on | 0.71% (0.69%-0.75%) | +0.00 | 0.71% | 72.62% | 0 / 0 | 1 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR (validator_v2 + rule) mean (min-max) | Δ vs ref (pp) | Valid ASR (validator_v2) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | nocap_rule_off | 0.03% (0.03%-0.03%) | - | 0.03% | 17.72% | - | - |
| cnn-s42 | p75 | nocap_rule_on | 0.03% (0.03%-0.03%) | +0.00 | 0.03% | 17.72% | 0 / 0 | 1 |
| cnn-s42 | unb | nocap_rule_off | 4.61% (4.41%-4.72%) | - | 4.87% | 68.11% | - | - |
| cnn-s42 | unb | nocap_rule_on | 4.61% (4.41%-4.72%) | +0.00 | 4.86% | 68.11% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | nocap_rule_off | 0.00% (0.00%-0.00%) | - | 0.00% | 0.36% | - | - |
| ft_transformer-s42 | p75 | nocap_rule_on | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0.36% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | nocap_rule_off | 0.09% (0.09%-0.09%) | - | 0.34% | 1.79% | - | - |
| ft_transformer-s42 | unb | nocap_rule_on | 0.09% (0.09%-0.09%) | +0.00 | 0.34% | 1.79% | 0 / 0 | 1 |
| mlp-s42 | p75 | nocap_rule_off | 0.20% (0.19%-0.22%) | - | 0.22% | 31.69% | - | - |
| mlp-s42 | p75 | nocap_rule_on | 0.20% (0.19%-0.22%) | +0.00 | 0.21% | 31.69% | 0 / 0 | 1 |
| mlp-s42 | unb | nocap_rule_off | 23.30% (23.22%-23.44%) | - | 38.52% | 66.61% | - | - |
| mlp-s42 | unb | nocap_rule_on | 23.30% (23.22%-23.44%) | +0.00 | 34.59% | 66.61% | 0 / 0 | 1 |

## validator_v2 successes the rule rejects (all seeds and classes pooled)

| dataset | victim | budget | arm | validator_v2 successes | rejected by the rule | validator_v2 + rule successes |
|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | capaware_rule_off | 1293 | 0 | 1293 |
| cicids2017_distrinet | cnn | p75 | capaware_rule_on | 1293 | 0 | 1293 |
| cicids2017_distrinet | cnn | p75 | nocap_rule_off | 259 | 0 | 259 |
| cicids2017_distrinet | cnn | p75 | nocap_rule_on | 259 | 0 | 259 |
| cicids2017_distrinet | cnn | unb | capaware_rule_off | 5754 | 0 | 5754 |
| cicids2017_distrinet | cnn | unb | capaware_rule_on | 5754 | 0 | 5754 |
| cicids2017_distrinet | cnn | unb | nocap_rule_off | 696 | 0 | 696 |
| cicids2017_distrinet | cnn | unb | nocap_rule_on | 696 | 0 | 696 |
| cicids2017_distrinet | ft_transformer | p75 | capaware_rule_off | 12 | 0 | 12 |
| cicids2017_distrinet | ft_transformer | p75 | capaware_rule_on | 12 | 0 | 12 |
| cicids2017_distrinet | ft_transformer | p75 | nocap_rule_off | 3 | 0 | 3 |
| cicids2017_distrinet | ft_transformer | p75 | nocap_rule_on | 3 | 0 | 3 |
| cicids2017_distrinet | ft_transformer | unb | capaware_rule_off | 53 | 0 | 53 |
| cicids2017_distrinet | ft_transformer | unb | capaware_rule_on | 53 | 0 | 53 |
| cicids2017_distrinet | ft_transformer | unb | nocap_rule_off | 3 | 0 | 3 |
| cicids2017_distrinet | ft_transformer | unb | nocap_rule_on | 3 | 0 | 3 |
| cicids2017_distrinet | mlp | p75 | capaware_rule_off | 393 | 0 | 393 |
| cicids2017_distrinet | mlp | p75 | capaware_rule_on | 393 | 0 | 393 |
| cicids2017_distrinet | mlp | p75 | nocap_rule_off | 18 | 0 | 18 |
| cicids2017_distrinet | mlp | p75 | nocap_rule_on | 18 | 0 | 18 |
| cicids2017_distrinet | mlp | unb | capaware_rule_off | 2205 | 0 | 2205 |
| cicids2017_distrinet | mlp | unb | capaware_rule_on | 2205 | 0 | 2205 |
| cicids2017_distrinet | mlp | unb | nocap_rule_off | 68 | 0 | 68 |
| cicids2017_distrinet | mlp | unb | nocap_rule_on | 68 | 0 | 68 |
| cicids2018_distrinet | cnn-s42 | p75 | capaware_rule_off | 111 | 0 | 111 |
| cicids2018_distrinet | cnn-s42 | p75 | capaware_rule_on | 111 | 0 | 111 |
| cicids2018_distrinet | cnn-s42 | p75 | nocap_rule_off | 3 | 0 | 3 |
| cicids2018_distrinet | cnn-s42 | p75 | nocap_rule_on | 3 | 0 | 3 |
| cicids2018_distrinet | cnn-s42 | unb | capaware_rule_off | 2514 | 0 | 2514 |
| cicids2018_distrinet | cnn-s42 | unb | capaware_rule_on | 2514 | 0 | 2514 |
| cicids2018_distrinet | cnn-s42 | unb | nocap_rule_off | 468 | 25 | 443 |
| cicids2018_distrinet | cnn-s42 | unb | nocap_rule_on | 467 | 24 | 443 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | capaware_rule_off | 0 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | capaware_rule_on | 0 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | nocap_rule_off | 0 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | nocap_rule_on | 0 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | capaware_rule_off | 12 | 0 | 12 |
| cicids2018_distrinet | ft_transformer-s42 | unb | capaware_rule_on | 12 | 0 | 12 |
| cicids2018_distrinet | ft_transformer-s42 | unb | nocap_rule_off | 33 | 24 | 9 |
| cicids2018_distrinet | ft_transformer-s42 | unb | nocap_rule_on | 33 | 24 | 9 |
| cicids2018_distrinet | mlp-s42 | p75 | capaware_rule_off | 243 | 0 | 243 |
| cicids2018_distrinet | mlp-s42 | p75 | capaware_rule_on | 243 | 0 | 243 |
| cicids2018_distrinet | mlp-s42 | p75 | nocap_rule_off | 21 | 2 | 19 |
| cicids2018_distrinet | mlp-s42 | p75 | nocap_rule_on | 20 | 1 | 19 |
| cicids2018_distrinet | mlp-s42 | unb | capaware_rule_off | 4259 | 0 | 4259 |
| cicids2018_distrinet | mlp-s42 | unb | capaware_rule_on | 4259 | 0 | 4259 |
| cicids2018_distrinet | mlp-s42 | unb | nocap_rule_off | 3698 | 1461 | 2237 |
| cicids2018_distrinet | mlp-s42 | unb | nocap_rule_on | 3321 | 1084 | 2237 |

## Rule-off arms reproduce the earlier runs

| arm | same as | dataset | cells | flows | identical adversarial flow |
|---|---|---|---|---|---|
| capaware_rule_off | reference/reference | cicids2017_distrinet | 72 | 57600 | 57600 |
| capaware_rule_off | reference/reference | cicids2018_distrinet | 72 | 57600 | 57600 |
| nocap_rule_off | D6_capability_inference/no_capability | cicids2017_distrinet | 72 | 57600 | 57600 |
| nocap_rule_off | D6_capability_inference/no_capability | cicids2018_distrinet | 72 | 57600 | 57600 |

## Rule on genuine flows (all splits, all classes incl. Benign)

| dataset | split | flows | single-forward-packet flows | violations |
|---|---|---|---|---|
| cicids2017_distrinet | test | 312056 | 69812 | 0 |
| cicids2017_distrinet | train | 1456265 | 311781 | 0 |
| cicids2017_distrinet | val | 312058 | 67199 | 0 |
| cicids2018_distrinet | test | 125032 | 23411 | 0 |
| cicids2018_distrinet | train | 583487 | 120798 | 0 |
| cicids2018_distrinet | val | 125033 | 23538 | 0 |

Per-class counts: `results/genuine_flow_check.csv`.

