# B1 - Convergence (steps x restarts x query budget)

Outcome: **Valid ASR** over the frozen clean-correct flows (4 classes x 800 per attack seed). Mean over attack seeds 42/2024/2026 with the seed range. Paired McNemar vs the reference arm at seed 42, Holm over this table's comparisons.

### cicids2017_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn | p75 | reference | 13.47% (13.47%-13.47%) | - | 13.47% | - | - |
| cnn | p75 | steps80_eval512 | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | steps160_eval1024 | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | steps40_eval512 | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | steps40_eval1024 | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | unb | reference | 59.94% (59.94%-59.94%) | - | 59.94% | - | - |
| cnn | unb | steps80_eval512 | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | steps160_eval1024 | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | steps40_eval512 | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | steps40_eval1024 | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| ft_transformer | p75 | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer | p75 | steps80_eval512 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | steps160_eval1024 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | steps40_eval512 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | steps40_eval1024 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | unb | reference | 0.55% (0.53%-0.56%) | - | 0.55% | - | - |
| ft_transformer | unb | steps80_eval512 | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 0 / 0 | 1 |
| ft_transformer | unb | steps160_eval1024 | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 0 / 0 | 1 |
| ft_transformer | unb | steps40_eval512 | 0.57% (0.56%-0.59%) | +0.02 | 0.57% | 0 / 0 | 1 |
| ft_transformer | unb | steps40_eval1024 | 0.58% (0.56%-0.59%) | +0.03 | 0.58% | 1 / 0 | 1 |
| mlp | p75 | reference | 4.09% (4.09%-4.09%) | - | 4.09% | - | - |
| mlp | p75 | steps80_eval512 | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | steps160_eval1024 | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | steps40_eval512 | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | steps40_eval1024 | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | unb | reference | 22.97% (22.97%-22.97%) | - | 22.97% | - | - |
| mlp | unb | steps80_eval512 | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | steps160_eval1024 | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | steps40_eval512 | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | steps40_eval1024 | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | reference | 1.16% (1.16%-1.16%) | - | 1.16% | - | - |
| cnn-s42 | p75 | steps80_eval512 | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | steps160_eval1024 | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | steps40_eval512 | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | steps40_eval1024 | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | unb | reference | 26.19% (26.16%-26.22%) | - | 26.19% | - | - |
| cnn-s42 | unb | steps80_eval512 | 26.19% (26.16%-26.22%) | +0.00 | 26.19% | 0 / 0 | 1 |
| cnn-s42 | unb | steps160_eval1024 | 26.19% (26.16%-26.22%) | +0.00 | 26.19% | 0 / 0 | 1 |
| cnn-s42 | unb | steps40_eval512 | 26.21% (26.19%-26.25%) | +0.02 | 26.21% | 1 / 0 | 1 |
| cnn-s42 | unb | steps40_eval1024 | 26.28% (26.25%-26.31%) | +0.09 | 26.28% | 5 / 0 | 1 |
| ft_transformer-s42 | p75 | reference | 0.00% (0.00%-0.00%) | - | 0.00% | - | - |
| ft_transformer-s42 | p75 | steps80_eval512 | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | steps160_eval1024 | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | steps40_eval512 | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | steps40_eval1024 | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer-s42 | unb | steps80_eval512 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | steps160_eval1024 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | steps40_eval512 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | steps40_eval1024 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| mlp-s42 | p75 | reference | 2.53% (2.53%-2.53%) | - | 2.53% | - | - |
| mlp-s42 | p75 | steps80_eval512 | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | steps160_eval1024 | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | steps40_eval512 | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | steps40_eval1024 | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | unb | reference | 44.36% (44.34%-44.38%) | - | 44.36% | - | - |
| mlp-s42 | unb | steps80_eval512 | 44.37% (44.38%-44.38%) | +0.01 | 44.37% | 1 / 0 | 1 |
| mlp-s42 | unb | steps160_eval1024 | 44.37% (44.38%-44.38%) | +0.01 | 44.37% | 1 / 0 | 1 |
| mlp-s42 | unb | steps40_eval512 | 44.37% (44.38%-44.38%) | +0.01 | 44.37% | 1 / 0 | 1 |
| mlp-s42 | unb | steps40_eval1024 | 44.37% (44.38%-44.38%) | +0.01 | 44.37% | 1 / 0 | 1 |

## Anytime curve: Valid ASR within the first k victim evaluations per flow (mean over seeds)

| dataset | victim | budget | arm | ≤1 | ≤2 | ≤4 | ≤8 | ≤16 | ≤32 | ≤64 | ≤128 | ≤256 | ≤512 | ≤1024 | final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 0.00 | 0.00 | 0.03 | 0.50 | 2.28 | 4.50 | 4.50 | 13.45 | 13.47 | 13.47 | 13.47 | 13.47 |
| cicids2017_distrinet | cnn | p75 | steps40_eval1024 | 0.00 | 0.00 | 0.03 | 0.50 | 2.28 | 4.50 | 4.50 | 13.45 | 13.47 | 13.47 | 13.47 | 13.47 |
| cicids2017_distrinet | cnn | p75 | steps160_eval1024 | 0.00 | 0.00 | 0.03 | 0.50 | 2.28 | 4.50 | 4.50 | 4.50 | 4.50 | 13.45 | 13.47 | 13.47 |
| cicids2017_distrinet | cnn | unb | reference | 0.00 | 0.00 | 3.00 | 13.22 | 31.81 | 34.28 | 34.28 | 59.93 | 59.94 | 59.94 | 59.94 | 59.94 |
| cicids2017_distrinet | cnn | unb | steps40_eval1024 | 0.00 | 0.00 | 3.00 | 13.22 | 31.81 | 34.28 | 34.28 | 59.93 | 59.94 | 59.94 | 59.94 | 59.94 |
| cicids2017_distrinet | cnn | unb | steps160_eval1024 | 0.00 | 0.00 | 3.00 | 13.22 | 31.81 | 34.28 | 34.28 | 34.28 | 34.28 | 59.93 | 59.94 | 59.94 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 0.00 | 0.00 | 0.00 | 0.09 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | p75 | steps40_eval1024 | 0.00 | 0.00 | 0.00 | 0.09 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | p75 | steps160_eval1024 | 0.00 | 0.00 | 0.00 | 0.09 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | unb | reference | 0.00 | 0.00 | 0.25 | 0.28 | 0.41 | 0.44 | 0.50 | 0.53 | 0.55 | 0.55 | 0.55 | 0.55 |
| cicids2017_distrinet | ft_transformer | unb | steps40_eval1024 | 0.00 | 0.00 | 0.25 | 0.28 | 0.41 | 0.44 | 0.50 | 0.53 | 0.55 | 0.57 | 0.58 | 0.58 |
| cicids2017_distrinet | ft_transformer | unb | steps160_eval1024 | 0.00 | 0.00 | 0.25 | 0.28 | 0.41 | 0.44 | 0.44 | 0.44 | 0.50 | 0.53 | 0.55 | 0.55 |
| cicids2017_distrinet | mlp | p75 | reference | 0.00 | 0.00 | 0.00 | 0.09 | 2.00 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 |
| cicids2017_distrinet | mlp | p75 | steps40_eval1024 | 0.00 | 0.00 | 0.00 | 0.09 | 2.00 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 |
| cicids2017_distrinet | mlp | p75 | steps160_eval1024 | 0.00 | 0.00 | 0.00 | 0.09 | 2.00 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 | 4.09 |
| cicids2017_distrinet | mlp | unb | reference | 0.00 | 0.00 | 1.94 | 9.22 | 19.09 | 22.94 | 22.94 | 22.97 | 22.97 | 22.97 | 22.97 | 22.97 |
| cicids2017_distrinet | mlp | unb | steps40_eval1024 | 0.00 | 0.00 | 1.94 | 9.22 | 19.09 | 22.94 | 22.94 | 22.97 | 22.97 | 22.97 | 22.97 | 22.97 |
| cicids2017_distrinet | mlp | unb | steps160_eval1024 | 0.00 | 0.00 | 1.94 | 9.22 | 19.09 | 22.94 | 22.94 | 22.94 | 22.94 | 22.97 | 22.97 | 22.97 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 1.06 | 1.06 | 1.16 | 1.16 | 1.16 | 1.16 | 1.16 |
| cicids2018_distrinet | cnn-s42 | p75 | steps40_eval1024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 1.06 | 1.06 | 1.16 | 1.16 | 1.16 | 1.16 | 1.16 |
| cicids2018_distrinet | cnn-s42 | p75 | steps160_eval1024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 | 1.06 | 1.06 | 1.06 | 1.06 | 1.16 | 1.16 | 1.16 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 0.00 | 0.00 | 9.97 | 25.41 | 25.78 | 25.87 | 25.87 | 26.17 | 26.19 | 26.19 | 26.19 | 26.19 |
| cicids2018_distrinet | cnn-s42 | unb | steps40_eval1024 | 0.00 | 0.00 | 9.97 | 25.41 | 25.77 | 25.86 | 25.86 | 26.16 | 26.18 | 26.20 | 26.28 | 26.28 |
| cicids2018_distrinet | cnn-s42 | unb | steps160_eval1024 | 0.00 | 0.00 | 9.97 | 25.41 | 25.78 | 25.87 | 25.87 | 25.87 | 25.87 | 26.17 | 26.19 | 26.19 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | steps40_eval1024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | steps160_eval1024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 0.00 | 0.00 | 0.09 | 0.09 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2018_distrinet | ft_transformer-s42 | unb | steps40_eval1024 | 0.00 | 0.00 | 0.09 | 0.09 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2018_distrinet | ft_transformer-s42 | unb | steps160_eval1024 | 0.00 | 0.00 | 0.09 | 0.09 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 0.00 | 0.00 | 0.03 | 0.22 | 1.09 | 2.16 | 2.16 | 2.53 | 2.53 | 2.53 | 2.53 | 2.53 |
| cicids2018_distrinet | mlp-s42 | p75 | steps40_eval1024 | 0.00 | 0.00 | 0.03 | 0.22 | 1.09 | 2.16 | 2.16 | 2.53 | 2.53 | 2.53 | 2.53 | 2.53 |
| cicids2018_distrinet | mlp-s42 | p75 | steps160_eval1024 | 0.00 | 0.00 | 0.03 | 0.22 | 1.09 | 2.16 | 2.16 | 2.16 | 2.16 | 2.53 | 2.53 | 2.53 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 0.00 | 0.00 | 11.91 | 20.16 | 36.62 | 43.94 | 43.94 | 44.34 | 44.36 | 44.36 | 44.36 | 44.36 |
| cicids2018_distrinet | mlp-s42 | unb | steps40_eval1024 | 0.00 | 0.00 | 11.91 | 20.16 | 36.62 | 43.94 | 43.94 | 44.34 | 44.36 | 44.37 | 44.37 | 44.37 |
| cicids2018_distrinet | mlp-s42 | unb | steps160_eval1024 | 0.00 | 0.00 | 11.91 | 20.16 | 36.62 | 43.94 | 43.94 | 43.94 | 43.94 | 44.34 | 44.37 | 44.37 |

Values in %. The reference spends at most 256 evaluations, so its curve is flat after 256.

