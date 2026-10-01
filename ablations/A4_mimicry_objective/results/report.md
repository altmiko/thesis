# A4 - Mimicry objective vs victim-margin objective

Outcome: **Valid ASR** over the frozen clean-correct flows (4 classes x 800 per attack seed). Mean over attack seeds 42/2024/2026 with the seed range. Paired McNemar vs the reference arm at seed 42, Holm over this table's comparisons.

### cicids2017_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn | p75 | reference | 13.47% (13.47%-13.47%) | - | 13.47% | - | - |
| cnn | p75 | loss_mimicry | 5.14% (5.09%-5.16%) | -8.33 | 5.14% | 0 / 268 | 8.43e-59 |
| cnn | unb | reference | 59.94% (59.94%-59.94%) | - | 59.94% | - | - |
| cnn | unb | loss_mimicry | 39.17% (39.06%-39.25%) | -20.77 | 39.17% | 0 / 668 | 8.88e-146 |
| ft_transformer | p75 | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer | p75 | loss_mimicry | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | unb | reference | 0.55% (0.53%-0.56%) | - | 0.55% | - | - |
| ft_transformer | unb | loss_mimicry | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 1 / 1 | 1 |
| mlp | p75 | reference | 4.09% (4.09%-4.09%) | - | 4.09% | - | - |
| mlp | p75 | loss_mimicry | 1.22% (1.19%-1.25%) | -2.88 | 1.22% | 0 / 93 | 1.14e-20 |
| mlp | unb | reference | 22.97% (22.97%-22.97%) | - | 22.97% | - | - |
| mlp | unb | loss_mimicry | 15.58% (15.44%-15.84%) | -7.39 | 15.58% | 0 / 228 | 3.99e-50 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | reference | 1.16% (1.16%-1.16%) | - | 1.16% | - | - |
| cnn-s42 | p75 | loss_mimicry | 0.11% (0.06%-0.19%) | -1.04 | 0.11% | 0 / 34 | 9.11e-08 |
| cnn-s42 | unb | reference | 26.19% (26.16%-26.22%) | - | 26.19% | - | - |
| cnn-s42 | unb | loss_mimicry | 26.40% (26.28%-26.47%) | +0.21 | 26.40% | 12 / 8 | 1 |
| ft_transformer-s42 | p75 | reference | 0.00% (0.00%-0.00%) | - | 0.00% | - | - |
| ft_transformer-s42 | p75 | loss_mimicry | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer-s42 | unb | loss_mimicry | 0.10% (0.09%-0.12%) | -0.02 | 0.10% | 0 / 1 | 1 |
| mlp-s42 | p75 | reference | 2.53% (2.53%-2.53%) | - | 2.53% | - | - |
| mlp-s42 | p75 | loss_mimicry | 1.01% (0.94%-1.09%) | -1.52 | 1.01% | 0 / 49 | 4.92e-11 |
| mlp-s42 | unb | reference | 44.36% (44.34%-44.38%) | - | 44.36% | - | - |
| mlp-s42 | unb | loss_mimicry | 31.52% (31.16%-31.78%) | -12.84 | 31.52% | 0 / 402 | 6.05e-88 |

## Cost profile of the successes (`valid_success`, all seeds pooled)

| dataset | victim | budget | condition | successes | median norm. cost | median rel. duration change | median delay (µs) | median shape | frac. using padding | median added bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.939 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | loss_mimicry | 493 | 0.442 | 0.301 | 8.75e+05 | 0.495 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | reference | 5754 | 0.379 | 6.01 | 7.76e+06 | 0.691 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | loss_mimicry | 3760 | 0.259 | 7.23 | 4.13e+06 | 0.498 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | loss_mimicry | 12 | 0.22 | 0.0513 | 3.44e+03 | 0.4 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | reference | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | loss_mimicry | 53 | 0.0146 | 246 | 1.07e+06 | 0.39 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | reference | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | loss_mimicry | 117 | 0.502 | 0.379 | 1.3e+06 | 0.591 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | reference | 2205 | 0.3 | 25.7 | 5.2e+06 | 0.664 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | loss_mimicry | 1496 | 0.228 | 20 | 4.1e+06 | 0.418 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | loss_mimicry | 11 | 0.318 | 0.454 | 1.04e+05 | 0.544 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 2514 | 0.166 | 51.2 | 1.86e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | loss_mimicry | 2534 | 0.0821 | 26.1 | 9.54e+06 | 1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | loss_mimicry | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | loss_mimicry | 10 | 0.0426 | 493 | 2.04e+06 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | loss_mimicry | 97 | 0.422 | 0.527 | 6.03e+05 | 0.531 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | loss_mimicry | 3026 | 0.0753 | 112 | 2.86e+06 | 0.553 | 0 | 0 |

