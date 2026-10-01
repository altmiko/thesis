# A2 - Fixed vs learned delay allocation (shape)

Outcome: **Valid ASR** over the frozen clean-correct flows (4 classes x 800 per attack seed). Mean over attack seeds 42/2024/2026 with the seed range. Paired McNemar vs the reference arm at seed 42, Holm over this table's comparisons.

### cicids2017_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn | p75 | reference | 13.47% (13.47%-13.47%) | - | 13.47% | - | - |
| cnn | p75 | shape_fixed_0 | 0.06% (0.06%-0.06%) | -13.41 | 0.06% | 0 / 429 | 2.48e-93 |
| cnn | p75 | shape_fixed_0p5 | 7.44% (7.44%-7.44%) | -6.03 | 7.44% | 0 / 193 | 5.95e-42 |
| cnn | p75 | shape_fixed_1 | 13.41% (13.41%-13.41%) | -0.06 | 13.41% | 0 / 2 | 1 |
| cnn | unb | reference | 59.94% (59.94%-59.94%) | - | 59.94% | - | - |
| cnn | unb | shape_fixed_0 | 1.28% (1.28%-1.28%) | -58.66 | 1.28% | 0 / 1877 | 0 |
| cnn | unb | shape_fixed_0p5 | 53.66% (53.66%-53.66%) | -6.28 | 53.66% | 0 / 201 | 1.1e-43 |
| cnn | unb | shape_fixed_1 | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| ft_transformer | p75 | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer | p75 | shape_fixed_0 | 0.03% (0.03%-0.03%) | -0.09 | 0.03% | 0 / 3 | 1 |
| ft_transformer | p75 | shape_fixed_0p5 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | shape_fixed_1 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | unb | reference | 0.55% (0.53%-0.56%) | - | 0.55% | - | - |
| ft_transformer | unb | shape_fixed_0 | 0.25% (0.25%-0.25%) | -0.30 | 0.25% | 0 / 10 | 0.0391 |
| ft_transformer | unb | shape_fixed_0p5 | 0.53% (0.53%-0.53%) | -0.02 | 0.53% | 0 / 1 | 1 |
| ft_transformer | unb | shape_fixed_1 | 0.59% (0.59%-0.59%) | +0.04 | 0.59% | 1 / 0 | 1 |
| mlp | p75 | reference | 4.09% (4.09%-4.09%) | - | 4.09% | - | - |
| mlp | p75 | shape_fixed_0 | 0.00% (0.00%-0.00%) | -4.09 | 0.00% | 0 / 131 | 1.96e-28 |
| mlp | p75 | shape_fixed_0p5 | 2.12% (2.12%-2.12%) | -1.97 | 2.12% | 0 / 63 | 1.47e-13 |
| mlp | p75 | shape_fixed_1 | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | unb | reference | 22.97% (22.97%-22.97%) | - | 22.97% | - | - |
| mlp | unb | shape_fixed_0 | 0.91% (0.91%-0.91%) | -22.06 | 0.91% | 0 / 706 | 1.41e-153 |
| mlp | unb | shape_fixed_0p5 | 19.25% (19.25%-19.25%) | -3.72 | 19.25% | 0 / 119 | 8e-26 |
| mlp | unb | shape_fixed_1 | 22.75% (22.75%-22.75%) | -0.22 | 22.75% | 0 / 7 | 0.297 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | reference | 1.16% (1.16%-1.16%) | - | 1.16% | - | - |
| cnn-s42 | p75 | shape_fixed_0 | 0.00% (0.00%-0.00%) | -1.16 | 0.00% | 0 / 37 | 7.8e-08 |
| cnn-s42 | p75 | shape_fixed_0p5 | 0.16% (0.16%-0.16%) | -1.00 | 0.16% | 0 / 32 | 9.35e-07 |
| cnn-s42 | p75 | shape_fixed_1 | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | unb | reference | 26.19% (26.16%-26.22%) | - | 26.19% | - | - |
| cnn-s42 | unb | shape_fixed_0 | 25.62% (25.62%-25.62%) | -0.56 | 25.62% | 0 / 17 | 0.00032 |
| cnn-s42 | unb | shape_fixed_0p5 | 26.01% (26.00%-26.03%) | -0.18 | 26.01% | 0 / 5 | 1 |
| cnn-s42 | unb | shape_fixed_1 | 26.27% (26.25%-26.28%) | +0.08 | 26.27% | 5 / 1 | 1 |
| ft_transformer-s42 | p75 | reference | 0.00% (0.00%-0.00%) | - | 0.00% | - | - |
| ft_transformer-s42 | p75 | shape_fixed_0 | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | shape_fixed_0p5 | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | shape_fixed_1 | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer-s42 | unb | shape_fixed_0 | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | shape_fixed_0p5 | 0.09% (0.09%-0.09%) | -0.03 | 0.09% | 0 / 1 | 1 |
| ft_transformer-s42 | unb | shape_fixed_1 | 0.09% (0.09%-0.09%) | -0.03 | 0.09% | 0 / 1 | 1 |
| mlp-s42 | p75 | reference | 2.53% (2.53%-2.53%) | - | 2.53% | - | - |
| mlp-s42 | p75 | shape_fixed_0 | 0.00% (0.00%-0.00%) | -2.53 | 0.00% | 0 / 81 | 1.67e-17 |
| mlp-s42 | p75 | shape_fixed_0p5 | 1.50% (1.50%-1.50%) | -1.03 | 1.50% | 0 / 33 | 5.84e-07 |
| mlp-s42 | p75 | shape_fixed_1 | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | unb | reference | 44.36% (44.34%-44.38%) | - | 44.36% | - | - |
| mlp-s42 | unb | shape_fixed_0 | 39.09% (39.09%-39.09%) | -5.27 | 39.09% | 0 / 168 | 1.65e-36 |
| mlp-s42 | unb | shape_fixed_0p5 | 42.59% (42.59%-42.59%) | -1.77 | 42.59% | 0 / 56 | 4.97e-12 |
| mlp-s42 | unb | shape_fixed_1 | 35.50% (35.50%-35.50%) | -8.86 | 35.50% | 1 / 284 | 4.03e-61 |

## Learned shape among the reference's timing successes (all seeds)

| dataset | victim | budget | timing successes | shape median | shape < 0.05 | shape > 0.95 | 0.05-0.95 |
|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | 1293 | 0.939 | 0.004 | 0.483 | 0.514 |
| cicids2017_distrinet | cnn | unb | 5754 | 0.691 | 0.000 | 0.219 | 0.780 |
| cicids2017_distrinet | ft_transformer | p75 | 12 | 0.399 | 0.000 | 0.083 | 0.917 |
| cicids2017_distrinet | ft_transformer | unb | 53 | 0.984 | 0.000 | 0.528 | 0.472 |
| cicids2017_distrinet | mlp | p75 | 393 | 0.937 | 0.000 | 0.478 | 0.522 |
| cicids2017_distrinet | mlp | unb | 2205 | 0.664 | 0.007 | 0.257 | 0.736 |
| cicids2018_distrinet | cnn-s42 | p75 | 111 | 1.000 | 0.000 | 0.874 | 0.126 |
| cicids2018_distrinet | cnn-s42 | unb | 2514 | 0.100 | 0.059 | 0.035 | 0.907 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 0 | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | unb | 12 | 0.100 | 0.250 | 0.000 | 0.750 |
| cicids2018_distrinet | mlp-s42 | p75 | 243 | 0.950 | 0.000 | 0.498 | 0.502 |
| cicids2018_distrinet | mlp-s42 | unb | 4259 | 0.000 | 0.564 | 0.018 | 0.418 |

## Cost profile of the successes (`valid_success`, all seeds pooled)

| dataset | victim | budget | condition | successes | median norm. cost | median rel. duration change | median delay (µs) | median shape | frac. using padding | median added bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.939 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | shape_fixed_0 | 6 | 0.3 | 0.16 | 360 | 0 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | shape_fixed_0p5 | 714 | 0.5 | 0.308 | 9.7e+05 | 0.5 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | shape_fixed_1 | 1287 | 0.418 | 0.275 | 6.22e+05 | 1 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | reference | 5754 | 0.379 | 6.01 | 7.76e+06 | 0.691 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | shape_fixed_0 | 123 | 0.1 | 713 | 4.43e+06 | 0 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | shape_fixed_0p5 | 5151 | 0.379 | 8.56 | 4.45e+06 | 0.5 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | shape_fixed_1 | 5754 | 0.2 | 3.28 | 2.72e+06 | 1 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | shape_fixed_0 | 3 | 0.6 | 0.14 | 2.9e+03 | 0 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | shape_fixed_0p5 | 12 | 0.173 | 0.0403 | 2.92e+03 | 0.5 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | shape_fixed_1 | 12 | 0.15 | 0.035 | 2.63e+03 | 1 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | reference | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | shape_fixed_0 | 24 | 0.05 | 1.03e+03 | 2.4e+06 | 0 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | shape_fixed_0p5 | 51 | 0.05 | 663 | 2.7e+06 | 0.5 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | shape_fixed_1 | 57 | 0.05 | 1.84e+03 | 2.88e+06 | 1 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | reference | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | shape_fixed_0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2017_distrinet | mlp | p75 | shape_fixed_0p5 | 204 | 0.663 | 0.481 | 1.68e+06 | 0.5 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | shape_fixed_1 | 393 | 0.5 | 0.391 | 9.42e+05 | 1 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | reference | 2205 | 0.3 | 25.7 | 5.2e+06 | 0.664 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | shape_fixed_0 | 87 | 0.1 | 3.03e+03 | 6.5e+06 | 0 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | shape_fixed_0p5 | 1848 | 0.2 | 16.1 | 4.12e+06 | 0.5 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | shape_fixed_1 | 2184 | 0.134 | 13.1 | 2.55e+06 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | shape_fixed_0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | cnn-s42 | p75 | shape_fixed_0p5 | 15 | 0.376 | 0.537 | 6.9e+04 | 0.5 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | shape_fixed_1 | 111 | 0.785 | 1.12 | 1.22e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 2514 | 0.166 | 51.2 | 1.86e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | shape_fixed_0 | 2460 | 0.2 | 60.7 | 2.34e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | shape_fixed_0p5 | 2497 | 0.1 | 31.4 | 1.17e+07 | 0.5 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | shape_fixed_1 | 2522 | 0.1 | 31.3 | 1.17e+07 | 1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | shape_fixed_0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | shape_fixed_0p5 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | shape_fixed_1 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | shape_fixed_0 | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | shape_fixed_0p5 | 9 | 0.1 | 1.65e+03 | 4.8e+06 | 0.5 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | shape_fixed_1 | 9 | 0.1 | 1.65e+03 | 4.8e+06 | 1 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | shape_fixed_0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | mlp-s42 | p75 | shape_fixed_0p5 | 144 | 0.477 | 0.596 | 4.82e+05 | 0.5 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | shape_fixed_1 | 243 | 0.5 | 0.659 | 3.85e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | shape_fixed_0 | 3753 | 0.5 | 187 | 5.55e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | shape_fixed_0p5 | 4089 | 0.4 | 172 | 4.67e+07 | 0.5 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | shape_fixed_1 | 3408 | 0.1 | 193 | 3.77e+06 | 1 | 0 | 0 |

