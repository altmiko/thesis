# A3 - Refinement loss (margin vs CE vs DLR)

Outcome: **Valid ASR** over the frozen clean-correct flows (4 classes x 800 per attack seed). Mean over attack seeds 42/2024/2026 with the seed range. Paired McNemar vs the reference arm at seed 42, Holm over this table's comparisons.

### cicids2017_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn | p75 | reference | 13.47% (13.47%-13.47%) | - | 13.47% | - | - |
| cnn | p75 | loss_ce | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | loss_dlr | 12.98% (12.94%-13.03%) | -0.49 | 12.98% | 0 / 16 | 0.000732 |
| cnn | unb | reference | 59.94% (59.94%-59.94%) | - | 59.94% | - | - |
| cnn | unb | loss_ce | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | loss_dlr | 59.48% (59.38%-59.62%) | -0.46 | 59.48% | 0 / 10 | 0.0449 |
| ft_transformer | p75 | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer | p75 | loss_ce | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | loss_dlr | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | unb | reference | 0.55% (0.53%-0.56%) | - | 0.55% | - | - |
| ft_transformer | unb | loss_ce | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 0 / 0 | 1 |
| ft_transformer | unb | loss_dlr | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 0 / 0 | 1 |
| mlp | p75 | reference | 4.09% (4.09%-4.09%) | - | 4.09% | - | - |
| mlp | p75 | loss_ce | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | loss_dlr | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | unb | reference | 22.97% (22.97%-22.97%) | - | 22.97% | - | - |
| mlp | unb | loss_ce | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | loss_dlr | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | reference | 1.16% (1.16%-1.16%) | - | 1.16% | - | - |
| cnn-s42 | p75 | loss_ce | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | loss_dlr | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | unb | reference | 26.19% (26.16%-26.22%) | - | 26.19% | - | - |
| cnn-s42 | unb | loss_ce | 26.25% (26.22%-26.31%) | +0.06 | 26.25% | 2 / 0 | 1 |
| cnn-s42 | unb | loss_dlr | 26.21% (26.19%-26.22%) | +0.02 | 26.21% | 2 / 0 | 1 |
| ft_transformer-s42 | p75 | reference | 0.00% (0.00%-0.00%) | - | 0.00% | - | - |
| ft_transformer-s42 | p75 | loss_ce | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | loss_dlr | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer-s42 | unb | loss_ce | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | loss_dlr | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| mlp-s42 | p75 | reference | 2.53% (2.53%-2.53%) | - | 2.53% | - | - |
| mlp-s42 | p75 | loss_ce | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | loss_dlr | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | unb | reference | 44.36% (44.34%-44.38%) | - | 44.36% | - | - |
| mlp-s42 | unb | loss_ce | 44.36% (44.34%-44.38%) | +0.00 | 44.36% | 0 / 0 | 1 |
| mlp-s42 | unb | loss_dlr | 44.32% (44.31%-44.34%) | -0.04 | 44.32% | 0 / 1 | 1 |

## Zero gradients during refinement (all seeds and classes)

A step counts as zero when the loss gradient is exactly 0 on every free control coordinate of the row.

| dataset | victim | budget | condition | gradient steps | zero-gradient steps | fraction |
|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | cnn | p75 | loss_ce | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | cnn | p75 | loss_dlr | 899922 | 3317 | 0.0037 |
| cicids2017_distrinet | cnn | unb | reference | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | cnn | unb | loss_ce | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | cnn | unb | loss_dlr | 899922 | 35487 | 0.0394 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 900303 | 0 | 0.0000 |
| cicids2017_distrinet | ft_transformer | p75 | loss_ce | 900303 | 0 | 0.0000 |
| cicids2017_distrinet | ft_transformer | p75 | loss_dlr | 900303 | 0 | 0.0000 |
| cicids2017_distrinet | ft_transformer | unb | reference | 900303 | 0 | 0.0000 |
| cicids2017_distrinet | ft_transformer | unb | loss_ce | 900303 | 0 | 0.0000 |
| cicids2017_distrinet | ft_transformer | unb | loss_dlr | 900303 | 0 | 0.0000 |
| cicids2017_distrinet | mlp | p75 | reference | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | mlp | p75 | loss_ce | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | mlp | p75 | loss_dlr | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | mlp | unb | reference | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | mlp | unb | loss_ce | 899922 | 0 | 0.0000 |
| cicids2017_distrinet | mlp | unb | loss_dlr | 899922 | 14053 | 0.0156 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 910590 | 0 | 0.0000 |
| cicids2018_distrinet | cnn-s42 | p75 | loss_ce | 910590 | 0 | 0.0000 |
| cicids2018_distrinet | cnn-s42 | p75 | loss_dlr | 910590 | 0 | 0.0000 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 910590 | 0 | 0.0000 |
| cicids2018_distrinet | cnn-s42 | unb | loss_ce | 910590 | 0 | 0.0000 |
| cicids2018_distrinet | cnn-s42 | unb | loss_dlr | 910590 | 118459 | 0.1301 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | loss_ce | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | loss_dlr | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | ft_transformer-s42 | unb | loss_ce | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | ft_transformer-s42 | unb | loss_dlr | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | mlp-s42 | p75 | loss_ce | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | mlp-s42 | p75 | loss_dlr | 910971 | 1388 | 0.0015 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | mlp-s42 | unb | loss_ce | 910971 | 0 | 0.0000 |
| cicids2018_distrinet | mlp-s42 | unb | loss_dlr | 910971 | 149068 | 0.1636 |

## Cost profile of the successes (`valid_success`, all seeds pooled)

| dataset | victim | budget | condition | successes | median norm. cost | median rel. duration change | median delay (µs) | median shape | frac. using padding | median added bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.939 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | loss_ce | 1293 | 0.6 | 0.41 | 9.81e+05 | 0.933 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | loss_dlr | 1246 | 0.611 | 0.431 | 9.76e+05 | 0.923 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | reference | 5754 | 0.379 | 6.01 | 7.76e+06 | 0.691 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | loss_ce | 5754 | 0.382 | 6 | 8.05e+06 | 0.697 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | loss_dlr | 5710 | 0.383 | 6.12 | 8.6e+06 | 0.7 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | loss_ce | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | loss_dlr | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | reference | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | loss_ce | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | loss_dlr | 53 | 0.05 | 767 | 2.67e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | reference | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | loss_ce | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | loss_dlr | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | reference | 2205 | 0.3 | 25.7 | 5.2e+06 | 0.664 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | loss_ce | 2205 | 0.3 | 25.7 | 5.19e+06 | 0.663 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | loss_dlr | 2205 | 0.3 | 26 | 5.18e+06 | 0.633 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | loss_ce | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | loss_dlr | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 2514 | 0.166 | 51.2 | 1.86e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | loss_ce | 2520 | 0.164 | 50.7 | 1.85e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | loss_dlr | 2516 | 0.162 | 49.6 | 1.81e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | loss_ce | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | loss_dlr | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | loss_ce | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | loss_dlr | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | loss_ce | 243 | 0.6 | 0.784 | 4.71e+05 | 0.945 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | loss_dlr | 243 | 0.6 | 0.786 | 4.86e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | loss_ce | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | loss_dlr | 4255 | 0.5 | 187 | 4.67e+07 | 0.12 | 0 | 0 |

