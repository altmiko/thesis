# A1 - Hybrid Search component leave-one-out

Outcome: **Valid ASR** over the frozen clean-correct flows (4 classes x 800 per attack seed). Mean over attack seeds 42/2024/2026 with the seed range. Paired McNemar vs the reference arm at seed 42, Holm over this table's comparisons.

### cicids2017_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn | p75 | reference | 13.47% (13.47%-13.47%) | - | 13.47% | - | - |
| cnn | p75 | no_padding_sweep | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | no_refinement | 0.00% (0.00%-0.00%) | -13.47 | 0.00% | 0 / 431 | 2.44e-93 |
| cnn | p75 | no_random_restarts | 4.50% (4.50%-4.50%) | -8.97 | 4.50% | 0 / 287 | 5.49e-62 |
| cnn | p75 | fixed_step | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | no_momentum | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | no_surrogate_floor | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | validator_post_hoc | 13.47% (13.47%-13.47%) | +0.00 | 13.47% | 0 / 0 | 1 |
| cnn | p75 | last_iterate | 12.33% (12.28%-12.44%) | -1.14 | 12.33% | 0 / 33 | 2.16e-06 |
| cnn | unb | reference | 59.94% (59.94%-59.94%) | - | 59.94% | - | - |
| cnn | unb | no_padding_sweep | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | no_refinement | 0.00% (0.00%-0.00%) | -59.94 | 0.00% | 0 / 1918 | 0 |
| cnn | unb | no_random_restarts | 34.28% (34.28%-34.28%) | -25.66 | 34.28% | 0 / 821 | 3.71e-178 |
| cnn | unb | fixed_step | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | no_momentum | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | no_surrogate_floor | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | validator_post_hoc | 59.94% (59.94%-59.94%) | +0.00 | 59.94% | 0 / 0 | 1 |
| cnn | unb | last_iterate | 59.03% (58.94%-59.13%) | -0.91 | 59.03% | 0 / 29 | 1.68e-05 |
| ft_transformer | p75 | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer | p75 | no_padding_sweep | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | no_refinement | 0.00% (0.00%-0.00%) | -0.12 | 0.00% | 0 / 4 | 1 |
| ft_transformer | p75 | no_random_restarts | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | fixed_step | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | no_momentum | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | no_surrogate_floor | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | validator_post_hoc | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | p75 | last_iterate | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer | unb | reference | 0.55% (0.53%-0.56%) | - | 0.55% | - | - |
| ft_transformer | unb | no_padding_sweep | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 0 / 0 | 1 |
| ft_transformer | unb | no_refinement | 0.00% (0.00%-0.00%) | -0.55 | 0.00% | 0 / 18 | 0.000626 |
| ft_transformer | unb | no_random_restarts | 0.50% (0.50%-0.50%) | -0.05 | 0.50% | 0 / 2 | 1 |
| ft_transformer | unb | fixed_step | 0.51% (0.50%-0.53%) | -0.04 | 0.51% | 0 / 2 | 1 |
| ft_transformer | unb | no_momentum | 0.59% (0.59%-0.59%) | +0.04 | 0.59% | 1 / 0 | 1 |
| ft_transformer | unb | no_surrogate_floor | 0.48% (0.47%-0.50%) | -0.07 | 0.48% | 0 / 2 | 1 |
| ft_transformer | unb | validator_post_hoc | 0.55% (0.53%-0.56%) | +0.00 | 0.55% | 0 / 0 | 1 |
| ft_transformer | unb | last_iterate | 0.32% (0.22%-0.38%) | -0.23 | 0.34% | 0 / 6 | 1 |
| mlp | p75 | reference | 4.09% (4.09%-4.09%) | - | 4.09% | - | - |
| mlp | p75 | no_padding_sweep | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | no_refinement | 0.00% (0.00%-0.00%) | -4.09 | 0.00% | 0 / 131 | 6.01e-28 |
| mlp | p75 | no_random_restarts | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | fixed_step | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | no_momentum | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | no_surrogate_floor | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | validator_post_hoc | 4.09% (4.09%-4.09%) | +0.00 | 4.09% | 0 / 0 | 1 |
| mlp | p75 | last_iterate | 3.89% (3.75%-3.97%) | -0.21 | 3.89% | 0 / 11 | 0.0771 |
| mlp | unb | reference | 22.97% (22.97%-22.97%) | - | 22.97% | - | - |
| mlp | unb | no_padding_sweep | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | no_refinement | 0.00% (0.00%-0.00%) | -22.97 | 0.00% | 0 / 735 | 1.83e-159 |
| mlp | unb | no_random_restarts | 22.94% (22.94%-22.94%) | -0.03 | 22.94% | 0 / 1 | 1 |
| mlp | unb | fixed_step | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | no_momentum | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | no_surrogate_floor | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | validator_post_hoc | 22.97% (22.97%-22.97%) | +0.00 | 22.97% | 0 / 0 | 1 |
| mlp | unb | last_iterate | 22.31% (22.25%-22.38%) | -0.66 | 22.32% | 0 / 19 | 0.000317 |

### cicids2018_distrinet

| victim | budget | condition | Valid ASR mean (min-max) | Δ vs ref (pp) | Raw ASR | seed-42 cond-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| cnn-s42 | p75 | reference | 1.16% (1.16%-1.16%) | - | 1.16% | - | - |
| cnn-s42 | p75 | no_padding_sweep | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | no_refinement | 0.00% (0.00%-0.00%) | -1.16 | 0.00% | 0 / 37 | 2.8e-07 |
| cnn-s42 | p75 | no_random_restarts | 1.06% (1.06%-1.06%) | -0.09 | 1.06% | 0 / 3 | 1 |
| cnn-s42 | p75 | fixed_step | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | no_momentum | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | no_surrogate_floor | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | validator_post_hoc | 1.16% (1.16%-1.16%) | +0.00 | 1.16% | 0 / 0 | 1 |
| cnn-s42 | p75 | last_iterate | 0.86% (0.81%-0.97%) | -0.29 | 0.86% | 0 / 6 | 1 |
| cnn-s42 | unb | reference | 26.19% (26.16%-26.22%) | - | 26.19% | - | - |
| cnn-s42 | unb | no_padding_sweep | 26.19% (26.16%-26.22%) | +0.00 | 26.19% | 0 / 0 | 1 |
| cnn-s42 | unb | no_refinement | 0.00% (0.00%-0.00%) | -26.19 | 0.00% | 0 / 837 | 1.25e-181 |
| cnn-s42 | unb | no_random_restarts | 25.87% (25.87%-25.87%) | -0.31 | 25.87% | 0 / 9 | 0.305 |
| cnn-s42 | unb | fixed_step | 26.18% (26.16%-26.22%) | -0.01 | 26.18% | 0 / 0 | 1 |
| cnn-s42 | unb | no_momentum | 26.23% (26.22%-26.25%) | +0.04 | 26.23% | 2 / 0 | 1 |
| cnn-s42 | unb | no_surrogate_floor | 26.18% (26.16%-26.22%) | -0.01 | 26.18% | 0 / 0 | 1 |
| cnn-s42 | unb | validator_post_hoc | 26.19% (26.16%-26.22%) | +0.00 | 26.19% | 0 / 0 | 1 |
| cnn-s42 | unb | last_iterate | 26.12% (26.12%-26.12%) | -0.06 | 26.12% | 0 / 1 | 1 |
| ft_transformer-s42 | p75 | reference | 0.00% (0.00%-0.00%) | - | 0.00% | - | - |
| ft_transformer-s42 | p75 | no_padding_sweep | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | no_refinement | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | no_random_restarts | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | fixed_step | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | no_momentum | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | no_surrogate_floor | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | validator_post_hoc | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | last_iterate | 0.00% (0.00%-0.00%) | +0.00 | 0.00% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | reference | 0.12% (0.12%-0.12%) | - | 0.12% | - | - |
| ft_transformer-s42 | unb | no_padding_sweep | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | no_refinement | 0.00% (0.00%-0.00%) | -0.12 | 0.00% | 0 / 4 | 1 |
| ft_transformer-s42 | unb | no_random_restarts | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | fixed_step | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | no_momentum | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | no_surrogate_floor | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | validator_post_hoc | 0.12% (0.12%-0.12%) | +0.00 | 0.12% | 0 / 0 | 1 |
| ft_transformer-s42 | unb | last_iterate | 0.09% (0.09%-0.09%) | -0.03 | 0.09% | 0 / 1 | 1 |
| mlp-s42 | p75 | reference | 2.53% (2.53%-2.53%) | - | 2.53% | - | - |
| mlp-s42 | p75 | no_padding_sweep | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | no_refinement | 0.00% (0.00%-0.00%) | -2.53 | 0.00% | 0 / 81 | 5.43e-17 |
| mlp-s42 | p75 | no_random_restarts | 2.16% (2.16%-2.16%) | -0.38 | 2.16% | 0 / 12 | 0.0391 |
| mlp-s42 | p75 | fixed_step | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | no_momentum | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | no_surrogate_floor | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | validator_post_hoc | 2.53% (2.53%-2.53%) | +0.00 | 2.53% | 0 / 0 | 1 |
| mlp-s42 | p75 | last_iterate | 2.26% (2.25%-2.28%) | -0.27 | 2.26% | 0 / 9 | 0.305 |
| mlp-s42 | unb | reference | 44.36% (44.34%-44.38%) | - | 44.36% | - | - |
| mlp-s42 | unb | no_padding_sweep | 44.36% (44.34%-44.38%) | +0.00 | 44.36% | 0 / 0 | 1 |
| mlp-s42 | unb | no_refinement | 0.00% (0.00%-0.00%) | -44.36 | 0.00% | 0 / 1419 | 4.04e-308 |
| mlp-s42 | unb | no_random_restarts | 43.94% (43.94%-43.94%) | -0.43 | 43.94% | 0 / 13 | 0.0198 |
| mlp-s42 | unb | fixed_step | 44.36% (44.34%-44.38%) | +0.00 | 44.36% | 0 / 0 | 1 |
| mlp-s42 | unb | no_momentum | 44.33% (44.31%-44.34%) | -0.03 | 44.33% | 0 / 1 | 1 |
| mlp-s42 | unb | no_surrogate_floor | 44.07% (44.03%-44.12%) | -0.29 | 44.07% | 0 / 9 | 0.305 |
| mlp-s42 | unb | validator_post_hoc | 44.36% (44.34%-44.38%) | +0.00 | 44.36% | 0 / 0 | 1 |
| mlp-s42 | unb | last_iterate | 42.69% (42.53%-42.88%) | -1.68 | 42.69% | 0 / 58 | 6.25e-12 |

## Coverage (seed 42, valid successes)

`covers ref` = |S_arm ∩ S_ref| / |S_ref|; `in ref` = |S_arm ∩ S_ref| / |S_arm|; `CAA cov.` = |S_arm| / |union of all arms| (CAA Fig. 4); `outside ref` = flows the arm breaks that the reference does not. `final_prim_pgd` = FINAL Exp A Prim-PGD.

| dataset | victim | budget | arm | successes | covers ref | in ref | CAA cov. | outside ref | union |
|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 431 | 1.000 | 1.000 | 1.000 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | no_padding_sweep | 431 | 1.000 | 1.000 | 1.000 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | no_random_restarts | 144 | 0.334 | 1.000 | 0.334 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | fixed_step | 431 | 1.000 | 1.000 | 1.000 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | no_momentum | 431 | 1.000 | 1.000 | 1.000 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | no_surrogate_floor | 431 | 1.000 | 1.000 | 1.000 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | validator_post_hoc | 431 | 1.000 | 1.000 | 1.000 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | last_iterate | 398 | 0.923 | 1.000 | 0.923 | 0 | 431 |
| cicids2017_distrinet | cnn | p75 | final_prim_pgd | 431 | 1.000 | 1.000 | 1.000 | 0 | 431 |
| cicids2017_distrinet | cnn | unb | reference | 1918 | 1.000 | 1.000 | 1.000 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | no_padding_sweep | 1918 | 1.000 | 1.000 | 1.000 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | no_random_restarts | 1097 | 0.572 | 1.000 | 0.572 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | fixed_step | 1918 | 1.000 | 1.000 | 1.000 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | no_momentum | 1918 | 1.000 | 1.000 | 1.000 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | no_surrogate_floor | 1918 | 1.000 | 1.000 | 1.000 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | validator_post_hoc | 1918 | 1.000 | 1.000 | 1.000 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | last_iterate | 1889 | 0.985 | 1.000 | 0.985 | 0 | 1918 |
| cicids2017_distrinet | cnn | unb | final_prim_pgd | 1918 | 1.000 | 1.000 | 1.000 | 0 | 1918 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | no_padding_sweep | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | no_random_restarts | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | fixed_step | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | no_momentum | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | no_surrogate_floor | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | validator_post_hoc | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | last_iterate | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | p75 | final_prim_pgd | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2017_distrinet | ft_transformer | unb | reference | 18 | 1.000 | 1.000 | 0.947 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | no_padding_sweep | 18 | 1.000 | 1.000 | 0.947 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | no_random_restarts | 16 | 0.889 | 1.000 | 0.842 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | fixed_step | 16 | 0.889 | 1.000 | 0.842 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | no_momentum | 19 | 1.000 | 0.947 | 1.000 | 1 | 19 |
| cicids2017_distrinet | ft_transformer | unb | no_surrogate_floor | 16 | 0.889 | 1.000 | 0.842 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | validator_post_hoc | 18 | 1.000 | 1.000 | 0.947 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | last_iterate | 12 | 0.667 | 1.000 | 0.632 | 0 | 19 |
| cicids2017_distrinet | ft_transformer | unb | final_prim_pgd | 19 | 1.000 | 0.947 | 1.000 | 1 | 19 |
| cicids2017_distrinet | mlp | p75 | reference | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | no_padding_sweep | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | no_random_restarts | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | fixed_step | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | no_momentum | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | no_surrogate_floor | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | validator_post_hoc | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | last_iterate | 120 | 0.916 | 1.000 | 0.916 | 0 | 131 |
| cicids2017_distrinet | mlp | p75 | final_prim_pgd | 131 | 1.000 | 1.000 | 1.000 | 0 | 131 |
| cicids2017_distrinet | mlp | unb | reference | 735 | 1.000 | 1.000 | 1.000 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | no_padding_sweep | 735 | 1.000 | 1.000 | 1.000 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | no_random_restarts | 734 | 0.999 | 1.000 | 0.999 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | fixed_step | 735 | 1.000 | 1.000 | 1.000 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | no_momentum | 735 | 1.000 | 1.000 | 1.000 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | no_surrogate_floor | 735 | 1.000 | 1.000 | 1.000 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | validator_post_hoc | 735 | 1.000 | 1.000 | 1.000 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | last_iterate | 716 | 0.974 | 1.000 | 0.974 | 0 | 735 |
| cicids2017_distrinet | mlp | unb | final_prim_pgd | 735 | 1.000 | 1.000 | 1.000 | 0 | 735 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 37 | 1.000 | 1.000 | 1.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | no_padding_sweep | 37 | 1.000 | 1.000 | 1.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | no_random_restarts | 34 | 0.919 | 1.000 | 0.919 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | fixed_step | 37 | 1.000 | 1.000 | 1.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | no_momentum | 37 | 1.000 | 1.000 | 1.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | no_surrogate_floor | 37 | 1.000 | 1.000 | 1.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | validator_post_hoc | 37 | 1.000 | 1.000 | 1.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | last_iterate | 31 | 0.838 | 1.000 | 0.838 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | p75 | final_prim_pgd | 37 | 1.000 | 1.000 | 1.000 | 0 | 37 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 837 | 1.000 | 1.000 | 0.995 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | no_padding_sweep | 837 | 1.000 | 1.000 | 0.995 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | no_random_restarts | 828 | 0.989 | 1.000 | 0.985 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | fixed_step | 837 | 1.000 | 1.000 | 0.995 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | no_momentum | 839 | 1.000 | 0.998 | 0.998 | 2 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | no_surrogate_floor | 837 | 1.000 | 1.000 | 0.995 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | validator_post_hoc | 837 | 1.000 | 1.000 | 0.995 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | last_iterate | 836 | 0.999 | 1.000 | 0.994 | 0 | 841 |
| cicids2018_distrinet | cnn-s42 | unb | final_prim_pgd | 841 | 1.000 | 0.995 | 1.000 | 4 | 841 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_padding_sweep | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_refinement | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_random_restarts | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | fixed_step | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_momentum | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_surrogate_floor | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | validator_post_hoc | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | last_iterate | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | final_prim_pgd | 0 | n/a | n/a | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_padding_sweep | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_random_restarts | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | fixed_step | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_momentum | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_surrogate_floor | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | validator_post_hoc | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | last_iterate | 3 | 0.750 | 1.000 | 0.750 | 0 | 4 |
| cicids2018_distrinet | ft_transformer-s42 | unb | final_prim_pgd | 4 | 1.000 | 1.000 | 1.000 | 0 | 4 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 81 | 1.000 | 1.000 | 1.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | no_padding_sweep | 81 | 1.000 | 1.000 | 1.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | no_random_restarts | 69 | 0.852 | 1.000 | 0.852 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | fixed_step | 81 | 1.000 | 1.000 | 1.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | no_momentum | 81 | 1.000 | 1.000 | 1.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | no_surrogate_floor | 81 | 1.000 | 1.000 | 1.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | validator_post_hoc | 81 | 1.000 | 1.000 | 1.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | last_iterate | 72 | 0.889 | 1.000 | 0.889 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | p75 | final_prim_pgd | 81 | 1.000 | 1.000 | 1.000 | 0 | 81 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 1419 | 1.000 | 1.000 | 1.000 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | no_padding_sweep | 1419 | 1.000 | 1.000 | 1.000 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | no_refinement | 0 | 0.000 | n/a | 0.000 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | no_random_restarts | 1406 | 0.991 | 1.000 | 0.991 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | fixed_step | 1419 | 1.000 | 1.000 | 1.000 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | no_momentum | 1418 | 0.999 | 1.000 | 0.999 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | no_surrogate_floor | 1410 | 0.994 | 1.000 | 0.994 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | validator_post_hoc | 1419 | 1.000 | 1.000 | 1.000 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | last_iterate | 1361 | 0.959 | 1.000 | 0.959 | 0 | 1419 |
| cicids2018_distrinet | mlp-s42 | unb | final_prim_pgd | 1418 | 0.999 | 1.000 | 0.999 | 0 | 1419 |

## Cost profile of the successes (`valid_success`, all seeds pooled)

| dataset | victim | budget | condition | successes | median norm. cost | median rel. duration change | median delay (µs) | median shape | frac. using padding | median added bytes |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | p75 | reference | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.939 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | no_padding_sweep | 1293 | 0.6 | 0.412 | 9.87e+05 | 0.942 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2017_distrinet | cnn | p75 | no_random_restarts | 432 | 0.7 | 0.619 | 8.81e+05 | 0.7 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | fixed_step | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.939 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | no_momentum | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.938 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | no_surrogate_floor | 1293 | 0.62 | 0.437 | 1.01e+06 | 0.945 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | validator_post_hoc | 1293 | 0.6 | 0.412 | 9.83e+05 | 0.939 | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | last_iterate | 1184 | 1 | 0.687 | 1.81e+06 | 1 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | reference | 5754 | 0.379 | 6.01 | 7.76e+06 | 0.691 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | no_padding_sweep | 5754 | 0.38 | 6.03 | 7.83e+06 | 0.696 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2017_distrinet | cnn | unb | no_random_restarts | 3291 | 0.5 | 7.78 | 7.86e+06 | 0.5 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | fixed_step | 5754 | 0.379 | 6.01 | 7.76e+06 | 0.691 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | no_momentum | 5754 | 0.377 | 6.03 | 7.72e+06 | 0.69 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | no_surrogate_floor | 5754 | 0.436 | 6.29 | 1.14e+07 | 0.769 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | validator_post_hoc | 5754 | 0.379 | 6.01 | 7.76e+06 | 0.691 | 0 | 0 |
| cicids2017_distrinet | cnn | unb | last_iterate | 5667 | 1 | 12.2 | 2.38e+07 | 1 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | reference | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | no_padding_sweep | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.36 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2017_distrinet | ft_transformer | p75 | no_random_restarts | 12 | 0.3 | 0.0701 | 6.72e+03 | 0.3 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | fixed_step | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | no_momentum | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | no_surrogate_floor | 12 | 0.324 | 0.0702 | 5.45e+03 | 0.599 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | validator_post_hoc | 12 | 0.298 | 0.0697 | 5.45e+03 | 0.399 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | last_iterate | 12 | 1 | 0.218 | 1.77e+04 | 1 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | reference | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | no_padding_sweep | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2017_distrinet | ft_transformer | unb | no_random_restarts | 48 | 0.05 | 1.31e+03 | 3.16e+06 | 0.85 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | fixed_step | 49 | 0.1 | 957 | 4.14e+06 | 0.777 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | no_momentum | 57 | 0.05 | 1.1e+03 | 2.77e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | no_surrogate_floor | 46 | 0.387 | 1.21e+03 | 1.39e+07 | 0.733 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | validator_post_hoc | 53 | 0.05 | 767 | 2.7e+06 | 0.984 | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | last_iterate | 31 | 1 | 4.26e+03 | 5.35e+07 | 0.737 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | reference | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | no_padding_sweep | 393 | 0.629 | 0.485 | 1.25e+06 | 0.947 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2017_distrinet | mlp | p75 | no_random_restarts | 393 | 0.8 | 0.55 | 1.48e+06 | 0.8 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | fixed_step | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | no_momentum | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | no_surrogate_floor | 393 | 0.671 | 0.51 | 1.27e+06 | 0.967 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | validator_post_hoc | 393 | 0.632 | 0.482 | 1.24e+06 | 0.937 | 0 | 0 |
| cicids2017_distrinet | mlp | p75 | last_iterate | 373 | 1 | 0.687 | 1.99e+06 | 1 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | reference | 2205 | 0.3 | 25.7 | 5.2e+06 | 0.664 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | no_padding_sweep | 2205 | 0.3 | 25.7 | 5.28e+06 | 0.627 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2017_distrinet | mlp | unb | no_random_restarts | 2202 | 0.4 | 29.3 | 6.65e+06 | 0.4 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | fixed_step | 2205 | 0.3 | 25.7 | 5.21e+06 | 0.664 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | no_momentum | 2205 | 0.3 | 25.7 | 5.21e+06 | 0.648 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | no_surrogate_floor | 2205 | 0.376 | 33.7 | 6.76e+06 | 0.813 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | validator_post_hoc | 2205 | 0.3 | 25.7 | 5.2e+06 | 0.664 | 0 | 0 |
| cicids2017_distrinet | mlp | unb | last_iterate | 2142 | 1 | 101 | 1.78e+07 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | reference | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | no_padding_sweep | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | cnn-s42 | p75 | no_random_restarts | 102 | 1 | 1.43 | 1.5e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | fixed_step | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | no_momentum | 111 | 0.829 | 1.18 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | no_surrogate_floor | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | validator_post_hoc | 111 | 0.836 | 1.19 | 1.27e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | last_iterate | 83 | 1 | 1.43 | 1.48e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | reference | 2514 | 0.166 | 51.2 | 1.86e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | no_padding_sweep | 2514 | 0.166 | 51.2 | 1.89e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | cnn-s42 | unb | no_random_restarts | 2484 | 0.2 | 59.7 | 2.34e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | fixed_step | 2513 | 0.166 | 51.2 | 1.87e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | no_momentum | 2518 | 0.166 | 51 | 1.86e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | no_surrogate_floor | 2513 | 0.305 | 95.6 | 3.41e+07 | 0.596 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | validator_post_hoc | 2514 | 0.166 | 51.2 | 1.86e+07 | 0.1 | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | last_iterate | 2508 | 1 | 307 | 1.17e+08 | 1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | reference | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_padding_sweep | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_random_restarts | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | fixed_step | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_momentum | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | no_surrogate_floor | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | validator_post_hoc | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | p75 | last_iterate | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | unb | reference | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_padding_sweep | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_random_restarts | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | fixed_step | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_momentum | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | no_surrogate_floor | 12 | 0.357 | 2.45e+03 | 1.71e+07 | 0.545 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | validator_post_hoc | 12 | 0.1 | 1.28e+03 | 4.8e+06 | 0.1 | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | last_iterate | 9 | 0.959 | 1.48e+04 | 4.6e+07 | 0.647 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | reference | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | no_padding_sweep | 243 | 0.6 | 0.784 | 4.82e+05 | 0.945 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | mlp-s42 | p75 | no_random_restarts | 207 | 0.7 | 0.991 | 5.31e+05 | 0.7 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | fixed_step | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | no_momentum | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | no_surrogate_floor | 243 | 0.652 | 0.837 | 5.15e+05 | 0.98 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | validator_post_hoc | 243 | 0.6 | 0.784 | 4.82e+05 | 0.95 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | last_iterate | 217 | 1 | 1.43 | 9.91e+05 | 1 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | reference | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | no_padding_sweep | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | no_refinement | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | mlp-s42 | unb | no_random_restarts | 4218 | 0.4 | 176 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | fixed_step | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | no_momentum | 4256 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | no_surrogate_floor | 4231 | 0.537 | 207 | 4.82e+07 | 0.378 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | validator_post_hoc | 4259 | 0.4 | 171 | 4.67e+07 | 0 | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | last_iterate | 4098 | 1 | 319 | 4.8e+07 | 0.5 | 0 | 0 |

