# D5 - Attacker overhead of PrimAttack's valid successes

FINAL-suite artifacts; overheads over valid successes (all attack seeds pooled), Valid ASR = mean over seeds 42/2024/2026 of successes / 3,200 attempted flows.

## Overhead per success (Amoeba DO/TO, PLAA relative changes)

| dataset | victim | arm | budget | Valid ASR (min-max) | successes | with padding | median DO | median TO | p90 TO | median added µs | Δ pkt length | Δ Flow IAT Mean | Δ Flow Bytes/s | Δ Flow Pkts/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | targeted_hybrid | p50 | 9.19% (9.19%-9.19%) | 882 | 0 | 0 | 0.196 | 0.303 | 1.03e+06 | 0 | 0.244 | -0.196 | -0.196 |
| cicids2017_distrinet | cnn | targeted_hybrid | p75 | 13.25% (13.25%-13.25%) | 1272 | 0 | 0 | 0.289 | 0.524 | 9.92e+05 | 0 | 0.405 | -0.289 | -0.289 |
| cicids2017_distrinet | cnn | targeted_hybrid | unb | 59.69% (59.69%-59.69%) | 5730 | 0 | 0 | 0.859 | 0.986 | 8.14e+06 | 0 | 6.09 | -0.858 | -0.859 |
| cicids2017_distrinet | cnn | targeted_pgd | p50 | 9.19% (9.19%-9.19%) | 882 | 0 | 0 | 0.204 | 0.303 | 1.07e+06 | 0 | 0.257 | -0.204 | -0.204 |
| cicids2017_distrinet | cnn | targeted_pgd | p75 | 13.25% (13.25%-13.25%) | 1272 | 0 | 0 | 0.307 | 0.525 | 1.03e+06 | 0 | 0.444 | -0.307 | -0.307 |
| cicids2017_distrinet | cnn | targeted_pgd | unb | 59.69% (59.69%-59.69%) | 5730 | 0 | 0 | 0.867 | 0.986 | 8.41e+06 | 0 | 6.52 | -0.867 | -0.867 |
| cicids2017_distrinet | cnn | untargeted_pgd | p75 | 13.47% (13.47%-13.47%) | 1293 | 0 | 0 | 0.307 | 0.524 | 1e+06 | 0 | 0.444 | -0.307 | -0.307 |
| cicids2017_distrinet | cnn | untargeted_pgd | unb | 59.94% (59.94%-59.94%) | 5754 | 0 | 0 | 0.864 | 0.986 | 8.04e+06 | 0 | 6.38 | -0.864 | -0.864 |
| cicids2017_distrinet | ft_transformer | targeted_hybrid | p50 | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.0386 | 0.0569 | 3.09e+03 | 0 | 0.0402 | -0.0386 | -0.0386 |
| cicids2017_distrinet | ft_transformer | targeted_hybrid | p75 | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.0652 | 0.0662 | 5.45e+03 | 0 | 0.0697 | -0.0652 | -0.0652 |
| cicids2017_distrinet | ft_transformer | targeted_hybrid | unb | 0.55% (0.53%-0.56%) | 53 | 0 | 0 | 0.999 | 1 | 2.7e+06 | 0 | 767 | -0.993 | -0.999 |
| cicids2017_distrinet | ft_transformer | targeted_pgd | p50 | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.0417 | 0.0564 | 3.66e+03 | 0 | 0.0436 | -0.0417 | -0.0417 |
| cicids2017_distrinet | ft_transformer | targeted_pgd | p75 | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.0552 | 0.0662 | 5.7e+03 | 0 | 0.0584 | -0.0552 | -0.0552 |
| cicids2017_distrinet | ft_transformer | targeted_pgd | unb | 0.59% (0.59%-0.59%) | 57 | 0 | 0 | 0.999 | 1 | 3.06e+06 | 0 | 1.66e+03 | -0.994 | -0.999 |
| cicids2017_distrinet | ft_transformer | untargeted_pgd | p75 | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.0552 | 0.0662 | 5.7e+03 | 0 | 0.0584 | -0.0552 | -0.0552 |
| cicids2017_distrinet | ft_transformer | untargeted_pgd | unb | 0.59% (0.59%-0.59%) | 57 | 0 | 0 | 0.999 | 1 | 3.06e+06 | 0 | 1.66e+03 | -0.994 | -0.999 |
| cicids2017_distrinet | mlp | targeted_hybrid | p50 | 2.31% (2.31%-2.31%) | 222 | 0 | 0 | 0.233 | 0.293 | 1.07e+06 | 0 | 0.304 | -0.233 | -0.233 |
| cicids2017_distrinet | mlp | targeted_hybrid | p75 | 4.09% (4.09%-4.09%) | 393 | 0 | 0 | 0.325 | 0.518 | 1.24e+06 | 0 | 0.482 | -0.325 | -0.325 |
| cicids2017_distrinet | mlp | targeted_hybrid | unb | 22.94% (22.94%-22.94%) | 2202 | 0 | 0 | 0.963 | 0.996 | 5.19e+06 | 0 | 25.7 | -0.962 | -0.963 |
| cicids2017_distrinet | mlp | targeted_pgd | p50 | 2.31% (2.31%-2.31%) | 222 | 0 | 0 | 0.234 | 0.289 | 1.07e+06 | 0 | 0.305 | -0.234 | -0.234 |
| cicids2017_distrinet | mlp | targeted_pgd | p75 | 4.09% (4.09%-4.09%) | 393 | 0 | 0 | 0.326 | 0.519 | 1.26e+06 | 0 | 0.483 | -0.326 | -0.326 |
| cicids2017_distrinet | mlp | targeted_pgd | unb | 22.94% (22.94%-22.94%) | 2202 | 0 | 0 | 0.959 | 0.996 | 5.12e+06 | 0 | 23.4 | -0.959 | -0.959 |
| cicids2017_distrinet | mlp | untargeted_pgd | p75 | 4.09% (4.09%-4.09%) | 393 | 0 | 0 | 0.326 | 0.519 | 1.26e+06 | 0 | 0.483 | -0.326 | -0.326 |
| cicids2017_distrinet | mlp | untargeted_pgd | unb | 22.97% (22.97%-22.97%) | 2205 | 0 | 0 | 0.959 | 0.996 | 5.12e+06 | 0 | 23.4 | -0.959 | -0.959 |
| cicids2018_distrinet | cnn-s42 | targeted_hybrid | p50 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | cnn-s42 | targeted_hybrid | p75 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | cnn-s42 | targeted_hybrid | unb | 26.09% (26.09%-26.09%) | 2505 | 0 | 0 | 0.98 | 0.985 | 1.83e+07 | 0 | 49.7 | -0.981 | -0.98 |
| cicids2018_distrinet | cnn-s42 | targeted_pgd | p50 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | cnn-s42 | targeted_pgd | p75 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | cnn-s42 | targeted_pgd | unb | 26.09% (26.09%-26.09%) | 2505 | 0 | 0 | 0.971 | 0.979 | 1.17e+07 | 0 | 34 | -0.971 | -0.971 |
| cicids2018_distrinet | cnn-s42 | untargeted_pgd | p75 | 1.16% (1.16%-1.16%) | 111 | 0 | 0 | 0.549 | 0.586 | 1.25e+05 | 0 | 1.22 | -0.549 | -0.549 |
| cicids2018_distrinet | cnn-s42 | untargeted_pgd | unb | 26.28% (26.28%-26.28%) | 2523 | 0 | 0 | 0.971 | 0.979 | 1.17e+07 | 0 | 33.9 | -0.971 | -0.971 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_hybrid | p50 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | targeted_hybrid | p75 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | targeted_hybrid | unb | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.999 | 1 | 4.8e+06 | 0 | 1.28e+03 | -0.999 | -0.999 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_pgd | p50 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | targeted_pgd | p75 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | targeted_pgd | unb | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.999 | 1 | 3.6e+06 | 0 | 853 | -0.999 | -0.999 |
| cicids2018_distrinet | ft_transformer-s42 | untargeted_pgd | p75 | 0.00% (0.00%-0.00%) | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| cicids2018_distrinet | ft_transformer-s42 | untargeted_pgd | unb | 0.12% (0.12%-0.12%) | 12 | 0 | 0 | 0.999 | 1 | 3.6e+06 | 0 | 853 | -0.999 | -0.999 |
| cicids2018_distrinet | mlp-s42 | targeted_hybrid | p50 | 0.69% (0.69%-0.69%) | 66 | 0 | 0 | 0.32 | 0.422 | 3.33e+06 | 0 | 0.472 | -0.32 | -0.32 |
| cicids2018_distrinet | mlp-s42 | targeted_hybrid | p75 | 0.78% (0.78%-0.78%) | 75 | 0 | 0 | 0.331 | 0.539 | 3.08e+06 | 0 | 0.496 | -0.331 | -0.331 |
| cicids2018_distrinet | mlp-s42 | targeted_hybrid | unb | 24.80% (24.78%-24.81%) | 2381 | 0 | 0 | 0.995 | 0.997 | 7.01e+07 | 0 | 193 | -0.995 | -0.995 |
| cicids2018_distrinet | mlp-s42 | targeted_pgd | p50 | 0.69% (0.69%-0.69%) | 66 | 0 | 0 | 0.305 | 0.423 | 2.74e+06 | 0 | 0.439 | -0.305 | -0.305 |
| cicids2018_distrinet | mlp-s42 | targeted_pgd | p75 | 0.78% (0.78%-0.78%) | 75 | 0 | 0 | 0.308 | 0.55 | 2.63e+06 | 0 | 0.446 | -0.308 | -0.308 |
| cicids2018_distrinet | mlp-s42 | targeted_pgd | unb | 24.76% (24.75%-24.78%) | 2377 | 0 | 0 | 0.995 | 0.997 | 7.01e+07 | 0 | 189 | -0.995 | -0.995 |
| cicids2018_distrinet | mlp-s42 | untargeted_pgd | p75 | 2.53% (2.53%-2.53%) | 243 | 0 | 0 | 0.449 | 0.57 | 5.08e+05 | 0 | 0.814 | -0.449 | -0.449 |
| cicids2018_distrinet | mlp-s42 | untargeted_pgd | unb | 44.32% (44.31%-44.34%) | 4255 | 0 | 0 | 0.994 | 0.997 | 4.41e+07 | 0 | 158 | -0.994 | -0.994 |

DO = added bytes / (forward + backward bytes + added bytes); TO = added flow time / (added time + original duration); Δ = (adversarial - original) / original.

## Diminishing returns over the budget ladder (targeted arms: p50 -> p75 -> unbounded)

| dataset | victim | arm | p50 | p75 | unbounded | gain p50->p75 (pp) | gain p75->unb (pp) | median TO p50 / p75 / unb |
|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | cnn | targeted_hybrid | 9.19% | 13.25% | 59.69% | +4.06 | +46.44 | 0.196 / 0.289 / 0.859 |
| cicids2017_distrinet | cnn | targeted_pgd | 9.19% | 13.25% | 59.69% | +4.06 | +46.44 | 0.204 / 0.307 / 0.867 |
| cicids2017_distrinet | ft_transformer | targeted_hybrid | 0.12% | 0.12% | 0.55% | +0.00 | +0.43 | 0.0386 / 0.0652 / 0.999 |
| cicids2017_distrinet | ft_transformer | targeted_pgd | 0.12% | 0.12% | 0.59% | +0.00 | +0.47 | 0.0417 / 0.0552 / 0.999 |
| cicids2017_distrinet | mlp | targeted_hybrid | 2.31% | 4.09% | 22.94% | +1.78 | +18.84 | 0.233 / 0.325 / 0.963 |
| cicids2017_distrinet | mlp | targeted_pgd | 2.31% | 4.09% | 22.94% | +1.78 | +18.84 | 0.234 / 0.326 / 0.959 |
| cicids2018_distrinet | cnn-s42 | targeted_hybrid | 0.00% | 0.00% | 26.09% | +0.00 | +26.09 | n/a / n/a / 0.98 |
| cicids2018_distrinet | cnn-s42 | targeted_pgd | 0.00% | 0.00% | 26.09% | +0.00 | +26.09 | n/a / n/a / 0.971 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_hybrid | 0.00% | 0.00% | 0.12% | +0.00 | +0.12 | n/a / n/a / 0.999 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_pgd | 0.00% | 0.00% | 0.12% | +0.00 | +0.12 | n/a / n/a / 0.999 |
| cicids2018_distrinet | mlp-s42 | targeted_hybrid | 0.69% | 0.78% | 24.80% | +0.09 | +24.02 | 0.32 / 0.331 / 0.995 |
| cicids2018_distrinet | mlp-s42 | targeted_pgd | 0.69% | 0.78% | 24.76% | +0.09 | +23.98 | 0.305 / 0.308 / 0.995 |

## Cost curve: Valid ASR when the attacker caps the time overhead TO

| dataset | victim | arm | budget | TO≤0.01 | TO≤0.05 | TO≤0.1 | TO≤0.25 | TO≤0.5 | TO≤0.75 | TO≤0.9 |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | untargeted_pgd | p75 | 0.00 | 0.02 | 0.09 | 0.86 | 3.58 | 4.09 | 4.09 |
| cicids2017_distrinet | mlp | untargeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.24 | 2.35 | 5.18 | 7.67 |
| cicids2017_distrinet | mlp | targeted_pgd | p50 | 0.00 | 0.06 | 0.10 | 1.46 | 2.31 | 2.31 | 2.31 |
| cicids2017_distrinet | mlp | targeted_pgd | p75 | 0.00 | 0.02 | 0.09 | 0.86 | 3.58 | 4.09 | 4.09 |
| cicids2017_distrinet | mlp | targeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.24 | 2.35 | 5.18 | 7.64 |
| cicids2017_distrinet | mlp | targeted_hybrid | p50 | 0.00 | 0.06 | 0.12 | 1.45 | 2.31 | 2.31 | 2.31 |
| cicids2017_distrinet | mlp | targeted_hybrid | p75 | 0.00 | 0.02 | 0.09 | 0.95 | 3.53 | 4.09 | 4.09 |
| cicids2017_distrinet | mlp | targeted_hybrid | unb | 0.00 | 0.00 | 0.00 | 0.20 | 2.19 | 5.17 | 7.55 |
| cicids2017_distrinet | cnn | untargeted_pgd | p75 | 0.00 | 0.22 | 0.92 | 5.05 | 11.57 | 13.47 | 13.47 |
| cicids2017_distrinet | cnn | untargeted_pgd | unb | 0.00 | 0.16 | 0.62 | 3.32 | 6.68 | 12.62 | 37.93 |
| cicids2017_distrinet | cnn | targeted_pgd | p50 | 0.00 | 0.31 | 1.16 | 6.36 | 9.19 | 9.19 | 9.19 |
| cicids2017_distrinet | cnn | targeted_pgd | p75 | 0.00 | 0.22 | 0.94 | 5.00 | 11.32 | 13.25 | 13.25 |
| cicids2017_distrinet | cnn | targeted_pgd | unb | 0.00 | 0.16 | 0.65 | 3.33 | 6.62 | 12.19 | 37.51 |
| cicids2017_distrinet | cnn | targeted_hybrid | p50 | 0.00 | 0.28 | 1.18 | 6.60 | 9.19 | 9.19 | 9.19 |
| cicids2017_distrinet | cnn | targeted_hybrid | p75 | 0.00 | 0.26 | 0.86 | 5.32 | 11.40 | 13.25 | 13.25 |
| cicids2017_distrinet | cnn | targeted_hybrid | unb | 0.00 | 0.18 | 0.61 | 3.53 | 6.76 | 11.93 | 37.52 |
| cicids2017_distrinet | ft_transformer | untargeted_pgd | p75 | 0.03 | 0.03 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | untargeted_pgd | unb | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.05 |
| cicids2017_distrinet | ft_transformer | targeted_pgd | p50 | 0.03 | 0.07 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | targeted_pgd | p75 | 0.03 | 0.03 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | targeted_pgd | unb | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.05 |
| cicids2017_distrinet | ft_transformer | targeted_hybrid | p50 | 0.03 | 0.07 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | targeted_hybrid | p75 | 0.03 | 0.03 | 0.12 | 0.12 | 0.12 | 0.12 | 0.12 |
| cicids2017_distrinet | ft_transformer | targeted_hybrid | unb | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.05 |
| cicids2018_distrinet | mlp-s42 | untargeted_pgd | p75 | 0.00 | 0.00 | 0.04 | 0.30 | 1.69 | 2.53 | 2.53 |
| cicids2018_distrinet | mlp-s42 | untargeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.12 | 0.32 | 0.73 | 2.00 |
| cicids2018_distrinet | mlp-s42 | targeted_pgd | p50 | 0.00 | 0.00 | 0.04 | 0.25 | 0.69 | 0.69 | 0.69 |
| cicids2018_distrinet | mlp-s42 | targeted_pgd | p75 | 0.00 | 0.00 | 0.04 | 0.25 | 0.69 | 0.78 | 0.78 |
| cicids2018_distrinet | mlp-s42 | targeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.12 | 0.32 | 0.62 | 0.89 |
| cicids2018_distrinet | mlp-s42 | targeted_hybrid | p50 | 0.00 | 0.00 | 0.04 | 0.23 | 0.69 | 0.69 | 0.69 |
| cicids2018_distrinet | mlp-s42 | targeted_hybrid | p75 | 0.00 | 0.00 | 0.04 | 0.23 | 0.69 | 0.78 | 0.78 |
| cicids2018_distrinet | mlp-s42 | targeted_hybrid | unb | 0.00 | 0.00 | 0.00 | 0.12 | 0.33 | 0.62 | 0.88 |
| cicids2018_distrinet | cnn-s42 | untargeted_pgd | p75 | 0.00 | 0.00 | 0.00 | 0.01 | 0.23 | 1.16 | 1.16 |
| cicids2018_distrinet | cnn-s42 | untargeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.02 | 0.05 | 0.09 | 0.33 |
| cicids2018_distrinet | cnn-s42 | targeted_pgd | p50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | cnn-s42 | targeted_pgd | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | cnn-s42 | targeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.25 |
| cicids2018_distrinet | cnn-s42 | targeted_hybrid | p50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | cnn-s42 | targeted_hybrid | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | cnn-s42 | targeted_hybrid | unb | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.25 |
| cicids2018_distrinet | ft_transformer-s42 | untargeted_pgd | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | untargeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_pgd | p50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_pgd | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_pgd | unb | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_hybrid | p50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_hybrid | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| cicids2018_distrinet | ft_transformer-s42 | targeted_hybrid | unb | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Cost-curve values are Valid ASR in % of attempted flows, pooled over seeds. No valid success adds bytes (column `with padding` above), so no data-overhead cap is needed.
