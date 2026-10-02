# P1 - Realization-aware search vs continuous-state search

Targeted -> Benign, Hybrid Search, frozen clean-correct flows (4 classes x 800 per attack seed). Mean ± SD over attack seeds 42/2024/2026 (%). `P1 cont` = objective hit of the continuous candidate before realization (never counted as success). Paired McNemar on valid success at seed 42 (exact binomial below 25 discordant pairs), Holm over the 12 cells.

## cicids2017_distrinet

| victim | budget | ref Raw | ref Valid | P1 cont | P1 Raw | P1 Valid | Δ Valid (pp) | seed-42 P1-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| mlp | p75 | 4.09 ± 0.00 | 4.09 ± 0.00 | 4.09 ± 0.00 | 4.09 ± 0.00 | 4.09 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| mlp | unb | 22.94 ± 0.00 | 22.94 ± 0.00 | 22.94 ± 0.00 | 22.94 ± 0.00 | 22.94 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| cnn | p75 | 13.25 ± 0.00 | 13.25 ± 0.00 | 13.25 ± 0.00 | 13.25 ± 0.00 | 13.25 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| cnn | unb | 59.69 ± 0.00 | 59.69 ± 0.00 | 59.69 ± 0.00 | 59.69 ± 0.00 | 59.69 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| ft_transformer | p75 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| ft_transformer | unb | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.55 ± 0.02 | +0.00 ± 0.00 | 0 / 0 | 1 |

## cicids2018_distrinet

| victim | budget | ref Raw | ref Valid | P1 cont | P1 Raw | P1 Valid | Δ Valid (pp) | seed-42 P1-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|
| mlp-s42 | p75 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.78 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| mlp-s42 | unb | 24.80 ± 0.02 | 24.80 ± 0.02 | 24.80 ± 0.02 | 24.80 ± 0.02 | 24.80 ± 0.02 | +0.00 ± 0.00 | 0 / 0 | 1 |
| cnn-s42 | p75 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| cnn-s42 | unb | 26.09 ± 0.00 | 26.09 ± 0.00 | 26.09 ± 0.00 | 26.09 ± 0.00 | 26.09 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| ft_transformer-s42 | p75 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |
| ft_transformer-s42 | unb | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | +0.00 ± 0.00 | 0 / 0 | 1 |

## Continuous -> realized (P1, all three seeds summed)

| dataset | victim | budget | flows | cont hits | -> realized valid | -> realized hit, invalid | -> realized miss | share lost | cont miss -> realized hit | prediction changed |
|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 9600 | 393 | 393 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2017_distrinet | mlp | unb | 9600 | 2202 | 2202 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2017_distrinet | cnn | p75 | 9600 | 1272 | 1272 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2017_distrinet | cnn | unb | 9600 | 5730 | 5730 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2017_distrinet | ft_transformer | p75 | 9600 | 12 | 12 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2017_distrinet | ft_transformer | unb | 9600 | 53 | 53 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | p75 | 9600 | 75 | 75 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2018_distrinet | mlp-s42 | unb | 9600 | 2381 | 2381 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | p75 | 9600 | 0 | 0 | 0 | 0 | n/a | 0 | 0 |
| cicids2018_distrinet | cnn-s42 | unb | 9600 | 2505 | 2505 | 0 | 0 | 0.00% | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 9600 | 0 | 0 | 0 | 0 | n/a | 0 | 0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 9600 | 12 | 12 | 0 | 0 | 0.00% | 0 | 0 |

## What the single final rounding changes (P1, all three seeds)

Continuous flows are judged by the same validator_v2 for diagnosis only. The margin is the targeted margin max(non-Benign) - Benign logit (negative = hit).

| dataset | victim | budget | max abs Δdelay (µs) | max abs Δp (bytes) | max abs Δmargin | min abs margin of cont hits | cont hits passing validator_v2 (SCHEMA) unrealized | final flow identical to ref (all / ref valid successes) |
|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 0.999 | 0.000 | 4.69e-03 | 1.13e-03 | 3/393 (3) | 9596/9600 / 393/393 |
| cicids2017_distrinet | mlp | unb | 0.500 | 0.000 | 6.87e-05 | 2.93e-04 | 78/2202 (78) | 9445/9600 / 2197/2202 |
| cicids2017_distrinet | cnn | p75 | 0.999 | 0.000 | 5.63e-03 | 2.69e-04 | 20/1272 (20) | 9599/9600 / 1272/1272 |
| cicids2017_distrinet | cnn | unb | 0.500 | 0.000 | 5.53e-03 | 9.01e-04 | 606/5730 (606) | 9600/9600 / 5730/5730 |
| cicids2017_distrinet | ft_transformer | p75 | 0.999 | 0.000 | 4.63e-03 | 1.49e-02 | 0/12 (0) | 9498/9600 / 12/12 |
| cicids2017_distrinet | ft_transformer | unb | 0.500 | 0.000 | 1.05e-03 | 7.90e-03 | 12/53 (12) | 9518/9600 / 53/53 |
| cicids2018_distrinet | mlp-s42 | p75 | 0.998 | 0.000 | 1.55e-03 | 4.88e-03 | 2/75 (2) | 9428/9600 / 75/75 |
| cicids2018_distrinet | mlp-s42 | unb | 0.500 | 0.000 | 1.47e-03 | 3.42e-04 | 86/2381 (86) | 9431/9600 / 2381/2381 |
| cicids2018_distrinet | cnn-s42 | p75 | 0.998 | 0.000 | 5.17e-03 | nan | 0/0 (0) | 9528/9600 / 0/0 |
| cicids2018_distrinet | cnn-s42 | unb | 0.500 | 0.000 | 6.48e-03 | 5.05e-05 | 27/2505 (27) | 9598/9600 / 2505/2505 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 0.998 | 0.000 | 4.78e-04 | nan | 0/0 (0) | 9264/9600 / 0/0 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 0.500 | 0.000 | 2.62e-04 | 5.39e-02 | 0/12 (0) | 9571/9600 / 12/12 |

## Per seed

| dataset | victim | budget | seed | ref Raw | ref Valid | P1 cont | P1 Raw | P1 Valid | P1-only / ref-only valid | cont hit -> miss | pred changed | median cost valid ref / P1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 42 | 4.09% | 4.09% | 4.09% | 4.09% | 4.09% | 0 / 0 | 0 | 0 | 0.629 / 0.629 |
| cicids2017_distrinet | mlp | p75 | 2024 | 4.09% | 4.09% | 4.09% | 4.09% | 4.09% | 0 / 0 | 0 | 0 | 0.603 / 0.603 |
| cicids2017_distrinet | mlp | p75 | 2026 | 4.09% | 4.09% | 4.09% | 4.09% | 4.09% | 0 / 0 | 0 | 0 | 0.672 / 0.672 |
| cicids2017_distrinet | mlp | unb | 42 | 22.94% | 22.94% | 22.94% | 22.94% | 22.94% | 0 / 0 | 0 | 0 | 0.300 / 0.300 |
| cicids2017_distrinet | mlp | unb | 2024 | 22.94% | 22.94% | 22.94% | 22.94% | 22.94% | 0 / 0 | 0 | 0 | 0.300 / 0.300 |
| cicids2017_distrinet | mlp | unb | 2026 | 22.94% | 22.94% | 22.94% | 22.94% | 22.94% | 0 / 0 | 0 | 0 | 0.300 / 0.300 |
| cicids2017_distrinet | cnn | p75 | 42 | 13.25% | 13.25% | 13.25% | 13.25% | 13.25% | 0 / 0 | 0 | 0 | 0.580 / 0.580 |
| cicids2017_distrinet | cnn | p75 | 2024 | 13.25% | 13.25% | 13.25% | 13.25% | 13.25% | 0 / 0 | 0 | 0 | 0.600 / 0.600 |
| cicids2017_distrinet | cnn | p75 | 2026 | 13.25% | 13.25% | 13.25% | 13.25% | 13.25% | 0 / 0 | 0 | 0 | 0.608 / 0.608 |
| cicids2017_distrinet | cnn | unb | 42 | 59.69% | 59.69% | 59.69% | 59.69% | 59.69% | 0 / 0 | 0 | 0 | 0.380 / 0.380 |
| cicids2017_distrinet | cnn | unb | 2024 | 59.69% | 59.69% | 59.69% | 59.69% | 59.69% | 0 / 0 | 0 | 0 | 0.383 / 0.383 |
| cicids2017_distrinet | cnn | unb | 2026 | 59.69% | 59.69% | 59.69% | 59.69% | 59.69% | 0 / 0 | 0 | 0 | 0.393 / 0.393 |
| cicids2017_distrinet | ft_transformer | p75 | 42 | 0.12% | 0.12% | 0.12% | 0.12% | 0.12% | 0 / 0 | 0 | 0 | 0.273 / 0.273 |
| cicids2017_distrinet | ft_transformer | p75 | 2024 | 0.12% | 0.12% | 0.12% | 0.12% | 0.12% | 0 / 0 | 0 | 0 | 0.298 / 0.298 |
| cicids2017_distrinet | ft_transformer | p75 | 2026 | 0.12% | 0.12% | 0.12% | 0.12% | 0.12% | 0 / 0 | 0 | 0 | 0.285 / 0.285 |
| cicids2017_distrinet | ft_transformer | unb | 42 | 0.56% | 0.56% | 0.56% | 0.56% | 0.56% | 0 / 0 | 0 | 0 | 0.047 / 0.047 |
| cicids2017_distrinet | ft_transformer | unb | 2024 | 0.53% | 0.53% | 0.53% | 0.53% | 0.53% | 0 / 0 | 0 | 0 | 0.050 / 0.050 |
| cicids2017_distrinet | ft_transformer | unb | 2026 | 0.56% | 0.56% | 0.56% | 0.56% | 0.56% | 0 / 0 | 0 | 0 | 0.050 / 0.050 |
| cicids2018_distrinet | mlp-s42 | p75 | 42 | 0.78% | 0.78% | 0.78% | 0.78% | 0.78% | 0 / 0 | 0 | 0 | 0.476 / 0.476 |
| cicids2018_distrinet | mlp-s42 | p75 | 2024 | 0.78% | 0.78% | 0.78% | 0.78% | 0.78% | 0 / 0 | 0 | 0 | 0.500 / 0.500 |
| cicids2018_distrinet | mlp-s42 | p75 | 2026 | 0.78% | 0.78% | 0.78% | 0.78% | 0.78% | 0 / 0 | 0 | 0 | 0.450 / 0.450 |
| cicids2018_distrinet | mlp-s42 | unb | 42 | 24.78% | 24.78% | 24.78% | 24.78% | 24.78% | 0 / 0 | 0 | 0 | 0.600 / 0.600 |
| cicids2018_distrinet | mlp-s42 | unb | 2024 | 24.81% | 24.81% | 24.81% | 24.81% | 24.81% | 0 / 0 | 0 | 0 | 0.600 / 0.600 |
| cicids2018_distrinet | mlp-s42 | unb | 2026 | 24.81% | 24.81% | 24.81% | 24.81% | 24.81% | 0 / 0 | 0 | 0 | 0.600 / 0.600 |
| cicids2018_distrinet | cnn-s42 | p75 | 42 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 | 0 | 0 | nan / nan |
| cicids2018_distrinet | cnn-s42 | p75 | 2024 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 | 0 | 0 | nan / nan |
| cicids2018_distrinet | cnn-s42 | p75 | 2026 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 | 0 | 0 | nan / nan |
| cicids2018_distrinet | cnn-s42 | unb | 42 | 26.09% | 26.09% | 26.09% | 26.09% | 26.09% | 0 / 0 | 0 | 0 | 0.158 / 0.158 |
| cicids2018_distrinet | cnn-s42 | unb | 2024 | 26.09% | 26.09% | 26.09% | 26.09% | 26.09% | 0 / 0 | 0 | 0 | 0.168 / 0.168 |
| cicids2018_distrinet | cnn-s42 | unb | 2026 | 26.09% | 26.09% | 26.09% | 26.09% | 26.09% | 0 / 0 | 0 | 0 | 0.163 / 0.163 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 42 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 | 0 | 0 | nan / nan |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 2024 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 | 0 | 0 | nan / nan |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 2026 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0 / 0 | 0 | 0 | nan / nan |
| cicids2018_distrinet | ft_transformer-s42 | unb | 42 | 0.12% | 0.12% | 0.12% | 0.12% | 0.12% | 0 / 0 | 0 | 0 | 0.100 / 0.100 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 2024 | 0.12% | 0.12% | 0.12% | 0.12% | 0.12% | 0 / 0 | 0 | 0 | 0.100 / 0.100 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 2026 | 0.12% | 0.12% | 0.12% | 0.12% | 0.12% | 0 / 0 | 0 | 0 | 0.100 / 0.100 |

## Victim evaluations per flow (mean over seeds; max over all flows)

| dataset | victim | budget | ref mean | ref max | P1 mean | P1 max | P1 cont candidates | P1 gradient | P1 final realization |
|---|---|---|---|---|---|---|---|---|---|
| cicids2017_distrinet | mlp | p75 | 188.5 | 255 | 189.5 | 256 | 94.8 | 93.7 | 1.00 |
| cicids2017_distrinet | mlp | unb | 188.5 | 255 | 189.5 | 256 | 94.8 | 93.7 | 1.00 |
| cicids2017_distrinet | cnn | p75 | 188.5 | 255 | 189.5 | 256 | 94.8 | 93.7 | 1.00 |
| cicids2017_distrinet | cnn | unb | 188.5 | 255 | 189.5 | 256 | 94.8 | 93.7 | 1.00 |
| cicids2017_distrinet | ft_transformer | p75 | 188.6 | 255 | 189.6 | 256 | 94.8 | 93.8 | 1.00 |
| cicids2017_distrinet | ft_transformer | unb | 188.6 | 255 | 189.6 | 256 | 94.8 | 93.8 | 1.00 |
| cicids2018_distrinet | mlp-s42 | p75 | 190.8 | 255 | 191.8 | 256 | 95.9 | 94.9 | 1.00 |
| cicids2018_distrinet | mlp-s42 | unb | 190.8 | 255 | 191.8 | 256 | 95.9 | 94.9 | 1.00 |
| cicids2018_distrinet | cnn-s42 | p75 | 190.7 | 255 | 191.7 | 256 | 95.9 | 94.9 | 1.00 |
| cicids2018_distrinet | cnn-s42 | unb | 190.7 | 255 | 191.7 | 256 | 95.9 | 94.9 | 1.00 |
| cicids2018_distrinet | ft_transformer-s42 | p75 | 190.8 | 255 | 191.8 | 256 | 95.9 | 94.9 | 1.00 |
| cicids2018_distrinet | ft_transformer-s42 | unb | 190.8 | 255 | 191.8 | 256 | 95.9 | 94.9 | 1.00 |

## Sanity checks

* cells x seeds checked: 36; flows per arm: 115200
* identical source ids in both arms: 36/36; frozen selection SHA-256 ok: 36/36
* identical capability mask M(x) / per-flow box in both arms: 36/36 / 36/36; M(x) recomputed from the source flows equals the stored mask: ref 115200/115200, P1 115200/115200
* max victim evaluations per flow: ref 255, P1 256 (cap 256); P1 final realization = 1 evaluation per flow in 36/36
* final validator_v2 recomputed on the stored realized flows equals the stored verdict: ref 115200/115200, P1 115200/115200
* realized controls integer (bytes, µs): ref 115200/115200, P1 115200/115200; stored controls = canonical projection of the requested controls: ref 115200/115200, P1 115200/115200; stored flow = quantized φ of the stored controls (bit-identical): ref 115200/115200, P1 115200/115200
* P1 raw success = realized victim prediction == Benign and search success = raw success: 36/36; valid successes of P1 whose realized flow misses (continuous-only): 0
* reference vs FINAL targeted Hybrid (`primattack_hybrid_objective_targeted` p75, `primattack_targeted_budgets` unb): flows compared 115200, adv flow identical 115200, prediction identical 115200, validity identical 115200, valid success identical 115200; FINAL artifacts missing 0

