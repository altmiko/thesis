# A2 - Fixed vs learned delay allocation (`shape`)

## Design

This follows two published designs:

- Nasr et al. (USENIX Security 2021, Sec. 7.2, Table 3), who vary the mean and the spread of
  the added delay separately;
- FRONT (USENIX Security 2020, Sec. 5.4), which moves the same budget in time.

Every arm uses the same total-delay budget. Only the way the delay is spread over the forward
gaps changes:

- `shape_fixed_0`: proportional dilation, g' = a·g;
- `shape_fixed_0p5`: an equal mix of proportional and uniform;
- `shape_fixed_1`: uniform per-gap delay, g' = g + b;
- reference: `shape` is optimized jointly with `p` and `delay`.

A pinned shape applies to every candidate and is never updated by gradient.

Run: `python ablations/thesis_ablations/A2_shape_allocation/run.py --device cuda`. Full tables:
`results/report.md`.

## Results

Valid ASR in % (mean of seeds). `*` marks a significant difference from the reference
(seed-42 McNemar, Holm over 36 comparisons).

| dataset | victim | budget | reference (learned) | fixed 0 | fixed 0.5 | fixed 1 |
|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09 | 0.00* | 2.12* | 4.09 |
| 2017 | mlp | unb | 22.97 | 0.91* | 19.25* | 22.75 |
| 2017 | cnn | p75 | 13.47 | 0.06* | 7.44* | 13.41 |
| 2017 | cnn | unb | 59.94 | 1.28* | 53.66* | 59.94 |
| 2017 | ft_transformer | p75 | 0.12 | 0.03 | 0.12 | 0.12 |
| 2017 | ft_transformer | unb | 0.55 | 0.25* | 0.53 | 0.59 |
| 2018 | mlp-s42 | p75 | 2.53 | 0.00* | 1.50* | 2.53 |
| 2018 | mlp-s42 | unb | 44.36 | 39.09* | 42.59* | 35.50* |
| 2018 | cnn-s42 | p75 | 1.16 | 0.00* | 0.16* | 1.16 |
| 2018 | cnn-s42 | unb | 26.19 | 25.62* | 26.01 | 26.27 |
| 2018 | ft_transformer-s42 | p75 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2018 | ft_transformer-s42 | unb | 0.12 | 0.12 | 0.09 | 0.09 |

How the reference's own timing successes set `shape` (all seeds; `results/report.md`):

- p75 budget: median shape is 0.94–1.00 for MLP/CNN on both datasets. 48–87% of successes are
  above 0.95; at most 0.4% are below 0.05.
- Unbounded budget, CICIDS2018 MLP: median shape 0.00; 56% of successes are below 0.05.

### Findings

- **How the delay is spread matters as much as how much is added.**
  - Proportional dilation (shape 0) stretches the existing gaps.
  - Under the calibrated p75 budget it almost never evades: Valid ASR ≤ 0.06% on every victim,
    against 1.16–13.47% for the reference.
  - Uniform per-gap delay (shape 1) matches the reference within 0.06 pp on every p75 cell
    (none significant).
- **Uniform delay is not universally best.** On CICIDS2018 MLP with the unbounded budget:
  - shape 1 loses 8.9 pp (35.50% vs 44.36%; 1 vs 284 discordant flows at seed 42);
  - shape 0 loses only 5.3 pp.

  The learned shape there is mostly 0. Which allocation evades depends on the victim and the
  budget.
- The learned `shape` is never significantly beaten by any fixed value: the fixed-only
  discordant flows are ≤ 5 at seed 42. Optimizing it is what lets one search cover both
  regimes.
