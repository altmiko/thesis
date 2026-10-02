# B1 - Convergence: steps, restarts and query budget

## Design

This follows four published sources:

- Carlini et al. (2019, Sec. 4.8): doubling the iterations must not raise success; success should
  be plotted against iterations.
- Tramèr et al. (NeurIPS 2020, Sec. 5): going from 40 to 400 steps exposed non-convergence.
- Amoeba (CoNEXT 2023, Fig. 7) and NetMasquerade (NDSS 2026, Fig. 10): query curves.

| Arm | Steps per restart | Per-flow evaluation budget | Effect |
|---|---|---|---|
| reference | 40 | 256 | ≈ 3 refinement restarts |
| `steps80_eval512` | 80 | 512 | longer restarts, same count |
| `steps160_eval1024` | 160 | 1024 | longer restarts, same count |
| `steps40_eval512` | 40 | 512 | ≈ 2× more restarts |
| `steps40_eval1024` | 40 | 1024 | ≈ 4× more restarts |

Restarts continue until the budget is spent.

Anytime curve: ASR(k) is the share of flows whose first validator-passing success occurs within
their first k victim evaluations (`first_success_evaluation`).

Run: `python ablations/B1_convergence/run.py --device cuda`. Full tables: `results/report.md`,
`results/anytime_curve.csv`.

## Results

**Valid ASR barely moves with 2–4× the budget.** Across 48 comparisons:

- the largest gain is +0.09 pp: 2018 CNN unbounded with `steps40_eval1024` (5 vs 0 discordant
  flows at seed 42);
- none is significant after Holm;
- every p75 cell is identical to the reference under all four arms;
- longer restarts (`steps80`/`steps160`) change at most one flow per cell;
- more 40-step restarts (`steps40_eval512`/`1024`) add at most 0.09 pp (2018 CNN unbounded) and
  0.03 pp (2017 FT unbounded).

**Anytime curve** (reference, mean over seeds): almost all successes appear within the first
32 evaluations of the clean-start restart, or right after the first random restart begins
(between 64 and 128 evaluations).

| dataset | victim | budget | ≤16 | ≤32 | ≤128 | ≤256 (final) |
|---|---|---|---|---|---|---|
| 2017 | mlp | unb | 19.09 | 22.94 | 22.97 | 22.97 |
| 2017 | cnn | p75 | 2.28 | 4.50 | 13.45 | 13.47 |
| 2017 | cnn | unb | 31.81 | 34.28 | 59.93 | 59.94 |
| 2018 | mlp-s42 | unb | 36.62 | 43.94 | 44.34 | 44.36 |
| 2018 | cnn-s42 | unb | 25.78 | 25.87 | 26.17 | 26.19 |

With 160-step restarts the CNN jump comes later, after 512 evaluations, but ends at the same
value. Starting from random points matters, not iterating longer from the clean start. This
agrees with A1's `no_random_restarts`.

### Findings

- **The attack has converged at the FINAL-suite budget.** Carlini's "double the iterations"
  check passes: four times more steps or restarts adds ≤ 0.09 pp of Valid ASR.
- The reported Valid ASR is therefore not an under-optimization artifact. Together with A1
  (search saturation and coverage), A3 (no zero gradients) and B5 (the validator never binds),
  the low Valid ASR is set by:
  - what the calibrated primitive box can reach;
  - the victim.
