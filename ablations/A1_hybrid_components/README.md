# A1 - Hybrid Search component leave-one-out

## Design

This follows two published ablations:

- CAA's CAPGD component ablation (Simonetto et al., NeurIPS 2024, App. B.1 Table 7), including
  its coverage metric (Fig. 4);
- NetMasquerade's stage knock-out (NDSS 2026, App. D Table VI).

Each arm removes exactly one component of the canonical Hybrid Search. Everything else is the
shared protocol (`../README.md`).

| Arm | Removed component |
|---|---|
| `no_padding_sweep` | exhaustive integer padding enumeration (padding left to the gradient) |
| `no_refinement` | the whole gradient stage (identity + padding enumeration only) |
| `no_random_restarts` | random restarts (clean-start refinement only) |
| `fixed_step` | stall-triggered step halving / reset to the restart's best point |
| `no_momentum` | momentum (0 instead of 0.75) |
| `no_surrogate_floor` | straight-through floor of the relaxation at zero controls |
| `validator_post_hoc` | validator_v2 in the search success predicate (judged afterwards only) |
| `last_iterate` | success-first lowest-cost incumbent (last evaluated candidate returned) |

The coverage analysis (seed 42) adds the FINAL suite's Prim-PGD (Exp A PrimAttack row) as an
external arm.

Run: `python ablations/A1_hybrid_components/run.py --device cuda`. Full tables:
`results/report.md`, `results/coverage_seed42.csv`.

## Results

Valid ASR in % (mean of seeds 42/2024/2026). `*` marks a significant difference from the
reference: seed-42 McNemar, Holm over all 96 comparisons. Arms equal to the reference in every
cell are omitted from the table.

| dataset | victim | budget | reference | no_refinement | no_random_restarts | last_iterate |
|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09 | 0.00* | 4.09 | 3.89 |
| 2017 | mlp | unb | 22.97 | 0.00* | 22.94 | 22.31* |
| 2017 | cnn | p75 | 13.47 | 0.00* | 4.50* | 12.33* |
| 2017 | cnn | unb | 59.94 | 0.00* | 34.28* | 59.03* |
| 2017 | ft_transformer | p75 | 0.12 | 0.00 | 0.12 | 0.12 |
| 2017 | ft_transformer | unb | 0.55 | 0.00* | 0.50 | 0.32 |
| 2018 | mlp-s42 | p75 | 2.53 | 0.00* | 2.16* | 2.26 |
| 2018 | mlp-s42 | unb | 44.36 | 0.00* | 43.94* | 42.69* |
| 2018 | cnn-s42 | p75 | 1.16 | 0.00* | 1.06 | 0.86 |
| 2018 | cnn-s42 | unb | 26.19 | 0.00* | 25.87 | 26.12 |
| 2018 | ft_transformer-s42 | p75 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2018 | ft_transformer-s42 | unb | 0.12 | 0.00 | 0.12 | 0.09 |

The other five arms never differ significantly from the reference:

- `no_padding_sweep` and `validator_post_hoc` are identical to the reference in every cell.
- `fixed_step`, `no_momentum` and `no_surrogate_floor` differ by at most 0.3 pp.

### Findings

- **The gradient refinement stage is the attack.** Without it Valid ASR is 0 everywhere. With
  capability-aware padding, no valid success uses padding (see `../D5_overhead`). The exhaustive
  padding enumeration contributes nothing; it only costs evaluations on flows that can pad.
- **Random restarts matter for CNN on CICIDS2017.** Without them Valid ASR drops:
  - p75: 13.47% → 4.50%;
  - unbounded: 59.94% → 34.28%.

  The CNN's margin surface is non-convex in the 3-D control space, so the clean start alone
  reaches only a third of the successes. Elsewhere the loss is ≤ 0.43 pp.
- **Success-first incumbent selection helps a little.** Returning the last iterate instead
  loses up to 1.7 pp. Refinement keeps iterating on successful flows and can walk back out of
  success.
- **Momentum, adaptive step size and the surrogate floor are inert at this budget.** The
  validator in the predicate is inert as well: putting it in the search changes no outcome,
  because under capability-aware primitives the realized flows pass validator_v2 anyway (see
  `../B5_validator_layers`).
- **Coverage (seed 42).**
  - The reference covers 100% of every ablated arm's successes, except 3 flows: 1 found by
    `no_momentum` on 2017 FT unbounded and 2 on 2018 CNN unbounded.
  - FINAL Prim-PGD covers the reference's successes completely, except 1 flow (2018 MLP
    unbounded).
  - Prim-PGD adds 1 flow (2017 FT unbounded) and 4 flows (2018 CNN unbounded) outside the
    reference.
  - The union of all nine Hybrid variants and Prim-PGD is at most 4 flows larger than the
    reference in any cell. The search is saturated: the low Valid ASR reflects how few flows
    are breakable at all, not a weak optimizer.
