# P1 - Realization-aware search ablation (`NoRealizedSearch`)

Design, arms and controls: `README.md`. Full tables: `results/report.md`. Per-flow data:
`results/per_flow.csv.gz`.

**Result.** Scoring and selecting realized integer states during the search does not change
the final Valid ASR on any of the 12 cells. Under the tested granularity (integer bytes,
integer µs), realization-aware search is empirically unnecessary. Realizing the returned
continuous candidate once gives the same successes, and in 99.0% of flows the same final flow.

## Setup in brief

- Targeted → Benign, Hybrid Search (`PRIM_ARGS`), joint mode, capability-aware M(x), p75 and
  unbounded boxes, 2 datasets × 3 victims × 2 budgets = 12 cells, attack seeds 42/2024/2026,
  4 classes × 800 frozen clean-correct flows per cell and seed (3,200; 9,600 over seeds).
- `reference` = FINAL behaviour. `no_realized_search` = same optimizer with every candidate
  scored and selected on the continuous state (no rounding, unquantized φ), then one canonical
  realization (integer bytes / µs, box, M(x), quantized φ) of the returned candidate.
- Both arms end with the same full validator_v2. At most 256 victim evaluations per flow in
  both arms (the continuous search gets 255; the final realization is the 256th).
- The shared ablation reference (`ablations/reference`) is untargeted. P1 runs its own targeted
  reference, so its numbers match FINAL Exp B / C (targeted Hybrid), not the untargeted
  headline.

## Valid ASR (mean ± SD over attack seeds, %)

| dataset | victim | budget | ref Raw | ref Valid | P1 continuous hit | P1 Raw | P1 Valid | Δ Valid (pp) | seed-42 P1-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09 ± 0.00 | 4.09 ± 0.00 | 4.09 ± 0.00 | 4.09 ± 0.00 | 4.09 ± 0.00 | 0.00 | 0 / 0 | 1 |
| 2017 | mlp | unb | 22.94 ± 0.00 | 22.94 ± 0.00 | 22.94 ± 0.00 | 22.94 ± 0.00 | 22.94 ± 0.00 | 0.00 | 0 / 0 | 1 |
| 2017 | cnn | p75 | 13.25 ± 0.00 | 13.25 ± 0.00 | 13.25 ± 0.00 | 13.25 ± 0.00 | 13.25 ± 0.00 | 0.00 | 0 / 0 | 1 |
| 2017 | cnn | unb | 59.69 ± 0.00 | 59.69 ± 0.00 | 59.69 ± 0.00 | 59.69 ± 0.00 | 59.69 ± 0.00 | 0.00 | 0 / 0 | 1 |
| 2017 | ft_transformer | p75 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.00 | 0 / 0 | 1 |
| 2017 | ft_transformer | unb | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.55 ± 0.02 | 0.00 | 0 / 0 | 1 |
| 2018 | mlp-s42 | p75 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.78 ± 0.00 | 0.00 | 0 / 0 | 1 |
| 2018 | mlp-s42 | unb | 24.80 ± 0.02 | 24.80 ± 0.02 | 24.80 ± 0.02 | 24.80 ± 0.02 | 24.80 ± 0.02 | 0.00 | 0 / 0 | 1 |
| 2018 | cnn-s42 | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 / 0 | 1 |
| 2018 | cnn-s42 | unb | 26.09 ± 0.00 | 26.09 ± 0.00 | 26.09 ± 0.00 | 26.09 ± 0.00 | 26.09 ± 0.00 | 0.00 | 0 / 0 | 1 |
| 2018 | ft_transformer-s42 | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 / 0 | 1 |
| 2018 | ft_transformer-s42 | unb | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.12 ± 0.00 | 0.00 | 0 / 0 | 1 |

The paired comparison finds no discordant flow at seed 42 in any cell. Seeds 2024 and 2026 also
have 0 / 0 discordant flows everywhere (`results/per_seed.csv`). McNemar p = 1 and Holm-adjusted
p = 1 in all 12 cells. The median normalized primitive cost of valid successes is identical
per cell and seed (e.g. 2017 CNN p75: 0.580 / 0.600 / 0.608 in both arms). See
`results/per_seed.csv`.

## Answers

**Does search-time realization materially improve Valid ASR?** No. Δ Valid ASR = 0.00 pp in
all 12 cells and all 3 seeds. The valid-success sets are identical flow for flow.

**How often does a continuous adversarial state stop working after integer realization?**
Never in this experiment. Of 14,635 continuous hits (all cells and seeds), 0 lose the Benign
prediction after realization, and 0 flows of any kind change their predicted class
(`results/realization_loss.csv`). No continuous miss turns into a realized hit either. The
rounding is real but small:

- 4,610 of the 14,635 continuous hits had a non-integer delay; rounding changes the delay by
  less than 1 µs (≤ 0.5 µs on unbounded cells; up to 0.999 µs on p75 cells, where the integer
  cap `floor(delay_hi)` binds);
- the smallest added delay among valid successes is 1,305 µs and the medians are 5.9 ms to 70 s,
  so the rounding is ≤ 0.08% of the delay;
- no cell has a joint row: every capability-aware flow is timing-only, padding-only or
  no-primitive. Padding is reached only by the exact integer enumeration, so the continuous
  and realized padding never differ (max |Δp| = 0);
- the victim margin moves by at most 6.5 × 10⁻³ under rounding. The closest continuous hit to
  the boundary has margin −5 × 10⁻⁵ (2018 CNN unbounded), so a flip was possible, but none
  occurred.

**Is any difference statistically significant?** No. There are 0 discordant pairs in every
cell, so McNemar cannot reject equality (p = 1, Holm p = 1).

**Does realization mainly affect classifier success, validity, or both?** In P1, realization
changes validity, not the classifier outcome. Before realization, 13,801 of the 14,635
continuous hits (94.3%) fail validator_v2. Every one of them fails the SCHEMA integer checks.
Each failing flow has a fractional value in an integer-typed µs feature: Fwd IAT Min (13,615
flows), Fwd IAT Max (6,898), and Fwd IAT Total, Flow Duration or Flow IAT Max (≈ 2,990). The
allocation scales the existing gaps and spreads the delay over them, so the IAT extremes are
fractional even when the total delay is an integer. After the one final realization, all
14,635 are valid. The classifier verdict is the same before and after for every flow. The
realization step is therefore required for validity. Doing it inside the search adds nothing at
this granularity.

## Why the arms coincide

Both arms differentiate the same unquantized relaxation from the same start points with the same
seeded restarts, so they share the gradient path until a decision that reads a margin or a
success differs: a stall checkpoint, a restart best or the incumbent. Rounding moves the delay
by less than 1 µs and the margin by at most 6.5 × 10⁻³, which almost never flips such a
decision. The final realized flow is bit-identical to the reference's in
114,076 of 115,200 flows (99.0%) and in 14,630 of 14,635 reference valid successes. The 5
differing successes (2017 MLP unbounded) are valid in both arms; 4 differ in the selected
`shape` and 1 by 2 µs of delay.

## Sanity checks (`results/sanity_checks.csv`)

| check | result |
|---|---|
| reference vs FINAL targeted Hybrid (`primattack_hybrid_objective_targeted` p75, `primattack_targeted_budgets` unb) | 115,200 / 115,200 flows: identical adversarial flow, prediction, validity, valid success |
| same source-flow IDs in both arms; frozen selection SHA-256 | 36 / 36 cell-seeds; 36 / 36 |
| same capability mask M(x) and per-flow box in both arms; M(x) recomputed from source = stored | 36 / 36 and 36 / 36; 115,200 / 115,200 per arm |
| victim evaluations per flow | max 255 (ref), 256 (P1); P1 = ≤ 255 continuous search + exactly 1 final realization on every flow. Mean 188.5–190.8 (ref) vs 189.5–191.8 (P1); both arms take the same number of gradient steps (2 evaluations each), and the P1 surplus is the final realization |
| same final validator_v2 | verdict recomputed with `structural_masks` on the stored flows equals the stored verdict, 115,200 / 115,200 per arm |
| P1 outputs integer-realized | realized bytes and µs integer, stored controls = canonical `project_controls` of the requested controls, stored flow = quantized φ of the stored controls (bit-identical): 115,200 / 115,200 each |
| no success counted from the continuous surrogate | P1 Raw success = realized prediction == Benign and = search success in 36 / 36; the continuous hit is stored only as `continuous_success` |

## Limits

- One primitive granularity (integer bytes, integer µs). With coarser timestamp resolution
  (e.g. ms) or with joint padding+timing rows in the refinement, rounding moves the flow
  further, and this result need not hold.
- Targeted → Benign only; the untargeted ablation suite was not rerun.
- The continuous search has no validator_v2 in its success predicate. The reference has it, and
  amendment A6 showed the gate never changes a FINAL flow. The continuous-flow verdict is a
  diagnostic.
- Feature-space proxies on CICFlowMeter aggregates; no PCAP edited or replayed.
