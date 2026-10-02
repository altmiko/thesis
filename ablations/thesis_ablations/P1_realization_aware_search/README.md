# P1 - Realization-aware search vs continuous-state search

## Question

Does PrimAttack need to score and select **realized** integer primitive states during the
search, or would a search on the continuous primitive surrogate, realized once at the end,
reach the same final Valid ASR?

## Arms

Both arms run Hybrid Search with the FINAL `PRIM_ARGS` (exact padding enumeration, 40-step
adaptive projected refinement, momentum 0.75, stall halving, restarts until the budget is spent),
targeted → Benign, joint mode, capability-aware M(x), train-fit p75 and envelope-only unbounded
boxes, on the frozen 800 clean-correct test flows per (dataset, victim, class), attack seeds
42/2024/2026.

- `reference`: the FINAL behaviour. Every candidate is projected to integer bytes / µs in the
  per-flow box and M(x) (`project_controls`), mapped through φ with quantization, and scored by
  the victim. The success test (objective hit ∧ validator_v2), the step-size checkpoints, the
  restart bests and the incumbent (success first, lowest normalized cost, best margin among
  failures) all use these realized flows.
- `no_realized_search` (`HybridConfig(realization_aware_search=False)`): the same optimizer and
  the same gradient on the same relaxation. Every candidate is scored on the **continuous**
  state instead: controls clamped to the continuous box `[0, bounds]` and M(x), φ with
  `quantize=False` (`ablations/common/hybrid.py:ContinuousSearch`). Search success = objective
  hit on the continuous flow; checkpoints, restart bests and incumbent use the continuous margin
  and cost. Nothing is rounded during the search. The returned candidate is realized exactly
  once by the canonical code path (`RealizedSearch._score`: integer bytes / µs, box, M(x),
  quantized φ, victim). Raw and Valid ASR count only this realized flow.

Controls held equal: flows, victims, classes, seeds, objective, capability inference, budgets,
φ, the 256-evaluation cap per flow (the continuous search gets 255, the final realization is the
256th) and the final validator_v2 (all four layers, source-conditioned).

The continuous search has no validator_v2 in its success predicate. validator_v2 judges
realized integer flows (its SCHEMA layer requires integer counts, bytes and µs). Amendment A6
showed that the gate never changed a FINAL flow. The continuous flow's validator_v2 verdict is
stored as a diagnostic (`continuous_validator_pass`).

The shared reference arm (`ablations/reference`) is untargeted. P1 therefore runs its own
targeted reference and checks it flow for flow against the FINAL targeted Hybrid cells
(`primattack_hybrid_objective_targeted` p75, `primattack_targeted_budgets` unbounded).

Statistics: paired McNemar on per-flow valid success at seed 42 (exact binomial below 25
discordant pairs), Newcombe 95% CI, Holm over the 12 (dataset, victim, budget) cells. This is the
convention of the other ablations and of amendment A5.

Run: `python ablations/thesis_ablations/P1_realization_aware_search/run.py --device cuda`. Report:
`P1_REALIZATION_AWARE_SEARCH.md`. Tables: `results/report.md`.

## Result

Δ Valid ASR = 0.00 pp in all 12 cells and 3 seeds, with 0 discordant flows (Holm p = 1). 0 of
14,635 continuous hits lose the Benign prediction at the final rounding. Analysis and limits:
`P1_REALIZATION_AWARE_SEARCH.md`.

## Outputs (`results/`)

| file | content |
|---|---|
| `<dataset>/artifacts/*.npz` | per-flow results per (victim, class, budget, arm, seed); P1 also stores the continuous state (`continuous_*`) (git-ignored) |
| `per_flow.csv.gz` | both arms paired per flow, every cell and seed |
| `per_seed.csv` | every metric per cell and seed |
| `summary.csv` | mean ± SD over seeds per cell |
| `tests.csv` | seed-42 McNemar, Newcombe CI, Holm |
| `realization_loss.csv` | continuous hit → realized valid / realized invalid hit / realized miss |
| `budget.csv` | victim evaluations per flow, by component |
| `sanity_checks.csv` | ids, M(x), box, budget, validator recomputation, integer realization, FINAL reproduction |
| `report.md` | all tables |
