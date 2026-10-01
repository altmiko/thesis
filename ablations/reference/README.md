# Reference arm

This folder holds the shared baseline that every ablation compares against. It is the FINAL
suite's Hybrid Search (amendment A5, untargeted), run at p75 and unbounded on both datasets:

- three victims each;
- four classes;
- 800 frozen flows per class;
- attack seeds 42/2024/2026;
- validator_v2 `hybrid_valid` in the success predicate;
- capability-aware primitives.

It runs through the ablation harness (`ablations/common`) with the default `HybridConfig`.

Run: `python ablations/reference/run.py --device cuda`.

## Reproduction check

The analysis compares every p75 reference cell with
`FINAL_OUTPUTS/runs/<dataset>/primattack_hybrid_objective_untargeted` (`results/reproduction_check.md`):

| dataset | cells | flows | identical adversarial flow / prediction / validity / success |
|---|---|---|---|
| cicids2017_distrinet | 36 | 28,800 | 28,800 / 28,800 / 28,800 / 28,800 |
| cicids2018_distrinet | 36 | 28,800 | 28,800 / 28,800 / 28,800 / 28,800 |

So the harness reproduces the canonical attack exactly, and every ablation differs from the
FINAL suite only in the component it names.

FINAL ran no untargeted Hybrid at the unbounded budget. Those reference cells are therefore new.
For comparison, Exp A's Prim-PGD unbounded Valid ASR is 22.97% / 59.94% / 0.59% on CICIDS2017
(MLP/CNN/FT). The Hybrid reference unbounded values are 22.97% / 59.94% / 0.55%.
