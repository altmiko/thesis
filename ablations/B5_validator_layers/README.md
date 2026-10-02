# B5 - Remove one validator_v2 layer at a time

## Design

This follows two published sources:

- Carlini et al., On Evaluating Adversarial Robustness (2019, Sec. 5.2–5.3): relax the
  constraints and check whether the success rate rises;
- Sheatsley et al. (JCS 2022, Fig. 4).

validator_v2 `hybrid_valid` = SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED. Each arm drops exactly one
layer, both from the search success predicate and from the measured validity:

| Arm | Validity used |
|---|---|
| `drop_schema` | EXTRACTOR ∧ PROTOCOL ∧ MINED |
| `drop_extractor` | SCHEMA ∧ PROTOCOL ∧ MINED |
| `drop_protocol` | SCHEMA ∧ EXTRACTOR ∧ MINED |
| `drop_mined` | SCHEMA ∧ EXTRACTOR ∧ PROTOCOL |
| reference | all four |

- Primary outcome: Valid ASR under the arm's reduced validator.
- Secondary outcomes:
  - Valid ASR of the same flows under the full validator_v2;
  - how many reduced-validator successes the dropped layer rejects.

Run: `python ablations/B5_validator_layers/run.py --device cuda`. Full tables:
`results/report.md`, `results/dropped_layer_audit.csv`.

## Results

**Dropping any single layer changes nothing.** Across 2 datasets × 3 victims × 2 budgets ×
4 layers (48 comparisons), every arm's Valid ASR is identical to the reference's:

- under the reduced validator;
- under the full validator;
- in Raw ASR.

There are 0 discordant flows at seed 42 in every comparison. No reduced-validator success fails
the dropped layer (`fail dropped` = 0 in all 48 cells).

The same holds without the validator in the search at all: A1's `validator_post_hoc` equals the
reference everywhere. Every layer's pass rate over the final flows is ≥ 99.6% in every cell.

### Findings

- **For capability-aware PrimAttack, validator_v2 never binds.**
  - The canonical map φ recomputes every dependent feature exactly.
  - The capability mask keeps primitives within what the source flow admits.
  - Together these mean no realized PrimAttack flow that meets the objective is rejected by any
    validator layer.
  - So the low Valid ASR is **not** caused by the validator. It is set by the calibrated budget
    and by the victim (`../D5_overhead`, `../A1_hybrid_components`). Raw ASR = Valid ASR in
    every cell.
- **The validator does bind once capability inference is removed** (`../thesis_ablations/D6_capability_inference`):
  - PROTOCOL rejects all capability-violating objective hits on CICIDS2017;
  - MINED and PROTOCOL reject most of them on CICIDS2018.

  Validity of PrimAttack's successes is guaranteed by construction (φ + M(x)) and confirmed by
  validator_v2, not filtered by it.
- **This is the opposite of the feature-space baselines.** For unconstrained PGD/C&W, Raw ASR
  ≈ 1 but Valid ASR = 0 (FINAL Exp A). There the validator is the whole story.
