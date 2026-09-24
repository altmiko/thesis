# PrimAttack experiment protocol

## Evaluation levels

The protocol reports four separate questions in this order:

- **Level 0 — classifier evasion:** did the realized adversarial flow reach Benign?
- **Level 1 — domain validity:** did validator_v2 mark the realized vector `hybrid_valid`?
- **Level 2 — primitive feasibility:** did final projection remain inside the calibrated
  primitive budget and did the primitive transform pass internal consistency checks?
- **Level 3 — flow-level semantic preservation proxy:** did every required available semantic
  test pass?

Level 3 is not packet-level functionality verification.

## 1. Freeze calibration

```text
PYTHONPATH=".;src" python -m attack.primattack_budget \
  --output artifacts/primattack/budget_calibration.json
```

Verify that `fit_split` is `train`, hashes identify `X_train_pristine.npy` and
`y_train_cat.npy`, and `selection_prohibited_inputs` includes test features, predictions, and
adversarial success.

## 2. Single calibrated PrimAttack run

```text
PYTHONPATH=".;src" python -m attack.run_cicids2017_primitive_attack \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn \
  --budget maximum-evaluated \
  --primitive-mode joint \
  --optimizer optimized \
  --seeds 42,43,44 \
  --output-dir outputs/primattack_calibrated_joint_max
```

The victim checkpoints and train-fitted scaler are reused. No model retraining occurs.

## 3. Primitive ablations and budget sensitivity

```text
PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn \
  --test-limit 512 \
  --steps 40 \
  --seeds 42 \
  --output-dir outputs/primattack_budget_sensitivity_full
```

This executes the Cartesian product:

- timing-only, with $p=0$ by hard bound;
- padding-only, with $\alpha=1$ by hard bound;
- joint;
- restricted, intermediate, maximum-evaluated budgets.

The source selector is deterministic and independent of condition. The script asserts exact
`sample_id` equality for every class/victim/seed across all nine configurations and writes
`source_id_consistency.json`.

## 4. Optional random feasible control

Use the same runner with `--optimizer random-feasible`. It samples uniformly inside the exact
same per-flow primitive box and goes through the same projection, transform, validators, and
semantic rules:

```text
PYTHONPATH=".;src" python -m attack.run_cicids2017_primitive_attack \
  --classes DoS,DDoS,Recon,BruteForce --victims mlp,cnn \
  --budget maximum-evaluated --primitive-mode joint \
  --optimizer random-feasible --seeds 42 \
  --output-dir outputs/primattack_random_control
```

## 5. Paired statistics

```text
PYTHONPATH=".;src" python scripts/analyze_primattack_experiments.py \
  --input-dir outputs/primattack_budget_sensitivity_full

PYTHONPATH=".;src" python scripts/build_primattack_budget_report.py \
  --input-dir outputs/primattack_budget_sensitivity_full \
  --output docs/primattack_budget_results.md
```

The analyzer enforces pairing by `sample_id × attack_class × victim_model × seed`.

- two binary conditions: repository McNemar implementation;
- more than two repeated binary conditions: Cochran's Q, then Holm-corrected pairwise McNemar;
- more than two repeated continuous conditions: Friedman, then Holm-corrected Wilcoxon
  signed-rank tests.

## 6. Verification commands

Focused contract suite:

```text
PYTHONPATH=".;src" python -m pytest \
  src/attack/tests/test_primitive_controls.py \
  src/attack/tests/test_primattack_transformation.py \
  src/attack/tests/test_flow_semantics.py \
  src/attack/tests/test_primattack_budget.py \
  src/attack/tests/test_vae_latent_primitive.py \
  -q -p no:faulthandler
```

Final audit also runs the repository unit suite, a small end-to-end attack smoke run, explicit
budget violations, semantic-invariant violations, non-finite checks, and source-ID checks.

## Reproducibility invariants

- Source rows: fixed by class ID and seed `42 + class_id`, unchanged across configurations.
- Victim, target class, scaler, validators, semantic rules, and calibration: unchanged.
- Only named budget and primitive mode vary in the sensitivity design.
- Final classifier success is computed after hard projection and discrete feature generation.
- Every artifact embeds source/checkpoint/calibration hashes and exact configuration.
