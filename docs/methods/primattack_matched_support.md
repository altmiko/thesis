# PrimAttack matched-support CAPGD and C-PGD

## Scope and methodological boundary

`capgd_prim_support` and `cpgd_prim_support` directly optimize feature coordinates:

\[
x'_S = x_S + \delta_S,
\]

where `S` is the set of classifier features that the canonical PrimAttack transform can
write. PrimAttack instead optimizes packet-size and timing primitives and evaluates

\[
x' = g(x, z).
\]

Its directionality, per-flow budgets, integer primitive projection, capability gates, and
deterministic downstream coupling remain part of `g`. The shared mask therefore matches
**downstream feature support only**. It does not give CAPGD or C-PGD PrimAttack's feasible
attack space.

## Canonical support mask

Source of truth: `attack.realizability.cicids2017.primattack_feature_support`. The mask is
resolved by feature name against each dataset's `FeatureManifest`; CICIDS2017-DistriNet and
CSE-CIC-IDS-2018 DistriNet share the same frozen 79-feature order. Indices below are both
zero-based (`index0`, for arrays/code) and one-based (`index1`, for thesis tables).

The derivation followed the executable path, not name matching:

1. `CICIDS2017PrimitiveModel.per_flow_bounds` intersects train-calibrated numeric budgets
   with per-flow semantic capabilities.
2. `project_controls` clamps and rounds padding `p` to integer bytes and total `delay` to
   integer microseconds; `shape` remains in `[0,1]` and is zero when delay is inactive.
3. `generate` applies the projected controls and writes the coordinates listed below.
4. The same function propagates combined packet-length statistics and rates before the
   RobustScaler transform and victim forward pass.

| index0 | index1 | Feature | Primitive effect | Recompute block in `CICIDS2017PrimitiveModel.generate` |
|---:|---:|---|---|---|
| 3 | 4 | Flow Duration | timing | affine allocation of total forward delay |
| 6 | 7 | Total Length of Fwd Packet | padding | forward packet-length augmentation |
| 8 | 9 | Fwd Packet Length Max | padding | forward packet-length augmentation |
| 9 | 10 | Fwd Packet Length Min | padding | forward packet-length augmentation |
| 10 | 11 | Fwd Packet Length Mean | padding | forward packet-length augmentation |
| 16 | 17 | Flow Bytes/s | both | rates |
| 17 | 18 | Flow Packets/s | timing | rates |
| 18 | 19 | Flow IAT Mean | timing | affine allocation of total forward delay |
| 20 | 21 | Flow IAT Max | timing | affine allocation of total forward delay |
| 22 | 23 | Fwd IAT Total | timing | affine allocation of total forward delay |
| 23 | 24 | Fwd IAT Mean | timing | affine allocation of total forward delay |
| 24 | 25 | Fwd IAT Std | timing | affine allocation of total forward delay |
| 25 | 26 | Fwd IAT Max | timing | affine allocation of total forward delay |
| 26 | 27 | Fwd IAT Min | timing | affine allocation of total forward delay |
| 38 | 39 | Fwd Packets/s | timing | rates |
| 39 | 40 | Bwd Packets/s | timing | rates |
| 40 | 41 | Packet Length Min | padding | combined packet-length statistics |
| 41 | 42 | Packet Length Max | padding | combined packet-length statistics |
| 42 | 43 | Packet Length Mean | padding | combined packet-length statistics |
| 43 | 44 | Packet Length Std | padding | combined packet-length statistics |
| 44 | 45 | Packet Length Variance | padding | combined packet-length statistics |
| 54 | 55 | Average Packet Size | padding | combined packet-length statistics |
| 55 | 56 | Fwd Segment Size Avg | padding | forward packet-length augmentation |

Counts: padding 12, timing 12, joint 23; `Flow Bytes/s` is the overlap. Public accessors:

- `primattack_padding_feature_mask(manifest)`
- `primattack_timing_feature_mask(manifest)`
- `primattack_joint_feature_mask(manifest)`

## CAPGD matched-support configuration

`build_capgd_prim_support_resources` reuses the existing CAPGD train-fitted min-max box,
feature types, relation constraints, scaler, loss, adaptive step-size logic, restarts, and
repair implementation. It changes only `Constraints.mutable_features` to
`primattack_joint_feature_mask`. `finalize_capgd_output` then restores every coordinate
outside the mask exactly and asserts bitwise equality. The pre-existing `capgd_native`
configuration and `capgd_prim_p75` primitive-space comparison are unchanged.

CAPGD defaults in the canonical runner:

| Parameter | Default |
|---|---:|
| norm | `L2` |
| epsilon | `0.5` in the train min-max space |
| iterations | `10` |
| restarts | `2` |
| loss | cross entropy (`ce`) |
| EOT iterations | `1` |
| adaptive-step `rho` | `0.75` |
| epsilon margin | `0.01` |
| random start / equality repair | enabled |
| batch size | `64` |

## C-PGD source and formulation

The reproduced method is **Constrained Projected Gradient Descent (C-PGD)** from
Simonetto, Dyrmishi, Ghamizi, Cordy, and Le Traon,
[A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space](https://doi.org/10.24963/ijcai.2022/183),
IJCAI 2022. The authors' public implementation is
[`serval-uni-lu/constrained-attacks`](https://github.com/serval-uni-lu/constrained-attacks).
This is not generic PGD and is separate from the later CAPGD method.

For untargeted evasion, this implementation maximizes

\[
\operatorname{CE}(h(x'), y) - \lambda C(x'),
\qquad \lVert x'-x\rVert_p \le \epsilon,
\]

which is equivalent to minimizing `attack_loss + lambda * constraint_violation_loss`
with `attack_loss = -CE`. `C(x')` is the differentiable sum of the explicitly encoded
CAPGD/C-PGD relation penalties: extractor equalities for forward packet mean, forward
segment mean, forward IAT mean, four rate identities, and the packet-length/IAT order
relations. Bounds and the mutable mask are projected every iteration. Declared integer
features are repaired at the end; if integer repair crosses the norm boundary, those
integer changes are reverted and the continuous coordinates are reprojected.

Validator-v2's schema, extractor, protocol, and mined verdicts remain independent
post-attack checks on the exact final sample. They are not silently approximated in the
C-PGD objective.

C-PGD defaults:

| Parameter | Default |
|---|---:|
| norm | `L2` |
| epsilon | `0.5` in the train min-max space |
| step size | `0.05` |
| iterations | `40` |
| constraint penalty weight | `1.0` |
| loss | untargeted cross entropy (`ce`) |
| random start | enabled |
| batch size | `64` |
| seeds | `42,2024,2026` |

## Runner names and commands

Both methods use `scripts/run_full_adversarial_eval.py` and its common evaluator/artifact
schema. `--attacks` restricts an expanded family roster to exact method names.

CICIDS2017-DistriNet smoke or bounded run:

```powershell
$Env:PYTHONPATH = ".;src"
python scripts/run_full_adversarial_eval.py `
  --dataset cicids2017 --families capgd,cpgd `
  --attacks capgd_prim_support,cpgd_prim_support `
  --victims mlp --classes DoS --n-per-class 8 `
  --seeds 42,2024,2026 `
  --output-dir outputs/smoke_matched_support/cicids2017_distrinet
```

CSE-CIC-IDS-2018 DistriNet:

```powershell
$Env:PYTHONPATH = ".;src"
python scripts/run_full_adversarial_eval.py `
  --dataset cicids2018 --families capgd,cpgd `
  --attacks capgd_prim_support,cpgd_prim_support `
  --victims mlp-s42 --classes DoS --n-per-class 8 `
  --seeds 42,2024,2026 `
  --output-dir outputs/smoke_matched_support/cicids2018_distrinet
```

Each matched-support artifact records the 79-bit mutable mask, 23 allowed features,
per-sample modified-feature count, zero-required outside-mask count, objective, parameters,
iterations/model evaluations where available, runtime, predictions, raw/valid success, and
the independent validator verdict.

## Focused tests and smoke results

Focused verification command:

```powershell
$Env:PYTHONPATH = ".;src"
python -m pytest src/attack/tests/test_primattack_support_mask.py `
  src/attack/tests/test_primattack_transformation.py `
  tests/test_capgd_comparison.py -q -p no:faulthandler
```

Observed on 2026-09-26: **19 passed**. The tests resolve the exact indices for both
manifests; compare observed padding/timing recomputation sensitivity with each complete
mask; exercise native and matched-support CAPGD; verify victim and constraint gradients;
and check C-PGD type, immutable-coordinate, L2/Linf projection, and evaluator contracts.

A broader primitive/CAPGD regression selection completed with **46 passed, 1 skipped**:
`test_primitive_controls.py`, `test_primitive_optimizer.py`,
`test_primattack_support_mask.py`, `test_primattack_transformation.py`, and
`test_capgd_comparison.py`. The only warning was NumPy's upstream
`__array_wrap__` deprecation inside the frozen CAPGD dependency.

The smoke commands above were reduced to `--n-per-class 4 --seeds 42
--capgd-steps 3 --cpgd-iterations 3` for execution. These are execution checks, not final
thesis estimates.

| Dataset / victim / class | Method | Rows | Modified features per row | Outside-mask modifications | Raw successes | Validator passes | Valid successes |
|---|---|---:|---|---|---:|---:|---:|
| CICIDS2017 / MLP / DoS | `capgd_prim_support` | 4 | 21–23 | 0 for every row | 4 | 0 | 0 |
| CICIDS2017 / MLP / DoS | `cpgd_prim_support` | 4 | 21–23 | 0 for every row | 0 | 0 | 0 |
| CICIDS2018 / MLP seed 42 / DoS | `capgd_prim_support` | 4 | 20–23 | 0 for every row | 2 | 0 | 0 |
| CICIDS2018 / MLP seed 42 / DoS | `cpgd_prim_support` | 4 | 21–23 | 0 for every row | 1 | 0 | 0 |

All four artifacts were non-empty, contained the required raw-success,
validator-pass, valid-success, parameter, failure-status, and support-count fields, and
reported 23 allowed features. `failures.json` was empty for both datasets. A separate
two-row CICIDS2017 `capgd_native` smoke run with three iterations also completed with no
failures and passed the runner's pairing assertions.
