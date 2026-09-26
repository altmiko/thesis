# C-PGD: Method, Implementation, Runner Integration, and Verification

## 1. Purpose

This document explains the constrained projected-gradient-descent baseline implemented as
`cpgd_prim_support` for CICIDS2017-DistriNet and CSE-CIC-IDS-2018 DistriNet. It covers:

- the exact published method being reproduced;
- the optimization objective and constraint penalty;
- the attack space and its relationship to PrimAttack;
- the PyTorch implementation;
- projection, bounds, feature types, and finalization;
- runner and evaluator integration;
- output artifacts, tests, smoke results, and limitations.

The implementation is in:

- `src/comparisons/cpgd_prim_support.py`
- `src/comparisons/capgd_cicids2017.py`
- `scripts/run_full_adversarial_eval.py`

The method identifier written to artifacts is `cpgd_prim_support`.

---

## 2. Which C-PGD is implemented

The implementation follows **Constrained Projected Gradient Descent (C-PGD)** from:

> Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, and Yves Le Traon,
> “A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space,”
> IJCAI 2022. DOI: [10.24963/ijcai.2022/183](https://doi.org/10.24963/ijcai.2022/183).

Public author implementation:

- [https://github.com/serval-uni-lu/constrained-attacks](https://github.com/serval-uni-lu/constrained-attacks)

This distinction matters:

- **C-PGD** augments projected gradient descent with a differentiable constraint-violation
  penalty.
- **CAPGD** is the later constrained adaptive PGD method with adaptive step-size behavior,
  momentum/restart logic, and other attack-specific machinery.
- **Generic PGD** optimizes only the classifier attack loss and has no constraint penalty.

The local implementation is therefore not an unconstrained PGD attack renamed as C-PGD.
Its gradient objective contains both the classifier loss and the differentiable relation
penalty.

---

## 3. Threat model and attack space

### 3.1 White-box access

C-PGD is a white-box attack. It differentiates through:

1. the train-fitted min-max inverse transform;
2. the raw-space relation penalty;
3. the victim’s existing RobustScaler transform;
4. the frozen neural classifier;
5. the untargeted cross-entropy loss.

The attack receives the true source class for every clean-correct malicious flow.

### 3.2 Direct feature-space control

Let the clean 79-feature vector be `x`, and let `S` be the 23-coordinate
PrimAttack-derived support mask. C-PGD directly optimizes:

\[
x'_S = x_S + \delta_S,
\qquad
x'_{\bar S}=x_{\bar S}.
\]

Every coordinate outside `S` is restored from the clean input after every projection and
again after type repair. Final output contains a bitwise-equality assertion for all
coordinates outside `S`.

### 3.3 This is not PrimAttack’s feasible space

PrimAttack optimizes primitive controls:

\[
z=(p,\text{delay},\text{shape}),
\qquad
x'=g(x,z).
\]

Its packet-size and timing controls impose:

- increase-only directions;
- per-flow capability gates;
- train-calibrated primitive budgets;
- integer bytes and integer microseconds;
- deterministic coupling between downstream features.

C-PGD does not inherit those restrictions. It can directly move any coordinate in `S`
subject to its own norm ball, train-derived bounds, declared feature types, and
constraint penalty. Consequently, `cpgd_prim_support` matches **which downstream
coordinates may change**, not **which joint vectors PrimAttack can realize**.

---

## 4. Optimization objective

For clean input `x`, true label `y`, classifier `h`, perturbation budget `epsilon`, norm
`p`, and differentiable constraint penalty `C`, the local attack maximizes:

\[
J(x') = \operatorname{CE}(h(x'),y)-\lambda C(x')
\]

subject to:

\[
\lVert x'-x\rVert_p \le \epsilon,
\qquad
x'_{\bar S}=x_{\bar S},
\qquad
x'\in\mathcal B_{\text{train}}.
\]

The equivalent minimization notation is:

\[
\underbrace{-\operatorname{CE}(h(x'),y)}_{\text{attack loss}}
+
\lambda\underbrace{C(x')}_{\text{constraint violation loss}}.
\]

This equivalence explains the source code:

```python
attack_loss = F.cross_entropy(logits, labels)
constraint_loss = constraint_violation(raw).mean()
score = attack_loss - constraint_penalty_weight * constraint_loss
```

The implementation performs gradient **ascent** on `score`.

### 4.1 Untargeted objective

The objective is untargeted. Cross entropy is evaluated against the true malicious class,
and ascent increases the classifier’s loss for that class. Runner success is:

\[
\text{raw success} = [\hat y(x') \ne y].
\]

The attack does not optimize specifically for the Benign class. The evaluator still
records `targeted_success` for comparability, but the declared objective and reported
headline result are untargeted evasion.

### 4.2 Constraint weight

`constraint_penalty_weight` is the coefficient `lambda`. Its default is `1.0`.

- `lambda = 0` reduces the gradient objective to masked PGD.
- Larger values place more gradient weight on reducing encoded relation violations.
- The independent validator remains authoritative regardless of `lambda`.

No automatic penalty-weight schedule is silently applied.

---

## 5. Differentiable constraints used by C-PGD

C-PGD reuses the relation definitions already supplied to the frozen TabularBench CAPGD
adapter. They are built by `_relations` in
`src/comparisons/capgd_cicids2017.py` and executed through TabularBench’s
`ConstraintsExecutor` with its `PytorchBackend`.

### 5.1 Equality relations

The following raw-space equalities contribute differentiable penalties:

| Target | Relation | Tolerance |
|---|---|---:|
| `Fwd Packet Length Mean` | `Total Length of Fwd Packet / Total Fwd Packet` | `1e-3` |
| `Fwd Segment Size Avg` | `Fwd Packet Length Mean` | `1e-3` |
| `Fwd IAT Mean` | `Fwd IAT Total / (Total Fwd Packet - 1)` | `1e-3` |
| `Fwd Packets/s` | `Total Fwd Packet * 1e6 / Flow Duration` | `0.1` |
| `Bwd Packets/s` | `Total Bwd packets * 1e6 / Flow Duration` | `0.1` |
| `Flow Packets/s` | `(Total Fwd Packet + Total Bwd packets) * 1e6 / Flow Duration` | `0.1` |
| `Flow Bytes/s` | `(Total Length of Fwd Packet + Total Length of Bwd Packet) * 1e6 / Flow Duration` | `0.1` |

Safe division uses a zero fill value when the divisor is not safely nonzero.

For equality `a = b` with tolerance `tau`, the backend penalty is:

\[
C_{=}(a,b)=\max(|a-b|-\tau,0).
\]

### 5.2 Ordering relations

Four inequality terms are included:

\[
\begin{aligned}
\text{Fwd Packet Length Min} &\le \text{Fwd Packet Length Mean},\\
\text{Fwd Packet Length Mean} &\le \text{Fwd Packet Length Max},\\
\text{Fwd IAT Min} &\le \text{Fwd IAT Mean},\\
\text{Fwd IAT Mean} &\le \text{Fwd IAT Max}.
\end{aligned}
\]

For `a <= b`, the backend penalty is:

\[
C_{\le}(a,b)=\max(a-b,0).
\]

### 5.3 Combined penalty

The relation list is wrapped in `AndConstraint`. The PyTorch backend implements logical
AND as a sum of violation terms, producing one differentiable scalar per sample:

\[
C(x')=\sum_i C_i(x').
\]

`constraint_violation` checks that the executor returns exactly one value per input row.

### 5.4 What is not in the gradient penalty

The following are not silently approximated:

- validator-v2 schema checks;
- extractor checks not represented by the relations above;
- protocol rules;
- mined validator rules;
- packet-level or PCAP semantics;
- PrimAttack primitive feasibility;
- PrimAttack semantic-preservation verdicts.

They remain independent post-attack checks. This prevents C-PGD from being credited with
constraints it did not actually optimize.

---

## 6. Coordinate systems and model path

Three coordinate systems are involved.

### 6.1 Pristine/raw feature space

Dataset arrays `X_*_pristine.npy` contain raw CICFlowMeter units. Relation penalties,
feature types, validator checks, and train min/max bounds are defined here.

### 6.2 Train min-max attack space

The shared TabularBench scaler is fitted only from `X_train_pristine.npy`:

\[
u_j = \frac{x_j - \min_{\text{train},j}}
{\max_{\text{train},j}-\min_{\text{train},j}}.
\]

C-PGD performs norm projection and gradient steps in this space. Under ordinary
train-range values, the box is `[0,1]^79`.

If a held-out clean coordinate is outside the train range, the projector includes the
clean value as an endpoint rather than forcing an unrelated move into `[0,1]`. The attack
cannot move that coordinate farther outside the fitted range. This avoids violating the
norm budget merely because a clean test value exceeds a train extremum.

### 6.3 Victim RobustScaler space

`RawCICIDSVictim` exposes the existing victim as a raw-space module:

\[
h_{\text{raw}}(x)=h_{\text{victim}}
\left(\frac{x-\text{center}}{\text{scale}}\right).
\]

The attack path is therefore:

```text
min-max attack variable
    -> inverse train min-max transform
    -> pristine/raw features
    -> existing victim RobustScaler transform
    -> frozen victim logits
    -> cross-entropy loss
```

The victim scaler and checkpoint are not refitted or modified.

---

## 7. Attack algorithm

### 7.1 Initialization

`CPGDPrimSupportAttack` requires resources whose configuration is exactly
`capgd_prim_support`. This fail-loud check prevents accidental execution with CAPGD’s
native mask.

The attack stores:

- the shared resource bundle;
- raw-space victim wrapper;
- validated `CPGDConfig`;
- attack seed and device;
- the 79-bit PrimAttack support mask;
- one differentiable relation executor.

### 7.2 Random start

Random initialization is enabled by default.

For L2:

1. sample a standard-normal vector;
2. zero every unmasked coordinate;
3. normalize to unit L2 length;
4. multiply by a uniform radius in `[0, epsilon]`.

For Linf:

1. sample each allowed coordinate uniformly in `[-1,1]`;
2. multiply by `epsilon`;
3. zero every unmasked coordinate.

The random candidate is immediately passed through the same projection used after every
optimization step.

### 7.3 Per-iteration update

For iteration `t`:

1. Enable gradients on the current min-max candidate `u_t`.
2. Invert it to raw space.
3. Evaluate victim logits and untargeted cross entropy.
4. Evaluate the differentiable relation penalty.
5. Form `CE - lambda * penalty`.
6. Differentiate with respect to `u_t`.
7. Fail if any gradient is NaN or infinite.
8. Zero all gradients outside the 23-feature mask.
9. Convert the gradient to the norm-specific direction.
10. Take one ascent step.
11. Project into the norm ball, train box, and immutable-coordinate set.
12. Detach before the next iteration.

L2 uses a normalized gradient:

\[
d_t=\frac{\nabla J(u_t)}{\lVert\nabla J(u_t)\rVert_2}.
\]

Linf uses the sign direction:

\[
d_t=\operatorname{sign}(\nabla J(u_t)).
\]

The update is:

\[
u_{t+1}=\Pi_{\mathcal F}
\left(u_t+\alpha d_t\right),
\]

where `F` is the intersection of the norm ball, feature box, and mutable support.

### 7.4 Norm projection

For L2, a perturbation outside the radius is radially scaled:

\[
\delta \leftarrow \delta
\min\left(1,\frac{\epsilon}{\lVert\delta\rVert_2}\right).
\]

For Linf, each allowed coordinate is clamped:

\[
\delta_j\leftarrow\operatorname{clip}(\delta_j,-\epsilon,\epsilon).
\]

After projection, `torch.where` copies all unmasked coordinates from the clean input.

---

## 8. Final type repair and epsilon preservation

The continuous optimizer can produce non-integral values for integer features. Finalization
uses TabularBench’s existing `fix_types` behavior to repair declared integer coordinates.

Integer repair can increase the normalized distance. The implementation therefore checks
the final L2 or Linf distance after repair.

For any row whose integer repair crosses `epsilon`:

1. revert changed integer support coordinates to their clean values;
2. reproject the remaining continuous coordinates;
3. invert back to raw space;
4. restore the clean integer coordinates again;
5. restore every unmasked coordinate.

Two final assertions enforce:

- normalized distance is at most `epsilon + 1e-5`;
- all unmasked raw coordinates are exactly equal to the clean sample.

This is conservative: a row may lose useful integer changes rather than violate the
attack’s declared norm or type contract.

---

## 9. Configuration and defaults

`CPGDConfig` is immutable and validates every field.

| Parameter | Default | Validation |
|---|---:|---|
| `epsilon` | `0.5` | non-negative |
| `norm` | `L2` | `L2` or `Linf` |
| `step_size` | `0.05` | positive |
| `iterations` | `40` | positive integer |
| `constraint_penalty_weight` | `1.0` | non-negative |
| `loss` | `ce` | cross entropy only |
| `random_start` | `True` | Boolean configuration |
| runner batch size | `64` | CLI integer |
| default attack seeds | `42,2024,2026` | parsed by runner |

Invalid values fail before attack execution.

Runner flags:

```text
--cpgd-epsilon
--cpgd-norm {L2,Linf}
--cpgd-step-size
--cpgd-iterations
--cpgd-constraint-weight
--cpgd-loss {ce}
--cpgd-batch-size
```

---

## 10. Runner integration

The canonical runner expands family `cpgd` to:

```text
name: cpgd_prim_support
kind: cpgd_prim_support
goal: untargeted
```

Resources are built once:

1. `build_capgd_resources` fits the train-only min/max box and loads feature types and
   relation definitions.
2. `build_capgd_prim_support_resources` replaces the mutable mask with the canonical
   PrimAttack joint support.
3. C-PGD and matched-support CAPGD consume the same resource object and therefore the same
   feature support, train bounds, feature types, and relation definitions.

The runner batches rows, runs C-PGD, concatenates results, and records:

- iterations per row;
- model evaluations per row;
- final differentiable constraint violation;
- exact attack parameters and mutable indices.

### 10.1 Example command

Small CICIDS2017 run:

```powershell
$Env:PYTHONPATH = ".;src"
python scripts/run_full_adversarial_eval.py `
  --dataset cicids2017 `
  --families cpgd `
  --attacks cpgd_prim_support `
  --victims mlp `
  --classes DoS `
  --n-per-class 8 `
  --seeds 42 `
  --cpgd-iterations 5 `
  --cpgd-step-size 0.05 `
  --device cpu `
  --output-dir outputs/smoke_matched_support/cicids2017_cpgd
```

Run both matched-support attacks:

```powershell
$Env:PYTHONPATH = ".;src"
python scripts/run_full_adversarial_eval.py `
  --dataset cicids2017 `
  --families capgd,cpgd `
  --attacks capgd_prim_support,cpgd_prim_support `
  --victims mlp `
  --classes DoS `
  --n-per-class 8 `
  --seeds 42,2024,2026 `
  --output-dir outputs/adv_campaign_matched_support/cicids2017_distrinet
```

The full thesis attack suite was not run during implementation or smoke verification.

---

## 11. Independent evaluation

The exact finalized `adversarial_raw` tensor is passed to the existing evaluator.

The evaluator computes:

- clean and adversarial predictions;
- untargeted evasion;
- targeted-to-Benign outcome for compatibility;
- validator-v2 `hybrid_valid`;
- validator-v2 hard structural validity;
- internal primitive-transform consistency as a separate diagnostic;
- scaled perturbation and cost summaries.

For C-PGD:

\[
\text{valid success}
=
\text{untargeted evasion}\land\text{hybrid valid}.
\]

Passing the differentiable penalty is not substituted for passing validator-v2.

---

## 12. Artifact schema

Each `cpgd_prim_support` NPZ artifact contains the common per-sample fields, including:

- `dataset`
- `victim`, `victim_arch`
- `source_class`, `source_label`
- `sample_id`, `positional_idx`
- `seed`, `method`, `objective`
- `clean_pred`, `adv_pred`
- `raw_success`
- `validator_pass`
- `valid_success`
- `perturbation_norm`, `l2_scaled`
- `attack_parameters`
- `iterations`
- `model_evaluations`
- `runtime_seconds`
- `failure_status`
- `mutable_feature_mask`
- `n_allowed_primattack_support_features`
- `n_features_modified`
- `n_modified_outside_primattack_mask`
- `cpgd_constraint_violation`

For every matched-support artifact, the runner asserts:

```text
n_modified_outside_primattack_mask == 0
```

before writing the result.

---

## 13. Tests

Focused coverage is in `tests/test_capgd_comparison.py`.

The C-PGD tests prove:

1. gradients reach the victim objective;
2. the combined C-PGD score has gradients on allowed coordinates;
3. the differentiable relation penalty receives gradients;
4. unmasked features remain exactly unchanged;
5. allowed coordinates can move;
6. L2 projection respects `epsilon`;
7. Linf projection respects `epsilon`;
8. the final output is accepted by the common evaluator interface;
9. validator and distance fields have one result per row.

The broader relevant regression selection produced:

```text
46 passed, 1 skipped
```

The only warning was a NumPy `__array_wrap__` deprecation emitted by the frozen upstream
CAPGD dependency.

---

## 14. Observed CICIDS2017 smoke run

A requested small smoke run used:

- dataset: CICIDS2017-DistriNet;
- victim: MLP;
- source class: DoS;
- 8 clean-correct samples;
- seed 42;
- 5 C-PGD iterations;
- L2 epsilon `0.5`;
- step size `0.05`;
- CPU execution.

Observed `cpgd_prim_support` output:

| Quantity | Observation |
|---|---:|
| rows | 8 |
| allowed support features | 23 |
| modified features per row | 21–23 |
| maximum modified outside support | 0 |
| raw untargeted successes | 2/8 |
| validator passes | 0/8 |
| valid successes | 0/8 |
| recorded iterations | 5 |
| recorded model evaluations | 5 |
| runtime | about 0.023 s |
| failure records | 0 |

The artifact was non-empty, all perturbation norms were finite, and the runner’s pairing
assertions passed. These values are smoke-test observations, not final attack estimates.

---

## 15. Methodological limitations

1. **Feature support is not primitive feasibility.** C-PGD has independent direct control
   over 23 coordinates and does not reproduce `g(x,z)`.
2. **The differentiable penalty is incomplete by design.** Validator-v2 contains checks
   that are non-differentiable or not represented by the eleven relation terms.
3. **Aggregate feature-space proxy only.** No PCAP is edited or replayed, and no packet-level
   realization is established.
4. **No malicious-function preservation claim.** A validator pass would establish only the
   implemented feature-domain verdict, not complete malicious semantics.
5. **Train-range box.** Bounds are train-fitted; they are not packet-protocol budgets.
6. **Penalty scaling matters.** Relation terms have different raw units. `lambda` is exposed
   and recorded; it is not claimed universally optimal.
7. **Small smoke results are not thesis results.** The eight-row run checks execution and
   invariants only.
8. **Victim and seed scope must be reported explicitly.** Attack-seed variation does not
   replace victim-training seed replication.

---

## 16. File map

| File | Responsibility |
|---|---|
| `src/comparisons/cpgd_prim_support.py` | C-PGD configuration, objective, penalty execution, random start, projection, type repair, result object |
| `src/comparisons/capgd_cicids2017.py` | train min/max resources, relation definitions, feature types, shared PrimAttack-support resource configuration |
| `src/attack/realizability/cicids2017.py` | canonical PrimAttack support mask |
| `scripts/run_full_adversarial_eval.py` | CLI, clean-correct selection, batching, execution, independent evaluation, NPZ/cell output |
| `tests/test_capgd_comparison.py` | gradient, support, projection, type, and evaluator tests |
| `docs/methods/primattack_matched_support.md` | concise combined matched-support reference |

---

## 17. Interpretation

`cpgd_prim_support` answers a narrow comparison question:

> How vulnerable is the classifier when C-PGD may directly optimize exactly the set of
> downstream coordinates that PrimAttack’s packet-size or timing primitives can affect?

It does **not** answer:

> How effective would C-PGD be under PrimAttack’s primitive-feasible transformation?

That second question requires optimizing through `g(x,z)` and is a different attack space.
