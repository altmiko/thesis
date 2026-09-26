# 2. PrimAttack: current implementation

PrimAttack is the thesis's **primitive-domain, targeted white-box attack** for
CICIDS2017-DistriNet. It does not optimize any of the 79 CICFlowMeter features
independently. It searches controls for two attacker operations:

1. add a uniform integer number of bytes `p` to every forward packet represented by
   the flow; and
2. add an integer total forward delay `delay`, with a continuous `shape` control that
   allocates that delay between proportional dilation and an equal additive shift of
   every forward inter-arrival gap.

The canonical transform

$$
x_{\mathrm{adv}}=\phi(x_0,p,\mathrm{delay},\mathrm{shape})
$$

recomputes the aggregate features declared as dependent on those operations. The
classifier only sees the transformed vector after the final controls have been
projected and the integer-valued outputs have been quantized.

This is a **feature-space proxy over aggregate flow summaries**. No packet is edited,
no PCAP is replayed, and no target service is exercised.

## 2.1 What changed

The current code replaces the earlier Adam/sigmoid search over `(p, alpha)`.

| Earlier implementation | Current implementation |
|---|---|
| timing control `alpha >= 1` uniformly multiplied every forward gap | `delay >= 0` adds an integer total delay and `shape in [0,1]` controls its allocation |
| two optimized controls `(p, alpha)` | three search controls `(p, delay, shape)` representing two physical operations |
| Adam over unconstrained sigmoid parameters | identity check, exhaustive integer-padding search, then adaptive projected sign-momentum refinement |
| targeted cross-entropy plus a primitive-cost penalty | targeted Benign logit margin; cost is used only to choose between successful candidates |
| continuous optimization followed by one final projection | every retained candidate is projected, quantized, and re-evaluated on the victim |
| only named p25/p50/p75 budget runs | named p50/p75 runs plus an evaluated envelope-only `unbounded` condition in the full paired driver |
| MLP/CNN sweep artifacts | paired MLP/CNN/FT-Transformer artifacts with one frozen clean-correct roster per victim/class |

The old and current joint/timing results differ in both optimizer and timing
parameterization. **Padding-only** comparisons isolate the optimizer change because
the padding primitive and its canonical map are shared.

## 2.2 Source map

| Concern | Current source |
|---|---|
| Dataset-agnostic primitive contracts, roles, capabilities, packet-backend interface | `src/attack/realizability/base.py` |
| CICFlowMeter primitive specifications, capabilities, bounds, projection, and canonical transform | `src/attack/realizability/cicids2017.py` (`CICIDS2017PrimitiveModel`) |
| Quantization-aware candidate search | `src/attack/primitive_optimizer.py` (`optimize_primitive_candidates`) |
| Internal transform-consistency checks | `src/attack/realizability/validator.py` (`RealizabilityValidator`) |
| Train-only budget fitting/loading | `src/attack/primattack_budget.py` |
| Flow-level semantic-preservation proxy | `src/attack/flow_semantics.py` |
| Standalone CICIDS2017 runner and detailed NPZ audit record | `src/attack/run_cicids2017_primitive_attack.py` |
| Canonical paired campaign driver | `scripts/run_full_adversarial_eval.py` |
| Same-box primitive CAPGD comparison | `src/comparisons/primitive_capgd.py` |
| Optimizer regression tests | `src/attack/tests/test_primitive_optimizer.py` |
| Transform, projection, and realizability tests | `src/attack/tests/test_primattack_transformation.py`, `src/attack/tests/test_primitive_controls.py` |

`CICIDS2017PrimitiveModel` takes feature positions from the dataset
`FeatureManifest`; it does not define a second feature order. The model can also
consume the CSE-CIC-IDS-2018 DistriNet manifest because that dataset has the same
extractor layout, but the results discussed here are the completed
CICIDS2017-DistriNet campaign.

## 2.3 Threat model and attack goal

- **Knowledge:** white-box access to the frozen category victim, preprocessing
  transform, primitive map, train-fitted calibration, and validation rules.
- **Source rows:** malicious test flows from `DoS`, `DDoS`, `Recon`, and
  `BruteForce`.
- **Eligibility:** the victim must classify the clean row as its true malicious
  category.
- **Target:** category id `0` (`Benign`).
- **Allowed operations:** forward length augmentation and forward delay only.
- **Forbidden operations:** shortening packets, accelerating traffic, adding/removing
  packets, changing backward traffic, ports, protocol, endpoints, flags, labels, or
  arbitrary aggregate features.
- **Primary success:** `adv_pred == Benign`, then success is successively gated by
  validator_v2 validity, primitive feasibility, and the semantic proxy.
- **Realism:** the per-class VAE Mahalanobis IDR gate is reported separately. It is not
  part of structural validity or primitive feasibility.

The direct attack loads the per-class Stage-A VAE only to compute the IDR mask. The
VAE is **not** in PrimAttack's generation or gradient path.

## 2.4 End-to-end data flow

The current paired driver follows this path for each `(victim, class)`:

```text
full malicious test rows
  -> victim(clean scaled row)
  -> retain clean-correct rows
  -> freeze at most N row IDs and their order
  -> infer per-row primitive capabilities once
  -> build per-row hard bounds from train-only calibration
  -> apply mode: joint | timing-only | padding-only
  -> search or random-feasible control
  -> project p and delay to the integer hard box; clamp shape
  -> phi(x0, projected controls, quantize=True)
  -> victim + validator_v2 + internal realizability + IDR + semantic proxy
  -> one per-cell NPZ containing per-row arrays and one cell summary
```

The current `scripts/run_full_adversarial_eval.py` default is a seeded uniform sample
from **all** clean-correct rows (`--selection random --selection-seed 42`). A
class-specific permutation is generated with seed `selection_seed + class_id`; each
victim keeps the first `N` rows in that permutation that it classifies correctly, then
the selected positions are sorted back into test order. `--selection head` instead
takes the first `N` clean-correct rows in test order.

The current runner records the selection method, seed, and rule in `config.json`.
`selection.json` records each victim/class roster itself: positional indices,
`sample_id`s, eligible/used counts, source-label composition, clean-validity rate, and
the SHA-256 of the ordered sample IDs. Every attack and attack seed receives that same
ordered roster within a victim/class.

The committed `outputs/full_adv_eval_primattack_v2` campaign used the legacy
**head-after-clean-correct** selection, not the current random default. Its older
`config.json` predates the explicit `selection` metadata block; the method is stated
in `FULL_ADVERSARIAL_EVALUATION_CICIDS2017_PRIMATTACK_V2.md`, while its exact rows and
hashes are in `outputs/full_adv_eval_primattack_v2/selection.json`.

The standalone runner is useful for focused experiments, but its selection procedure
is different: `_class_rows` first samples rows by true class, then `clean_correct`
defines the denominator inside that sample. Use the full paired driver for cross-method
claims.

For one PrimAttack cell:

1. `infer_capabilities(x0)` determines whether each physical operation is supported by
   the source row.
2. `per_flow_bounds(x0, class_config, capabilities)` intersects class budget,
   train-p99 envelope headroom, capability gates, and any DoS/DDoS rate floor.
3. `_apply_primitive_mode` zeroes the irrelevant **upper bounds** for timing-only or
   padding-only ablations.
4. `optimize_primitive_candidates` searches the realized integer attack, or
   `random_feasible_primitives` supplies the control condition.
5. `project_controls` produces legal controls.
6. `generate(..., quantize=True)` produces the final raw 79-vector.
7. The raw vector is transformed by the training-fitted scaler and passed to the
   victim.
8. Structural validity, internal consistency, IDR, primitive feasibility, and semantic
   status are computed independently.

## 2.5 Primitive contracts

### 2.5.1 Padding `p`

`p` is uniform forward packet-length augmentation.

| Property | Contract |
|---|---|
| Units | bytes per forward packet |
| Type | discrete integer after projection |
| Identity | `0` |
| Direction | increase only |
| Absolute range | `[0, +inf)` before per-flow bounds |
| Capability | at least one forward packet, positive total forward length, and positive forward mean length |
| Disabled reasons | `NO_FORWARD_PAYLOAD` or `INSUFFICIENT_FWD_PACKETS` |
| Projection | `min(round(max(p,0)), floor(p_hi))`, then capability mask |

The capability check is:

$$
m_p =
[N_f \ge 1]
\land [L_f > 0]
\land [\bar l_f > 0].
$$

If it fails, `p_hi=0` before the optimizer sees the row. A zero-payload SYN-like flow
is therefore not treated as paddable merely because a numeric envelope has headroom.

The numeric upper bound is:

$$
\begin{aligned}
p_{\mathrm{hi}}=\min\{&
E_{\mathrm{fwd,max}}-\mathrm{fwdmax}_0,\;
E_{\mathrm{fwd,min}}-\mathrm{fwdmin}_0,\\
&E_{\mathrm{fwd,mean}}-\mathrm{fwdmean}_0,\;
(E_{\mathrm{TL,fwd}}-L_{f,0})/\max(N_f,1),\;
p_{\max}\},
\end{aligned}
$$

clamped below at zero and multiplied by the capability mask. Every `E` value is a
global-training p99 upper envelope; `p_max` comes from the selected class budget.

For projected `p`, the canonical map computes:

```text
Total Length of Fwd Packet = TL_fwd0 + N_f * p
Fwd Packet Length Min      = min0 + p
Fwd Packet Length Max      = max0 + p
Fwd Packet Length Mean     = TL_fwd / max(N_f, 1)
Fwd Segment Size Avg       = Fwd Packet Length Mean
Fwd Packet Length Std      = unchanged
```

A uniform shift preserves the forward-length standard deviation. Combined packet
length min/max branch on whether each direction is present. Combined mean uses total
forward plus backward bytes. Combined variance uses the exact pooled sample-variance
decomposition from forward/backward counts, means, and standard deviations. Packet
Length Std is the square root of that variance. `Flow Bytes/s` is then recomputed from
the updated byte total and projected duration.

### 2.5.2 Total forward delay `delay`

`delay` is the total number of microseconds added across the forward IAT sequence.

| Property | Contract |
|---|---|
| Units | total microseconds of additional forward delay |
| Type | discrete integer after projection |
| Identity | `0` |
| Direction | increase only |
| Capability | at least two forward packets and positive `Fwd IAT Total` |
| Disabled reasons | `SINGLE_FWD_PACKET` or `ZERO_TIMING_HEADROOM` |
| Projection | `min(round(max(delay,0)), floor(delay_hi))`, then capability mask |

The timing capability is:

$$
m_t=[N_f\ge2]\land[T_f>0].
$$

If it fails, both `delay_hi` and `shape_hi` become zero.

### 2.5.3 Delay allocation `shape`

`shape` does not create more delay. It allocates the selected total:

- `shape=0`: proportional dilation of existing forward gaps;
- `shape=1`: equal additive delay for every forward gap;
- `0<shape<1`: affine mixture of both endpoints.

For $m=\max(N_f-1,1)$ gaps with original total $T_f$:

$$
a=1+(1-\mathrm{shape})\frac{\mathrm{delay}}{T_f},
\qquad
b=\mathrm{shape}\frac{\mathrm{delay}}{m},
$$

$$
g_i'=a g_i+b.
$$

Therefore:

$$
\sum_i g_i'=T_f+\mathrm{delay}.
$$

Both endpoints add delay only and preserve gap order because `a >= 1` and `b >= 0`.
`shape` is clamped to `[0,1]`, capped by `shape_hi`, and forced to zero whenever the
projected delay is zero.

The timing recomputation is:

```text
Fwd IAT Total = fit0 + delay
Fwd IAT Max   = a * fimax0 + b
Fwd IAT Min   = a * fimin0 + b
Fwd IAT Std   = a * fistd0
Fwd IAT Mean  = Fwd IAT Total / max(N_f - 1, 1)

Flow Duration = max(duration0 + delay,
                    Fwd IAT Total,
                    Bwd IAT Total,
                    1 microsecond)
Flow IAT Mean = Flow Duration / max(N_f + N_b - 1, 1)
Flow IAT Max  = Flow IAT Max0 + max(Flow Duration - duration0, 0)

Fwd Packets/s  = N_f / duration_seconds
Bwd Packets/s  = N_b / duration_seconds
Flow Packets/s = (N_f + N_b) / duration_seconds
Flow Bytes/s   = (TL_fwd + TL_bwd) / duration_seconds
```

`Flow Duration` and `Flow IAT Max` are conservative aggregate projections. The merged
forward/backward packet order is absent, so they are not claimed as exact packet-trace
re-extractions.

## 2.6 Hard per-flow timing bounds

`delay_hi` is the minimum of:

1. named relative-duration budget
   `max_relative_duration_change * max(Flow Duration, 1us)`;
2. global train-p99 headroom for `Fwd IAT Total`;
3. `gaps *` global train-p99 headroom for `Fwd IAT Mean`;
4. global train-p99 headroom for `Flow Duration`;
5. `Fwd IAT Max` headroom divided by the worst allocation coefficient
   `max(Fwd IAT Max / Fwd IAT Total, 1/gaps)`;
6. `Fwd IAT Std` headroom divided by its proportional-allocation coefficient; and
7. for DoS/DDoS, the delay allowed before `Flow Packets/s` falls below the
   class-training p05.

The max/std terms use the worst coefficient over every `shape in [0,1]`. Consequently,
the returned `delay x shape` rectangle is intended to be feasible without a soft
constraint penalty. `Fwd IAT Min` needs no separate upper cap because it cannot exceed
the bounded maximum under the affine order-preserving map.

`per_flow_bounds` returns semantic bounds (`p`, `delay`, `shape`) and pre-capability
numeric bounds (`p_numeric`, `delay_numeric`, `shape_numeric`) for auditability.

## 2.7 Canonical feature map and role system

`generate` starts with `raw.clone()` and writes only declared dependencies on rows where
the relevant projected control is non-identity. If both `p=0` and `delay=0`, it returns
the source row exactly.

| Role | Meaning | Examples |
|---|---|---|
| `DERIVED_P` (`Dp`) | exact padding-derived value | forward and combined length statistics |
| `DERIVED_T` (`Dt`) | exact affine-timing-derived value | forward IAT summaries |
| `DERIVED` (`D`) | other exact algebraic derivation | reserved role in the generic contract |
| `CONDITIONAL` (`C`) | branch-dependent or conservative reconstruction | combined min/max, duration, Flow IAT Max |
| `RATE` (`R`) | count/bytes over projected duration | bytes/s and packet rates |
| `INVARIANT` (`I`) | mathematically unchanged | Fwd Packet Length Std under uniform padding |
| `FROZEN` (`F`) | genuinely unaffected by allowed operations | ports, protocol, counts, flags, backward-only fields |
| `LEVEL_C` (`F^C`) | would require packet sequence/re-extraction; held constant, not claimed invariant | subflow, bulk, active/idle, merged Flow IAT Std/Min |

The explicit `LEVEL_C` set includes:

- `Fwd Act Data Pkts`;
- `Subflow Fwd Bytes`;
- forward bulk byte/packet/rate averages;
- `Flow IAT Std` and `Flow IAT Min`; and
- all Active/Idle mean/std/max/min fields.

These held values are the main boundary between **aggregate algebraic consistency** and
packet-level realizability. `NullPacketBackend.available()` is false, so Level-C
verification is unavailable.

When `quantize=True`, the transform rounds projected `p` and `delay` and the
integer-valued fields it writes: forward/combined length extrema and totals, forward IAT
total/extrema, flow duration, and flow IAT maximum. The internal validator independently
checks discreteness on the complete data-mined integer feature set.

## 2.8 Current optimizer

`optimize_primitive_candidates` (the **Hybrid Search**) searches the **realized attack**. Its
differentiable relaxation supplies gradients, but candidate selection always uses
`project_controls` followed by `generate(..., quantize=True)`. All PrimAttack optimizers
(Hybrid, and the ablation baselines `optimize_primitive_pgd` / `optimize_primitive_cw`) run
on one `RealizedSearch` object that owns projection, quantized realization, victim scoring,
the success predicate, the per-flow incumbent, and per-flow evaluation counting.

### Targeted objective

For target class `0`:

$$
m(x)=\max_{k\ne0} z_k(x)-z_0(x).
$$

Lower is better. Actual success is tested on the realized flow as
`argmax(logits) == 0` **and** the injected validity gate (every runner passes
`hybrid_valid_gate(dataset)`, i.e. validator_v2 `hybrid_valid`) — the same predicate as the
reported valid ASR. Before this fix, selection used `argmax == 0` alone, so a
validator-invalid padding "success" could stop the padding sweep and block timing
refinement (material on CICIDS2018, where padding breaks the mined
`Fwd Packet Length Min == Packet Length Min` rule).

### Untargeted objective

`AttackObjective("untargeted", y)` (source class `y`) replaces the margin with

$$
m(x)=z_y(x)-\max_{k\ne y} z_k(x)
$$

and the realized-flow success test with `argmax(logits) != y` **and** the same validity gate.
Everything else (projection, quantized realization, incumbent ordering, evaluation budget) is
unchanged; all three optimizers accept `objective=`. The default objective is
`AttackObjective("targeted", 0)` (Benign). `scripts/run_primattack_optimizer_ablation.py
--objective untargeted` runs it (final suite Exp A/D).

There is no cost term in the gradient objective. Hard bounds define feasibility.
Per-row candidate selection is lexicographic:

1. a (valid, targeted) success replaces any failure;
2. among successes, minimize
   `p/p_hi + delay/delay_hi`;
3. if success costs tie, use lower margin;
4. among failures, use lower margin.

`shape` has no direct cost because it reallocates a fixed total delay.

### Stage 1: identity

The exact no-op candidate `(0,0,0)` is projected, generated, quantized, and scored. It
is the initial best candidate.

### Stage 2: exhaustive integer padding

For each row, the search evaluates:

$$
p=1,2,\ldots,\lfloor p_{\mathrm{hi}}\rfloor,
\qquad \mathrm{delay}=0,\quad \mathrm{shape}=0.
$$

Values are evaluated in increasing cost order. For `m` currently unresolved rows, the
implementation batches `k=max(1,min(values_remaining,4096//m))` consecutive padding
values into one victim call. In the completed campaign (`m <= 800`) this caps each
call at 4096 candidate rows. A row leaves the sweep after its first (valid, targeted)
success or after exhausting its cap; the batched values after a row's first success are
discarded and not charged, so the per-flow evaluation count equals sequential scoring.
This yields the minimum-padding successful padding-only
attack wherever one exists; otherwise it retains the padding value with the lowest
targeted margin.

In timing-only mode `p_hi=0`, so this stage is only the identity.

### Stage 3: adaptive projected refinement

Only rows still unresolved after exhaustive padding and with
`delay_hi >= 1 microsecond` enter this stage. Controls are normalized:

$$
q\in[0,1]^3,\qquad
(p,\mathrm{delay},\mathrm{shape})
=(p_{\mathrm{hi}}q_p,\mathrm{delay}_{\mathrm{hi}}q_d,
\mathrm{shape}_{\mathrm{hi}}q_s).
$$

At each iteration:

```text
relaxed = phi(x0, controls(max(q, 1e-3)), quantize=False)   # straight-through floor
g       = gradient_q sum(targeted_margin(victim(scale(relaxed))))
v       = 0.75 * v + g / max(mean(abs(g)), 1e-12)
q       = clamp(q - step_size * sign(v), 0, 1)

realized_controls = project_controls(controls(q))
realized_candidate = phi(x0, realized_controls, quantize=True)
score and commit the realized candidate
```

Every `max(5, steps // 4)` iterations, rows whose best realized margin has stalled
halve their step size, return to their restart-best `q`, and clear momentum.

- Restart 0 begins from the best identity/exact-padding control (`adaptive-clean`).
- Later restarts begin from seeded uniform normalized controls
  (`adaptive-random`).
- Controls whose integer headroom is absent are pinned to zero.
- Successful rows are not removed from adaptive iterations; the success-first,
  lowest-cost selector still decides what survives.

The canonical map copies a row verbatim where `p == 0` / `delay == 0`, so the relaxation's
gradient is exactly zero at a zero control. Surrogate evaluations therefore use
`max(q, SURROGATE_FLOOR=1e-3)` on coordinates with headroom, with the gradient passed
straight through to `q`; realized scoring never sees the floor. Without it, restart 0
(which starts at `delay = 0`) could never move timing.

Defaults are `steps=40`, `learning_rate=0.1`, and `restarts=2`; `restarts=None` with an
`eval_budget` keeps starting random restarts until each refined row has spent the per-flow
budget (used by the optimizer ablation). The search records the winning source (`identity`,
`exact-padding`, `adaptive-clean`, or `adaptive-random`), realized target margin,
normalized cost, and per-flow counts of realized forward, surrogate forward, and backward
victim evaluations plus the evaluation index of the first success. The seed only affects
random restarts; identity, exact padding, and the clean restart are deterministic given the
runtime/model.

The controlled optimizer ablation (Hybrid vs Prim-PGD vs Prim-C&W at a matched per-flow
evaluation budget) is documented in the root report `primattack_optimizer_ablation.md`.

### Random-feasible control

`random_feasible_primitives` draws three independent continuous `U(0,1)` fractions,
multiplies them by `p_hi`, `delay_hi`, and `shape_hi`, then applies the same projection
and quantized transform. Because `p` and `delay` are rounded afterward, this is uniform
in the continuous pre-projection box, not exactly uniform over discrete integer values.
It tests whether directed search improves over one random feasible candidate.

## 2.9 Calibration and evaluated boxes

`primattack_budget.calibrate` reads only `X_train_pristine.npy` and
`y_train_cat.npy`. The frozen artifact records `fit_split="train"` and explicitly
prohibits validation/test features, victim predictions, and adversarial success from
budget selection.

For each attack class:

- padding reference population: positive class-conditional
  `Fwd Packet Length Mean` on rows that pass the padding evidence check;
- timing reference population:
  `abs(Flow Duration - class_median) / class_median`;
- `restricted`, `intermediate`, `maximum-evaluated`: empirical p25, p50, p75;
- padding quantiles are rounded to the nearest legal byte;
- a shared complete-training p99 feature envelope supplies physical/plausibility
  headroom;
- DoS/DDoS receive an additional class-training p05 `Flow Packets/s` floor.

Frozen values in `artifacts/primattack/budget_calibration.json`:

| Class (train n) | Restricted p25 (`p / relative duration`) | Intermediate p50 | Maximum-evaluated p75 | Rate floor required |
|---|---:|---:|---:|---|
| DoS (120,093) | `41 / 0.0427` | `47 / 0.5671` | `54 / 1.2682` | yes |
| DDoS (66,568) | `2 / 0.2220` | `2 / 0.4353` | `3 / 0.6874` | yes |
| Recon (111,311) | `2 / 0.0851` | `2 / 0.2128` | `10 / 0.5319` | no |
| BruteForce (4,862) | `11 / 0.0617` | `12 / 0.1199` | `91 / 0.2337` | no |

These are empirical evaluation budgets, not MTU limits or universal physical maxima.

### Named budgets versus `unbounded`

The standalone runner accepts only the three named calibrated budgets. The full paired
driver additionally calls `unbounded_calibration`:

```text
p_max = +inf
max_relative_duration_change = +inf
```

This removes the class p25/p50/p75 caps but retains:

- the same train-p99 feature envelope;
- per-flow capability gates;
- exact integer projection;
- and the DoS/DDoS minimum-rate floor.

`unbounded` therefore means **envelope-only**, not unconstrained feature-space attack.
It is an evaluated condition in `outputs/full_adv_eval_primattack_v2`, not a
hypothetical one. It must not be described as compliant with a named empirical budget.
The fully unconstrained comparison is input-space PGD/C&W, which does not use the
primitive map.

## 2.10 Validation, feasibility, semantics, and metrics

The gates are deliberately separate.

### Structural validity

`validation.attack_interface.structural_masks` evaluates validator_v2:

$$
\mathrm{hybrid\_valid}
=\mathrm{SCHEMA}\land\mathrm{EXTRACTOR}\land\mathrm{PROTOCOL}\land\mathrm{MINED}.
$$

The runner stores this as `domain_valid`. PrimAttack is designed to preserve many of
these identities, but external validator acceptance is still measured rather than
assumed.

### Internal primitive consistency

`RealizabilityValidator` independently checks:

- declared algebraic dependencies;
- packet-summary ordering/non-negativity;
- timing ordering and totals;
- non-negative rates;
- integrality of integer-valued features; and
- exact preservation of `FROZEN`, `INVARIANT`, and held `LEVEL_C` columns.

This produces `primitive_transform_consistent`. It is not packet-level verification.

### Primitive feasibility

`FlowSemanticValidator` checks projected controls against per-row bounds, integer
requirements, shape bounds, represented-byte monotonicity, and the selected relative
duration budget. The reported primitive-feasible mask is:

$$
\mathrm{primitive\_feasible}
=\mathrm{budget\_compliance}
\land\mathrm{primitive\_transform\_consistent}.
$$

For `unbounded`, the relative-duration budget is infinite, but envelope and capability
bounds still apply.

### Flow-level semantic proxy

Generic required checks preserve:

- attack-label metadata;
- protocol, source/destination ports, endpoints, and direction;
- packet counts and TCP flag aggregates;
- finite generated values;
- non-decreasing represented traffic volume; and
- the rule that only declared primitive dependencies may change.

Class-specific checks are:

- **DoS/DDoS:** adversarial `Flow Packets/s` must remain at or above the
  class-training p05 absolute floor.
- **Recon:** complete scanned-port set, scan order, and distinct attempt sequence are
  unavailable from one aggregate row.
- **BruteForce:** authentication attempts, credentials/payload semantics, and service
  outcome are unavailable.

The result is `PASS`, `FAIL`, or `NOT_FULLY_TESTABLE`. Recon/BruteForce rows remain in
the denominator; they are never silently dropped to inflate SP-ASR.

### Nested rates

On one frozen clean-correct denominator:

$$
\mathrm{ASR}_{raw}
\supseteq
\mathrm{ASR}_{hybrid\ valid}
\supseteq
\mathrm{ASR}_{primitive\ feasible}
\supseteq
\mathrm{SP\mbox{-}ASR}.
$$

For targeted PrimAttack:

```text
raw              = targeted_success
valid            = targeted_success & domain_valid
primitive        = targeted_success & domain_valid & primitive_feasible
SP-ASR           = targeted_success & domain_valid & primitive_feasible & semantic_PASS
True-IDSR        = targeted_success & domain_valid & in_distribution
```

IDR is intentionally outside structural validity and outside SP-ASR.

## 2.11 Runners and artifacts

### Focused standalone run

```powershell
$Env:PYTHONPATH = "src"
python -m attack.run_cicids2017_primitive_attack `
  --classes DoS,DDoS,Recon,BruteForce `
  --victims mlp,cnn,ft_transformer `
  --budget maximum-evaluated `
  --primitive-mode joint `
  --optimizer search `
  --steps 40 `
  --learning-rate 0.1 `
  --restarts 2 `
  --seeds 42,123,2024 `
  --output-dir outputs/primattack_calibrated
```

The output directory is required to be fresh. It contains:

- `run_manifest.json`: source/config/checkpoint/scaler/manifest provenance;
- `attack_results.json`: per-cell rates and cost summaries;
- `attack_artifacts/<class>_<victim>_seed<seed>.npz`: detailed per-row audit.

The standalone NPZ includes clean/adversarial raw vectors, scaled adversarial vectors,
sample IDs, predictions/logits, requested and projected controls, bounds, capability
reasons, candidate source/margin, validity and feasibility masks/reasons, semantic
status/reasons, IDR, costs, changed features, and provenance hashes.

### Canonical paired campaign

To run only the current PrimAttack family with the current driver:

```powershell
$Env:PYTHONPATH = "src"
python scripts/run_full_adversarial_eval.py `
  --dataset cicids2017 `
  --families primattack `
  --budgets intermediate,maximum-evaluated,unbounded `
  --modes joint,timing-only,padding-only `
  --optimizers search,random-feasible `
  --selection random `
  --selection-seed 42 `
  --seeds 42,123,2024
```

The full driver writes `config.json`, `selection.json`, `cells.json`,
`failures.json`, and one artifact named
`<victim>__<class>__<attack>__seed<seed>.npz`. `assert_pairing` reloads artifacts
and fails if sample IDs, row order, or clean-correct eligibility differ within a
victim/class.

The full driver can also run `capgd_prim_p75`, an untargeted CAPGD optimizer over the
same normalized `(p, delay, shape)` p75 joint box. The canonical transform, projection,
quantization, and final validators remain unchanged, so this is an optimizer-control
comparison rather than native feature-space CAPGD.

## 2.12 Current measured behavior

The committed v2 campaign is
`outputs/full_adv_eval_primattack_v2`:

- three victims: MLP, CNN, FT-Transformer;
- four attack classes;
- 800 frozen clean-correct rows per victim/class;
- legacy head selection: first 800 clean-correct rows in test order per victim/class;
- attack seeds `42, 123, 2024`;
- p50, p75, and envelope-only boxes;
- joint, timing-only, and padding-only modes;
- search and random-feasible control;
- 648 PrimAttack cells and no recorded failures.

In those artifacts, domain validity and primitive feasibility are both 100% in every
PrimAttack cell. This is an **observed result for this campaign**, not a universal
guarantee of the transform.

Reference-seed (`42`) class-pooled results, `N=3,200` per victim:

| Victim | Condition | Valid targeted ASR | SP-ASR | True-IDSR |
|---|---|---:|---:|---:|
| MLP | search joint p50 | 4.19% | 4.09% | 0.00% |
| MLP | search joint p75 | 6.28% | 6.19% | 0.00% |
| MLP | search joint envelope-only | 23.22% | 18.38% | 0.00% |
| CNN | search joint p50 | 12.31% | 7.56% | 0.03% |
| CNN | search joint p75 | 34.91% | 10.97% | 0.06% |
| CNN | search joint envelope-only | 57.66% | 24.78% | 0.09% |
| FT-Transformer | search joint p50 | 5.03% | 0.22% | 0.00% |
| FT-Transformer | search joint p75 | 5.19% | 0.22% | 0.00% |
| FT-Transformer | search joint envelope-only | 5.44% | 0.22% | 0.00% |

At p75, the random-feasible valid targeted rates were 1.50% (MLP), 5.56% (CNN),
and 3.69% (FT-Transformer), versus 6.28%, 34.91%, and 5.19% for search.

The replaced `(p, alpha)` Adam run in `outputs/full_adv_eval` produced p75 joint valid
targeted rates of 0.16%, 1.19%, and 4.66% at the same reference seed. The paired report
is `PRIMATTACK_V2_OPTIMIZER_COMPARISON.md`. Interpret joint/timing differences as a
combined optimizer-and-timing-model change; use padding-only rows for the clean
optimizer comparison.

The substantive result is not “PrimAttack always works.” Valid evasion is strongly
victim- and class-dependent, and almost every successful flow is outside the
val-anchored per-class VAE IDR gate. PrimAttack v2 is therefore evidence for
**validity-constrained robustness evaluation**, not an in-distribution evasion claim.

## 2.13 Verified invariants

The dedicated tests cover:

- exact no-op identity, including quantized generation;
- completeness of each primitive's declared dependency set;
- affine timing endpoints and equations;
- padding and pooled-length equations;
- hard integer projection and capability masking;
- finite-difference agreement with autograd through the scaler and real victim;
- non-finite input/control rejection and tiny-duration boundaries;
- internal algebraic, packet-summary, timing, rate, discreteness, and frozen checks;
- machine-readable capability reasons;
- result regeneration from returned requested/projected controls;
- hard-box compliance; and
- optimizer dominance over identity and every exhaustively enumerated padding
  candidate.

The budget tests reproduce the frozen train-only artifact and verify monotone p25/p50/p75
levels and prohibited-input provenance.

## 2.14 Assumptions, limitations, and defensible claims

### Assumptions

1. Forward uniform padding and forward delay allocation are meaningful attacker
   operations for the evaluated flows.
2. Positive forward payload and a non-zero forward-IAT sequence are sufficient evidence
   to enable the corresponding aggregate primitive.
3. The conservative duration/Flow-IAT-Max construction is acceptable for a flow-level
   proxy.
4. Training p99 envelopes and class quantiles are defensible evaluation bounds for this
   dataset, not physical constants.

### Limitations

- No packet trace is edited or re-extracted; `NullPacketBackend` declares packet-level
  verification unavailable.
- Level-C sequence-dependent fields are held constant because they cannot be recovered
  from one aggregate row.
- Complete malicious functionality is unobservable, especially for Recon and
  BruteForce.
- Only forward padding and delay are modeled; there is no packet injection/splitting,
  payload rewrite, backward edit, flag edit, or target-state feedback.
- The p25/p50/p75 budgets are empirical dataset envelopes. `unbounded` removes only
  those class caps and remains constrained by train-p99/capability/rate rules.
- Results are feature-space, closed-set, and victim-specific. Victims must not be pooled
  as independent replicates.
- The committed campaign uses one frozen row roster that was inspected diagnostically
  before the v2 design; optimizer comparisons are post-hoc rather than held-out
  confirmation.
- Seed variation changes random restarts/control samples, not victim training. It does
  not establish training-seed robustness for CICIDS2017 victims.

### Can claim

- PrimAttack searches interpretable primitive controls rather than arbitrary features.
- Final candidates obey the declared integer hard box and are re-evaluated after
  quantization.
- The canonical map preserves its declared aggregate dependencies and frozen columns
  under the tested contract.
- The completed v2 campaign observed 100% validator_v2 and internal primitive-feasibility
  acceptance for PrimAttack outputs.
- Valid targeted evasion is measurable but highly victim-dependent.

### Must not claim

- packet-level realizability or PCAP validity;
- complete CICFlowMeter re-extraction equivalence;
- preserved malicious functionality or deployment behavior;
- in-distribution evasion when True-IDSR is approximately zero;
- an unconstrained primitive attack: even `unbounded` retains the train-p99 envelope,
  capability gates, integer projection, and DoS/DDoS rate floor; or
- universal robustness/generalization from one dataset split, one frozen source roster,
  or pooled victims.
