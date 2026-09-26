# PrimAttack Feature Support: Derivation, Canonical Mask, and Matched-Support Attacks

## 1. Purpose

This document explains how the PrimAttack downstream feature-support mask was derived and
implemented. It answers four questions:

1. Which of the 79 classifier inputs can change after PrimAttack applies its packet-size
   and timing primitives?
2. How was that set obtained from the executable PrimAttack path rather than guessed from
   feature names?
3. How is the set represented as a canonical, schema-checked mask?
4. How do `capgd_prim_support` and `cpgd_prim_support` consume the mask without being
   misrepresented as primitive-feasible attacks?

The main implementation is in:

- `src/attack/realizability/cicids2017.py`
- `src/comparisons/capgd_cicids2017.py`
- `src/comparisons/cpgd_prim_support.py`
- `scripts/run_full_adversarial_eval.py`

The canonical public mask is:

```python
primattack_joint_feature_mask(manifest)
```

It contains **23 of the 79 classifier features**.

---

## 2. Definition of feature support

For a classifier feature `j`, the joint mask is true when an allowed PrimAttack primitive
can cause the canonical transform to write a value different from the clean value:

\[
S_j=1
\iff
\exists z\in\mathcal Z(x): g_j(x,z)\ne x_j.
\]

Here:

- `x` is a clean pristine/raw flow vector;
- `z = (p, delay, shape)` is a projected primitive-control vector;
- `Z(x)` is the per-flow primitive domain after capabilities and budgets;
- `g` is `CICIDS2017PrimitiveModel.generate`;
- `g_j` is output coordinate `j`.

This is a **potential downstream write-support** definition. It does not mean every feature
moves on every attacked row. For example:

- `Packet Length Max` changes only when the changed forward maximum determines the combined
  maximum;
- `Bwd Packets/s` changes under timing only when the flow has backward packets;
- all timing outputs remain unchanged when the source flow has no timing capability;
- all padding outputs remain unchanged when the source flow has no supported forward
  payload.

A false mask entry means the canonical PrimAttack transform does not write that coordinate.

---

## 3. Why feature names were insufficient

The mask was not inferred from names containing words such as `Fwd`, `Packet`, `IAT`, or
`Duration`. Such inference would be wrong for several reasons:

- `Fwd Packet Length Std` is invariant under adding the same padding amount to every
  forward packet and is therefore not written.
- `Flow IAT Std` could change in a real packet trace, but the aggregate input does not
  contain enough information to reconstruct the merged forward/backward gap sequence. The
  canonical transform deliberately holds it constant.
- `Flow Bytes/s` changes under both padding and timing even though neither primitive is
  named in the feature.
- `Bwd Packets/s` changes under a forward timing delay because its denominator is the
  projected whole-flow duration.
- active/idle and bulk statistics could be affected at packet level but cannot be uniquely
  recomputed from the available aggregates. They are Level-C held-constant features, not
  supported outputs of the canonical map.

The only defensible source is the executed primitive-to-feature transformation.

---

## 4. Executable path that was traced

The derivation followed the full PrimAttack path:

```text
primitive variables
    -> semantic capability gates
    -> train-calibrated per-flow budgets
    -> hard primitive projection
    -> integer rounding/discretization
    -> CICIDS2017PrimitiveModel.generate
    -> combined-statistic and rate propagation
    -> RobustScaler classifier transform
    -> victim model
```

The support set is determined at the `generate` stage. Earlier stages determine whether a
primitive may be nonzero for a particular row; later stages preserve the same feature
positions while scaling values for the classifier.

---

## 5. Primitive controls

PrimAttack exposes three controls representing two attacker operations.

### 5.1 Uniform forward packet padding `p`

`p` is an increase-only number of bytes added to every forward packet:

\[
p\ge 0.
\]

After projection it is an integer number of bytes. It is active only when the source flow
has at least one forward packet and positive forward payload evidence.

### 5.2 Total added forward delay `delay`

`delay` is increase-only total forward inter-arrival delay in microseconds:

\[
delay\ge 0.
\]

After projection it is an integer number of microseconds. It is active only when the flow
has at least two forward packets and a positive forward-IAT total.

### 5.3 Delay allocation `shape`

`shape` determines how total delay is distributed between proportional dilation and equal
per-gap offset:

\[
shape\in[0,1].
\]

For `m = N_f - 1` forward gaps and original total `T`:

\[
scale = 1 + (1-shape)\frac{delay}{T},
\]

\[
offset = shape\frac{delay}{m},
\]

\[
iat'_i = scale\cdot iat_i+offset.
\]

The total becomes exactly `T + delay`. `shape` is zeroed whenever delay is inactive.

---

## 6. Capability gates and primitive budgets

### 6.1 Padding capability

`infer_capabilities` permits padding only when:

- `Total Fwd Packet >= 1`; and
- `Total Length of Fwd Packet > 0`; and
- `Fwd Packet Length Mean > 0`.

Otherwise the projected padding value is exactly zero.

### 6.2 Timing capability

Timing is permitted only when:

- `Total Fwd Packet >= 2`; and
- `Fwd IAT Total > 0`.

Otherwise both `delay` and `shape` are zero.

### 6.3 Per-flow bounds

`per_flow_bounds` intersects the configured primitive budget with source-flow headroom.
Padding headroom considers forward total, min, max, and mean envelopes. Timing headroom
considers:

- maximum relative duration change;
- forward IAT total and mean envelopes;
- projected flow duration;
- worst-case forward IAT max and standard-deviation effects over all shape values;
- optional minimum flow-rate semantics.

The mask does not encode budget magnitudes. It encodes only which coordinates the
canonical map may write when a primitive is active.

---

## 7. Projection and discretization

`project_controls` performs the final primitive projection before realized generation.

For padding:

\[
p' = \min(\operatorname{round}(\max(p,0)),\lfloor p_{hi}\rfloor).
\]

For timing:

\[
delay' = \min(\operatorname{round}(\max(delay,0)),\lfloor delay_{hi}\rfloor).
\]

For shape:

\[
shape'=\min(\max(shape,0),shape_{hi}),
\]

then `shape' = 0` when `delay' = 0`.

Capability masks are reapplied during projection and generation. The realized classifier
sample therefore uses integer padding and integer total delay.

---

## 8. Padding recomputation path

Let:

- `Nf` be `Total Fwd Packet`;
- `Nb` be `Total Bwd packets`;
- `TLf` and `TLb` be total forward and backward lengths;
- `p` be projected padding bytes per forward packet.

### 8.1 Forward-direction packet statistics

The canonical transform computes:

\[
TLf' = TLf + N_f p,
\]

\[
FwdMin' = FwdMin+p,
\qquad
FwdMax' = FwdMax+p,
\]

\[
FwdMean' = \frac{TLf'}{\max(N_f,1)},
\]

\[
FwdSegmentAvg' = FwdMean'.
\]

`Fwd Packet Length Std` is deliberately unchanged because a uniform shift preserves the
within-forward standard deviation.

### 8.2 Combined packet-length extrema

Combined minimum and maximum are recomputed with direction-presence branches:

\[
PacketMax' = \max(FwdMax',BwdMax)
\]

when both directions exist, with the existing direction selected when only one direction
exists. `PacketMin'` uses the analogous minimum branch.

These features are in support even though a particular row may retain the same combined
extremum when the backward direction remains dominant.

### 8.3 Combined mean and variance

The combined mean is:

\[
PacketMean' = \frac{TLf'+TLb}{\max(N_f+N_b,1)}.
\]

`Average Packet Size` is set to the same value.

The transform recomputes pooled sample variance from:

- forward count, changed forward mean, unchanged forward standard deviation;
- backward count, mean, and standard deviation;
- the between-group sum of squares.

Then:

\[
PacketStd'=\sqrt{PacketVariance'}.
\]

### 8.4 Flow byte rate

Padding changes the byte numerator of:

\[
FlowBytesPerSecond'
=
\frac{TLf'+TLb}{Duration'/10^6}.
\]

With padding only, duration is unchanged. With joint padding and timing, both numerator
and denominator may change.

---

## 9. Timing recomputation path

Let:

- `FIT` be `Fwd IAT Total`;
- `m = max(Nf - 1, 1)`;
- `scale` and `offset` be the affine delay-allocation terms.

The transform computes:

\[
FIT' = FIT + delay,
\]

\[
FwdIATMax' = scale\cdot FwdIATMax+offset,
\]

\[
FwdIATMin' = scale\cdot FwdIATMin+offset,
\]

\[
FwdIATStd' = scale\cdot FwdIATStd,
\]

\[
FwdIATMean' = \frac{FIT'}{m}.
\]

The projected duration is conservatively bounded below by the changed forward IAT total,
backward IAT total, and one-microsecond duration floor:

\[
Duration' = \max(Duration+delay,FIT',BwdIATTotal,1).
\]

Then:

\[
FlowIATMean' =
\frac{Duration'}{\max(N_f+N_b-1,1)}.
\]

`Flow IAT Max` is increased by the non-negative projected duration change.

### 9.1 Rate propagation

Timing changes the shared duration denominator used by:

\[
FwdPacketsPerSecond' = \frac{N_f}{Duration'/10^6},
\]

\[
BwdPacketsPerSecond' = \frac{N_b}{Duration'/10^6},
\]

\[
FlowPacketsPerSecond' = \frac{N_f+N_b}{Duration'/10^6},
\]

\[
FlowBytesPerSecond' = \frac{TLf'+TLb}{Duration'/10^6}.
\]

This is why backward and whole-flow rates occur in the timing mask even though only
forward delays are controlled.

---

## 10. Exact canonical feature table

Indices are resolved against
`FeatureManifest`/`preprocessing_manifest.json:modelling_feature_names`.
`index0` is the zero-based NumPy/PyTorch index; `index1` is the one-based thesis position.

| index0 | index1 | Classifier feature | Padding | Timing | Recompute block |
|---:|---:|---|:---:|:---:|---|
| 3 | 4 | `Flow Duration` |  | yes | affine forward-delay allocation |
| 6 | 7 | `Total Length of Fwd Packet` | yes |  | forward packet-length augmentation |
| 8 | 9 | `Fwd Packet Length Max` | yes |  | forward packet-length augmentation |
| 9 | 10 | `Fwd Packet Length Min` | yes |  | forward packet-length augmentation |
| 10 | 11 | `Fwd Packet Length Mean` | yes |  | forward packet-length augmentation |
| 16 | 17 | `Flow Bytes/s` | yes | yes | rates |
| 17 | 18 | `Flow Packets/s` |  | yes | rates |
| 18 | 19 | `Flow IAT Mean` |  | yes | affine forward-delay allocation |
| 20 | 21 | `Flow IAT Max` |  | yes | affine forward-delay allocation |
| 22 | 23 | `Fwd IAT Total` |  | yes | affine forward-delay allocation |
| 23 | 24 | `Fwd IAT Mean` |  | yes | affine forward-delay allocation |
| 24 | 25 | `Fwd IAT Std` |  | yes | affine forward-delay allocation |
| 25 | 26 | `Fwd IAT Max` |  | yes | affine forward-delay allocation |
| 26 | 27 | `Fwd IAT Min` |  | yes | affine forward-delay allocation |
| 38 | 39 | `Fwd Packets/s` |  | yes | rates |
| 39 | 40 | `Bwd Packets/s` |  | yes | rates |
| 40 | 41 | `Packet Length Min` | yes |  | combined packet-length statistics |
| 41 | 42 | `Packet Length Max` | yes |  | combined packet-length statistics |
| 42 | 43 | `Packet Length Mean` | yes |  | combined packet-length statistics |
| 43 | 44 | `Packet Length Std` | yes |  | combined packet-length statistics |
| 44 | 45 | `Packet Length Variance` | yes |  | combined packet-length statistics |
| 54 | 55 | `Average Packet Size` | yes |  | combined packet-length statistics |
| 55 | 56 | `Fwd Segment Size Avg` | yes |  | forward packet-length augmentation |

Counts:

- padding support: 12;
- timing support: 12;
- overlap: 1 (`Flow Bytes/s`);
- joint support: 23.

---

## 11. Features deliberately excluded

### 11.1 Proven invariant

`Fwd Packet Length Std` is excluded because adding the same `p` to every forward packet
preserves standard deviation.

### 11.2 Genuinely unaffected fields

Ports, protocol, packet counts, TCP flags, backward packet-length statistics, header
fields, and window fields are not written by the primitive transform.

### 11.3 Level-C held-constant fields

Some aggregates might change after real packet editing but cannot be reconstructed from a
single aggregate flow vector. PrimAttack deliberately holds them constant:

- `Subflow Fwd Bytes`;
- forward bulk byte/packet/rate aggregates;
- `Flow IAT Std` and `Flow IAT Min`;
- active and idle mean/std/max/min;
- packet-sequence-dependent burst summaries.

Excluding these is not a claim that physical packet edits leave them invariant. It is a
statement that the canonical feature-space transformation cannot determine their new
values from the available aggregates.

---

## 12. Canonical implementation

### 12.1 Provenance record

`PrimAttackFeatureSupport` records, for each supported coordinate:

- feature name;
- resolved index;
- padding effect Boolean;
- timing effect Boolean;
- responsible `generate` recomputation block.

### 12.2 Canonical write maps

Two module-level maps define the support by actual write block:

```python
_PADDING_RECOMPUTATION
_TIMING_RECOMPUTATION
```

These maps are consumed by both:

- `CICIDS2017PrimitiveModel.primitives()` dependency metadata; and
- the public support-mask functions.

This avoids a separate CAPGD/C-PGD feature list that could drift from PrimAttack.

### 12.3 Manifest resolution

`primattack_feature_support(manifest)`:

1. forms the union of padding and timing write names;
2. verifies that every name exists in the supplied manifest;
3. walks the manifest’s frozen feature order;
4. emits index-bearing support records in classifier order.

An unknown or missing name raises immediately. No positional fallback is used.

### 12.4 Public masks

```python
primattack_padding_feature_mask(manifest)
primattack_timing_feature_mask(manifest)
primattack_joint_feature_mask(manifest)
```

Each returns a 79-element `torch.bool` tensor where `True` means the canonical PrimAttack
transform may write that classifier coordinate.

### 12.5 Primitive model integration

`CICIDS2017PrimitiveModel.__init__` derives:

- `controlled_idx` from `primattack_joint_feature_mask`;
- `frozen_idx` as its complement.

The primitive model, matched-support attacks, tests, and runner therefore consume the same
source of truth.

---

## 13. Dataset applicability

CICIDS2017-DistriNet and CSE-CIC-IDS-2018 DistriNet use the same corrected 79-feature
CICFlowMeter layout. The primitive model is shared across both datasets, and support is
resolved independently against each dataset’s manifest.

The support mask is not applied to the legacy CICIoT2023 pipeline, which has a different
39-feature schema and different attack semantics.

---

## 14. Use by matched-support CAPGD

`build_capgd_prim_support_resources` starts from the existing native CAPGD resources and
reuses:

- train-only min/max bounds;
- feature types;
- TabularBench scaler;
- differentiable relation constraints;
- validator object;
- upstream CAPGD implementation and parameters.

It replaces `Constraints.mutable_features` with the NumPy form of
`primattack_joint_feature_mask`.

CAPGD then directly optimizes all allowed feature coordinates. It does not call
`CICIDS2017PrimitiveModel.generate` and does not enforce primitive coupling.

After CAPGD returns a candidate, `finalize_capgd_output`:

1. clones the candidate;
2. copies all 56 unmasked coordinates from the clean input;
3. asserts exact equality outside the mask.

The configuration name is `capgd_prim_support`.

The existing `capgd_native` resource path remains separate. Its original direct/derived
mask and dependency repair continue to be used when the prim-support configuration is not
selected.

---

## 15. Use by matched-support C-PGD

`CPGDPrimSupportAttack` refuses any resource configuration other than
`capgd_prim_support`. Its gradient is multiplied by the exact same 79-bit support mask.

Every projection performs:

```text
delta = (candidate - clean) * support_mask
```

and returns clean values outside the mask. Final type repair repeats the restoration and
asserts exact equality.

C-PGD and CAPGD therefore share support, but retain different optimizers and attack logic.

---

## 16. Runner enforcement and artifact fields

The canonical runner builds the mask directly from the active dataset manifest:

```python
prim_support_mask = primattack_joint_feature_mask(manifest)
```

For `capgd_prim_support` and `cpgd_prim_support`, it computes exact per-row movement:

```python
changed = adv_raw != raw
outside_changed = changed[:, ~prim_support_mask].sum(dim=1)
```

If any outside-mask count is nonzero, the runner raises before writing an artifact.

Artifacts include:

- the complete 79-bit `mutable_feature_mask`;
- `n_allowed_primattack_support_features` — always 23;
- `n_features_modified` per sample;
- `n_modified_outside_primattack_mask` per sample — required to be zero;
- the attack’s exact parameter JSON.

The independent validator receives the same finalized sample used for prediction and
artifact metrics.

---

## 17. Verification tests

Primary mask coverage is in:

- `src/attack/tests/test_primattack_support_mask.py`
- `src/attack/tests/test_primattack_transformation.py`
- `tests/test_capgd_comparison.py`

### 17.1 Index contract

For both final datasets, tests assert the zero-based joint index tuple:

```text
(3, 6, 8, 9, 10, 16, 17, 18, 20, 22, 23, 24, 25, 26,
 38, 39, 40, 41, 42, 43, 44, 54, 55)
```

They also verify:

- each support record name matches the manifest at its resolved index;
- every record has recomputation provenance;
- padding count is 12;
- timing count is 12;
- joint count is 23.

### 17.2 Dynamic recomputation agreement

A representative test loads 8,192 pristine CICIDS2017 test flows and applies:

- feasible integer padding of 17 bytes per forward packet;
- feasible integer delay of 1,009 microseconds;
- delay shape `0.37`.

It independently computes which output columns differ from the clean batch and asserts:

```text
observed padding changes == padding mask
observed timing changes  == timing mask
observed union           == joint mask
```

This proves both directions on representative flows:

- no actual canonical write escapes the mask;
- every declared mask coordinate is observed to change for at least one representative
  feasible flow.

### 17.3 Matched-support attack checks

Tests also assert:

- CAPGD and C-PGD can move allowed coordinates;
- neither changes an unmasked coordinate;
- CAPGD still receives gradients;
- C-PGD victim and constraint penalties receive gradients;
- native CAPGD resources are not mutated by deriving matched-support resources;
- C-PGD respects L2 and Linf epsilon projection;
- outputs remain compatible with the evaluator.

The broader relevant regression selection completed with:

```text
46 passed, 1 skipped
```

---

## 18. Observed CICIDS2017 small-sample run

A requested smoke run used:

- CICIDS2017-DistriNet;
- MLP victim;
- DoS source class;
- 8 clean-correct samples;
- attack seed 42;
- 5 CAPGD steps;
- 5 C-PGD iterations;
- CPU execution.

| Method | Rows | Allowed | Modified per row | Max outside mask | Raw success | Validator pass | Valid success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `capgd_prim_support` | 8 | 23 | 20–22 | 0 | 7/8 | 0/8 | 0/8 |
| `cpgd_prim_support` | 8 | 23 | 21–23 | 0 | 2/8 | 0/8 | 0/8 |

Both artifacts were non-empty, all required schema fields were present, all perturbation
norms were finite, `failures.json` was empty, and pairing assertions passed.

The result confirms implementation behavior, not attack quality. In particular, raw
success did not survive the independent validity gate in this small sample.

---

## 19. Methodological interpretation

The matched-support comparison controls one factor:

> the set of downstream classifier coordinates exposed to direct optimization.

It does not control:

- feasible direction;
- primitive budget;
- packet/timing capability;
- quantization of primitive controls;
- deterministic relationships among changed features;
- semantic preservation;
- packet-level realizability.

Accordingly, correct language is:

> CAPGD/C-PGD were restricted to the set of features that PrimAttack’s canonical
> primitive transform can affect.

Incorrect language is:

> CAPGD/C-PGD used the same feasible attack space as PrimAttack.

---

## 20. Limitations

1. **Aggregate-flow proxy:** the mask describes writes in the canonical 79-feature
   transformation, not observed PCAP edits.
2. **Level-C omissions:** packet-sequence-dependent features are held constant because they
   cannot be reconstructed from aggregates.
3. **Conditional movement:** a true mask bit indicates possible movement, not guaranteed
   movement on every row.
4. **Dataset scope:** the mask applies to the shared CICIDS2017/2018 DistriNet schema only.
5. **Validity separation:** support membership does not imply validator validity.
6. **No malicious-function claim:** support and validator checks do not prove preservation
   of complete attack behavior.
7. **No campaign-generalization claim:** the dataset split is chronological within source
   label, not a global forward-time or new-campaign split.
8. **Small smoke samples:** observed smoke rates are execution evidence only.

---

## 21. File map

| File | Responsibility |
|---|---|
| `src/attack/realizability/cicids2017.py` | primitive capabilities, bounds, projection, recomputation, support provenance, public masks |
| `src/attack/tests/test_primattack_support_mask.py` | exact indices and dynamic support completeness |
| `src/attack/tests/test_primattack_transformation.py` | primitive equations, projection, and victim-gradient behavior |
| `src/comparisons/capgd_cicids2017.py` | native and matched-support CAPGD resources and finalization |
| `src/comparisons/cpgd_prim_support.py` | matched-support C-PGD mask enforcement |
| `scripts/run_full_adversarial_eval.py` | roster, shared mask construction, outside-mask assertion, metrics, artifacts |
| `docs/methods/primattack_matched_support.md` | concise combined reference |

---

## 22. Summary

The joint PrimAttack support mask is a manifest-resolved representation of the exact write
set of `CICIDS2017PrimitiveModel.generate` after primitive capabilities, budgets, and
projection. It contains 23 features: 12 padding-affected, 12 timing-affected, with
`Flow Bytes/s` shared by both.

The mask is canonical because the same name-based write maps drive primitive dependency
metadata, primitive-model controlled indices, matched-support CAPGD, matched-support
C-PGD, runner assertions, and tests. This provides a support-matched comparison while
preserving the essential methodological distinction between direct feature optimization
and primitive-coupled generation.
