# PrimAttack primitive audit

> **Historical pre-v2 audit.** This file records the replaced `(p, alpha)` /
> Adam-sigmoid implementation and its then-open corrections. It is retained for
> provenance, not as the current specification. See
> [`../full_thesis_methodology/02_primattack.md`](../full_thesis_methodology/02_primattack.md)
> for the active `(p, delay, shape)` implementation and completed corrections.

## Scope and evidence

This audit covers the active CICIDS2017-DistriNet primitive paths:

- direct primitive optimization: `src/attack/run_cicids2017_primitive_attack.py`;
- latent optimization followed by primitive collapse: `src/attack/vae_latent_primitive.py`;
- canonical transform: `src/attack/realizability/cicids2017.py`;
- primitive interface: `src/attack/realizability/base.py`;
- internal consistency validator: `src/attack/realizability/validator.py`;
- regression tests: `src/attack/tests/test_primitive_controls.py`.

The feature order and split provenance come from
`data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json`. Feature semantics are
cross-checked against `docs/cicids2017_feature_reference.md`, the corrected-release
preprocessing record in
`docs/data/cicids2017distrinet/cicids2017distrinet_preprocessing.md`, and the relationships
already associated with this project in `docs/attack_system/02_constraint_discovery.md` and
`05_realizability_model.md`. The codebase does not contain packet traces or a complete packet
sequence for each saved flow. Dependencies that require those data are marked **UNKNOWN** and
must not be represented as exact CICFlowMeter reconstruction.

## Pipeline audit

### Direct primitive path

```text
X_test_pristine row
  -> train-only per-flow envelope bounds
  -> unconstrained leaves u,v
  -> p = p_hi sigmoid(u), alpha = 1 + (alpha_hi - 1) sigmoid(v)
  -> CICIDS2017PrimitiveModel.generate(raw, controls, quantize=False)
  -> train-fitted RobustScaler affine transform
  -> victim logits
  -> targeted cross-entropy + normalized feature-change penalty
  -> final primitive projection
  -> generate(..., quantize=True)
  -> victim and validators
```

`u` and `v` are the only optimizer leaves. No CICFlowMeter feature tensor is optimized
independently. The final vector is always produced by `generate`; unrelated columns begin as a
clone of the pristine row and are not assigned afterward.

The direct method uses the C&W change-of-variable pattern for the primitive box, but its
classification term is targeted cross-entropy rather than the canonical C&W logit-margin loss.
Calling the whole method “C&W” without that qualification would be imprecise.

### Latent primitive path

```text
z_adv (only optimizer leaf)
  -> VAE decoder
  -> infer_primitives_from_decoded
  -> p, alpha
  -> CICIDS2017PrimitiveModel.generate
  -> train-fitted RobustScaler affine transform
  -> victim logits
  -> targeted CE or C&W margin plus latent/cost/realism terms
```

The decoded 79-feature vector is never classified directly. It only proposes two controls;
`generate` remains the only path into the victim. The graph is therefore genuinely
`loss -> victim -> transformed features -> primitive map -> primitive controls`; for the latent
variant it continues through the decoder to `z_adv`.

### Scaling and loss

Both paths transform raw generated features using `(x_raw - center) / scale`, where the center
and scale come from the training-fitted `RobustScaler`. They do not fit or update preprocessing
on validation/test data. Direct PrimAttack uses targeted cross-entropy to Benign plus a soft
normalized control-position penalty. Latent PrimAttack supports targeted C&W margin or targeted
cross-entropy and adds latent, feature-cost, realism, and optional decoder-movement terms.

### Current final projection

`project_controls` rounds `p` to whole bytes and keeps `alpha` continuous. Final
`generate(..., quantize=True)` rounds integer-valued length/timing outputs to integer bytes or
microseconds. Projection is performed only for final evaluation; optimization uses the
continuous differentiable map. The current per-flow box already acts as a hard upper bound
through sigmoid/clamp operations, but the box is based on a train maximum plus CLI ceilings,
not the requested robust, class-conditional calibrated budgets.

## Primitive inventory

### Forward packet-length augmentation

| Field | Audit result |
|---|---|
| Primitive name | `p` |
| Symbol | $p$ |
| Interpretation | Uniform non-negative length augmentation applied to every forward packet represented by the flow. It is a flow-level padding/length model, not proof of application-payload insertion. |
| Units | bytes added per forward packet |
| Type | Discrete at final evaluation; continuously relaxed during optimization |
| Identity | $p=0$ |
| Allowed direction | Increase only |
| Implementation | `CICIDS2017PrimitiveModel.primitives`, `project_controls`, and `generate` in `src/attack/realizability/cicids2017.py` |
| Current absolute limits | Spec lower bound 0; no spec upper bound. Runner default CLI ceiling 1460 bytes. Per-flow cap further intersects this ceiling with train-envelope headroom and optional MTU-style headroom. |
| Capability gate | Enabled only if `Total Fwd Packet >= 1`, `Total Length of Fwd Packet > 0`, and `Fwd Packet Length Mean > 0`. Otherwise projected to 0. |
| Projection | Clamp to non-negative; capability mask; final `round(p)` |
| Semantic risk | Uniform padding may be impossible for some packets; aggregate features do not identify packet payload/header composition. It does not establish preserved payload semantics. |
| Current correctness | Correct as a declared uniform flow-level length-shift model for known algebraic dependencies. Not packet-realization verified. |
| Required correction | Extend `PrimitiveSpec`; replace hard-coded/train-maximum budget selection with frozen train-only calibration; emit requested/projected values and explicit budget compliance; preserve UNKNOWN dependency status. |

Known recomputation under $p$:

- `Total Length of Fwd Packet`:
  $L_f' = L_f + N_f p$.
- `Fwd Packet Length Min`, `Fwd Packet Length Max`:
  $m_f'=m_f+p$, $M_f'=M_f+p$.
- `Fwd Packet Length Mean` and `Fwd Segment Size Avg`:
  $\bar l_f'=L_f'/\max(N_f,1)$.
- `Fwd Packet Length Std`: unchanged; a uniform shift is variance-invariant.
- `Packet Length Mean` and `Average Packet Size`:
  $(L_f'+L_b)/\max(N_f+N_b,1)$.
- `Packet Length Min`/`Max`: branch on whether forward/backward directions are present, then
  take the extrema of the updated forward and unchanged backward extrema.
- `Packet Length Variance`/`Std`: pooled sample variance/std from forward and backward count,
  mean, and std.
- `Flow Bytes/s`: recomputed using updated total bytes and projected duration.

Known unaffected values include counts, protocol, ports, direction metadata outside the model
matrix, backward length statistics, flags, headers, and initial-window fields.

Dependencies that cannot be established from the aggregate row are **UNKNOWN** rather than
independent invariants: `Fwd Act Data Pkts`, forward bulk statistics, and `Subflow Fwd Bytes`.
The current implementation holds these values constant and labels them `LEVEL_C`; that is
honest bookkeeping, but it is not a packet-extractor equivalence claim.

### Forward timing dilation

| Field | Audit result |
|---|---|
| Primitive name | `alpha` |
| Symbol | $\alpha$ |
| Interpretation | Multiplicative dilation of every forward inter-arrival gap represented by the flow |
| Units | dimensionless ratio |
| Type | Continuous control; dependent extractor timing fields are quantized to microseconds for final evaluation |
| Identity | $\alpha=1$ |
| Allowed direction | Delay/dilation only; never timing compression |
| Implementation | `CICIDS2017PrimitiveModel.primitives`, `project_controls`, and `generate` in `src/attack/realizability/cicids2017.py` |
| Current absolute limits | Spec lower bound 1; no spec upper bound. Runner default CLI ceiling 100. Per-flow cap intersects this with train-envelope headroom. |
| Capability gate | Enabled only if `Total Fwd Packet >= 2` and `Fwd IAT Total > 0`. Otherwise projected to 1. |
| Projection | Clamp to at least 1; capability mask; `alpha` remains continuous; dependent integer timing fields are rounded to microseconds |
| Semantic risk | The aggregate row does not contain the merged forward/backward packet sequence. Duration and flow-IAT effects therefore cannot be uniquely reconstructed from the row. Excessive dilation can destroy attack-like traffic intensity. |
| Current correctness | Exact for scaling the saved forward-IAT summary under the declared uniform dilation. Duration and `Flow IAT Max` are conservative constructions, not confirmed extractor-exact transformations. |
| Required correction | Extend `PrimitiveSpec`; calibrate robust class-specific timing envelopes from train only; add duration/rate semantic retention checks; explicitly preserve UNKNOWN status for merged-sequence statistics. |

Known recomputation under $\alpha$:

- `Fwd IAT Total`, `Fwd IAT Max`, `Fwd IAT Min`, and `Fwd IAT Std` are multiplied by
  $\alpha$.
- `Fwd IAT Mean` is recomputed as
  $\mathrm{FwdIATTotal}'/\max(N_f-1,1)$.
- The implemented conservative duration projection is
  $D'=\max(D+(\alpha-1)\,\mathrm{FwdIATTotal},
  \mathrm{FwdIATTotal}',\mathrm{BwdIATTotal},1\,\mu s)$.
- `Flow IAT Mean` is recomputed as $D'/\max(N_f+N_b-1,1)$.
- `Flow IAT Max` is implemented as the original maximum plus non-negative added duration. This
  is one conservative feasible construction, not an extractor identity established from the
  aggregate row.
- `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, and `Bwd Packets/s` are recomputed over
  $D'/10^6$ seconds.

`Flow IAT Std`, `Flow IAT Min`, active/idle statistics, subflow statistics, and bulk statistics
require the unavailable packet order/burst segmentation. Their transformed values are
**UNKNOWN**. The current map holds them constant under the explicit `LEVEL_C` role. They must
not be described as physically invariant or fully recomputed.

## Dependency-completeness assessment

The transform recomputes every dependency for which the repository has either an established
CICFlowMeter identity or a declared primitive shift equation. Earlier defects—frozen combined
packet-length statistics and frozen `Flow IAT Mean`—are already corrected.

The remaining held fields divide into two groups:

1. **True immutable/unaffected fields:** protocol, ports, packet counts, backward-only
   statistics, flags, headers, initial windows, and other fields not changed by the declared
   primitives.
2. **Unavailable sequence-dependent fields:** bulk, subflow, active/idle, merged Flow-IAT
   dispersion/minimum, and data-bearing-packet semantics. These are not demonstrably immutable.
   The transform keeps them unchanged because the aggregate dataset cannot reconstruct them and
   marks them `LEVEL_C`.

Accordingly, `phi` is a deterministic flow-feature transformation/proxy. It is not a complete
CICFlowMeter re-extraction of a modified packet trace.

## Bugs and methodological gaps found

1. **PrimitiveSpec is only partial.** It lacks explicit dtype, direction, dependency set,
   projection function identifier, and semantic-risk fields required for a frozen contract.
2. **Budget selection is not yet defensible enough.** `train_envelope` uses a global train
   maximum, while `p_max=1460` and `alpha_max=100` are CLI defaults. These are not robust,
   class-conditional, semantic-retention-calibrated budgets.
3. **The identity contract has a boundary risk.** `generate` floors duration to 1 microsecond.
   If a pristine row has zero duration, `phi(x, identity)` changes it. The all-row identity
   property must be checked and the no-op path made exact without permitting unsafe division.
4. **Requested versus projected controls are incomplete in artifacts.** Existing NPZ files
   store continuous and realized values, but not a general budget name, calibrated limit,
   compliance reason, primitive costs, semantic status, or complete per-sample audit record.
5. **Primitive feasibility and semantic preservation are conflated in prose.** Capability
   masks answer whether an operation is representable from available flow evidence; they do
   not prove preservation of attack behavior.
6. **No separate flow-semantic validator exists.** Current realizability/domain validators
   check representation consistency, not attack-class semantic retention.
7. **No finite-difference victim-loss gradient check exists.** Current tests prove non-zero
   primitive gradients but do not compare autograd against numerical derivatives through the
   actual victim/scaler path.
8. **No explicit numerical-safety suite covers zero/tiny duration and boundary controls.**
9. **Current strict metric naming is inconsistent.** The runner’s saved `strict_valid` value is
   assigned from `mined_valid`; semantic feasibility and preservation require separate gates and
   names.

## Corrections to retain versus defer

Retain:

- the two primitives `p` and `alpha`;
- packet-count/injection primitives disabled;
- one canonical `generate` path;
- continuous optimization followed by final discrete projection and reclassification;
- training-fitted scaler use;
- fail-closed capability masks;
- explicit `LEVEL_C`/UNKNOWN features.

Implement next:

- canonical expanded `PrimitiveSpec` contract and budget projection;
- robust train-only calibration artifact and named levels;
- transformation, finite-difference, projection, and numerical-safety tests;
- separate `FlowSemanticValidator` with tri-state checks;
- class-specific flow-level proxy rules for DoS, DDoS, Recon (PortScan category), and
  BruteForce, using only available fields;
- per-sample audit records, staged ASRs, testability coverage, matched ablations, and budget
  sensitivity outputs.

Do not implement packet crafting, PCAP editing, replay, service interaction, scanning,
exploitation, or malware execution.
