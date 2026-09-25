# PrimAttack transformation verification

> **Superseded by PrimAttack v2.** The equations below describe the former
> proportional-only `alpha` timing map. The active transform uses integer total
> `delay` plus allocation `shape`; see
> [`../full_thesis_methodology/02_primattack.md`](../full_thesis_methodology/02_primattack.md).

## Canonical contract

The canonical feature transform is

$$x_{adv}=\phi(x_0,p,\alpha)$$

implemented only by `CICIDS2017PrimitiveModel.generate` in
`src/attack/realizability/cicids2017.py`. Direct PrimAttack optimizes two unconstrained leaves
that map into $(p,\alpha)$. The latent variant optimizes only $z_{adv}$, collapses decoder
movement into the same two controls, and calls the same `generate` method. Neither path sends an
independently edited CICFlowMeter vector to the victim.

The frozen `PrimitiveSpec` contract now records, for each primitive: units, dtype, direction,
identity, absolute bounds, complete writable dependency set, projection function, semantic risk,
and description.

## Verified equations

### Padding/size primitive

For uniform forward packet-length augmentation $p\ge0$:

$$L_f'=L_f+N_fp$$

$$l_{f,min}'=l_{f,min}+p,\quad l_{f,max}'=l_{f,max}+p$$

$$\bar l_f'=\frac{L_f'}{\max(N_f,1)}$$

The forward standard deviation is invariant under a uniform shift. Combined mean, extrema,
pooled sample variance/std, `Average Packet Size`, and `Flow Bytes/s` are recomputed. Packet
counts, protocol, ports, flags, headers, backward-only summaries, and explicitly unresolved
Level-C fields are copied from the source.

### Timing primitive

For forward-IAT dilation $\alpha\ge1$:

$$T_f'=\alpha T_f$$

The forward IAT min/max/std scale by $\alpha$ and the mean is recomputed as

$$\overline{IAT}_f'=\frac{T_f'}{\max(N_f-1,1)}.$$

The implemented flow-duration projection is

$$D'=\max\{D+(T_f'-T_f),T_f',T_b,1\ \mu s\}.$$

`Flow IAT Mean` and all packet/byte rates are then recomputed from $D'$. `Flow IAT Max` is a
conservative construction that adds the duration increase to the original maximum. This is not
claimed to be the unique packet-sequence result.

## Primitive-specific write isolation

A verification failure exposed that the earlier implementation recomputed the entire timing
block during a padding-only operation and the entire length block during a timing-only
operation. Even when the inactive primitive was at identity, small source/extractor residuals
could make unrelated features change. `generate` now writes:

- length/combined-length dependencies only on rows where $p\ne0$;
- timing and packet-rate dependencies only on rows where $\alpha\ne1$;
- `Flow Bytes/s` when either primitive is active;
- no feature on exact identity rows.

This makes the declared per-primitive dependency sets executable rather than descriptive.

## Automated checks

Tests are in:

- `src/attack/tests/test_primitive_controls.py`;
- `src/attack/tests/test_primattack_transformation.py`;
- `src/attack/tests/test_vae_latent_primitive.py`.

They verify:

1. **Zero-perturbation identity.** Both continuous and quantized
   `phi(x,{p=0,alpha=1})` are exactly equal to the source row, including a synthetic zero-duration
   boundary case.
2. **Immutability.** `FROZEN`, proven `INVARIANT`, and explicitly unresolved `LEVEL_C` fields
   remain byte-identical.
3. **Dependency completeness.** Every observed changed feature belongs to the active
   primitive's frozen `PrimitiveSpec.dependencies`; no arbitrary feature write is accepted.
4. **Timing consistency.** Forward-IAT scaling, mean recomputation, non-negative timing,
   non-decreasing duration, and ordering are checked from the implemented equations. The tests
   do not assert that every IAT statistic must increase independently.
5. **Padding/size consistency.** Total forward bytes, min/max/mean, combined length statistics,
   and no-volume-decrease are checked.
6. **Derived-feature consistency.** Only extractor relationships confirmed in the repository
   are asserted, including packet variance/std and average/mean identities.
7. **Discrete projection.** Padding is rounded to integer bytes and capped by the floor of its
   hard budget; integer-valued generated fields are integral after final projection.
8. **Autograd.** Targeted victim cross-entropy gradients with respect to both continuous
   controls are compared against central finite differences through
   `primitive map -> train-fitted scaler -> trained MLP victim`.
9. **Numerical safety.** Zero/tiny duration, identity boundaries, hard-limit projection,
   finite outputs, and explicit rejection of NaN/Inf controls are covered.

Verification command:

```text
PYTHONPATH=".;src" python -m pytest \
  src/attack/tests/test_primitive_controls.py \
  src/attack/tests/test_primattack_transformation.py \
  src/attack/tests/test_flow_semantics.py \
  src/attack/tests/test_primattack_budget.py \
  src/attack/tests/test_vae_latent_primitive.py \
  -q -p no:faulthandler
```

Observed result on 2026-09-24: **41 passed, 1 skipped**. The skipped case is a data-conditional
legacy test whose sampled rows contained no zero-forward-payload flow; the explicit synthetic
boundary tests still ran.

## Unknown/unverifiable dependencies

The aggregate row does not reveal the packet order or burst/subflow assignment required to
recompute `Flow IAT Std`, `Flow IAT Min`, active/idle summaries, bulk summaries, or subflow byte
summaries. It also cannot prove which packets can carry padding without changing application
semantics. These fields remain unchanged under an explicit `LEVEL_C`/UNKNOWN designation. Their
constancy is not presented as packet-extractor equivalence.

No generated artifact may contain NaN or Inf. The transform raises on non-finite source/control
inputs; the runner performs an additional final artifact check before saving.
