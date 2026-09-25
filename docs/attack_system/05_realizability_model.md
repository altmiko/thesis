# 05 — The Realizability Model (the `(p, alpha) -> 79 features` map)

> **Superseded PrimAttack-v1 explanation.** This page documents the replaced
> `(p, alpha)` map. The current `(p, delay, shape)` canonical map, bounds, equations,
> roles, tests, and limitations are documented in
> [`../full_thesis_methodology/02_primattack.md`](../full_thesis_methodology/02_primattack.md).

**Source:** `src/attack/realizability/cicids2017.py` (`CICIDS2017PrimitiveModel`),
`src/attack/realizability/base.py` (contracts).

This is the physical heart of the primitive attacks. It is a **differentiable, deterministic
function** that takes a pristine raw flow plus the two attacker controls and produces a new
raw flow in which **every feature affected by a control is recomputed** and everything else is
byte-identical to the source. It is the single component that makes "perturb 2 numbers,
regenerate 79 consistently" possible.

---

## 1. The two primitives (`primitives()`)

| Primitive | Meaning | identity | range | integer |
|-----------|---------|----------|-------|---------|
| `p` | forward packet-length augmentation — bytes added to every forward packet's length | 0 (no change) | `[0, ∞)` | yes (rounded on projection) |
| `alpha` | forward inter-arrival timing dilation — multiply every forward IAT gap | 1 (no change) | `[1, ∞)` | no (kept continuous; realizability via µs quantization of derived features) |

**Semantics chosen conservatively.** `p` is *length* augmentation (padding), **not** asserted
payload insertion — so `Fwd Act Data Pkts` is held frozen and flagged a packet-level (Level-C)
limitation rather than over-claimed. `alpha >= 1` means the attacker can only **delay**, never
compress, so duration stays positive and rates stay finite.

**Primitive-realizable, not functionality-verified.** A successful sample here is
*semantically-admissible primitive-realizable*: the discrete primitives produce internally
consistent CICFlowMeter features (Levels A + B) **and** each applied primitive is supported by
evidence in the source flow (§4.1). It is **not** a claim that the modified traffic still
carries out the malicious behaviour — establishing that requires packet-level replay
(Level C, §7). Scope claims as *feature-space, semantically-admissible, primitive-realizable
evasion under a conservative padding+timing threat model* — never universally "real-world
realizable".

---

## 2. Feature roles (`roles()`) — how each feature reacts

`base.py::FeatureRole` defines eight tags. No CICFlowMeter feature is ever an attack variable;
these describe how each feature is *derived from* or *reacts to* the two primitives:

| Tag | Enum | Meaning |
|-----|------|---------|
| `Dp` | `DERIVED_P` | exactly derived from `p` |
| `Dt` | `DERIVED_T` | exactly derived from `alpha` |
| `D` | `DERIVED` | other exact algebraic derived |
| `C` | `CONDITIONAL` | conditionally / conservatively reconstructed |
| `R` | `RATE` | rate (count/byte over projected duration) |
| `I` | `INVARIANT` | **proven** invariant under the relevant primitive |
| `F` | `FROZEN` | genuinely unaffected |
| `Fᶜ` | `LEVEL_C` | **unresolved**, held constant — *would* change under real packet edits but is not reconstructable from the aggregate flow (**not** claimed physically invariant) |

Selected assignments (`roles()`):

- **Dp:** `Total Length of Fwd Packet` (`= TL_fwd0 + Nf·p`), `Fwd Packet Length Min/Max`
  (`+= p`), `Fwd Packet Length Mean`, `Fwd Segment Size Avg`, `Packet Length Mean`,
  `Average Packet Size`, `Packet Length Variance`, `Packet Length Std`.
- **I (proven invariant):** `Fwd Packet Length Std` — a uniform shift preserves std.
- **Dt:** `Fwd IAT Total/Max/Min/Std` (`= alpha··`), `Fwd IAT Mean`.
- **C (conservative):** `Packet Length Max/Min` (direction-presence branch),
  `Flow Duration` (delay projection), `Flow IAT Max` (added delay lands in the largest gap —
  keeps `mean ≤ max ≤ dur`).
- **R:** `Flow Bytes/s`, `Flow Packets/s`, `Fwd Packets/s`, `Bwd Packets/s`.
- **Fᶜ (Level-C held constant):** `Fwd Act Data Pkts`, `Subflow Fwd Bytes`, `Fwd
  Bytes/Bulk Avg`, `Fwd Bulk Rate Avg`, `Fwd Packet/Bulk Avg`, `Flow IAT Std`, `Flow IAT Min`,
  and all 8 `Active/Idle *`. These depend on packet-level timing/size *sequences* that cannot
  be reconstructed from aggregates, so they are held constant and honestly flagged, never
  fabricated.
- **F:** everything else (backward stats, counts, ports/protocol, flags, headers, window,
  down/up ratio, ...).

`controlled_idx` = every feature with a Dp/Dt/D/C/R role (the transform *writes* these);
`frozen_idx` = everything else (the transform never touches these).

---

## 3. Exact identities re-checked independently (`algebraic_identities()`)

The model returns a tuple of `IdentityCheck(target, parents, fn, atol, rtol)`. Each is the
single source of truth for both *recompute intent* and *independent validation* — but the
validator recomputes `fn` from the **adversarial** vector, so a generator that forgets to
update a dependent feature is caught. Examples with their tolerances:

```
Fwd Packet Length Mean = TL_fwd / max(Nf, 1)                     atol 1e-3 rtol 1e-4
Packet Length Variance = Packet Length Std**2                    atol 1.0  rtol 1e-3
Packet Length Max/Min  = ext(fwd±, bwd) branch on direction      (default)
Fwd IAT Mean           = Fwd IAT Total / max(Nf-1, 1)            atol 1.0  rtol 1e-4
Flow IAT Mean          = Flow Duration / max(Nf+Nb-1, 1)         atol 1.0  rtol 1e-4
Fwd/Bwd/Flow Packets/s = count / (dur_us/1e6)                    atol 1e-3 rtol 1e-3
Flow Bytes/s           = (TL_fwd+TL_bwd) / (dur_us/1e6)          atol 1e-2 rtol 1e-3
```

Absolute tolerances scale with the physical magnitude of each quantity (variance and IAT are
large; rates are small).

---

## 4. Semantic capabilities, activity masks & per-flow bounds

### 4.1 Semantic capabilities (`infer_capabilities(raw)`)

Numerical feasibility (inside the train envelope) is **necessary but not sufficient**: a
primitive can be numerically applicable yet unsupported by the source flow. `infer_capabilities`
is a per-flow, **conservative (fail-closed)** gate applied to the bounds *before the optimizer
ever sees them*, returning a `PrimitiveCapabilities` object (`pad_allowed`, `timing_allowed`
bool tensors + one reason code per flow):

- **`p` (forward-length augmentation)** is admissible iff the source has forward packets
  (`Total Fwd Packet >= MIN_FWD_PACKETS_FOR_PADDING`, default 1) **and** non-zero forward
  payload (`Total Length of Fwd Packet > 0` **and** `Fwd Packet Length Mean > 0`). A flow with
  no forward payload (e.g. a single-SYN Recon probe: `Nf=1`, `TL_fwd=0`) has no forward data to
  augment → `pad_allowed = False` (reason `NO_FORWARD_PAYLOAD`). This is the fix for the
  pathological case where `p=28` was permitted on a zero-payload flow.
- **`alpha` (forward timing dilation)** is admissible iff there are ≥2 forward packets (a
  forward IAT sequence exists; else `SINGLE_FWD_PACKET`) **and** that sequence is non-zero
  (`Fwd IAT Total > 0`; else `ZERO_TIMING_HEADROOM`).

Aggregate CICFlowMeter features cannot reveal *which* forward packets carry modifiable
application data, so this is a **conservative feature-level model of primitive realizability**,
not a packet-semantics claim; ambiguous flows fall to the identity. The rule is
class-agnostic — a DoS, DDoS, Recon, or BruteForce flow with the same structure is treated
identically (no per-class special cases).

### 4.2 Activity masks & bounds

`active_mask(raw, primitive, capabilities=None)` returns the capability mask directly
(`pad_allowed` for `p`, `timing_allowed` for `alpha`); it infers capabilities from `raw` when
none are passed. Inadmissible primitives are clamped to the identity **inside the graph**, so
the optimizer wastes no effort and reports no learned effect.

`per_flow_bounds(raw, config, capabilities=None)` computes **data-mined per-flow feasible caps**
so padding/dilation keeps the flow inside the **train envelope**, then applies the semantic gate:

- `p` headroom = min over the four forward-length features of `(train_env - current)`
  (`Total Length of Fwd Packet` headroom divided by `Nf`), optionally also `mtu_cap - fwd_max`
  if an MTU cap is set; then clamped to `[0, p_max]`.
- `alpha` headroom = min over `{Fwd IAT Total/Max/Std/Mean}` of `env / current`, plus a
  duration-based cap `(env_dur - dur)/fwd_iat_total + 1`; clamped to `[1, alpha_max]`.
- **Semantic gate (enforced twice):** `p_hi := p_hi · pad_allowed` (→ 0 where inadmissible)
  and `alpha_hi := 1` where `~timing_allowed`. So `p_hi^semantic(x) = p_hi^envelope(x) · m_p(x)`
  with `m_p(x) ∈ {0,1}`. The pre-gate caps are also returned as `p_numeric` / `alpha_numeric`
  so artifacts record both the numeric feasibility *and* the semantic decision (the NPZ stores
  `p_hi_numeric` vs `p_hi_semantic`, `pad_semantic_allowed`, and `pad_disable_reason`).

`train_envelope(raw_train, i)` (in the runner) mines the caps as the **train max** of the
controlled forward length/timing features — leakage-safe (train only).

---

## 5. Decoder → primitive inference (used by the latent attack)

`infer_primitives_from_decoded(raw0, decoded_adv_raw, decoded_base_raw, bounds)` collapses a
VAE decoder proposal into feasible `(p, alpha)` **differentiably**. Movement is measured
relative to `decode(z0)` (not `raw0`) so the constant VAE reconstruction bias cancels and the
primitive is driven purely by the latent displacement `z_adv - z0`:

- `p_hat = relu(mean of four forward-length movement signals)`, then `min(p_hat, bounds[p])`,
  masked to 0 where `p` is inactive. `relu` is the correct map: padding cannot shorten packets,
  so negative proposed movement → `p = 0`.
- `alpha_hat = exp(relu(mean log-ratio of {Fwd IAT Total, Fwd IAT Mean, Flow Duration}))` —
  geometric mean of timing ratios is the exact conversion back to a multiplicative delay
  factor; `min(alpha_hat, bounds[alpha])`, masked to 1 where inactive.

---

## 6. Projection and generation

`project_controls(raw, controls)` → realizable controls: `p` **rounded to integer bytes**,
`alpha` kept continuous (its realizability is enforced by integer-microsecond quantization of
the derived timing features), both re-masked to identity where inactive.

`generate(raw, controls, quantize)` → the adversarial raw vector. It:
1. re-applies activity masks and clamps (`p >= 0`, `alpha >= 1`; rounds `p` if `quantize`);
2. **forward length:** `tl_fwd = TL_fwd0 + Nf·p`, `fmin/fmax += p`, `fmean = tl_fwd/Nf`,
   `fstd` unchanged; writes the 5 forward-length features;
3. **forward timing:** `fit = alpha·fit0`, scales `Fwd IAT Max/Min/Std`, projects
   `dur = max(dur0 + (fit - fit0), fit, bwd_iat_total, dur_floor)`, grows `Flow IAT Max` by the
   added delay, recomputes `Fwd IAT Mean` and `Flow IAT Mean`; if `quantize`, rounds timing to
   integer µs (`_DUR_FLOOR_US = 1.0`);
4. **combined length stats:** conditional `Packet Length Max/Min` (direction branch),
   `Packet Length Mean`, `Average Packet Size`, pooled `Packet Length Variance`/`Std`;
5. **rates:** recomputed from the new totals and projected duration
   (`dur_s = max(dur/1e6, 1e-12)`).

Every other column is left equal to `raw` — hence the runner's hard assertion that all frozen
features are byte-identical to the pristine source (within `SCALER_ATOL = 1e-6`).

---

## 7. Why this is "realizability-aware," not "packet-verified"

The map operates in **feature space** (Levels A + B: domains + algebraic consistency). Actual
packet-level replay (Level C — edit the PCAP, re-run CICFlowMeter, re-extract) is *not*
performed: `base.py::NullPacketBackend` declares Level-C unavailable, and `PacketEditPlan` /
`PacketVerificationBackend` are the plug-in seams for a future backend. No current experiment
asserts Level-C realizability — this is stated plainly in the runner docstring and is an
honest limitation to report. The `Fᶜ` role and the frozen `Fwd Act Data Pkts` exist precisely
because we refuse to over-claim what feature-space edits guarantee at the packet level.
