# 2. PrimAttack (MAXIMUM DETAIL)

PrimAttack is a **primitive-domain white-box evasion attack**. Instead of perturbing
the 79-dim feature vector directly, it optimizes **two physically-interpretable
attacker primitives** — forward-packet **padding** `p` and forward-timing
**dilation** `α` — and *deterministically recomputes* all dependent CICFlowMeter
features through a differentiable transform `φ(x₀, p, α)`. This keeps every
adversarial vector inside the CICFlowMeter algebra by construction.

### Source files
| Concern | File |
|---|---|
| Primitive contracts, roles, capabilities, `NullPacketBackend` | `src/attack/realizability/base.py` |
| The transform φ, capabilities, per-flow bounds, projection | `src/attack/realizability/cicids2017.py` (`CICIDS2017PrimitiveModel`) |
| Internal primitive-consistency validator | `src/attack/realizability/validator.py` (`RealizabilityValidator`) |
| Train-only budget calibration | `src/attack/primattack_budget.py` |
| Flow-level semantic proxy | `src/attack/flow_semantics.py` |
| Runner + metrics + npz artifacts | `src/attack/run_cicids2017_primitive_attack.py` |
| Budget/mode sweep | `scripts/budget_sweep_primitive.py` |

---

## 2.1 One sample end-to-end

Trace of a single source flow `x₀` (raw/pristine 79-vector), for one
(class, victim, budget, mode, seed) cell (`run :302-491`):

1. **Clean-correct eligibility** — victim must classify the *clean* flow correctly:
   `clean_correct = victim(scale(x₀)).argmax == class_id` (`evaluate_cell :164`).
   ASR denominators use only eligible rows.
2. **Capability inference** — `infer_capabilities(x₀)` (`:292-336`) decides per-flow
   whether `p` and `α` are *semantically admissible* (below).
3. **Per-flow hard bounds** — `per_flow_bounds(x₀, class_cfg, caps)` (`:349-433`)
   returns `p_hi, α_hi` (the feasible box), intersecting the class budget, the
   global-train p99 envelope, and (DoS/DDoS) a minimum-rate cap; capability gates
   force unsupported primitives to identity.
4. **Mode masking** — `_apply_primitive_mode` (`:123-135`) zeroes `p` (timing-only) or
   pins `α=1` (padding-only); `joint` keeps both.
5. **Optimization variables** — two unconstrained leaves `u, v`; controls via sigmoid.
6. **Attack loop** — `optimize_primitives` (`:65-106`): Adam over `u,v`, targeted-Benign
   cross-entropy + primitive-cost penalty; φ is applied every step (differentiable,
   `quantize=False`).
7. **Projection / quantization** — `project_controls` (`:494-520`) then
   `generate(..., quantize=True)` (`:523-624`) → final integer-consistent adversarial
   raw vector `x_adv`.
8. **Frozen-feature assertion** — features φ does not write must equal x₀ within
   `SCALER_ATOL=1e-6` (`run :345-350`).
9. **Victim evaluation** — targeted success `adv_pred==0(Benign)`, evasion
   `adv_pred≠class_id` (`evaluate_cell :155-166`).
10. **validator_v2** — `structural_masks(x_adv)` → `hybrid_valid` (doc 3).
11. **Primitive feasibility** — `FlowSemanticValidator.evaluate(...).primitive_feasible`
    ∧ internal transform consistency (`run :368-374`).
12. **Semantic proxy** — `semantic_status ∈ {PASS, FAIL, NOT_FULLY_TESTABLE}` (doc 6).
13. **Metrics + npz** — nested ASRs + full per-sample audit written to
    `attack_artifacts/<class>_<victim>_seed<seed>.npz` (`run :408-491`).

---

## 2.2 Padding primitive `p`

- **Physical meaning**: uniform **forward packet-length augmentation** — add `p` bytes
  to *every* forward packet's length (a feature-level model of forward padding: MSS
  padding / filler). Docstring `cicids2017.py:5-9`.
- **Units**: `bytes_per_forward_packet`, `dtype = discrete_integer`
  (`PrimitiveSpec`, `:149-164`).
- **Direction**: `increase_only`; `identity = 0`; `absolute_lower_bound = 0`; no fixed
  upper bound (bounded per-flow). Padding can never shorten packets.
- **Capability mask** (`infer_capabilities :313-315`): `pad_allowed = (Total Fwd Packet
  ≥ 1) ∧ (Total Length of Fwd Packet > 0) ∧ (Fwd Packet Length Mean > 0)`. A flow with
  zero forward payload (e.g. a single-SYN Recon probe) has nothing to pad →
  `p_hi = 0`. Reason codes `PAD_ALLOWED / NO_FORWARD_PAYLOAD / INSUFFICIENT_FWD_PACKETS`.
- **Legal range**: `0 ≤ p ≤ p_hi` where (`per_flow_bounds :372-381`)
  `p_hi = min( env_headroom_maxlen, env_headroom_minlen, env_headroom_meanlen,
  (env_TL_fwd − TL_fwd)/N_f ) clamped to [0, p_max]`, then ×`pad_allowed`.
  Envelopes are global-train p99 of the relevant length features; `p_max` is the class
  budget (below).
- **Integer projection** (`project_controls :513-518`): `p ← min(round(p),
  floor(p_hi))`, clamped ≥0, forced to 0 where `¬pad_allowed`.
- **Every feature p writes** (`generate :547-610`, roles `:198-211`):
  Total Length of Fwd Packet, Fwd Packet Length {Min,Max,Mean}, Fwd Segment Size Avg,
  Packet Length {Min,Max,Mean,Variance,Std}, Average Packet Size, and (via totals)
  Flow Bytes/s. **Fwd Packet Length Std is PROVEN-invariant** under a uniform shift.

### Exact padding recomputation (`generate`)
Let `N_f = Total Fwd Packet`, and subscript 0 = source value.
```
TL_fwd      = TL_fwd0 + N_f · p                    # accumulator (:548)
fmin        = fmin0 + p ;  fmax = fmax0 + p        # uniform shift (:549-550)
fmean       = TL_fwd / N_f                          # exact (:553)
Fwd Seg Size Avg = fmean                            # extractor identity (:559)
fstd        = fstd0                                 # shift-invariant (:554)
# combined fwd+bwd length stats (:587-610), N = N_f+N_b, has_f/has_b = direction present
Packet Length Max = ext(fmax, bwd_max) ;  Packet Length Min = ext(fmin, bwd_min)
Packet Length Mean = (TL_fwd + TL_bwd)/N ;  Average Packet Size = Packet Length Mean
Packet Length Variance = pooled_sample_variance(N_f,fmean,fstd ; N_b,bmean,bstd)  # (:597-603)
Packet Length Std = sqrt(Variance)
Flow Bytes/s = (TL_fwd + TL_bwd) / (duration_us/1e6)   # (:618-621)
```
`ext(...)` is the direction-branching min/max (`:591-594`). Pooled variance uses the
exact between+within decomposition (`:599-603`).

---

## 2.3 Timing primitive `α`

- **Physical meaning**: uniform **forward inter-arrival timing dilation** — stretch
  every forward IAT gap by `α` (delay only). Docstring `:10-12`.
- **Units**: `dimensionless_ratio`, `dtype = continuous`.
- **Direction**: `increase_only`; `identity = 1`; `absolute_lower_bound = 1`. `α ≥ 1`
  can never compress a flow, so duration stays positive and rates finite.
- **Capability mask** (`:317-319`): `timing_allowed = (Total Fwd Packet ≥ 2) ∧
  (Fwd IAT Total > 0)` — a forward IAT sequence must exist and be non-zero. Reason
  codes `TIMING_ALLOWED / SINGLE_FWD_PACKET / ZERO_TIMING_HEADROOM`.
- **Legal range**: `1 ≤ α ≤ α_hi` where `α_hi` is the min (`per_flow_bounds :383-421`) of
  - relative-duration cap `1 + B_D · Duration/Fwd IAT Total` (`B_D` = class timing budget),
  - per-feature envelope caps `env_{Fwd IAT Total,Max,Std,Mean}/current`,
  - a duration-envelope cap `(env_Flow Duration − Duration)/Fwd IAT Total + 1`,
  - **(DoS/DDoS only)** a minimum-rate cap: duration may not grow past
    `N·1e6/min_flow_packets_per_second` (`:409-420`),
  then ×`timing_allowed` (else 1).
- **Projection** (`:516`): `α ← min(α, α_hi)`, clamped ≥1; integer microsecond fields
  quantized by `generate(quantize=True)`.
- **Every feature α writes** (`generate :561-617`, roles `:212-223`):
  Fwd IAT {Total,Mean,Std,Max,Min}, Flow Duration, Flow IAT Mean, Flow IAT Max,
  Flow Bytes/s, Flow Packets/s, Fwd Packets/s, Bwd Packets/s.

### Exact timing recomputation (`generate`)
```
fit   = α · fit0                                   # Fwd IAT Total (:563)
fimax = α · fimax0 ; fimin = α · fimin0            # (:564)
fistd = α · fistd0                                 # scale-equivariant (:565)
dur   = Duration0 + (fit − fit0)                   # conservative delay projection (:567)
dur   = max(dur, fit, Bwd IAT Total)  ≥ dur_floor_us (=1µs)   # (:568-569)
Flow IAT Max = Flow IAT Max0 + max(dur − Duration0, 0)        # added delay → largest gap (:571)
fimean = fit / max(N_f − 1, 1)                     # Fwd IAT Mean (:576)
Flow IAT Mean = dur / max(N − 1, 1)                # (:577)
dur_s = max(dur,·)/1e6
Fwd Packets/s = N_f/dur_s ; Bwd Packets/s = N_b/dur_s ; Flow Packets/s = N/dur_s   # (:615-617)
Flow Bytes/s  = (TL_fwd+TL_bwd)/dur_s                                              # (:618-621)
```
**Timing assumptions** made explicit in the docstring (`:28-34`): Flow Duration under
dilation is a *conservative packet-sequence projection* (all added forward delay
extends the flow); Flow IAT Max grows by the same delay so `mean ≤ max ≤ duration`
stays exact. Both are labelled **CONDITIONAL_DERIVED / Level-C-approximate**.

---

## 2.4 φ(x₀, p, α): per-feature map

`generate` (`:523-624`) writes only rows where the primitive is active
(`padding_rows = p≠0`, `timing_rows = α≠1`); rows with `p=0 ∧ α=1` are returned as an
exact copy (`identity_rows`, `:624`). Every write is differentiable except the optional
`round` under `quantize=True`.

| Feature | Primitive | Original inputs | Equation | Differentiable? | Projection |
|---|---|---|---|---|---|
| Total Length of Fwd Packet | p | TL_fwd0, N_f, p | `TL_fwd0 + N_f·p` | yes | round (quantize) |
| Fwd Packet Length Min/Max | p | min0/max0, p | `+ p` | yes | round |
| Fwd Packet Length Mean | p | TL_fwd, N_f | `TL_fwd/N_f` | yes | — |
| Fwd Segment Size Avg | p | Fwd Pkt Len Mean | `= mean` (EXT identity) | yes | — |
| Fwd Packet Length Std | p | — | **invariant** (unchanged) | n/a | — |
| Packet Length Min/Max | p | fwd±p, bwd | direction-branched ext | yes | round (len) |
| Packet Length Mean / Average Packet Size | p | TL_fwd,TL_bwd,N | `(TL_fwd+TL_bwd)/N` | yes | — |
| Packet Length Variance / Std | p | N_f,fmean,fstd,N_b,… | pooled sample variance | yes | — |
| Fwd IAT Total | α | fit0 | `α·fit0` | yes | round |
| Fwd IAT Max/Min | α | fimax0/fimin0 | `α··` | yes | round |
| Fwd IAT Std | α | fistd0 | `α·fistd0` | yes | — |
| Fwd IAT Mean | α | fit, N_f | `fit/(N_f−1)` | yes | — |
| Flow Duration | α | Dur0, fit−fit0, bwd | `Dur0+(fit−fit0)`, floored | yes | round |
| Flow IAT Mean | α | dur, N | `dur/(N−1)` | yes | — |
| Flow IAT Max | α | Flow IAT Max0, Δdur | `+ max(Δdur,0)` | yes | round |
| Fwd/Bwd/Flow Packets/s | α | counts, dur | `count/dur_s` | yes | — |
| Flow Bytes/s | p, α | TL_fwd+TL_bwd, dur | `bytes/dur_s` | yes | — |

Roles enum (`base.py FeatureRole`): `DERIVED_P (Dp)`, `DERIVED_T (Dt)`, `DERIVED (D)`,
`CONDITIONAL (C)`, `RATE (R)`, `INVARIANT (I)`, `FROZEN (F)`, `LEVEL_C (Fᶜ)`.

### Frozen / immutable / held-constant features
- **Frozen (F)** — genuinely unaffected (ports, protocol, packet counts, flags, header
  lengths, backward-length stats, Init-Win bytes, …): `frozen_idx` = every column φ does
  not write (`:112`); asserted unchanged post-attack (`run :345-350`).
- **Invariant (I)** — Fwd Packet Length Std, *proven* unchanged under a uniform shift.
- **Level-C held-constant (Fᶜ)** (`roles :224-235`): **UNRESOLVED** features that *would*
  change under real packet edits but are **not reconstructable from aggregate flow**, so
  held constant and flagged (never fabricated): `Fwd Act Data Pkts` (p is length
  augmentation, *not* asserted payload insertion), `Subflow Fwd Bytes`,
  `Fwd {Bytes,Packet,Bulk Rate}/Bulk Avg`, `Flow IAT Std`, `Flow IAT Min`, and all
  Active/Idle {Mean,Std,Max,Min}.

**Assumptions these introduce**: holding Fᶜ constant means the adversarial vector is
*internally CICFlowMeter-consistent* but not *packet-trace-consistent* — real padding
would move `Fwd Act Data Pkts`/subflow/bulk/active-idle; these are the explicit
Level-C limitations. Duration and Flow-IAT-Max effects are conservative projections,
not derived from the merged packet order (which the aggregate row does not contain).

---

## 2.5 Optimization

### Variables `u, v` and the sigmoid parameterization (`optimize_primitives :65-106`)
Two unconstrained scalar leaves per flow, initialized `u,v = −2 + 0.5·𝒩(0,1)`
(`init_noise=0.5`, seeded generator). Controls are mapped into the **hard per-flow box**
so gradients are unconstrained but outputs are always feasible:
```
p(u)  = p_hi · σ(u) · 1[pad_allowed]
α(v)  = 1 + (α_hi − 1) · σ(v) · 1[timing_allowed]
```
`σ` = logistic sigmoid. Initializing at `−2` makes `σ(−2)≈0.12`, i.e. attacks start
near identity (small perturbation) and grow only if it helps.

### Objective (Adam / C&W-style) (`:92-103`)
Per-flow loss, minimized by **Adam** over `[u,v]`:
```
L = CE( victim( (φ(x₀,p(u),α(v)) − center)/scale ), target=Benign )
    + cost_weight · ( σ(u)·1[pad] + σ(v)·1[timing] )
```
- **Targeted cross-entropy** toward class 0 (Benign) — this is a *targeted→Benign*
  attack (`target = zeros`, `:81`).
- **Primitive-cost penalty** `cost_weight·(σ(u)+σ(v))` (default `cost_weight=0.01`)
  discourages large primitives — the C&W-style trade-off term, here on the *normalized*
  primitive magnitude rather than an L₂ norm.
- `loss.sum().backward(); optimizer.step()` for `steps=40` iterations, `lr=0.1`
  (the committed sweep values).

### Gradient flow / where gradients stop
- Gradients flow: `u,v → σ → p,α → φ (generate) → scaler → victim logits → CE`. All of
  φ's writes are differentiable; the victim is frozen (`requires_grad_(False)`) but
  **input gradients pass through** (`cicids2017d_victims.load_category_victim`).
- Gradients **stop** at: the discrete `round` (only applied at projection/quantization,
  *outside* the loop, `quantize=False` during optimization); the capability masks
  (constant 0/1 multipliers); and the per-flow bounds `p_hi, α_hi` (precomputed
  constants). The identity-row `torch.where` (`:624`) also blocks gradient on no-op rows.

### Discrete projection (after the loop)
```
requested = {p(u), α(v)}                      # continuous, in-box
projected = project_controls(x₀, requested, bounds)   # p←min(round(p),floor(p_hi)); α←min(α,α_hi)
x_adv     = generate(x₀, projected, quantize=True)    # integer µs / byte fields rounded
```
Success is then **re-evaluated on the projected, quantized vector** — the reported ASR
reflects realizable integer primitives, not the continuous relaxation.

### One-iteration pseudocode
```
for t in 1..steps:
    p   = p_hi * sigmoid(u) * pad_active
    a   = 1 + (a_hi - 1) * sigmoid(v) * timing_active
    xadv = generate(x0, {p, a}, quantize=False)      # φ, differentiable
    logits = victim((xadv - center) / scale)
    loss = CE(logits, Benign) + cost_weight*(sigmoid(u)*pad_active + sigmoid(v)*timing_active)
    u,v <- Adam.step(∇_{u,v} loss.sum())
# after loop:
p,a  = project_controls(x0, {p_hi*σ(u)*pad, 1+(a_hi-1)*σ(v)*timing}, bounds)
xadv = generate(x0, {p,a}, quantize=True)
```

### Worked numerical example (padding + timing, illustrative)
Source BruteForce flow: `N_f=5, TL_fwd0=300, mean0=60, min0=40, max0=100,
Fwd IAT Total0=1000µs, Flow Duration0=2000µs, N_b=0`. Suppose the optimizer settles on
`p=10, α=1.1` (within a maximum-evaluated BruteForce box `p_max=91`, `B_D=0.2337`):
```
TL_fwd = 300 + 5·10 = 350 ;  mean = 350/5 = 70 ; min = 50 ; max = 110 ; std unchanged
Fwd IAT Total = 1.1·1000 = 1100 ;  Fwd IAT Mean = 1100/4 = 275
Flow Duration = 2000 + (1100−1000) = 2100 ;  Flow IAT Mean = 2100/4 = 525
Flow IAT Max += (2100−2000) = +100
Fwd Packets/s = 5 / (2100/1e6) = 2380.95 ;  Flow Bytes/s = 350 / 0.0021 = 166 666.7
```
Relative duration change = 100/2000 = 0.05 ≤ 0.2337 (feasible). Represented bytes rose
(350>300) → `TRAFFIC_VOLUME_DECREASED` passes. Ports/counts/flags unchanged (frozen).

---

## 2.6 "Original / non-budget" vs budgeted PrimAttack

**Finding (discrepancy):** there is **no separate committed "original / non-budget"
PrimAttack experiment.** `run(...)` always requires
`budget_name ∈ {restricted, intermediate, maximum-evaluated}` and always applies a
calibrated hard box; "original" in the source docs means the *original source flow*
x₀. See `00_OPEN_ISSUES.md#A2`.

What differs across the (existing) conditions is **only the size of the feasible box**,
via two knobs consumed by `per_flow_bounds`:
- `p_max` — the class padding budget (caps `p_hi`),
- `max_relative_duration_change = B_D` — the class timing budget (caps `α_hi` via
  `1 + B_D·Duration/Fwd IAT Total`).

| Condition | `p_max` | `B_D` | Feasible box |
|---|---|---|---|
| restricted | train p25 | train p25 | smallest |
| intermediate | train p50 | train p50 | medium |
| maximum-evaluated | train p75 | train p75 | **largest evaluated** |
| *hypothetical non-budget* | ∞ (→ envelope p99 only) | large | bounded only by global-train p99 envelope + capability + (DoS/DDoS) rate cap |

**Why an unbudgeted attack could have much higher valid ASR:** removing `p_max`/`B_D`
lets `p_hi`/`α_hi` grow to the p99 envelope, so the sigmoid range widens and the
optimizer can push far larger padding/dilation — more classifier movement toward
Benign. Crucially, **validity is preserved regardless of box size** (φ keeps the
CICFlowMeter algebra intact and validator_v2 rejected *none* of the classifier
successes even at maximum budget), so any extra raw evasion would also be *valid*
evasion. The budget therefore trades attacker success for a *defensible, train-derived
plausibility bound*. The historically high ASRs in the archive were against the
**retired LSTM/serial** victims, not the current MLP/CNN (see `00_OPEN_ISSUES.md#C13`).

**Measured budgeted results** (`docs/primattack_budget_results.md`, mlp+cnn, seed 42,
512 rows/class, pooled N=4064): timing-only 0/4064 at every budget; padding-only and
joint 10/4064 raw=valid=feasible (0.25%) *only* at maximum-evaluated, **SP-ASR 0**
(all 10 are BruteForce, `NOT_FULLY_TESTABLE`). Budgeted PrimAttack essentially does
not evade the current victims.

---

## 2.7 Budget calibration (`primattack_budget.py`)

- **Training-only** (`calibrate :152-287`): reads `X_train_pristine.npy` +
  `y_train_cat.npy` only; `load_calibration` asserts `fit_split=="train"`
  (`:301-302`). `selection_prohibited_inputs` explicitly bars val/test features,
  victim predictions, and adversarial success (`:258-263`).
- **Budget levels = training quantiles** (`_LEVEL_QUANTILES :21`):
  restricted=p25, intermediate=p50, maximum-evaluated=p75.
- **Padding statistic** (`_padding_population :131-137`): population = *positive*
  `Fwd Packet Length Mean` over class rows with forward payload;
  `padding_bytes_per_forward_packet = round(quantile(pop, q))` (`_rounded_empirical_budget
  :146-149`) — discrete bytes, rounded to nearest legal byte, **not** derived from an
  MTU constant or from attack success.
- **Timing statistic** (`_relative_duration_variation :140-143`): population =
  `|Flow Duration − median| / median` over class rows; `max_relative_duration_change =
  quantile(pop, q)`.
- **Per-class** for DoS/DDoS/Recon/BruteForce; also stores a **global-train p99
  envelope** per length/timing feature (`_ENVELOPE_FEATURES`, `:165-168`) shared across
  classes to keep budgets comparable and avoid a class-local structural zero acting as a
  physical ceiling.
- **Semantic thresholds** (`:241-251`): class-train p05 `Flow Packets/s` and
  `Flow Bytes/s` (lower), p99 `Flow Duration` (upper), `rate_retention_required` =
  class∈{DoS,DDoS}, and a `critical_not_testable` list for Recon/BruteForce.

### Frozen calibrated budgets (`artifacts/primattack/budget_calibration.json`)
`padding bytes / max relative duration change`:

| Class (n train) | Restricted (p25) | Intermediate (p50) | Max-eval (p75) | rate-req | pps p05 |
|---|---|---|---|---|---|
| DoS (120,093) | 41 / 0.0427 | 47 / 0.5671 | 54 / 1.2682 | yes | 0.625 |
| DDoS (66,568) | 2 / 0.2220 | 2 / 0.4353 | 3 / 0.6874 | yes | 1.065 |
| Recon (111,311) | 2 / 0.0851 | 2 / 0.2128 | 10 / 0.5319 | no | 22 471.9 |
| BruteForce (4,862) | 11 / 0.0617 | 12 / 0.1199 | 91 / 0.2337 | no | 2.848 |

**Why these are evaluation budgets, not physical realism:** they are *empirical
quantiles of already-observed benign-vs-attack flow variation on the training split*,
chosen so the attacker's edit stays within the range the dataset already exhibits. They
bound how far the attack strays from the training distribution; they do **not** prove a
packet-level attacker could realize exactly that padding/delay while preserving the
attack. The calibration manifest itself notes: *"These are evaluated flow-level
envelopes, not universal physical maxima."* (`:282-284`).

---

## 2.8 PrimAttack variants (what's implemented; what's held constant)

Two optimizers over the **same feasible box** (`OPTIMIZERS`, `:43`):

1. **`optimized`** (Adam / C&W-style) — `optimize_primitives`, §2.5. `method_id =
   "primitive_direct"`.
2. **`random-feasible`** (control) — `random_feasible_primitives :109-120`:
   `p = p_hi·U(0,1)`, `α = 1+(α_hi−1)·U(0,1)`, seeded. `method_id = "primitive_random"`.
   Committed at `outputs/primattack_random_control/` (older commit, MLP-only). Isolates
   "does the optimizer help beyond random sampling inside the box?".

There is **no Prim-PGD** variant in the current code — the only gradient optimizer is
the Adam/sigmoid one above. (Feature-space PGD/C&W are separate baselines, doc 8; VAE
latent→primitive is a separate method, doc 9.)

**Held constant across variants (fair comparison):** the primitive contract
(`PrimitiveSpec`), φ (`generate`), capability inference, per-flow bounds, projection,
quantization, the frozen-feature assertion, validator_v2, the semantic proxy, the
victim checkpoints, and the eligible source rows (`source_id_consistency.json`:
`identical_across_all_configurations = true`). Only `u,v` optimization vs uniform
sampling differs.

---

## 2.9 Assumptions · Limitations · Claims

**Assumptions**: (i) forward padding/timing are the attacker's only levers; (ii) aggregate
flow features suffice to *define* feasibility; (iii) capability gates (forward payload,
≥2 fwd packets) are the right admissibility evidence; (iv) conservative duration/IAT-Max
projections are acceptable stand-ins for the unknown packet order.

**Limitations**: Level-C features (Fwd Act Data Pkts, subflow, bulk, active/idle,
Flow IAT Std/Min) are held constant, so realism at the packet level is unproven
(`NullPacketBackend`); budgets are dataset-empirical, not physical; only p and α are
modeled (no packet injection, no backward-direction edits, no flag changes); success is
near-zero against the current victims.

**Can claim**: a principled, differentiable, validity-preserving primitive-domain
attack with train-only calibrated hard budgets; every adversarial vector is
CICFlowMeter-algebra-consistent and validator_v2-valid; a clean 4-level evaluation
(evasion → validity → feasibility → semantic proxy) with full per-sample provenance.

**Must NOT claim**: packet-level realizability, PCAP validity, preserved malicious
functionality, or that PrimAttack is a *high-success* evasion attack on MLP/CNN. Do not
present a "non-budget PrimAttack" as an executed experiment.
