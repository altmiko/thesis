# 02 — How the Constraints Are Discovered

This is the document your report needs for "how the constraints are discovered and how we put
them into the attack." There are **four independent kinds** of constraint in this system, and
**every one is derived from the TRAIN split only** — nothing is hand-authored as a magic
number that was tuned on val/test.

| # | Kind | Discovered from | Enforced by | File(s) |
|---|------|-----------------|-------------|---------|
| A | **Algebraic identities** (`p, delay, shape` -> feature map) | CICFlowMeter extractor semantics + 0% train violation across 400k–600k rows | `generate()` recomputes them; realizability validator re-checks | `attack/realizability/cicids2017.py` |
| B | **Exact derivations** (manifest) | 0% train violation, 100% val pass | Layer-0 projector recomputes | `datasets/cicids2017.py::_DERIVATIONS`, `constraints/layer0.py` |
| C | **Mined density rules** (Layer 2) | fit on 1,456,265 pristine train rows; keep if train violation-rate ≤ 1% | `ConstraintEngine` Layer 2 (validation only) | `old_constraints/cicids2017_distrinet/mined.json` |
| D | **Robust tail bound** (Layer 1) | median/IQR/tau fit on first 200,000 pristine train rows | `ConstraintEngine` Layer 1 (validation) | `constraints/layer1.py::RobustTailBound.fit` |

(CFF — *which* features are perturbable — is a fifth, separate data-driven ingredient; see
[`01_cff_conditional_feature_freedom.md`](01_cff_conditional_feature_freedom.md).)

The recurring principle (from `CLAUDE.md`, "THE REFACTOR PRINCIPLE"): **everything
mined/fitted on TRAIN ONLY; nothing hand-authored; each kind flows through one artifact.**

---

## A. Algebraic identities (the realizability map)

These are the exact CICFlowMeter relationships that let three controls for two physical
operations regenerate their declared dependent features. They are declared in
`attack/realizability/cicids2017.py` and were **mined on the pristine TRAIN split** —
the module docstring states each identity had **0% violation across 400k–600k train
rows**. Examples (exact):

```
Fwd Packet Length Mean = Total Length of Fwd Packet / Total Fwd Packet
Fwd Segment Size Avg   = Fwd Packet Length Mean
Packet Length Mean     = (TL_fwd + TL_bwd) / (Nf + Nb)
Average Packet Size    = Packet Length Mean
Packet Length Variance = pooled sample variance(nf, mf, sf, nb, mb, sb)
Fwd IAT Mean           = Fwd IAT Total / (Nf - 1)
Flow IAT Mean          = Flow Duration / (Nf + Nb - 1)
<dir> Packets/s        = count / (duration_us / 1e6)
Flow Bytes/s           = (TL_fwd + TL_bwd) / (duration_us / 1e6)
```

Two sub-kinds:
- **Extractor identities** (above) — properties of how CICFlowMeter computes a feature.
- **Padding/dilation shift identities** — *threat-model definitions* that follow from the
  attacker's action, e.g. adding `p` bytes to every forward packet gives
  `TL_fwd += Nf*p`, `fwd_min += p`, `fwd_max += p`, and fwd length **std is invariant**
  (a uniform shift preserves standard deviation — this one is labelled `INVARIANT`, i.e.
  *proven*, not merely held constant).

Discovery is verified two ways: (1) the identity holds on train, and (2) the validator
recomputes it *from the adversarial vector* (§ "How they enter the attack" below), so a
generator bug that forgets to update a dependent feature is caught rather than hidden.

The full identity/role tables (Dp/Dt/D/C/R/I/F/Fᶜ) are in
[`05_realizability_model.md`](05_realizability_model.md).

---

## B. Exact manifest derivations (Layer 0)

`datasets/cicids2017.py::_DERIVATIONS` (lines 124–139) declares six identities that had
**zero violations across all 1,456,265 train rows and 100% pass across all 312,058
validation rows**:

```python
"Packet Length Variance"       = square(Packet Length Std)
"Average Packet Size"          = identity(Packet Length Mean)
"Fwd Segment Size Avg"         = identity(Fwd Packet Length Mean)
"Bwd Segment Size Avg"         = identity(Bwd Packet Length Mean)
"Total Length of Fwd Packet"   = product(Total Fwd Packet, Fwd Packet Length Mean)
"Total Length of Bwd Packet"   = product(Total Bwd packets, Bwd Packet Length Mean)
```

Each is expressed as a **generic derivation tag** (`identity` / `square` / `product`) with
named parents. `Layer0Projector` (constraints/layer0.py, lines 44–56, 89–98) reads these
from the manifest and recomputes the target from post-projection parents. Only three tags
exist and each validates its parent arity — anything else raises.

> Note: these manifest derivations and the realizability map (A) are two *views* of the same
> physics. The masked attack uses the manifest/Layer-0 view; the primitive attacks use the
> realizability-model view. `masks/base.py::generator_projector()` deliberately *strips* the
> manifest's inverse derivations when a mask treats a feature as directly perturbable, so the
> two views never fight.

---

## C. Mined density rules (Layer 2) — `mined.json`

This is the file people usually mean by "the mined constraints." It lives at
`old_constraints/cicids2017_distrinet/mined.json` and is loaded verbatim by the engine
(`constraints/layer2.py::load_layer2` → `registry.build_constraint`).

### The mining procedure

The rule *format* is designed so a mining script can emit the file with no engine changes
(`layer2.py` docstring). The procedure (recorded in the file's `mining` block and in
`docs/constraint_miner.md`):

1. **Enumerate candidate relations** of the generic types that already exist in Layer 1:
   ordered triples (`MonotoneNondecreasing`), product/identity equalities (`ProductEquality`),
   and spread bounds (`HalfRangeBound`).
2. **Instantiate each candidate** against the manifest feature names.
3. **Measure the per-rule violation rate on the pristine TRAIN array**
   (`X_train_pristine.npy`, `fit_rows = 1,456,265`).
4. **Keep a rule iff its train violation rate ≤ ε = 0.01** (`keep_if_violation_rate_lte`).
5. **Serialize survivors** in the Layer-2 envelope (`dump_layer2`).

### What survived and what was rejected (from the file's `mining` block)

```json
"fit_split": "train", "fit_rows": 1456265, "source": "X_train_pristine.npy",
"keep_if_violation_rate_lte": 0.01, "product_rtol": 0.05, "monotone_tol": 1e-06,
"retained_rules": 14, "retained_max_violation_rate": 0.0,
"rejected_half_range_violation_rate_range": [0.02352..., 0.42790...]
```

- **14 rules retained**, every one with **0.0 train violation rate**.
- **All `HalfRangeBound` (Std ≤ half-range) candidates were rejected** — their train
  violation rates ranged from **2.35% to 42.79%**, all above ε, so they are *absent* from
  the file. This is the soundness/completeness accounting the target architecture asks for:
  a rule is kept only if it is (almost) never violated by real training traffic.

### The 14 retained rules

**8 `MonotoneNondecreasing` (min ≤ mean ≤ max), `tol = 1e-6`:**
`fwd_packet_length_order`, `bwd_packet_length_order`, `packet_length_order`,
`flow_iat_order`, `fwd_iat_order`, `bwd_iat_order`, `active_order`, `idle_order`.

**6 `ProductEquality` (target ≈ ∏ factors), `rtol = 0.05`:**

| name | identity |
|------|----------|
| `packet_variance_eq_std_squared` | `Packet Length Variance = Packet Length Std²` |
| `avg_packet_size_eq_packet_mean` | `Average Packet Size = Packet Length Mean` |
| `fwd_segment_avg_eq_fwd_mean` | `Fwd Segment Size Avg = Fwd Packet Length Mean` |
| `bwd_segment_avg_eq_bwd_mean` | `Bwd Segment Size Avg = Bwd Packet Length Mean` |
| `fwd_total_eq_count_mean` | `Total Length of Fwd Packet = Total Fwd Packet × Fwd Packet Length Mean` |
| `bwd_total_eq_count_mean` | `Total Length of Bwd Packet = Total Bwd packets × Bwd Packet Length Mean` |

The `ProductEquality.validate` test is a **relative** one:
`|target - ∏factors| / (|target| + 1) < rtol` (`layer1.py` lines 126–135).

### Serialized format

```json
{
  "schema_version": "1.0",
  "dataset": "cicids2017_distrinet",
  "mining": { ... provenance of the mining run ... },
  "constraints": [ { "type": "...", "name": "...", "params": { ... } }, ... ]
}
```

`load_layer2` checks the declared `dataset` matches the manifest, rebuilds each rule via the
registry, and tags it `layer = 2` regardless of its underlying class. The CICIoT2023 sibling
file `old_constraints/ciciot2023/mined.json` uses the same format.

---

## D. Robust tail bound (Layer 1) — fit at runtime

Unlike C (a static file), the Layer-1 constraint is **fit each run on TRAIN rows**. In the
A4 engine used by the attacks (`experiments/ablations.py` lines 121–125):

```python
RobustTailBound.fit(manifest, layer1_fit_x_raw, tau=None, coverage=0.99)
```

- `layer1_fit_x_raw` = **first 200,000 rows** of `X_train_pristine.npy`
  (`run_*_attack.py`: `raw_train[:200000]`).
- Per feature: `median`, `IQR = Q75 - Q25`; scale falls back to `std` if `IQR == 0`, and to
  `1.0` only for truly constant columns (avoids treating sparse-but-variable columns as
  all-outlier).
- `tau=None` + `coverage=0.99` means **tau is discovered from the data**: it is set to the
  0.99 quantile (`method="higher"`) of the per-row maximum robust z-score
  `max_i |x_i - median_i| / scale_i`. So the bound is calibrated to *cover 99% of train
  rows*, not hand-set. (`build_engine`'s standalone default is `tau=6.0`, but the attack path
  always uses the coverage-calibrated form.)
- Validation: a row passes iff **every** feature's robust z-score ≤ `tau`.

`RobustTailBound` is (de)serializable (`to_config`/`from_config`) so a fitted bound can be
persisted, but the attack refits it deterministically from train each run.

---

## How the discovered constraints enter the attack

There are two enforcement roles, kept strictly separate (`constraints/base.py` docstring):

1. **Generation-time (hard, by construction).**
   - The realizability model's `generate()` **recomputes** every A-identity from the
     projected primitives — the adversarial vector is *built* consistent, not penalized into
     consistency.
   - Layer-0 `project()` clamps value-type domains, restores frozen features from the source
     row, and recomputes B-derivations. It is differentiable (clamps have sub-gradients), so
     it can sit inside a gradient attack.
2. **Evaluation-time (independent validation).**
   - The A4 `ConstraintEngine.validate(adv_raw)` re-checks Layer 0 (domains) + Layer 1 (robust
     tail, D) + Layer 2 (mined density, C) and returns `pass_l0_l1_l2` per sample. This is
     the `mined_valid` mask.
   - The realizability validator independently re-derives the A-identities from the
     adversarial vector (so a generator bug cannot pass).
   - PAVE independently checks raw ranges + integer/binary types.

Crucially, **the density rules (C, D) and PAVE are validators, not generator penalties, in
the primitive/latent attacks.** The generator only guarantees structural realizability;
whether the result also lands in the mined density envelope and passes PAVE is *measured*,
not enforced. That is the thesis "separation guarantee": a high realizability rate does not
trivially imply high mined/PAVE validity.

For how these masks combine into `strict_valid`, IDR, and True-IDSR, see
[`06_validators.md`](06_validators.md).
