# Mined constraint rules (Layer 0 / 1 / 2)

How the repository derives, serializes, and enforces **feature-relationship constraints**
from data rather than hand-authoring them — the `ConstraintMiner → MinedValidator` half
of THE REFACTOR PRINCIPLE (constraints C in `CLAUDE.md`). Everything numeric is fit on the
**train split only**; validation/test are checked, never fitted on.

- **Package:** `src/constraints/` — `base.py`, `layer0.py`, `layer1.py`, `layer2.py`,
  `engine.py`, `registry.py`.
- **Serialized rule sets:** `old_constraints/<dataset>/mined.json`
  (`old_constraints/ciciot2023/mined.json`, `old_constraints/cicids2017_distrinet/mined.json`).
- **Sibling (perturbability mining, not rules):**
  `src/preprocessing/conditional_feature_freedom.py` (§ CFF).
- **Tests:** `src/constraints/tests/test_constraints.py`, `test_toggles.py`.

## The three-layer model

Constraints operate in **raw feature space** (post inverse-transform), the same space the
independent validators use, and resolve features by **manifest name**, never hardcoded
index (`base.py`). Two roles are kept strictly distinct per constraint:

- **generation-time:** Layer 0 is a hard *projector* `P0`; Layer 1/2 contribute
  differentiable soft *penalties* `C1`/`C2` for losses / attack objectives.
- **evaluation-time:** every constraint also exposes `validate(x_raw) -> (N,) bool` so
  validity is checked **independently** of whether the generator projected — never by
  trusting the generator's own projection.

| layer | nature | source of truth | code |
|---|---|---|---|
| **0** | hard, semantically inviolable | datatype / representation / exact identity (manifest `value_type`, derivations) | `layer0.py` |
| **1** | soft, generic algorithm + train-fit params | universal algorithm; medians/IQRs/tolerances fit on TRAIN | `layer1.py` |
| **2** | soft, dataset/extractor-specific or **mined** | per-dataset `old_constraints/<dataset>/mined.json` | `layer2.py` + `registry.py` |

The distinction that makes Layers 1/2 "mined": the **algorithm is dataset-independent**;
only the numeric parameters (and which features a relation references) come from data.

## Layer 0 — hard projector (`Layer0Projector`)

Not mined and not a penalty — structural feasibility from the manifest. At construction it
reads value-type index groups from the manifest and, for `bounded_continuous`, the
`lower`/`upper` arrays. `project(x_raw, x_source, mutable_mask)`:

1. clamp `probability → [0,1]`, `binary → [0,1]`, `positive_continuous`/`integer_count →
   [0,∞)`, `bounded_continuous → [lower, upper]` (all differentiable via clamp subgradients);
2. **immutable preservation** — where `mutable_mask` is 0, restore the value from
   `x_source` (attack setting: frozen features cannot move);
3. **exact derived recompute** — for each manifest `derived` feature with a declared
   `derivation`, recompute the target from its (post-projection) parents. Generic op tags:
   `identity` (copy parent), `square` (parent²), else product of parents.

`validate` mirrors these domains with a tolerance (`tol=1e-4`) and returns a per-sample
mask; `soft_penalty` (`C0`) is the mean pre-projection domain violation, a differentiable
regularizer to keep the decoder in-domain before projection.

## Layer 1 — generic soft constraints, train-fit parameters (`layer1.py`)

Four deliberately-few, strongly-justified constraint classes. Each exposes
`penalty` (differentiable ≥0), `validate` (per-sample bool), and `to_config` /
`from_config` for serialization.

### `RobustTailBound` — per-feature robust tail

The only class with a `fit` classmethod (the true "miner" for Layer 1). For the requested
features it fits, **on train**:

```
median = median(X[:, cols])
IQR    = Q75 - Q25
scale  = IQR            where IQR > eps
       = std            where IQR == 0 but std > 0   (sparse-but-variable columns)
       = 1              where truly constant
z_i    = |x_i - median_i| / (scale_i + eps)
```

`penalty = relu(z - tau).mean()`; `validate = (z <= tau).all(dim=1)`. `tau` defaults to
`4.0`. If `tau=None`, it is **calibrated on train** to a coverage quantile: `tau =
quantile(max_z_per_row, coverage, method="higher")` (default `coverage=0.99`), i.e. the
smallest bound admitting 99% of training rows. All parameters (median, IQR/scale, tau)
serialize into the config, so evaluation reuses the exact train-fit values.

### Cross-feature relations (parameters = which features)

These carry no fitted statistics beyond tolerances; "mining" them means discovering which
feature triples/products satisfy the relation on train (§ Layer 2 mining procedure).

- **`MonotoneNondecreasing(features)`** — ordered features non-decreasing (e.g.
  `Min ≤ AVG ≤ Max`). `validate`: consecutive diffs `≥ -tol` (`tol=1e-6`). `penalty`:
  scale-normalized `relu(prev - next)`.
- **`ProductEquality(target, factors, rtol)`** — `target ≈ prod(factors)` in relative
  terms: `|target - prod| / (|target| + 1) < rtol` (`rtol=0.05`). Encodes identities like
  `Tot = Number·AVG`, `Variance = Std·Std`, `AvgPacketSize = PacketLengthMean·1`,
  `TotalLenFwd = TotalFwdPkts·FwdPktLenMean`.
- **`HalfRangeBound(value, lo, hi, rtol)`** — spread bound
  `value ≤ 0.5·(hi − lo)·(1+rtol)` (e.g. `Std` vs `Min`/`Max`). `validate`: `excess ≤ 0`.

## Layer 2 — dataset-specific / mined rule sets (`layer2.py` + `registry.py`)

Layer 2 rules are **not baked into code**. They are a serializable list of
`{"type", "name", "params"}` entries, rebuilt against a manifest through the registry.

### Serialized format (`old_constraints/<dataset>/mined.json`)

```json
{
  "schema_version": "1.0",
  "dataset": "cicids2017_distrinet",
  "mining": { … provenance metadata … },
  "constraints": [
    {"type": "MonotoneNondecreasing", "name": "fwd_packet_length_order",
     "params": {"features": ["Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Max"], "tol": 1e-06}},
    {"type": "ProductEquality", "name": "packet_variance_eq_std_squared",
     "params": {"target": "Packet Length Variance", "factors": ["Packet Length Std", "Packet Length Std"], "rtol": 0.05}}
  ]
}
```

- `load_layer2(source, manifest)` accepts a path, dict, or list. It validates that a
  declared `dataset` matches `manifest.dataset_name` (raises otherwise), builds each rule
  via `build_constraint`, and **tags every result `layer = 2`** regardless of the
  underlying class (a `MonotoneNondecreasing` used as a dataset rule reports as Layer 2).
- `dump_layer2(constraints, dataset_name)` emits the same envelope
  (`schema_version`, `dataset`, `constraints=[c.to_config() …]`), so a mined-in-memory rule
  list round-trips to disk (`test_layer2_load_and_engine_hierarchical`).

### Registry (`registry.py`)

`CONSTRAINT_REGISTRY` maps the `type` string to the class:
`RobustTailBound`, `ProductEquality`, `MonotoneNondecreasing`, `HalfRangeBound`.
`build_constraint(manifest, cfg)` looks up the type and calls its `from_config`.
`register_constraint(cls)` adds new types **without editing the generic engine** — this is
the extension point that lets a mining script emit new rule kinds.

## The mining procedure (how `mined.json` is produced)

The rule format is designed so that "an automatic constraint-mining script can overwrite
this file in-place with the same format, with no code changes" (`layer2.py` docstring).
Provenance is recorded in the file's `mining` block. The CICIDS rule set documents the
exact procedure that was run:

```json
"mining": {
  "fit_split": "train",
  "fit_rows": 1456265,
  "source": "X_train_pristine.npy",
  "keep_if_violation_rate_lte": 0.01,
  "product_rtol": 0.05,
  "monotone_tol": 1e-06,
  "retained_rules": 14,
  "retained_max_violation_rate": 0.0,
  "rejected_half_range_violation_rate_range": [0.0235…, 0.4279…]
}
```

The mining algorithm (per THE REFACTOR PRINCIPLE):

1. **Enumerate candidate relations** from the manifest's semantic feature families —
   `{Min, Mean, Max}` triples per measurement group → `MonotoneNondecreasing`; known
   flow-statistic identities (`Total = Count·Mean`, `Variance = Std²`, `Avg… = …Mean`) →
   `ProductEquality`; spread-vs-range pairs → `HalfRangeBound`.
2. **Measure the empirical violation rate of each candidate on the train split only**,
   using that constraint's own `validate` at the retained tolerance (`product_rtol=0.05`,
   `monotone_tol=1e-6`).
3. **Keep a rule iff its train violation rate ≤ ε** (`keep_if_violation_rate_lte=0.01`).
   For CICIDS, 14 order/product rules were retained at a **maximum violation rate of 0.0**
   (zero train violations); all `HalfRangeBound` candidates were **rejected** (violation
   rates 2.35%–42.79% > ε) and do not appear in the file.
4. **Serialize** the survivors in the Layer-2 envelope via `dump_layer2`.

The soundness/completeness/F1 accounting the target architecture calls for is exactly
this per-rule train violation rate and the kept/rejected split recorded in `mining`.

The CICIoT2023 rule set is currently **hand-seeded** (`note` field) but in the identical
format — four soft rules (`Min ≤ AVG ≤ Max`, `Std` half-range, `Tot sum = Number·AVG`,
`Variance = Std²`). Exact identities are kept as Layer-2 **soft** checks pending promotion
to Layer-0 **derived recompute** after clean-data exactness verification (this is exactly
what happened for CICIDS: `cicids2017.py` promotes six exact CICFlowMeter identities to
`derived` after zero train violations + 100% validation pass, so Layer 0 recomputes them).

## Composition & toggling (`ConstraintEngine`, `engine.py`)

`ConstraintEngine(manifest, layer0, layer1, layer2, active_layers)` composes the layers
with a **runtime on/off selector** so one built engine serves every ablation
(`{0}`, `{0,1}`, `{0,1,2}`, `{0,2}`, …). `parse_layers` accepts `"012"`, `"0,1,2"`,
`[0,1,2]`, or `None` (→ all present layers). The active set is honored consistently:

- `project(x_raw, x_source, mutable_mask)` — Layer-0 hard projection only if 0 active;
- `penalty(x_raw, w1, w2)` — sums `c1` over Layer 1 (if active), `c2` over Layer 2 (if
  active), returns `{c1, c2, total = w1·c1 + w2·c2, per_constraint}`;
- `validate(x_raw)` — an **inactive layer contributes an all-pass mask**, so hierarchical
  rates are exactly the active layers. Returns per-sample masks `pass_l0`,
  `pass_l0_l1`, `pass_l0_l1_l2` and their rates `rate_l0`, `rate_l0_l1`,
  `rate_l0_l1_l2`, plus a `per_constraint` dict.

`build_engine(manifest, layers, layer1_fit_x_raw, layer1_features, layer1_tau,
layer2_source)` constructs an engine with **only** the requested layers: Layer 1 requires
train raw features (fits a `RobustTailBound`), Layer 2 requires a `layer2_source`
(path/dict/list). This is the standard way callers wire the mined rules.

`test_constraints.py` asserts real clean data passes the Layer-0 domain "essentially
always," and `test_toggles.py` verifies the toggle semantics.

## Who consumes the mined rules

Downstream code loads **only** the serialized rule set (never re-hardcodes rules):

- `evaluation/run_pave_validity.py:_build_mined_checker` → Layer-2-only engine as the
  optional `mined_checker` in the PAVE-style strict conjunction (see
  `docs/pave_validity.md`).
- `experiments/ablations.py` (A4 "+ Layer 2 dataset/mined constraints"),
  `attack/run_cicids2017_vae_attacks.py`, and the Stage-B / manifold-loss tests all load
  `old_constraints/<dataset>/mined.json` via `load_layer2`.

## CFF — perturbability mining (sibling, not rule mining)

`src/preprocessing/conditional_feature_freedom.py` mines the **perturbability tier**
(mutability), the *other* half of the manifest, not the C-rules above. It is included here
because it is the repository's other train-only "miner." Per feature `X_i` and class `c`,
LightGBM predicts `X_i` from the other features with out-of-fold predictions, and:

```
r_i    = X_i - f_i(X_-i)                               # OOF residual
W_r(i) = Q0.95(r_i) - Q0.05(r_i)                       # robust residual width
W_x(i) = Q0.95(X_i) - Q0.05(X_i)                       # robust natural width
CFF_i  = clip(W_r(i) / (W_x(i) + eps), 0, 1)           # conditional freedom
```

Higher CFF = more conditional freedom; lower = more structural constraint/predictability.
Class-conditional ranks feed the top-k selection fractions used to build per-class
mutability masks. Fit on **train artifacts only** (`X_train.npy`, `y_train_cat.npy`);
validation/test are never opened. CFF explicitly does **not** prove attacker accessibility,
packet-level mutability, or causal controllability — those need separate problem-space
validation. See `docs/CFF.md` for full detail.

## Invariants

- **Train-only fit.** Every statistic/threshold (median, IQR, tau, kept-rule set) comes
  from the train split; a rule survives only if its **train** violation rate ≤ ε.
- **Manifest-addressed.** Constraints resolve features by name through the manifest; no
  hardcoded indices, no dataset code inside the generic engine.
- **Generation ≠ evaluation.** `project`/`penalty` (generation) and `validate`
  (evaluation) are independent; validity never trusts the projector.
- **Serializable & extensible.** Rules live in `old_constraints/<dataset>/mined.json`; new
  rule kinds register via `register_constraint` with no engine edits.
