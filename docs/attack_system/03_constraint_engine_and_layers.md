# 03 — The Constraint Engine and Its Layers

**Source:** `src/constraints/{base,layer0,layer1,layer2,registry,engine}.py`.
**Serialized rules:** `old_constraints/<dataset>/mined.json`.

The constraint engine is the machinery that *applies* the discovered constraints (doc 02). It
implements a **three-layer hierarchy** and keeps **generation** (projection + soft penalties)
strictly separate from **evaluation** (independent per-sample validation). All constraints
operate in **raw feature space** (post inverse-transform), resolve features **by manifest
name** (never hard-coded indices), and expose both a `penalty` (differentiable, for losses)
and a `validate` (boolean per-sample mask, for evaluation).

---

## 1. The layer taxonomy (`constraints/base.py`)

| Layer | Nature | Source of parameters | Role |
|-------|--------|----------------------|------|
| **0** | Hard, semantically inviolable | representation / datatype / exact identity — *no* observed ranges | **Projector** `P0` (generation) + domain validator |
| **1** | Generic soft constraints | universal algorithm; medians/IQR/tau **fit on TRAIN** | soft penalty `C1` + validator |
| **2** | Dataset-specific / **mined** | per-dataset `mined.json` | soft penalty `C2` + validator |

Every `Constraint` subclass implements:
- `penalty(x_raw, ctx) -> scalar >= 0` — differentiable, used in generation losses.
- `validate(x_raw, ctx) -> bool[N]` — per-sample pass mask, used in evaluation.
- `report(...)` → `ConstraintReport(name, layer, per_sample, pass_rate)`.
- `to_config()` / `from_config()` — for the serializable ones (Layer 1/2 kinds).

---

## 2. Layer 0 — the hard projector (`layer0.py`)

`Layer0Projector` is built directly from the manifest. It groups features by `value_type`:

- `probability` → clamp to `[0, 1]`
- `binary` → clamp to `[0, 1]` (straight-through handled by the caller)
- `positive_continuous` + `integer_count` → clamp to `[0, ∞)`
- `bounded_continuous` → clamp to `[lower, upper]` (semantic bounds, e.g. ports ≤ 65535,
  protocol ≤ 255, window bytes ≤ 65535)
- `real` → untouched

`project(x_raw, x_source, mutable_mask)`:
1. clamp all value-type domains;
2. **immutable preservation** — `x = where(mutable_mask, x, x_source)`: any frozen feature is
   copied verbatim from the source sample (the attack setting);
3. **exact derived recompute** — for every manifest derivation (`identity`/`square`/`product`)
   overwrite the target from its post-projection parents.

`validate(x_raw, tol=1e-4)` returns per-sample "all features within their Layer-0 domain."
`soft_penalty(x_raw)` is the differentiable pre-projection regularizer `C0` (mean domain
violation), used to encourage the decoder to produce in-domain values *before* projection.

**Layer 0 is structural, not statistical.** By construction it makes Layer-0 validity ~100%.

---

## 3. Layer 1 — generic soft constraints (`layer1.py`)

Four generic, dataset-independent constraint *algorithms*; only their numeric parameters or
referenced features come from data. The spec explicitly says "do not over-engineer Layer 1."

| Class | Rule | `validate` test |
|-------|------|-----------------|
| `RobustTailBound` | per-feature robust tail | `|x - median| / IQR ≤ tau` for all features |
| `ProductEquality` | `target ≈ ∏ factors` | `|t - ∏f| / (|t| + 1) < rtol` |
| `MonotoneNondecreasing` | ordered features non-decreasing | `diff(features) ≥ -tol` |
| `HalfRangeBound` | spread bound | `value ≤ 0.5·(hi - lo)·(1+rtol)` |

`RobustTailBound.fit` is the only fitted one; its calibration (train median/IQR, `tau=None` +
`coverage=0.99`) is described in doc 02 §D. The other three carry only their feature lists and
tolerances and are what the miner *instantiates* to produce Layer-2 rules.

The soft `penalty` of each is a mean of ReLU'd violations normalized by feature scale, so it
is bounded and differentiable — safe to add to a VAE ELBO or an attack objective.

---

## 4. Layer 2 — dataset-specific / mined (`layer2.py`, `registry.py`)

Layer 2 is **not new code per dataset** — it is a serialized list of Layer-1-kind rules
loaded from `mined.json`. `load_layer2(source, manifest)`:
- accepts a path, dict, or list;
- validates the declared `dataset` matches the manifest;
- rebuilds each rule through `registry.build_constraint` (`CONSTRAINT_REGISTRY` maps a `type`
  string → class → `from_config`);
- tags each rule `layer = 2` regardless of its underlying class.

`dump_layer2(constraints, dataset_name)` emits the same envelope, so an in-memory mined rule
list round-trips to disk. New rule kinds register via `register_constraint` with **no engine
edits** — this is the extensibility that lets the miner grow the rule set.

For CICIDS2017 the file contains the **14 mined rules** (8 monotone + 6 product) documented in
doc 02 §C.

---

## 5. The engine (`engine.py`)

`ConstraintEngine(manifest, layer0, layer1, layer2, active_layers)` composes the layers with
**per-run on/off toggles** (`active_layers`, e.g. `{0}`, `{0,1}`, `{0,1,2}`, `{0,2}`). The
active set is honored consistently:

- `project(x_raw, x_source, mutable_mask)` — applies Layer-0 projection **only if 0 is
  active** (else returns input unchanged).
- `penalty(x_raw, ctx, w1, w2)` — `c1` counts only if layer 1 active, `c2` only if 2 active;
  returns `{c1, c2, total = w1·c1 + w2·c2, per_constraint}`.
- `validate(x_raw, ctx)` — an **inactive layer contributes an all-pass mask**, so the
  hierarchical rates reflect exactly the active layers. Returns:

```python
{
  "active_layers": [...],
  "layer0", "layer1", "layer2":        # per-layer bool[N]
  "pass_l0":        l0,
  "pass_l0_l1":     l0 & l1,
  "pass_l0_l1_l2":  l0 & l1 & l2,       # <- this is the "mined_valid" mask in the attacks
  "rate_l0", "rate_l0_l1", "rate_l0_l1_l2",
  "per_constraint": { name: bool[N], ... },
}
```

**Generation and evaluation are independent:** `validate` runs its own checks and never trusts
that the projector was applied. This is what makes the density validity an honest measurement.

`build_engine(manifest, layers, layer1_fit_x_raw, ...)` is a convenience constructor that
builds only the requested layers and marks them active; Layer 1 requires TRAIN raw features,
Layer 2 requires a rule source.

---

## 6. How the attacks build the engine (A4)

Both primitive attacks build the engine through the ablation ladder:

```python
engine = build_ablation(
    "A4", adapter, encoder_input_transform="asinh",
    layer1_fit_x_raw=raw_train[:200000],                      # TRAIN-only Layer-1 fit
    layer2_path=repo/"constraints"/adapter.name/"mined.json", # mined Layer-2 rules
).engine
```

`A4 = typed decoder + Layer0 + Layer1 + Layer2` (`experiments/ablations.py::PRESETS`). The
attack then calls `engine.validate(adv_raw)["pass_l0_l1_l2"]` to get the per-sample
`mined_valid` mask. The full A0–A6 ladder is documented in
[`09_attack_pipeline_and_rows.md`](09_attack_pipeline_and_rows.md).

Stage-A VAE *training* uses the same engine class but with `active_layers=set()` — the VAE is
trained with **no active constraint penalty** (the physics loss is dormant in this thesis).
The engine there only carries the Layer-0 projector for its manifold-ELBO plumbing.
