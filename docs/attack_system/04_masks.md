# 04 — Perturbation Masks

**Source:** `src/attack/masks/{base,cicids2017_distrinet}.py`, `src/attack/masks/__init__.py`.

A *mask* declares, **by feature name**, the attacker-facing role of every one of the 79
features. It is the mechanism the **masked VAE attack ladder** (`run_cicids2017_vae_attacks.py`)
uses to decide what may move. (The primitive attacks in docs 07–08 do not use `DatasetMask`;
they use the realizability model's roles, doc 05. Both express the same physics.)

Two mask *sources* exist and are selectable with `--mask-source`:
- `config` → the hand-verified `DatasetMask` in `masks/cicids2017_distrinet.py`;
- `cff` → the train-mined CFF `.npy` masks (doc 01), selected by `--mask-tier`.

---

## 1. The three feature states (`masks/base.py::FeatureState`)

| State | Meaning | In the attack |
|-------|---------|---------------|
| `PERTURBABLE` | attacker-controlled degree of freedom | receives a direct latent/residual update |
| `DERIVED_EXACT` | recomputed from perturbable/frozen parents by an exact formula | not independently optimized; gradients flow *through* it to its parents |
| `FROZEN` | copied verbatim from the original sample | never changes |

These three must **exactly partition** the feature vector (checked in `resolve`).

---

## 2. `DatasetMask` and `resolve()` — fail-loud wiring

`DatasetMask` is declarative: `perturbable` names, `derived` (topologically ordered
`DerivedFeature`s), and `expected_perturbable_index1` (the 1-indexed positions in the frozen
79-feature contract). `resolve(manifest)` performs strict validation and can **never silently
mis-index** a checkpoint or scaler:

1. dataset name + feature count match the manifest;
2. every declared feature exists, is unique, and no feature is both perturbable and derived;
3. **expected 1-indexed positions match the manifest exactly** — a reordered contract is
   rejected;
4. derived parents resolve, and a derived parent may only be perturbable, frozen, or an
   *earlier* derived feature (no forward/cyclic reference);
5. perturbable ∪ derived ∪ frozen is an exact partition of all 79 features.

The result is a `ResolvedMask` bound to concrete indices/tensors.

---

## 3. The CICIDS2017 config mask (`masks/cicids2017_distrinet.py`)

**9 PERTURBABLE features** (a small forward-direction / timing set), verified at 1-indexed
positions `(4, 7, 9, 10, 12, 23, 25, 26, 27)`:

```
Flow Duration, Total Length of Fwd Packet, Fwd Packet Length Max,
Fwd Packet Length Min, Fwd Packet Length Std,
Fwd IAT Total, Fwd IAT Std, Fwd IAT Max, Fwd IAT Min
```

**7 DERIVED_EXACT features**, each an exact CICFlowMeter identity with **zero TRAIN violations
at `rtol=1e-4`** (verified against `X_train_pristine.npy`, 1,456,265 rows):

| Derived | Formula |
|---------|---------|
| `Fwd Packet Length Mean` | `Total Length of Fwd Packet / max(Total Fwd Packet, 1)` |
| `Fwd Segment Size Avg` | `= Fwd Packet Length Mean` |
| `Fwd IAT Mean` | `Fwd IAT Total / max(Total Fwd Packet - 1, 1)` |
| `Fwd Packets/s` | `Total Fwd Packet / (Flow Duration / 1e6)`  [0 if dur≈0] |
| `Bwd Packets/s` | `Total Bwd packets / (Flow Duration / 1e6)`  [0 if dur≈0] |
| `Flow Packets/s` | `(Fwd+Bwd) / (Flow Duration / 1e6)`  [0 if dur≈0] |
| `Flow Bytes/s` | `(TL_fwd+TL_bwd) / (Flow Duration / 1e6)`  [0 if dur≈0] |

Everything else (backward stats, counts, ports/protocol, TCP flags, window/header fields,
std/min/max aggregates, bulk, subflow, active/idle) is **FROZEN** — it cannot be uniquely
reconstructed from the perturbable aggregates.

### Zero-duration convention

Rates use `_DUR_FLOOR_US = 0.5`: `rate := 0` when `Flow Duration <= 0.5 us`. This reproduces
CICFlowMeter's convention and removes a float32 blow-up: at ~0 duration the RobustScaler scale
for duration is huge (~5e6), so a scaled duration is numerically indistinguishable from zero,
and an unclamped `count / duration` would explode.

---

## 4. `ResolvedMask` — the differentiable dependency stage

The masked attack calls these (all differentiable so classifier gradients reach perturbable
parents):

- `perturbable_mask()`, `frozen_mask()` → bool[79].
- `restore_frozen(raw, raw_original)` → copy frozen columns back from the source row.
- `recompute(raw)` → overwrite every `DERIVED_EXACT` column with its formula, **in declared
  order** (so a derived feature can read a just-recomputed one).
- `apply(raw_generated, raw_original)` = `recompute(restore_frozen(...))` — the full stage
  used in the attack loop (`run_cicids2017_vae_attacks.py::_apply_dependencies`).
- `generator_projector()` → a Layer-0 projector built from a *primitive-ized* manifest (its
  inverse derivations stripped) so the manifest's derivations don't clobber a feature this
  mask treats as directly perturbable.

### Independent sanity checks (evaluation)

- `frozen_violation_mask(raw_adv, raw_original, atol=1e-5, rtol=1e-4)` → True where any frozen
  feature moved. The runner asserts this is **0** (`_assert_frozen_unchanged`).
- `derived_consistency_mask(raw_adv, scale, tol=1e-3)` → True where a derived feature
  disagrees with its own formula (compared in **model space** when a scale is given, matching
  the float32 precision the classifier actually sees).
- `derived_consistency_counts(...)`, `per_feature_perturbation_frequency(...)` → diagnostics.

---

## 5. CFF masks vs the config mask

`load_cff_mask(repo, manifest, class_name, tier)`:
- reads `outputs/cff_cicids2017distrinet/feature_order.json` and asserts it matches the
  manifest name order;
- loads `masks/<class>_<tier>.npy` (`tier ∈ {top10, top25, top50, eligible}`);
- **forces `mask[derived_indices] = False`** so the attacker can never *directly* move a
  DERIVED feature (it can only move it via its parents).

So the config mask is *hand-verified physics* (same 9 perturbable set for every class), while
the CFF mask is *per-class, data-mined* (different features per class, top-25% by default). The
ladder can run either and compare — this is a key ablation.

---

## 6. `get_dataset_mask` registry

`masks/__init__.py::get_dataset_mask(name)` returns the dataset's `DatasetMask` (built via
`cicids2017_distrinet.build_mask()`), so the runner never imports dataset code directly. The
CICIoT2023 mask (`masks/ciciot2023.py`) plugs into the same interface.
