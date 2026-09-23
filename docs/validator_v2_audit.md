# validator_v2 — Audit of the current validation stack (Phase A)

Read-only audit of every piece of validation logic in the repository, why it is
untrustworthy as a single thesis artifact, and what `validator_v2`
(`validation/`) retains, replaces, or deprecates. **No existing validator was
modified**; the current code is preserved for comparison (see §7).

Scope of this rebuild: **CICIDS2017 (DistriNet-corrected CICFlowMeter)**. The
CICIoT2023 stack is documented here only to show what is being separated out.

---

## 1. Current files that contribute validation logic

### CICIDS2017 stack (the "current" validator for this dataset)
| Path | Role | Provenance clarity |
|---|---|---|
| `src/datasets/feature_manifest.py` | SCHEMA source of truth: `FeatureSpec` (value_type, bounds, derivation), ordered `FeatureManifest` | clear |
| `src/datasets/cicids2017.py` | SCHEMA `_TYPING` + EXTRACTOR `_DERIVATIONS` (6 exact identities) | clear (0% train / 100% val) |
| `old_constraints/cicids2017_distrinet/mined.json` | MINED Layer-2 rules: 14 (8 monotone order + 6 product identities) | artifact clear, **generating script absent** |
| `src/constraints/{engine,layer0,layer1,layer2,registry,base}.py` | 3-layer ConstraintEngine that *applies* the mined rules; Layer-1 `RobustTailBound.fit` (train tail) | clear |
| `src/evaluation/pave_style_validator.py` | Independent raw-space range/type validator; mixes SCHEMA + MINED (train min/max fallback) + PROTOCOL (ports/ttl/proto table) into one `valid_mask` | clear but **mixed** |
| `src/evaluation/run_pave_validity.py` | PAVE CLI + `_build_mined_checker` (Layer-2-only engine) | clear |
| `src/attack/realizability/{validator,base,cicids2017}.py` | EXTRACTOR internal categorized realizability (algebraic/packet/timing/rate/discreteness/frozen) | clear (independent by design) |
| `src/attack/masks/{cicids2017_distrinet,base}.py` | EXTRACTOR perturbation mask + derived-feature recompute | clear |
| `src/experiments/ablations.py` | wires VAE + ConstraintEngine (A4 = +Layer2 mined) | clear |
| `src/attack/run_cicids2017_*.py` (primitive, input_baseline, vae_latent_attack, latent_variants, vae_attacks) | attack runners that assemble `strict_valid` | clear |
| `scripts/audit_results.py` (+ root copy) | independent metric re-computation from NPZ | clear |
| `src/preprocessing/conditional_feature_freedom.py` | CFF perturbability miner (sibling, mutability not rules) | clear |

### Legacy CICIoT2023 stack (hand-coded, mixed — deprecate)
| Path | Role | Provenance clarity |
|---|---|---|
| `src/attack/validator.py` | G-rule structural validator (`R_min_leq_max`, `R_avg_in_range`, `R_var_eq_std_sq`, structural min/max) | **unknown / hand-set** (`FLOAT_TOL=0.01`, `VALID_PROTOCOLS`) |
| `src/vae/physics_validator.py` | P1–P8 physics plausibility (`PROTOCOL_MIN_HEADER`, flag pairs); several checks disabled | hand-authored |
| `src/vae/calibrate_physics_validator.py` | physics threshold calibration; inlines duplicate `FEATURE_NAMES` | hand |
| `src/attack/latent_infra.py` | `ProtocolValidator`, hardcoded `PerturbationMask` tiers | hand (CLAUDE.md smell) |
| `src/attack/constrained_input_baselines.py` | `VAEConstraintProjection` (generation-time projection used as validity) | hand |
| `src/vae/schema.py` | `PROTOCOL_ALLOWLIST`, `raw_postprocess`, `get_partition` | hand (CLAUDE.md smell) |

### PAVE provenance
- Repo-root `PAVE_artifact/` is **empty**. The real upstream code is at
  `old_root_files/old_p3_expt/external/PAVE_artifact/` (`code/attacks/constraints.py`,
  `code/pave_audit.py`).
- `docs/pave_validity.md` states PAVE was *"adapted as an evaluation baseline, not
  a claimed methodological novelty"* — inspired by `constraints.py` /
  `pave_audit.py`, observational-only (never repairs), train-only fallbacks.

---

## 2. Data flow (CICIDS2017 attack → validator)

Every CICIDS2017 runner assembles `strict_valid` from independent evaluators plus
a **separate** realism gate:

```
adv_raw ─┬─ PAVEStyleValidator.validate_batch()          → pave_valid   (SCHEMA+MINED+PROTOCOL, mixed)
         ├─ ConstraintEngine.validate()["pass_l0_l1_l2"] → mined_valid  (Layer0 schema + Layer1 tail + Layer2 mined)
         ├─ RealizabilityValidator.validate().valid       → realizable   (EXTRACTOR identities/ordering/discreteness)
         └─ _idr_mask() (val-anchored Mahalanobis)         → in_distribution  (PLAUSIBILITY, kept separate)

strict_valid = pave_valid & mined_valid & realizable         (primitive / input / latent-primitive)
             = pave_valid & mined_valid & mask_valid          (latent raw/masked variants)
IDR (realism) reported as an INDEPENDENT metric.
```

Exception / main mixing point: `run_cicids2017_vae_attacks.py` sets
`joint_valid = engine.pass_l0_l1_l2`, which folds **Layer-1 `RobustTailBound`**
(a distributional tail bound = plausibility) **into "validity"**, then computes
`True_IDSR = evasion & joint_valid & in_distribution`.

---

## 3. Existing rule sources & provenance

| Source | Where | Provenance |
|---|---|---|
| SCHEMA / type | `feature_manifest.py`, `cicids2017.py::_TYPING`, `layer0.py` clamps, PAVE type grid | typed contract — clear |
| MINED / empirical | `mined.json`, `layer1.py::RobustTailBound.fit`, CFF, PAVE train min/max fallback | artifact clear; **no re-runnable mining script** |
| EXTRACTOR / formula | `cicids2017.py::_DERIVATIONS`, `masks/cicids2017_distrinet.py`, `realizability/cicids2017.py`, legacy `attack/validator.py::R_var_eq_std_sq` | extractor definitions — clear but duplicated |
| PROTOCOL / domain | PAVE `_EXACT_UNIVERSAL`/`_DATASET_INTEGER_OVERRIDES`, legacy `VALID_PROTOCOLS`, `physics_validator.PROTOCOL_MIN_HEADER`, `vae/schema.PROTOCOL_ALLOWLIST` | RFC/port knowledge — hand, scattered |
| PLAUSIBILITY | `_idr_mask` (Mahalanobis), `layer1.RobustTailBound`, `physics_validator` | val-anchored / hand |

**Old miner status:** `docs/constraint_miner.md` documents the mining *procedure*
that produced `mined.json` (enumerate semantic-family candidates → measure train
violation rate → keep if ≤ ε=0.01 → `dump_layer2`), and the file's `mining` block
records provenance (1,456,265 train rows; 14 retained at 0.0 violation; all
`HalfRangeBound` candidates rejected at 2.35%–42.79%). **But no standalone mining
script exists in the tree** — only `dump_layer2` (serializer, tests only) and
`RobustTailBound.fit`. This is exactly the "the original miner was deleted"
situation: the artifact and its format survive, the generator does not.

---

## 4. What is understandable vs unknown provenance

- **Understandable:** the manifest schema, the 6 extractor `_DERIVATIONS`
  (verified 0% train / 100% val), the `mined.json` 14 rules (format + `mining`
  block), the realizability categories, `_idr_mask`.
- **Unknown / opaque provenance:** legacy `attack/validator.py` `FLOAT_TOL=0.01`
  and `VALID_PROTOCOLS`; `physics_validator.py` disabled checks and RFC header
  constants; PAVE `_EXACT_UNIVERSAL` port/ttl table origins; the exact code that
  produced `mined.json` (deleted). Why any single sample is `valid=True/False`
  currently requires reading up to five modules.

---

## 5. Duplicated rules (same relation in multiple files)

- **Var = Std²:** `mined.json` + `cicids2017.py::_DERIVATIONS` + legacy
  `attack/validator.py::R_var_eq_std_sq` + `layer1.ProductEquality`.
- **Total Length = Count × Mean (fwd/bwd):** `mined.json` + `_DERIVATIONS` +
  `masks/cicids2017_distrinet` + `realizability/cicids2017` + `realizability/validator`.
- **Avg = Mean identities:** `mined.json` + `_DERIVATIONS` + masks.
- **Min ≤ Mean ≤ Max ordering:** `mined.json` (8 rules) + `realizability/validator`
  + legacy `R_avg_in_range`/`R_min_leq_max` + `layer1.MonotoneNondecreasing`.
- **Port/Protocol domains:** PAVE + `_TYPING` + legacy `ProtocolValidator` +
  `vae/schema.PROTOCOL_ALLOWLIST` + `physics_validator.PROTOCOL_MIN_HEADER`.

Each CICFlowMeter identity appears in **4–5 modules**. Some duplication is
intentional (generation vs independent evaluation), but the rule *definitions*
are copied, so provenance and tolerance drift silently.

---

## 6. Validity vs plausibility — where they are mixed

1. **`run_cicids2017_vae_attacks.py`**: `joint_valid` includes Layer-1
   `RobustTailBound` (distributional) — the main CICIDS mixing.
2. **`PAVEStyleValidator`**: one `valid_mask` blends SCHEMA + MINED (train min/max
   fallback) + PROTOCOL. Auditable per-feature but conceptually merged.
3. **Legacy CICIoT2023**: G-rules (structural) + physics (plausibility) + protocol
   + Mahalanobis historically merged; CLAUDE.md flags shipped
   `he_idsr = E·(1−O)` as defective (omits validity), and
   `thesis_visualizations.py IDSR = (~evasion).mean()` as a separate bogus (= 1−ASR).

`validator_v2` fixes this by returning `schema/mined/extractor/protocol` +
`hard_structural_valid` + `hybrid_valid` separately, and keeping
`in_distribution`/`plausibility_score` strictly out of structural validity.

---

## 7. Retain vs deprecate

**RETAIN (foundations / references):**
- `src/datasets/feature_manifest.py`, `src/datasets/cicids2017.py` — schema truth.
- `old_constraints/cicids2017_distrinet/mined.json` — the legacy MINED artifact
  (validator_v2 compares against it via `validation/legacy.py`).
- `src/attack/realizability/*` — the EXTRACTOR realizability leg.
- `_idr_mask` / `vae/cicids2017_stage_a.fit_idr` — the plausibility realism gate,
  kept separate.
- `scripts/audit_results.py` — independent verifier.

**DEPRECATE / migrate off (do not modify now; superseded by validator_v2 profiles):**
- `src/attack/validator.py`, `src/vae/physics_validator.py`,
  `src/vae/calibrate_physics_validator.py`, `src/attack/latent_infra.py`
  `ProtocolValidator`, `src/attack/constrained_input_baselines.py`
  `VAEConstraintProjection`, `src/vae/schema.py` hardcoded protocol/partition.

**Legacy preservation:** none of the above is renamed or edited. `validation/legacy.py`
provides a faithful, self-contained reimplementation of the legacy CICIDS2017
mined-engine `validate()` (monotone + product) purely for the legacy-vs-v2
comparison (§15), so the original `src/` code stays untouched and importable by
the attack runners.

---

## 8. What validator_v2 builds (gap closed)

1. A concrete, re-runnable **mining script** (`validation/mining/run_mining.py`)
   with a restricted, interpretable grammar — replacing the deleted miner.
2. One **Rule object** with explicit provenance (SCHEMA/MINED/EXTRACTOR/PROTOCOL),
   tolerance, and evidence — replacing scattered `if` statements.
3. **Explainable results** (`SampleResult.to_markdown()`), separating
   `hard_structural_valid` / `hybrid_valid` / `in_distribution`.
4. **Human-readable reports** (mining, clean-acceptance, synthetic, legacy-vs-v2)
   and a **feature reference** for all 79 features.
