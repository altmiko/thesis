# 06 — The Validators (and how validity/IDR/True-IDSR compose)

Adversarial "success" in this thesis is deliberately hard to earn. A modified flow must be
**misclassified**, **valid**, and **realistic**. Validity is checked by **three independent
validators**, none of which is embedded into the generator. This document explains each
validator and then how their masks combine into the headline metrics.

| Validator | Level | Question it answers | Source |
|-----------|-------|---------------------|--------|
| **Realizability validator** | internal (A+B) | Is the flow internally self-consistent under the primitive physics? | `attack/realizability/validator.py` |
| **PAVE validator** | Level-A | Are raw values in-range and correctly typed (integer/binary)? | `evaluation/pave_style_validator.py` |
| **Mined constraint engine (A4)** | density | Does the flow obey the mined density/order/product rules? | `constraints/*` + `mined.json` (doc 03) |
| **VAE realism gate (IDR)** | realism | Does the flow's latent code sit inside the val-anchored 95% Mahalanobis region? | `vae/cicids2017_stage_a.py::fit_idr`, `run_*::_idr_mask` |

Plus a fifth, dataset-specific structural validator for the older CICIoT2023 pipeline
(`attack/validator.py`) documented at the end.

---

## 1. Realizability validator (`attack/realizability/validator.py`)

Given the model and an adversarial vector, it returns a **per-category FAIL mask** so a
failing sample reports *why*. Categories (True = violation):

| Category | Checks |
|----------|--------|
| `algebraic_dependency_fail` | every `IdentityCheck` recomputed **from the adversarial vector** (Level B) — catches a generator that forgot to update a dependent feature |
| `packet_summary_fail` | `fwd min ≤ mean ≤ max`, `fwd max ≤ TL_fwd`, non-negativity, `min ≤ mean ≤ max` for combined lengths, std/variance ≥ 0 |
| `timing_fail` | `fwd IAT min ≤ mean ≤ max ≤ total`, `flow IAT min ≤ mean ≤ max`, `duration > 0`, `fwd/bwd IAT total ≤ duration`, `flow IAT max ≤ duration` |
| `negative_rate_fail` | all four rates ≥ 0 |
| `discreteness_fail` | every integer-valued feature is integral (after projection) |
| `frozen_fail` | every frozen / invariant / Level-C feature is **byte-identical to the pristine raw row** |

Tolerances (constructor defaults, used by the attacks): `atol=1e-3`, `rtol=1e-4`,
`frozen_atol=1e-6`, `int_atol=1e-3`. The `le(a,b)` helper uses `a <= b + (atol + rtol·|b|)`.

`RealizabilityReport.valid = ~any_fail` is the `realizable` mask. The "preserved-exactly" set
(`frozen_names`) is every feature whose role is `FROZEN`, `INVARIANT`, or `LEVEL_C` — i.e.
everything `generate()` does not write. The runner separately **asserts** frozen features are
unchanged (a hard guarantee, not just a soft check).

`discreteness` integer set = `_INTEGER_FEATURES` in `realizability/cicids2017.py`
(data-mined: ≥ 99.999% integral on ~600k train rows).

---

## 2. PAVE validator (`evaluation/pave_style_validator.py`)

An **independent, observational** raw-space range + type validator (never projects, clips,
rounds, or mutates). Fit on pristine train only.

- `fit(X_train_raw, feature_names, schema=manifest)` builds one `FeatureConstraint` per
  feature. Bounds come from **semantic manifest bounds where available**, falling back to
  **train min/max** otherwise (the `source` field records which: `dataset_schema`,
  `training_range`, or both). CICIDS integer/protocol overrides
  (`_DATASET_INTEGER_OVERRIDES`: ports, protocol, window bytes) force integer typing even
  though the manifest types them as bounded_continuous.
- `validate_batch(X)` returns `range_valid` (every feature within `[lower - tol, upper + tol]`,
  finite) and `type_valid` (binary features ≈ {0,1}; integer features integral within
  tolerance). `valid = range_valid & type_valid`.
- Tolerances in the attacks: `integer_tolerance = range_tolerance = SCALER_ATOL = 1e-6`.
- It also produces per-feature and per-reason violation counts for diagnostics.

`evaluate_mined_constraints(x_raw, checker)` lets PAVE optionally *consume* the mined
`ConstraintEngine`'s result (`pass_l0_l1_l2`) to report a combined `strict_valid` — but it
**copies none of the engine's rules**, preserving independence.

The attacks call PAVE directly (`pave.validate_batch(adv_raw)["valid_mask"]`) to get the
`pave_valid` mask, and the mined engine separately for `mined_valid`.

---

## 3. Mined constraint engine as a validator

Built as the **A4** engine (Layer 0 + Layer 1 + Layer 2, doc 03). The attacks use
`engine.validate(adv_raw)["pass_l0_l1_l2"]` as the `mined_valid` mask. This is the "dataset
density" gate: domains (L0) + robust tail coverage (L1) + mined order/product rules (L2).

---

## 4. VAE realism gate — IDR / True-IDSR

Realism is measured with the per-class Stage-A β-VAE, **not** with any of the validators
above. `fit_idr` (`vae/cicids2017_stage_a.py`) calibrates the gate on the **validation** rows
of that class (val-anchored, *not* train — this is called out explicitly in `CLAUDE.md`):

1. encode val rows to posterior means `z`;
2. `mean`, `cov` (+ `1e-4·I` regularization), `precision = inv(cov)`;
3. per-row squared Mahalanobis distance `d² = (z-µ)ᵀ Σ⁻¹ (z-µ)`;
4. `threshold_sq = 95th percentile` of `d²` (`method="higher"`).

At attack time `_idr_mask(model, x_adv_scaled, idr_path)` encodes the adversarial sample and
returns `d² <= threshold_sq` — the `in_dist` mask. `IDR` = rate of in-distribution
adversarials over the clean-correct denominator.

**Caveat, honestly stated in the latent runner:** in the *VAE latent* attack the attack
generator and the IDR gate are the **same** per-class VAE, so IDR there is
*generator-relative*, reported as `IDR_generator_relative`, not independent evidence. In the
*direct primitive* attack the VAE is used only as the gate (never in the gradient path), so
IDR is a cleaner realism signal.

---

## 5. How the masks compose (the metrics)

Per `(class, victim, seed)` the runner computes these per-sample boolean masks:

```
clean_correct = victim(clean) == true_class          # denominator
evasion       = victim(adv)  != true_class            # untargeted success
benign        = victim(adv)  == 0                      # targeted Attack->Benign success
pave_valid, mined_valid, realizable                    # the three validators
in_dist       = _idr_mask(...)                         # realism gate
```

The composite definitions (identical in the primitive and latent runners):

```
strict_valid = pave_valid & mined_valid & realizable
true_idsr    = benign & strict_valid & in_dist         # the honest success metric
```

Every rate is conditioned on `clean_correct` and divided by `denom = clean_correct.sum()`:

| Reported metric | Definition |
|-----------------|------------|
| `untargeted_asr` | `mean(evasion | clean_correct)` |
| `targeted_benign_asr` | `mean(benign | clean_correct)` |
| `targeted_strict_valid_asr` | `mean(benign & strict_valid | clean_correct)` |
| `pave_validity`, `mined_validity`, `realizability_aware_validity`, `strict_validity` | per-validator rates |
| `IDR` | `mean(in_dist | clean_correct)` |
| `true_idsr` | `mean(benign & strict_valid & in_dist | clean_correct)` |

Note the deliberate escalation: `untargeted_asr ≥ targeted_benign_asr ≥
targeted_strict_valid_asr ≥ true_idsr`. The gap between the first and last is the whole point
of the thesis — raw evasion is easy; evasion that is *also valid and realistic* is hard.

`fail_counts` additionally reports, per realizability category, how many clean-correct rows
failed — so a low realizability rate can be attributed to a specific cause (dependency,
packet, timing, rate, discreteness, frozen).

---

## 6. The older CICIoT2023 structural validator (`attack/validator.py`)

For completeness (different dataset, 39 window-aggregated features). `validate_batch(X,
feature_names)` checks **structural** rules only — no percentiles, no train min/max, no
integer-protocol rule:

- per-feature `expected_min` / `expected_max` from `FEATURE_METADATA` (± `FLOAT_TOL = 0.01`);
- `R_min_leq_max`: `Min ≤ Max`;
- `R_avg_in_range`: `Min ≤ AVG ≤ Max`;
- `R_var_eq_std_sq`: `Variance ≈ Std²` (abs tol `0.01`, rel tol `0.05`).

`VALID_PROTOCOLS = {0,1,2,6,17,47}` is retained for artifact readers only, **not** used as a
validity rule. `ValidationResult` exposes `overall_valid`, `validity_rate`, and per-rule
violation rates. This validator is not part of the CICIDS2017 primitive-attack path; it is the
CICIoT2023 analogue of the realizability validator.
