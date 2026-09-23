# PAVE-style validity

Independent, raw-space domain/type validity baseline for adversarial network flows.
Inspired by the feature-constraint methodology in
[robotumbel/PAVE_artifact](https://github.com/robotumbel/PAVE_artifact)
(`code/attacks/constraints.py`, `code/pave_audit.py`); adapted as an **evaluation
baseline**, not a claimed methodological novelty.

- **Module:** `src/evaluation/pave_style_validator.py` (`PAVEStyleValidator`,
  `FeatureConstraint`, `evaluate_mined_constraints`).
- **Runner:** `src/evaluation/run_pave_validity.py` (fit / audit / attack-gating CLI).
- **Tests:** `src/evaluation/tests/test_pave_style_validator.py`.

## What it is (and is not)

The validator answers exactly one question per sample: *does this vector fall inside
each feature's declared domain and datatype in original (raw) units?* It is purely
**observational**:

- It **never** projects, clips, rounds, repairs, or mutates a sample. Contrast with
  the upstream PAVE `FeatureConstraintClassifier.project_tensor`, which repairs into the
  feasible domain; here validity is measured, not enforced.
- It **never** fits a scaler/statistic on validation, test, or adversarial data.
  All train-derived fallbacks come from the **train split only** (THE REFACTOR
  PRINCIPLE). Held-out and adversarial data are transformed/evaluated, never fitted on.
- It is **separate from** the mined constraint engine (`src/constraints`). Mined rules
  are evaluated only if present, and only combined with PAVE-style validity **at
  reporting time** via a strict conjunction (§ Mined-constraint interop).

## The per-feature constraint (`FeatureConstraint`)

`fit()` resolves one `FeatureConstraint` per feature (in manifest order):

| field | meaning |
|---|---|
| `name` | feature name (manifest order) |
| `kind` | semantic label: `binary`, `statistical_integer`, `protocol_integer`, `probability`, `continuous` |
| `lower` / `upper` | inclusive bounds in raw units (`None` = unbounded on that side) |
| `integer` | value must be (near-)integral |
| `binary` | value must be (near-)`0` or `1` |
| `source` | provenance string, e.g. `dataset_schema`, `training_range`, `dataset_schema+training_range` |
| `uncertain` | `True` when no semantic bound was known and both bounds fell back to the train min/max |

### Bound resolution precedence (in `fit`, `pave_style_validator.py:135-283`)

For every feature the fitter first computes the **finite** train min/max of that column
(`train_lower`, `train_upper`; raises if a column has no finite training value). It then
picks the domain from the highest-priority source available:

1. **Repository manifest spec** (`_from_manifest_spec`, preferred). When `fit(schema=…)`
   receives a `FeatureManifest`-like object, `value_type` maps to `kind` + type flags:
   - `binary → binary` (integer + binary),
   - `integer_count → statistical_integer` (integer),
   - `categorical → protocol_integer` (integer),
   - `probability → probability`,
   - `bounded_continuous / positive_continuous / derived / real → continuous`.

   Manifest `lower`/`upper` are used when present; a missing side falls back to the train
   min/max and the source is tagged `…+training_range`. A `_DATASET_INTEGER_OVERRIDES`
   set forces integer/`protocol_integer` semantics for known integer-coded fields whose
   generic `value_type` is `bounded_continuous` (currently the CICIDS ports/protocol and
   init-window-bytes columns).

2. **Mapping schema or `FeatureConstraint`** supplied per name (`_from_mapping_or_name`),
   for callers without a manifest.

3. **`_EXACT_UNIVERSAL` fallback table** — exact **normalized-name** matches only
   (deliberately no substring tests). Covers universal protocol domains, e.g.
   `ttl`/`time_to_live → [0,255]` integer, `src_port`/`dst_port → [0,65535]` integer,
   `protocol → [0,255]` integer, `binary_flag → {0,1}`, `packet_count → [0,∞)` integer,
   `byte_count`/`duration`/`rate → [0,∞)` continuous.

4. **Train min/max only** — if nothing above recognizes the feature, both bounds fall
   back to the observed training range and the constraint is flagged `uncertain=True`
   (surfaced in the audit as "unrecognized / uncertain features").

Name normalization (`_normalize_name`) lowercases and collapses any non-alphanumeric run
to `_`, so `Time_To_Live`, `time to live`, `TIME-TO-LIVE` all resolve to `time_to_live`.

## The validity check (`validate_batch`, `pave_style_validator.py:299-411`)

Given a raw batch `X` (shape `(N, n_features)`), the validator builds two independent
boolean cell grids, both seeded from `np.isfinite(X)`:

- **Range grid** — for each feature with a bound, `value >= lower - range_tolerance` and
  `value <= upper + range_tolerance`.
- **Type grid** —
  - `binary`: value within `integer_tolerance` of `0` or `1` (`np.isclose`, `rtol=0`);
  - `integer`: `|value - round(value)| <= integer_tolerance`;
  - continuous: no type check.

Then:

```
range_mask = range_grid.all(axis=1)     # every feature in range
type_mask  = type_grid.all(axis=1)      # every feature correct datatype
valid_mask = range_mask & type_mask     # combined PAVE-style validity
```

Non-finite cells (`NaN`/`±inf`) fail **both** grids, so any non-finite value invalidates
the sample with reason `"non-finite value"`.

The result dict reports masks, counts, and rates:
`validity_rate`, `range_validity_rate`, `type_validity_rate`, `combined_validity_rate`
(equal to `validity_rate`), plus per-sample `violations` and aggregate
`violation_counts_by_feature` / `violation_counts_by_reason`. Violation reasons are
one of: `non-finite value`, `below lower bound <L>`, `above upper bound <U>`,
`expected binary value`, `expected integer value`.

### Design consequences (verified by tests)

- `TTL = 0` and `TTL = 255` are valid; `-1`, `256` fail **range**
  (`test_ttl_*`).
- `binary_flag ∈ {0,1}` valid; `0.5` is **in range but type-invalid**
  (`test_fractional_binary_is_type_invalid_but_in_range`).
- Integer counts reject fractional and negative values
  (`test_packet_count_requires_nonnegative_integer`).
- An unrecognized feature is bounded by the **train min/max**, so a value above the
  training maximum is range-invalid (`test_unknown_feature_uses_training_minmax`).

Tolerances are constructor knobs (`integer_tolerance`, `range_tolerance`, default
`1e-6`, must be non-negative).

## Sample / scaled entrypoints

- `validate_sample(x, mined_checker=None)` — one raw vector; wraps `validate_batch`.
- `validate_batch(X, mined_checker=None)` — untouched raw batch (the core).
- `validate_scaled_batch(X_scaled, transform, mined_checker=None)` — inverse-transforms
  with an **existing train-fit** `FeatureTransform` (`transform.inverse_transform`)
  first, then validates raw. This is how scaled attack artifacts are audited without
  refitting any scaler. `test_scaled_batch_matches_manual_inverse` confirms the scaled
  path equals manually inverse-transforming then validating.

## Mined-constraint interop (optional, kept separate)

`validate_batch` accepts an optional `mined_checker`. `evaluate_mined_constraints`
(`pave_style_validator.py:82-115`) adapts an existing checker (a callable, or an object
with `.validate`, e.g. `constraints.ConstraintEngine`) into a per-sample mask **without
copying any of its rules into this module**. It:

- feeds the raw batch as a `torch.float32` tensor (falls back to numpy without torch),
- reads the validity mask from a returned mapping using the first present of
  `pass_l0_l1_l2`, `strict_valid`, `valid`, `layer2` (or treats a bare array as the mask),
- optionally reads `per_constraint` to count per-rule violations.

The combination is a **strict conjunction only at reporting**:

```
strict_valid_mask = valid_mask & mined_valid_mask
```

reported alongside `mined_constraint_validity_rate` and `strict_validity_rate`. PAVE-style
validity and mined validity remain independently computed and independently reported
(`test_existing_checker_remains_separate_and_combines_only_at_reporting`).

## Persistence

`summary()` returns the full fitted registry (dataset, tolerances, all constraints,
uncertain-feature list). `save(path)` writes it as JSON; `load(path)` reconstructs an
already-fitted validator (no data needed). Round-trip is exact
(`test_fitted_registry_round_trip`). `format_audit()` renders a human-readable schema
table (feature, kind, lower, upper, source) plus the uncertain-feature list.

## Attack gating and metrics (`run_pave_validity.py`)

`evaluate_attack_arrays` (`run_pave_validity.py:68-119`) gates ASR on validity using the
**originally-correct denominator**:

```
clean_correct = (y_pred_clean == y_true)
successful    = clean_correct & (y_pred_adv != y_true)      # evasion among clean-correct
raw_asr             = successful / clean_correct
pave_valid_asr      = (successful & pave_valid) / clean_correct
valid_asr           = (successful & strict_valid) / clean_correct   # strict = PAVE ∧ mined
```

`over_clean_correct` divides by `clean_correct.sum()` (returns `0.0` when nothing was
clean-correct), so a valid but never-correct sample cannot inflate ASR
(`test_attack_metrics_use_originally_correct_denominator`). When no mined checker is
active, `strict_valid` defaults to the PAVE-style mask and `mined_validity_rate` /
`strict_validity_rate` report `None`.

### CLI

`_raw_split` prefers saved pristine raw arrays (`X_<split>_pristine.npy` under the
adapter's processed dir) and otherwise inverse-transforms the scaled split with the
train-fit transform. `_build_mined_checker` loads `old_constraints/<dataset>/mined.json` as a
**Layer-2-only** `ConstraintEngine` (`active_layers={2}`) when the file exists (unless
`--no-mined`).

Fit on train, audit held-out genuine data, save the fitted registry:

```bash
PYTHONPATH=".;src" python -m evaluation.run_pave_validity \
  --dataset ciciot2023 \
  --heldout-limit 10000 \
  --output-dir outputs/pave_validity
```

Audit untouched scaled attack artifacts (`.npz` with `X_adv`/`x_adv`, `y_true`,
`y_pred_clean`/`y_pred_before`, `y_pred_adv`/`y_pred_after`), reusing the saved validator:

```bash
PYTHONPATH=".;src" python -m evaluation.run_pave_validity \
  --dataset ciciot2023 \
  --load-validator outputs/pave_validity/ciciot2023_pave_validator.json \
  --attack-file path/to/attack_results.npz \
  --attack-space scaled \
  --output-dir outputs/pave_validity
```

Key flags: `--attack-space {scaled,raw}` (use `raw` only when stored vectors are already
in original units), `--attack-dir` (glob every `.npz` in a directory), `--heldout-split
{val,test}` + `--heldout-limit`, `--integer-tolerance` / `--range-tolerance`,
`--mined-constraints <path>` / `--no-mined`, `--load-validator`.

### Outputs

Written under `--output-dir`:

- `<dataset>_pave_validator.json` — fitted registry (reloadable).
- `<dataset>_feature_audit.txt` — human-readable schema table.
- `<dataset>_pave_report.json` — dataset name, fit split, held-out validity, per-attack
  metrics + compact validation, and whether mined constraints were enabled.

Reported rates per audited set: range / type / combined PAVE-style validity, per-feature
and per-reason failure counts, mined and strict validity (when enabled), raw ASR, and
validity-gated ASR over the clean-correct denominator.

## Relationship to the wider validity stack

PAVE-style validity is the **independent, lightweight, raw-space** baseline. It is
distinct from and complementary to:

- **Layer 0/1/2 mined constraints** (`src/constraints`) — see `docs/constraint_miner.md`;
  supplied here only through the optional `mined_checker` strict conjunction.
- **True-IDSR realism gate** (Mahalanobis in-distribution) — a separate, val-anchored
  gate; PAVE-style validity does **not** implement realism, only domain/type feasibility.
