# Validator v2 & Rule Miner — How Every Rule Is Found (Detailed)

**Bottom line:** validator v2 is a **profile-driven, 235-rule CICIDS2017 validator**.
Only **16 rules are empirically mined**. The other 219 come from inferred feature
types, authored extractor identities, and a domain-level non-negativity template.

Current composition:

| Layer | Rules | How obtained |
|---|---:|---|
| SCHEMA | 133 | Feature types inferred from a 200k-row train sample |
| EXTRACTOR | 7 | Manually specified CICFlowMeter algebraic definitions, empirically verified |
| PROTOCOL | 79 | One non-negativity rule instantiated for every feature |
| MINED | 16 | Selected from 7,586 candidate relationships |
| **Total structural rules** | **235** | Loaded by one generic engine |
| Plausibility profile | Separate | Per-feature train p0.1–p99.9 bands; not a structural rule |

The implementation is under `validation/`. It targets the current **79-feature
CICIDS2017-DistriNet profile**, not the older 39-feature CICIoT validator.

---

## 1. End-to-end architecture

```text
preprocessing_manifest.json
        │
        ├── modelling_feature_names ──► fixed 79-column order
        │
X_train_pristine.npy
X_val_pristine.npy
        │
        ▼
validation.mining.run_mining
        │
        ├── schema/cicids2017_distrinet.yaml
        ├── rules/.../mined_rules.json
        ├── rules/.../protocol_rules.yaml
        ├── rules/.../plausibility_profile.json
        └── uses existing extractor_rules.yaml
        │
        ▼
load_validator("cicids2017_distrinet")
        │
        ▼
Validator.validate_batch(raw_X)
        │
        ├── evaluates each Rule → satisfied, eligible masks
        ├── schema_valid
        ├── extractor_valid
        ├── protocol_valid
        ├── mined_valid
        ├── hard_structural_valid
        ├── hybrid_valid
        └── in_distribution, separately
```

Primary files:

- Mining orchestrator: `validation/mining/run_mining.py`
- Candidate grammar: `validation/mining/candidate_templates.py`
- Candidate families:
  - `validation/mining/mine_pairwise.py`
  - `validation/mining/mine_arithmetic.py`
  - `validation/mining/mine_implications.py`
  - `validation/mining/mine_unary.py`
- Feature-family registry: `validation/mining/feature_registry.py`
- Rule scoring: `validation/mining/evaluate_candidates.py`
- Pruning: `validation/mining/prune_rules.py`
- Rule semantics: `validation/validator/rule.py`
- Runtime engine: `validation/validator/engine.py`
- Result aggregation: `validation/validator/result.py`
- Plausibility: `validation/validator/plausibility.py`
- Attack adapter: `validation/attack_interface.py`

The engine itself contains no CICIDS feature names. Dataset knowledge lives in
YAML/JSON profiles and the authored feature registry.

---

## 2. Data used by the miner

`validation/mining/data_access.py:20-45` loads:

- Feature order from
  `data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json`
- Raw, unscaled feature arrays:
  - `X_train_pristine.npy`
  - `X_val_pristine.npy`
  - `X_test_pristine.npy`

The test split is guarded: loading it during mining without
`allow_test_for_final_reporting=True` raises `PermissionError`.

The current artifact records:

- Full train: **1,456,265 rows**
- Full validation: **312,058 rows**
- Discovery sample: **200,000 train rows**
- Confirmation sample: **200,000 validation rows**
- Prefilter sample: **20,000 train rows**
- Seed: **42**
- Test used during mining: **false**

See `validation/rules/cicids2017_distrinet/mined_rules.json:4-24`.

No labels are loaded by `run_mining.main()`. Therefore:

> All v2 rules are global dataset-level rules. There is no per-class or
> per-attack-class mining.

---

## 3. The authored feature registry

Before discovering relationships, the miner needs to know which comparisons are
semantically reasonable. `validation/mining/feature_registry.py` assigns every
feature: category, units, description, extractor-derived flag, uncertainty flag.

Examples:

| Feature | Category |
|---|---|
| `Total Fwd Packet` | `packet_count` |
| `Total Length of Fwd Packet` | `byte_count` |
| `Fwd Packet Length Mean` | `packet_size` |
| `Flow Packets/s` | `rate` |
| `Bwd IAT Mean` | `iat` |
| `PSH Flag Count` | `tcp_flags` |
| `Active Mean` | `active_idle` |

This registry does **not** declare the final empirical rules. It constrains the
search space; e.g. pairwise rules are normally considered only within the same
category. Uncertain bulk and subflow features are explicitly flagged rather than
assigned invented formulas (`feature_registry.py:169-191`).

---

## 4. SCHEMA rules: how the 133 type rules are found

SCHEMA inference is separate from relationship mining.

### 4.1 Type classifier

For each feature column, `mine_unary.classify_feature()` computes: finiteness,
distinct-value count, fraction integral, fraction nonnegative, observed train
min/max. Thresholds (`validation/mining/mine_unary.py:17-19`):

```python
INT_SUPPORT = 0.99999
BINARY_DOMAIN = {0.0, 1.0}
MAX_CATEGORICAL = 8
```

Classification order (`mine_unary.py:49-68`):

1. Exactly one distinct finite value → `constant`
2. Values are a subset of `{0,1}` → `binary`
3. At most eight distinct values and ≥99.999% integral → `categorical`
4. ≥99.999% integral → `integer`
5. Otherwise → `numeric`

Guard in `infer_schema.py:37-44`: a small integer value set is treated as a hard
categorical domain only for `flow_identity` features; a count feature with few
observed values stays `integer`, so unseen larger counts remain valid.

### 4.2 Resulting type counts

| Type | Features |
|---|---:|
| Integer | 48 |
| Numeric | 25 |
| Constant | 3 |
| Binary | 2 |
| Categorical | 1 |

Special cases:

- Categorical: `Protocol`
- Constants: `Fwd URG Flags = 0`, `Bwd URG Flags = 0`, `URG Flag Count = 0`
- Binary: `Subflow Fwd Packets`, `Subflow Bwd Packets`

### 4.3 Conversion into runtime rules

`schema_rules_from_profile()` (`validation/validator/engine.py:39-81`) emits:

- one `finite` rule per feature: 79 rules;
- one type rule per non-numeric feature: 48 integer + 3 constant + 2 binary + 1 categorical.

So `79 + 48 + 3 + 2 + 1 = 133`. Numeric features get only the finite rule.
Observed train min/max are stored as evidence, not as hard limits.

---

## 5. Features excluded from relationship mining

Before generating cross-feature candidates, the miner removes features for which
one value occupies at least 99.9% of the 200k discovery sample:

```python
counts.max() / col.size >= 0.999
```

(`run_mining.py:133-145`). Current excluded list:

1. `Fwd URG Flags`
2. `Bwd URG Flags`
3. `URG Flag Count`
4. `CWR Flag Count`
5. `ECE Flag Count`
6. `Subflow Bwd Packets`

Reason: an almost-always-zero feature would make thousands of accidental relations
(`A = 0×B`, `A = B`) appear high-support without being structural. These features
still retain SCHEMA and PROTOCOL checks.

---

## 6. Candidate grammar

No open-ended symbolic regression — a restricted set of interpretable templates.
After near-constant exclusion, exactly **7,586 candidates**:

| Family | Candidates |
|---|---:|
| Pairwise | 4,425 |
| Arithmetic | 3,140 |
| Implication | 21 |

### 6.1 Pairwise candidates (`validation/mining/mine_pairwise.py`)

Within each feature category, for features `A,B`:

- **Equality** `A ≈ B` (one direction, lexical ordering avoids dups)
- **Scaled equality** `A ≈ kB` for `k ∈ {2, 0.5, 10, 100, 1000, 1e6}` (both directions)
- **Square relation** `A ≈ B²` (both directions)

Breakdown:

| Template | Count |
|---|---:|
| Scaled equality | 3,540 |
| Square relation | 590 |
| Equality | 295 |
| **Total** | **4,425** |

Arbitrary pairwise `A ≤ B` generation was intentionally removed
(`mine_pairwise.py:49-53`); ordering is only produced for Min/Mean/Max groups.

### 6.2 Arithmetic candidates (`validation/mining/mine_arithmetic.py`)

- **A. Statistical groups** — suffix scan for `Min/Mean/Max/Std/Variance`;
  emits `Min ≤ Mean ≤ Max` and `Variance ≈ Std²` (`mine_arithmetic.py:41-58`).
- **B. Directional sums** — normalizes `Fwd`/`Bwd`; proposes
  `A_combined ≈ A_fwd + A_bwd` (e.g. `Packet Length Max ≈ Fwd + Bwd`). Data rejects
  these.
- **C. Guided products** — `Total Length ≈ Count × Packet Length Mean` for fwd/bwd.
  Discovered but later pruned in favor of EXTRACTOR versions.
- **D. Category-local sums** — for each category with ≥3 features, `A ≈ B + C` for
  each target and each unordered pair; capped at 4,000/category. Discovers
  `PSH Flag Count = Fwd PSH Flags + Bwd PSH Flags` and rediscovers
  `Flow Packets/s = Fwd + Bwd` (pruned to EXTRACTOR).

**Implementation vs comments:** the module header mentions general
difference/product/ratio scans, but `category_arithmetic()`
(`mine_arithmetic.py:116-136`) emits **only sum equalities**. The only product
candidates are the two guided total-length formulas.

Breakdown:

| Template | Count |
|---|---:|
| Monotone chain | 8 |
| Square relation | 1 |
| Sum equality | 3,129 |
| Guided product equality | 2 |
| **Total** | **3,140** |

### 6.3 Implication candidates (`validation/mining/mine_implications.py`)

Only `A = 0 ⇒ B = 0`. Antecedents are `packet_count` features; consequents are
direction-matched `byte_count`/`rate`/`iat` features. 21 candidates = seven targets
each for `Total Fwd Packet`, `Total Bwd packets`, `Fwd Act Data Pkts`. Not every
permutation is generated.

---

## 7. Candidate prefiltering

Cheap first stage over 20,000 train rows. Constants (`run_mining.py:45-63`):

```text
PREFILTER                 = 0.99
PREFILTER_SAMPLE          = 20,000
DISCOVERY_SAMPLE          = 200,000
VAL_SAMPLE                = 200,000
MIN_ANTECEDENT_RATE       = 0.01
MIN_ANTECEDENT_ROWS       = 1,000
REL_TIGHT                 = 0.001
REL_TIGHT_VAL             = 0.002
LOGICAL_ABS_TOL           = 1e-6
```

### 7.1 Logical candidates

`support = #(eligible ∧ satisfied) / #(eligible)`; survive if support ≥ 0.99.
Logical comparisons permit absolute slack `1e-6`.

### 7.2 Approximate candidates

No fitted tolerance yet; compute a symmetric scale-free residual

```text
r_i = |o_i - e_i| / (|o_i| + |e_i| + 1e-6)
```

where `o_i` is the observed LHS and `e_i` the candidate expression. Prefilter score
= fraction of rows with `r_i ≤ 1e-3`; must be ≥ 0.99. Prevents a bad relation from
surviving via a huge fitted tolerance.

### 7.3 Implication coverage

Prefilter: antecedent rate ≥ 1% and ≥ 5 eligible rows. Discovery: ≥ 1,000 eligible
train rows and antecedent rate ≥ 1%. Blocks vacuous rules whose antecedent never
occurs.

---

## 8. Tolerance estimation

Approximate rules get absolute + relative tolerance from the 200k train sample
(`validation/validator/tolerance.py:65-87`). With residuals `d_i = |o_i - e_i|`:

```text
absolute a = clip( Q99.9(d),               1e-6, 1e9 )
relative r = clip( Q99.9( d_i/(|e_i|+1) ), 1e-6, 0.10 )
```

Runtime closeness: `|o - e| ≤ a + r·|e|`. Floors block float-noise rejections;
caps block effectively-unlimited tolerances. The one retained approximate MINED
rule (PSH sum) is exact, so tolerance collapses to floors `absolute=1e-6,
relative=1e-6`. EXTRACTOR tolerances are authored in `extractor_rules.yaml`, not
fitted here.

---

## 9. Train/validation acceptance

### Logical rules (`run_mining.py:187-206`)

Accept if `train support ≥ 0.999` and `validation support ≥ 0.995`. Implications
also need the antecedent coverage gates.

### Approximate rules (`run_mining.py:208-226`)

Accept via residual tightness: `Q99.9(r_train) ≤ 1e-3` and
`Q99.9(r_val) ≤ 2e-3`. Ordinary support is recorded but not the explicit gate for
approximate rules.

---

## 10. Pruning

24 accepted → 16 retained; 8 removed (`validation/mining/prune_rules.py:43-93`):

1. Collapse exact duplicates.
2. Prefer EXTRACTOR provenance over an identical MINED relation.
3. Equality dominates matching `≤`.
4. A monotone chain dominates its pairwise order components.
5. Plain equality dominates scaled equality on the same pair.

The eight pruned candidates:

1. `Fwd Packet Length Mean ≈ Fwd Segment Size Avg`
2. `Bwd Packet Length Mean ≈ Bwd Segment Size Avg`
3. `Packet Length Variance ≈ Packet Length Std²`
4. `Average Packet Size ≈ Packet Length Mean`
5. A second generated copy of variance/std
6. `Total Length Fwd ≈ Total Fwd Packet × Fwd Packet Length Mean`
7. `Total Length Bwd ≈ Total Bwd packets × Bwd Packet Length Mean`
8. `Flow Packets/s ≈ Fwd Packets/s + Bwd Packets/s`

All have stronger EXTRACTOR equivalents. Variance/std appears twice because
deduplication happens inside each family, not across families.

---

## 11. The 16 final mined rules

All report full train and validation support of 1.0.

### 11.1 Eight Min–Mean–Max chains

1. `Fwd Packet Length Min ≤ Mean ≤ Max`
2. `Bwd Packet Length Min ≤ Mean ≤ Max`
3. `Flow IAT Min ≤ Mean ≤ Max`
4. `Fwd IAT Min ≤ Mean ≤ Max`
5. `Bwd IAT Min ≤ Mean ≤ Max`
6. `Packet Length Min ≤ Mean ≤ Max`
7. `Active Min ≤ Mean ≤ Max`
8. `Idle Min ≤ Mean ≤ Max`

From `statistical_groups()` (`mine_arithmetic.py:41-58`); each applies to every row
with absolute slack `1e-6`.

### 11.2 One flag-count sum

```text
PSH Flag Count ≈ Fwd PSH Flags + Bwd PSH Flags
```

Residuals exactly zero: train p50/p99/max abs-error = 0; train & val support = 1.0.

### 11.3 Seven backward-zero implications

Antecedent `Total Bwd packets == 0`; consequents:

1. `Total Length of Bwd Packet == 0`
2. `Bwd IAT Total == 0`
3. `Bwd IAT Mean == 0`
4. `Bwd IAT Std == 0`
5. `Bwd IAT Max == 0`
6. `Bwd IAT Min == 0`
7. `Bwd Packets/s == 0`

Discovery sample: 2,065 eligible rows, antecedent rate 1.0325%, conditional support
1.0. Full splits: 14,566 eligible train / 32,750 eligible val, conditional support
1.0 both. The `Total Fwd Packet == 0` implications were rejected at prefilter for
too-rare antecedents.

---

## 12. EXTRACTOR rules

Authored in `validation/rules/cicids2017_distrinet/extractor_rules.yaml`, not mined:

1. `Packet Length Variance ≈ Packet Length Std²`
2. `Average Packet Size ≈ Packet Length Mean`
3. `Fwd Segment Size Avg ≈ Fwd Packet Length Mean`
4. `Bwd Segment Size Avg ≈ Bwd Packet Length Mean`
5. `Flow Packets/s ≈ Fwd Packets/s + Bwd Packets/s`
6. `Total Length Fwd ≈ Total Fwd Packet × Fwd Packet Length Mean`
7. `Total Length Bwd ≈ Total Bwd packets × Bwd Packet Length Mean`

Provenance = extractor semantics. 100% support over full train and validation at
authored tolerances. The miner loads them only to prune empirical duplicates
(`run_mining.py:230-233,287-296`).

---

## 13. PROTOCOL rules

`_build_protocol_rules()` (`run_mining.py:299-319`) emits `x_j ≥ -1e-6` for every
feature → 79 rules. Not discovered; asserts a domain principle (CICFlowMeter counts,
lengths, durations, rates, ratios, ports, protocol codes cannot be negative).
Support is computed and stored but not used as an acceptance gate — the rules are
always emitted.

---

## 14. Runtime rule evaluation

Every rule is a `validation.validator.rule.Rule` with: id, provenance layer,
template, params, feature refs, hardness, tolerance, description, provenance,
evidence. Templates (`rule.py:34-42`):

```text
finite integer binary nonnegative nonpositive constant categorical
le ge equality scaled_equality square_relation sqrt_relation
sum_equality difference_equality product_equality ratio_equality
monotone_chain implication_zero implication_pos
```

Evaluation returns `satisfied[N]` and `eligible[N]`; a violation is
`eligible ∧ ¬satisfied`.

Eligibility examples:

- **Ratio**: eligible only if `|denominator| > max(a, 1e-6)`; near-zero denom = inapplicable.
- **Zero implication**: eligible only where `A ≈ 0`; rows with `A ≠ 0` cannot violate.
- **Square root**: eligible only where base ≥ `-a`.

Equalities, chains, type checks and non-negativity are eligible on every row.

---

## 15. Validity outputs

`validation/validator/result.py:44-89` builds independent layer masks:

```text
V_schema    = AND over SCHEMA    rules of ¬violation
V_extractor = AND over EXTRACTOR rules of ¬violation
V_protocol  = AND over PROTOCOL  rules of ¬violation
V_mined     = AND over MINED     rules of ¬violation

V_hard   = V_schema ∧ V_extractor ∧ V_protocol
V_hybrid = V_hard   ∧ V_mined
```

`structurally_valid` defaults to `hybrid_valid` (configurable to hard-only).

- **hard structural** = type/extractor/domain definitions.
- **hybrid** = hard plus empirical train/val invariants.

A per-sample result records the failed rule, expression, feature values, observed
value, expected value, and numerical errors.

---

## 16. Plausibility is separate

`PlausibilityProfile.fit()` (`validation/validator/plausibility.py:95-105`) stores
per feature: p0.1, p99.9, median, IQR. For sample `x`:

```text
within_ij       = [ q0.001_j ≤ x_ij ≤ q0.999_j ]
plausibility_i  = mean_j within_ij
in_distribution = AND_j within_ij
```

One feature outside its band → `in_distribution = False`. This mask is deliberately
excluded from `hard_structural_valid` and `hybrid_valid`. The profile is fit on the
200k train sample, not the full train matrix.

---

## 17. Attack integration

CICIDS attack runners call `structural_masks(adv_raw)["hybrid_valid"]`
(`validation/attack_interface.py:65-84`), which returns `schema_valid`,
`extractor_valid`, `protocol_valid`, `mined_valid`, `hard_structural_valid`,
`hybrid_valid`, `in_distribution`.

Current consumers:

- `src/attack/run_cicids2017_primitive_attack.py`
- `src/attack/run_cicids2017_latent_variants.py`
- `src/attack/run_cicids2017_vae_attacks.py`

`hybrid_valid` is the domain-validity gate; plausibility is reported separately, not
folded into structural validity.

---

## 18. Important implementation caveats

### 18.1 Only 16 rules are actually mined
"Rules are data" describes representation/execution, not statistical discovery of
all 235: 16 MINED + 133 inferred types + 79 domain non-negativity + 7 authored
extractor. That provenance split is explicit in code and the result object.

### 18.2 Validation confirmation recorded but not enforced for SCHEMA inference
`infer_schema.py:35-63` records `val_type_confirmed`, but an unconfirmed
train-inferred type is still written and loaded as a hard SCHEMA rule. Measured on
the full validation split, SCHEMA validity = `311,964 / 312,058 = 0.9996987739`,
with failures:

| Rule | Validation failures |
|---|---:|
| `Fwd URG Flags == 0` | 93 |
| `Bwd URG Flags == 0` | 1 |
| `URG Flag Count == 0` | 94 |

Those columns are constant-zero across full train but not validation, so the claim
that every empirical fact is "confirmed on validation" is not strictly true for
SCHEMA constants. The stored 20k-row test report still showed 100% SCHEMA
acceptance, which does not remove the validation discrepancy.

### 18.3 Schema inference uses the 200k sample
`run_mining.py:124-130` infers types on `tr_sample`/`va_sample`, not the full 1.46M
rows. MINED survivors are re-scored on full splits; SCHEMA types are not re-scored
before serialization.

### 18.4 Full-split rescoring is informational
After pruning, full train/val support is stored in evidence
(`run_mining.py:235-249`) but is not a gate; the decision was made from the 200k
samples. All 16 survivors happen to have full support 1.0.

### 18.5 Approximate acceptance differs from the headline support policy
Logical rules use 0.999/0.995 support thresholds; approximate rules are accepted by
p99.9 relative-residual tightness instead (`run_mining.py:208-226`). Their support
is measured but not part of the `if`.

### 18.6 Prefilter sampling is slightly asymmetric
`sample_rows()` sorts random indices; `pre_sample = tr_sample[:20000]` takes the
earliest 20k of the sorted 200k, not a fresh uniform 20k. Cannot cause a false
final acceptance (survivors are re-checked on full 200k train/val) but could
prematurely reject a relation absent from that subsection.

### 18.7 Grammar coverage is deliberately incomplete
No arbitrary learned coefficients; no general symbolic expressions; no
class-conditional rules; no arbitrary logical predicates; no general
difference/ratio scan in the current implementation; no reliable bulk/subflow
extractor equations; no packet-level constraints. A vector can pass all rules yet be
unrealizable as packets.

---

## 19. Existing empirical checks

**Clean test acceptance** (20,000 untouched test flows,
`validation/reports/cicids2017_distrinet/clean_acceptance_report.md`):

- SCHEMA 1.0, PROTOCOL 1.0, EXTRACTOR 1.0, MINED 1.0
- hard structural 1.0, hybrid 1.0
- in distribution 0.9749

**Synthetic corruptions**: ten one-feature corruption families, mean detection
**0.9998** (min>max, negative count, variance≠std², rate inconsistency, fractional
count, total length≠count×mean, etc.). Shows rules fire on intended local
violations; does not prove completeness against coordinated multi-feature attacks.

---

## 20. Focused runtime verification

Loaded the current profile through the public API:

```text
SCHEMA:    133
EXTRACTOR:   7
PROTOCOL:   79
MINED:      16
TOTAL:     235
```

A pristine train row:

```text
hard_structural_valid = True
hybrid_valid          = True
in_distribution       = False   (genuine row can lie outside p0.1–p99.9 band)
failed_rules          = 0
```

Setting `Fwd Packet Length Min = Fwd Packet Length Max + 1`:

```text
hybrid_valid = False
failed rule  = MINED_0001
expression   = Fwd Packet Length Min <= Fwd Packet Length Mean <= Fwd Packet Length Max
```

Confirms the loaded artifact, generic rule engine, eligibility/violation
aggregation, and final hybrid mask behave as described.
