# 3. Validator v2 (MAXIMUM DETAIL)

Validator v2 (`validation/` package) decides whether a raw 79-feature vector is a
*domain-valid* CICIDS2017 flow. Its defining property: **rules are data, not code.**
Every rule is a `Rule` dataclass with a declared template, parameters, tolerance,
provenance, and evidence; the engine evaluates them uniformly and reports exactly which
rule rejected a sample and why. All empirical content is **mined from the train split
only** and confirmed on validation.

### Source files
| Concern | File |
|---|---|
| Rule dataclass + evaluation templates | `validation/validator/rule.py` |
| Tolerance model + mining helper | `validation/validator/tolerance.py` |
| Engine (compose rules, validate) | `validation/validator/engine.py` |
| Result aggregation (validity concepts) | `validation/validator/result.py` |
| Plausibility (distributional, separate) | `validation/validator/plausibility.py` |
| Mining pipeline | `validation/mining/{run_mining,candidate_templates,mine_*,evaluate_candidates,prune_rules,infer_schema}.py` |
| Rule artifacts | `validation/schema/cicids2017_distrinet.yaml`, `validation/rules/cicids2017_distrinet/{extractor_rules.yaml,protocol_rules.yaml,mined_rules.json,plausibility_profile.json}` |
| Attack-side adapter | `validation/attack_interface.py` (`structural_masks`, `evaluate_attack`) |
| Self-tests / reports | `validation/evaluation/{clean_acceptance,synthetic_violations}.py`, `validation/reports/cicids2017_distrinet/*.md` |

Load: `load_validator("cicids2017_distrinet")` → `Validator.from_profiles`
(`engine.py:122-138`) reads the schema YAML + three rule files + plausibility profile
and composes one `Validator`.

---

## 3.1 Architecture / dataflow

```
dataset profile (schema YAML, inferred on train)
   │  schema_rules_from_profile()            → SCHEMA rules  (per-feature type facts)
extractor_rules.yaml   → EXTRACTOR rules      (CICFlowMeter algebraic identities)
protocol_rules.yaml    → PROTOCOL rules       (domain non-negativity)
mined_rules.json       → MINED rules          (empirical train invariants)
plausibility_profile.json → PlausibilityProfile (distributional band)
   │
   ▼
Validator.validate_batch(X)  → per-rule (satisfied, eligible) masks
   │
   ▼
BatchResult  → schema_valid, extractor_valid, protocol_valid, mined_valid,
               hard_structural_valid, hybrid_valid, in_distribution, per-rule rates
```
Per rule, `evaluate(X, idx)` returns two per-sample boolean masks
(`rule.py:68-152`): `satisfied` and `eligible`. **A sample violates a rule iff it is
eligible ∧ ¬satisfied.** Ineligible rows (e.g. a conditional whose antecedent is false,
or a ratio with ~0 denominator) are *never* violations — this is what makes conditional
and ratio rules safe.

Rule counts actually loaded for `cicids2017_distrinet`:
**SCHEMA 133** (79 finite + type rules), **EXTRACTOR 7**, **PROTOCOL 79**, **MINED 16**.

---

## 3.2 SCHEMA layer

Synthesized from the inferred schema profile by `schema_rules_from_profile`
(`engine.py:39-81`). The profile (`validation/schema/cicids2017_distrinet.yaml`,
inferred on the train sample by `mining/infer_schema.py`) assigns each of the 79
features a `value_type`: **48 integer, 25 numeric, 3 constant, 2 binary, 1 categorical**.

Per feature the engine emits:
- a **`finite`** rule always (`isfinite`);
- plus one of: **`constant`** (fixed value, e.g. the 3 all-zero URG fields),
  **`binary`** (∈{0,1}), **`categorical`** (∈ observed code set, e.g. Protocol),
  **`integer`** (`x ≈ round(x)`).

All SCHEMA rules are `hardness="HARD"`, provenance `origin="inferred from train split",
automatically_mined=True`. These are *definitional type facts*, not empirical relations.

---

## 3.3 PROTOCOL layer

`_build_protocol_rules` (`run_mining.py:299-319`) instantiates **one principled
template — non-negativity — per feature** (79 `nonnegative` rules, tol `1e-6`).
Justification recorded in each rule: CICFlowMeter flow statistics (counts, sizes,
durations, rates, ratios, ports/protocol codes) cannot be negative by construction.
`hardness="PROTOCOL"`, `automatically_mined=False` (domain knowledge), but each is
**verified** to hold on train and validation (support stored in `evidence`). This is a
minimal, domain-justified layer — not hand-invented per feature.

---

## 3.4 EXTRACTOR layer

`extractor_rules.yaml` — **7 exact CICFlowMeter algebraic/derived-feature identities**
(`hardness="HARD"`, provenance = extractor definition). These encode how CICFlowMeter
*computes* one feature from others:

| id | type | Relationship |
|---|---|---|
| EXT_0001 | `square_relation` | `Packet Length Variance = (Packet Length Std)²` |
| EXT_0002 | `equality` | `Average Packet Size = Packet Length Mean` |
| EXT_0003 | `equality` | `Fwd Segment Size Avg = Fwd Packet Length Mean` |
| EXT_0004 | `equality` | `Bwd Segment Size Avg = Bwd Packet Length Mean` |
| EXT_0005 | `sum_equality` | `Flow Packets/s = Fwd Packets/s + Bwd Packets/s` |
| EXT_0006 | `product_equality` | `Total Length of Fwd Packet = Total Fwd Packet × Fwd Packet Length Mean` |
| EXT_0007 | `product_equality` | `Total Length of Bwd Packet = Total Bwd packets × Bwd Packet Length Mean` |

These are exactly the identities PrimAttack's φ preserves by construction (doc 2 §2.4).
They carry explicit tolerances (approximate types) because CICFlowMeter emits rounded
floats.

---

## 3.5 MINED layer

`mined_rules.json` — **16 empirical invariants**, mined + confirmed automatically.
Meta (recorded in the file): `fit_split=train, confirm_split=val, test_used=false,
discovery_sample=200000, val_sample=200000, candidates_tested=7586
(pairwise 4425 / arithmetic 3140 / implication 21), accepted_pre_prune=24,
retained=16, seed=42`.

### Miner trace (`mining/run_mining.py:main`)
1. **Schema inference** → schema YAML (`:129-131`).
2. **Candidate generation** (`candidate_templates.generate_candidates :146`) over a
   *restricted grammar* (not open-ended symbolic regression), grouped into families:
   `mine_pairwise` (`le/ge/equality/scaled_equality` on feature pairs),
   `mine_arithmetic` (`sum/difference/product/ratio/square/sqrt` equalities),
   `mine_implications` (`implication_zero`: `A=0 ⇒ B=0`). Near-constant columns
   (≥99.9% one value) are **excluded** from relational mining (`:138-145`) so relations
   aren't driven by a structural constant.
3. **Prefilter** on a 20k sample (`PREFILTER=0.99`): logical rules on plain support;
   approximate rules on scale-free relative-residual tightness (`pre_support :159-169`).
   `implication_zero` also needs antecedent coverage ≥ `MIN_ANTECEDENT_RATE=0.01`.
4. **Tolerance estimation** for approximate rules (`derive_tolerance` →
   `suggest_tolerance`, `tolerance.py:65-87`): `abs = clip(p99.9(|residual|),
   ABS_FLOOR=1e-6, ABS_CAP=1e9)`; `rel = clip(p99.9(|residual|/(|expected|+1)),
   REL_FLOOR=1e-6, REL_CAP=0.10)`.
5. **Train/val confirmation** (`:187-226`): logical rules kept iff `train_support ≥
   THR_TRAIN=0.999 ∧ val_support ≥ THR_VAL=0.995`; approximate rules kept iff the
   99.9th-percentile *relative residual* is tight on **both** train (`REL_TIGHT=1e-3`)
   and val (`REL_TIGHT_VAL=2e-3`) — a self-derived tolerance cannot make a false
   relation vacuously pass.
6. **Redundancy pruning** (`prune_rules.prune :43-93`): an equality dominates the
   matching `A≤B`/`B≤A`; a `Min≤Mean≤Max` chain dominates its pairwise `≤`; an
   EXTRACTOR identity dominates an empirically-mined duplicate (stronger provenance);
   duplicate expressions collapse. (24 accepted → 16 kept.)
7. **Full-split re-support** for survivors, id assignment `MINED_0001…`, `hardness=
   "EMPIRICAL"`, evidence updated with full-split support (`:236-249`).

### The 16 retained MINED rules (all train & val support = 1.0)
- 8 **monotone_chain** `Min ≤ Mean ≤ Max`: Fwd Pkt Len, Bwd Pkt Len, Packet Len,
  Flow IAT, Fwd IAT, Bwd IAT, Active, Idle.
- 1 **sum_equality**: `PSH Flag Count = Fwd PSH Flags + Bwd PSH Flags`.
- 7 **implication_zero** (`Total Bwd packets = 0 ⇒ X = 0`): X ∈ {Total Length of Bwd
  Packet, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,
  Bwd Packets/s}.

---

## 3.6 Rule templates and tolerance evaluation

`RULE_TYPES` (`rule.py:39-42`) with per-template semantics (`evaluate :85-152`):
`finite, integer, binary, nonnegative, nonpositive, constant, categorical, le, ge,
equality, scaled_equality, square_relation, sqrt_relation, sum_equality,
difference_equality, product_equality, ratio_equality, monotone_chain,
implication_zero, implication_pos`.

`APPROXIMATE_TYPES` use the tolerance; logical types use a `1e-6` absolute slack.
Eligibility subtleties: `ratio_equality` eligible only where `|denominator| > max(at,
ABS_FLOOR)`; `implication_zero/pos` eligible only where the antecedent holds;
`sqrt_relation` eligible only where the base ≥ −at.

**Tolerance equation** (numpy-style, `tolerance.is_close :48-52`):
```
close  ⇔  |observed − expected| ≤ absolute + relative · |expected|
```
Values come from the **train residual distribution** (§3.5 step 4): `absolute` from the
99.9th percentile of `|residual|`, `relative` from the 99.9th percentile of
`|residual|/(|expected|+1)`, each clamped to `[floor, cap]`. Floors prevent a perfectly
clean residual from producing an absurdly tight rule that rejects genuine data on float
noise; caps prevent a vacuous "within a mile" rule.

---

## 3.7 Validity concepts (`result.py`)

Per-provenance conjunction `_group_valid` (`:44-49`): a group is valid for a sample iff
none of its rules is violated. Then:

| Concept | Definition | Code |
|---|---|---|
| `schema_valid` / `extractor_valid` / `protocol_valid` / `mined_valid` | conjunction within that source | `:51-65` |
| **`hard_structural_valid`** | `SCHEMA ∧ EXTRACTOR ∧ PROTOCOL` (definitional only) | `:67-69` |
| **`hybrid_valid`** | `hard_structural ∧ MINED` (+ empirical invariants) | `:71-73` |
| `structurally_valid` | alias, default = `hybrid_valid` | `:75-77` |
| **`in_distribution`** | plausibility band, **never folded into structural validity** | `:79-83` |
| `plausibility_score` | fraction of features inside the band | `:85-89` |

**HARD** = SCHEMA + EXTRACTOR (definitional). **PROTOCOL** = domain non-negativity.
**EMPIRICAL** = MINED (train-mined, val-confirmed). **plausibility / in-distribution** =
a *separate* distributional notion, deliberately not part of validity.

The attack side uses `structural_masks(x_adv)` (`attack_interface.py:65-84`) and reads
`hybrid_valid` as the domain-validity gate (PrimAttack `domain_valid`, doc 2).

---

## 3.8 Plausibility / in-distribution

`PlausibilityProfile` (`plausibility.py`) fit on the **train sample** with per-feature
low/high quantiles `low_q=0.001, high_q=0.999` plus median/IQR (`fit :95-105`).
`evaluate(X)` (`:31-57`): `within = (X ≥ low) ∧ (X ≤ high)` per feature;
**`in_distribution = within.all(axis=1)`** (every feature inside its band);
`score = within.mean(axis=1)`; a robust z `|x−median|/(IQR+1e-6)` is reported for OOD
features. Observed train min/max are deliberately *not* hard validity constraints — they
live here as plausibility, never in `structurally_valid` (docstring `:7-8`).

---

## 3.9 What PrimAttack already enforces vs what the validator independently checks

Because PrimAttack's φ *constructs* vectors inside the CICFlowMeter algebra, several
validator layers are **guaranteed to pass** and therefore add little independent
evidence for PrimAttack outputs:
- **EXTRACTOR identities** — φ recomputes Fwd/Bwd Segment Size Avg, Average Packet Size,
  Packet Length Variance/Std, Flow Packets/s, and the totals=count×mean products exactly
  (doc 2 §2.4). PrimAttack cannot violate EXT_0001–0007.
- **PROTOCOL non-negativity** — φ only adds bytes / dilates time (increase-only), and
  frozen features are copied; outputs stay non-negative.
- **MINED monotone chains / PSH-sum** — preserved because φ shifts min/max/mean uniformly
  and never touches flags/backward fields.
- **MINED backward implications** — φ never touches backward features, so
  `N_b=0 ⇒ Bwd·=0` is inherited from x₀.

**Genuinely independent checks** for PrimAttack are essentially only:
- **SCHEMA integer/binary/categorical/constant** on the features φ *writes* — the
  quantization step (`generate(quantize=True)`) is what actually satisfies these; a bug
  in rounding would be caught here.
- **plausibility / in_distribution** — the p0.1/p99.9 band is not enforced by φ at all;
  a large primitive could push a written feature outside the band. This is the one
  validator signal that can independently fail a PrimAttack vector.

Empirically the two are aligned: in the sweep, `raw = valid = primitive-feasible` counts
are **equal** — validator_v2 rejected none of the classifier successes. That is expected
given the above, and is exactly why **100% validator validity does not prove packet-level
realizability**: validity here means "consistent with the CICFlowMeter feature algebra
and inside the training band," which φ was designed to guarantee. It says nothing about
whether a packet sequence realizing that exact padding/timing exists or preserves the
attack (Level-C, `NullPacketBackend`). For *unconstrained* attacks (feature-space PGD/C&W,
doc 8) the validator is strongly independent — those achieve raw ASR≈1.0 but valid ASR=0.

---

## 3.10 Validator experiments

### Clean acceptance (`validation/reports/.../clean_acceptance_report.md`)
20,000 untouched genuine **test**-split flows (pure evaluation, not used for mining):
SCHEMA/PROTOCOL/EXTRACTOR/MINED acceptance = **1.000000** each; hard_structural =
hybrid = **1.000000**; in_distribution (plausibility) = 0.9749. **No structural rule
rejects any genuine sample.** (Confirms soundness: the validator does not reject real
data.)

### Synthetic violations (`synthetic_violation_report.md`)
10,000 accepted base flows, each given **one** controlled corruption; detection =
fraction for which `structurally_valid` flips to FALSE. **Overall mean detection
0.9998** across 10 corruption types, e.g.: min>max (chain) 1.0000; negative count
(PROTOCOL) 1.0000; variance≠std² (EXTRACTOR) 1.0000; a *subtle* 2% variance error
0.9980; totlen≠count×mean 1.0000; fractional count (SCHEMA integer) 1.0000; mean 0.01
below min 1.0000. (Confirms completeness on these injected violations.)

**Limitations of these tests:** (i) corruptions are *single, synthetic, one-feature*
edits — they exercise each rule in isolation, not correlated multi-feature adversarial
manipulations; (ii) subtle within-tolerance corruptions (the 2% variance case) can slip
through by construction — that is the tolerance doing its job, but it bounds sensitivity;
(iii) detection is measured on corruptions *designed to hit a known layer*, so 0.9998 is
a lower bound on evadability, not a guarantee against an adaptive attacker; (iv) neither
test speaks to packet-level realizability.

---

## 3.11 Assumptions · Limitations · Claims

**Assumptions**: train-mined relations that hold at ≥99.9%/≥99.5% train/val support are
genuine domain invariants; the restricted grammar covers the relevant relations; the
p0.1/p99.9 band is a reasonable plausibility envelope.

**Limitations**: mined rules are only as complete as the grammar (no 3-way arithmetic
beyond templates, no cross-direction conditional beyond `A=0⇒B=0`); tolerances admit
small within-band manipulation; the miner used 200k samples of train/val (not full split)
for discovery/confirmation, then re-scored survivors on the full split; validity is a
feature-space property.

**Can claim**: a transparent, data-driven, train-only-fitted validator with explainable
per-rule verdicts; 100% clean acceptance and ~99.98% single-violation detection; a clean
separation of definitional (HARD), domain (PROTOCOL), empirical (MINED), and
distributional (plausibility) validity.

**Must NOT claim**: that `hybrid_valid=100%` implies packet-level realizability or
preserved malicious behavior; that the validator is complete against adaptive
multi-feature attacks; that plausibility is part of structural validity.
