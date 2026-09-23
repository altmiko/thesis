# validator_v2 — Methodology

An **interpretable hybrid validity framework** for adversarial network-flow
feature vectors. This document describes the method in thesis-defence language.
Implementation: `validation/`. Dataset: CICIDS2017 (DistriNet-corrected
CICFlowMeter, 79 modelling features, raw/pristine feature space).

## Contribution (stated conservatively)

We do **not** claim to introduce automatic constraint mining — prior work on
denial-constraint / invariant mining already exists. The contribution is an
**interpretable hybrid validity framework** that:

1. automatically discovers simple, high-support invariants from **training** data
   using a small, human-readable candidate grammar;
2. independently **confirms** each discovered invariant on the held-out
   **validation** split before accepting it;
3. **distinguishes** empirically-mined invariants from extractor-defined
   identities and protocol/domain constraints, and from automatically-inferred
   schema/type facts;
4. records explicit **provenance** for every rule (which of the four sources it
   comes from, plus any external reference and its train/validation evidence);
5. produces a **human-readable explanation** for every validity decision;
6. keeps **structural validity separate from distributional plausibility**;
7. supports reporting **raw ASR vs validity-aware ASR** with a single, fixed
   denominator;
8. is architected for **reuse across NIDS feature representations** by swapping
   dataset/extractor profiles, not engine code.

## Four rule sources (exactly one primary provenance each)

| Source | Meaning | How obtained | Enters `hard_structural`? |
|---|---|---|---|
| **SCHEMA** | representation/type facts: finite, integer, binary, constant, categorical code domain | auto-inferred on train, confirmed on val | yes |
| **EXTRACTOR** | exact algebraic identities the extractor imposes (e.g. `Variance = Std²`) | authored from the CICFlowMeter definition, then verified 100% on train **and** val | yes |
| **PROTOCOL** | minimal domain law: non-negativity of flow statistics | one principled template per feature | yes |
| **MINED** | empirical cross-feature invariants (orderings, sums, zero-implications) | discovered on train, confirmed on val | only in `hybrid` |

Any external influence (e.g. PAVE) is recorded inside a rule's `provenance`
metadata; the rule still belongs to exactly one of the four semantic categories.

## The restricted, interpretable candidate grammar

The miner is **not** an opaque symbolic-regression system and uses **no LLM**.
It tests a small set of understandable templates:

- **Pairwise:** `A ~= B`, `A ~= k·B` (k ∈ {2, 0.5, 10, 100, 1000, 1e6}), `A ~= B²`.
- **Three-feature arithmetic:** `A ~= B + C`, `A ~= B − C`, `A ~= B·C`, `A ~= B / C`
  (division guarded by eligibility).
- **Statistical ordering:** auto-detected `Min ≤ Mean ≤ Max` chains and
  `Variance ~= Std²`.
- **Zero-implications:** `A == 0 ⇒ B == 0`, restricted to defensible directional
  count→byte/rate/IAT pairs.

The bare inequality `A ≤ B` over arbitrary pairs is deliberately **excluded**:
an inequality that merely happens to hold is correlation, not a structural
invariant (§18). Candidates that involve constant / near-constant columns are
excluded from relational mining — those are captured by the SCHEMA layer instead.

## Train / validation / test discipline

```
TRAIN  → discover candidates, derive tolerances, measure support
VAL    → independently confirm / prune discovered candidates
TEST   → validation ONLY, never discovery/threshold/tolerance selection
```

Leakage is prevented in code: `validation/mining/data_access.load_split("test")`
raises unless an explicit `allow_test_for_final_reporting=True` flag is passed
(used only by the final evaluation reports, never by the miner).

## Acceptance criterion and tolerance

- **Logical rules** (chains, implications) use a fixed `1e-6` floor and must reach
  train support ≥ 0.999 and validation support ≥ 0.995. Implications additionally
  require meaningful antecedent coverage (rate ≥ 0.01 and ≥ 1000 eligible rows), so
  a rule is never "supported" merely because its antecedent almost never occurs.
- **Approximate rules** are gated on **scale-free relative-residual tightness**:
  the 99.9th-percentile of `|obs − exp| / (|obs| + |exp|)` must be ≤ 1e-3 on train
  and ≤ 2e-3 on validation. This is essential: deriving a tolerance from the
  residual and then testing "support at that tolerance" is circular — a spurious
  relation would receive a huge tolerance and pass vacuously. The tightness gate
  is scale-free and cannot be gamed this way.
- Stored tolerances for accepted approximate rules are derived from the train
  residual distribution (`max(floor, p99.9(|residual|))`), then capped, so a
  clean relation with float noise is not rejected while a broad rule cannot form.

## Pruning (deterministic, recorded)

After acceptance, redundant rules are removed with a recorded `why_pruned`:
an equality dominates the corresponding inequalities; a `Min ≤ Mean ≤ Max` chain
dominates its pairwise components; an EXTRACTOR identity dominates an empirically
re-discovered duplicate (the extractor version carries stronger provenance);
duplicate expressions collapse. For CICIDS2017 this prunes the 6 extractor
identities the miner re-discovers empirically, leaving a clean MINED set.

## Validity vs plausibility (kept separate)

The engine returns, per sample:

```
schema_valid, mined_valid, extractor_valid, protocol_valid
hard_structural_valid = SCHEMA ∧ EXTRACTOR ∧ PROTOCOL
hybrid_valid          = hard_structural ∧ MINED
in_distribution, plausibility_score        (distributional, SEPARATE)
```

`hard_structural` uses only definitional/domain rules; `hybrid` adds empirical
invariants. `in_distribution` is a robust per-feature quantile-band check that
**never** changes structural validity. Observed train min/max are used **only**
for plausibility, never as hard validity bounds.

## Explainability

Every decision is a structured `SampleResult` with `to_dict/to_json/to_markdown`.
The markdown lists per-source pass/fail counts and, for each failed rule, the
observed value, expected value, absolute/relative error, allowed tolerance and a
plain-English explanation (see `validation/validator/report.py`). The mining and
evaluation reports let a reader understand the validator without opening the code.

## Cross-dataset portability

The engine, Rule object, grammar, evaluation and reporting are dataset-agnostic.
Porting to another representation (e.g. CICIDS2017 → CICIoT2023) replaces only
the profiles: feature registry, schema profile, extractor rules, dataset-specific
mined rules, and plausibility profile. `General engine = SAME; dataset profile =
DIFFERENT; mined rules = GENERATED PER DATASET.`

## Scientific safeguards (what this framework does NOT do)

- does not mine from test; does not use adversarial samples to decide rules;
- does not treat training-set ranges as physical constraints;
- does not treat correlation as validity;
- does not mix OOD detection into structural validity;
- does not claim packet-level realizability merely because a vector passes these
  constraints (structural validity is necessary, not sufficient, for realizability);
- was designed independently of whether the attack performs well.

## Reproduce

```
python -m validation.mining.run_mining                 # schema + mined + protocol + plausibility + mining report
python -m validation.evaluation.clean_acceptance       # clean acceptance report
python -m validation.evaluation.synthetic_violations   # synthetic corruption report
python -m validation.evaluation.legacy_vs_v2           # legacy vs v2 report
python -m validation.evaluation.feature_reference      # feature reference doc
python -m pytest validation/tests -q                   # unit + regression tests
```
