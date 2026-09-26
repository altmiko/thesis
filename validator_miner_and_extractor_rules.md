# Validator v2: the rule miner and the EXTRACTOR rules

This document covers three questions:

1. How does the validator's rule miner work?
2. Is there code for it?
3. What are the EXTRACTOR rules, and how were they derived?

Everything below was checked against the code and the committed rule artifacts on 2026-09-26. Where
older write-ups disagree with the code, see §6. All numbers come from the committed artifacts or from
re-running the checks on the full train/val splits (§5.3).

---

## 0. Short answers

- **Yes, there is code for the miner.** It is `validation/mining/`, and you run it with
  `python -m validation.mining.run_mining [--dataset cicids2018_distrinet]`.
  - It is a *restricted-grammar* miner, not symbolic regression. It generates about 7.6k–8.0k
    interpretable candidate relations, scores them on a 200k-row **train** sample, confirms them on a
    200k-row **val** sample, and prunes redundant rules. It then re-scores the survivors on the full
    splits.
  - Its outputs are `mined_rules.json`, `protocol_rules.yaml`, `plausibility_profile.json`, the
    schema YAML and a mining report.
  - **Test is never read.** `data_access.load_split("test")` raises `PermissionError` unless an
    explicit override flag is passed.
- **There are 7 EXTRACTOR rules.** All are exact CICFlowMeter algebraic identities (full list in §4.1):
  - `Var = Std²`
  - `Average Packet Size = Packet Length Mean`
  - the two `Segment Size Avg = Packet Length Mean` rules
  - `Flow Pkts/s = Fwd + Bwd`
  - the two `Total Length = count × mean` rules
- **How the EXTRACTOR rules were derived:**
  - They were **hand-authored** from the CICFlowMeter feature definitions. They are *not* output by
    the miner.
  - Each was then **verified** to hold on 100% of the full train and val splits.
  - **The miner independently rediscovers all 7.** They pass its acceptance gates, and `prune_rules`
    then drops the mined copies as "dominated by EXTRACTOR rule EXT_000x".
  - CICIDS2018 **inherits** the 2017 file. `run_mining` copies each rule over only if it again has
    100% train+val support; all 7 passed (`dropped: []`).

---

## 1. Where the code is

| Concern | File |
|---|---|
| Orchestrator (all steps, thresholds, outputs) | `validation/mining/run_mining.py` |
| Split access + test-leak guard, deterministic sampling | `validation/mining/data_access.py` |
| Authored feature registry (category, units, `uncertain` flags) | `validation/mining/feature_registry.py` |
| SCHEMA type inference (unary grammar) | `validation/mining/infer_schema.py`, `mine_unary.py` |
| Candidate grammar | `candidate_templates.py` → `mine_pairwise.py`, `mine_arithmetic.py`, `mine_implications.py` |
| Support, residuals, train-derived tolerance | `validation/mining/evaluate_candidates.py`, `validation/validator/tolerance.py` |
| Redundancy pruning | `validation/mining/prune_rules.py` |
| Mining report | `validation/mining/report_builder.py` |
| Rule semantics (`evaluate` → satisfied/eligible masks) | `validation/validator/rule.py` |
| Runtime loader/engine | `validation/validator/engine.py` (`load_validator`) |
| Tests | `validation/tests/test_rule_mining.py`, `test_rule_engine.py`, `test_empty_forward_packet_rule.py`, … |

The miner writes these artifacts per dataset (`cicids2017_distrinet`, `cicids2018_distrinet`):

- `validation/schema/<ds>.yaml`: the inferred type profile, which becomes the SCHEMA rules.
- `validation/rules/<ds>/mined_rules.json`: the MINED layer.
- `validation/rules/<ds>/protocol_rules.yaml`: the PROTOCOL layer.
- `validation/rules/<ds>/plausibility_profile.json`: the p0.1/p99.9 band. This is **not** part of
  validity.
- `validation/reports/<ds>/mining_report.md`
- `validation/rules/<ds>/extractor_rules.yaml`: read by the miner, not written by it, except for 2018
  (§4.4).

Recorded wall time: 27.2 s (2017) and 12.7 s (2018).

---

## 2. How the miner works (`run_mining.main`)

### 2.0 Pseudocode summary

This condenses `validation/mining/run_mining.py:main` and its helpers. Constants are the values in
the code; `support(r, X)` = #(eligible ∧ satisfied) / #eligible, using `Rule.evaluate`.

```text
CONSTANTS
  THR_TRAIN = 0.999   THR_VAL = 0.995   PREFILTER = 0.99
  DISCOVERY_SAMPLE = VAL_SAMPLE = 200_000   PREFILTER_SAMPLE = 20_000   SEED = 42
  MIN_ANTECEDENT_RATE = 0.01   MIN_ANTECEDENT_ROWS = 1000
  REL_TIGHT = 1e-3 (train)   REL_TIGHT_VAL = 2e-3 (val)   TOL_PERCENTILE = 99.9
  LOGICAL = {le, ge, monotone_chain, implication_zero, implication_pos}

rel_resid(r, X)  = |obs - exp| / (|obs| + |exp| + 1e-6)          # scale-free, in [0, 1]

MINE(dataset):
  names   ← preprocessing_manifest.modelling_feature_names         # 79, fixed order
  Xtr, Xva ← X_train_pristine, X_val_pristine                      # raw; TEST access raises
  S_tr ← sorted uniform sample of 200k train rows (seed 42)
  S_va ← sorted uniform sample of 200k val rows (seed 42)
  S_pre ← first 20k rows of S_tr

  # 1. SCHEMA (unary type facts)
  for each feature f:
      type_f ← classify(S_tr[:, f])   # constant | binary | categorical(≤8 vals, integral) | integer | numeric
      if type_f = categorical and category(f) ≠ flow_identity: type_f ← integer
      record val type (val_type_confirmed; recorded, not enforced)
  write schema/<ds>.yaml                                          # → finite + type rules at load time

  # 2. candidate grammar
  near_const ← { f : most frequent value of S_tr[:, f] covers ≥ 99.9% }
  C ← generate(names \ near_const, category = feature_registry):
        pairwise    : A≈B, A≈k·B (k∈{2,.5,10,100,1e3,1e6}), A≈B²     within a category
        arithmetic  : Min≤Mean≤Max chains, Var≈Std², Combined≈Fwd+Bwd,
                      TotalLen_dir≈Count_dir·Mean_dir, A≈B+C within a category (≤4000/cat)
        implication : Count_dir = 0 ⇒ X_dir = 0   (X ∈ byte_count ∪ rate ∪ iat, same direction)
      dedup identical expressions within each family

  # 3–6. prefilter → tolerance → train/val acceptance
  accepted ← [], rejected ← []
  for r in C:
      if r ∈ LOGICAL: r.tol ← (abs 1e-6, rel 0)
      if r is implication_zero and (antecedent_rate(S_pre) < 1% or eligible(S_pre) < 5):
          reject(r, "low antecedent coverage (prefilter)"); continue
      pre ← support(r, S_pre)                               if r ∈ LOGICAL
            mean(rel_resid(r, S_pre) ≤ REL_TIGHT)           otherwise
      if pre < PREFILTER: reject(r, "prefilter below threshold"); continue

      if r ∈ LOGICAL:
          ok ← support(r, S_tr) ≥ THR_TRAIN and support(r, S_va) ≥ THR_VAL
          if r is implication_zero:
              ok ← ok and antecedent_rows(S_tr) ≥ 1000 and antecedent_rate(S_tr) ≥ 1%
      else:                                                # approximate equalities
          resid ← |obs - exp| on S_tr
          r.tol ← ( clip(p99.9(resid), 1e-6, 1e9),
                    clip(p99.9(resid / (|exp| + 1)), 1e-6, 0.10) )
          ok ← p99.9(rel_resid(r, S_tr)) ≤ REL_TIGHT and p99.9(rel_resid(r, S_va)) ≤ REL_TIGHT_VAL
      if ok: accepted += r (with sample evidence) else: reject(r, reason)

  # 7. pruning
  E ← load extractor_rules.yaml
      (if missing and dataset ≠ cicids2017: inherit the 2017 rules, keeping each one only if its
       support = 1.0 on the FULL train AND val splits; write the file with per-rule evidence)
  kept ← []
  for r in accepted:
      drop if same signature as an already-kept rule         # duplicate
      drop if same signature as a rule in E                   # dominated by EXTRACTOR
      drop if r is A≤B and (equality on {A,B} exists or a Min≤Mean≤Max chain covers it)
      drop if r is A≈k·B and a plain equality on {A,B} exists
      otherwise kept += r

  # 8. full-split re-score (informational; the decision was made above)
  for j, r in enumerate(kept, 1):
      r.id ← MINED_{j:04d};  r.hardness ← EMPIRICAL
      r.evidence += support(r, Xtr), support(r, Xva)         # chunked over the full splits
  write rules/<ds>/mined_rules.json

  # 9. PROTOCOL, plausibility, report
  P ← []
  for each feature f:
      r ← nonnegative(f)  (f ≥ -1e-6)
      if support(r, Xtr) = 1.0 and support(r, Xva) = 1.0: P += r else: excluded += f
  P += zero_preserved(Fwd Packet Length Min)   # if source flow has it = 0, perturbed flow keeps 0
  write rules/<ds>/protocol_rules.yaml
  write rules/<ds>/plausibility_profile.json  # per-feature p0.1/p99.9 of S_tr; never validity
  write reports/<ds>/mining_report.md

RUNTIME (validator, not the miner):
  rule r is violated by row x  ⇔  eligible_r(x) ∧ ¬satisfied_r(x)
  approximate satisfied ⇔ |obs - exp| ≤ tol.abs + tol.rel · |exp|
  hybrid_valid(x) ⇔ no SCHEMA, EXTRACTOR, PROTOCOL or MINED rule is violated
```

### 2.1 Inputs and splits

- **Feature order** comes from `preprocessing_manifest.json:modelling_feature_names` (79 features).
- **Data** is the raw, unscaled `X_{train,val}_pristine.npy`. Rules live in raw feature space.
- **Samples** (seed 42):
  - `DISCOVERY_SAMPLE`: 200,000 train rows.
  - `VAL_SAMPLE`: 200,000 val rows.
  - `PREFILTER_SAMPLE`: the first 20,000 of the 200k train sample. Indices are sorted, so this is
    *not* a fresh uniform draw (§6).
- **No labels are loaded.** Every rule is global; there are no per-class rules.

### 2.2 Step 1: SCHEMA inference (unary type facts)

`mine_unary.classify_feature` classifies each column on the train sample. The checks are applied in
this order:

1. Exactly one distinct value → `constant`.
2. All values in {0, 1} → `binary`.
3. At most 8 distinct values and ≥ 99.999% integral → `categorical`.
4. ≥ 99.999% integral → `integer`.
5. Anything else → `numeric`.

A guard in `infer_schema.py` then downgrades `categorical` back to `integer` unless the registry
category is `flow_identity`. In practice only `Protocol` stays categorical, so a rare count value
unseen in train is not rejected.

- The engine emits one `finite` rule per feature, plus one type rule per non-numeric feature. That
  gives **133 SCHEMA rules** on both datasets.
- Observed min/max are stored as evidence only and never become hard bounds.
- The val type is recorded (`val_type_confirmed`) but **not enforced** (see §6).

### 2.3 Step 2: near-constant exclusion

Any column where one value accounts for ≥ 99.9% of the train sample is excluded from *relational*
mining. It keeps its SCHEMA and PROTOCOL rules. Without this exclusion, "almost always 0" columns
would produce thousands of vacuous `A = k·B` relations.

| Dataset | Excluded columns |
|---|---|
| 2017 | Fwd URG Flags, Bwd URG Flags, URG Flag Count, CWR Flag Count, ECE Flag Count, Subflow Bwd Packets |
| 2018 | Fwd URG Flags, Bwd URG Flags, URG Flag Count, Subflow Bwd Packets |

### 2.4 Step 3: candidate grammar (the whole hypothesis space)

Candidates are generated within categories from the authored registry: `packet_count`, `byte_count`,
`packet_size`, `rate`, `iat`, `tcp_flags`, and so on. Identical expressions are removed within each
family.

| Family | Template | Generator | 2017 | 2018 |
|---|---|---|---:|---:|
| pairwise | `A ≈ k·B`, k ∈ {2, 0.5, 10, 100, 1000, 1e6} | `mine_pairwise` | 3540 | 3720 |
| pairwise | `A ≈ B²` | `mine_pairwise` | 590 | 620 |
| pairwise | `A ≈ B` | `mine_pairwise` | 295 | 310 |
| arithmetic | `Min ≤ Mean ≤ Max` chains, found by name suffix | `statistical_groups` | 8 | 8 |
| arithmetic | `Variance ≈ Std²`, found by name suffix | `statistical_groups` | 1 | 1 |
| arithmetic | `A ≈ B + C` within a category, capped at 4000 per category, plus 4 `Combined ≈ Fwd + Bwd` directional sums | `category_arithmetic`, `directional_sums` | 3129 | 3276 |
| arithmetic | `TotalLen_dir ≈ Count_dir × PacketLengthMean_dir` | `guided_products` | 2 | 2 |
| implication | `A = 0 ⇒ B = 0` | `mine_implications` | 21 | 21 |
| **total** | | | **7586** | **7958** |

- **Implication pairs.** A is a directional packet count (`Total Fwd Packet`, `Total Bwd packets`,
  `Fwd Act Data Pkts`). B is a same-direction `byte_count`, `rate` or `iat` feature.
- **Deliberately absent:**
  - Bare `A ≤ B` inequalities. They were removed because they are correlation, not structure;
    ordering only enters through the Min/Mean/Max chains.
  - General difference, product and ratio scans. The `mine_arithmetic` docstring lists them, but
    the code emits none.
  - Fitted coefficients.
  - Class-conditional rules.

### 2.5 Step 4: prefilter (20k train rows, threshold `PREFILTER = 0.99`)

- **Logical rules** (`le`, `ge`, `monotone_chain`, `implication_*`) use plain support: satisfied ÷
  eligible, with 1e-6 slack.
- **Approximate rules** use the fraction of rows whose scale-free residual
  `|o − e| / (|o| + |e| + 1e-6)` is ≤ `REL_TIGHT = 1e-3`.
- **`implication_zero`** also needs antecedent rate ≥ 1% and at least 5 eligible rows.

### 2.6 Step 5: tolerance, then train/val acceptance

- **Logical rules:**
  - Tolerance is fixed at `abs = 1e-6`, `rel = 0`.
  - Accept iff train support ≥ **0.999** and val support ≥ **0.995**.
  - Implications also need ≥ **1000** antecedent rows and antecedent rate ≥ **1%** on train.
- **Approximate rules:**
  - The tolerance comes from the train residual distribution (`tolerance.suggest_tolerance`):
    `abs = clip(p99.9|o−e|, 1e-6, 1e9)` and `rel = clip(p99.9(|o−e|/(|e|+1)), 1e-6, 0.10)`.
  - Acceptance uses **residual tightness**, not support: p99.9 of the scale-free residual must be
    ≤ 1e-3 on train **and** ≤ 2e-3 on val.
  - This stops a self-derived tolerance from making a false relation pass vacuously.
- **Runtime check:** `|observed − expected| ≤ abs + rel·|expected|`. A sample violates a rule iff it
  is `eligible ∧ ¬satisfied`, so an implication whose antecedent is false can never be violated.

### 2.7 Step 6: pruning (`prune_rules.prune`)

Every removal records a `why_pruned` reason. The checks are applied in this order:

1. An exact duplicate of an already-kept rule is removed.
2. A rule that duplicates an EXTRACTOR rule is removed, because the extractor version has stronger
   provenance.
3. `A ≤ B` is removed if an equality exists on the same pair, or if a Min/Mean/Max chain covers it.
4. A scaled equality is removed if a plain equality exists on the same pair.

### 2.8 Step 7: full-split re-scoring and outputs

- **Survivors** get ids `MINED_0001…`, hardness `EMPIRICAL`, and full-split train/val support in
  their evidence. This full-split re-score is informational only; the accept decision was already
  made on the 200k samples.
- **PROTOCOL** (`_build_protocol_rules`):
  - One `nonnegative` rule per feature, emitted **only if** it holds on *every* full train and val
    row.
  - Plus the source-conditioned transition rule `zero_preserved::Fwd Packet Length Min` (amendment
    A2): if the source flow has `Fwd Packet Length Min = 0`, the perturbed flow must keep it at 0.
    This rule is only eligible when `validate_batch(X, source)` is given the source rows.
- **Plausibility profile:** per-feature p0.1/p99.9 bands, fitted on the train sample. It is reported
  separately and never folded into `hybrid_valid`.

### 2.9 What the miner produced

**Funnel:**

| | 2017 | 2018 |
|---|---:|---:|
| Candidates | 7586 | 7958 |
| Accepted before pruning | 24 | 18 |
| Pruned | 8 | 8 |
| **Retained MINED rules** | **16** | **10** |

In both datasets, the 8 pruned rules are the 7 EXTRACTOR identities, with `Var ≈ Std²` appearing
twice because it was generated by two families.

**CICIDS2017: 16 MINED rules.** Every one has full-split train and val support of 1.0.

| ids | rule |
|---|---|
| MINED_0001–0008 | `Min ≤ Mean ≤ Max` for Fwd Pkt Len, Bwd Pkt Len, Flow IAT, Fwd IAT, Bwd IAT, Packet Len, Active, Idle |
| MINED_0009 | `PSH Flag Count = Fwd PSH Flags + Bwd PSH Flags` (residual exactly 0) |
| MINED_0010–0016 | `Total Bwd packets = 0 ⇒ X = 0`, X ∈ {Total Length of Bwd Packet, Bwd IAT Total/Mean/Std/Max/Min, Bwd Packets/s} (14,566 eligible train rows) |

**CICIDS2018: 10 MINED rules.**

| ids | rule | full train / val support |
|---|---|---|
| MINED_0001 | `Fwd Packet Length Min = Packet Length Min` | 0.99943 / 0.99933 (84 val rows violate) |
| MINED_0002–0009 | the same 8 `Min ≤ Mean ≤ Max` chains | 1.0 / 1.0 |
| MINED_0010 | PSH sum | 1.0 / 1.0 |

**Why 2018 has no `Total Bwd packets = 0 ⇒ …` rules.** They hold at 100%, but the antecedent is too
rare:

| Share of flows with `Total Bwd packets = 0` | 2017 | 2018 |
|---|---:|---:|
| Full train split | 1.0002% | 0.839% |
| 200k discovery sample | 1.0325% | 0.857% |
| 20k prefilter sample | 1.145% | 0.325% |

The 2018 rules fail the 1% coverage gate at the prefilter. The 2017 rules pass it by only about 0.03
percentage points, so their inclusion depends on the threshold.

The `Total Fwd Packet = 0 ⇒ …` implications are rejected on both datasets for the same reason: the
antecedent rate is 0.002% (2017) and 0.026% (2018).

**Loaded validator totals (`load_validator`, verified):**

| | SCHEMA | EXTRACTOR | PROTOCOL | MINED | Total |
|---|---:|---:|---:|---:|---:|
| 2017 | 133 | 7 | 80 (79 non-negativity + 1 transition) | 16 | **236** |
| 2018 | 133 | 7 | 78 (77 non-negativity + 1 transition) | 10 | **228** |

In 2018, `Fwd Header Length` and `Bwd Header Length` fail non-negativity because they are
int16-wrapped, so they are listed under `excluded` in `protocol_rules.yaml`.

### 2.10 Sample SCHEMA and PROTOCOL rules

These rules were taken verbatim from the loaded `load_validator("cicids2017_distrinet")` objects.
Every rule of both layers is evaluated on every row with no eligibility condition, except
`zero_preserved`, which applies only to rows that come with a source flow.

**How SCHEMA rules are built.** `engine.schema_rules_from_profile` turns
`validation/schema/cicids2017_distrinet.yaml` into rules while the validator loads. For each feature
in manifest order it emits:

- a `finite` rule, always;
- at most one type rule (`constant` / `binary` / `categorical` / `integer`), none for `numeric`.

Ids are sequential (`SCH_0001…SCH_0133`). All are `hardness: HARD`, provenance
`origin: inferred from train split`, `automatically_mined: true`, tolerance `abs = rel = 1e-6`.
The evidence is the train-sample `classify_feature` statistics (200,000 rows).

| id | rule | check at runtime | evidence (train sample) |
|---|---|---|---|
| `SCH_0010` | `integer::Total Fwd Packet`: "Total Fwd Packet is integer-valued." | `\|x − round(x)\| ≤ 1e-6` | 423 distinct values, `frac_integral = 1.0`, observed range 0–4272 (range recorded only, **not** a bound); val type `integer`, `val_type_confirmed: true` |
| `SCH_0006` | `categorical::Protocol`: "Protocol takes one of the observed categorical codes [0.0, 6.0, 17.0]." | `x ∈ {0, 6, 17}` (±1e-6) | 3 distinct values; the only `categorical` feature, since the `flow_identity` guard downgrades every other small-domain column to `integer`; val confirmed |
| `SCH_0058` | `constant::Fwd URG Flags`: "Fwd URG Flags is constant at 0.0 across the whole train split." | `\|x − 0\| ≤ 1e-6` | 1 distinct value. On val the column is **not** constant (`val_value_type: categorical`, `val_type_confirmed: false`), yet the rule is still hard and fails on 93 full-val rows (§6 #1) |
| `SCH_0107` | `binary::Subflow Fwd Packets`: "Subflow Fwd Packets takes only the values 0 or 1." | `x ∈ {0, 1}` (±1e-6) | 2 distinct values {0, 1}. This is why no `Subflow = Total` extractor identity exists (§4.5). Val confirmed |

(The paired finite rule, e.g. `SCH_0009 finite::Total Fwd Packet`, checks `isfinite(x)`; every
feature has one.)

**How PROTOCOL rules are built.** `run_mining._build_protocol_rules` applies one template,
`nonnegative` (`x ≥ −1e-6`), with id `PROTO_{feature_index+1:04d}`. A feature gets the rule only if
it holds on 100% of the full train and val rows. After that comes the single transition rule
`zero_preserved`. All are `hardness: PROTOCOL` with provenance "networking / extractor domain
knowledge", `automatically_mined: false`, and tolerance `abs = 1e-6, rel = 0`.

| id | rule | check at runtime | evidence |
|---|---|---|---|
| `PROTO_0004` | `nonnegative::Flow Duration`: "Flow Duration is non-negative: CICFlowMeter flow statistics (counts, sizes, durations, rates, ratios, ports/protocol codes) cannot be negative by construction." | `x ≥ −1e-6` | full train support 1.0, full val support 1.0 |
| `PROTO_0005` | `nonnegative::Total Fwd Packet` (same template and description) | `x ≥ −1e-6` | full train 1.0, full val 1.0 |
| `PROTO_0018` | `nonnegative::Flow Packets/s` (same template and description) | `x ≥ −1e-6` | full train 1.0, full val 1.0 |
| `PROTO_0080` | `zero_preserved::Fwd Packet Length Min`: "An empty forward packet stays empty: if the SOURCE flow has Fwd Packet Length Min == 0 (…e.g. a pure ACK), the perturbed flow must keep Fwd Packet Length Min == 0." | eligible iff `\|source − 0\| ≤ 1e-6`; satisfied iff `\|x − 0\| ≤ 1e-6`. With `source=None`, no row is eligible | `applies_to: (perturbed flow, unperturbed source flow) pairs only`. The source condition (`Fwd Packet Length Min = 0`) holds on 50.5% of train flows and 63.8% of val flows; `unperturbed_flow_violations: 0` |

**The PROTOCOL gate in action (CICIDS2018).** `Fwd Header Length` and `Bwd Header Length` would be
`PROTO_0037` / `PROTO_0038`, but the non-negativity check fails on them:

| Feature | full train support | full val support |
|---|---:|---:|
| `Fwd Header Length` | 0.99974 | 0.99971 |
| `Bwd Header Length` | 0.99990 | 0.99992 |

The values are int16-wrapped; the train minimum of `Fwd Header Length` is −32,488. Both rules
therefore go under `excluded` in `validation/rules/cicids2018_distrinet/protocol_rules.yaml`
instead of being emitted. For 2018, the transition rule is also `PROTO_0080`; there it is the 78th
rule, and its source condition holds on 86.7% of train and 88.3% of val flows.

---

## 3. How the layers combine (`validation/validator/result.py`)

```
schema_valid, extractor_valid, protocol_valid, mined_valid  = conjunction over each layer's rules
hard_structural_valid = SCHEMA ∧ EXTRACTOR ∧ PROTOCOL        (definitional)
hybrid_valid          = hard_structural_valid ∧ MINED         (headline validity gate for attacks)
in_distribution       = plausibility band, reported separately, never part of validity
```

- **Hardness by layer:** EXTRACTOR and SCHEMA are `HARD`, PROTOCOL is `PROTOCOL`, MINED is
  `EMPIRICAL`.
- **Attack-side entry point:** `validation/attack_interface.py:structural_masks(x_adv,
  source_raw=...)`.

---

## 4. EXTRACTOR rules

### 4.1 Full list (identical for 2017 and 2018)

Source files: `validation/rules/cicids2017_distrinet/extractor_rules.yaml`, and
`validation/rules/cicids2018_distrinet/extractor_rules.yaml` (inherited from 2017).

| id | name | template | relation | tolerance (abs, rel) |
|---|---|---|---|---|
| EXT_0001 | packet_variance_eq_std_squared | `square_relation` | `Packet Length Variance = (Packet Length Std)²` | 1.0, 1e-3 |
| EXT_0002 | average_packet_size_eq_packet_length_mean | `equality` | `Average Packet Size = Packet Length Mean` | 1e-3, 1e-4 |
| EXT_0003 | fwd_segment_size_avg_eq_fwd_packet_length_mean | `equality` | `Fwd Segment Size Avg = Fwd Packet Length Mean` | 1e-3, 1e-4 |
| EXT_0004 | bwd_segment_size_avg_eq_bwd_packet_length_mean | `equality` | `Bwd Segment Size Avg = Bwd Packet Length Mean` | 1e-3, 1e-4 |
| EXT_0005 | flow_packets_per_s_eq_fwd_plus_bwd | `sum_equality` | `Flow Packets/s = Fwd Packets/s + Bwd Packets/s` | 0.1, 1e-3 |
| EXT_0006 | total_length_fwd_eq_count_times_mean | `product_equality` | `Total Length of Fwd Packet = Total Fwd Packet × Fwd Packet Length Mean` | 1.0, 1e-3 |
| EXT_0007 | total_length_bwd_eq_count_times_mean | `product_equality` | `Total Length of Bwd Packet = Total Bwd packets × Bwd Packet Length Mean` | 1.0, 1e-3 |

Every rule has `source_type: EXTRACTOR`, `hardness: HARD`, `automatically_mined: false`, and
provenance origin "DistriNet CICFlowMeter feature definition". All are eligible on every row; none
has an eligibility condition.

### 4.2 Why each identity holds (the CICFlowMeter definitions)

| Rule | Definition it encodes |
|---|---|
| EXT_0001 | CICFlowMeter reports both the variance and the standard deviation of the same packet-length sample, and the std is the square root of the variance. |
| EXT_0002 | "Average Packet Size" is computed as total bytes ÷ total packets over both directions, which is the same quantity as `Packet Length Mean` in this extractor release. |
| EXT_0003 / 0004 | "Segment Size Avg" per direction is the mean packet length of that direction: `TotalLen_dir / Count_dir`. |
| EXT_0005 | All three rates share one denominator, the flow duration. So `(Nf + Nb)/d = Nf/d + Nb/d`. |
| EXT_0006 / 0007 | Mean is defined as sum ÷ count, so `sum = count × mean` for each direction. When count = 0, both sides are 0. |

These are the same identities PrimAttack's canonical transform φ rebuilds exactly
(`src/attack/realizability/cicids2017.py`: the docstring at lines 23–31, the `write_when(...)` calls
around lines 670–740, and the `IdentityCheck` list around lines 359–383). As a result, PrimAttack
outputs cannot violate EXT_0001–0007 by construction.

### 4.3 How they were derived: a four-part evidence chain

1. **Authored from extractor semantics, not searched for.** The 2017 YAML is hand-written, with a
   description and provenance per rule. Its header states explicitly: "NOT empirically mined —
   their provenance is the extractor definition".
   - The `evidence` blocks in that file record 100% support on the full train split (1,456,265 rows)
     and val split (312,058 rows).
   - No in-tree script writes the 2017 evidence block; only the 2018 inheritance path does (§4.4).
     [INFERENCE] The 2017 evidence was therefore computed by a one-off check when the file was
     authored (commit `619c1ec`, "Made validator v2 + fixed primattack").
2. **Re-verified today on the full splits** of both datasets, using the committed rule objects:

   | rule | 2017 train max \|o−e\| | 2017 val max \|o−e\| | 2018 train max \|o−e\| | max relative residual (all) | violations |
   |---|---:|---:|---:|---:|---:|
   | EXT_0001 | 2.20 | 1.27 | 4.43 | ≤ 8.9e-8 | 0 |
   | EXT_0002 | 0 | 0 | 0 | 0 | 0 |
   | EXT_0003 | 0 | 0 | 0 | 0 | 0 |
   | EXT_0004 | 0 | 0 | 0 | 0 | 0 |
   | EXT_0005 | 0.0039 | 0.031 | 0.0020 | ≤ 4.5e-8 | 0 |
   | EXT_0006 | 0.051 | 0.57 | 0.15 | ≤ 2.9e-8 | 0 |
   | EXT_0007 | 3.38 | 0.54 | 2.20 | ≤ 4.1e-8 | 0 |

   - Every residual is float32 rounding: a relative error of about 1e-7, i.e. float32 machine
     epsilon, on large magnitudes.
   - No row uses even 0.1% of its tolerance.
3. **The miner rediscovered all 7 independently.** In both mining reports (§7 "Pruned rules"), the
   grammar produced each identity from generic templates:
   - `equality` for EXT_0002/0003/0004
   - `square_relation` for EXT_0001, from both `statistical_groups` and the pairwise family
   - `category_arithmetic` sum for EXT_0005
   - `guided_products` for EXT_0006/0007

   All of them passed the train/val tightness gates, and then `prune_rules` removed them as
   "dominated by EXTRACTOR rule EXT_000x (same relation, stronger provenance)". The authored list and
   the data-driven miner therefore agree exactly: the miner finds no extractor-grade identity outside
   these 7, and no authored identity that the data rejects.
4. **Cross-dataset re-verification.** CICIDS2018 has no hand-written extractor file.
   `run_mining._verify_reference_extractor_rules` copies each 2017 rule and keeps it **only if**
   support is exactly 1.0 on the full 2018 train split (583,487 rows) and val split (125,033 rows).
   All 7 passed, and the per-dataset evidence is written next to each rule.

### 4.4 Tolerances: authored, not fitted

The EXTRACTOR tolerances were set by hand, loosely enough to absorb float rounding. For comparison,
this is what the miner's `suggest_tolerance` (p99.9 of the train residual) would have fitted:

| rule | authored (abs, rel) | miner-fitted on 2017 train | miner-fitted on 2018 train |
|---|---|---|---|
| EXT_0001 | 1.0, 1e-3 | 0.806, 1e-6 | 0.048, 1e-6 |
| EXT_0002–0004 | 1e-3, 1e-4 | 1e-6, 1e-6 (floors) | 1e-6, 1e-6 |
| EXT_0005 | 0.1, 1e-3 | 1.2e-4, 1e-6 | 3.1e-5, 1e-6 |
| EXT_0006 | 1.0, 1e-3 | 7.3e-4, 1e-6 | 3.7e-4, 1e-6 |
| EXT_0007 | 1.0, 1e-3 | 0.014, 1e-6 | 3.4e-3, 1e-6 |

- **Consequence:** the relative slack of 1e-3 permits roughly a 0.1% inconsistency on these
  identities. That is far above the data's ~1e-7 noise, so the rules are conservative against false
  rejections but also leave a small manipulation margin.
- **Measured detection:** the synthetic-violation report detects a 2% variance error in 99.8% of
  cases.
- **For PrimAttack** this margin is irrelevant, because φ satisfies the identities exactly.

### 4.5 Identities deliberately NOT encoded

The YAML header lists these:

| Candidate | Why it is not a rule |
|---|---|
| `Fwd/Bwd IAT Total = IAT Mean × (count − 1)` | Holds on 100% of train rows with count ≥ 2 on both datasets (re-checked: 2017 fwd 1,144,484 rows, bwd 1,126,998; 2018 fwd 462,689, bwd 457,117). But the `count − 1` offset is outside the rule grammar (`product_equality` has only a multiplicative `k`), so it was left out rather than faked. PrimAttack's φ uses it (`Fwd IAT Mean = Fwd IAT Total / (Nf − 1)`). |
| Bulk features (`*/Bulk Avg`, `Bulk Rate Avg`) | Outputs of CICFlowMeter's bulk state machine. The formulas are not reliably reproducible on this release, and the best mined relations only reach 0.955 (2017) / 0.979 (2018) prefilter support. The registry marks them `uncertain`. |
| `Subflow Fwd/Bwd Packets = Total Fwd/Bwd` | Subflow packet counts are binary {0, 1} in both releases. `Subflow Fwd Packets == Total Fwd Packet` on only 0.002% (2017) / 0.026% (2018) of train rows. The registry marks them `uncertain`. |

---

## 5. Reproduce

```powershell
# repo root, thesis env active
$Env:PYTHONPATH = "src;."
python -m validation.mining.run_mining                               # cicids2017_distrinet
python -m validation.mining.run_mining --dataset cicids2018_distrinet
python -m pytest validation/tests -q
```

### 5.1 Warning about re-running

Re-running **overwrites** the files below. Frozen FINAL_OUTPUTS results were validated against the
current rule files, so regenerating them changes the validator under those results.

- `schema/<ds>.yaml`
- `mined_rules.json`
- `protocol_rules.yaml`
- `plausibility_profile.json`
- the mining report

### 5.2 Loading the validator

```python
from validation import load_validator
v = load_validator("cicids2017_distrinet")
res = v.validate_batch(X_raw, source=X_source_raw)   # source needed for PROTO transition rule
```

### 5.3 How the numbers in this document were checked

Every figure in §2.9 and §4.3–4.5 was re-computed on 2026-09-26 by:

- loading both validators;
- evaluating every rule on the full train and val splits;
- regenerating the candidate grammar;
- recomputing the near-constant lists and antecedent rates.

---

## 6. Discrepancies and caveats found while writing this

| # | Issue | Evidence |
|---|---|---|
| 1 | **SCHEMA val confirmation is recorded but not enforced.** The three URG columns are constant 0 on train, so they become hard `== 0` rules. On the full 2017 val split they fail on 93 (`Fwd URG Flags`), 1 (`Bwd URG Flags`) and 94 (`URG Flag Count`) rows, so 2017 val is not 100% `hybrid_valid`. 2018 val violates only `MINED_0001`, on 84 rows (support 0.9993 ≥ 0.995, accepted by design). | `infer_schema.py` records `val_type_confirmed` but still writes the type; re-check above |
| 2 | `report_builder.py` hard-codes the 2017 split sizes (`1,456,265` / `312,058`), so `validation/reports/cicids2018_distrinet/mining_report.md` states the wrong row counts. The real 2018 counts are 583,487 / 125,033. | `report_builder.py:72-73` |
| 3 | The mining reports' "Grammar families" line advertises `A<=B`, `A~=B-C`, `A~=B*C` and `A~=B/C`. None of these is generated (the only products are the 2 guided ones). | `report_builder.py:83`, vs the §2.4 counts |
| 4 | The mining reports list PROTOCOL = 79 (2017) and 77 (2018), which predates the `PROTO_0080` / `PROTO_0078` transition rule. The current files contain 80 and 78. | `protocol_rules.yaml` vs the reports |
| 5 | The 2017 `extractor_rules.yaml` header says the IAT-total identity "is documented in docs/cicids2017_feature_reference.md". It is not mentioned there. | grep of that file |
| 6 | The prefilter sample is the *first* 20k of a sorted 200k draw, not a uniform 20k. For 2018 this lowers the `Total Bwd packets = 0` rate from 0.86% to 0.33%. The outcome is unchanged, since both are below 1%. | `run_mining.py:133`, `data_access.sample_rows` |
| 7 | Whether 2017 gets the backward-zero implications (MINED_0010–0016) depends on the 1% antecedent-rate threshold: the full-split rate is 1.0002%. | §2.9 |
| 8 | The older root doc `validator_v2_rule_miner_explained.md` is stale in places. It reports PROTOCOL 79 / 235 total rules (now 80 / 236). It says PROTOCOL support "is not used as an acceptance gate — always emitted"; non-negativity is now emitted only if it holds on every train and val row. It predates CICIDS2018 and the transition rule. | this document, §2.8–2.9 |

Claim boundary: every rule here is a **feature-space** consistency check on aggregate CICFlowMeter
statistics. Passing all 236 (or 228) rules says nothing about whether a packet sequence exists that
produces the vector (`NullPacketBackend`, Level-C unavailable).
