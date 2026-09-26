# Final Experiment F — Validator evaluation (descriptive)

validator_v2 is the independent domain validator used by every experiment. It combines general network-flow consistency constraints (SCHEMA domain/type rules, CICFlowMeter EXTRACTOR identities, PROTOCOL rules) with automatically mined, train-only dataset-specific invariants (MINED). This experiment applies it to **every genuine held-out flow** of both datasets (validation and test splits, all classes including Benign). Neither split was used for rule mining or tolerance fitting. No hypothesis test is used. The results are counts and percentages.

## Rule inventory

| Dataset | Category | Layer | Rules | of which source-conditioned (perturbed flows only) |
|---|---|---|---|---|
| CICIDS2017 | SCHEMA | general flow consistency | 133 | 0 |
| CICIDS2017 | EXTRACTOR | general flow consistency | 7 | 0 |
| CICIDS2017 | PROTOCOL | general flow consistency | 80 | 1 |
| CICIDS2017 | MINED | dataset-specific (train-mined) | 16 | 0 |
| CICIDS2018 | SCHEMA | general flow consistency | 133 | 0 |
| CICIDS2018 | EXTRACTOR | general flow consistency | 7 | 0 |
| CICIDS2018 | PROTOCOL | general flow consistency | 78 | 1 |
| CICIDS2018 | MINED | dataset-specific (train-mined) | 10 | 0 |

The source-conditioned PROTOCOL rule `PROTO_0080` (an empty forward packet stays empty: source `Fwd Packet Length Min = 0` ⇒ perturbed `Fwd Packet Length Min = 0`) constrains perturbed flows against their source. A genuine flow is its own source, so the rule is never eligible here and cannot change genuine-flow acceptance.

## Acceptance / rejection of genuine held-out flows

| Dataset | Split | Validator | Genuine flows | Accepted | Rejected | Acceptance | Rejection |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | val | general only (SCHEMA ∧ EXTRACTOR ∧ PROTOCOL) | 312,058 | 311,964 | 94 | 99.9699% | 0.0301% |
| CICIDS2017 | val | general + dataset-specific (hybrid_valid) | 312,058 | 311,964 | 94 | 99.9699% | 0.0301% |
| CICIDS2017 | test | general only (SCHEMA ∧ EXTRACTOR ∧ PROTOCOL) | 312,056 | 312,056 | 0 | 100.0000% | 0.0000% |
| CICIDS2017 | test | general + dataset-specific (hybrid_valid) | 312,056 | 312,056 | 0 | 100.0000% | 0.0000% |
| CICIDS2018 | val | general only (SCHEMA ∧ EXTRACTOR ∧ PROTOCOL) | 125,033 | 125,033 | 0 | 100.0000% | 0.0000% |
| CICIDS2018 | val | general + dataset-specific (hybrid_valid) | 125,033 | 124,949 | 84 | 99.9328% | 0.0672% |
| CICIDS2018 | test | general only (SCHEMA ∧ EXTRACTOR ∧ PROTOCOL) | 125,032 | 125,032 | 0 | 100.0000% | 0.0000% |
| CICIDS2018 | test | general + dataset-specific (hybrid_valid) | 125,032 | 124,952 | 80 | 99.9360% | 0.0640% |

## Rejection categories

A flow counts under a category when it violates at least one rule of that category. `Rejected only by this category` counts flows that every other category accepts.

| Dataset | Split | Rejection category | Rejected (≥1 rule of category) | Rate | Rejected only by this category |
|---|---|---|---|---|---|
| CICIDS2017 | val | SCHEMA | 94 | 0.0301% | 94 |
| CICIDS2017 | val | EXTRACTOR | 0 | 0.0000% | 0 |
| CICIDS2017 | val | PROTOCOL | 0 | 0.0000% | 0 |
| CICIDS2017 | val | MINED | 0 | 0.0000% | 0 |
| CICIDS2017 | test | SCHEMA | 0 | 0.0000% | 0 |
| CICIDS2017 | test | EXTRACTOR | 0 | 0.0000% | 0 |
| CICIDS2017 | test | PROTOCOL | 0 | 0.0000% | 0 |
| CICIDS2017 | test | MINED | 0 | 0.0000% | 0 |
| CICIDS2018 | val | SCHEMA | 0 | 0.0000% | 0 |
| CICIDS2018 | val | EXTRACTOR | 0 | 0.0000% | 0 |
| CICIDS2018 | val | PROTOCOL | 0 | 0.0000% | 0 |
| CICIDS2018 | val | MINED | 84 | 0.0672% | 84 |
| CICIDS2018 | test | SCHEMA | 0 | 0.0000% | 0 |
| CICIDS2018 | test | EXTRACTOR | 0 | 0.0000% | 0 |
| CICIDS2018 | test | PROTOCOL | 0 | 0.0000% | 0 |
| CICIDS2018 | test | MINED | 80 | 0.0640% | 80 |

## Rules that reject genuine flows

| Dataset | Split | Rule | Category | Expression | Rejected | Rate |
|---|---|---|---|---|---|---|
| CICIDS2017 | val | SCH_0085 | SCHEMA | `URG Flag Count == 0.0` | 94 | 0.0301% |
| CICIDS2017 | val | SCH_0058 | SCHEMA | `Fwd URG Flags == 0.0` | 93 | 0.0298% |
| CICIDS2017 | val | SCH_0060 | SCHEMA | `Bwd URG Flags == 0.0` | 1 | 0.0003% |
| CICIDS2018 | val | MINED_0001 | MINED | `Fwd Packet Length Min ~= Packet Length Min` | 84 | 0.0672% |
| CICIDS2018 | test | MINED_0001 | MINED | `Fwd Packet Length Min ~= Packet Length Min` | 80 | 0.0640% |

## Class-wise acceptance

| Dataset | Split | Class | Genuine flows | General-only acceptance | General + dataset-specific acceptance | Rejected (hybrid) |
|---|---|---|---|---|---|---|
| CICIDS2017 | val | Benign | 247,164 | 99.9620% | 99.9620% | 94 |
| CICIDS2017 | val | DoS | 25,733 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | val | DDoS | 14,265 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | val | Recon | 23,853 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | val | BruteForce | 1,043 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | test | Benign | 247,164 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | test | DoS | 25,733 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | test | DDoS | 14,265 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | test | Recon | 23,852 | 100.0000% | 100.0000% | 0 |
| CICIDS2017 | test | BruteForce | 1,042 | 100.0000% | 100.0000% | 0 |
| CICIDS2018 | val | Benign | 37,500 | 100.0000% | 99.8747% | 47 |
| CICIDS2018 | val | DoS | 30,000 | 100.0000% | 99.8767% | 37 |
| CICIDS2018 | val | DDoS | 30,000 | 100.0000% | 100.0000% | 0 |
| CICIDS2018 | val | Recon | 13,403 | 100.0000% | 100.0000% | 0 |
| CICIDS2018 | val | BruteForce | 14,130 | 100.0000% | 100.0000% | 0 |
| CICIDS2018 | test | Benign | 37,500 | 100.0000% | 99.8853% | 43 |
| CICIDS2018 | test | DoS | 30,000 | 100.0000% | 99.8767% | 37 |
| CICIDS2018 | test | DDoS | 30,000 | 100.0000% | 100.0000% | 0 |
| CICIDS2018 | test | Recon | 13,403 | 100.0000% | 100.0000% | 0 |
| CICIDS2018 | test | BruteForce | 14,129 | 100.0000% | 100.0000% | 0 |

## Machine-readable outputs

`per_sample.parquet` (one verdict row per genuine flow, per category), `validator_acceptance.csv`, `rejection_categories.csv`, `rule_rejections.csv`, `classwise_acceptance.csv`, `rule_inventory.csv`.

## Interpretation

**Genuine held-out traffic is almost always accepted.** validator_v2 accepts 100.0000% of the
CICIDS2017 test split (312,056 flows) and 99.9699% of its validation split (94 of 312,058 rejected).
On CICIDS2018 it accepts 99.9360% of the test split (80 of 125,032 rejected) and 99.9328% of the
validation split (84 of 125,033 rejected). No EXTRACTOR or PROTOCOL rule rejects any genuine
flow of either dataset.

**Rejection categories.**
- CICIDS2017: all 94 validation rejections are Benign flows. Each violates a SCHEMA rule that
  pins the URG-flag features to 0 (`URG Flag Count == 0` for 94 flows; `Fwd URG Flags == 0` for
  93; `Bwd URG Flags == 0` for 1). The value comes from the train-derived schema profile, so the
  rule is general in form but train-derived in value.
- CICIDS2018: all rejections come from one dataset-specific rule, `MINED_0001`
  (`Fwd Packet Length Min ≈ Packet Length Min`). It rejects 84 validation flows (47 Benign, 37 DoS)
  and 80 test flows (43 Benign, 37 DoS). The general layer alone accepts 100% of both splits.

**The new transition rule.** `PROTO_0080` (amendment A2: a source flow's zero-length forward
packet must stay empty) is a source-conditioned PROTOCOL rule. A genuine flow is its own source,
so it is never eligible here and changes no acceptance figure above; it only constrains
perturbed flows (Exp A–E). It is the dataset-independent form of what `MINED_0001` enforces on
CICIDS2018 only, without `MINED_0001`'s cost of rejecting ~0.06% of genuine flows.

**General-only vs general + dataset-specific.** Adding the mined rules costs 0.0000 pp of
acceptance on CICIDS2017 and 0.064–0.067 pp on CICIDS2018. On adversarial flows the mined layer
flags 84–100% of invalid CAPGD/C-PGD examples (which also fail EXTRACTOR rules) and ≥ 98.5% of
invalid PGD/C&W examples (which fail every category). After the capability fix no **successful**
PrimAttack output is rejected by any rule (the small number of invalid best-margin failures are
not successes), so `MINED_0001`'s false-rejection rate no longer drives a PrimAttack validity gap.

**Optimizer scope.** Experiment F evaluates validator acceptance, not attack optimization.
Hybrid and Prim-PGD's matching attack outcomes therefore do not provide two validator estimates:
they use the same rules and, on almost entirely timing-only rows, reach the same targeted success
set through closely related sign-momentum searches. Their equality belongs to Exp B's attack-space
analysis; genuine-flow acceptance here is independent of either optimizer.

**Reading (Contribution 4).** The validator combines general flow-consistency constraints
(133 SCHEMA, 7 EXTRACTOR and 78–80 PROTOCOL rules, one of them source-conditioned) with
automatically mined, train-only dataset-specific invariants (16 rules on CICIDS2017, 10 on
CICIDS2018). On data it has never seen, its false-rejection rate is ≤ 0.07%, so "invalid"
verdicts on adversarial flows are not an artefact of an over-strict validator. Acceptance is a
necessary structural check, not a proof of realizability or malicious functionality: a flow can
pass every rule and still not be producible by any real packet trace.
