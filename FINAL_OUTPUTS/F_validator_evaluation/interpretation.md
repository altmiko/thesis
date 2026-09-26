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

**Reading (Contribution 4).** The validator combines general flow-consistency constraints
(133 SCHEMA, 7 EXTRACTOR and 78–80 PROTOCOL rules, one of them source-conditioned) with
automatically mined, train-only dataset-specific invariants (16 rules on CICIDS2017, 10 on
CICIDS2018). On data it has never seen, its false-rejection rate is ≤ 0.07%, so "invalid"
verdicts on adversarial flows are not an artefact of an over-strict validator. Acceptance is a
necessary structural check, not a proof of realizability or malicious functionality: a flow can
pass every rule and still not be producible by any real packet trace.
