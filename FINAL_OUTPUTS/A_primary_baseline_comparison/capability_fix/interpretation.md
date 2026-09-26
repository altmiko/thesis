**Methodological conclusion.** The correction changes the interpretation of PrimAttack from
"uniform padding or timing whenever a flow carries forward payload" to a conservative,
source-conditioned primitive model. A flow with `Fwd Packet Length Min = 0` contains at least one
empty forward packet. Because the aggregate record cannot identify that packet, adding `p` bytes
to every forward packet cannot be defended as padding existing payload: it inserts bytes into the
empty packet. The correct response is not to let the optimizer propose this operation and reject
it later, but to remove padding from that flow's attack space before search. validator_v2 then
checks the same semantic transition independently for every attack.

**Capability impact.** The restriction is severe by design. On the canonical attack lists,
99.97% of CICIDS2017 and 99.63% of CICIDS2018 flows have zero forward minimum. Only 0.03% and
0.38%, respectively, are padding-capable; after intersecting capabilities with the p75 box there
is only one paddable CICIDS2017 flow and 12 paddable CICIDS2018 flows per victim. About 74–75% of
all attacked flows are timing-only and 25–26% have no p75 primitive headroom. The no-primitive
rows are mostly Recon flows; DoS, DDoS and BruteForce are usually timing-capable despite being
padding-ineligible.

**Fresh optimization matters.** Simply filtering the relaxed CICIDS2017 result gave the earlier
lower bounds 0.16% / 2.31% / 0.03% (MLP / CNN / FT-Transformer; recomputed 0.15% / 2.31% /
0.03% across seeds). The fresh search reallocates the existing evaluation schedule to timing and
reaches 4.09% / 13.47% / 0.12%. It recovers 126–127, 348–366 and 3 valid successes per seed over
the filter while still remaining 63–75% below the relaxed result. On CICIDS2018, where relaxed
padding was already rejected by `MINED_0001`, the same reallocation raises Valid ASR from 0.19% /
0.03% / 0.00% to 2.53% / 1.16% / 0.00%. Like-for-like targeted runs with the same optimizer show
the same recovery, so it is not merely the Hybrid→Prim-PGD selection change.

**Why optimizer sophistication matters little here.** Hybrid and Prim-PGD tie exactly on Exp B:
1,752 valid targeted successes each, with identical success masks. Capability-aware padding
removes Hybrid's exact integer-padding phase from almost every row; both then optimize only
`(delay, shape)` through projected sign-momentum steps, momentum 0.75, clean/random starts and
the same realized-flow incumbent. Their delay/shape values and primitive costs can differ, so
this is an observed outcome equivalence under the current timing-dominated protocol—not an
algorithmic equivalence. Prim-PGD becomes canonical only through the pre-registered evaluation
tie-break (188.5 vs 189.6 mean victim evaluations).

**Constrained feature-space reachability vs primitive-domain reachability.** CAPGD-PrimSupport and
PrimAttack are paired on the same flows, victims, seeds, validator, metrics and *potential*
23-feature downstream support. They do not share a feasible set. CAPGD directly moves allowed
feature coordinates within its norm ball; PrimAttack moves delay (and, on the few eligible flows,
padding), and the 23 coordinates can move only through deterministic recomputation and
primitive-induced coupling. The comparison therefore measures the effect of primitive-domain
parameterization. At p75 PrimAttack is higher on four victims (MLP/CNN of both datasets), lower
on CICIDS2017 FT by 0.05 pp and tied at zero on CICIDS2018 FT. This result is not an assumption of
the method: a direct feature-space attack is allowed to win. Native CAPGD, with a different
16-feature configuration mask, is higher than every inferential attack on every victim and is
reported only descriptively.

**Five concepts must remain separate.** (1) *Validator validity* means SCHEMA, EXTRACTOR,
PROTOCOL and MINED rules accept the aggregate flow. (2) *Feature support* is the union of
classifier coordinates φ may write on some eligible flow; it does not say every coordinate is
available on every flow. (3) *Primitive applicability* is the source-conditioned per-flow gate
(`pad_allowed`, `timing_allowed`). (4) *Primitive-induced coupling* means downstream values are
not independent: they are outputs of φ from a small control vector. (5) *Packet-level
realization* would require editing/replaying packets and re-extracting features; it was not done.

**Defensible thesis claim.** PrimAttack is a more restrictive, realizability-oriented **flow-level
abstraction**. Its successful downstream feature changes must arise from modeled attacker-facing
primitives, per-flow applicability checks and deterministic recomputation. The correction removes
the dominant but indefensible relaxed-padding path and leaves a reproducible timing-based attack.
It does not prove packet-level realizability, preserve complete malicious functionality, or show
that a concrete packet trace exists for every accepted vector.
