**Main findings (per contribution).**

1. **PrimAttack (Contribution 1).** After the empty-forward-packet capability correction,
   PrimAttack is almost entirely a timing attack: only 0.03% (CICIDS2017) and 0.38%
   (CICIDS2018) of its attacked flows are padding-eligible, and none of its valid successes uses
   padding. At the p75 train-calibrated budget its untargeted Valid ASR is 4.09% / 13.47% / 0.12%
   (CICIDS2017 MLP / CNN / FT-Transformer) and 2.53% / 1.16% / 0.00% (CICIDS2018). It exceeds
   CAPGD-PrimSupport significantly on the MLP/CNN of both datasets, is statistically
   indistinguishable on CICIDS2017 FT and ties at zero on CICIDS2018 FT. Hybrid and Prim-PGD tie
   exactly on the selection outcome (1,752 / 57,600 valid targeted successes); fewer mean victim
   evaluations selects Prim-PGD.
2. **Evaluation methodology (Contribution 2).** Threat-model restrictions reverse the raw-success
   ranking. PGD/C&W reach 53.59–100% Raw ASR but 0% Valid ASR. CAPGD/C-PGD restricted to
   PrimAttack's 23-feature support retain high raw success but little or no valid success.
   PrimAttack has much lower Raw ASR but zero gap between Raw and Valid ASR for every successful
   cell. Targeted and untargeted objectives coincide on CICIDS2017 except for 7 CNN flows, but
   differ by 1.16–1.75 pp on CICIDS2018 MLP/CNN because untargeted DDoS flows can move to DoS.
3. **Paired validity gap (Contribution 3).** On the same source samples, PGD/C&W lose 53.6–100 pp
   and matched-support CAPGD/C-PGD lose 1.7–92.4 pp from Raw to Valid ASR; capability-aware
   PrimAttack loses 0 pp. The new transition rule removes only 0.06–0.17 pp from
   CAPGD-PrimSupport on CICIDS2017 and nothing elsewhere; those direct feature-space outputs
   already fail other identities in almost every case.
4. **Independent validator (Contribution 4).** On 874,179 genuine held-out flows validator_v2
   accepts 99.93–100% per split. `PROTO_0080` is source-conditioned (an empty forward packet stays
   empty) and cannot reject genuine flows. Rejections remain the train-constant URG schema rules
   on CICIDS2017 or `MINED_0001` on CICIDS2018. The transition rule is independently applied to
   every attack, while PrimAttack also removes unsupported padding before search.
5. **Budget analysis (Contribution 5).** Valid Targeted ASR never decreases with the budget.
   The unbounded timing box raises p75 Valid ASR from 4.09% to 22.94% (CICIDS2017 MLP), 13.25% to
   59.69% (CICIDS2017 CNN), 0.78% to 24.76% (CICIDS2018 MLP) and 0% to 26.09%
   (CICIDS2018 CNN). The p75 result is therefore a conservative calibrated-budget result, not a
   bound on timing-based evasion.

**Effect of amendment A2.** The relaxed pre-fix PrimAttack had 11.06% / 36.67% / 0.50% Valid ASR
on CICIDS2017, dominated by padding empty packets. The capability-aware result is 63–75% lower,
but fresh timing optimization recovers 126–127 / 348–366 / 3 valid successes per seed beyond
simply filtering the old outputs. On CICIDS2018 the correction raises Valid ASR because it stops
spending evaluations on padding that `MINED_0001` rejects. This is the methodological point:
capability restrictions belong in the attack space, not only in validation.

**Matched-support interpretation.** CAPGD-PrimSupport directly optimizes the 23 downstream
features; PrimAttack reaches those potential coordinates only through source-applicable
primitives and coupled recomputation. Their comparison quantifies constrained feature-space
reachability versus primitive-domain reachability; it does not require PrimAttack to win. Native
CAPGD (descriptive, its different 16-feature configuration mask) has the highest Valid ASR on all
six victims (0.45–28.42%), showing that the chosen support and parameterization materially define
the threat model.

**Victim dependence.** FT-Transformer stays at ≤ 0.59% Valid ASR under any PrimAttack budget and
≤ 0.18% under the five inferential p75 attacks. CICIDS2017 CNN is the most exposed victim. Results
are reported per victim and dataset and are never pooled.

**Claim boundary.** PrimAttack is a restrictive, realizability-oriented **flow-level abstraction**:
feature changes must arise from modeled padding/timing primitives and deterministic
recomputation. No PCAP is edited, replayed or re-extracted; aggregate features do not identify
individual payload packets; complete malicious functionality and packet-level realizability are
not established. The conservative `Fwd Packet Length Min > 0` condition sacrifices possible
legitimate data-packet padding on mixed empty/data flows. Seeds are attack seeds on one frozen
victim per architecture, not victim-training seeds. PrimAttack queries validator_v2 during
search; baselines do not.

**Amendments.** A1 fixed non-finite gradients at pinned controls before optimizer selection. A2
added capability-aware padding and `PROTO_0080`, preserved the old run as
`superseded_relaxed_padding/`, and reran every stage from scratch. A3 adds native CAPGD only as a
descriptive row; the locked five-method inferential family is unchanged. See `00_PROTOCOL.md` §7
and `../primattack_empty_packet_fix_report.md`.
