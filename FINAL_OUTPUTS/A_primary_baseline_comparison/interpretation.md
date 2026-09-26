Numbers below are seed means over the six victims (3 per dataset, n = 3,200 flows per victim
and seed). All tests are McNemar on seed-42 Valid Success, Holm over the four
PrimAttack-vs-baseline comparisons.

**1. Unrestricted feature-space attacks: high raw success, no valid success.** PGD reaches a
Raw ASR of 91.70–100.00% and C&W 53.59–99.94% on all six victims. Every one of these
adversarial flows is rejected by validator_v2 (Valid ASR = 0.00% for all 12 PGD/C&W cells), so
the Validity Gap equals the Raw ASR. Among the rejected seed-42 examples, 100% violate SCHEMA,
EXTRACTOR and PROTOCOL rules and ≥ 98.5% violate MINED rules (Exp E breakdown). Moving all 79
scaled features independently yields vectors that lie outside the feature domain (SCHEMA),
contradict CICFlowMeter's own aggregate identities (EXTRACTOR), and break protocol rules
(PROTOCOL).

**2. Public constrained attacks restricted to PrimAttack's 23 downstream coordinates.**
Limiting CAPGD and C-PGD to the canonical `primattack_joint_feature_mask` lowers raw success
without removing it. CAPGD-PrimSupport reaches 9.74–96.53% and C-PGD-PrimSupport 1.65–60.42%.
No feature outside the mask changed in any flow (max = 0, asserted). Almost none of this success
is valid. C-PGD reaches a Valid ASR of 0.00% on every victim. CAPGD reaches at most 5.21% (CICIDS2017
CNN), 2.18% on CICIDS2017 MLP and ≤ 0.29% elsewhere. The resulting Validity Gaps are 9.74–92.23 pp
(CAPGD) and 1.65–60.42 pp (C-PGD). Their invalid examples mostly break EXTRACTOR identities
(67–100%) and MINED invariants (84–100%). Changing the 23 coordinates independently decouples
features that one packet-level change moves together. For example, padding changes forward
total length, mean, max, min and byte rate jointly. C-PGD's differentiable penalty covers only
11 of those relations (λ = 1) and does not prevent this. Restricting the support therefore
constrains which features move but not whether they stay mutually consistent.

**3. Reaching the same coordinates only through packet-size/timing primitives.** PrimAttack
(Hybrid Search, p75, untargeted) has far lower raw success than any feature-space baseline:
11.06% / 36.67% / 0.50% on CICIDS2017 MLP / CNN / FT-Transformer, and 12.00% / 15.31% / 0.33% on
CICIDS2018. On CICIDS2017 every raw success is also valid (gap 0.00 pp). That makes it the method
with the highest **Valid** ASR on all three CICIDS2017 victims. Cochran's Q is significant for
each victim, and PrimAttack beats every baseline after Holm correction. The margins are +11.06 pp
over PGD, C&W and C-PGD and +8.88 pp over CAPGD on MLP. On CNN they are +36.69 pp and +31.44 pp.
On FT-Transformer they are +0.50 pp and +0.34 pp (Holm p = 0.027 against CAPGD). On
CICIDS2018 at p75, Valid ASR is near zero for every method: PrimAttack 0.19% / 0.03% / 0.00%,
CAPGD 0.14% / 0.29% / 0.00%. Cochran's Q is significant for MLP and CNN, but none of the four planned
PrimAttack comparisons survives Holm correction (all |Δ| ≤ 0.19 pp). FT-Transformer has no valid
success under any attack (Q = 0, tests not performed). The CICIDS2018 PrimAttack raw successes
are rejected by a single dataset-specific rule, `MINED_0001` (`Fwd Packet Length Min ≈ Packet
Length Min`). Every rejected example uses padding. The p75 timing budget is also small
(median per-flow delay cap about 31 ms vs about 0.9 s on CICIDS2017). The descriptive unbounded run
confirms that the budget, not the method, limits CICIDS2018. Valid ASR rises to 23.51% (MLP) and
5.01% (CNN) there, and to 47.91% / 73.41% on CICIDS2017 MLP / CNN (plot A6).

**4. How much Raw ASR disappears under the validator.** For PGD and C&W all of it: 53.59–100 pp
per victim. For CAPGD 9.74–92.23 pp and for C-PGD 1.65–60.42 pp. For PrimAttack 0.00 pp on CICIDS2017
and 0.33–15.28 pp on CICIDS2018. PrimAttack's raw successes survive validation far more
often. PrimAttack is **not** valid by construction, though: when no valid success exists, its
incumbent is the best-margin failure, which can evade and still be invalid.

**How to read these differences.** The attack with the highest Raw ASR (PGD) is not the best
attack. It is the one with the least constrained threat model, and none of its evasions is a valid
flow. The attacks also differ in validator access. PrimAttack's search uses validator_v2 as part
of its success predicate. The baselines never query it, and C-PGD only sees a differentiable
subset. PrimAttack's valid-success advantage therefore reflects both the primitive
parameterization and this validity-aware search. The comparison controls downstream feature
support. It does not give CAPGD/C-PGD packet-level realizability, and it does not give
PrimAttack realizability beyond the feature-space proxy (no PCAP is modified). Valid evasion is
strongly victim-dependent. FT-Transformer resists every validity-preserving attack
(≤ 0.50% Valid ASR at p75 and ≤ 0.97% even unbounded), while the CICIDS2017 CNN is the most
exposed victim.
