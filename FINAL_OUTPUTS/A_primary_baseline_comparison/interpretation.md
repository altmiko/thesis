Numbers are seed means (n = 3,200 flows per victim and seed). Tests are McNemar on seed-42
Valid Success, Holm over the four PrimAttack-vs-baseline comparisons. PrimAttack is the
capability-aware version (amendment A2): padding only for flows without a zero-length forward
packet, every other flow timing-only.

**1. Unrestricted feature-space attacks: high raw success, no valid success.** PGD reaches a Raw
ASR of 91.70–100.00% and C&W 53.59–99.94% on all six victims. validator_v2 rejects every one of
these flows (Valid ASR 0.00% in all 12 cells), so the Validity Gap equals the Raw ASR. Moving all
79 scaled features independently leaves the feature domain (SCHEMA), contradicts CICFlowMeter's
aggregate identities (EXTRACTOR) and breaks protocol rules (PROTOCOL): 100% of the seed-42
invalid examples fail each of these categories (Exp E).

**2. Public constrained attacks restricted to PrimAttack's 23 downstream coordinates.**
CAPGD-PrimSupport reaches a Raw ASR of 9.74–96.53% and C-PGD-PrimSupport 1.65–60.42%, with no
feature changed outside the mask (asserted). Almost none of it is valid. C-PGD has a Valid ASR of
0.00% on every victim. CAPGD reaches 2.01% / 5.15% / 0.18% on CICIDS2017 MLP / CNN /
FT-Transformer and 0.14% / 0.29% / 0.00% on CICIDS2018. Their invalid examples mostly break
EXTRACTOR identities (67–100%) and MINED invariants (84–100%). Changing the 23 coordinates
independently decouples features that one packet-level change moves together. The new
empty-forward-packet rule `PROTO_0080` removes only 16 (MLP) and 6 (CNN) of CAPGD's CICIDS2017
valid successes over three seeds (−0.17 / −0.06 pp) and none elsewhere.

**3. Capability-aware PrimAttack is a timing attack on almost every attack flow.** 99.97%
(CICIDS2017) and 99.63% (CICIDS2018) of the attacked flows contain a zero-length forward packet,
so only 1 of 3,200 (CICIDS2017) and 12 of 3,200 (CICIDS2018) flows per victim may be padded. At
p75, 73.8–74.7% of the flows are searched timing-only and 25.0–26.2% have no primitive at all
(mostly Recon: single-packet or zero-IAT probes). Every valid success uses timing only
(p = 0, delay > 0); none fills an empty packet (asserted). PrimAttack (Prim-PGD, p75, untargeted)
reaches a Valid ASR of 4.09% / 13.47% / 0.12% on CICIDS2017 and 2.53% / 1.16% / 0.00% on
CICIDS2018, with a Validity Gap of 0.00 pp everywhere. Its successes come from DoS and DDoS (and
21 CICIDS2018 Recon flows on MLP); BruteForce and Recon barely move.

**4. Inference.** Cochran's Q is significant on five victims (not on CICIDS2018 FT-Transformer,
where none of the five attacks has a valid success). PrimAttack has a higher Valid ASR than PGD, C&W and C-PGD on
CICIDS2017 MLP / CNN and CICIDS2018 MLP / CNN (Holm p ≤ 1.3e-8) and than CAPGD-PrimSupport on the
same four victims: +2.06, +8.28, +2.41 and +0.94 pp at seed 42 (364 vs 99 discordant flows on the
CICIDS2017 CNN; Holm p ≤ 1.3e-5). On CICIDS2017 FT-Transformer none of the four comparisons is
significant (4 PrimAttack-only flows; vs CAPGD Δ = −0.03 pp, Holm p = 1).

**5. What the capability fix changed.** The relaxed pre-fix PrimAttack reached 11.06% / 36.67% /
0.50% on CICIDS2017, almost all of it by padding empty packets. Re-judging those flows with the
new rule leaves 0.15% / 2.31% / 0.03%. The fresh timing-focused re-run recovers 126–127, 348–366
and 3 valid successes per seed above that filter and lands at 4.09% / 13.47% / 0.12%: 63–75% below
the relaxed result, but 27×, 5.8× and 4× the post-hoc lower bound. On CICIDS2018 the fix raises
PrimAttack from 0.19% / 0.03% to 2.53% / 1.16%. There the relaxed search spent its evaluations on
padding that `MINED_0001` always rejected; now the whole budget goes to timing. The selected
optimizer also changed (Hybrid → Prim-PGD, a tie broken by evaluations; Exp B). The like-for-like
targeted comparison with a fixed optimizer shows the same direction (e.g. Hybrid on CICIDS2018
MLP 0.29% → 0.78%).

**6. Descriptive rows.** Native CAPGD (†, its own 16-feature configuration mask, not the PrimAttack
support) reaches a Valid ASR of 9.53% / 19.24% / 5.71% (CICIDS2017) and 28.42% / 16.30% / 0.45%
(CICIDS2018): higher than every inferential attack on every victim. It is not a matched-support
comparison. `PROTO_0080` removes 131 / 260 / 29 of its CICIDS2017 valid successes over three seeds
(−1.37 / −2.71 / −0.30 pp). Unbounded PrimAttack reaches 22.97% / 59.94% / 0.59% and 44.32% /
26.28% / 0.12%, so the p75 timing box (median per-flow delay cap about 0.9 s on CICIDS2017 and
31 ms on CICIDS2018), not the search, limits p75 success.

**How to read these differences.** The highest Raw ASR (PGD) belongs to the least constrained
threat model, and none of its evasions is a valid flow. PrimAttack's search includes validator_v2
in its success predicate; the baselines never query it, and C-PGD sees only a differentiable
subset. PrimAttack's advantage over CAPGD-PrimSupport therefore reflects the primitive
parameterization together with this validity-aware search, and it is limited to the victims
where timing alone moves the decision. Matched support does not give CAPGD/C-PGD packet-level
realizability, and PrimAttack's results are flow-level proxies (no PCAP is modified). Valid
evasion is strongly victim-dependent: FT-Transformer resists every validity-preserving attack
(≤ 0.18% Valid ASR at p75 across the inferential attacks).
