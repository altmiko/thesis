**Size of the gap.** On identical source flows, the share of raw successes that validator_v2
rejects depends almost entirely on the attack's threat model:

- PGD and C&W: every raw success is invalid on every victim (Validity Gap = Raw ASR,
  53.59–100.00 pp; McNemar p < 1e-300 on all 12 cells).
- CAPGD-PrimSupport: gaps of 9.74–92.40 pp; at seed 42 between 5 (CICIDS2017 FT-Transformer)
  and 166 (CICIDS2017 CNN) of up to 3,121 raw successes survive.
- C-PGD-PrimSupport: gaps of 1.65–60.42 pp; no raw success survives on any victim.
- Capability-aware PrimAttack (Exp A untargeted and all three Exp B optimizers, targeted): gap
  0.00 pp in all 24 cells; every raw success is valid (McNemar p = 1, no discordant flow).

**Optimizer duplication is deliberate, not independent replication.** In Exp B, Hybrid and
Prim-PGD have identical targeted success masks: 1,752 valid successes each and no discordant
flow. Nearly all rows are timing-only, so Hybrid skips exact padding enumeration and both reduce
to closely related sign-momentum timing searches. Their two zero-gap rows are retained because
the optimizer comparison was pre-registered; they must not be treated as two independent pieces
of evidence for validator performance. Prim-C&W follows the same validity gate but finds fewer
CICIDS2017 CNN successes because its cost-penalized trajectory differs.

**Why raw successes are rejected.** PGD and C&W examples fail every validator category (SCHEMA,
EXTRACTOR and PROTOCOL for 100% of them; MINED for ≥ 98.5%). The matched-support attacks never
fail SCHEMA (their type repair works) but mostly break CICFlowMeter identities (EXTRACTOR:
67–100% of CAPGD, 100% of C-PGD invalid examples) and mined invariants (MINED: 84–100%).
PROTOCOL failures (11–82%) all come from the empty-forward-packet rule `PROTO_0080` (0% before
amendment A2 on the identical adversarial flows): both attacks raise `Fwd Packet Length Min` of
flows that had a zero-length forward packet (CAPGD-PrimSupport in about 454 of 3,021 raw
successes per seed on CICIDS2017 MLP). The rule changes almost no verdict on its own: those flows
usually also violate EXTRACTOR or MINED rules, and only 16 / 6 CAPGD-PrimSupport valid successes
(CICIDS2017 MLP / CNN, three seeds) were lost to it alone.

**Why PrimAttack has no gap.** It constructs every candidate through the canonical recomputation
φ (identities hold by construction), keeps only validator-accepted incumbents in its search, and
under the capability rule never pads a flow with an empty forward packet. Before amendment A2 the
relaxed PrimAttack had a 0.33–15.28 pp gap on CICIDS2018, entirely from `MINED_0001` rejecting
padded flows; that source of invalidity is now excluded before optimization.

**Reading (Contribution 3).** A high Raw ASR says little about constrained evasion: the attacks
with the highest raw success produce no valid flow, and matched feature support alone does not
keep flows consistent. The validity gap is a property of the attack parameterization, measured on
the same flows, not an artefact of different samples.
