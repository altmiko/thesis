**How much classifier-level success disappears when validity is required.** On identical
adversarial examples (seed-42 McNemar, raw vs valid):

- **Unconstrained feature-space attacks (PGD, C&W):** all of it. 1,715–3,200 raw successes per
  victim, 0 valid, gap 53.59–100.00 pp. Every test is significant (p < 1e-300).
- **Matched-support constrained attacks:** almost all of it. CAPGD-PrimSupport keeps 0–168 valid
  of 335–3,121 raw successes (gap 9.74–92.23 pp). C-PGD-PrimSupport keeps none of 39–2,032
  (gap 1.65–60.42 pp). Every test is significant (p ≤ 1.2e-9). The invalid examples fail
  EXTRACTOR identities (67–100%) and MINED invariants (84–100%). They never fail SCHEMA or
  PROTOCOL: the attacks' train-range box and type repair keep single features in-domain, but not
  their mutual consistency.
- **PrimAttack (untargeted and all three targeted optimizers, p75):** none of it on CICIDS2017.
  There is no raw-success-but-invalid example on any victim (p = 1, gap 0.00 pp). On
  CICIDS2018 a systematic gap appears. Untargeted: 379 of 385 (MLP), 489 of 490 (CNN) and 11 of 11 (FT)
  raw successes are invalid (gap 11.81 / 15.28 / 0.33 pp; p ≤ 0.001). Targeted: gaps of
  0.56–1.05 pp (MLP), 13.81–15.01 pp (CNN) and 0.19–0.33 pp (FT) across the three optimizers.
  Every such example violates only the MINED category, specifically `MINED_0001`
  (`Fwd Packet Length Min ≈ Packet Length Min`), and uses padding.

**Is the loss systematic?** Yes, wherever it exists. Because Valid ⊆ Raw, the discordant cell
"raw success, invalid" is one-sided by construction, and every non-zero count is significant.
The seed replicates show the same picture: the Validity Gap SD across seeds is ≤ 0.03 pp for
every PrimAttack condition and ≤ 4.32 pp for every baseline. What matters is the size of the
gap, not the p-value. The feature-space baselines lose essentially their whole Raw ASR, and
PrimAttack loses nothing on CICIDS2017 and 0.2–15.3 pp on CICIDS2018.

**What the gap measures.** Raw ASR alone would rank PGD (≥ 91.70%) far above PrimAttack
(≤ 36.67% at p75). Valid ASR reverses that ranking on CICIDS2017 and flattens it to near zero on
CICIDS2018. The paired validity gap separates two failure modes: classifier robustness (low
raw success) and domain invalidity (high raw, low valid). It also shows where the independent
validator adds information that the attack's own constraints do not encode. For C-PGD, the
differentiable relation penalty did not prevent EXTRACTOR failures. For PrimAttack on
CICIDS2018, the train-mined regularity `Fwd Packet Length Min ≈ Packet Length Min` is not one of
the constraints PrimAttack's recomputation φ preserves. Padding raises the forward minimum, and
the validator rejects the resulting flow (Contribution 3).
