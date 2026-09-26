**Main findings (per contribution).**

1. **PrimAttack (Contribution 1).** At the p75 train-calibrated budget, PrimAttack is the only
   attack that produces validator-valid evasions at scale on CICIDS2017. Valid ASR is 11.06%
   (MLP), 36.67% (CNN) and 0.50% (FT-Transformer), untargeted, and it beats every baseline
   significantly after Holm correction. PGD, C&W, CAPGD-PrimSupport and C-PGD-PrimSupport reach
   up to 100% Raw ASR but at most 5.21% Valid ASR. On CICIDS2018 at p75 every method is near zero
   valid (≤ 0.29%), with no significant PrimAttack-vs-baseline difference. The three optimizers
   are practically tied on aggregate Valid Targeted ASR: Hybrid Search 7.990% vs Prim-PGD 7.977%,
   with Prim-C&W at 4.411%. Hybrid Search is selected by the locked criterion and also uses the
   fewest victim evaluations.
2. **Evaluation methodology (Contribution 2).** Changing objective, validity requirement and
   budget moves results in different directions. The objective barely changes Valid ASR
   (Δ ≤ 0.53 pp; one significant victim) but can change Raw ASR ten-fold (CICIDS2018 MLP: 1.28%
   targeted vs 12.00% untargeted raw). The budget is a first-order control of Valid ASR:
   CICIDS2017 CNN goes from 13.10% to 36.15% to 70.48%, and CICIDS2018 MLP/CNN rise from ≈ 0% to
   22–25% only when the budget is unbounded.
3. **Paired validity gap (Contribution 3).** On identical adversarial examples, feature-space
   baselines lose 53.6–100 pp (PGD/C&W) and 1.7–92.2 pp (matched-support CAPGD/C-PGD) of their
   Raw ASR to the validator. PrimAttack loses 0 pp on CICIDS2017 and 0.2–15.3 pp on CICIDS2018.
   The loss is systematic wherever it occurs (McNemar p ≤ 0.032; every non-zero case one-sided).
   On CICIDS2017, Raw ASR ranks the attacks almost in reverse order of their valid success.
4. **Independent validator (Contribution 4).** On 874,179 genuine held-out flows it accepts
   99.93–100% per split. Rejections come from the train-constant URG-flag schema rules
   (CICIDS2017) or a single mined rule, `MINED_0001` (CICIDS2018). That same mined rule causes
   PrimAttack's entire CICIDS2018 validity gap. Its 0.06% false-rejection rate on genuine
   CICIDS2018 traffic is the relevant caveat.
5. **Budget analysis (Contribution 5).** Valid Targeted ASR never decreases with the budget.
   In 23 of 24 adjacent comparisons every discordant flow favors the larger budget; the single
   exception is one flow. The CICIDS2018 p50/p75 calibrations are too tight for valid
   timing-only evasion, and padding violates `MINED_0001` there.

**Victim dependence.** Valid evasion depends strongly on the victim. The FT-Transformer stays at
≤ 0.97% Valid ASR under every PrimAttack configuration and at 0.00–0.18% under every baseline. The
CICIDS2017 CNN is the most exposed victim. Results are reported per victim and dataset and are
never pooled.

**Claim boundary.** All results are feature-space proxies on CICFlowMeter aggregates of a
chronological within-label test split. No PCAP is edited or replayed. Neither packet-level
realizability nor preserved malicious functionality is claimed, for PrimAttack or for the
matched-support baselines. The seeds are attack seeds on one frozen victim per architecture
(CICIDS2018: the training-seed-42 replicate). They quantify attack-run variability, which is
tiny (SD ≤ 0.20 pp for any PrimAttack Valid ASR, ≤ 4.33 pp for any baseline Raw ASR). They do
not quantify victim-training variability. PrimAttack's search queries validator_v2, while the
baselines do not. This is part of its threat model and part of its valid-success advantage.

**Protocol amendment disclosed.** During the first Exp B run, a methodological bug surfaced in
the shared PrimAttack optimizer core. A pinned (zero-headroom) padding control on CICIDS2018
flows without backward packets produced a NaN surrogate gradient. That crashed Prim-C&W and would
silently freeze Hybrid/Prim-PGD rows (sign(NaN) = 0). The fix, which cuts the autograd path of
pinned controls and fails loudly on any non-finite gradient, was applied before any optimizer
selection. Stage B was then re-run from scratch on both datasets. See `00_PROTOCOL.md` §7.
