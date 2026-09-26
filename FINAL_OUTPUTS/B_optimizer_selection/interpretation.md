**Selection (locked criterion).** Hybrid Search has the highest aggregate Valid Targeted ASR
at p75 over both datasets, all victims, classes and seeds: 4,602 / 57,600 = 7.990%. Prim-PGD
follows with 4,595 = 7.977% and Prim-C&W with 2,541 = 4.411%. Hybrid Search is therefore the
PrimAttack optimizer for Exp A and D, and Hybrid and Prim-PGD are the Exp C optimizers. The
margin between Hybrid and Prim-PGD is 7 flow-instances (0.012 pp) across 57,600 attacked
instances. The selection follows the pre-registered rule. It is not evidence that Hybrid is
the more effective search.

**Supporting paired evidence (seed 42, per dataset × victim).** Cochran's Q is significant for
CICIDS2017 CNN (Q = 1380.2) and CICIDS2018 MLP (Q = 24.4). It is not significant for the other
four victims. On CICIDS2017 FT and CICIDS2018 CNN/FT all three optimizers solve identical flow
sets (Q = 0). On CICIDS2017 MLP they differ by 2 flows (Q = 4.0, p = 0.135).
- CICIDS2017 CNN: Hybrid vs Prim-PGD is not significant (8 vs 4 discordant flows, +0.13 pp,
  Holm p = 0.39). Both beat Prim-C&W by about 22 pp (Hybrid +21.84 pp, 699 vs 0; Prim-PGD
  +21.72 pp, 696 vs 1; Holm p < 1e-150).
- CICIDS2018 MLP: Prim-C&W is higher than Hybrid (+0.41 pp, 13 vs 0, Holm p = 0.0005) and
  than Prim-PGD (+0.44 pp, 14 vs 0, Holm p = 0.0004). The effect is statistically clear but
  practically small (under 0.5 pp of 3,200 flows).

**Why.** Hybrid and Prim-PGD share the same projected sign-momentum update and differ mainly in
Hybrid's exact padding enumeration and step adaptation. They find essentially the same flows.
Prim-C&W restarts every binary-search stage from the clean flow and minimizes primitive cost
jointly with the margin. It is markedly weaker on CICIDS2017 CNN (−21.8 pp). On CICIDS2018 MLP
every valid success of every optimizer is timing-only (p = 0) and mostly Recon, while every
invalid raw success uses padding and fails a MINED rule (`MINED_0001` in every Hybrid case
checked). Prim-C&W's cost term penalizes padding. At seed 42 it finds 23 timing-only valid
successes, against 10 for Hybrid and 9 for Prim-PGD. Validity gaps are similar across
optimizers: there is none on CICIDS2017, and on CICIDS2018 gaps reach 15.01 pp on CNN. Every
invalid example there fails only the MINED category (Exp E).

**Cost.** Under the matched cap of 256 victim evaluations per flow, Hybrid uses the fewest
evaluations (188.2–188.4 on CICIDS2017, 192.8–192.9 on CICIDS2018, vs 190.6 / 197.4 for Prim-PGD
and Prim-C&W). It is also the fastest on the FT-Transformer (12.2–12.8 vs 13.7–14.0 ms/flow);
on MLP/CNN all three take about 3 ms/flow. Seed variability is negligible (SD ≤ 0.13 pp in Valid
Targeted ASR). Prim-C&W is deterministic (SD 0).

**Thesis reading (Contribution 1).** Given this primitive parameterization, the valid-success
ceiling is set mainly by the attack space (budget, capabilities, validator) and by the victim,
not by the optimizer. Two of the three gradient searches reach the same flows. The selected
Hybrid Search is at least as effective as the strongest alternative, at the lowest query cost.
