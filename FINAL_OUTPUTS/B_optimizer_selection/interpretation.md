**Selection.** Under the pre-registered rule, Hybrid Search and Prim-PGD tie exactly: 1,752 valid
targeted successes each out of 57,600 attempts (3.042%). The tie-break (fewer mean victim
evaluations per flow) selects **Prim-PGD** (188.5 vs 189.6). Prim-C&W is third with 861 (1.495%).
Budget sensitivity (Exp C) therefore uses Prim-PGD and Hybrid Search. Before amendment A2 the
same rule had selected Hybrid (7.990% vs 7.977%); the capability fix, not the rule, changed the
outcome.

**Why Hybrid and Prim-PGD coincide.** With the capability-aware padding rule, only 1 (CICIDS2017)
or 12 (CICIDS2018) of 3,200 attacked flows per victim may be padded, so Hybrid's distinguishing
component, the exhaustive integer padding enumeration, has almost nothing to enumerate. What
remains for both is projected sign-momentum descent on the timing controls (delay, shape) from
the clean flow plus restarts, with the same 256-evaluation cap. The two reach identical success
sets: on every victim their Valid Targeted ASR is equal (4.09% / 13.25% / 0.12% on CICIDS2017,
0.78% / 0.00% / 0.00% on CICIDS2018), and on the CICIDS2017 CNN, where the planned McNemar tests
run, Hybrid vs Prim-PGD has 0 discordant flows (Holm p = 1).

**Prim-C&W.** Its cost-penalized objective (normalized primitive cost + c · margin) finds fewer
successes where the needed delay is large: on the CICIDS2017 CNN it reaches 4.00% vs 13.25% for
the other two (296 vs 0 discordant flows; Holm p = 2e-65 for both comparisons). Elsewhere it ties
within 0.03 pp (CICIDS2018 MLP 0.75% vs 0.78%; Cochran's Q p = 0.368). Cochran's Q is not
significant, or not computable because all three are identical, on the other five victims.

**Validity.** Every targeted raw success of every optimizer is valid (Validity Gap 0.00 pp in all
18 cells): the realized-flow search keeps only validator-accepted incumbents, and the timing-only
flows cannot trigger the empty-packet rule. The optimizers spend a similar per-flow budget
(187–191 victim evaluations) because flows without primitive headroom stop at the identity.

**Reading.** After the fix, the optimizer choice matters little: timing is a low-dimensional
search that simple projected descent already saturates within the budget. The victim and the
timing budget determine success (Exp C), not the optimizer.
