**Budget is the dominant factor.** With capability-aware PrimAttack almost every attacked flow is
timing-only, so the budget is essentially the added-delay cap. Valid Targeted ASR (Prim-PGD;
Hybrid is identical except where noted) grows monotonically with the budget:

- CICIDS2017 MLP 2.31% (p50) → 4.09% (p75) → 22.94% (unbounded); CNN 9.19% → 13.25% → 59.69%;
  FT-Transformer 0.12% → 0.12% → 0.59% (Hybrid 0.55%).
- CICIDS2018 MLP 0.69% → 0.78% → 24.76% (Hybrid 24.80%); CNN 0.00% → 0.00% → 26.09%;
  FT-Transformer 0.00% → 0.00% → 0.12%.

**Why the two optimizer curves overlap.** Hybrid's unique exact integer-padding enumeration is
inactive on almost every attacked flow, because `p` is pinned to zero. The remaining problem is
the same low-dimensional timing search used by Prim-PGD, and both use projected sign-momentum
updates with momentum 0.75, clean/random starts, the same realized-flow incumbent and the same
evaluation cap. Consequently they have identical p50 and p75 success sets and nearly identical
unbounded results. The only visible differences are CICIDS2017 FT-Transformer unbounded
(Prim-PGD 0.59%, Hybrid 0.55%) and CICIDS2018 MLP unbounded (24.76% vs 24.80%). These tiny
differences come from their fixed versus adaptive step/restart schedules, not from different
threat models.

**Inference (seed 42, Holm over the two adjacent comparisons, per optimizer).** Cochran's Q is
significant on every victim for both optimizers. p75 beats p50 only on the CICIDS2017 MLP and CNN
(57 and 130 flows gained, none lost; Holm p = 1.2e-13 and 1.1e-29); elsewhere the p50→p75 step
adds at most 3 flows (n.s.). Unbounded beats p75 on every victim (Holm p ≤ 2.4e-4) except
CICIDS2018 FT-Transformer (4 flows, Holm p = 0.25). No flow is ever lost by a larger budget
(B-only = 0 in every test), as expected from nested boxes.

**Why CICIDS2018 needs the unbounded box.** The calibrated p75 delay cap is small on CICIDS2018
(median per-flow cap about 31 ms vs about 0.9 s on CICIDS2017), and at p50/p75 the CICIDS2018 CNN
and FT-Transformer have no valid targeted success at all. Without the calibrated class budget
(envelope-only box) the same timing search reaches about a quarter of the MLP and CNN flows. The
p75 results therefore measure a deliberately conservative budget, not the limit of timing
manipulation.

**Validity at every budget.** The Validity Gap is 0.00 pp in all 36 cells: larger delays stay
inside the extractor identities and mined invariants, and no flow can be padded into an empty
packet. Valid success grows because the victims respond to larger delays, not because the
validator is relaxed. FT-Transformer remains the most robust victim at every budget (≤ 0.59%).
