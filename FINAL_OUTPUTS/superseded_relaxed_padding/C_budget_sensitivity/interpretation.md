**Valid Targeted ASR grows with attacker capability, but the budget that matters differs by
dataset.** Values are for Hybrid Search. Prim-PGD differs by ≤ 0.12 pp everywhere except CICIDS2018
CNN unbounded, where it reaches 25.10% vs 22.68%.

| Victim | p50 | p75 | unbounded |
|---|---|---|---|
| CICIDS2017 MLP | 6.31% | 11.06% | 41.90% |
| CICIDS2017 CNN | 13.10% | 36.15% | 70.48% |
| CICIDS2017 FT-Transformer | 0.16% | 0.44% | 0.88% |
| CICIDS2018 MLP | 0.17% | 0.29% | 24.05% |
| CICIDS2018 CNN | 0.00% | 0.00% | 22.68% |
| CICIDS2018 FT-Transformer | 0.00% | 0.00% | 0.09% |

**Planned tests (seed 42, Holm over the two adjacent comparisons).** On CICIDS2017 both steps
are significant for every victim and both optimizers. p50 → p75 adds +4.75 pp (MLP), +23.09 pp
(CNN) and +0.28 pp (FT). p75 → unbounded adds +30.94 / +34.34 / +0.44 pp. On CICIDS2018
p50 → p75 changes little. It is significant only for Prim-PGD on MLP (+0.19 pp, Holm p = 0.031).
Hybrid MLP (+0.16 pp, p = 0.063), CNN (0 discordant flows) and FT are not significant.
p75 → unbounded is large and significant for MLP (+23.81 pp) and CNN (+22.66 pp; Prim-PGD
+25.09 pp). It is not significant for FT (+0.09 pp, 3 vs 0 flows; Holm p = 0.5), although its
Cochran's Q is borderline (p = 0.0498). Across all 24 adjacent comparisons, every discordant
flow favors the larger budget except one (Prim-PGD, CICIDS2018 MLP, 764 vs 1). A larger box
almost never loses a success.

**Validity gap vs budget.** On CICIDS2017 the gap is 0.00 pp at every budget: every raw
success is valid. On CICIDS2018 CNN, raw targeted success is already 14.05% at p50 and 15.00% at
p75, but none of it is valid (gap 14.05 / 15.00 pp). At p75 every such example uses padding and
breaks the train-mined `MINED_0001` invariant (Exp E). Unbounded raises raw success to 45.85% and valid
success to 22.68%. The gap widens to 23.18 pp, so validity absorbs a large share of the extra
raw success.

**Reading.** The train-calibrated p50/p75 budgets on CICIDS2018 allow little timing freedom.
The median per-flow delay cap at p75 is about 31 ms, against about 0.9 s on CICIDS2017 (Exp A
primitive-cost table). Valid evasion there needs larger timing changes, which only the
envelope-only unbounded budget allows. On CICIDS2017 the same percentile budgets already allow
sizable valid evasion of MLP and CNN.
FT-Transformer stays at or below 0.88% under every budget on both datasets. Budget is thus a
first-order control of valid attack success (Contribution 5). The p75 headline figures are
budget-conditional and should always be reported with their budget. The unbounded budget is
a stress test bounded only by the train-p99 feature envelope and the DoS/DDoS min-rate floor.
It is not a realistic attacker budget.
