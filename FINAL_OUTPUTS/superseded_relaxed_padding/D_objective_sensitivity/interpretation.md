**Valid success hardly depends on the objective.** With Hybrid Search at p75, Valid ASR is:

| Victim | Targeted → Benign | Untargeted | Δ | Planned McNemar test (seed 42) |
|---|---|---|---|---|
| CICIDS2017 MLP | 11.06% | 11.06% | 0.00 pp | 0 vs 0 discordant, p = 1 |
| CICIDS2017 CNN | 36.15% | 36.67% | −0.53 pp | 0 vs 17, p = 1.5e-5 |
| CICIDS2017 FT-Transformer | 0.44% | 0.50% | −0.06 pp | 0 vs 2, p = 0.5 |
| CICIDS2018 MLP | 0.29% | 0.19% | +0.13 pp | 4 vs 0, p = 0.125 |
| CICIDS2018 CNN | 0.00% | 0.03% | −0.03 pp | 0 vs 1, p = 1 |
| CICIDS2018 FT-Transformer | 0.00% | 0.00% | 0.00 pp | 0 vs 0, p = 1 |

Only CICIDS2017 CNN shows a significant difference: 17 flows that the untargeted search
evades validly but the targeted search does not. That is a small effect (0.53 pp). Its sign
agrees on seeds 2024 (−0.50 pp) and 2026 (−0.53 pp). A valid evasion that PrimAttack finds for a
malicious flow almost always lands in the Benign class. On five of six victims, requiring
"→ Benign" instead of "any other class" costs nothing measurable.

**Raw success depends on the objective.** Raw untargeted ASR is much higher than raw targeted
ASR on CICIDS2018 MLP (12.00% vs 1.28%). The two are similar elsewhere (e.g. CICIDS2018 CNN
15.31% vs 15.00%; CICIDS2017 identical to within 0.52 pp). At seed 42, 343 of the 379 invalid raw
untargeted successes on CICIDS2018 MLP are DDoS flows pushed into the DoS class, all by padding,
and all fail the validator (`MINED_0001`). The Validity Gap is therefore 11.81 pp untargeted vs
0.99 pp targeted. Without the validator, the untargeted objective would look far stronger. With
it, the difference disappears. This is a direct example of why objective and validity must be
reported separately (Contribution 2).

**Reading.** The targeted-to-Benign setting is the operationally relevant evasion goal and
PrimAttack's primary contribution. On these victims it costs almost no valid success compared
with the easier untargeted goal. The weak valid results on CICIDS2018 at p75 and on
FT-Transformer are therefore not an artefact of choosing the harder targeted objective.
