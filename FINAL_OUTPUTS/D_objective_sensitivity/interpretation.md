**Targeted→Benign vs untargeted (Prim-PGD, p75, same flows and seeds).** On CICIDS2017 the two
objectives give almost the same Valid ASR: MLP 4.09% vs 4.09% and FT-Transformer 0.12% vs 0.12%
(identical success sets, 0 discordant flows), CNN 13.25% vs 13.47% (7 flows succeed only
untargeted; McNemar p = 0.016). There, the flows that timing moves out of their class almost all
move to Benign (seed 42: 423 of 431 untargeted CNN successes; the rest are DDoS↔Recon). On
CICIDS2018 the untargeted objective is clearly easier: MLP 0.78% vs 2.53% (56 vs 0 discordant
flows, p = 2.0e-13) and CNN 0.00% vs 1.16% (37 vs 0, p = 3.3e-9). These extra untargeted
successes are DDoS flows pushed into DoS, i.e. into another attack class, not into Benign; the 25
targeted successes on the MLP (4 DDoS, 21 Recon) reach Benign under both objectives.
FT-Transformer has no valid success under either objective on CICIDS2018.

**Direction of the effect.** No flow is valid-targeted-only on any victim (targeted-only = 0
everywhere), consistent with Benign being one of the classes an untargeted success may reach.

**Validity.** Both objectives have a Validity Gap of 0.00 pp on every victim. The objective
changes which flows succeed, not whether the successes are valid.

**Reading.** Reporting untargeted evasion (Exp A) and targeted evasion (Exp B/C) separately
matters only on CICIDS2018, where up to 1.75 pp of the untargeted Valid ASR is class-to-class
confusion between attack categories rather than evasion to Benign.
