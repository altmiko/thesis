# D5 - Attacker overhead of the valid successes

## Design

Analysis only. This experiment runs no new attack; it reads the canonical FINAL-suite per-row
artifacts. The metrics come from these sources:

- Amoeba (CoNEXT 2023, Table 1):
  - data overhead DO = added bytes / (forward + backward bytes + added bytes);
  - time overhead TO = added flow time / (added time + original duration).
- PLAA (2026, Table VI): the relative change of packet length, inter-arrival time and rate
  between the adversarial and the original flow.
- Tamaraw (CCS 2014, Fig. 4) and FRONT (USENIX Security 2020, Fig. 7): cost curves. Valid ASR is
  shown as a function of the overhead the attacker accepts.
- DeTorrent (PETS 2024, Fig. 6): diminishing returns over the budget ladder.

| Arm | FINAL stage | Budgets |
|---|---|---|
| `untargeted_pgd` | Exp A PrimAttack (`primattack_untargeted`, Prim-PGD) | p75, unbounded |
| `targeted_pgd` | Exp B/C (`primattack_targeted_optimizers` + `_budgets`) | p50, p75, unbounded |
| `targeted_hybrid` | Exp B/C | p50, p75, unbounded |

- Overheads are computed from each realized adversarial flow and its source test flow. Added
  bytes = Δ`Total Length of Fwd Packet`; added time = Δ`Flow Duration`.
- Valid ASR is over 3,200 attempted flows per victim and seed.
- Overhead statistics are taken over the valid successes, with the three attack seeds pooled.

Run: `python ablations/D5_overhead/run.py`. Tables: `results/report.md`,
`results/overhead_summary.csv`, `results/cost_curves.csv`.

## Results

### 1. Byte overhead is zero

- No valid success in any arm, budget, victim or dataset adds a byte (`with padding` = 0).
  DO = 0 everywhere.
- Every success is a timing manipulation. This agrees with the FINAL suite's
  timing-only finding.

### 2. Time overhead is large

Medians over valid successes:

| dataset | victim | Exp A (untargeted Prim-PGD) p75: Valid ASR / median TO | unbounded: Valid ASR / median TO |
|---|---|---|---|
| 2017 | mlp | 4.09% / 0.33 | 22.97% / 0.96 |
| 2017 | cnn | 13.47% / 0.31 | 59.94% / 0.86 |
| 2017 | ft_transformer | 0.12% / 0.06 | 0.59% / 0.999 |
| 2018 | mlp-s42 | 2.53% / 0.45 | 44.32% / 0.99 |
| 2018 | cnn-s42 | 1.16% / 0.55 | 26.28% / 0.97 |
| 2018 | ft_transformer-s42 | 0% / n/a | 0.12% / 0.999 |

- At p75, a typical success stretches the flow by 31–55% of its new duration on MLP and CNN.
  `Flow Bytes/s` and `Flow Packets/s` drop by the same fraction.
- Without the budget cap, a typical success is mostly added delay: TO ≥ 0.86. Example: the
  2018 MLP median success turns a flow into one that is ~99% idle time
  (median `Flow IAT Mean` ≈ 159× the original).

### 3. Diminishing returns are inverted

The budget ladder (targeted arms: p50 → p75 → unbounded) shows small gains up to p75 and a
large jump beyond it:

| dataset | victim | p50 → p75 | p75 → unbounded |
|---|---|---|---|
| 2017 | mlp | +1.78 pp | +18.84 pp |
| 2017 | cnn | +4.06 pp | +46.44 pp |
| 2018 | mlp-s42 | +0.09 pp | ≈ +24 pp |
| 2018 | cnn-s42 | 0 | +26.09 pp |

- The extra evasion beyond p75 is paid for with TO ≈ 0.86–0.99, i.e. delays far outside the
  per-class train p75 duration change.
- The calibrated budget, not the optimizer, keeps Valid ASR low.

### 4. Cost curves

These are in `results/report.md`. At an attacker-accepted TO ≤ 0.10, Valid ASR is:

- ≤ 0.12% on every victim;
- except the 2017 CNN: ≤ 1.18% at p50/p75.

Low-overhead evasion is essentially absent.

Unbounded runs can have lower ASR than p75 runs at the same TO cap. Example: 2017 MLP at
TO ≤ 0.5 reaches 2.35% unbounded vs 3.58% at p75. The reason is how the search scores and
moves:

- It is success-first and ranks successes by box-normalized cost.
- In a larger box, its normalized steps are coarser, so it settles on larger absolute delays.

So unbounded overheads are upper bounds on the overhead an attacker needs, not minima.
