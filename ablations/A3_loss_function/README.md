# A3 - Refinement loss: margin vs cross-entropy vs DLR

## Design

This follows two published sources:

- Croce & Hein, AutoAttack (ICML 2020): the CE vs CW-margin vs DLR losses (Sec. 4.2,
  Tables 9–11) and the fraction of zero gradients (Fig. 2);
- Pintor et al., Indicators of Attack Failure (I1, unavailable gradients).

Only the loss that drives the refinement gradient changes. Everything else stays as in the
reference:

- the success predicate;
- the incumbent ordering, which uses the objective margin;
- the step adaptation, also on the margin;
- the padding enumeration.

| Arm | Loss (untargeted, source class y) |
|---|---|
| reference | `z_y − max_{i≠y} z_i` |
| `loss_ce` | `log softmax(z)_y` |
| `loss_dlr` | `(z_y − max_{i≠y} z_i) / (z_π1 − z_π3)` |

Every arm counts refinement steps whose gradient is exactly zero on all free control
coordinates.

Run: `python ablations/A3_loss_function/run.py --device cuda`. Full tables:
`results/report.md`, `results/zero_gradients.csv`.

## Results

**Valid ASR**:

- `loss_ce` equals the reference in 11 of 12 cells. The exception is 2018 CNN unbounded:
  +0.06 pp (2 vs 0 discordant flows at seed 42, n.s.).
- `loss_dlr` is equal or within 0.04 pp of the reference everywhere except 2017 CNN, where it
  is slightly worse:
  - p75: 12.98% vs 13.47% (0 vs 16 discordant flows, p_Holm = 7.3e-4);
  - unbounded: 59.48% vs 59.94% (p_Holm = 0.045).

**Zero gradients** (all seeds and classes; about 0.9 M refinement steps per cell):

- Margin and CE: 0 zero-gradient steps on every victim, dataset and budget.
- DLR: up to 16.4% of steps (2018 MLP unbounded). Other examples:
  - 2018 CNN unbounded: 13.0%;
  - 2017 CNN unbounded: 3.9%.

  DLR's normalization `z_π1 − z_π3` makes the loss flat or saturated far from the decision
  boundary.
- FT-Transformer: 0 zero-gradient steps under all three losses.

### Findings

- **The loss is not what limits PrimAttack.** CE matches the margin loss, and DLR is never
  better.
- **No gradient masking.** The FT-Transformer's near-zero Valid ASR (0.12% / 0% at p75) comes
  with fully available gradients. Its robustness is not a gradient artifact; the reachable
  primitive box simply does not contain flows it misclassifies.
