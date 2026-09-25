# PrimAttack budget calibration

> **Partly superseded.** The frozen train-only calibration values remain applicable,
> but the `alpha` conversion below belongs to the replaced timing model. The active
> code bounds integer total `delay` and supports an evaluated envelope-only condition;
> see [`../full_thesis_methodology/02_primattack.md`](../full_thesis_methodology/02_primattack.md).

## Frozen evidence source

The calibration command reads only:

- `data/processed/CICIDS_2017_Distrinet/X_train_pristine.npy`;
- `data/processed/CICIDS_2017_Distrinet/y_train_cat.npy`;
- the frozen feature order and class mapping.

It does not load validation/test features, victim predictions, attack artifacts, or adversarial
success. The machine-readable output is
`artifacts/primattack/budget_calibration.json`. SHA-256 identifiers for the training arrays and
preprocessing manifest are embedded in that artifact.

Regenerate with:

```text
PYTHONPATH=".;src" python -m attack.primattack_budget \
  --output artifacts/primattack/budget_calibration.json
```

## Robust summaries

For every retained class (`DoS`, `DDoS`, `Recon`/PortScan, `BruteForce`) the artifact records
$n$, median, Q1, Q3, IQR, MAD, and P01/P05/P10/P25/P50/P75/P90/P95/P99 for duration, rates,
packet counts, byte totals, forward and combined packet-length measures, forward-IAT measures,
selected flags, and active-data-packet count. Protocol frequencies are stored separately.

Means and standard deviations are not used to choose the budgets because these quantities are
strongly skewed.

## Empirical named budgets

Padding uses the empirical distribution of positive class-conditional
`Fwd Packet Length Mean`. Since $p$ is discrete, each selected quantile is rounded to the
nearest legal byte. Timing uses the empirical distribution

$$r_i=\frac{|D_i-\operatorname{median}(D)|}{\max(\operatorname{median}(D),1\ \mu s)}.$$

The named levels are empirical quantiles, not arbitrary fractions of a tuned optimum:

- **restricted:** P25;
- **intermediate:** P50;
- **maximum-evaluated:** P75.

The P75 level is the maximum evaluated flow-level budget. It is not a universal network or
physical maximum.

### Exact frozen values

Each cell is `padding bytes per forward packet / maximum relative duration increase`.

| Class | Restricted | Intermediate | Maximum-evaluated |
|---|---:|---:|---:|
| DoS | 41 / 0.042725609756097564 | 47 / 0.5671219512195121 | 54 / 1.2682256097560975 |
| DDoS | 2 / 0.22199772960647784 | 2 / 0.4352960736292373 | 3 / 0.6874252351844157 |
| Recon (PortScan) | 2 / 0.0851063829787234 | 2 / 0.2127659574468085 | 10 / 0.5319148936170213 |
| BruteForce | 11 / 0.06174194934404961 | 12 / 0.11989485807317402 | 91 / 0.23366757427033438 |

Repeated discrete values are retained when empirical quantiles round to the same legal byte;
levels are not artificially separated.

## Calibrated feasible envelope

Named budgets are intersected per source flow with mathematical capability and train
plausibility:

$$B_{max}(x)=\min(B_{named},B_{train\text{-}p99}(x),B_{semantic}(x)).$$

For padding, the per-flow cap is the minimum remaining headroom to the complete-training P99
for forward max/min/mean and forward total bytes (the total-byte headroom is divided by
$N_f$), then capped by the named class budget. No imported MTU constant is used.

For timing, the named relative-duration limit is converted to an alpha cap:

$$\alpha_{relative}(x)=1+r_{named}\frac{\max(D,1)}{\max(T_f,\epsilon)}.$$

This cap is intersected with complete-training P99 headroom for the forward-IAT summaries and
flow duration. DoS/DDoS also intersect a semantic rate cap:

$$\alpha_{rate}(x)=1+\frac{D_{rate\ cap}-D}{\max(T_f,\epsilon)},\qquad
D_{rate\ cap}=\frac{(N_f+N_b)10^6}{R_{class,P05}}.$$

The fixed class P05 packet-rate thresholds are:

| Class | Minimum `Flow Packets/s` |
|---|---:|
| DoS | 0.6247806906700134 |
| DDoS | 1.0654268741607666 |

These thresholds are fitted before attacks and never selected using ASR.

## Hard enforcement

Both optimizers receive the per-flow feasible box. Final
`project_controls(raw, requested, bounds)` then independently clamps requested values, rounds
$p$, caps it by `floor(p_hi)`, clamps $\alpha$, and reapplies capability masks. Final features
are regenerated and reclassified after projection.

Every artifact records requested/projected controls, per-flow caps, actual relative duration
change, actual byte change, normalized primitive magnitudes, and primitive compliance.

## Primitive costs

Duration cost:

$$\Delta D=D_{adv}-D_0,\qquad c_D=\frac{\Delta D}{\max(D_0,\epsilon)}.$$

Byte/size cost uses the represented total forward plus backward byte quantity:

$$\Delta B=B_{adv}-B_0,\qquad c_B=\frac{\Delta B}{\max(B_0,\epsilon)}.$$

Also reported: $100p/\max(\text{Fwd Packet Length Mean},\epsilon)$, normalized $p$ and
$\alpha$ positions inside their per-flow caps, and rate retention. No exact packet-level bytes
added per individual packet are fabricated beyond the declared uniform-$p$ model.
