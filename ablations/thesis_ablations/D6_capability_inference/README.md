# D6 - Capability inference removed

## Design

| Arm | Controls |
|---|---|
| reference | $u = M(x) \odot (p, D, s)$ |
| `no_capability` | $u = (p, D, s)$, i.e. $M(x) = 1$ for every flow |

$M(x)$ is `CICIDS2017PrimitiveModel.infer_capabilities`. It allows padding only when the flow has
forward packets, forward payload and no zero-length forward packet
(`Fwd Packet Length Min > 0`). It allows timing only when the flow has at least two forward
packets and a positive `Fwd IAT Total`. The reference applies the mask in three places: the
per-flow box, the projection and the canonical map φ. The ablated arm builds the box from the
numeric train-envelope/budget headroom alone. Its φ then applies padding or delay to any flow.

Everything else is the shared protocol (`../../README.md`):

- the same Hybrid Search, with validator_v2 `hybrid_valid` in the success predicate;
- the same frozen flows, victims, seeds and budgets;
- **the same final validator_v2** (SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED, source-conditioned)
  scores both arms.

There is one necessary deviation. Outside the capability-admissible set φ is not differentiable
everywhere. Example: padding a flow whose packet-length variance is 0 gives
d sqrt(var)/dp = ∞·0. The canonical search raises on a non-finite gradient. The ablated arm
instead zeroes the affected gradient coordinates and counts them. On CICIDS2018 this happened
for 69 flows per (victim, budget), pooled over seeds; on CICIDS2017 it never happened.

Run: `python ablations/thesis_ablations/D6_capability_inference/run.py --device cuda`. Full tables are in
`results/report.md`.

## Results

**Valid ASR** (mean of seeds 42/2024/2026; 3,200 flows per seed). Paired McNemar at seed 42 with
Holm correction. The 2018 MLP unbounded row is discussed below.

| dataset | victim | budget | reference | no_capability | Δ (pp) | Raw ASR, no_capability | p (Holm) |
|---|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09% | 0.19% | −3.91 | 13.97% | 6.7e-28 |
| 2017 | mlp | unb | 22.97% | 0.71% | −22.26 | 72.62% | 3.6e-155 |
| 2017 | cnn | p75 | 13.47% | 2.70% | −10.77 | 49.07% | 2.1e-75 |
| 2017 | cnn | unb | 59.94% | 7.25% | −52.69 | 98.16% | <1e-300 |
| 2017 | ft_transformer | p75 | 0.12% | 0.03% | −0.09 | 25.25% | 0.5 |
| 2017 | ft_transformer | unb | 0.55% | 0.03% | −0.52 | 25.69% | 6.1e-05 |
| 2018 | mlp-s42 | p75 | 2.53% | 0.22% | −2.31 | 31.69% | 9.0e-17 |
| 2018 | mlp-s42 | unb | 44.36% | 38.52% | −5.84 | 66.61% | 4.5e-06 |
| 2018 | cnn-s42 | p75 | 1.16% | 0.03% | −1.12 | 17.72% | 3.3e-08 |
| 2018 | cnn-s42 | unb | 26.19% | 4.87% | −21.31 | 68.11% | 5.2e-144 |
| 2018 | ft_transformer-s42 | p75 | 0.00% | 0.00% | 0.00 | 0.36% | 1 |
| 2018 | ft_transformer-s42 | unb | 0.12% | 0.34% | +0.22 | 1.79% | 0.12 |

### Findings

1. **Removing $M(x)$ lowers Valid ASR in 10 of 12 cells. Raw ASR explodes at the same time.**
   - Without the mask, the gradient pushes padding into flows that cannot carry it. The
     objective is met (Raw ASR up to 98%), but validator_v2 rejects those flows.
   - On CICIDS2017 PROTOCOL rejects every capability-violating objective hit, at both budgets
     and for all three victims. On CICIDS2018 MINED rejects most of them as well.
   - The search then spends its 256 evaluations in an infeasible part of the space. Valid
     successes inside the capability set drop sharply. Example: 2017 CNN, unbounded budget,
     flows whose source permits timing: 81.20% → 9.82%.
   - Every timing-capable flow still received refinement steps (`never refined` = 0). The loss
     comes from the direction of search, not from a starved budget.
   - So $M(x)$ is not redundant with the validator. A validator works as an accept/reject gate
     and cannot steer the search. $M(x)$ restricts the search to the feasible subspace.

2. **validator_v2 rejects all padding violations but misses one kind of timing violation.**
   - Zero valid successes in the ablated arm use padding the source forbids. PROTO_0080 and
     MINED catch filled empty packets and padding without payload.
   - On CICIDS2018, 1,512 valid successes add forward delay to flows with a single forward
     packet (source reason `SINGLE_FWD_PACKET`):
     - MLP, unbounded: 1,461 of 3,698. All are Recon; at seed 42, 500 of the 800 Recon flows.
     - CNN, unbounded: 25.
     - FT-Transformer, unbounded: 24.
     - MLP, p75: 2.
   - These flows have no forward inter-arrival gap. Example (2018 Recon, TCP, 1 forward +
     1 backward packet):

     | feature | source | adversarial |
     |---|---|---|
     | `Fwd IAT Total` | 0 | 1.2e7 µs |
     | `Fwd IAT Mean` / `Max` / `Min` | 0 | 1.2e7 µs |
     | `Flow Duration` | 2 µs | 1.2e7 µs |

   - The 2018 and 2017 validators both accept this flow; checked directly on this example. No
     validator_v2 layer has the rule `Total Fwd Packet = 1 ⇒ Fwd IAT Total = 0`.
   - The realizability/primitive-feasibility check also passes these flows: the identity
     `Fwd IAT Mean = Fwd IAT Total / max(N_f − 1, 1)` holds trivially when `N_f = 1`.
   - On these flows, capability inference is the **only** safeguard against a physically
     impossible manipulation.
   - The 2018 MLP unbounded Valid ASR (38.52%) therefore overstates achievable evasion. Only
     2,237 of its 3,698 valid successes respect the source's capability.
   - The FT-Transformer 2018 unbounded "gain" (0.12% → 0.34%) is the same artifact: 24 of the
     33 ablated successes add delay to single-packet flows.

3. **Thesis implication.** The capability mask is load-bearing in two ways:
   - **efficiency:** search stays in the feasible subspace;
   - **soundness:** it blocks single-forward-packet timing manipulations that validator_v2 does
     not encode.

   Anyone reusing validator_v2 as the only feasibility gate (for example for other attacks)
   inherits this gap.
