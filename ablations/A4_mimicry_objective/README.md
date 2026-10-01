# A4 - Mimicry objective

## Design

This follows the guidance ablation of Traffic Manipulator (Han et al., IEEE JSAC 2021,
Sec. VIII-B, Fig. 4). It asks whether victim-agnostic imitation of benign traffic, inside the
same primitive box, evades as often as the victim-guided search.

The `loss_mimicry` arm keeps the reference except for the loss of the refinement stage. That
loss becomes the squared distance in `asinh((x − center)/scale)` space between the relaxed
adversarial flow and a fixed anchor:

- The anchor is the nearest Benign flow of the **train** split to the source flow.
- The search covers all 1,153,431 Benign train flows on CICIDS2017 and 175,000 on CICIDS2018.

The following stay exactly as in the reference:

- the victim-based success predicate (untargeted ∧ validator_v2);
- the incumbent ordering;
- the step-size adaptation, keyed on the realized victim margin;
- the padding enumeration.

Run: `python ablations/A4_mimicry_objective/run.py --device cuda`. Full tables are in
`results/report.md`.

## Results

**Valid ASR** (mean of seeds 42/2024/2026). Paired McNemar at seed 42 with Holm correction.

| dataset | victim | budget | reference | loss_mimicry | Δ (pp) | seed 42: mimicry-only / ref-only | p (Holm) |
|---|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09% | 1.22% | −2.88 | 0 / 93 | 1.1e-20 |
| 2017 | mlp | unb | 22.97% | 15.58% | −7.39 | 0 / 228 | 4.0e-50 |
| 2017 | cnn | p75 | 13.47% | 5.14% | −8.33 | 0 / 268 | 8.4e-59 |
| 2017 | cnn | unb | 59.94% | 39.17% | −20.77 | 0 / 668 | 8.9e-146 |
| 2017 | ft_transformer | p75 | 0.12% | 0.12% | 0.00 | 0 / 0 | 1 |
| 2017 | ft_transformer | unb | 0.55% | 0.55% | 0.00 | 1 / 1 | 1 |
| 2018 | mlp-s42 | p75 | 2.53% | 1.01% | −1.52 | 0 / 49 | 4.9e-11 |
| 2018 | mlp-s42 | unb | 44.36% | 31.52% | −12.84 | 0 / 402 | 6.1e-88 |
| 2018 | cnn-s42 | p75 | 1.16% | 0.11% | −1.04 | 0 / 34 | 9.1e-08 |
| 2018 | cnn-s42 | unb | 26.19% | 26.40% | +0.21 | 12 / 8 | 1 |
| 2018 | ft_transformer-s42 | p75 | 0.00% | 0.00% | 0.00 | 0 / 0 | 1 |
| 2018 | ft_transformer-s42 | unb | 0.12% | 0.10% | −0.02 | 0 / 1 | 1 |

### Findings

- **Victim guidance matters for MLP and CNN.** Imitating the nearest Benign train flow loses
  1.0–20.8 pp of Valid ASR in 7 of 12 cells. In those cells, every flow that mimicry breaks is
  also broken by the reference (mimicry-only = 0 at seed 42). Mimicry finds a strict subset of
  the reference's successes.
- **Exceptions:**
  - FT-Transformer: unchanged, since nearly nothing is breakable at all.
  - CICIDS2018 CNN, unbounded budget: equal (26.40% vs 26.19%; 12 vs 8 discordant flows,
    n.s.). There, the reachable successes are the large timing dilations that any search
    reaches.
- **Successes found by mimicry are cheaper.** Their median normalized cost is lower in every
  cell that has successes. For example, 2018 MLP, unbounded budget:
  - median added delay: 2.9e6 µs (mimicry) vs 4.7e7 µs (reference);
  - median relative duration change: 112 vs 171.

  Benign imitation reaches the decision boundary only for the flows that lie close to it. It
  does so with smaller timing changes.
- This is consistent with Traffic Manipulator's finding that guidance toward the target model
  adds success over pure benign imitation.
