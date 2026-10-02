# P2 - Coupled feature-recomputation ablation (`NoCoupledPhi`)

## Design

The canonical map φ (`CICIDS2017PrimitiveModel.generate`) writes 23 features. P2 removes the
propagation from the statistics each primitive sets directly to the statistics φ derives from
them, and keeps the rest of the attack.

- `ablations/common/phi_mapping.py` traces φ's data flow from its source code. A write is
  *direct* if its value depends only on the primitives and source columns, *derived* if φ
  computes it from another written value. Result: 7 direct features (Total Length of Fwd
  Packet, Fwd Packet Length Min/Max, Fwd IAT Total/Std/Max/Min) and 16 derived ones
  (`results/phi_mapping.md`, written before the run).
- `full_phi` (reference, `recompute_mode="full_phi"`): FINAL targeted Hybrid configuration.
- `direct_only` (P2, `recompute_mode="direct_only"`): the same search sees a map that keeps φ's
  direct writes and holds the derived features at the source values
  (`DirectOnlyPrimitiveModel`). validator_v2 is not in its success predicate (A6: removing the
  gate leaves every FINAL Hybrid flow unchanged).
- The primitives P2 returns are realized again through canonical φ
  (`phi_mapping.realize_full_phi`) and scored by the victim and the unchanged validator_v2. Only
  that full-φ outcome is compared with the reference; the reduced flow is kept as a diagnostic.

Protocol: targeted → Benign, both datasets, three victims, four classes × 800 frozen
clean-correct flows, p75 and unbounded, attack seeds 42/2024/2026, 256 evaluations per flow.

Run: `python ablations/thesis_ablations/P2_coupled_phi/run.py --device cuda` (re-analyze: `--skip-run`). Full
report: `results/P2_COUPLED_PHI_ABLATION.md`; generated tables: `results/report.md`.

## Results

Valid ASR in % (mean of seeds; P2 = full-φ Valid of the P2 primitives). `*` = significant
seed-42 McNemar after Holm over 12 cells. Reduced-space Raw is what the P2 search believed.

| dataset | victim | budget | reference | P2 | P2 reduced-space Raw |
|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09 | 3.08* | 4.16 |
| 2017 | mlp | unb | 22.94 | 14.90* | 28.91 |
| 2017 | cnn | p75 | 13.25 | 12.07* | 13.41 |
| 2017 | cnn | unb | 59.69 | 53.75* | 60.81 |
| 2017 | ft_transformer | p75 | 0.12 | 0.12 | 0.12 |
| 2017 | ft_transformer | unb | 0.55 | 0.45 | 0.41 |
| 2018 | mlp-s42 | p75 | 0.78 | 0.69 | 0.69 |
| 2018 | mlp-s42 | unb | 24.80 | 24.02* | 1.61 |
| 2018 | cnn-s42 | p75 | 0.00 | 0.00 | 0.00 |
| 2018 | cnn-s42 | unb | 26.09 | 25.64* | 20.03 |
| 2018 | ft_transformer-s42 | p75 | 0.00 | 0.00 | 0.00 |
| 2018 | ft_transformer-s42 | unb | 0.12 | 0.12 | 0.00 |

### Findings

- Coupled recomputation raises Valid ASR in 6 of 12 cells: 1.0–8.0 pp on the 2017 MLP and
  CNN, 0.45–0.78 pp on the 2018 MLP and CNN unbounded cells. P2 has no valid success the
  reference lacks (0 P2-only flows in all 36 seed × cell comparisons). FT-Transformer and the
  2018 p75 cells show no difference.
- Of the successes the P2 search saw on the reduced flow, 76.6% (2017) and 93.8% (2018) survive
  when the same primitives go through φ. Every loss is a changed victim prediction; validator_v2
  accepts every full-φ flow.
- The reduced search misjudges evasion in both directions: it overstates it on the 2017 MLP
  (28.91% seen vs 14.90% real, unbounded) and misses most of it on the 2018 MLP (1.61% seen vs
  24.02% real, unbounded).
- The derived timing features of φ's delay block (Fwd IAT Mean, Flow Duration, Flow IAT Mean,
  Flow IAT Max) account for the lost successes: their full-φ values alone end 96.9% (2017) and
  100% (2018) of them. Padding-derived features never matter (no valid success pads).
- validator_v2 accepts 8.3–78.5% of the reduced flows although ≥ 99.6% of the accepted ones
  break φ's identity `Fwd IAT Mean = Fwd IAT Total / (Nf − 1)`. No EXTRACTOR rule covers forward
  IAT means, IAT totals or flow duration. Capability-aware PrimAttack is unaffected: its φ flows
  pass the Level-B identity and timing checks on every row.
- Sanity: `full_phi` reproduces the FINAL targeted Hybrid cells on all 115,200 flows; pairing,
  primitive identity, capability masking, validator recomputation and final counting checks
  all pass (`results/sanity_checks.json`).
