# PrimAttack ablations

Ablation and sensitivity experiments for PrimAttack. Component ablations cite their
published design sources in each experiment's `run.py` docstring and `README.md`.
Each experiment has its own folder with `run.py`, `README.md` and `results/`
(`report.md`, `summary.csv`, `tests.csv`, extra tables and per-dataset artifacts).
The per-row npz artifacts are git-ignored but retained locally.

| Folder | Question | New attack runs |
|---|---|---|
| `reference/` | Shared reference arm; reproduces the FINAL suite's Hybrid cells flow-for-flow | yes |
| `A1_hybrid_components/` | What does each Hybrid Search component contribute? (leave-one-out + coverage) | yes |
| `A2_shape_allocation/` | Does the learned delay allocation `shape` matter vs a fixed allocation? | yes |
| `A3_loss_function/` | Margin vs cross-entropy vs DLR refinement loss; are gradients zero? | yes |
| `A4_mimicry_objective/` | Victim-guided search vs imitating the nearest Benign train flow | yes |
| `B1_convergence/` | Does more steps / restarts / queries raise Valid ASR? (anytime curve) | yes |
| `B5_validator_layers/` | Valid ASR when one validator_v2 layer (SCHEMA, EXTRACTOR, PROTOCOL, MINED) is removed | yes |
| `D5_overhead/` | Bytes and time the attacker pays per valid success (cost curves) | no (FINAL artifacts) |
| `D6_capability_inference/` | `u = M(x) ⊙ (p, D, s)` vs `u = (p, D, s)`, both judged by the same validator_v2 | yes |
| `V1_single_fwd_packet_rule/` | Toggle-able validator rule `Total Fwd Packet ≤ 1 ⇒ forward IAT = 0` (closes the D6 gap), on capability-aware and capability-ablated PrimAttack | yes |
| `seeded_rows/` | How does random clean-correct row selection change p75 and unbounded Prim-PGD Valid ASR? | yes (new cohorts for seeds 2024/2026; seed 42 copied from FINAL) |

## Key results

| Experiment | Result |
|---|---|
| reference | Reproduces all 72 FINAL p75 Hybrid cells flow-for-flow (57,600 flows) |
| A1 | The gradient refinement is the whole attack (Valid ASR 0 without it). Random restarts matter for 2017 CNN (13.47% → 4.50% at p75). Padding sweep, momentum, adaptive step, surrogate floor and the in-search validator change nothing. The reference covers every arm's successes except ≤ 4 flows per cell. |
| A2 | Proportional delay (shape 0) almost never evades at p75 (≤ 0.06%). Uniform delay matches the learned shape at p75 but loses 8.9 pp on 2018 MLP unbounded. Learning `shape` matters. |
| A3 | CE equals the margin loss; DLR is never better. Margin/CE have 0 zero-gradient steps on every victim, including FT-Transformer (no gradient masking). |
| A4 | Benign mimicry loses 1.0–20.8 pp in 7 of 12 cells and finds a subset of the reference's successes. Victim guidance matters. |
| B1 | 2–4× steps/restarts/queries add ≤ 0.09 pp (n.s.). The attack has converged. |
| B5 | Dropping any validator_v2 layer changes nothing (0 discordant flows in 48 comparisons). For capability-aware PrimAttack the validator never binds. |
| D5 | No valid success adds a byte. Median time overhead of MLP/CNN successes is 0.31–0.55 at p75 and ≥ 0.86 unbounded. Valid ASR at TO ≤ 0.10 is ≤ 1.18%. |
| D6 | Removing M(x) lowers Valid ASR in 10 of 12 cells (e.g. 2017 CNN unbounded 59.94% → 7.25%) while Raw ASR rises up to 98%. validator_v2 misses delay added to single-forward-packet flows: 1,512 such "valid" successes on CICIDS2018. |
| V1 | The rule `Total Fwd Packet ≤ 1 ⇒ forward IAT = 0` accepts all 2.91 M genuine flows and never binds for capability-aware PrimAttack (identical Valid ASR in all 12 cells). It removes all 1,512 single-packet timing successes of the capability-ablated attack (2018 MLP unbounded 38.52% → 23.30%). |
| seeded_rows | New seed-specific random cohorts change Valid ASR (2017 CNN p75: 13.47%, 12.47%, 12.88%; 2018 MLP p75: 2.53%, 2.25%, 1.88%). Paired p75 vs unbounded Valid ASR differs in 5/6 victims after Holm on seed 42; this is not a cross-method or cross-seed paired test. |

## Shared protocol

Unless an arm changes it, every cell is the FINAL suite's Hybrid configuration (amendment A5,
untargeted arm):

* datasets CICIDS2017-DistriNet and CICIDS2018-DistriNet; victims `mlp`, `cnn`, `ft_transformer`
  (2017 canonical checkpoints) and `mlp-s42`, `cnn-s42`, `ft_transformer-s42` (2018);
* the frozen 800 clean-correct test flows per (dataset, victim, class) from
  `FINAL_OUTPUTS/runs/<dataset>/baselines_untargeted/selection.json` (sha256-verified), classes
  DoS, DDoS, Recon, BruteForce;
* attack seeds 42, 2024, 2026; untargeted objective (realized flow leaves its source class);
  joint primitive mode; capability-aware padding/timing;
* budgets p75 (`maximum-evaluated`, the headline) and envelope-only `unbounded`, both from the
  train-fit calibration;
* Hybrid Search: 40 steps, lr 0.1, restarts until 256 victim evaluations per flow are spent;
  validator_v2 `hybrid_valid` in the search success predicate.

Every realized flow, in every arm, is scored afterwards by the same full validator_v2
(SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED, given its source flow); each layer is stored per row.

Analysis: per (dataset, victim, budget), the four classes are pooled (3,200 flows per seed).
ASR = mean over the three attack seeds (seed range shown). Each arm is compared with the
reference on the same flows by a paired McNemar test at seed 42 (exact binomial below 25
discordant pairs) with a Newcombe 95% CI; Holm corrects within one experiment. Victims and
datasets are never pooled.

These ablations are added analyses, not part of the locked FINAL protocol
(`FINAL_OUTPUTS/00_PROTOCOL.md`). The global claim boundary of `CLAUDE.md` applies: feature-space
proxies on CICFlowMeter aggregates, no PCAP edited or replayed.

## Code

* `common/hybrid.py` - `HybridConfig` + `optimize_hybrid_ablation`: the canonical Hybrid Search
  (`attack.primitive_optimizer.optimize_primitive_candidates`) with switchable components. The
  canonical module is not modified. Default config = canonical behaviour.
* `common/runner.py` - cell runner (frozen flows, victims, calibration, search, final
  validator_v2, npz + `cells.json`); `Condition` = one arm (Hybrid config, validator layers in the
  search gate, capability masking on/off). Finished cells are skipped on re-runs.
* `common/analysis.py` - comparison with the reference, McNemar/Newcombe/Holm, report tables.
* `common/cli.py` - shared command line of every `run.py`.

## Running

From the repo root in the `thesis` env (`PYTHONPATH` is set by the scripts):

```
python ablations/reference/run.py --device cuda          # first: the shared reference arm
python ablations/A1_hybrid_components/run.py --device cuda
...                                                       # A2, A3, A4, B1, B5, D6 likewise
python ablations/D5_overhead/run.py                      # analysis of FINAL artifacts only
python ablations/seeded_rows/run.py --device cuda  # new random row cohort per seed
python ablations/seeded_rows/analyze.py --device cuda
```

Options: `--datasets`, `--budgets`, `--seeds`, `--victims`, `--classes`, `--conditions`,
`--limit-rows` (smoke tests; use a separate `--results-dir`), `--skip-run` (re-analyze only),
`--skip-analysis`. `reference/run.py`'s analysis writes `results/reproduction_check.md`: every
p75 reference cell vs `FINAL_OUTPUTS/runs/<dataset>/primattack_hybrid_objective_untargeted`.
