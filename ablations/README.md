# PrimAttack ablations

Ablation and sensitivity experiments for PrimAttack. Component ablations cite their
published design sources in each experiment's `run.py` docstring and `README.md`.
Each experiment has its own folder with `run.py`, `README.md` and `results/`
(`report.md`, `summary.csv`, `tests.csv`, extra tables and per-dataset artifacts).
The per-row npz artifacts are git-ignored but retained locally.

D6, A2, P1 and P2 live in `thesis_ablations/`; `thesis_ablations/All_Ablations.md` describes
their design, evaluation and results together.

| Folder | Question | New attack runs |
|---|---|---|
| `reference/` | Shared reference arm; reproduces the FINAL suite's Hybrid cells flow-for-flow | yes |
| `A1_hybrid_components/` | What does each Hybrid Search component contribute? (leave-one-out + coverage) | yes |
| `thesis_ablations/A2_shape_allocation/` | Does the learned delay allocation `shape` matter vs a fixed allocation? | yes |
| `A3_loss_function/` | Margin vs cross-entropy vs DLR refinement loss; are gradients zero? | yes |
| `A4_mimicry_objective/` | Victim-guided search vs imitating the nearest Benign train flow | yes |
| `B1_convergence/` | Does more steps / restarts / queries raise Valid ASR? (anytime curve) | yes |
| `B5_validator_layers/` | Valid ASR when one validator_v2 layer (SCHEMA, EXTRACTOR, PROTOCOL, MINED) is removed | yes |
| `D5_overhead/` | Bytes and time the attacker pays per valid success (cost curves) | no (FINAL artifacts) |
| `thesis_ablations/D6_capability_inference/` | `u = M(x) ⊙ (p, D, s)` vs `u = (p, D, s)`, both judged by the same validator_v2 | yes |
| `V1_single_fwd_packet_rule/` | Toggle-able validator rule `Total Fwd Packet ≤ 1 ⇒ forward IAT = 0` (closes the D6 gap), on capability-aware and capability-ablated PrimAttack | yes |
| `seeded_rows/` | How does random clean-correct row selection change p75 and unbounded Prim-PGD Valid ASR? | yes (new cohorts for seeds 2024/2026; seed 42 copied from FINAL) |
| `thesis_ablations/P1_realization_aware_search/` | Does scoring/selecting realized integer states during the search matter vs a continuous-state search realized once at the end? (targeted → Benign; own targeted reference arm) | yes |
| `thesis_ablations/P2_coupled_phi/` | Is φ's coupled recomputation of dependent features needed, vs a search that moves only each primitive's direct statistics? (targeted → Benign; own targeted reference arm; returned primitives re-realized through full φ) | yes |

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
| P1 | Targeted → Benign. A continuous-state search realized once at the end gives the reference's valid successes flow-for-flow in all 12 cells × 3 seeds (Δ Valid ASR 0.00 pp, 0 discordant flows, Holm p = 1). 0 of 14,635 continuous hits lose the Benign prediction after rounding (< 1 µs). 94.3% of them fail validator_v2 SCHEMA before realization. Realization is needed for validity, but doing it inside the search adds nothing at µs granularity. Its targeted reference reproduces all 115,200 FINAL targeted Hybrid flows. |
| P2 | Targeted → Benign. φ's 23 writes traced from code: 7 direct, 16 derived. A search that holds the derived features at source values loses 1.0–8.0 pp Valid ASR on 2017 MLP/CNN and 0.45–0.78 pp on 2018 MLP/CNN unbounded (6/12 cells significant, 0 P2-only valid flows); FT-Transformer and 2018 p75 unchanged. 76.6% (2017) / 93.8% (2018) of its reduced-space successes survive full φ, all losses through the victim's prediction. It overstates evasion on 2017 MLP (28.91% seen vs 14.90% real, unbounded) and misses it on 2018 MLP (1.61% vs 24.02%). The derived delay features (Fwd IAT Mean, Flow Duration, Flow IAT Mean/Max) account for ≥ 96.9% of the lost successes. validator_v2 accepts 8.3–78.5% of the inconsistent reduced flows (no EXTRACTOR rule on forward IAT mean or flow duration). Its targeted reference reproduces all 115,200 FINAL targeted Hybrid flows. |

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
  `realization_aware_search=False` (P1) scores candidates on the continuous state
  (`ContinuousSearch`) and realizes the returned candidate once (`realize_once`).
* `common/runner.py` - cell runner (frozen flows, victims, calibration, search, final
  validator_v2, npz + `cells.json`); `Condition` = one arm (Hybrid config, validator layers in the
  search gate, capability masking on/off). Finished cells are skipped on re-runs. `objective`
  selects untargeted (default) or targeted → Benign cells (P1); per-arm extra per-row arrays
  (`SearchDiagnostics.extra`) are stored in the npz. `Condition.recompute_mode="direct_only"`
  (P2) runs the search on `phi_mapping.DirectOnlyPrimitiveModel` and re-realizes the returned
  primitives through canonical φ (`realize_full_phi`); the reduced flow is stored as `reduced_*`.
* `common/phi_mapping.py` - traces the data flow of `CICIDS2017PrimitiveModel.generate` (φ) and
  classifies each write as direct or derived (P2).
* `common/analysis.py` - comparison with the reference, McNemar/Newcombe/Holm, report tables.
* `common/cli.py` - shared command line of every `run.py`.

## Running

From the repo root in the `thesis` env (`PYTHONPATH` is set by the scripts):

```
python ablations/reference/run.py --device cuda          # first: the shared reference arm
python ablations/A1_hybrid_components/run.py --device cuda
...                                                       # A3, A4, B1, B5 likewise
python ablations/thesis_ablations/A2_shape_allocation/run.py --device cuda
python ablations/thesis_ablations/D6_capability_inference/run.py --device cuda
python ablations/D5_overhead/run.py                      # analysis of FINAL artifacts only
python ablations/seeded_rows/run.py --device cuda  # new random row cohort per seed
python ablations/seeded_rows/analyze.py --device cuda
python ablations/thesis_ablations/P1_realization_aware_search/run.py --device cuda  # targeted
python ablations/thesis_ablations/P2_coupled_phi/run.py --device cuda  # targeted; own full_phi reference arm
```

Options: `--datasets`, `--budgets`, `--seeds`, `--victims`, `--classes`, `--conditions`,
`--limit-rows` (smoke tests; use a separate `--results-dir`), `--skip-run` (re-analyze only),
`--skip-analysis`. `reference/run.py`'s analysis writes `results/reproduction_check.md`: every
p75 reference cell vs `FINAL_OUTPUTS/runs/<dataset>/primattack_hybrid_objective_untargeted`.
