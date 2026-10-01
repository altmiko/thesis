# PrimAttack ablations

Ablation experiments for PrimAttack. Each design is taken from published work (sources are
cited in each experiment's `run.py` docstring and `README.md`). Every experiment has its own
folder with `run.py`, `README.md` (design and results) and `results/` (`report.md`,
`summary.csv`, `tests.csv`, extra tables, `<dataset>/cells.json`, `<dataset>/config.json`).
The per-row npz artifacts (`results/<dataset>/artifacts/`) are git-ignored.

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
```

Options: `--datasets`, `--budgets`, `--seeds`, `--victims`, `--classes`, `--conditions`,
`--limit-rows` (smoke tests; use a separate `--results-dir`), `--skip-run` (re-analyze only),
`--skip-analysis`. `reference/run.py`'s analysis writes `results/reproduction_check.md`: every
p75 reference cell vs `FINAL_OUTPUTS/runs/<dataset>/primattack_hybrid_objective_untargeted`.
