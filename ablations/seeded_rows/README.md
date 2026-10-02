# Seed-dependent source-row selection for PrimAttack

This post-run sensitivity study varies the **test-flow cohort** as well as the attack seed. It does not replace the locked `FINAL_OUTPUTS/` protocol or its attack comparisons. The fixed victim checkpoints and train-fitted budgets are unchanged.

For each dataset (`cicids2017_distrinet`, `cicids2018_distrinet`), category victim (three per dataset), class (DoS, DDoS, Recon, BruteForce), and seed (42, 2024, 2026), `run.py` applies the FINAL uniform-selection rule: shuffle all class test rows with `numpy.random.default_rng(seed + class_id)`, retain the first 800 clean-correct rows for that victim, and sort them into test-row order. Seed 42 is checked field-for-field against `FINAL_OUTPUTS/runs/<dataset>/baselines_untargeted/selection.json`; its already-computed p75 and unbounded artifacts are copied, not hardlinked or re-executed. For seeds 2024 and 2026, only Prim-PGD (the selected FINAL optimizer) is executed, with the untargeted objective, joint capability-aware primitives, the full validator gate, the FINAL hyperparameters and 256 victim evaluations per flow. The two budgets are p75 (`maximum-evaluated`) and envelope-only `unbounded`; the latter is **not** an unlimited-query attack. Both budgets use identical rows within each seed. All generated selections, copied and new attack artifacts, logs, audits, CSVs and reports are saved under `ablations/seeded_rows/results/`.

Run from the repository root in the `thesis` Python environment:

```text
python ablations/seeded_rows/run.py --device cuda
python ablations/seeded_rows/analyze.py --device cuda
```

`run.py --prepare-only` generates and verifies selections without attacks. `--run-only` resumes from prepared selections, and the PrimAttack runner resumes completed new cells. Seed-42 reuse checks the stored FINAL configuration and source-cell count and copies its exact artifacts into this experiment's directory. The analyzer requires all 144 cells before writing any report; it verifies IDs, indices, row hashes, clean labels and predictions, checkpoint hashes, final victim predictions, and source-conditioned validator-v2 decisions. It records cross-seed row overlap in `results/audit.json`.

The primary descriptive output is per-seed Valid and Raw ASR for each victim and class and the three-seed mean ± sample SD. Reference-seed-42 p75 versus unbounded Valid ASR is compared by paired McNemar, with a Newcombe interval and Holm adjustment over the six dataset–victim contrasts; raw/valid and raw-budget tests are reported as unadjusted diagnostics. Reference-seed-42 Clopper–Pearson/Wilson and within-class bootstrap intervals describe source-row uncertainty conditional on the held-out split and fixed victim. The `results/report.md` also gives context from the earlier fixed-attack-seed full-pool p75 row-resampling study. The three cohorts overlap and have different members: **Cochran's Q over seeds, Fleiss' κ and paired cross-seed McNemar are not valid here**. The selection seed and search seed change together, so the three-run SD does not by itself separate their effects. No new baseline attack is run on the new rows, and existing between-method tests cannot be transplanted onto them. Victim training, new-campaign generalization, and packet-level realizability are not tested.

## Observed results

The analyzer audited all 144 cells (115,200 attempted flows), including re-prediction and source-conditioned validator-v2 rechecks for every artifact row. Valid ASR equals Raw ASR in every cell. Per-victim Valid-ASR means and sample SD over the three newly selected cohorts:

| Dataset | Victim | p75 mean ± SD | Unbounded mean ± SD |
|---|---|---:|---:|
| CICIDS2017 | mlp | 4.11% ± 0.07% | 23.01% ± 0.59% |
| CICIDS2017 | cnn | 12.94% ± 0.50% | 59.67% ± 0.42% |
| CICIDS2017 | ft_transformer | 0.15% ± 0.02% | 0.60% ± 0.02% |
| CICIDS2018 | mlp-s42 | 2.22% ± 0.33% | 44.44% ± 0.27% |
| CICIDS2018 | cnn-s42 | 1.11% ± 0.10% | 26.32% ± 0.04% |
| CICIDS2018 | ft_transformer-s42 | 0.04% ± 0.05% | 0.26% ± 0.13% |

Seed-42 paired McNemar comparisons of Valid ASR (p75 against unbounded, Holm over six victims) reject equal marginal success rates for five victims. CICIDS2018 ft_transformer-s42 does not (4 discordant flows; adjusted p = 0.125). This is a **budget contrast**, not a comparison with PGD/C&W/CAPGD baselines. The different-row cohort rates show observed variability in all 12 victim–budget conditions, but three overlapping selections are not independent replicates for a cross-seed significance test. The fixed-search-seed full-pool p75 resampling provides the separate row-selection check: the canonical CICIDS2018 MLP p75 result is at the 94.1st percentile of its 10,000 class-stratified random selections. The new MLP cohorts at 2.25% and 1.88% versus the canonical 2.53% corroborate that the canonical estimate is relatively high; claiming that every high ASR is unrelated to row selection would contradict these data.

Full seed-by-seed rates, uncertainty intervals, exact/asymptotic test variants, discordances and sample overlap: `results/report.md`, `results/summary.csv`, `results/tests.csv`, `results/audit.json`. Original and newly executed per-row NPZs are in `results/<dataset>/seed<seed>/primattack/artifacts/` (git-ignored; retained locally).
