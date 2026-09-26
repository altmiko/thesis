# Final experiment summary

Bachelor's thesis: constrained adversarial attacks against NIDS classifiers. This is the definitive final suite defined in `master_experiments.md`, run under the locked protocol `00_PROTOCOL.md`. Every number below is regenerated from `runs/` by `scripts/analyze_final_suite.py`. All results are feature-space proxies on CICFlowMeter aggregates. No PCAP is edited or replayed, and no packet-level realizability or complete malicious functionality is claimed.

## Reports

| Experiment | Report |
|---|---|
| A — primary baseline comparison (untargeted) | `A_primary_baseline_comparison/primary_baseline_comparison.md` |
| B — PrimAttack optimizer selection | `B_optimizer_selection/primattack_optimizer_selection.md` |
| C — budget sensitivity | `C_budget_sensitivity/primattack_budget_sensitivity.md` |
| D — objective sensitivity | `D_objective_sensitivity/objective_sensitivity.md` |
| E — paired validity gap | `E_paired_validity_gap/paired_validity_gap_analysis.md` |
| F — validator evaluation | `F_validator_evaluation/validator_evaluation.md` |

## Scope

- Datasets: CICIDS2017-DistriNet, CSE-CIC-IDS-2018-DistriNet.
- Victims: MLP, CNN and FT-Transformer category classifiers, one frozen checkpoint per architecture and dataset (2018: training-seed-42 replicates).
- Source classes: DoS, DDoS, Recon, BruteForce. There are 800 canonical clean-correct test flows per (dataset, victim, class), identical for every method.
- Seeds 42, 2024 and 2026 are attack seeds used for every run.
- Metrics: Raw ASR, Valid ASR (success ∧ validator_v2 `hybrid_valid`, same denominator) and Validity Gap = Raw − Valid (pp).

## Statistical methodology

- mean ± SD across seeds 42, 2024 and 2026 for run-to-run variability;
- paired sample-level binary inference (one outcome per source flow: reference seed 42; seeds are never treated as independent observations and seed means are never the statistical sample);
- Cochran's Q for 3+ paired conditions (Exp A: 5 attacks; Exp B: 3 optimizers; Exp C: 3 budgets);
- planned McNemar comparisons for two-condition contrasts (A: PrimAttack vs each baseline; B: 3 optimizer pairs; C: adjacent budgets; D: targeted vs untargeted; E: raw vs valid);
- Holm correction only within logical families of multiple planned McNemar comparisons (A: 4, B: 3, C: 2 per dataset × victim);
- alpha = 0.05;
- Valid Success is the primary inferential outcome;
- Raw ASR is descriptive except in the dedicated validity-gap analysis (E);
- F is descriptive (no test).

Inferential tests actually computed: A 26, B 12, C 36, D 6, E 48 (see each `statistical_tests.csv`).

## Headline: Experiment A (Raw ASR → Valid ASR, untargeted, mean ± SD)

| Dataset | Victim | PrimAttack (Hybrid Search, p75) | PGD | C&W | CAPGD-PrimSupport | C-PGD-PrimSupport |
|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 11.06% ± 0.00% → 11.06% ± 0.00% | 100.00% ± 0.00% → 0.00% ± 0.00% | 99.94% ± 0.00% → 0.00% ± 0.00% | 94.41% ± 0.71% → 2.18% ± 0.08% | 50.80% ± 2.04% → 0.00% ± 0.00% |
| CICIDS2017 | cnn | 36.67% ± 0.04% → 36.67% ± 0.04% | 96.12% ± 0.09% → 0.00% ± 0.00% | 95.53% ± 0.00% → 0.00% ± 0.00% | 96.53% ± 1.07% → 5.21% ± 0.07% | 60.42% ± 2.98% → 0.00% ± 0.00% |
| CICIDS2017 | ft_transformer | 0.50% ± 0.00% → 0.50% ± 0.00% | 97.36% ± 0.28% → 0.00% ± 0.00% | 77.16% ± 0.00% → 0.00% ± 0.00% | 52.21% ± 4.33% → 0.18% ± 0.02% | 21.61% ± 0.31% → 0.00% ± 0.00% |
| CICIDS2018 | mlp-s42 | 12.00% ± 0.03% → 0.19% ± 0.03% | 94.34% ± 0.25% → 0.00% ± 0.00% | 87.63% ± 0.00% → 0.00% ± 0.00% | 91.57% ± 0.84% → 0.14% ± 0.02% | 28.25% ± 0.51% → 0.00% ± 0.00% |
| CICIDS2018 | cnn-s42 | 15.31% ± 0.00% → 0.03% ± 0.00% | 99.70% ± 0.02% → 0.00% ± 0.00% | 99.16% ± 0.00% → 0.00% ± 0.00% | 76.01% ± 1.68% → 0.29% ± 0.07% | 50.54% ± 3.30% → 0.00% ± 0.00% |
| CICIDS2018 | ft_transformer-s42 | 0.33% ± 0.02% → 0.00% ± 0.00% | 91.70% ± 0.18% → 0.00% ± 0.00% | 53.59% ± 0.00% → 0.00% ± 0.00% | 9.74% ± 0.70% → 0.00% ± 0.00% | 1.65% ± 0.42% → 0.00% ± 0.00% |

Selected PrimAttack optimizer (Exp B, pre-registered aggregate-Valid-Targeted-ASR rule): **Hybrid Search**. Ranking: Hybrid Search > Prim-PGD > Prim-C&W.

## Thesis contribution mapping

| Contribution | Evidence |
|---|---|
| 1. PrimAttack: constrained adversarial attack framework over attacker-controllable packet-size and timing primitives | Exp A (`primary_baseline_comparison.md`), Exp B (`primattack_optimizer_selection.md`) |
| 2. Evaluation methodology separating attack objective, domain validity and constraint-valid success under controlled perturbation conditions | Exp A, Exp C (`primattack_budget_sensitivity.md`), Exp D (`objective_sensitivity.md`) |
| 3. Paired validity-gap analysis (raw vs constraint-valid success on identical source samples) | `paired_validity_gap_analysis.md`; paired Raw/Valid/Gap columns in every report |
| 4. Independent domain-validation framework (general flow-consistency + automatically derived dataset-specific constraints) | `validator_evaluation.md` |
| 5. Controlled PrimAttack perturbation-budget analysis | `primattack_budget_sensitivity.md` |

## Provenance and integrity audit

- Per-sample artifacts read: 936 files, 748,800 attacked flow-instances.
- validator_v2 recomputed on the stored final adversarial flow for 748,800 rows: 0 mismatches (analysis aborts on any).
- Raw/valid success recomputed from stored predictions for 748,800 rows: 0 mismatches (analysis aborts on any).
- Victim re-prediction of the stored final flows: 748,800 rows, 0 mismatches.
- Pairing asserted per experiment: identical canonical sample IDs (order, no duplicates), clean-input SHA-256, labels, clean predictions, victim checkpoint SHA-256, seed set {42, 2024, 2026} and equal denominators.
- Run configurations with every hyperparameter: `runs/final_suite_config.json`, `runs/<dataset>/<stage>/config.json`; run logs: `runs/<dataset>/logs/`.

## Interpretation

**Main findings (per contribution).**

1. **PrimAttack (Contribution 1).** At the p75 train-calibrated budget, PrimAttack is the only
   attack that produces validator-valid evasions at scale on CICIDS2017. Valid ASR is 11.06%
   (MLP), 36.67% (CNN) and 0.50% (FT-Transformer), untargeted, and it beats every baseline
   significantly after Holm correction. PGD, C&W, CAPGD-PrimSupport and C-PGD-PrimSupport reach
   up to 100% Raw ASR but at most 5.21% Valid ASR. On CICIDS2018 at p75 every method is near zero
   valid (≤ 0.29%), with no significant PrimAttack-vs-baseline difference. The three optimizers
   are practically tied on aggregate Valid Targeted ASR: Hybrid Search 7.990% vs Prim-PGD 7.977%,
   with Prim-C&W at 4.411%. Hybrid Search is selected by the locked criterion and also uses the
   fewest victim evaluations.
2. **Evaluation methodology (Contribution 2).** Changing objective, validity requirement and
   budget moves results in different directions. The objective barely changes Valid ASR
   (Δ ≤ 0.53 pp; one significant victim) but can change Raw ASR ten-fold (CICIDS2018 MLP: 1.28%
   targeted vs 12.00% untargeted raw). The budget is a first-order control of Valid ASR:
   CICIDS2017 CNN goes from 13.10% to 36.15% to 70.48%, and CICIDS2018 MLP/CNN rise from ≈ 0% to
   22–25% only when the budget is unbounded.
3. **Paired validity gap (Contribution 3).** On identical adversarial examples, feature-space
   baselines lose 53.6–100 pp (PGD/C&W) and 1.7–92.2 pp (matched-support CAPGD/C-PGD) of their
   Raw ASR to the validator. PrimAttack loses 0 pp on CICIDS2017 and 0.2–15.3 pp on CICIDS2018.
   The loss is systematic wherever it occurs (McNemar p ≤ 0.032; every non-zero case one-sided).
   On CICIDS2017, Raw ASR ranks the attacks almost in reverse order of their valid success.
4. **Independent validator (Contribution 4).** On 874,179 genuine held-out flows it accepts
   99.93–100% per split. Rejections come from the train-constant URG-flag schema rules
   (CICIDS2017) or a single mined rule, `MINED_0001` (CICIDS2018). That same mined rule causes
   PrimAttack's entire CICIDS2018 validity gap. Its 0.06% false-rejection rate on genuine
   CICIDS2018 traffic is the relevant caveat.
5. **Budget analysis (Contribution 5).** Valid Targeted ASR never decreases with the budget.
   In 23 of 24 adjacent comparisons every discordant flow favors the larger budget; the single
   exception is one flow. The CICIDS2018 p50/p75 calibrations are too tight for valid
   timing-only evasion, and padding violates `MINED_0001` there.

**Victim dependence.** Valid evasion depends strongly on the victim. The FT-Transformer stays at
≤ 0.97% Valid ASR under every PrimAttack configuration and at 0.00–0.18% under every baseline. The
CICIDS2017 CNN is the most exposed victim. Results are reported per victim and dataset and are
never pooled.

**Claim boundary.** All results are feature-space proxies on CICFlowMeter aggregates of a
chronological within-label test split. No PCAP is edited or replayed. Neither packet-level
realizability nor preserved malicious functionality is claimed, for PrimAttack or for the
matched-support baselines. The seeds are attack seeds on one frozen victim per architecture
(CICIDS2018: the training-seed-42 replicate). They quantify attack-run variability, which is
tiny (SD ≤ 0.20 pp for any PrimAttack Valid ASR, ≤ 4.33 pp for any baseline Raw ASR). They do
not quantify victim-training variability. PrimAttack's search queries validator_v2, while the
baselines do not. This is part of its threat model and part of its valid-success advantage.

**Protocol amendment disclosed.** During the first Exp B run, a methodological bug surfaced in
the shared PrimAttack optimizer core. A pinned (zero-headroom) padding control on CICIDS2018
flows without backward packets produced a NaN surrogate gradient. That crashed Prim-C&W and would
silently freeze Hybrid/Prim-PGD rows (sign(NaN) = 0). The fix, which cuts the autograd path of
pinned controls and fails loudly on any non-finite gradient, was applied before any optimizer
selection. Stage B was then re-run from scratch on both datasets. See `00_PROTOCOL.md` §7.
