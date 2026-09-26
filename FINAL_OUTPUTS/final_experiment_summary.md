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

Inferential tests actually computed: A 26, B 9, C 36, D 6, E 48 (see each `statistical_tests.csv`).

## Headline: Experiment A (Raw ASR → Valid ASR, untargeted, mean ± SD)

| Dataset | Victim | PrimAttack (Prim-PGD, p75) | PGD | C&W | CAPGD-PrimSupport | C-PGD-PrimSupport | CAPGD (native) † |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 4.09% ± 0.00% → 4.09% ± 0.00% | 100.00% ± 0.00% → 0.00% ± 0.00% | 99.94% ± 0.00% → 0.00% ± 0.00% | 94.41% ± 0.71% → 2.01% ± 0.10% | 50.80% ± 2.04% → 0.00% ± 0.00% | 94.65% ± 0.88% → 9.53% ± 0.85% |
| CICIDS2017 | cnn | 13.47% ± 0.00% → 13.47% ± 0.00% | 96.12% ± 0.09% → 0.00% ± 0.00% | 95.53% ± 0.00% → 0.00% ± 0.00% | 96.53% ± 1.07% → 5.15% ± 0.07% | 60.42% ± 2.98% → 0.00% ± 0.00% | 96.25% ± 1.19% → 19.24% ± 0.13% |
| CICIDS2017 | ft_transformer | 0.12% ± 0.00% → 0.12% ± 0.00% | 97.36% ± 0.28% → 0.00% ± 0.00% | 77.16% ± 0.00% → 0.00% ± 0.00% | 52.21% ± 4.33% → 0.18% ± 0.02% | 21.61% ± 0.31% → 0.00% ± 0.00% | 50.61% ± 3.39% → 5.71% ± 1.07% |
| CICIDS2018 | mlp-s42 | 2.53% ± 0.00% → 2.53% ± 0.00% | 94.34% ± 0.25% → 0.00% ± 0.00% | 87.63% ± 0.00% → 0.00% ± 0.00% | 91.57% ± 0.84% → 0.14% ± 0.02% | 28.25% ± 0.51% → 0.00% ± 0.00% | 80.42% ± 0.69% → 28.42% ± 1.13% |
| CICIDS2018 | cnn-s42 | 1.16% ± 0.00% → 1.16% ± 0.00% | 99.70% ± 0.02% → 0.00% ± 0.00% | 99.16% ± 0.00% → 0.00% ± 0.00% | 76.01% ± 1.68% → 0.29% ± 0.07% | 50.54% ± 3.30% → 0.00% ± 0.00% | 66.44% ± 1.28% → 16.30% ± 3.22% |
| CICIDS2018 | ft_transformer-s42 | 0.00% ± 0.00% → 0.00% ± 0.00% | 91.70% ± 0.18% → 0.00% ± 0.00% | 53.59% ± 0.00% → 0.00% ± 0.00% | 9.74% ± 0.70% → 0.00% ± 0.00% | 1.65% ± 0.42% → 0.00% ± 0.00% | 5.96% ± 0.53% → 0.45% ± 0.45% |

† CAPGD (native): descriptive row (amendment A3), not in the inferential family.

**Capability-aware PrimAttack (amendment A2).** Padding adds `p` bytes to *every* forward packet. A source flow with `Fwd Packet Length Min = 0` contains at least one zero-length forward packet (e.g. a pure ACK), and aggregate features do not say which one, so padding would put bytes into an empty packet (payload insertion, not length augmentation). PrimAttack therefore infers `pad_allowed = payload present ∧ Fwd Packet Length Min > 0` before optimization; such flows are attacked timing-only with the full per-flow budget. validator_v2 independently rejects any attack output that turns a source minimum of 0 into a positive value (source-conditioned PROTOCOL rule `PROTO_0080`, both datasets). The pre-fix run is kept only as the `PrimAttack-relaxed-padding` sensitivity result (`superseded_relaxed_padding/`).

Capability-fix analyses (eligibility, primitive use, primitive ablation, relaxed vs capability-aware PrimAttack, validator-rule impact, CAPGD-PrimSupport fairness): `A_primary_baseline_comparison/primary_baseline_comparison.md` and `A_primary_baseline_comparison/capability_fix/`; root report `primattack_empty_packet_fix_report.md`.

Selected PrimAttack optimizer (Exp B, pre-registered aggregate-Valid-Targeted-ASR rule): **Prim-PGD**. Ranking: Prim-PGD > Hybrid Search > Prim-C&W.

## Thesis contribution mapping

| Contribution | Evidence |
|---|---|
| 1. PrimAttack: constrained adversarial attack framework over attacker-controllable packet-size and timing primitives | Exp A (`primary_baseline_comparison.md`), Exp B (`primattack_optimizer_selection.md`) |
| 2. Evaluation methodology separating attack objective, domain validity and constraint-valid success under controlled perturbation conditions | Exp A, Exp C (`primattack_budget_sensitivity.md`), Exp D (`objective_sensitivity.md`) |
| 3. Paired validity-gap analysis (raw vs constraint-valid success on identical source samples) | `paired_validity_gap_analysis.md`; paired Raw/Valid/Gap columns in every report |
| 4. Independent domain-validation framework (general flow-consistency + automatically derived dataset-specific constraints) | `validator_evaluation.md` |
| 5. Controlled PrimAttack perturbation-budget analysis | `primattack_budget_sensitivity.md` |

## Provenance and integrity audit

- Per-sample artifacts read: 1152 files, 921,600 attacked flow-instances.
- validator_v2 recomputed on the stored final adversarial flow for 921,600 rows: 0 mismatches (analysis aborts on any).
- Raw/valid success recomputed from stored predictions for 921,600 rows: 0 mismatches (analysis aborts on any).
- Victim re-prediction of the stored final flows: 921,600 rows, 0 mismatches.
- Pairing asserted per experiment: identical canonical sample IDs (order, no duplicates), clean-input SHA-256, labels, clean predictions, victim checkpoint SHA-256, seed set {42, 2024, 2026} and equal denominators.
- Run configurations with every hyperparameter: `runs/final_suite_config.json`, `runs/<dataset>/<stage>/config.json`; run logs: `runs/<dataset>/logs/`.

## Interpretation

**Main findings (per contribution).**

1. **PrimAttack (Contribution 1).** After the empty-forward-packet capability correction,
   PrimAttack is almost entirely a timing attack: only 0.03% (CICIDS2017) and 0.38%
   (CICIDS2018) of its attacked flows are padding-eligible, and none of its valid successes uses
   padding. At the p75 train-calibrated budget its untargeted Valid ASR is 4.09% / 13.47% / 0.12%
   (CICIDS2017 MLP / CNN / FT-Transformer) and 2.53% / 1.16% / 0.00% (CICIDS2018). It exceeds
   CAPGD-PrimSupport significantly on the MLP/CNN of both datasets, is statistically
   indistinguishable on CICIDS2017 FT and ties at zero on CICIDS2018 FT. Hybrid and Prim-PGD tie
   exactly on the selection outcome (1,752 / 57,600 valid targeted successes); fewer mean victim
   evaluations selects Prim-PGD.
2. **Evaluation methodology (Contribution 2).** Threat-model restrictions reverse the raw-success
   ranking. PGD/C&W reach 53.59–100% Raw ASR but 0% Valid ASR. CAPGD/C-PGD restricted to
   PrimAttack's 23-feature support retain high raw success but little or no valid success.
   PrimAttack has much lower Raw ASR but zero gap between Raw and Valid ASR for every successful
   cell. Targeted and untargeted objectives coincide on CICIDS2017 except for 7 CNN flows, but
   differ by 1.16–1.75 pp on CICIDS2018 MLP/CNN because untargeted DDoS flows can move to DoS.
3. **Paired validity gap (Contribution 3).** On the same source samples, PGD/C&W lose 53.6–100 pp
   and matched-support CAPGD/C-PGD lose 1.7–92.4 pp from Raw to Valid ASR; capability-aware
   PrimAttack loses 0 pp. The new transition rule removes only 0.06–0.17 pp from
   CAPGD-PrimSupport on CICIDS2017 and nothing elsewhere; those direct feature-space outputs
   already fail other identities in almost every case.
4. **Independent validator (Contribution 4).** On 874,179 genuine held-out flows validator_v2
   accepts 99.93–100% per split. `PROTO_0080` is source-conditioned (an empty forward packet stays
   empty) and cannot reject genuine flows. Rejections remain the train-constant URG schema rules
   on CICIDS2017 or `MINED_0001` on CICIDS2018. The transition rule is independently applied to
   every attack, while PrimAttack also removes unsupported padding before search.
5. **Budget analysis (Contribution 5).** Valid Targeted ASR never decreases with the budget.
   The unbounded timing box raises p75 Valid ASR from 4.09% to 22.94% (CICIDS2017 MLP), 13.25% to
   59.69% (CICIDS2017 CNN), 0.78% to 24.76% (CICIDS2018 MLP) and 0% to 26.09%
   (CICIDS2018 CNN). The p75 result is therefore a conservative calibrated-budget result, not a
   bound on timing-based evasion.

**Effect of amendment A2.** The relaxed pre-fix PrimAttack had 11.06% / 36.67% / 0.50% Valid ASR
on CICIDS2017, dominated by padding empty packets. The capability-aware result is 63–75% lower,
but fresh timing optimization recovers 126–127 / 348–366 / 3 valid successes per seed beyond
simply filtering the old outputs. On CICIDS2018 the correction raises Valid ASR because it stops
spending evaluations on padding that `MINED_0001` rejects. This is the methodological point:
capability restrictions belong in the attack space, not only in validation.

**Matched-support interpretation.** CAPGD-PrimSupport directly optimizes the 23 downstream
features; PrimAttack reaches those potential coordinates only through source-applicable
primitives and coupled recomputation. Their comparison quantifies constrained feature-space
reachability versus primitive-domain reachability; it does not require PrimAttack to win. Native
CAPGD (descriptive, its different 16-feature configuration mask) has the highest Valid ASR on all
six victims (0.45–28.42%), showing that the chosen support and parameterization materially define
the threat model.

**Victim dependence.** FT-Transformer stays at ≤ 0.59% Valid ASR under any PrimAttack budget and
≤ 0.18% under the five inferential p75 attacks. CICIDS2017 CNN is the most exposed victim. Results
are reported per victim and dataset and are never pooled.

**Claim boundary.** PrimAttack is a restrictive, realizability-oriented **flow-level abstraction**:
feature changes must arise from modeled padding/timing primitives and deterministic
recomputation. No PCAP is edited, replayed or re-extracted; aggregate features do not identify
individual payload packets; complete malicious functionality and packet-level realizability are
not established. The conservative `Fwd Packet Length Min > 0` condition sacrifices possible
legitimate data-packet padding on mixed empty/data flows. Seeds are attack seeds on one frozen
victim per architecture, not victim-training seeds. PrimAttack queries validator_v2 during
search; baselines do not.

**Amendments.** A1 fixed non-finite gradients at pinned controls before optimizer selection. A2
added capability-aware padding and `PROTO_0080`, preserved the old run as
`superseded_relaxed_padding/`, and reran every stage from scratch. A3 adds native CAPGD only as a
descriptive row; the locked five-method inferential family is unchanged. See `00_PROTOCOL.md` §7
and `../primattack_empty_packet_fix_report.md`.
