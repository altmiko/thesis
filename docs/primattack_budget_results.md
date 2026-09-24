# Full PrimAttack budget and primitive-ablation results

## Scope and claim boundary

This report contains the predeclared full PrimAttack budget-sensitivity experiment: four retained
attack classes, the active MLP and CNN victims, 512 fixed source rows per class, 40 optimization
steps, and seed 42. It evaluates three train-calibrated budgets under timing-only, padding-only,
and joint primitive modes. No victim was retrained.

SP-ASR is a **flow-level semantic-preservation proxy ASR**. It does not establish complete
malicious functionality or packet-trace behavior. Recon/PortScan and BruteForce retain critical
properties that CICIDS2017 aggregate rows cannot test; these rows are conservatively
`NOT_FULLY_TESTABLE` rather than counted as semantic PASS.

## Reproduction

```text
PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn \
  --test-limit 512 --steps 40 --seeds 42 \
  --output-dir outputs/primattack_budget_sensitivity_full

PYTHONPATH=".;src" python scripts/analyze_primattack_experiments.py \
  --input-dir outputs/primattack_budget_sensitivity_full

PYTHONPATH=".;src" python scripts/build_primattack_budget_report.py \
  --input-dir outputs/primattack_budget_sensitivity_full \
  --output docs/primattack_budget_results.md
```

Source pairing: **true** across
8 class/victim/seed cells. The same source IDs are used in every
budget/mode condition.

## Frozen class budgets

Each entry is `padding bytes per forward packet / maximum relative duration increase`.

| Class | Restricted | Intermediate | Maximum-evaluated |
|---|---|---|---|
| DoS | 41 B / 0.0427256 | 47 B / 0.567122 | 54 B / 1.26823 |
| DDoS | 2 B / 0.221998 | 2 B / 0.435296 | 3 B / 0.687425 |
| Recon | 2 B / 0.0851064 | 2 B / 0.212766 | 10 B / 0.531915 |
| BruteForce | 11 B / 0.0617419 | 12 B / 0.119895 | 91 B / 0.233668 |

Budgets and semantic thresholds were fitted on `X_train_pristine.npy` and
`y_train_cat.npy` only. Attack success did not participate in calibration.

## Main findings

- Timing-only produced **0/4064** targeted successes at
  maximum-evaluated budget (0.00%).
- Padding-only produced **10/4064** raw, domain-valid, and
  primitive-feasible successes (0.25%); **0**
  survived the semantic proxy (0.00%).
- Joint PrimAttack produced **10/4064** raw, domain-valid, and
  primitive-feasible successes (0.25%); **0** survived
  all gates (0.00%).
- At maximum-evaluated joint budget, BruteForce contributed
  **10** raw successes but zero SP successes because its critical
  application semantics are not testable from aggregate flow rows. DoS contributed
  **0** raw successes, of which **0** passed every
  flow-level proxy gate.
- Raw, valid, and primitive-feasible counts are equal in every pooled condition: validator_v2
  and hard primitive feasibility rejected none of the classifier successes.
- Padding is the effective evasion primitive in this experiment. Joint optimization adds only
  0 raw and
  0 SP successes over padding-only at the maximum budget.

## Statistical interpretation

- At maximum-evaluated budget, primitive mode affected targeted success
  (Cochran's Q $p=4.53999e-05$).
- Joint and padding-only each exceeded timing-only after Holm correction
  (`joint vs timing` Holm $p=0.00585938$;
  `padding vs timing` Holm $p=0.00585938$).
- The 0-success raw difference between joint and
  padding-only had Holm $p=1$. Their
  0-success SP-ASR difference had Holm
  $p=1$.
- For joint PrimAttack, maximum-evaluated exceeded intermediate budget after correction
  (Holm $p=0.00585938$).


## Pooled primary results

| Mode | Budget | N | Raw n | Raw ASR | Valid ASR | Feasible ASR | SP n | SP-ASR | Semantic PASS | Semantic FAIL | Not fully testable | Testability | Median Δduration | Median Δbytes | Median rate retention | Median changed features |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| timing-only | restricted | 4064 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0.000591466 | 0 | 0.999409 | 11 |
| timing-only | intermediate | 4064 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0.00339406 | 0 | 0.996617 | 11 |
| timing-only | maximum-evaluated | 4064 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0.00623234 | 0 | 0.993806 | 11 |
| padding-only | restricted | 4064 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0 | 0.000838082 | 1 | 10 |
| padding-only | intermediate | 4064 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0 | 0.00120534 | 1 | 10 |
| padding-only | maximum-evaluated | 4064 | 10 | 0.25% | 0.25% | 0.25% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0 | 0.00206629 | 1 | 10 |
| joint | restricted | 4064 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0.000616571 | 0.000838082 | 0.999384 | 20 |
| joint | intermediate | 4064 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0.00363767 | 0.00120534 | 0.996376 | 20 |
| joint | maximum-evaluated | 4064 | 10 | 0.25% | 0.25% | 0.25% | 0 | 0.00% | 48.18% | 2.17% | 49.66% | 50.34% | 0.007007 | 0.00206629 | 0.993042 | 20 |

All ASRs use eligible clean-correct malicious sources as the denominator. `Valid ASR` adds
validator_v2 domain validity. `Feasible ASR` additionally requires hard primitive compliance and
internal transform consistency. SP-ASR additionally requires semantic status `PASS`.

## Maximum-evaluated joint results by source class

| Class | N | Raw n | Raw ASR | Valid ASR | Feasible ASR | SP n | SP-ASR | Semantic PASS | Semantic FAIL | Not fully testable | Testability |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BruteForce | 998 | 10 | 1.00% | 1.00% | 1.00% | 0 | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% |
| DDoS | 1022 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 94.72% | 5.28% | 0.00% | 100.00% |
| DoS | 1024 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 96.68% | 3.32% | 0.00% | 100.00% |
| Recon | 1020 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 0.00% | 100.00% | 0.00% |

## Maximum-evaluated joint results by victim

| Victim | N | Raw n | Raw ASR | Valid ASR | Feasible ASR | SP n | SP-ASR | Median Δduration | Median Δbytes |
|---|---|---|---|---|---|---|---|---|---|
| cnn | 2030 | 4 | 0.20% | 0.20% | 0.20% | 0 | 0.00% | 0.009997 | 0.00206629 |
| mlp | 2034 | 6 | 0.29% | 0.29% | 0.29% | 0 | 0.00% | 0.00352975 | 0.00180801 |

## Maximum-evaluated primitive ablation details

| Mode | Class | Victim | N | Raw n | Raw ASR | Valid ASR | Feasible ASR | SP n | SP-ASR | Semantic PASS | Not fully testable | Median Δduration | Median Δbytes | Median rate retention |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| joint | BruteForce | cnn | 498 | 4 | 0.80% | 0.80% | 0.80% | 0 | 0.00% | 0.00% | 100.00% | 0.209205 | 0.854369 | 0.82699 |
| joint | BruteForce | mlp | 500 | 6 | 1.20% | 1.20% | 1.20% | 0 | 0.00% | 0.00% | 100.00% | 0.00199563 | 0.338733 | 0.998008 |
| joint | DDoS | cnn | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 94.72% | 0.00% | 0.00721106 | 0.00206629 | 0.992841 |
| joint | DDoS | mlp | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 94.72% | 0.00% | 0.451936 | 0.00206629 | 0.688736 |
| joint | DoS | cnn | 512 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 96.68% | 0.00% | 0.015079 | 0.00115201 | 0.985145 |
| joint | DoS | mlp | 512 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 96.68% | 0.00% | 1.09949 | 0 | 0.476306 |
| joint | Recon | cnn | 509 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0 | 1 |
| joint | Recon | mlp | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0 | 1 |
| padding-only | BruteForce | cnn | 498 | 4 | 0.80% | 0.80% | 0.80% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0.884244 | 1 |
| padding-only | BruteForce | mlp | 500 | 6 | 1.20% | 1.20% | 1.20% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0.333894 | 1 |
| padding-only | DDoS | cnn | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 94.72% | 0.00% | 0 | 0.00206629 | 1 |
| padding-only | DDoS | mlp | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 94.72% | 0.00% | 0 | 0.00206629 | 1 |
| padding-only | DoS | cnn | 512 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 96.68% | 0.00% | 0 | 0.000838542 | 1 |
| padding-only | DoS | mlp | 512 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 96.68% | 0.00% | 0 | 0 | 1 |
| padding-only | Recon | cnn | 509 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0 | 1 |
| padding-only | Recon | mlp | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0 | 1 |
| timing-only | BruteForce | cnn | 498 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0.204387 | 0 | 0.830298 |
| timing-only | BruteForce | mlp | 500 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0.00197098 | 0 | 0.998033 |
| timing-only | DDoS | cnn | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 94.72% | 0.00% | 0.00710375 | 0 | 0.992946 |
| timing-only | DDoS | mlp | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 94.72% | 0.00% | 0.451588 | 0 | 0.688901 |
| timing-only | DoS | cnn | 512 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 96.68% | 0.00% | 0.0174107 | 0 | 0.982887 |
| timing-only | DoS | mlp | 512 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 96.68% | 0.00% | 1.10295 | 0 | 0.475523 |
| timing-only | Recon | cnn | 509 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0 | 1 |
| timing-only | Recon | mlp | 511 | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00% | 100.00% | 0 | 0 | 1 |

## Paired statistical tests

Pairing key: `sample_id × attack_class × victim × seed`. Binary omnibus tests use Cochran's Q,
with repository McNemar tests for pairwise follow-ups. Continuous omnibus tests use Friedman,
with Wilcoxon signed-rank follow-ups. Pairwise p-values are Holm corrected within each family.

### Omnibus tests

| Family | Test | Fixed axis | Fixed value | Outcome | N | Statistic | p |
|---|---|---|---|---|---|---|---|
| binary | Cochran's Q | budget_name | intermediate | targeted_success | 4064 | 0 | 1 |
| binary | Cochran's Q | budget_name | intermediate | sp_success | 4064 | 0 | 1 |
| binary | Cochran's Q | budget_name | maximum-evaluated | targeted_success | 4064 | 20 | 4.53999e-05 |
| binary | Cochran's Q | budget_name | maximum-evaluated | sp_success | 4064 | 0 | 1 |
| binary | Cochran's Q | budget_name | restricted | targeted_success | 4064 | 0 | 1 |
| binary | Cochran's Q | budget_name | restricted | sp_success | 4064 | 0 | 1 |
| binary | Cochran's Q | primitive_mode | joint | targeted_success | 4064 | 20 | 4.53999e-05 |
| binary | Cochran's Q | primitive_mode | joint | sp_success | 4064 | 0 | 1 |
| binary | Cochran's Q | primitive_mode | padding-only | targeted_success | 4064 | 20 | 4.53999e-05 |
| binary | Cochran's Q | primitive_mode | padding-only | sp_success | 4064 | 0 | 1 |
| binary | Cochran's Q | primitive_mode | timing-only | targeted_success | 4064 | 0 | 1 |
| binary | Cochran's Q | primitive_mode | timing-only | sp_success | 4064 | 0 | 1 |
| continuous | Friedman | budget_name | intermediate | relative_duration_change | 4064 | 4588.08 | 0 |
| continuous | Friedman | budget_name | intermediate | relative_byte_change | 4064 | 4372.42 | 0 |
| continuous | Friedman | budget_name | intermediate | rate_retention | 4064 | 4589.86 | 0 |
| continuous | Friedman | budget_name | maximum-evaluated | relative_duration_change | 4064 | 4493.43 | 0 |
| continuous | Friedman | budget_name | maximum-evaluated | relative_byte_change | 4064 | 4929.74 | 0 |
| continuous | Friedman | budget_name | maximum-evaluated | rate_retention | 4064 | 4496.19 | 0 |
| continuous | Friedman | budget_name | restricted | relative_duration_change | 4064 | 4682.31 | 0 |
| continuous | Friedman | budget_name | restricted | relative_byte_change | 4064 | 4268.03 | 0 |
| continuous | Friedman | budget_name | restricted | rate_retention | 4064 | 4685.41 | 0 |
| continuous | Friedman | primitive_mode | joint | relative_duration_change | 4064 | 5155.73 | 0 |
| continuous | Friedman | primitive_mode | joint | relative_byte_change | 4064 | 4426.47 | 0 |
| continuous | Friedman | primitive_mode | joint | rate_retention | 4064 | 5163.88 | 0 |
| continuous | Friedman | primitive_mode | padding-only | relative_duration_change | 4064 | 0 | 1 |
| continuous | Friedman | primitive_mode | padding-only | relative_byte_change | 4064 | 4406.28 | 0 |
| continuous | Friedman | primitive_mode | padding-only | rate_retention | 4064 | 0 | 1 |
| continuous | Friedman | primitive_mode | timing-only | relative_duration_change | 4064 | 5306.17 | 0 |
| continuous | Friedman | primitive_mode | timing-only | relative_byte_change | 4064 | 0 | 1 |
| continuous | Friedman | primitive_mode | timing-only | rate_retention | 4064 | 5306.17 | 0 |

### Holm-significant pairwise tests

| Family | Fixed axis | Fixed value | Outcome | Comparison | p | Holm p |
|---|---|---|---|---|---|---|
| binary | budget_name | maximum-evaluated | targeted_success | joint vs timing-only | 0.00195312 | 0.00585938 |
| binary | budget_name | maximum-evaluated | targeted_success | padding-only vs timing-only | 0.00195312 | 0.00585938 |
| binary | primitive_mode | joint | targeted_success | intermediate vs maximum-evaluated | 0.00195312 | 0.00585938 |
| binary | primitive_mode | joint | targeted_success | maximum-evaluated vs restricted | 0.00195312 | 0.00585938 |
| binary | primitive_mode | padding-only | targeted_success | intermediate vs maximum-evaluated | 0.00195312 | 0.00585938 |
| binary | primitive_mode | padding-only | targeted_success | maximum-evaluated vs restricted | 0.00195312 | 0.00585938 |
| continuous | budget_name | intermediate | relative_duration_change | joint vs padding-only | 0 | 0 |
| continuous | budget_name | intermediate | relative_duration_change | joint vs timing-only | 2.76528e-06 | 2.76528e-06 |
| continuous | budget_name | intermediate | relative_duration_change | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | intermediate | relative_byte_change | joint vs timing-only | 0 | 0 |
| continuous | budget_name | intermediate | relative_byte_change | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | intermediate | rate_retention | joint vs padding-only | 0 | 0 |
| continuous | budget_name | intermediate | rate_retention | joint vs timing-only | 2.44504e-06 | 2.44504e-06 |
| continuous | budget_name | intermediate | rate_retention | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | maximum-evaluated | relative_duration_change | joint vs padding-only | 0 | 0 |
| continuous | budget_name | maximum-evaluated | relative_duration_change | joint vs timing-only | 7.0896e-07 | 7.0896e-07 |
| continuous | budget_name | maximum-evaluated | relative_duration_change | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | maximum-evaluated | relative_byte_change | joint vs padding-only | 8.49965e-05 | 8.49965e-05 |
| continuous | budget_name | maximum-evaluated | relative_byte_change | joint vs timing-only | 0 | 0 |
| continuous | budget_name | maximum-evaluated | relative_byte_change | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | maximum-evaluated | rate_retention | joint vs padding-only | 0 | 0 |
| continuous | budget_name | maximum-evaluated | rate_retention | joint vs timing-only | 7.42025e-07 | 7.42025e-07 |
| continuous | budget_name | maximum-evaluated | rate_retention | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | restricted | relative_duration_change | joint vs padding-only | 0 | 0 |
| continuous | budget_name | restricted | relative_duration_change | joint vs timing-only | 9.77568e-07 | 9.77568e-07 |
| continuous | budget_name | restricted | relative_duration_change | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | restricted | relative_byte_change | joint vs timing-only | 0 | 0 |
| continuous | budget_name | restricted | relative_byte_change | padding-only vs timing-only | 0 | 0 |
| continuous | budget_name | restricted | rate_retention | joint vs padding-only | 0 | 0 |
| continuous | budget_name | restricted | rate_retention | joint vs timing-only | 6.35846e-07 | 6.35846e-07 |
| continuous | budget_name | restricted | rate_retention | padding-only vs timing-only | 0 | 0 |
| continuous | primitive_mode | joint | relative_duration_change | intermediate vs maximum-evaluated | 0 | 0 |
| continuous | primitive_mode | joint | relative_duration_change | intermediate vs restricted | 0 | 0 |
| continuous | primitive_mode | joint | relative_duration_change | maximum-evaluated vs restricted | 0 | 0 |
| continuous | primitive_mode | joint | relative_byte_change | intermediate vs maximum-evaluated | 0 | 0 |
| continuous | primitive_mode | joint | relative_byte_change | intermediate vs restricted | 1.38169e-214 | 1.38169e-214 |
| continuous | primitive_mode | joint | relative_byte_change | maximum-evaluated vs restricted | 0 | 0 |
| continuous | primitive_mode | joint | rate_retention | intermediate vs maximum-evaluated | 0 | 0 |
| continuous | primitive_mode | joint | rate_retention | intermediate vs restricted | 0 | 0 |
| continuous | primitive_mode | joint | rate_retention | maximum-evaluated vs restricted | 0 | 0 |
| continuous | primitive_mode | padding-only | relative_byte_change | intermediate vs maximum-evaluated | 0 | 0 |
| continuous | primitive_mode | padding-only | relative_byte_change | intermediate vs restricted | 1.43481e-215 | 1.43481e-215 |
| continuous | primitive_mode | padding-only | relative_byte_change | maximum-evaluated vs restricted | 0 | 0 |
| continuous | primitive_mode | timing-only | relative_duration_change | intermediate vs maximum-evaluated | 0 | 0 |
| continuous | primitive_mode | timing-only | relative_duration_change | intermediate vs restricted | 0 | 0 |
| continuous | primitive_mode | timing-only | relative_duration_change | maximum-evaluated vs restricted | 0 | 0 |
| continuous | primitive_mode | timing-only | rate_retention | intermediate vs maximum-evaluated | 0 | 0 |
| continuous | primitive_mode | timing-only | rate_retention | intermediate vs restricted | 0 | 0 |
| continuous | primitive_mode | timing-only | rate_retention | maximum-evaluated vs restricted | 0 | 0 |

## Integrity and reproducibility checks

- Artifacts checked: **72** NPZ files containing **36864** saved rows.
- Every projected sample passed primitive feasibility: **true**.
- No NaN or Inf occurred in final vectors, primitive controls, costs, or retention quantities.
- Timing-only artifacts had $p=0$ and padding-only artifacts had $\alpha=1$ for every row.
- Exact source-ID pairing held across all 16 class/victim/seed cells and all nine conditions.
- Final repository test suite: **217 passed, 1 skipped**; one unrelated existing PyTorch
  convolution warning.

## Figures

1. [Raw Target-Benign ASR vs budget](../outputs/primattack_budget_sensitivity_full/01_raw_asr_vs_budget.png)
2. [Valid Target-Benign ASR vs budget](../outputs/primattack_budget_sensitivity_full/02_valid_asr_vs_budget.png)
3. [SP-ASR vs budget](../outputs/primattack_budget_sensitivity_full/03_sp_asr_vs_budget.png)
4. [Semantic PASS rate vs budget](../outputs/primattack_budget_sensitivity_full/04_semantic_pass_vs_budget.png)
5. [Rate retention vs timing budget](../outputs/primattack_budget_sensitivity_full/05_rate_retention_vs_timing_budget.png)
6. [Primitive cost vs ASR](../outputs/primattack_budget_sensitivity_full/06_primitive_cost_vs_asr.png)
7. [Timing-only vs padding-only vs joint](../outputs/primattack_budget_sensitivity_full/07_timing_padding_combined.png)

## Interpretation

- Compare raw, valid, feasible, and SP-ASR in order; later gates never replace earlier metrics.
- A zero SP-ASR can coexist with raw success when success occurs in a class whose critical
  semantics are unavailable from aggregate flow data.
- Timing-only, padding-only, and joint conditions use identical sources and hard budgets, so
  differences isolate primitive contribution rather than sample selection.
- The maximum-evaluated level is the training-derived P75 experimental envelope, not a universal
  physical maximum.

## Limitations

The transformation is an offline flow-feature model. Complete scan structure, authentication
attempt semantics, payload behavior, merged packet ordering, and target response are unavailable
from CICIDS2017 aggregate rows. Establishing those properties would require packet realization,
feature re-extraction, and isolated replay, which are outside this thesis.
