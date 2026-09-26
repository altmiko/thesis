# Statistical testing guide

This directory contains the minimum paired inferential analysis for the canonical final-suite attack artifacts under `FINAL_OUTPUTS/runs/`.

## Files

| File | Purpose |
|---|---|
| `statistical_tests.csv` | Machine-readable test results, one row per omnibus test or planned paired comparison. |
| `statistical_summary.md` | Thesis-ready protocol, descriptive mean ± SD tables, inferential results, and interpretation. |
| `README.md` | Usage and interpretation guide. |

The analysis implementation is `scripts/analyze_minimum_final_statistics.py`.

## Reproduce the analysis

From the repository root in the `thesis` environment:

```powershell
$Env:PYTHONPATH = "src"
python scripts/analyze_minimum_final_statistics.py
```

This command reads stored NPZ artifacts only. It does not train victims, generate adversarial examples, or rerun attacks. It rewrites `statistical_tests.csv` and `statistical_summary.md` after all integrity checks pass.

## Analysis unit and seed handling

- Paired unit: one canonical clean-correct source flow.
- Classes pooled within each victim: DoS, DDoS, Recon, and BruteForce.
- N per dataset–victim test: 3,200 flows (800 per class).
- Datasets are analyzed separately.
- Victim models are analyzed separately.
- Inferential seed: attack seed 42 only.
- Seeds 42, 2024, and 2026 are used only to report mean ± sample SD of rates.
- Seeds and seed-level means are never treated as independent observations.
- Significance threshold: α = 0.05.

## Pairing and integrity checks

Before any statistic is computed, every loaded artifact is checked against the canonical `selection.json` for its dataset. The checks cover:

1. exact sample-ID order;
2. absence of duplicate sample IDs;
3. positional indices;
4. clean-input SHA-256;
5. stored seed, objective, and attack method;
6. budget and primitive mode where applicable;
7. equal outcome lengths; and
8. the invariant `valid_success ⊆ raw_success`.

The analyzer then requires identical qualified sample IDs across every condition in a paired family for all three seeds. A mismatch raises an exception before either output file is written. Conditions are never joined approximately or paired by row position without canonical-ID verification.

## Statistical protocol

### Primitive-mode ablation

Conditions:

- timing-only;
- padding-only; and
- joint.

All use the selected canonical PrimAttack optimizer, the untargeted objective, and the maximum-evaluated budget.

1. Cochran’s Q tests the null that the three paired `valid_success` probabilities are equal.
2. Pairwise testing proceeds only when Q is significant.
3. Planned McNemar comparisons are:
   - joint versus timing-only;
   - joint versus padding-only.
4. Holm correction is applied across exactly these two comparisons within each dataset–victim family.

The original locked final-suite protocol classified the mode ablation as descriptive. This inferential analysis was added after the final run and must be identified as a post-run analysis rather than described as pre-registered.

### Budget sensitivity

The requested conditions are the train-calibrated budgets:

- restricted = p25;
- intermediate = p50; and
- maximum-evaluated = p75.

The intended protocol is Cochran’s Q followed, when significant, by the adjacent McNemar comparisons restricted versus intermediate and intermediate versus maximum-evaluated, with Holm correction across the two comparisons.

This family is **not present in `statistical_tests.csv`** because the final suite has no stored restricted/p25 per-sample artifacts. Seventy-two required files are absent. The stored suite contains intermediate/p50, maximum-evaluated/p75, and unbounded conditions. Unbounded is not a valid substitute for restricted, so no three-condition budget test was fabricated. The available p50 and p75 rates remain descriptive in `statistical_summary.md`.

### Raw-vs-Valid validity gap

For each canonical headline attack condition, McNemar compares paired seed-42 outcomes from the same generated adversarial examples:

- A = `raw_success`;
- B = `valid_success`.

The included conditions are the five inferential Experiment-A methods and the three maximum-evaluated targeted PrimAttack optimizers. No Holm adjustment is applied because each condition answers its own dedicated Raw-vs-Valid question.

Because `valid_success = raw_success ∧ validator_pass`, `A fails/B succeeds` must equal zero. `A succeeds/B fails` is the number of apparent attack successes removed by validity enforcement.

## McNemar variant

- Fewer than 25 discordant pairs: exact two-sided binomial McNemar test.
- At least 25 discordant pairs: continuity-corrected χ² McNemar test.

For exact tests, `test_statistic` is blank because the reported result is based directly on the binomial discordance count. For asymptotic tests, `test_statistic` contains the continuity-corrected χ² value.

## CSV columns

| Column | Meaning |
|---|---|
| `analysis` | Primitive-mode ablation, budget sensitivity, or Raw-vs-Valid validity gap. |
| `test` | `Cochran's Q` or `McNemar`. |
| `dataset` | Dataset analyzed; never pooled. |
| `victim` | Victim checkpoint/model analyzed; never pooled. |
| `objective` | Targeted or untargeted attack objective. |
| `optimizer` | PrimAttack optimizer or baseline attack identifier. |
| `conditions` | Ordered semicolon-separated condition labels. Counts and rates use the same order. |
| `outcome` | Binary outcome used by the test. |
| `reference_seed` | Inferential attack seed; always 42. |
| `n_paired` | Number of paired source flows. |
| `success_counts` | Semicolon-separated success counts in condition order. |
| `success_rates` | Semicolon-separated success proportions in condition order. |
| `test_statistic` | Cochran Q or continuity-corrected McNemar χ²; blank for exact McNemar tests. |
| `df` | Degrees of freedom for Cochran’s Q; blank for McNemar. |
| `p_value` | Raw p-value. |
| `p_holm` | Holm-adjusted p-value for gated primitive-mode pairwise tests; blank when not applicable. |
| `significant_alpha_0_05` | `yes` or `no`, using Holm-adjusted p when Holm applies. |
| `a_succeeds_b_fails` | McNemar A-only directional discordance. |
| `a_fails_b_succeeds` | McNemar B-only directional discordance. |
| `status` | Performed test and variant, or reason a planned test was not performed. |

Semicolon-delimited values must be interpreted using the order in `conditions`. Rates are proportions in the CSV and percentages in the Markdown report.

## Reading the results

### Cochran’s Q

A significant Q establishes evidence that at least one of the three paired condition success probabilities differs. It does not identify which conditions differ. Only the predeclared McNemar contrasts provide that follow-up evidence.

When Q is not significant, the planned pairwise rows remain in the CSV with `status = not performed: omnibus Cochran's Q not significant`. Their p-value and significance fields are intentionally blank.

### McNemar

The directional discordances determine the result:

- `a_succeeds_b_fails` favors condition A;
- `a_fails_b_succeeds` favors condition B.

Equal marginal success rates do not by themselves prove identical per-sample behavior; inspect both directional counts. For Raw-vs-Valid, however, the subset invariant makes the direction one-sided in observed structure even though the reported McNemar p-value is two-sided.

### Holm-adjusted results

For primitive-mode planned comparisons, use `p_holm` and `significant_alpha_0_05`, not the unadjusted p-value, for the final inferential decision. Holm adjustment is local to the two planned comparisons for one dataset and one victim. There is no pooling or cross-victim correction.

## Statistical versus practical significance

Statistical significance addresses evidence against equal paired success probabilities. Practical significance is the magnitude of the difference:

- absolute success-rate difference;
- number of discordant flows;
- baseline success rate; and
- whether the difference changes the thesis conclusion.

With N = 3,200, a small difference can be statistically significant. A non-significant result does not establish equivalence. Mean ± SD over the three attack seeds describes run-to-run variability; it is not a confidence interval and does not change the paired seed-42 hypothesis test.

## Thesis reporting checklist

When citing these outputs:

1. State that inference uses paired per-flow outcomes from reference attack seed 42.
2. State that seeds 42, 2024, and 2026 contribute only descriptive mean ± SD.
3. Report dataset and victim separately.
4. Report N, success counts/rates, directional discordances, test variant, raw p-value, and Holm-adjusted p-value where applicable.
5. Use `valid_success` as the primary outcome except for the dedicated Raw-vs-Valid comparison.
6. Do not describe the primitive-mode inferential tests as pre-registered.
7. Do not claim a restricted/intermediate/maximum budget test was performed; restricted/p25 artifacts are missing.
8. Distinguish statistical significance from effect magnitude and operational relevance.
