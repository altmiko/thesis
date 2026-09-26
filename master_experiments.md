You are now setting up the FINAL experimental suite for this bachelor's thesis. Treat this as the definitive run from which all final thesis tables, statistics, figures, and conclusions will be generated.

Do not expand the experiment scope beyond what is defined here unless a methodological bug makes a result invalid. Do not silently change seeds, samples, budgets, attack objectives, denominators, or statistical procedures.

# 1. Global experimental rules

Use exactly these three seeds for every experiment:

- 42
- 2024
- 2026

The same three seeds must be used consistently across both datasets, all victim models, all attack methods, all budgets, and all experiments.

All comparisons must use paired samples.

For every dataset/model/class combination, construct or load one canonical source-sample list before running attacks. The identical source sample IDs must be used across every method or condition being compared.

The canonical pairing key should contain at minimum:

`dataset, victim_model, source_class, sample_id, seed`

No method may silently drop samples because an attack fails, a gradient fails, the optimizer stalls, the validator rejects a sample, or a result is missing.

Before aggregating any comparison, assert that the paired sample-ID sets are identical.

Fail loudly if there are:
- missing sample IDs
- duplicate sample IDs
- mismatched sample sets
- inconsistent seeds
- differing denominators
- incompatible result schemas
- partially completed runs being mistaken for complete experiments

Do not silently continue with incomplete paired data.

# 2. Metrics — LOCK THESE

Only use these two primary attack-success metrics:

## Raw ASR

`Raw ASR = number of successful adversarial attacks / number of attempted source samples`

## Valid ASR

`Valid ASR = number of adversarial samples that both succeed AND pass the independent domain validator / number of attempted source samples`

The denominator MUST remain the same attempted source-sample set.

Do NOT calculate Valid ASR only over already-valid adversarial samples.

Do not use IDR, IDSR, SP, semantic-preservation scores, realism scores, or any previous auxiliary attack-success metrics in the final headline tables.

Those may remain internally in legacy code if required, but they must not replace or confuse the two locked final metrics.

For every result table, report:

`mean ± standard deviation`

across the three seeds.

Example:

`42.18% ± 1.37%`

Also preserve the individual result for each of the three seeds in the markdown and machine-readable outputs.

# 3. Success definitions — LOCK THESE

## Targeted attack

The source is malicious.

A targeted success occurs only when:

`prediction(adversarial_sample) == Benign`

The primary scientific contribution of PrimAttack is VALID TARGETED SUCCESS, i.e.:

`targeted success AND domain-validator valid`

## Untargeted attack

An untargeted success occurs when:

`prediction(adversarial_sample) != original malicious class`

Do not count already-misclassified clean samples as successful attacks unless the existing evaluation methodology explicitly defines and justifies this. Prefer attacking only correctly classified source samples and document the exact rule consistently.

## Valid success

For either objective:

`valid_success = attack_success AND validator_pass`

The exact same final adversarial sample must be used for both the attack-success check and the validator check.

# 4. Validity-gap definition

For every relevant experiment calculate:

`Validity Gap = Raw ASR - Valid ASR`

Report this in percentage points.

This is a core thesis quantity and directly supports the paired validity-gap contribution.

Because Raw Success and Valid Success correspond to the same source attacks, preserve sample-level binary outcomes so paired statistical analysis can be performed.

#5. Statistical analysis — LOCK THIS

Keep the statistical analysis deliberately minimal, interpretable, and directly tied to the thesis research questions.

The primary inferential outcome is:

VALID SUCCESS

Raw ASR remains an important descriptive metric and must always be reported as mean ± SD, but do NOT duplicate every inferential test for Raw ASR.

The only statistical tools permitted in the final experimental suite are:

Cochran's Q test

McNemar's test

Holm correction for planned families of multiple McNemar comparisons

Do not introduce:

t-tests

ANOVA

Friedman tests

Wilcoxon tests

Mann–Whitney tests

bootstrap hypothesis tests

additional effect-size frameworks

other statistical tests unless an unavoidable methodological issue makes one necessary

Use:

alpha = 0.05

Pairing

All inferential tests must use identical paired source samples.

Do NOT perform the main statistical tests on the three seed-level aggregate means.

The three seeds:

42

2024

2026

are used to characterize run-to-run variability through:

mean ± SD

The inferential tests instead use paired sample-level binary success/failure outcomes.

Do not treat the three seed means as n=3 observations for hypothesis testing.

Document exactly how repeated seeded observations are handled so that observations are not incorrectly treated as independent.

Cochran's Q

Use Cochran's Q only when comparing 3 or more paired methods or conditions.

Its role is to answer:

"Is there evidence that valid attack success differs somewhere among these paired methods/conditions?"

Do not run Cochran's Q unnecessarily when only two conditions are compared.

McNemar's test

Use McNemar's test for planned two-condition comparisons on paired binary Valid Success outcomes.

For every McNemar comparison report:

conditions/methods compared

number of paired samples

successes unique to condition A

successes unique to condition B

paired success-rate difference in percentage points

McNemar test statistic where applicable

raw p-value

adjusted p-value where applicable

concise interpretation

The paired percentage-point difference must always accompany the p-value so that statistical significance is not presented without practical magnitude.

Holm correction

Use Holm correction only when multiple planned McNemar comparisons belong to the same logical experiment family.

Do not correct unrelated tests across the entire thesis together.

Raw ASR

For normal between-method/budget/objective comparisons:

report Raw ASR as mean ± SD

report Valid ASR as mean ± SD

perform inferential tests primarily on Valid Success

Do NOT duplicate the entire statistical analysis for Raw Success.

The exception is the dedicated paired validity-gap analysis, where Raw Success versus Valid Success is itself the research question.

Reporting

Do not output only "significant" or "not significant."

Always preserve:

test

sample count

comparison

p-value

Holm-adjusted p-value where applicable

paired percentage-point difference

discordant counts for McNemar

concise interpretation

Keep the statistical section simple enough to explain clearly in the thesis defense.

# 6. Machine-readable provenance

Every experiment must save machine-readable results in addition to the markdown.

At minimum save:
- per-sample results CSV or Parquet
- seed-level aggregate CSV
- table-level aggregate CSV
- statistical-test results CSV

The per-sample file must contain enough information to regenerate all final tables without rerunning attacks.

Include fields such as:
- dataset
- model
- source class
- sample ID
- seed
- method
- budget
- objective
- primitive mode
- clean prediction
- adversarial prediction
- raw success
- validator pass
- valid success
- relevant optimizer/search metadata
- runtime where available
- number of model evaluations where available

Preserve exact run configuration and hyperparameters.

# 7. Baseline fairness rules

The established attacks and PrimAttack do NOT necessarily share identical native threat models or perturbation budgets.

Do not falsely imply otherwise.

For primary baseline comparisons:
- use the exact same paired source samples
- use the same victim models
- use the same dataset splits
- use the same preprocessing/scaling
- use the same objective for that experiment
- evaluate all final adversarial samples using the same independent validator
- preserve each baseline's properly defined perturbation constraint unless there is already a principled common mapping

For PGD, C&W, CAPGD, and PrimAttack, explicitly document:
- perturbation norm/budget or primitive budget
- attack iterations
- restarts
- loss/objective
- targeted or untargeted setting
- feature mask
- projection/clamping
- model evaluation/query count where measurable
- stopping rule
- implementation/source

Include a clear fairness guide in each relevant markdown explaining what is held constant and what inherently differs between attack families.

# 8. PrimAttack realization rules

PrimAttack optimizes only its defined primitive-domain controls.

All PrimAttack optimizer variants must:
- use the same primitive parameterization
- use the same source samples
- use the same primitive budgets
- use the same primitive-to-feature recomputation
- use the same rounding/discretization
- use the same victim model
- use the same domain validator
- use the same success definition

All candidates used for final success reporting must correspond to actual rounded/discretized/recomputed flows.

Do not report success only on a smooth continuous surrogate.

Maintain the best candidate encountered per flow rather than checking only the final optimization step.

A successful candidate beats a failed candidate.

Among successful candidates, retain the lowest-cost success according to the existing PrimAttack primitive-cost rule.

Among failed candidates, retain the candidate with the best attack objective/margin.

# 9. FINAL EXPERIMENT A — PRIMARY BASELINE COMPARISON

This is the main cross-method experiment for the thesis.

The purpose is to compare PrimAttack against conventional white-box attacks and two public constrained white-box baselines under a common paired evaluation scenario.

## Methods

Compare:

- PrimAttack
- PGD
- C&W
- CAPGD-PrimSupport
- C-PGD-PrimSupport

`CAPGD-PrimSupport` and `C-PGD-PrimSupport` MUST use the canonical `primattack_joint_feature_mask` generated from the audited PrimAttack implementation.

Only downstream classifier features that can actually be changed/recomputed by PrimAttack may be modified by CAPGD or C-PGD in this experiment.

All features outside that mask must remain exactly unchanged.

Do NOT claim that CAPGD/C-PGD and PrimAttack have identical feasible attack spaces.

The controlled distinction is:

- CAPGD/C-PGD directly optimize the allowed downstream feature coordinates;
- PrimAttack optimizes packet-size/timing primitives and reaches those downstream features only through its coupled recomputation function.

This experiment therefore controls downstream feature support while preserving the fundamentally different attack parameterizations.

## Objective

Use the same **untargeted evasion objective** for all five attacks in this experiment.

A successful attack satisfies:

`prediction(adversarial_sample) != original malicious class`

Use the exact same correctly classified malicious source samples for every method.

Do not compare PrimAttack targeted-to-Benign results to untargeted baseline results in this experiment.

PrimAttack's main targeted-to-Benign evaluation remains in its dedicated experiments elsewhere in the master suite.

## Pairing

For every:

`dataset, victim_model, source_class, sample_id, seed`

the exact same source observation must be evaluated under every attack.

Before aggregation, assert:

- identical sample IDs;
- identical clean inputs;
- identical labels;
- identical victim checkpoints;
- identical seeds;
- identical denominators.

No method may silently drop failures.

## Fairness

Hold constant:

- source samples
- victim models
- preprocessing/scaling
- dataset split
- objective
- seed
- success definitions
- final independent validator
- aggregation rules

The attack families may retain different optimization algorithms and native budget definitions.

Do not force numerically identical epsilon values across incomparable perturbation representations.

Explicitly document for every method:

- attack norm/budget
- iterations
- step size where applicable
- restarts
- loss
- feature mask
- projection/clamping
- stopping rule
- implementation/source
- gradient/model-evaluation count where measurable

For CAPGD and C-PGD explicitly document that their feature mask is the PrimAttack-derived affected-feature mask.

## Metrics

Use only the locked headline metrics:

### Raw ASR

`successful attacks / attempted source samples`

### Valid ASR

`successful AND validator-valid attacks / attempted source samples`

Use the same denominator.

Also calculate:

`Validity Gap = Raw ASR - Valid ASR`

Report:

`mean ± SD`

across seeds:

- 42
- 2024
- 2026

Preserve each individual seed result.

## Primary inferential outcome

The primary inferential outcome is:

**Valid Success**

Raw ASR is descriptive in this experiment.

Do not duplicate the inferential tests for Raw Success.

## Statistical analysis

There are now five paired attack methods:

- PrimAttack
- PGD
- C&W
- CAPGD-PrimSupport
- C-PGD-PrimSupport

First run:

**Cochran's Q**

across all five methods using paired sample-level binary **Valid Success** outcomes.

Its purpose is to test whether valid attack success differs somewhere among the five paired methods.

If the overall Cochran's Q comparison is significant, perform ONLY these four planned McNemar comparisons:

- PrimAttack vs PGD
- PrimAttack vs C&W
- PrimAttack vs CAPGD-PrimSupport
- PrimAttack vs C-PGD-PrimSupport

Do NOT perform all possible baseline-vs-baseline pairwise tests.

Apply **Holm correction** across these four planned PrimAttack-versus-baseline McNemar tests.

For every comparison report:

- methods compared
- paired sample count
- PrimAttack-only valid successes
- baseline-only valid successes
- Valid ASR difference in percentage points
- McNemar statistic where applicable
- raw p-value
- Holm-adjusted p-value
- concise interpretation

Use:

`alpha = 0.05`

Do not treat the three seed means as n=3 statistical observations.

The three seeds characterize run-to-run variability through mean ± SD.

The inferential tests must use paired sample-level binary outcomes according to the master statistical rules.

## Required diagnostic analysis

For CAPGD-PrimSupport and C-PGD-PrimSupport report:

- number of mutable features in the PrimAttack support mask;
- number of actually modified features;
- confirmation that zero features outside the mask changed;
- Raw ASR;
- Valid ASR;
- Validity Gap.

For PrimAttack additionally report its existing primitive-domain cost/budget information.

This allows the report to distinguish:

1. direct optimization over the PrimAttack-affected downstream features;
2. primitive-domain optimization whose downstream changes are coupled by recomputation.

## Required tables

### Main baseline table

Include:

- Dataset
- Victim
- Attack
- Budget/configuration
- Raw ASR mean ± SD
- Valid ASR mean ± SD
- Validity Gap
- relevant perturbation/budget metric

for:

- PGD
- C&W
- CAPGD-PrimSupport
- C-PGD-PrimSupport
- PrimAttack

### Constrained-baseline diagnostic table

Compare:

- CAPGD-PrimSupport
- C-PGD-PrimSupport
- PrimAttack

Include:

- allowed downstream feature count
- mean number of modified features
- Raw ASR
- Valid ASR
- Validity Gap
- validator pass rate
- attack/model evaluation count where measurable

Do not imply that modified-feature count is directly equivalent to PrimAttack primitive cost.

## Required plots

Update the baseline plots to include C-PGD:

- Raw ASR by attack
- Valid ASR by attack
- class-wise Valid ASR
- model-wise Valid ASR
- Validity Gap by attack

Avoid redundant figures.

## Final interpretation

The thesis-ready interpretation of this experiment should answer:

1. How much attack success is obtained by ordinary unrestricted feature-space attacks?
2. What happens when public constrained attacks are restricted to only the downstream feature coordinates that PrimAttack can influence?
3. What happens when the attacker is further restricted to reaching those features indirectly through PrimAttack's small packet-size/timing primitive domain?
4. How much of each method's Raw ASR disappears once the independent domain validator is required?

Do not automatically describe the method with the highest ASR as the "best" attack.

Interpret differences in relation to the threat model and allowed attack space.

Do not claim that the PrimAttack feature mask establishes packet-level realizability for CAPGD/C-PGD.

It only provides a controlled matched-support comparison.

All other global experimental, provenance, validity-gap, and statistical rules from the master final-evaluation prompt remain unchanged.

10. FINAL EXPERIMENT B — PRIMATTACK OPTIMIZER SELECTION

Statistics — REPLACE THE EXISTING STATISTICS SUBSECTION WITH THIS

The primary inferential outcome is Valid Targeted Success.

Methods:

Hybrid Search

Prim-PGD

Prim-C&W

First run:

Cochran's Q

across the three optimizers.

If significant, perform these three planned paired McNemar comparisons:

Hybrid vs Prim-PGD

Hybrid vs Prim-C&W

Prim-PGD vs Prim-C&W

Apply Holm correction across these three tests.

Report:

paired sample counts

discordant success counts

Valid Targeted ASR difference in percentage points

raw p-values

Holm-adjusted p-values

interpretation

Raw Targeted ASR should be reported as:

mean ± SD

but does not require a duplicate set of inferential tests.

Optimizer selection remains based on the predefined:

aggregate Valid Targeted ASR

not on p-value alone.

Statistical testing provides supporting evidence and uncertainty; it does not replace the locked optimizer-selection criterion.

11. FINAL EXPERIMENT C — BUDGET SENSITIVITY

Statistics — REPLACE THE EXISTING STATISTICS SUBSECTION WITH THIS

The primary inferential outcome is Valid Targeted Success.

For each optimizer being evaluated, compare:

p50

p75

unbounded

using identical paired source samples.

First run:

Cochran's Q

across the three budgets.

If significant, perform ONLY these planned adjacent McNemar comparisons:

p50 vs p75

p75 vs unbounded

Apply Holm correction across these two budget comparisons.

These comparisons are intended to test whether each increase in attacker capability materially changes valid targeted attack success.

Do not automatically test every possible budget pair unless needed to resolve an unexpected pattern.

For the two optimizers used in this experiment, their Valid ASR values at each budget should be reported side by side.

A direct optimizer-vs-optimizer McNemar comparison at every budget is NOT required as part of the minimum statistical suite.

Only perform such a comparison if it is necessary to explain the central budget-sensitivity result.

Raw ASR remains descriptive:

mean ± SD

Do not duplicate the budget hypothesis tests for Raw Success.

12. FINAL EXPERIMENT D — OBJECTIVE SENSITIVITY

Statistics — REPLACE THE EXISTING STATISTICS SUBSECTION WITH THIS

This experiment contains only two paired conditions:

targeted-to-Benign

untargeted

Use ONE paired McNemar test comparing:

Valid targeted success vs Valid untargeted success

using identical source samples.

Report:

number of paired samples

targeted-only valid successes

untargeted-only valid successes

difference in Valid ASR in percentage points

McNemar p-value

interpretation

No Cochran's Q test is required.

No Holm correction is required because there is only one planned inferential comparison.

Report Raw targeted ASR and Raw untargeted ASR as:

mean ± SD

but do not run an additional Raw Success McNemar test.

13. FINAL EXPERIMENT E — PAIRED VALIDITY-GAP ANALYSIS

Statistics — REPLACE THE EXISTING STATISTICAL INSTRUCTIONS WITH THIS

This is the one experiment where Raw Success versus Valid Success is directly tested because the validity gap itself is the research question.

For each main attack/method condition, use a paired McNemar test comparing:

Raw Success

against:

Valid Success

on the exact same generated adversarial examples.

Remember:

Valid Success is necessarily a subset of Raw Success.

Therefore the important discordant category is:

raw success = 1, valid success = 0

which represents attacks that successfully fool the classifier but fail the independent domain validator.

Report:

number of paired adversarial examples

Raw ASR

Valid ASR

Validity Gap in percentage points

number of raw-success-but-invalid cases

number of valid-success cases

McNemar p-value

concise interpretation

Do not run Cochran's Q here.

Do not introduce additional statistical tests.

If many attack/method conditions are tested independently, clearly organize them by experiment rather than creating an unnecessarily large cross-thesis multiple-testing family.

The purpose of this analysis is simple:

"How much classifier-level attack success disappears when domain validity is required, and is that paired loss systematic?"

14. FINAL EXPERIMENT F — VALIDATOR EVALUATION

Statistics

No inferential hypothesis test is required for the validator evaluation.

This experiment is descriptive.

Report:

number of held-out genuine samples

accepted samples

rejected samples

acceptance rate

rejection rate

rejection categories

general-only validation results where available

general + dataset-specific validation results where available

Use counts, percentages, and clear descriptive comparisons.

Do not add statistical tests simply to increase the amount of analysis.

# 15. Required plots

Keep plots useful and limited.

For the primary baseline comparison:
- Raw ASR by attack
- Valid ASR by attack
- class-wise Valid ASR
- model-wise Valid ASR
- p75 vs unbounded PrimAttack comparison

For optimizer selection:
- Raw ASR by optimizer
- Valid ASR by optimizer
- runtime/model-evaluation comparison

For budget sensitivity:
- Raw ASR versus budget
- Valid ASR versus budget
- Validity Gap versus budget

For objective sensitivity:
- targeted vs untargeted Raw ASR
- targeted vs untargeted Valid ASR

For validity-gap analysis:
- Raw ASR versus Valid ASR
- validity-gap by attack/method

Do not create dozens of redundant figures.

# 16. Thesis contribution mapping

The final reports must explicitly map back to these thesis contributions:

## Contribution 1
PrimAttack: a constrained adversarial attack framework operating over attacker-controllable packet-size and timing primitives.

Evidence:
- primary baseline comparison
- optimizer-selection experiment

## Contribution 2
Adversarial evaluation methodology separating attack objective, domain validity and constraint-valid attack success under controlled perturbation conditions.

Evidence:
- baseline comparison
- budget sensitivity
- objective sensitivity

## Contribution 3
Paired validity-gap analysis quantifying the difference between raw attack success and constraint-valid attack success on identical source samples.

Evidence:
- paired_validity_gap_analysis.md
- paired outputs embedded throughout other reports

## Contribution 4
Independent domain-validation framework combining general network-flow consistency constraints with automatically derived dataset-specific constraints.

Evidence:
- validator_evaluation.md

## Contribution 5
Controlled PrimAttack perturbation-budget analysis.

Evidence:
- primattack_budget_sensitivity.md

Include this mapping in the final experiment-summary documentation.


17. Final master summary — STATISTICS REQUIREMENT

The final master report must summarize only the statistical tests required by the experiment design.

The statistical methodology should be described as:

mean ± SD across seeds 42, 2024 and 2026 for run-to-run variability

paired sample-level binary inference

Cochran's Q for 3+ paired conditions

planned McNemar comparisons for two-condition contrasts

Holm correction only for logical families containing multiple planned McNemar comparisons

alpha = 0.05

Valid Success as the primary inferential outcome

Raw ASR primarily descriptive except in the dedicated validity-gap analysis

Do not introduce additional statistical tests during final report generation.

20. Final instruction — ADD THIS STATISTICAL RULE

These are the FINAL thesis experiments.

The statistical analysis must remain deliberately minimal and interpretable.

The final permitted inferential toolkit is:

Cochran's Q + McNemar's test + Holm correction

Use Valid Success as the main inferential outcome.

Use the three seeds to report:

mean ± SD

Do not treat the three seed means as the statistical sample.

Do not add further hypothesis tests merely because they are available.

The goal is to provide sufficient statistical evidence for the thesis claims while keeping the methodology transparent, paired, reproducible, and straightforward to defend.