I need you to produce a complete, thesis-ready Markdown draft of ONLY the following parts of my B.Sc. thesis:

1. **Proposed Methodology**
2. **Results and Evaluation / Result Analysis**

Do NOT write the Introduction, Literature Review, Requirements/Impacts chapter, Conclusion, Abstract, or bibliography.

The output must be a single detailed Markdown file at the repository root:

```text
thesis_methodology_results.md
```

The document must be written as an actual thesis chapter draft, not as developer documentation, notes, or a code audit.

---

# PRIMARY REQUIREMENT

Before writing anything, perform a full repository audit.

The repository is the authoritative source for:

- datasets;
- preprocessing;
- train/validation/test construction;
- feature definitions;
- victim classifiers;
- attack implementations;
- PrimAttack;
- primitive capability inference;
- feature recomputation;
- validator;
- attack baselines;
- experiment definitions;
- seeds;
- budgets;
- metrics;
- statistical tests;
- result artifacts;
- ablations;
- figures;
- tables;
- configuration files;
- reproducibility information.

Do not rely on assumptions from filenames alone.

Trace the implementation and experiment artifacts.

If documentation conflicts with executable code or final result artifacts, identify the conflict and use the final canonical implementation/result while leaving a short TODO/note for me where clarification is necessary.

Do NOT fabricate missing results.

Use explicit placeholders where final results, figures, tables, or values are not yet available.

---

# WRITING STYLE

Write in formal but readable B.Sc. thesis language.

The writing should:

- explain what was done;
- explain why it was done;
- explain how it works;
- give enough technical detail for reproduction;
- avoid unnecessary complicated vocabulary;
- avoid AI-like filler;
- avoid repeatedly saying "novel", "robust", "significant", etc.;
- distinguish methodology from interpretation;
- distinguish assumptions from demonstrated facts;
- avoid claims stronger than the experiments support.

Use paragraphs for the thesis narrative.

Use equations where they make the method easier to understand.

Use tables where appropriate.

Do not turn every subsection into bullet points.

---

# THESIS TEMPLATE / ORGANIZATION

Use a structure similar to a conventional BRAC University CSE thesis:

```text
# Chapter 4 — Proposed Methodology

## 4.1 Methodology Overview
## 4.2 Research and Experimental Design
## 4.3 Datasets
## 4.4 Dataset Preprocessing
## 4.5 Victim NIDS Models
## 4.6 Adversarial Threat Model
## 4.7 Evaluation Framework
## 4.8 Domain Validity Framework
## 4.9 Baseline Adversarial Attacks
## 4.10 PrimAttack
## 4.11 Experimental Protocol
## 4.12 Statistical Analysis
## 4.13 Reproducibility and Implementation Environment

# Chapter 5 — Results and Evaluation

## 5.1 Evaluation Overview
## 5.2 Dataset and Victim Model Results
## 5.3 Experiment A — Main Attack Comparison
## 5.4 Raw vs Constraint-Valid Attack Success
## 5.5 PrimAttack Primitive Ablation
## 5.6 PrimAttack Budget Analysis
## 5.7 PrimAttack Optimizer / Search Ablation
## 5.8 Capability-Aware Padding Analysis
## 5.9 Statistical Evaluation
## 5.10 Cross-Dataset Analysis
## 5.11 Synthesis and Discussion
## 5.12 Limitations
```

Change subsection numbering/names if the repository shows that another organization is more accurate.

However, preserve the basic separation:

**Methodology = what was designed and how experiments were conducted.**

**Results = what happened and what the measurements mean.**

Do NOT mix results into the methodology except where a small descriptive dataset statistic is necessary.

---

# IMPORTANT: FIGURE AND TABLE PLACEHOLDERS

Throughout the document, explicitly identify where figures and tables should go.

Use this exact general style:

```markdown
> **[FIGURE 4.X PLACEHOLDER — Overall experimental pipeline]**
> Suggested content: Dataset → preprocessing → victim training → attacks → validator → raw/valid metrics → statistical analysis.
> Suggested source/artifact: `...`
> Purpose: Gives the reader a single overview of the experimental workflow.
```

and:

```markdown
> **[TABLE 4.X PLACEHOLDER — Dataset split statistics]**
> Suggested columns: Dataset, class, train, validation, test, percentage.
> Suggested source/artifact: `...`
> Purpose: Documents the final experimental population.
```

Do this for EVERY figure/table that would materially improve the thesis.

Do NOT invent figure numbers permanently if final numbering is uncertain; use sequential chapter placeholders.

Where an existing plot/table artifact already exists, give its exact path.

Where it does not exist, explain exactly what should be generated.

---

# CHAPTER 4 — PROPOSED METHODOLOGY

Write this chapter in substantial detail.

---

## 4.1 Methodology Overview

Explain the complete research pipeline from beginning to end.

The central research question is approximately:

> How does adversarial attack effectiveness against flow-based machine-learning/deep-learning NIDS change as progressively stronger validity, feature-control, and primitive-domain restrictions are imposed on the adversary?

Determine the exact final framing from the repository/current experiment design.

Explain the progression from:

```text
dataset
→ leakage-aware preprocessing
→ victim classifier training
→ feature-space adversarial attacks
→ constraint-aware attacks
→ matched-support attacks
→ PrimAttack
→ domain validation
→ raw vs valid success
→ paired statistical evaluation
```

Include:

> **[FIGURE PLACEHOLDER — Complete research pipeline]**

---

## 4.2 Research and Experimental Design

Explain:

- white-box threat model;
- source samples;
- unit of analysis;
- paired/matched experimental design;
- attack classes;
- victim models;
- datasets;
- seeds;
- repeated runs;
- why the same source samples are reused where applicable;
- how targeted and untargeted evaluations differ;
- why raw and valid ASR are measured separately.

Clarify the distinction between:

```text
classifier success
domain validity
feature support
primitive capability
primitive reachability
packet-level realization
```

Do not collapse these concepts together.

---

# 4.3 Datasets

Audit and describe every canonical dataset currently used.

This likely includes:

- CICIDS2017 DistriNet/corrected release;
- CICIDS2018 DistriNet/corrected release;

and any other dataset actually still present in the final experiment.

Do NOT include abandoned datasets simply because old code remains.

For each dataset explain:

- source;
- flow extractor;
- feature count;
- traffic classes;
- source labels;
- class grouping;
- final labels used in this thesis;
- size;
- known dataset issues;
- why it is suitable for this evaluation.

Include:

> **[TABLE PLACEHOLDER — Dataset overview]**

and:

> **[TABLE PLACEHOLDER — Final class mapping by source attack label]**

---

# 4.4 Dataset Preprocessing

THIS SECTION MUST BE VERY DETAILED.

Audit all preprocessing code.

Explain from raw input to final tensors/tables.

Include, where applicable:

- loading;
- corrected/DistriNet inputs;
- label normalization;
- class mapping;
- invalid values;
- NaN/Inf treatment;
- duplicate handling;
- leakage prevention;
- temporal splitting;
- source-ID construction;
- train/validation/test separation;
- Benign sampling/balancing;
- class balancing;
- scaling;
- transformations;
- feature filtering;
- categorical/binary handling;
- feature ordering;
- persisted artifacts;
- immutable pristine test set;
- sample selection for attacks.

Explain exactly how the final split prevents leakage.

If the final design uses approximately 30% Benign composition per split, verify this from final artifacts before writing it.

Do not copy old preprocessing descriptions if they no longer match code.

Include:

> **[FIGURE PLACEHOLDER — Dataset preprocessing pipeline]**

> **[TABLE PLACEHOLDER — Train/validation/test class counts]**

> **[TABLE PLACEHOLDER — Final feature set and feature categories]**

If 2017 and 2018 require materially different preprocessing, give separate subsections.

---

# 4.5 Victim NIDS Models

Audit final victim classifiers.

Likely models include some subset of:

- MLP;
- CNN;
- FT-Transformer;
- CNN-LSTM;

but use only the actual final models.

For each model describe:

- input;
- architecture;
- layers;
- activation;
- normalization;
- dropout;
- hidden dimensions;
- output;
- loss;
- optimizer;
- learning rate;
- batch size;
- epochs;
- early stopping;
- class weighting;
- thresholding if used;
- random seeds.

Explain why multiple victim architectures are used.

Do not spend excessive space on standard architectures compared with PrimAttack.

Include:

> **[TABLE PLACEHOLDER — Victim architecture and training hyperparameters]**

> **[TABLE PLACEHOLDER — Victim performance on validation/test data]**

Potential figure:

> **[FIGURE PLACEHOLDER — Simplified victim-model architectures]**

Only include if useful.

---

# 4.6 Adversarial Threat Model

Define the final adversary precisely.

Explain:

- white-box access;
- classifier access;
- gradients;
- access to feature extractor relationships;
- source malicious flow;
- untargeted evasion;
- targeted malicious-to-Benign attack where applicable;
- attacker capability assumptions;
- what the adversary cannot change.

Clearly state that this is OFFLINE FEATURE/FLOW-LEVEL RESEARCH.

Do not claim actual live-network exploitation.

Define the representation levels:

```text
packet level
flow level
feature space
primitive domain
```

Explain that packet construction/replay is outside scope.

---

# 4.7 Evaluation Framework

Define every final metric mathematically.

At minimum inspect whether the thesis uses:

### Raw ASR

For untargeted attacks:

\[
ASR_{raw} =
\frac{\#\{\text{successful misclassifications}\}}
{\#\{\text{eligible attempted malicious samples}\}}
\]

Use the project's exact denominator.

### Valid ASR

\[
ASR_{valid} =
\frac{\#\{\text{successful AND valid adversarial samples}\}}
{\#\{\text{eligible attempted samples}\}}
\]

Again verify exact implementation.

### Targeted-to-Benign ASR

Define properly.

### Validity rate

Define.

### Perturbation magnitude/cost

Document all final metrics such as:

- L1;
- L2;
- L∞;
- normalized perturbation;
- primitive cost;
- timing modification;
- padding modification;
- distributional cost;

only if actually retained.

Explain why Valid ASR is one of the central thesis measurements.

Include:

> **[TABLE PLACEHOLDER — Complete metric definitions]**

---

# 4.8 Domain Validity Framework

THIS SECTION MUST BE VERY DETAILED.

Audit the full validator implementation.

Explain:

1. purpose;
2. source vs adversarial comparison;
3. constraint categories;
4. mechanically derived rules;
5. mined/data-derived rules;
6. protocol/domain rules;
7. extractor identities;
8. min/mean/max ordering constraints;
9. bounds;
10. binary/discrete rules;
11. source-conditioned rules;
12. any class-conditioned constraints;
13. primitive-specific restrictions where relevant.

For every rule family explain:

- what it checks;
- why it exists;
- whether it is derived from extractor mathematics, protocol semantics, training data, or manually specified logic.

Clearly distinguish:

### Validator validity

from:

### PrimAttack primitive capability.

This distinction is now very important.

The validator answers approximately:

> Is the resulting flow-feature vector consistent with the implemented domain requirements?

PrimAttack capability inference answers:

> Is this particular modeled primitive available for this source flow?

Do not imply that passing the validator proves packet-level realizability.

Include:

> **[FIGURE PLACEHOLDER — Validator architecture / rule families]**

> **[TABLE PLACEHOLDER — Validator rule categories and examples]**

> **[TABLE PLACEHOLDER — Number of rules by source/category/dataset]**

if artifacts allow it.

---

# 4.9 Baseline Adversarial Attacks

Audit every baseline retained in the final Experiment A.

Potential methods include:

- PGD;
- C&W;
- CPGD;
- CAPGD;
- CAPGD-PrimSupport.

Use exact implementations.

For each baseline explain:

- objective;
- targeted/untargeted mode;
- norm;
- epsilon/budget;
- iterations;
- step size;
- initialization;
- projection;
- constraints;
- feature mask;
- restarts;
- stopping criterion.

Explain native CAPGD/CPGD separately from any matched-support variants.

---

## 4.9.X CAPGD-PrimSupport / Matched-Support Baseline

This subsection is important.

Explain that matched-support CAPGD is used to reduce one major confound:

> differences caused merely by the set of downstream features available to the attacker.

Audit exactly how the mask is constructed from PrimAttack.

State whether it is:

- a static feature-name support mask;
- sample-specific;
- capability-specific;

and do not assume these are equivalent.

Explain:

> CAPGD-PrimSupport can directly optimize permitted downstream feature values, while PrimAttack obtains feature changes through primitive variables and deterministic recomputation.

Therefore:

```text
same support ≠ same feasible set
```

if that is truly supported by the implementation.

This is essential for interpreting Experiment A.

---

# 4.10 PrimAttack

THIS IS THE MOST IMPORTANT METHODOLOGY SECTION.

Give it the highest level of detail.

Break into subsections.

---

## 4.10.1 Motivation

Explain why direct perturbation of arbitrary extracted flow features is problematic.

Discuss:

- dependency between extracted features;
- attacker control;
- derived statistics;
- feature consistency;
- reachability.

State that PrimAttack attempts to move adversarial optimization from independent downstream feature manipulation toward a small set of attacker-facing flow-level controls.

Do not claim full packet realizability.

---

## 4.10.2 Primitive Domain

Define "primitive" carefully.

Explain exactly which primitives exist in the final implementation.

Likely:

- packet-length/padding augmentation;
- timing/delay modification;

but inspect current code.

For each primitive explain:

- variable;
- physical/traffic interpretation;
- directionality;
- admissible range;
- capability condition;
- affected downstream features.

Include a formal parameter vector, e.g.:

\[
z = (p,t)
\]

only if it reflects implementation.

---

## 4.10.3 Primitive-to-Feature Mapping

This must be extremely detailed.

Read the code and derive the actual equations.

For padding explain all affected statistics, for example where applicable:

\[
L'_{\text{fwd,total}} =
L_{\text{fwd,total}} + N_f p
\]

and all corresponding:

- min;
- max;
- mean;
- totals;
- rates;
- dependent statistics;

that are recomputed.

Do not invent equations.

For timing similarly explain:

- duration;
- IAT statistics;
- flow rate changes;
- dependent quantities.

Explain which quantities remain invariant.

Include:

> **[TABLE PLACEHOLDER — Primitive → directly affected features → recomputed dependent features]**

and:

> **[FIGURE PLACEHOLDER — Primitive-to-feature recomputation graph]**

This should be one of the central thesis figures.

---

## 4.10.4 Capability Inference

Explain the source-conditioned capability system.

Particularly document the new padding condition.

Padding must NOT be allowed merely because a flow contains some forward payload.

The final implementation should enforce:

```text
Fwd Packet Length Min > 0
```

in addition to the existing capability requirements.

Explain why.

A zero minimum tells us that at least one forward packet contributes zero length.

Since aggregate flow data do not identify which specific packet is empty, uniformly adding `p` to all forward packets would necessarily change an originally empty packet into a positive-length one.

Therefore padding is conservatively disabled.

Avoid claiming the zero-length packet is definitely a SYN or ACK.

The flow representation cannot establish that.

State that this is a conservative capability rule.

Explain:

- timing-only capability;
- padding-only capability;
- joint capability;
- no-primitive case.

Include:

> **[TABLE PLACEHOLDER — Primitive capability rules]**

---

## 4.10.5 Primitive Budgets

Explain all final budget levels.

If the final design uses:

- restricted;
- intermediate;
- maximum/envelope;

or:

- p50;
- p75;
- envelope;

use the actual implementation.

For every bound explain where it comes from.

For example, if padding uses limits involving:

```text
Fwd Packet Length Max
Fwd Packet Length Min
Fwd Packet Length Mean
Total Length of Fwd Packets
N_f
p_max
```

derive the bound explicitly from the code.

Explain all p99/global training envelopes or class-calibrated limits exactly.

Explain why all statistics used for budget calibration are computed from training data only.

Include:

> **[TABLE PLACEHOLDER — PrimAttack budget levels and definitions]**

---

## 4.10.6 Optimization Objective

Explain the canonical optimizer.

If PrimAttack is a C&W-style optimization with Adam, derive the full objective.

Explain:

- adversarial loss;
- targeted vs untargeted term;
- primitive cost;
- constraint terms;
- initialization;
- number of steps;
- learning rate;
- projection/clamping;
- early stopping;
- restarts.

Do NOT casually call it C&W if the implementation only uses a C&W-like margin loss.

Use technically precise wording.

---

## 4.10.7 Timing-Only Optimization

Document what happens when padding is unavailable.

This is now important.

For a source flow where:

```text
padding_allowed = False
timing_allowed = True
```

PrimAttack must become a genuine timing-only optimization.

Explain how the implementation ensures:

- `p = 0`;
- padding receives no effective search effort;
- timing receives the available optimization iterations/restarts;
- the overall canonical attack budget is not artificially increased.

This behavior should be explicitly described.

---

## 4.10.8 PrimAttack Output and Validation

Explain sequence:

```text
source flow
→ capability inference
→ primitive optimization
→ deterministic recomputation
→ adversarial classifier prediction
→ domain validator
→ attack metrics
```

Include:

> **[FIGURE PLACEHOLDER — Detailed PrimAttack pipeline]**

---

## 4.10.9 Realizability Scope

Include a careful thesis-level statement.

Explain that PrimAttack is:

> a realizability-oriented flow-level abstraction.

It is closer to traffic manipulation than independent feature optimization because downstream changes are generated from primitive controls and recomputation.

However:

- packets are not directly constructed;
- traffic is not replayed;
- CICFlowMeter is not rerun on modified PCAPs as part of the canonical experiment;
- therefore packet-level realizability is not established.

This limitation must remain explicit.

---

# 4.11 Experimental Protocol

Create a master experiment table.

Include all final experiments.

At minimum inspect for:

### Experiment A — Main attack comparison
Main baselines versus PrimAttack.

### Raw vs valid attack success
Quantifying how many apparent successes disappear under validity enforcement.

### Primitive ablation
Timing-only vs padding-only vs joint.

### Budget sweep
Effect of primitive budgets.

### Optimizer/search ablation
For example:
- C&W-style/Adam;
- Prim-PGD;
- random search;

ONLY if these are still part of the final thesis.

### Capability / empty-packet experiment
Old relaxed padding versus capability-aware PrimAttack if retained.

### Dataset comparison
CICIDS2017 vs CICIDS2018.

### Victim architecture comparison
If intentional.

For every experiment state:

- purpose;
- independent variable;
- controlled variables;
- source samples;
- datasets;
- victims;
- seeds;
- attack configuration;
- outcomes;
- statistical comparison.

Include:

> **[TABLE PLACEHOLDER — Complete experimental matrix]**

This table should be sufficient for a reviewer to understand the entire evaluation.

---

# 4.12 Statistical Analysis

Use ONLY the final minimal statistical protocol already adopted by the project.

Do not invent unnecessary ANOVA/testing.

Audit current statistical code and final design.

Explain:

- unit of analysis;
- pairing key;
- why observations are paired;
- seeds;
- binary outcomes;
- continuous outcomes;
- test(s) used;
- effect size if retained;
- multiple comparison handling if retained;
- significance threshold;
- mean ± SD descriptive reporting.

If McNemar is used for paired binary success outcomes, explain it.

If Wilcoxon signed-rank is used for paired continuous outcomes, explain it.

If another final test replaced them, use that instead.

Explain what statistical significance establishes and what it does NOT establish.

Include equations where appropriate.

Include:

> **[TABLE PLACEHOLDER — Statistical questions, outcome type, test and effect measure]**

---

# 4.13 Reproducibility and Implementation Environment

Document:

- OS/environment;
- Python;
- PyTorch;
- major libraries;
- hardware if recorded;
- seeds;
- deterministic controls;
- experiment manifests;
- artifact directories;
- model checkpoints;
- config files;
- sample IDs;
- provenance;
- commands where useful.

Do not include machine-specific private paths unless necessary.

---

# CHAPTER 5 — RESULTS AND EVALUATION

This chapter must report actual final artifacts.

Do not invent numbers.

Any experiment that is currently running or invalidated by the new PrimAttack capability rule must receive a clear placeholder:

```markdown
> **[RESULT PENDING — rerun capability-aware PrimAttack with seeds 42, 2024, 2026]**
```

Do not quietly reuse obsolete results.

---

# 5.1 Evaluation Overview

Briefly state what this chapter evaluates.

Explain that emphasis is placed on:

- victim quality;
- raw adversarial success;
- constraint-valid success;
- primitive-domain success;
- perturbation/cost;
- consistency across victims/datasets;
- statistical reliability.

Include a compact roadmap table if useful.

---

# 5.2 Dataset and Victim Model Results

Report final dataset distributions and classifier performance.

For each dataset/victim include suitable metrics such as:

- accuracy;
- balanced accuracy;
- macro F1;
- per-class recall;
- confusion matrix;

based on what the repository actually computes.

Do not over-focus on classifier benchmarking—the victims exist to support the adversarial evaluation.

Placeholders:

> **[TABLE — Final dataset split distributions]**

> **[TABLE — Victim model test performance, mean ± SD if applicable]**

> **[FIGURE — Confusion matrix for each canonical victim/dataset, if worth including]**

Interpret enough to establish that the victims are meaningful attack targets.

---

# 5.3 Experiment A — Main Attack Comparison

THIS IS THE MAIN RESULTS SECTION.

Identify the exact purpose of Experiment A from code/docs.

Create final tables by:

```text
Dataset × Victim × Attack Method
```

Report:

- attempted N;
- raw ASR;
- valid ASR;
- targeted ASR if applicable;
- valid targeted ASR if applicable;
- mean ± SD;
- validity rate;
- perturbation cost if relevant.

Include:

- PGD;
- C&W;
- CPGD;
- CAPGD;
- CAPGD-PrimSupport;
- PrimAttack;

only as applicable.

Mark superseded PrimAttack results clearly and do not use them as canonical.

Include:

> **[TABLE — Experiment A, CICIDS2017]**

> **[TABLE — Experiment A, CICIDS2018]**

> **[FIGURE — Raw vs Valid ASR grouped by method and victim]**

Potential figure:

> **[FIGURE — Constraint ladder showing ASR reduction across progressively restricted attacks]**

only if the comparisons genuinely support such a ladder.

---

# 5.3.X CAPGD-PrimSupport vs PrimAttack

Give this comparison its own subsection.

Explain:

- what is matched;
- what is not matched;
- feature support;
- validator;
- source samples;
- victim;
- seeds;
- optimization representation.

If CAPGD-PrimSupport achieves a higher Valid ASR than PrimAttack, DO NOT frame this as a failed experiment.

Interpret it accurately:

> direct constrained optimization can locate valid adversarial feature vectors that are not necessarily reachable through PrimAttack's lower-dimensional primitive mapping.

However, do NOT automatically claim every such CAPGD solution is physically unrealizable.

The evidence only supports that PrimAttack enforces additional modeled primitive structure.

Explicitly distinguish:

```text
valid feature-space point
vs
PrimAttack-reachable point
vs
packet-realizable point
```

Include:

> **[TABLE — CAPGD-PrimSupport vs PrimAttack paired comparison]**

---

# 5.4 Raw vs Constraint-Valid Attack Success

Quantify the gap between raw and valid success.

For every method show:

\[
\Delta ASR = ASR_{raw} - ASR_{valid}
\]

where useful.

Explain which attacks lose most of their apparent success under validation.

Break failures down by validator rule family if artifacts permit.

Potential:

> **[FIGURE — Raw ASR vs Valid ASR]**

> **[TABLE — Invalid successful examples by validator failure category]**

Do not claim invalid feature examples are impossible network traffic unless the specific rule justifies that conclusion.

---

# 5.5 PrimAttack Primitive Ablation

Report:

```text
timing-only
padding-only
joint
```

using seeds 42, 2024, 2026 if this is the locked protocol.

Report both:

- ASR over all attempted source flows;
- where meaningful, conditional performance among flows eligible for a primitive.

Also report:

```text
% timing eligible
% padding eligible
% joint eligible
% neither eligible
```

by dataset/class/victim as appropriate.

This eligibility context is essential after the new padding rule.

Include:

> **[TABLE — Primitive eligibility by dataset and attack class]**

> **[TABLE — Timing vs padding vs joint ablation]**

> **[FIGURE — Valid ASR by primitive mode]**

Explain whether joint optimization provides additional success beyond timing alone.

---

# 5.6 PrimAttack Budget Analysis

Report each locked primitive budget.

For each show:

- Raw ASR;
- Valid ASR;
- perturbation magnitude;
- primitive cost;
- targeted results where applicable.

Explain the trade-off:

```text
larger attacker capability/budget
↔
attack effectiveness
```

Do not assume monotonicity; report what happened.

Include:

> **[TABLE — Budget sweep results]**

> **[FIGURE — Valid ASR versus primitive budget]**

> **[FIGURE — Attack success versus perturbation cost]**

if available.

---

# 5.7 PrimAttack Optimizer / Search Ablation

If final experiments compare:

- canonical PrimAttack optimizer;
- Prim-PGD;
- random search;
- other underlying optimizer;

write the full section.

Explain whether the primitive formulation itself or the chosen optimizer drives the result.

Use identical source sets/budgets where applicable.

Include:

> **[TABLE — Optimizer ablation]**

> **[FIGURE — Valid ASR across PrimAttack optimizers]**

If this experiment was dropped, OMIT the section rather than inventing content.

---

# 5.8 Capability-Aware Padding Analysis

This is now an important section.

Explain the discovered issue in neutral academic language.

Do NOT call it a catastrophic bug.

Explain:

The previous capability test allowed padding whenever the flow contained forward payload, while uniform padding increased all modeled forward packet lengths.

For flows satisfying:

```text
Fwd Packet Length Min = 0
```

this implied increasing at least one originally empty forward packet.

Because aggregate flow statistics do not identify individual packet roles/content, the revised capability-aware method conservatively disables uniform padding on these flows.

Report:

- fraction of flows with `Fwd Packet Length Min = 0`;
- fraction originally padding-eligible;
- fraction newly padding-ineligible;
- breakdown by dataset;
- breakdown by attack class/source label;
- effect on PrimAttack;
- effect on baseline valid results.

Use the actual final numbers.

Include:

> **[TABLE — Empty-forward-packet prevalence by class]**

> **[TABLE — Padding capability before vs after revised rule]**

> **[TABLE — Relaxed PrimAttack vs capability-aware PrimAttack]**

> **[FIGURE — Effect of capability restriction on Valid ASR]**

The old relaxed result may be retained as a sensitivity analysis, but it must be clearly labeled as superseded/non-canonical.

Explain that a fresh rerun is necessary because simply deleting old padding-dependent successes underestimates what a timing-focused optimizer can recover.

Quantify recovered timing-based successes from the rerun.

---

# 5.9 Statistical Evaluation

Report final statistical tests.

For each planned comparison provide:

- hypothesis;
- N;
- test;
- statistic;
- p-value;
- effect size if retained;
- descriptive values;
- interpretation.

Do NOT say statistical significance proves practical importance.

Do NOT use p-values alone.

Include:

> **[TABLE — Statistical test results]**

If tests compare primitive modes or budget levels, organize them clearly rather than dumping all pairwise tests.

---

# 5.10 Cross-Dataset Analysis

Compare CICIDS2017 and CICIDS2018 carefully.

Discuss:

- victim behavior;
- baseline attacks;
- Valid ASR;
- validator effects;
- PrimAttack;
- primitive eligibility;
- padding availability;
- timing dependence.

Do not treat raw percentage differences as automatically generalizable.

Explain differences in dataset composition/extractor characteristics where supported.

The existing CICIDS2018 empty-packet validation behavior should be discussed where relevant.

Include:

> **[TABLE — Cross-dataset summary]**

Potential:

> **[FIGURE — Valid ASR by method across datasets]**

---

# 5.11 Synthesis and Discussion

Synthesize results around the research questions instead of repeating tables.

Organize around findings such as:

### 1. Effect of validity enforcement
How much raw ASR disappears after domain constraints?

### 2. Effect of constrained feature support
How do native feature-space attacks compare with matched-support attacks?

### 3. Effect of primitive-domain restriction
What changes when downstream feature movements must arise from PrimAttack's primitive mapping?

### 4. Primitive contribution
How much comes from timing vs padding?

### 5. Budget/effectiveness trade-off
How does attack capability affect success?

### 6. Dataset dependence
Are findings stable across CICIDS2017 and CICIDS2018?

Do not force the results to prove PrimAttack is "better".

If CAPGD achieves higher Valid ASR, explain accurately.

The central contribution may instead be that increasingly restrictive attack representations yield materially different estimates of adversarial vulnerability.

---

# 5.12 Limitations

Write a substantial limitations section.

Include all limitations actually relevant to the final design, including:

### Flow-level rather than packet-level realization
No direct packet modification/replay/re-extraction.

### Aggregate-information limitation
The flow vector does not preserve individual packet identities/order/content sufficiently for exact inverse mapping.

### Conservative padding capability
`Fwd Packet Length Min > 0` may exclude some genuinely modifiable traffic because it sacrifices opportunity for defensibility.

### Primitive coverage
Timing and padding represent only a subset of all possible attacker actions.

### White-box assumption
Requires information/gradient access stronger than many practical attackers possess.

### Feature extractor dependence
PrimAttack's recomputation rules depend on the extractor/feature definition.

### Validator completeness
Passing validation does not prove true protocol/application functionality.

### Dataset limitations
Public benchmark traffic differs from operational networks.

### Statistical/sample limitations
Whatever applies from the final protocol.

### Computational constraints
If relevant.

Do not hide negative findings.

---

# REQUIRED TABLE/FIGURE INVENTORY

At the very end of the Markdown file create:

```markdown
# Proposed Figure and Table Inventory
```

Make two master tables.

## Figure inventory

Columns:

```text
ID
Chapter/Section
Suggested caption
Purpose
Existing artifact path
Needs generation?
Data source
Priority (Essential / Useful / Optional)
```

## Table inventory

Same structure.

At minimum consider:

### Methodology figures
- overall thesis pipeline;
- preprocessing pipeline;
- validator pipeline;
- primitive-domain mapping;
- detailed PrimAttack pipeline;
- experimental design.

### Methodology tables
- dataset overview;
- label mappings;
- split distributions;
- feature groups;
- victim configurations;
- baseline configurations;
- validator rule categories;
- primitive-feature mappings;
- capability rules;
- budgets;
- experiment matrix;
- statistical tests.

### Results figures
- victim performance where useful;
- raw vs valid ASR;
- Experiment A;
- primitive ablation;
- budget sweep;
- optimizer ablation;
- capability-aware padding impact;
- cross-dataset comparison.

### Results tables
- final victim metrics;
- main Experiment A;
- CAPGD-PrimSupport vs PrimAttack;
- raw vs valid;
- primitive ablation;
- eligibility;
- budgets;
- optimizer comparison;
- empty-packet/capability results;
- statistical tests;
- cross-dataset summary.

---

# RESULT STATUS AUDIT

At the end also create:

```markdown
# Experiment Completion and Rerun Audit
```

Construct a table:

```text
Experiment
Dataset
Method
Current artifact
Valid under final methodology?
Rerun required?
Reason
Expected output path
```

Inspect all experiments.

In particular, because PrimAttack's padding capability has changed:

- identify every PrimAttack experiment generated under the old padding behavior;
- mark it for rerun;
- identify statistical tests depending on those outputs;
- identify plots/tables depending on those outputs;
- determine whether baseline adversarial artifacts can simply be revalidated or require regeneration;
- inspect whether CAPGD-PrimSupport's attack mask depends dynamically on PrimAttack capability inference.

Do not automatically rerun anything as part of THIS task unless necessary to inspect artifacts.

This task is to write the thesis draft and accurately flag missing/outdated results.

---

# SOURCE TRACEABILITY

For every subsection add an HTML comment at the end containing the primary repository evidence, for example:

```html
<!--
Repository evidence:
- src/attack/realizability/cicids2017.py
- scripts/run_experiment_a.py
- results/experiment_a/summary.csv
-->
```

These comments are for my drafting process and can later be removed before LaTeX conversion.

Do not clutter the visible prose with file paths unless necessary.

---

# PLACEHOLDER POLICY

Use clear placeholders when something is missing.

Examples:

```markdown
> **[TABLE 5.X PENDING — Final capability-aware Experiment A results]**
> Requires rerun of PrimAttack for seeds 42, 2024, and 2026.
> Do not use the pre-capability-fix result here.
```

or:

```markdown
> **[VALUE PENDING: final mean ± SD Valid ASR after rerun]**
```

Never invent values.

Never silently use a single-seed result where the final experiment requires three seeds.

Never substitute an obsolete run for the final result.

---

# EQUATIONS

Use proper Markdown/LaTeX equations.

For every important equation:

1. define all symbols;
2. state where the quantity comes from;
3. explain its purpose in plain language afterward.

This is especially important for:

- ASR;
- Valid ASR;
- attack objective;
- padding transformation;
- timing transformation;
- capability bounds;
- primitive budgets;
- statistical tests/effect sizes.

All PrimAttack equations must be derived from actual implementation.

---

# FINAL QUALITY AUDIT

Before saving the Markdown, perform a final consistency check.

Confirm:

```text
[ ] Only methodology and results/evaluation chapters are written.
[ ] Every final attack used in the thesis is documented.
[ ] Every final experiment has a methodology subsection.
[ ] Every result has a corresponding methodology description.
[ ] No obsolete PrimAttack result is presented as canonical.
[ ] Empty-forward-packet capability fix is represented correctly.
[ ] Timing-only behavior is documented.
[ ] Validator validity is not confused with primitive reachability.
[ ] PrimAttack is not claimed to prove packet-level realization.
[ ] CAPGD-PrimSupport comparison is explained fairly.
[ ] Seeds are consistent everywhere.
[ ] Dataset splits are consistent everywhere.
[ ] Metrics use the correct denominators.
[ ] Statistical tests match current code.
[ ] Mean ± SD values come from actual artifacts.
[ ] All figures have placeholders or actual artifact paths.
[ ] All tables have placeholders or actual artifact paths.
[ ] Missing experiments are clearly marked.
[ ] All technical equations match executable code.
[ ] All result numbers are traceable to files.
[ ] Limitations are explicit.
```

Then write:

```text
thesis_methodology_results.md
```

Do not merely give me a summary of what you inspected.

The main deliverable is the full thesis-ready Markdown draft.