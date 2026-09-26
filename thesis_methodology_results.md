# Chapter 4 — Proposed Methodology

## 4.1 Methodology Overview

This study evaluates how estimates of adversarial vulnerability change when progressively stronger restrictions are placed on attacks against flow-based network-intrusion-detection systems (NIDS). The central question is: **How does measured attack success change when classifier evasion is separated from domain validity, constrained feature support, source-conditioned primitive capability, and primitive-domain reachability?** The final study uses two corrected DistriNet releases, three neural victim architectures, unconstrained and constrained feature-space baselines, the proposed PrimAttack method, an independent rule-based validator, paired source flows, and paired statistical tests.

The pipeline begins with corrected CICFlowMeter aggregate-flow records. Each dataset is cleaned and split chronologically within source labels before any fitted statistic is computed. A `RobustScaler` is fitted on training data only. Category victims are trained on the resulting 79-feature representation. Attacks are then run only on malicious test flows that the relevant victim classifies correctly. One canonical list of 800 flows per dataset, victim, and malicious class is frozen and reused across all compared methods and all attack seeds.

The evaluation deliberately separates four questions. First, does an adversarial vector change the classifier decision? Second, does the vector satisfy the implemented schema, extractor, protocol, and train-mined constraints? Third, is the change available under the source flow's primitive capabilities? Fourth, can it be reached through PrimAttack's low-dimensional primitive-to-feature mapping? Passing these stages does not establish packet-level realizability, because no packet trace is edited, replayed, or re-extracted.

> **[FIGURE 4.1 PLACEHOLDER — Complete research pipeline]**
> Suggested content: Corrected DistriNet CSVs → cleaning and chronological within-label split → train-only scaling → victim training → canonical clean-correct sample selection → feature-space and primitive-domain attacks → validator_v2 → Raw ASR and Valid ASR → paired statistics.
> Suggested source/artifact: `scripts/run_final_suite.py`, `scripts/analyze_final_suite.py`, `FINAL_OUTPUTS/00_PROTOCOL.md`.
> Purpose: Gives the reader a single overview of the experimental workflow and its leakage boundary.

<!--
Repository evidence:
- FINAL_OUTPUTS/00_PROTOCOL.md
- FINAL_OUTPUTS/runs/final_suite_config.json
- scripts/run_final_suite.py
- scripts/analyze_final_suite.py
- master_experiments.md
-->

## 4.2 Research and Experimental Design

The final evaluation is a white-box, offline, flow-level experiment. The adversary knows the frozen victim, preprocessing transformation, feature definitions, PrimAttack mapping, train-calibrated budgets, and validator rules. Gradients through the victim and through differentiable attack representations are available. The adversary starts from a malicious test flow belonging to DoS, DDoS, Recon, or BruteForce. Benign flows are not attack sources.

The unit of analysis is one clean-correct source flow under one frozen victim. For each dataset–victim–class combination, a seed-42 uniform permutation of all class test rows is created. The victim takes the first 800 correctly classified rows and the chosen positions are restored to test order. This produces 3,200 attempted flows per dataset–victim cell. A failed attack, invalid output, stalled optimizer, or flow with no primitive headroom remains in the denominator as a failure. The same ordered sample identifiers, raw-input hash, clean predictions, and checkpoint hash are verified for every compared condition.

Attack seeds 42, 2024, and 2026 control attack initialization and restarts. They are not victim-training seeds. CICIDS2017 uses one frozen checkpoint per architecture. CICIDS2018 has three trained replicates in the repository, but the final attack suite intentionally uses only the training-seed-42 MLP, CNN, and FT-Transformer checkpoints. Mean and sample standard deviation across attack seeds therefore describe optimization variability, not model-training uncertainty.

Two objectives are retained. Untargeted evasion succeeds when the adversarial prediction differs from the true malicious class. Targeted evasion succeeds only when the prediction is Benign. Both are evaluated on the same source flows, but they answer different questions: an untargeted DDoS-to-DoS transition is classifier evasion, whereas it is not targeted evasion to Benign.

The following concepts remain distinct throughout the study:

| Concept | Question answered | What it does not establish |
|---|---|---|
| Classifier success | Did the prediction meet the targeted or untargeted objective? | Domain consistency |
| Domain validity | Did validator_v2 accept the aggregate vector? | Primitive reachability or packet realization |
| Feature support | Which downstream coordinates may a method modify? | That coordinates are independently feasible |
| Primitive capability | Is padding or timing available for this source flow? | That every bounded control has a packet trace |
| Primitive reachability | Can the vector be generated by the implemented mapping $\phi$? | PCAP validity or preserved application behavior |
| Packet-level realization | Can packets be edited, replayed, and re-extracted consistently? | Not evaluated in this thesis |

> **[TABLE 4.1 PLACEHOLDER — Paired experimental design]**
> Suggested columns: Dataset, victim, source class, clean-correct flows/class, attack seeds, pairing key, objective.
> Suggested source/artifact: `FINAL_OUTPUTS/00_PROTOCOL.md`, `FINAL_OUTPUTS/runs/*/baselines_untargeted/selection.json`.
> Purpose: Makes the observational unit and repeated-measures design explicit.

<!--
Repository evidence:
- FINAL_OUTPUTS/00_PROTOCOL.md
- FINAL_OUTPUTS/runs/cicids2017_distrinet/baselines_untargeted/config.json
- FINAL_OUTPUTS/runs/cicids2018_distrinet/baselines_untargeted/config.json
- scripts/analyze_final_suite.py
-->

## 4.3 Datasets

### 4.3.1 CICIDS2017-DistriNet

The primary dataset is the corrected and relabelled five-file DistriNet CIC-IDS-2017 release. It contains CICFlowMeter aggregate statistics collected over five working-day files. After unsupported labels, invalid rows, and exact duplicates are removed, 2,080,379 flows remain. The model input contains 79 ordered numerical features. Metadata such as flow identifiers, endpoints, timestamps, labels, and source-file positions is retained separately and excluded from the classifier input.

Nine retained source labels are grouped into five closed-set categories. Four DoS labels are merged into DoS, the two Patator labels into BruteForce, PortScan into Recon, DDoS remains DDoS, and BENIGN remains Benign. A binary label is also produced, but the final adversarial suite attacks the five-class category heads.

<!--
Repository evidence:
- data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json
- docs/full_thesis_methodology/01_preprocessing_cicids2017_distrinet.md
-->

### 4.3.2 CSE-CIC-IDS-2018-DistriNet

The secondary dataset is the corrected DistriNet CSE-CIC-IDS-2018 release. The pipeline reads ten files containing 63,195,145 raw rows and 91 columns. It preserves the same 79-feature order used for CICIDS2017 and excludes dataset-specific ICMP and TCP-flow-time fields. After cleaning and global feature-plus-category deduplication, 62,393,756 rows remain before controlled class-size reduction.

CICIDS2018 is reduced only after chronological splitting. Benign, DoS, and DDoS are sampled to fixed totals of 250,000, 200,000, and 200,000, apportioned 70/15/15. Recon and BruteForce retain all rows. Consequently, each final split is approximately 30% Benign, 24% DoS, 24% DDoS, 11% Recon, and 11% BruteForce. This is a controlled evaluation population, not the natural CICIDS2018 prevalence.

A release-specific issue affects `Fwd Header Length` and `Bwd Header Length`: values can wrap in signed 16-bit storage on long flows. The pipeline retains and flags these values because dropping negative rows would remove traffic selectively without fixing positive wrapped values. No correction is attempted because the number of wraps is unknown.

| Dataset | Raw corrected inputs | Final modelling features | Final classes | Final rows used after preprocessing | Role |
|---|---:|---:|---|---:|---|
| CICIDS2017-DistriNet | 5 CSV files; 2,096,133 retained before deduplication | 79 | Benign, DoS, DDoS, Recon, BruteForce | 2,080,379 | Primary dataset |
| CSE-CIC-IDS-2018-DistriNet | 10 CSV files; 63,195,145 raw rows | 79 | Benign, DoS, DDoS, Recon, BruteForce | 833,552 after controlled reduction | Secondary dataset |

> **[TABLE 4.2 PLACEHOLDER — Final source-label mapping]**
> Suggested content: the mapping below, expanded with dropped labels and Attempted-label handling.
> Suggested source/artifact: both preprocessing manifests and preprocessing reports.
> Purpose: Documents how heterogeneous attack names become the common five-class label space.

| Dataset | Final class | Retained source labels |
|---|---|---|
| 2017 | Benign | BENIGN; labels ending in ` - Attempted` under the configured benign policy |
| 2017 | DoS | DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest |
| 2017 | DDoS | DDoS |
| 2017 | Recon | PortScan |
| 2017 | BruteForce | FTP-Patator, SSH-Patator |
| 2018 | Benign | BENIGN and retained Attempted variants |
| 2018 | DoS | DoS Hulk, DoS GoldenEye, DoS Slowloris |
| 2018 | DDoS | DDoS-HOIC, DDoS-LOIC-HTTP, DDoS-LOIC-UDP |
| 2018 | Recon | Infiltration - NMAP Portscan |
| 2018 | BruteForce | SSH-BruteForce |

<!--
Repository evidence:
- data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json
- data/processed/CSECICIDS_2018_Distrinet/preprocessing_manifest.json
- docs/full_thesis_methodology/01_preprocessing_cicids2017_distrinet.md
- docs/cicids2018distrinet/cicids2018distrinet_preprocessing.md
-->

## 4.4 Dataset Preprocessing

### 4.4.1 Shared leakage boundary and feature contract

Both pipelines enforce the same ordering principle: row-local cleaning, label mapping, float32 canonicalization, deduplication, and split assignment occur before any fitted statistic. The first fitted transformation is `RobustScaler`, trained only on the final training matrix. Its median and interquartile range are then frozen for validation and test data. Class weights, validator profiles, mined rules, train min–max attack boxes, and PrimAttack budgets are also training-only. Validation is used for model or attack selection; test is used for final reporting.

The 79 feature names and their order are load-bearing. The CICIDS2017 preprocessing manifest is the primary source, and the CICIDS2018 pipeline verifies that its shared features occur in the same relative order. Pristine raw arrays and scaled arrays are both persisted. Models consume scaled arrays and apply a differentiable `asinh` transformation internally. Attacks that operate in raw feature space use the same frozen scaler when querying victims.

> **[FIGURE 4.2 PLACEHOLDER — Dataset preprocessing pipeline]**
> Suggested content: CSV validation → row-local cleaning → label mapping → float32 canonicalization → global deduplication → chronological within-source-label split → optional within-split class reduction for 2018 → train-only RobustScaler → arrays, Parquet, audits, and manifests.
> Suggested source/artifact: `src/preprocessing/preprocess_cicids2017_distrinet.py`, `src/preprocessing/preprocess_cicids2018_distrinet.py`.
> Purpose: Shows processing order and the point at which fitted statistics begin.

<!--
Repository evidence:
- src/preprocessing/preprocess_cicids2017_distrinet.py
- src/preprocessing/preprocess_cicids2018_distrinet.py
- data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json
- data/processed/CSECICIDS_2018_Distrinet/preprocessing_manifest.json
-->

### 4.4.2 CICIDS2017 preprocessing

The CICIDS2017 pipeline requires exactly the five expected files and identical normalized headers. Feature values are coerced to numeric form. A row is removed if any of the 79 features is non-finite, if its timestamp cannot be parsed, if any physical feature is negative, or if its mapped category is unsupported. No imputation, clipping, winsorization, or sampling-based balancing is used.

Rows are converted to float32 before duplicate detection so equality matches the representation used by the models. Exact duplicates are defined by all 79 features plus the final category; the earliest chronological occurrence is retained. This removes 15,754 rows, or 0.752% of the cleaned population.

The remaining rows are globally ordered by timestamp, with source-day order and original row number as deterministic tie-breakers. For each retained source label independently, the first 70% becomes training, the next 15% validation, and the last 15% test. Largest-remainder allocation and a minimum of one row per split preserve every source label. Thus, training precedes validation and test within each source label. This is not a global forward-time split because different source labels occupy overlapping wall-clock periods.

The leakage audit asserts disjoint sample identifiers, no feature-plus-category overlap between splits, exhaustive split membership, and chronological ordering within every source label. Feature-only duplicates with different labels may remain by design. The final CICIDS2017 split is strongly imbalanced; BruteForce has only 4,862 training rows and receives a large training weight.

<!--
Repository evidence:
- src/preprocessing/preprocess_cicids2017_distrinet.py
- data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json
- data/processed/CICIDS_2017_Distrinet/leakage_audit.json
- data/processed/CICIDS_2017_Distrinet/duplicate_audit.json
-->

### 4.4.3 CICIDS2018 preprocessing

CICIDS2018 is processed in two streaming passes. The first pass stores compact provenance, timestamps, labels, header flags, and feature hashes. Cleaning, deduplication, splitting, and sample selection operate on those keys. The second pass rereads the CSVs and writes only selected rows, recomputing hashes to detect pass-to-pass drift.

Fifty-seven non-finite rows are removed; all are Benign zero-duration flows with infinite rate features. Another 143,493 rows belong to unsupported final classes. Features are converted to float32 and negative zero is folded into positive zero before hashing. Global exact deduplication on the 79-feature vector plus category removes 657,839 rows.

Chronological 70/15/15 membership is assigned independently within each of nine retained source labels. Only afterward are Benign, DoS, and DDoS reduced inside each split. Sampling is uniform without replacement within strata defined by source label, source file, and UTC hour. Hamilton allocation preserves proportional strata, and selected rows are restored to chronological order. This operation never moves a flow across splits and never synthesizes or duplicates a row.

| Dataset | Split | Benign | DoS | DDoS | Recon | BruteForce | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| CICIDS2017 | Train | 1,153,431 | 120,093 | 66,568 | 111,311 | 4,862 | 1,456,265 |
| CICIDS2017 | Validation | 247,164 | 25,733 | 14,265 | 23,853 | 1,043 | 312,058 |
| CICIDS2017 | Test | 247,164 | 25,733 | 14,265 | 23,852 | 1,042 | 312,056 |
| CICIDS2018 | Train | 175,000 | 140,000 | 140,000 | 62,549 | 65,938 | 583,487 |
| CICIDS2018 | Validation | 37,500 | 30,000 | 30,000 | 13,403 | 14,130 | 125,033 |
| CICIDS2018 | Test | 37,500 | 30,000 | 30,000 | 13,403 | 14,129 | 125,032 |

> **[TABLE 4.3 PLACEHOLDER — Final feature set and inferred feature categories]**
> Suggested columns: Index, feature name, inferred schema type, PrimAttack padding support, PrimAttack timing support, frozen/derived role.
> Suggested source/artifact: preprocessing manifests, validator schema profiles, `primattack_feature_support`.
> Purpose: Records the exact model input order and connects it to attack semantics.

<!--
Repository evidence:
- src/preprocessing/preprocess_cicids2017_distrinet.py
- src/preprocessing/preprocess_cicids2018_distrinet.py
- data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json
- data/processed/CSECICIDS_2018_Distrinet/preprocessing_manifest.json
- docs/cicids2018distrinet/cicids2018distrinet_preprocessing.md
-->

## 4.5 Victim NIDS Models

Three differentiable five-class neural architectures are used to reduce architecture-specific conclusions. All receive 79 RobustScaler-space features and apply $\operatorname{asinh}(x)$ internally. Their output is a vector of five logits in the order Benign, DoS, DDoS, Recon, and BruteForce.

The SimpleMLP contains hidden layers of 256, 128, and 64 units. Every hidden layer uses a linear transformation, ReLU, and dropout 0.3. The CNN treats the fixed-order feature vector as a one-dimensional signal: two same-padded convolutions, 1→32 and 32→64 channels with kernel size 3, are followed by ReLU, eight-bin adaptive max pooling, a 64-unit fully connected layer, ReLU, dropout 0.3, and the output layer.

The FT-Transformer tokenizes every scalar feature independently as $T_j=b_j+x_jW_j$ with token width 192, appends a learned classification token, and applies three pre-normalized Transformer blocks. Each block uses eight-head self-attention with dropout 0.2 and a ReGLU feed-forward layer of width 256 with dropout 0.1. Residual dropout is zero. The final head is LayerNorm, ReLU, and a linear layer applied only to the classification token.

Training uses balanced, training-only inverse-frequency weights in cross-entropy loss. MLP and CNN use Adam with learning rate $10^{-3}$ and no weight decay. FT-Transformer uses AdamW with learning rate $10^{-4}$ and weight decay $10^{-5}$, excluding tokenizer, classification token, bias, and LayerNorm parameters from decay. Batch size is 2,048; gradients are clipped to norm 5. `ReduceLROnPlateau` monitors validation macro-F1 with factor 0.5 and patience 1. Runs request ten epochs and early-stop with patience 3. The best validation macro-F1, with validation loss as a tie-break, selects the checkpoint; test data is evaluated only afterward.

| Model | Main architecture | Dropout | Optimizer | Learning rate | Parameters, 2017 category head |
|---|---|---:|---|---:|---:|
| SimpleMLP | 79→256→128→64→5 | 0.3 | Adam | $10^{-3}$ | 61,957 |
| CNNOnly | Conv1d 32/64, adaptive pool 8, FC 64 | 0.3 | Adam | $10^{-3}$ | 39,493 |
| FT-Transformer | 79 tokens + CLS; 3 blocks; width 192; 8 heads | 0.2 attention; 0.1 FFN | AdamW | $10^{-4}$ | 922,949 |

> **[FIGURE 4.3 PLACEHOLDER — Simplified victim architectures]**
> Suggested content: compact diagrams for MLP, CNN, and FT-Transformer, emphasizing that each consumes one independent 79-feature flow vector.
> Suggested source/artifact: `src/classifiers/models.py`, `src/classifiers/ft_transformer.py`.
> Purpose: Clarifies architectural diversity without over-emphasizing standard classifier design.

<!--
Repository evidence:
- src/classifiers/models.py
- src/classifiers/ft_transformer.py
- src/classifiers/cicids2017d_experiments.py
- outputs/cicids2017distrinet/classifier_run_manifest.json
- outputs/cicids2017distrinet_ft/classifier_run_manifest.json
- outputs/cicids2018distrinet/classifiers_multiseed/multiseed_manifest.json
-->

## 4.6 Adversarial Threat Model

The evaluated adversary has white-box access to one frozen category classifier, its input transformation, and gradients. Feature-space baselines directly optimize scaled or train-min–max-normalized feature vectors. PrimAttack additionally knows the primitive mapping, training-derived envelopes, class-calibrated budgets, and validator. PrimAttack uses validator acceptance in its search predicate; the baselines do not, except that C-PGD includes a differentiable subset of relations in its objective. This asymmetry is part of the methods' native threat models and is not presented as equal query access.

The source is an existing malicious flow. Allowed PrimAttack actions are increase-only forward packet-length augmentation and added forward timing delay. It cannot shorten packets, accelerate traffic, add or remove packets, change backward traffic, ports, protocol, endpoints, labels, flags, or arbitrary extracted features. A targeted attack seeks Benign; an untargeted attack seeks any class other than the source class.

Four representation levels are distinguished. Packets are concrete events with order, headers, and payload. A flow is an aggregate connection record. Feature space is the 79-dimensional CICFlowMeter vector. The primitive domain is the low-dimensional control vector from which PrimAttack deterministically recomputes downstream features. The experiment operates on flow aggregates and feature vectors. Packet construction, replay, target-state observation, and CICFlowMeter re-extraction are outside scope.

<!--
Repository evidence:
- FINAL_OUTPUTS/00_PROTOCOL.md
- src/attack/realizability/base.py
- src/attack/realizability/cicids2017.py
- src/attack/primitive_optimizer.py
-->

## 4.7 Evaluation Framework

Let $\mathcal{E}$ be the frozen set of attempted clean-correct malicious flows for one dataset, victim, and reported scope, and let $N=|\mathcal{E}|$. For flow $i$, let $y_i$ be its true malicious class, $\hat y_i^{adv}$ the adversarial prediction, and $V_i$ the validator_v2 `hybrid_valid` verdict.

Untargeted and targeted raw success indicators are

\[
U_i=\mathbf{1}[\hat y_i^{adv}\neq y_i],
\qquad
T_i=\mathbf{1}[\hat y_i^{adv}=\mathrm{Benign}].
\]

The corresponding raw attack success rates are

\[
\mathrm{ASR}^{unt}_{raw}=\frac{1}{N}\sum_{i\in\mathcal{E}}U_i,
\qquad
\mathrm{ASR}^{tar}_{raw}=\frac{1}{N}\sum_{i\in\mathcal{E}}T_i.
\]

For whichever objective is active, write its raw indicator as $S_i$. Valid success is $S_iV_i$, and

\[
\mathrm{ASR}_{valid}=\frac{1}{N}\sum_{i\in\mathcal{E}}S_iV_i.
\]

The denominator does not change when an attack fails or the validator rejects its output. The validity gap is

\[
\Delta_{valid}=100\left(\mathrm{ASR}_{raw}-\mathrm{ASR}_{valid}\right)
\]

in percentage points. Validator pass rate, reported separately, is $N^{-1}\sum_i V_i$ and must not be confused with Valid ASR because a valid vector need not fool the victim.

Perturbation reporting follows each method's native representation. Input PGD uses $L_\infty$ in RobustScaler space. Input C&W reports $L_2$ in RobustScaler space. CAPGD and C-PGD use a training-fitted min–max space. PrimAttack reports the integer controls, changed-feature support, added delay, and normalized primitive cost

\[
C_i=\frac{p_i}{p_{hi,i}}+\frac{d_i}{d_{hi,i}},
\]

with an inactive term defined as zero. Shape has no direct cost because it redistributes a fixed total delay.

| Metric | Numerator | Denominator | Interpretation |
|---|---|---|---|
| Raw ASR | Objective successes | Attempted clean-correct flows | Classifier-level attack success |
| Valid ASR | Objective success ∧ `hybrid_valid` | Same attempted flows | Classifier success that also passes implemented domain rules |
| Validity gap | Raw ASR − Valid ASR | Percentage-point difference | Apparent success removed by validation |
| Validator pass rate | Validator-accepted outputs | Attempted flows | Output validity independent of attack success |
| Normalized primitive cost | $p/p_{hi}+d/d_{hi}$ | Per successful flow | Fraction of available primitive box used |

<!--
Repository evidence:
- FINAL_OUTPUTS/00_PROTOCOL.md
- scripts/analyze_final_suite.py
- validation/metrics.py
- src/attack/primitive_optimizer.py
- master_experiments.md
-->

## 4.8 Domain Validity Framework

Validator_v2 is an independent, data-driven rule engine over pristine 79-feature vectors. A rule is stored as data with a stable identifier, template, parameters, tolerance, provenance, and evidence. Evaluation returns a satisfied mask and an eligible mask. A sample violates a rule only when the rule is eligible and unsatisfied. Conditional rules therefore do not reject rows for which their antecedent does not apply.

The headline verdict is

\[
V=V_{schema}\land V_{extractor}\land V_{protocol}\land V_{mined}.
\]

Distributional plausibility is implemented separately and is not included in this verdict or in the final headline metrics. The final no-IDR policy likewise excludes the earlier VAE IDR and True-IDSR measures.

### 4.8.1 SCHEMA rules

SCHEMA rules are synthesized from a profile inferred on training data. Every feature must be finite. Integer, binary, categorical, or train-constant features receive an additional type rule. Both datasets load 133 SCHEMA rules. CICIDS2017's train-constant URG fields illustrate a limitation: 94 genuine validation flows are rejected because values unseen in training appear later.

<!--
Repository evidence:
- validation/schema/cicids2017_distrinet.yaml
- validation/schema/cicids2018_distrinet.yaml
- validation/validator/engine.py
-->

### 4.8.2 EXTRACTOR rules

Seven extractor identities encode CICFlowMeter algebra, including variance–standard-deviation consistency, equality of average packet size and packet-length mean, equality of directional segment-size and packet-length means, flow packet rate as the sum of directional rates, and total directional bytes as count times mean. Approximate rules use explicit tolerances:

\[
|x_{obs}-x_{exp}|\le a+r|x_{exp}|,
\]

where $a$ and $r$ are serialized absolute and relative tolerances.

<!--
Repository evidence:
- validation/rules/cicids2017_distrinet/extractor_rules.yaml
- validation/rules/cicids2018_distrinet/extractor_rules.yaml
- validation/validator/tolerance.py
-->

### 4.8.3 PROTOCOL rules

The protocol layer applies non-negativity to features for which it is valid in the dataset release. CICIDS2017 loads 79 non-negativity rules plus one source-conditioned transition rule; CICIDS2018 loads 77 non-negativity rules plus the transition rule because wrapped header-length fields can be negative.

The transition rule `PROTO_0080` compares an adversarial flow with its source:

\[
L^{src}_{fwd,min}=0 \Longrightarrow L^{adv}_{fwd,min}=0.
\]

It prevents an attack from turning a source flow's empty forward packet into a positive-length packet. It is ineligible when validating an unperturbed flow without a separate source, so it does not lower genuine-flow acceptance.

<!--
Repository evidence:
- validation/rules/cicids2017_distrinet/protocol_rules.yaml
- validation/rules/cicids2018_distrinet/protocol_rules.yaml
- validation/validator/rule.py
-->

### 4.8.4 MINED rules

Mined rules are discovered on training data and confirmed on validation data. Candidate families include pairwise order/equality, arithmetic identities, monotone chains, and zero implications. Approximate tolerances are derived from the 99.9th percentile of training residuals and capped to avoid vacuous rules. Retained rules require at least 0.999 training support and 0.995 validation support, and test data is not used.

CICIDS2017 retains 16 rules: eight min–mean–max chains, one PSH-flag sum, and seven implications requiring backward totals and timing/rate fields to remain zero when the backward packet count is zero. CICIDS2018 retains ten rules; notably, `MINED_0001` approximately equates `Fwd Packet Length Min` and `Packet Length Min` and causes the small held-out rejection rate reported later.

<!--
Repository evidence:
- validation/rules/cicids2017_distrinet/mined_rules.json
- validation/rules/cicids2018_distrinet/mined_rules.json
- validation/mining/run_mining.py
-->

| Dataset | SCHEMA | EXTRACTOR | PROTOCOL | MINED | Source-conditioned rules |
|---|---:|---:|---:|---:|---:|
| CICIDS2017 | 133 | 7 | 80 | 16 | 1 |
| CICIDS2018 | 133 | 7 | 78 | 10 | 1 |

> **[FIGURE 4.4 PLACEHOLDER — Validator architecture and rule families]**
> Suggested content: schema profile, extractor rules, protocol rules, mined rules → per-rule eligibility/satisfaction → per-layer conjunctions → `hybrid_valid`; show plausibility outside the conjunction.
> Suggested source/artifact: `validation/validator/`, `FINAL_OUTPUTS/F_validator_evaluation/rule_inventory.csv`.
> Purpose: Prevents conflation of structural validity with distributional plausibility or primitive reachability.

Validator validity and PrimAttack capability answer different questions. The validator asks whether a final aggregate vector is consistent with implemented requirements. Capability inference asks whether a source flow supports a modeled operation before search. A vector may pass validator_v2 yet lie outside PrimAttack's reachable set, and neither property proves a packet trace exists.

<!--
Repository evidence:
- validation/validator/engine.py
- validation/validator/rule.py
- validation/validator/result.py
- validation/validator/tolerance.py
- validation/rules/cicids2017_distrinet/
- validation/rules/cicids2018_distrinet/
- FINAL_OUTPUTS/F_validator_evaluation/rule_inventory.csv
-->

## 4.9 Baseline Adversarial Attacks

The final inferential Experiment A contains four baselines; native CAPGD is added as a descriptive fifth baseline row but is not included in the locked inferential family.

**Input PGD** performs untargeted cross-entropy ascent over all 79 RobustScaler-space features. It uses $L_\infty$ radius 0.5, step size 0.05, 40 steps, and one uniform random start. Projection returns the vector to the $L_\infty$ ball only; there is no raw-domain box, type repair, feature mask, or validator in the optimization loop.

**Input C&W** performs untargeted Adam optimization over an additive feature-space perturbation. Its loss is

\[
\lambda\max(z_y-\max_{k\ne y}z_k+\kappa,0)+\|\delta\|_2^2,
\]

with $\lambda=1$, $\kappa=0$, learning rate 0.01, at most 60 iterations, and convergence threshold $10^{-5}$. There is no hard $L_2$ radius. The lowest-$L_2$ successful perturbation found for each sample is retained.

**CAPGD-PrimSupport** uses the frozen TabularBench CAPGD implementation in a train-fitted min–max space. It directly optimizes the 23 coordinates that PrimAttack's recomputation may write, with $L_2$ radius 0.5, ten steps, and two restarts. It projects to the norm ball and train box, freezes all other coordinates, repairs feature types, and uses encoded relation repair. The independent validator is applied after attack generation.

**C-PGD-PrimSupport** uses the same 23-feature mask, $L_2$ radius 0.5, 40 steps, step size 0.05, one random start, and a differentiable objective of cross-entropy minus a relation-violation penalty with weight 1. It repairs integer types and preserves the mask after optimization.

**Native CAPGD**, reported descriptively, uses its own 16-feature configuration mask rather than PrimAttack support. It is useful for showing how threat-model support affects results, but it is not a matched-support or inferential comparison.

| Method | Space/support | Budget | Search | Validator used during search? |
|---|---|---|---|---|
| PGD | 79 RobustScaler features | $L_\infty=0.5$ | 40 steps, $\alpha=0.05$, random start | No |
| C&W | 79 RobustScaler features | Unbounded $L_2$ penalty | ≤60 Adam steps, lr 0.01 | No |
| CAPGD-PrimSupport | 23 train-min–max coordinates | $L_2=0.5$ | 10 steps, 2 restarts | No |
| C-PGD-PrimSupport | Same 23 coordinates | $L_2=0.5$ | 40 steps, step 0.05 | Differentiable subset only |
| CAPGD native † | Native 16-feature mask | $L_2=0.5$ | 10 steps, 2 restarts | No |

### 4.9.1 Matched support is not matched feasibility

The 23-feature mask is generated from the actual write sites of PrimAttack's canonical map. It is static at feature-name level, not sample-specific. CAPGD and C-PGD may move any allowed coordinate directly, subject to their norm, box, type, and relation mechanisms. PrimAttack can alter those potential coordinates only through source-applicable padding or timing and deterministic coupled recomputation. Therefore,

\[
\text{same potential support}\neq\text{same feasible set}.
\]

The comparison controls one important confound—the set of downstream coordinates—but not directionality, coupling, source capability, primitive budgets, or validator-aware selection.

<!--
Repository evidence:
- src/attack/input_baselines.py
- src/comparisons/capgd_cicids2017.py
- src/comparisons/cpgd_prim_support.py
- FINAL_OUTPUTS/runs/final_suite_config.json
- FINAL_OUTPUTS/A_primary_baseline_comparison/primary_baseline_comparison.md
-->

## 4.10 PrimAttack

### 4.10.1 Motivation

Directly changing extracted flow features treats dependent quantities as independent controls. A feature-space optimizer may alter a total without its mean, a rate without its duration, or a variance without its standard deviation. It may also modify features that an attacker cannot directly set. PrimAttack instead optimizes a small set of attacker-facing flow controls and computes affected CICFlowMeter features through a deterministic map. This narrows the attack from arbitrary feature perturbation toward a realizability-oriented flow-level abstraction.

<!--
Repository evidence:
- src/attack/realizability/cicids2017.py
- docs/full_thesis_methodology/02_primattack.md
-->

### 4.10.2 Primitive domain

PrimAttack models two operations with three controls:

\[
z=(p,d,s),
\]

where $p$ is a non-negative integer number of bytes added uniformly to every forward packet, $d$ is a non-negative integer total forward delay in microseconds, and $s\in[0,1]$ allocates the delay between proportional dilation and equal additive delay per forward gap. Both physical operations are increase-only.

Padding does not add packets and is not a payload rewrite model. Timing does not accelerate or reorder packets. Backward traffic and all undeclared features are copied from the source. Projected $p$ and $d$ are rounded and capped by per-flow upper bounds; shape is clamped and forced to zero when delay is zero.

<!--
Repository evidence:
- src/attack/realizability/cicids2017.py
- src/attack/realizability/base.py
-->

### 4.10.3 Primitive-to-feature mapping

The adversarial vector is

\[
x_{adv}=\phi(x_0,p,d,s).
\]

For $N_f$ forward packets, uniform padding recomputes

\[
L'_{fwd}=L_{fwd}+N_fp,
\quad
l'_{fwd,min}=l_{fwd,min}+p,
\quad
l'_{fwd,max}=l_{fwd,max}+p,
\quad
\bar l'_{fwd}=\frac{L'_{fwd}}{\max(N_f,1)}.
\]

`Fwd Segment Size Avg` is set to the new mean. Forward packet-length standard deviation is unchanged because a uniform shift preserves variance. Combined packet-length min and max branch on whether forward and backward directions are present. The combined mean is

\[
\bar l'=\frac{L'_{fwd}+L_{bwd}}{\max(N_f+N_b,1)}.
\]

Let $m_f,m_b$, $s_f,s_b$, and $N_f,N_b$ denote directional means, standard deviations, and counts. The combined sample variance is recomputed by pooled decomposition:

\[
s'^2=\frac{(N_f-1)s_f^2+(N_b-1)s_b^2+N_f(m_f-m)^2+N_b(m_b-m)^2}{\max(N_f+N_b-1,1)},
\]

where $m=(N_fm_f+N_bm_b)/\max(N_f+N_b,1)$. `Packet Length Std` is $\sqrt{s'^2}$ and `Average Packet Size` equals the combined mean.

For timing, let the source contain $m=\max(N_f-1,1)$ forward gaps $g_i$ with total $T_f>0$. PrimAttack defines

\[
a=1+(1-s)\frac{d}{T_f},
\qquad
b=s\frac{d}{m},
\qquad
g'_i=ag_i+b.
\]

Thus $\sum_i g'_i=T_f+d$, with $a\ge1$ and $b\ge0$. The aggregate recomputation is

\[
T'_f=T_f+d,
\quad I'_{max}=aI_{max}+b,
\quad I'_{min}=aI_{min}+b,
\quad I'_{std}=aI_{std},
\quad I'_{mean}=\frac{T'_f}{m}.
\]

Duration is conservatively set to

\[
D'=\max(D+d,T'_f,T_{bwd},1\ \mu s),
\]

and `Flow IAT Mean` is $D'/\max(N_f+N_b-1,1)$. `Flow IAT Max` is increased by the realized duration increase. Rates are recomputed as count or bytes divided by $D'/10^6$ seconds:

\[
R'_{pkt}=\frac{N_f+N_b}{D'/10^6},
\qquad
R'_{byte}=\frac{L'_{fwd}+L_{bwd}}{D'/10^6}.
\]

The mapping writes 23 unique downstream coordinates. Sequence-dependent fields such as bulk, subflow, active/idle, merged Flow-IAT standard deviation/minimum, and `Fwd Act Data Pkts` are held constant and explicitly marked as packet-level limitations rather than claimed invariants.

| Primitive | Direct controls | Main recomputed features | Main invariants/held fields |
|---|---|---|---|
| Padding | $p$ bytes/forward packet | Forward total/min/max/mean, forward segment mean, combined min/max/mean/std/variance, average packet size, flow byte rate | Packet counts, forward length std, backward-only values, flags/ports |
| Timing | $d$ total µs, allocation $s$ | Forward IAT total/mean/std/min/max, duration, flow IAT mean/max, directional and flow rates | Packet lengths, counts, flags/ports; sequence-dependent Level-C fields held |

> **[FIGURE 4.5 PLACEHOLDER — Primitive-to-feature recomputation graph]**
> Suggested content: $p$ and $(d,s)$ nodes feeding the 23 written features; mark exact, conditional, rate, invariant, frozen, and Level-C roles.
> Suggested source/artifact: `src/attack/realizability/cicids2017.py:primattack_feature_support`, `roles`, and `generate`.
> Purpose: Makes the coupled feasible set visible and explains why 23-feature support is not 23 independent controls.

<!--
Repository evidence:
- src/attack/realizability/cicids2017.py
- src/attack/tests/test_primattack_transformation.py
- src/attack/tests/test_primattack_support_mask.py
-->

### 4.10.4 Capability inference

Capability is inferred from each unmodified source flow before optimization. Padding is allowed only when

\[
m_p=[N_f\ge1]\land[L_{fwd}>0]\land[\bar l_{fwd}>0]\land[l_{fwd,min}>0].
\]

The final condition is conservative. If `Fwd Packet Length Min = 0`, at least one forward packet is empty. Aggregate data do not reveal which packet is empty or whether another packet could safely be padded. Because uniform $p$ affects every forward packet, the modeled operation would necessarily make the empty packet positive. Padding is therefore disabled with reason `EMPTY_FWD_PACKET`. The analysis does not claim that the empty packet is definitely a SYN or ACK.

Timing is allowed when

\[
m_t=[N_f\ge2]\land[T_f>0].
\]

After intersecting these capabilities with budget and envelope headroom, every row is classified as joint, timing-only, padding-only, or no-primitive. A coordinate without at least one integer unit of headroom is pinned to zero.

| Effective mode | Condition | Search behavior |
|---|---|---|
| Joint | $p_{hi}\ge1$ and $d_{hi}\ge1$ | Search padding and timing controls |
| Timing-only | $p_{hi}<1$ and $d_{hi}\ge1$ | $p=0$; all available evaluations go to timing |
| Padding-only | $p_{hi}\ge1$ and $d_{hi}<1$ | Search integer padding only |
| No primitive | Both upper bounds below 1 | Return identity candidate |

<!--
Repository evidence:
- src/attack/realizability/cicids2017.py:infer_capabilities
- src/attack/primitive_optimizer.py:row_primitive_modes
- validation/rules/cicids2017_distrinet/protocol_rules.yaml
- validation/rules/cicids2018_distrinet/protocol_rules.yaml
-->

### 4.10.5 Primitive budgets

Budget calibration reads only pristine training features and training labels. For each malicious class, padding uses quantiles of positive forward mean length on capability-evidence rows. Timing uses quantiles of absolute relative deviation from the class median flow duration. Restricted, intermediate, and maximum-evaluated correspond to p25, p50, and p75. The final suite stores p50 and p75, plus an envelope-only condition called `unbounded`; it does not store p25 attack artifacts.

For a source flow, the padding upper bound is

\[
p_{hi}=\max\left(0,\min\left\{
E_{max}-l_{max},
E_{min}-l_{min},
E_{mean}-\bar l,
\frac{E_{total}-L_{fwd}}{\max(N_f,1)},
p_{max}
\right\}\right)m_p,
\]

where each $E$ is a complete-training p99 envelope and $p_{max}$ is the class budget.

The timing bound is the non-negative minimum of the relative-duration cap, train-p99 headroom for forward-IAT total/mean/max/std and flow duration, and, for DoS/DDoS, the remaining delay before flow packet rate would drop below the class-training p05. Worst-case coefficients over all shape values make the returned delay-by-shape box feasible without a soft constraint penalty.

| Dataset | Class | p25: bytes / relative duration | p50 | p75 |
|---|---|---:|---:|---:|
| 2017 | DoS | 41 / 0.0427 | 47 / 0.5671 | 54 / 1.2682 |
| 2017 | DDoS | 2 / 0.2220 | 2 / 0.4353 | 3 / 0.6874 |
| 2017 | Recon | 2 / 0.0851 | 2 / 0.2128 | 10 / 0.5319 |
| 2017 | BruteForce | 11 / 0.0617 | 12 / 0.1199 | 91 / 0.2337 |
| 2018 | DoS | 62 / 0.1220 | 69 / 0.2657 | 74 / 0.5174 |
| 2018 | DDoS | 50 / 0.4132 | 58 / 0.8087 | 63 / 1.4270 |
| 2018 | Recon | 11 / 0.1586 | 22 / 0.9912 | 50 / 0.9912 |
| 2018 | BruteForce | 81 / 0.0170 | 84 / 0.0390 | 85 / 0.0792 |

The envelope-only condition sets class $p_{max}$ and relative-duration caps to infinity but retains train-p99 envelopes, capability gates, integer projection, and DoS/DDoS minimum-rate floors. It is therefore not an unconstrained attack.

<!--
Repository evidence:
- src/attack/primattack_budget.py
- artifacts/primattack/budget_calibration.json
- artifacts/primattack/budget_calibration_cicids2018.json
- src/attack/realizability/cicids2017.py:per_flow_bounds
-->

### 4.10.6 Optimization objective and search methods

For targeted class $t=0$, all PrimAttack optimizers minimize the logit margin

\[
m_t(x)=\max_{k\ne t}z_k(x)-z_t(x).
\]

For untargeted attack from source class $y$, they minimize

\[
m_u(x)=z_y(x)-\max_{k\ne y}z_k(x).
\]

A negative margin indicates the classifier objective, but recorded success additionally requires validator_v2 acceptance on the realized, quantized flow. Candidate selection is lexicographic: a valid success beats a failure; among successes, lower normalized primitive cost wins, then lower margin; among failures, lower margin wins.

The proposed Hybrid Search first scores identity, then enumerates integer padding from one byte to $\lfloor p_{hi}\rfloor$ in increasing cost order. Unresolved timing-capable rows enter adaptive projected sign-momentum refinement. The continuous relaxation supplies gradients, but every retained candidate is projected, quantized, recomputed, and re-evaluated. Step size is halved after stalled checkpoints, and restarts continue until the 256-forward-evaluation cap is consumed.

Prim-PGD is a controlled alternative with three restarts, 42 steps each, step size 0.05, and momentum 0.75. The first restart begins at identity and later restarts are uniform in the normalized box. Prim-C&W uses three 42-step projected-Adam stages, learning rate 0.5, initial $c=1$, and $\kappa=0$ for the objective $C(q)+c\max(m(q)+\kappa,0)$. Prim-PGD and Prim-C&W hyperparameters were selected on CICIDS2017 validation data and transferred unchanged to CICIDS2018.

<!--
Repository evidence:
- src/attack/primitive_optimizer.py
- FINAL_OUTPUTS/runs/cicids2017_distrinet/primattack_targeted_optimizers/config.json
- FINAL_OUTPUTS/B_optimizer_selection/optimizer_selection.json
-->

### 4.10.7 Timing-only optimization

When padding is unavailable but timing is available, the padding upper bound is zero, the normalized padding coordinate is detached and pinned, and random starts and gradient normalization use only free timing coordinates. Hybrid's exact-padding stage costs only the already-scored identity. Prim-PGD and Hybrid then spend the available per-flow evaluation schedule on delay and shape. The total cap remains 256 victim forward evaluations; timing-only behavior reallocates the existing budget rather than increasing it.

<!--
Repository evidence:
- src/attack/primitive_optimizer.py:_mask_q
- src/attack/primitive_optimizer.py:_free_grad_scale
- src/attack/primitive_optimizer.py:optimize_primitive_candidates
- primattack_empty_packet_fix_report.md
-->

### 4.10.8 PrimAttack output and validation

The execution sequence is source flow → capability inference → hard per-flow bounds → selected primitive mode → optimizer → integer projection → deterministic recomputation → victim prediction → validator verdict → Raw/Valid metrics. Artifacts preserve sample IDs, source and adversarial vectors, requested and projected controls, bounds, capability reasons, per-row effective mode, predictions, success masks, validator verdicts, costs, evaluation counts, and checkpoint/input hashes.

> **[FIGURE 4.6 PLACEHOLDER — Detailed PrimAttack pipeline]**
> Suggested content: source $x_0$ → infer $(m_p,m_t)$ → compute $(p_{hi},d_{hi},s_{hi})$ → optimizer in normalized controls → project/round → $\phi$ → frozen scaler/victim and validator → incumbent and metrics.
> Suggested source/artifact: `src/attack/primitive_optimizer.py`, `scripts/run_primattack_optimizer_ablation.py`.
> Purpose: Shows where capabilities, quantization, classifier evaluation, and validity enter the search.

<!--
Repository evidence:
- src/attack/primitive_optimizer.py
- scripts/run_primattack_optimizer_ablation.py
- FINAL_OUTPUTS/runs/*/primattack_*/config.json
-->

### 4.10.9 Realizability scope

PrimAttack is a realizability-oriented flow-level abstraction. Its downstream changes must be generated from modeled padding or timing controls, source-conditioned applicability checks, and deterministic recomputation. This is more restrictive than independent feature optimization. However, packets are not constructed, PCAPs are not edited or replayed, CICFlowMeter is not rerun on modified traffic, and target or application state is not observed. Aggregate features do not uniquely identify packet order, roles, or payload. The method therefore does not establish packet-level realizability or complete preservation of malicious functionality.

<!--
Repository evidence:
- src/attack/realizability/base.py:NullPacketBackend
- src/attack/realizability/cicids2017.py
- docs/full_thesis_methodology/06_semantic_preservation.md
-->

## 4.11 Experimental Protocol

The final suite is generated by `scripts/run_final_suite.py` under the locked protocol. All experiments use both datasets, the three category victims, the four malicious source classes, the canonical 800-flow class rosters, and attack seeds 42, 2024, and 2026 unless stated otherwise.

| Experiment | Purpose | Conditions | Objective / budget | Primary outcome | Planned inference |
|---|---|---|---|---|---|
| A | Main attack comparison | PrimAttack, PGD, C&W, CAPGD-PrimSupport, C-PGD-PrimSupport; native CAPGD descriptive | Untargeted; PrimAttack p75 joint | Valid success | Cochran Q; four PrimAttack-vs-baseline McNemar tests with Holm |
| B | Select PrimAttack optimizer | Hybrid, Prim-PGD, Prim-C&W | Targeted-to-Benign; p75 joint; 256 evaluations | Valid targeted success | Cochran Q; three pairwise McNemar tests with Holm when gated |
| C | Budget sensitivity | p50, p75, envelope-only for top two optimizers | Targeted-to-Benign; joint | Valid targeted success | Cochran Q; p75-vs-p50 and envelope-vs-p75 McNemar with Holm |
| D | Objective sensitivity | Targeted vs untargeted Prim-PGD | p75 joint | Valid success | One McNemar test per dataset–victim |
| E | Paired validity gap | Raw vs Valid outcome of each main adversarial output | A and B conditions | Loss under validator gate | One McNemar test per condition and dataset–victim |
| F | Validator evaluation | General-only and hybrid validator on all genuine val/test flows | No attack | Acceptance/rejection | Descriptive |
| A2 addition | Primitive-mode ablation | Joint, timing-only, padding-only | Untargeted Prim-PGD p75 | Valid success | Descriptive in locked protocol; post-run Cochran/McNemar analysis explicitly labelled exploratory |

The optimizer-selection rule is pre-specified: maximize aggregate Valid Targeted ASR across both datasets, victims, classes, and attack seeds; ties use fewer mean victim evaluations and then fixed order Hybrid, Prim-PGD, Prim-C&W. P-values do not select the optimizer.

**Protocol discrepancy note.** The locked protocol and canonical Experiment C report define the tested family as p50, p75, and envelope-only. A later auxiliary file, `FINAL_OUTPUTS/statistics/statistical_summary.md`, instead requests p25, p50, and p75 and states that this alternative family could not be computed because p25 artifacts are absent. This chapter follows the pre-run locked protocol and `scripts/analyze_final_suite.py`. The p25 condition is reported as not run, not silently substituted. **TODO for final submission:** decide whether the auxiliary `statistics/` wording should be corrected or retained as an explicit post-run alternative analysis request.

> **[FIGURE 4.7 PLACEHOLDER — Final experimental design]**
> Suggested content: shared source roster branching into A–F, with reused cells between B/C/D and the validator recheck path.
> Suggested source/artifact: `FINAL_OUTPUTS/00_PROTOCOL.md`.
> Purpose: Shows which results are independent runs and which reuse identical stored cells.

<!--
Repository evidence:
- FINAL_OUTPUTS/00_PROTOCOL.md
- FINAL_OUTPUTS/statistics/README.md
- FINAL_OUTPUTS/statistics/statistical_summary.md
- scripts/run_final_suite.py
- scripts/analyze_final_suite.py
-->

## 4.12 Statistical Analysis

The inferential unit is one source flow. Classes are concatenated within a victim to $N=3,200$, while datasets and victims are analyzed separately. The pre-specified reference attack seed 42 contributes one binary outcome per flow to inferential tests. Seeds 2024 and 2026 contribute only descriptive mean and sample standard deviation and descriptive paired differences; the three seed means are never treated as $n=3$ independent observations.

For $K\ge3$ paired binary conditions, Cochran's Q tests equal marginal success probabilities. With condition totals $C_j$ and row totals $R_i$,

\[
Q=\frac{(K-1)\left(K\sum_jC_j^2-(\sum_jC_j)^2\right)}{K\sum_iR_i-\sum_iR_i^2},
\]

which is compared with $\chi^2_{K-1}$. Planned McNemar comparisons are performed only when the relevant omnibus Q is significant.

For two paired conditions, let $b$ be A-success/B-failure flows and $c$ A-failure/B-success flows. When $b+c<25$, the exact two-sided binomial McNemar test is used. Otherwise, the continuity-corrected statistic is

\[
\chi^2=\frac{(|b-c|-1)^2}{b+c}.
\]

Holm adjustment is applied only within each predeclared family for one dataset and one victim. The threshold is $\alpha=0.05$. Every interpretation includes the paired rate difference and directional discordant counts; significance is not treated as practical importance or as evidence of generalization.

The primitive-mode inferential analysis in `FINAL_OUTPUTS/statistics/` was added after the locked run and is labelled post-run. The locked protocol had treated the mode ablation descriptively.

| Question | Outcome | Test | Multiplicity |
|---|---|---|---|
| Do ≥3 methods/optimizers/budgets differ? | Per-flow Valid Success | Cochran's Q | Omnibus gate |
| Which planned paired conditions differ? | Per-flow Valid Success | McNemar | Holm within local planned family |
| How much raw success is removed by validity? | Raw Success vs Valid Success on same output | McNemar | No cross-condition correction |
| Does validator accept genuine held-out flows? | Acceptance count/rate | Descriptive | None |

<!--
Repository evidence:
- FINAL_OUTPUTS/00_PROTOCOL.md
- scripts/analyze_final_suite.py
- src/evaluation/paired_validity_gap.py
- FINAL_OUTPUTS/statistics/README.md
-->

## 4.13 Reproducibility and Implementation Environment

The repository environment specifies Python 3.11 and PyTorch 2.5.1+cu121, with NumPy 2.4.4, pandas 3.0.3, scikit-learn 1.9.0, SciPy 1.17.1, PyArrow 24.0.0, and Matplotlib 3.11.0. Final run configurations record Python 3.11.15, PyTorch 2.5.1+cu121, and CUDA execution. Victim-training manifests record Windows, an NVIDIA GeForce RTX 4070 Ti SUPER, deterministic cuDNN settings, and successful checkpoint reloads.

Preprocessing uses seed 42. CICIDS2017 victims use training seed 42; CICIDS2018 provides training seeds 42, 123, and 2024, although only seed-42 victims enter final attacks. Attack seeds are 42, 2024, and 2026. The driver sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`; model training seeds Python, NumPy, CPU/CUDA PyTorch, and deterministic cuDNN behavior.

Reproducibility is anchored by SHA-256 identities. Selection files store ordered sample IDs, positional indices, and clean raw hashes. Attack artifacts store checkpoint and configuration metadata. Before analysis, the canonical analyzer rechecks sample order, duplicates, inputs, labels, predictions, seeds, budgets, objectives, checkpoint hashes, and `valid_success ⊆ raw_success`. It recomputes validator verdicts and victim predictions from stored final flows.

The full analysis audit covers 1,152 NPZ files and 921,600 attacked flow-instances, with 921,600 validator and raw-success rechecks and zero prediction mismatches. Machine-readable outputs include per-sample Parquet, seed-level CSV, table-level CSV, statistical CSV, configuration JSON, selections, cells, and logs. Machine-specific absolute paths appear inside manifests but are not required to understand the method.

<!--
Repository evidence:
- environment.yml
- FINAL_OUTPUTS/analysis_audit.json
- FINAL_OUTPUTS/runs/final_suite_config.json
- outputs/cicids2017distrinet/classifier_run_manifest.json
- outputs/cicids2018distrinet/classifiers_multiseed/multiseed_manifest.json
-->

# Chapter 5 — Results and Evaluation

## 5.1 Evaluation Overview

This chapter evaluates victim quality, classifier-level attack success, constraint-valid success, primitive eligibility, optimizer and budget sensitivity, objective sensitivity, validator behavior, and paired statistical evidence. The main distinction is between Raw ASR and Valid ASR on the same attempted flows. All headline attack values are mean ± sample standard deviation across attack seeds 42, 2024, and 2026. Superseded relaxed-padding outputs are not used as canonical results.

<!--
Repository evidence:
- FINAL_OUTPUTS/final_experiment_summary.md
- FINAL_OUTPUTS/interpretation.md
- FINAL_OUTPUTS/00_PROTOCOL.md
-->

## 5.2 Dataset and Victim Model Results

The split counts in Table 4.3 show two different evaluation populations. CICIDS2017 retains its naturally imbalanced post-cleaning distribution, with Benign comprising about 79.2% of every split and BruteForce about 0.33%. CICIDS2018 deliberately uses controlled class totals, giving approximately 30/24/24/11/11% across the five classes. Metrics across these datasets should not be interpreted as estimates under the same operational class prior.

Only five-class category models are attacked. CICIDS2017 uses one training run per architecture. CICIDS2018 has three training replicates, but the final attacks use the seed-42 checkpoints shown separately below.

| Dataset | Category victim | Test accuracy | Balanced accuracy | Macro-F1 | Scope |
|---|---|---:|---:|---:|---|
| CICIDS2017 | SimpleMLP | 98.448% | 99.031% | 97.725% | One training seed |
| CICIDS2017 | CNNOnly | 98.433% | 98.914% | 97.528% | One training seed |
| CICIDS2017 | FT-Transformer | 98.458% | 99.099% | 97.796% | One training seed |
| CICIDS2018 | SimpleMLP | 99.793% ± 0.018% | 99.819% ± 0.018% | 99.802% ± 0.015% | Three training seeds; attacks use s42 |
| CICIDS2018 | CNNOnly | 99.784% ± 0.082% | 99.816% ± 0.059% | 99.793% ± 0.069% | Three training seeds; attacks use s42 |
| CICIDS2018 | FT-Transformer | 99.960% ± 0.009% | 99.967% ± 0.005% | 99.965% ± 0.009% | Three training seeds; attacks use s42 |

For the specific CICIDS2018 seed-42 checkpoints attacked in the final suite, macro-F1 is 99.788% for MLP, 99.873% for CNN, and 99.956% for FT-Transformer. The models therefore provide high-quality clean targets. This does not imply calibrated probabilities, deployment robustness, or training-seed robustness for the CICIDS2017 victims.

> **[FIGURE 5.1 PLACEHOLDER — Category-victim confusion matrices]**
> Suggested content: one normalized test confusion matrix per attacked victim and dataset, preferably grouped by dataset.
> Suggested source/artifact: classifier output `plots/` and confusion-matrix JSON files under the 2017 and 2018 classifier directories.
> Purpose: Demonstrates that attack denominators are drawn from meaningful classifiers and reveals residual class confusions.

<!--
Repository evidence:
- outputs/cicids2017distrinet/classifier_metrics_summary.csv
- outputs/cicids2017distrinet_ft/classifier_metrics_summary.csv
- outputs/cicids2017distrinet/per_class_metrics.csv
- outputs/cicids2018distrinet/classifiers_multiseed/cicids2018_multiseed_classifier_results.md
- outputs/cicids2018distrinet/classifiers_multiseed/runs/seed_42/classifier_metrics_summary.csv
-->

## 5.3 Experiment A — Main Attack Comparison

Experiment A compares five inferential methods under an untargeted objective on identical paired flows. Native CAPGD is included as a descriptive row. Every dataset–victim–method cell attempts 3,200 flows per seed.

### 5.3.1 CICIDS2017

| Victim | Method | Raw ASR | Valid ASR | Gap |
|---|---|---:|---:|---:|
| MLP | PrimAttack, Prim-PGD p75 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 pp |
| MLP | PGD | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 pp |
| MLP | C&W | 99.94% ± 0.00% | 0.00% ± 0.00% | 99.94 pp |
| MLP | CAPGD-PrimSupport | 94.41% ± 0.71% | 2.01% ± 0.10% | 92.40 pp |
| MLP | C-PGD-PrimSupport | 50.80% ± 2.04% | 0.00% ± 0.00% | 50.80 pp |
| MLP | CAPGD native † | 94.65% ± 0.88% | 9.53% ± 0.85% | 85.11 pp |
| CNN | PrimAttack, Prim-PGD p75 | 13.47% ± 0.00% | 13.47% ± 0.00% | 0.00 pp |
| CNN | PGD | 96.12% ± 0.09% | 0.00% ± 0.00% | 96.12 pp |
| CNN | C&W | 95.53% ± 0.00% | 0.00% ± 0.00% | 95.53 pp |
| CNN | CAPGD-PrimSupport | 96.53% ± 1.07% | 5.15% ± 0.07% | 91.39 pp |
| CNN | C-PGD-PrimSupport | 60.42% ± 2.98% | 0.00% ± 0.00% | 60.42 pp |
| CNN | CAPGD native † | 96.25% ± 1.19% | 19.24% ± 0.13% | 77.01 pp |
| FT-Transformer | PrimAttack, Prim-PGD p75 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 pp |
| FT-Transformer | PGD | 97.36% ± 0.28% | 0.00% ± 0.00% | 97.36 pp |
| FT-Transformer | C&W | 77.16% ± 0.00% | 0.00% ± 0.00% | 77.16 pp |
| FT-Transformer | CAPGD-PrimSupport | 52.21% ± 4.33% | 0.18% ± 0.02% | 52.03 pp |
| FT-Transformer | C-PGD-PrimSupport | 21.61% ± 0.31% | 0.00% ± 0.00% | 21.61 pp |
| FT-Transformer | CAPGD native † | 50.61% ± 3.39% | 5.71% ± 1.07% | 44.91 pp |

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/table_level.csv
- FINAL_OUTPUTS/A_primary_baseline_comparison/primary_baseline_comparison.md
-->

### 5.3.2 CICIDS2018

| Victim | Method | Raw ASR | Valid ASR | Gap |
|---|---|---:|---:|---:|
| MLP-s42 | PrimAttack, Prim-PGD p75 | 2.53% ± 0.00% | 2.53% ± 0.00% | 0.00 pp |
| MLP-s42 | PGD | 94.34% ± 0.25% | 0.00% ± 0.00% | 94.34 pp |
| MLP-s42 | C&W | 87.63% ± 0.00% | 0.00% ± 0.00% | 87.62 pp |
| MLP-s42 | CAPGD-PrimSupport | 91.57% ± 0.84% | 0.14% ± 0.02% | 91.44 pp |
| MLP-s42 | C-PGD-PrimSupport | 28.25% ± 0.51% | 0.00% ± 0.00% | 28.25 pp |
| MLP-s42 | CAPGD native † | 80.42% ± 0.69% | 28.42% ± 1.13% | 52.00 pp |
| CNN-s42 | PrimAttack, Prim-PGD p75 | 1.16% ± 0.00% | 1.16% ± 0.00% | 0.00 pp |
| CNN-s42 | PGD | 99.70% ± 0.02% | 0.00% ± 0.00% | 99.70 pp |
| CNN-s42 | C&W | 99.16% ± 0.00% | 0.00% ± 0.00% | 99.16 pp |
| CNN-s42 | CAPGD-PrimSupport | 76.01% ± 1.68% | 0.29% ± 0.07% | 75.72 pp |
| CNN-s42 | C-PGD-PrimSupport | 50.54% ± 3.30% | 0.00% ± 0.00% | 50.54 pp |
| CNN-s42 | CAPGD native † | 66.44% ± 1.28% | 16.30% ± 3.22% | 50.14 pp |
| FT-Transformer-s42 | PrimAttack, Prim-PGD p75 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp |
| FT-Transformer-s42 | PGD | 91.70% ± 0.18% | 0.00% ± 0.00% | 91.70 pp |
| FT-Transformer-s42 | C&W | 53.59% ± 0.00% | 0.00% ± 0.00% | 53.59 pp |
| FT-Transformer-s42 | CAPGD-PrimSupport | 9.74% ± 0.70% | 0.00% ± 0.00% | 9.74 pp |
| FT-Transformer-s42 | C-PGD-PrimSupport | 1.65% ± 0.42% | 0.00% ± 0.00% | 1.65 pp |
| FT-Transformer-s42 | CAPGD native † | 5.96% ± 0.53% | 0.45% ± 0.45% | 5.51 pp |

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/table_level.csv
- FINAL_OUTPUTS/A_primary_baseline_comparison/primary_baseline_comparison.md
-->

† Native CAPGD is descriptive and was not included in Experiment A's Cochran-Q/Holm family.

The raw ranking favors the least restricted attacks, but validity reverses the interpretation. PGD and C&W fool the classifiers on 53.59–100% of flows yet produce no valid success. Matched-support methods retain large Raw ASR but little valid success. PrimAttack has a lower Raw ASR, but every successful p75 output is validator-valid. Native CAPGD achieves the highest Valid ASR on all six victims, demonstrating that the support and parameterization materially define the threat model rather than proving one method universally superior.

> **[FIGURE 5.2 — Raw ASR by attack and victim]**
> Existing artifact: `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A1_raw_asr_by_attack.png`.
> Purpose: Displays classifier-level vulnerability before validity enforcement.

> **[FIGURE 5.3 — Valid ASR by attack and victim]**
> Existing artifact: `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A2_valid_asr_by_attack.png`.
> Purpose: Shows how the ranking changes under the domain-validity requirement.

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/primary_baseline_comparison.md
- FINAL_OUTPUTS/A_primary_baseline_comparison/table_level.csv
- FINAL_OUTPUTS/final_experiment_summary.md
-->

### 5.3.3 CAPGD-PrimSupport versus PrimAttack

This comparison matches source flows, victims, attack seeds, final validator, and the 23-coordinate potential write support. It does not match the feasible set: CAPGD directly chooses downstream values, whereas PrimAttack reaches them through source-applicable primitives and coupled recomputation.

| Dataset | Victim | PrimAttack Valid ASR | CAPGD-PrimSupport Valid ASR | Difference, PrimAttack−CAPGD | Seed-42 discordance, Prim-only/CAPGD-only | Holm $p$ |
|---|---|---:|---:|---:|---:|---:|
| 2017 | MLP | 4.09% ± 0.00% | 2.01% ± 0.10% | +2.08 pp | 125 / 59 | $1.65\times10^{-6}$ |
| 2017 | CNN | 13.47% ± 0.00% | 5.15% ± 0.07% | +8.32 pp | 364 / 99 | $1.33\times10^{-34}$ |
| 2017 | FT-Transformer | 0.12% ± 0.00% | 0.18% ± 0.02% | −0.05 pp | 4 / 5 | 1.0 |
| 2018 | MLP-s42 | 2.53% ± 0.00% | 0.14% ± 0.02% | +2.40 pp | 80 / 3 | $7.30\times10^{-17}$ |
| 2018 | CNN-s42 | 1.16% ± 0.00% | 0.29% ± 0.07% | +0.86 pp | 37 / 7 | $1.23\times10^{-5}$ |
| 2018 | FT-Transformer-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 pp | — | Not performed; omnibus Q not significant |

The results show that direct constrained feature optimization can locate valid feature vectors that differ from PrimAttack-reachable vectors, while PrimAttack can outperform the direct baseline when validator-aware timing search aligns with a victim boundary. Neither direction establishes physical realizability. The evidence supports three nested notions: validator-valid feature point, PrimAttack-reachable point, and packet-realizable point; only the first two are represented experimentally.

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/capability_fix/capgd_primsupport_fairness.csv
- FINAL_OUTPUTS/A_primary_baseline_comparison/statistical_tests.csv
-->

## 5.4 Raw versus Constraint-Valid Attack Success

The paired validity gap is largest for attacks that directly manipulate many or independently selected features. For PGD and C&W, every seed-42 raw success fails validation on every victim; mean gaps range from 53.59 to 100 percentage points. CAPGD-PrimSupport gaps range from 9.74 to 92.40 points. C-PGD-PrimSupport gaps range from 1.65 to 60.42 points and no raw success survives. Capability-aware PrimAttack has a zero gap for all untargeted and targeted cells.

Failure categories explain the difference. All invalid PGD and C&W successes fail SCHEMA, EXTRACTOR, and PROTOCOL rules; at least 98.5% also fail MINED rules. Matched-support methods pass SCHEMA after type repair but frequently violate extractor identities and mined invariants. Among their invalid seed-42 successes, 67–100% of CAPGD-PrimSupport and 100% of C-PGD-PrimSupport fail EXTRACTOR rules; 84–100% fail MINED rules. Category shares overlap because one sample can violate several rule families.

`PROTO_0080` detects matched-support attacks that raise a zero source forward minimum. However, removing that rule changes almost no final verdict because most affected examples already fail extractor or mined rules. Across three seeds it alone removes 16 CAPGD-PrimSupport valid successes for the CICIDS2017 MLP and six for the CNN, corresponding to 0.17 and 0.06 percentage points.

> **[FIGURE 5.4 — Raw ASR versus Valid ASR]**
> Existing artifact: `FINAL_OUTPUTS/E_paired_validity_gap/plots/E1_raw_vs_valid_asr.png`.
> Purpose: Directly displays how far each method falls below the raw-equals-valid diagonal.

> **[TABLE 5.1 PLACEHOLDER — Invalid raw successes by validator category]**
> Suggested columns: Dataset, victim, method, raw-success-but-invalid count, SCHEMA %, EXTRACTOR %, PROTOCOL %, MINED %.
> Existing source: `FINAL_OUTPUTS/E_paired_validity_gap/rejection_categories_of_invalid_successes.csv`.
> Purpose: Explains which consistency layers remove apparent successes.

<!--
Repository evidence:
- FINAL_OUTPUTS/E_paired_validity_gap/paired_validity_gap_analysis.md
- FINAL_OUTPUTS/E_paired_validity_gap/rejection_categories_of_invalid_successes.csv
- FINAL_OUTPUTS/E_paired_validity_gap/statistical_tests.csv
-->

## 5.5 PrimAttack Primitive Ablation and Eligibility

The revised padding rule makes the final attack almost entirely timing-based. In the canonical rosters, 99.97% of CICIDS2017 flows and 99.63% of CICIDS2018 flows have zero forward minimum. Only one of 3,200 CICIDS2017 flows and 12 of 3,200 CICIDS2018 flows per victim are padding-eligible. After p75 bounds, approximately 73.8–74.7% of flows are timing-only and 25.0–26.2% have no primitive headroom. The no-primitive group is dominated by Recon flows.

| Dataset | Victim | Timing eligible | Padding eligible | Neither eligible | p75 timing-only | p75 padding-only | p75 no primitive |
|---|---|---:|---:|---:|---:|---:|---:|
| 2017 | MLP | 75.28% | 0.03% | 24.72% | 73.81% | 0.03% | 26.16% |
| 2017 | CNN | 75.25% | 0.03% | 24.75% | 73.81% | 0.03% | 26.16% |
| 2017 | FT-Transformer | 75.28% | 0.03% | 24.72% | 73.84% | 0.03% | 26.13% |
| 2018 | MLP-s42 | 77.78% | 0.38% | 21.97% | 74.72% | 0.25% | 25.03% |
| 2018 | CNN-s42 | 77.78% | 0.38% | 21.97% | 74.69% | 0.25% | 25.06% |
| 2018 | FT-Transformer-s42 | 77.78% | 0.38% | 21.97% | 74.72% | 0.25% | 25.03% |

Joint and timing-only modes produce identical success masks on all six victims. Padding-only produces zero raw and valid successes, including when conditioned on the one or 12 padding-eligible flows.

| Dataset | Victim | Joint Valid ASR | Timing-only Valid ASR | Padding-only Valid ASR |
|---|---|---:|---:|---:|
| 2017 | MLP | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00% ± 0.00% |
| 2017 | CNN | 13.47% ± 0.00% | 13.47% ± 0.00% | 0.00% ± 0.00% |
| 2017 | FT-Transformer | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00% ± 0.00% |
| 2018 | MLP-s42 | 2.53% ± 0.00% | 2.53% ± 0.00% | 0.00% ± 0.00% |
| 2018 | CNN-s42 | 1.16% ± 0.00% | 1.16% ± 0.00% | 0.00% ± 0.00% |
| 2018 | FT-Transformer-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% |

The post-run primitive-mode inference finds significant omnibus differences on five victims, driven by joint/padding and timing/padding contrasts. Joint versus timing-only has zero discordant flows and $p=1$ wherever tested. Because this inferential analysis was added after the run, the thesis should present it as exploratory support for the descriptive ablation, not as pre-registered evidence.

> **[FIGURE 5.5 — Valid ASR by primitive mode]**
> Existing artifact: `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A7_primitive_ablation.png`.
> Purpose: Shows that all observed p75 success is attributable to timing.

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/capability_fix/eligibility.csv
- FINAL_OUTPUTS/A_primary_baseline_comparison/capability_fix/primitive_ablation.csv
- FINAL_OUTPUTS/statistics/statistical_summary.md
-->

## 5.6 PrimAttack Budget Analysis

Experiment C compares p50, p75, and the envelope-only box under the targeted-to-Benign objective for Prim-PGD and Hybrid Search. Raw and Valid ASR are identical in every cell. The two optimizers coincide at p50 and p75 and differ only slightly in two envelope-only cells.

| Dataset | Victim | p50 Valid targeted ASR | p75 | Envelope-only |
|---|---|---:|---:|---:|
| 2017 | MLP | 2.31% ± 0.00% | 4.09% ± 0.00% | 22.94% ± 0.00% |
| 2017 | CNN | 9.19% ± 0.00% | 13.25% ± 0.00% | 59.69% ± 0.00% |
| 2017 | FT-Transformer | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.59% ± 0.00% |
| 2018 | MLP-s42 | 0.69% ± 0.00% | 0.78% ± 0.00% | 24.76% ± 0.02% |
| 2018 | CNN-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 26.09% ± 0.00% |
| 2018 | FT-Transformer-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.12% ± 0.00% |

These Prim-PGD results are monotone. At seed 42, p75 significantly exceeds p50 only for CICIDS2017 MLP and CNN: 57 and 130 flows are gained, with no losses and Holm-adjusted $p=1.19\times10^{-13}$ and $1.12\times10^{-29}$. Envelope-only significantly exceeds p75 on every victim except CICIDS2018 FT-Transformer, where four flows are gained and Holm-adjusted $p=0.25$.

The large envelope-only increases indicate that the p75 delay cap is a first-order restriction. Median per-flow p75 delay caps are around 0.9 seconds for CICIDS2017 but around 31 milliseconds for CICIDS2018. Envelope-only remains constrained by training p99 feature envelopes and DoS/DDoS rate floors; these results do not represent unlimited timing manipulation.

> **[FIGURE 5.6 — Valid targeted ASR versus primitive budget]**
> Existing artifact: `FINAL_OUTPUTS/C_budget_sensitivity/plots/C2_valid_asr_vs_budget.png`.
> Purpose: Displays the effectiveness–budget relationship and victim dependence.

> **[RESULT STATUS NOTE — Restricted/p25 attacks were not run]**
> The frozen calibration includes p25, but the locked final suite stores p50, p75, and envelope-only. No p25 result is inferred or substituted.

<!--
Repository evidence:
- FINAL_OUTPUTS/C_budget_sensitivity/primattack_budget_sensitivity.md
- FINAL_OUTPUTS/C_budget_sensitivity/statistical_tests.csv
- artifacts/primattack/budget_calibration.json
- artifacts/primattack/budget_calibration_cicids2018.json
-->

## 5.7 PrimAttack Optimizer Selection

Hybrid Search and Prim-PGD each produce 1,752 valid targeted successes in 57,600 attempts, an aggregate 3.042%. Their success masks are identical. The pre-registered tie-break selects Prim-PGD because it uses fewer mean victim evaluations per flow: 188.5 versus 189.6. Prim-C&W is third with 861 successes, or 1.495%.

| Dataset | Victim | Hybrid Valid ASR | Prim-PGD Valid ASR | Prim-C&W Valid ASR |
|---|---|---:|---:|---:|
| 2017 | MLP | 4.09% | 4.09% | 4.09% |
| 2017 | CNN | 13.25% | 13.25% | 4.00% |
| 2017 | FT-Transformer | 0.12% | 0.12% | 0.12% |
| 2018 | MLP-s42 | 0.78% | 0.78% | 0.75% |
| 2018 | CNN-s42 | 0.00% | 0.00% | 0.00% |
| 2018 | FT-Transformer-s42 | 0.00% | 0.00% | 0.00% |

Only the CICIDS2017 CNN has a significant omnibus optimizer effect. Hybrid and Prim-PGD have no discordant flows, while each exceeds Prim-C&W by 296 flows; Holm-adjusted $p=2.0\times10^{-65}$. The capability rule removes Hybrid's distinguishing exact-padding phase from almost every flow, leaving both leading methods to solve essentially the same two-control timing problem. Their trajectories and costs can still differ, so the observed equality is an outcome tie, not an algorithmic equivalence.

> **[FIGURE 5.7 — Valid targeted ASR by PrimAttack optimizer]**
> Existing artifact: `FINAL_OUTPUTS/B_optimizer_selection/plots/B2_valid_asr_by_optimizer.png`.
> Purpose: Supports the pre-registered optimizer selection.

<!--
Repository evidence:
- FINAL_OUTPUTS/B_optimizer_selection/primattack_optimizer_selection.md
- FINAL_OUTPUTS/B_optimizer_selection/optimizer_selection.json
- FINAL_OUTPUTS/B_optimizer_selection/statistical_tests.csv
-->

## 5.8 Objective Sensitivity

Targeted and untargeted Prim-PGD produce nearly identical results on CICIDS2017. MLP and FT-Transformer have identical success sets. On CNN, untargeted ASR is 13.47% versus 13.25% targeted; seven flows succeed only untargeted ($p=0.0156$).

CICIDS2018 shows a clearer distinction. MLP untargeted Valid ASR is 2.53% versus 0.78% targeted, with 56 seed-42 untargeted-only successes ($p=1.99\times10^{-13}$). CNN has 1.16% untargeted and 0% targeted, with 37 untargeted-only successes ($p=3.25\times10^{-9}$). These additional successes are DDoS flows moved to DoS, not to Benign. FT-Transformer has no success under either objective.

| Dataset | Victim | Targeted Valid ASR | Untargeted Valid ASR | Seed-42 targeted-only / untargeted-only | McNemar $p$ |
|---|---|---:|---:|---:|---:|
| 2017 | MLP | 4.09% | 4.09% | 0 / 0 | 1 |
| 2017 | CNN | 13.25% | 13.47% | 0 / 7 | 0.0156 |
| 2017 | FT-Transformer | 0.12% | 0.12% | 0 / 0 | 1 |
| 2018 | MLP-s42 | 0.78% | 2.53% | 0 / 56 | $1.99\times10^{-13}$ |
| 2018 | CNN-s42 | 0.00% | 1.16% | 0 / 37 | $3.25\times10^{-9}$ |
| 2018 | FT-Transformer-s42 | 0.00% | 0.00% | 0 / 0 | 1 |

<!--
Repository evidence:
- FINAL_OUTPUTS/D_objective_sensitivity/objective_sensitivity.md
- FINAL_OUTPUTS/D_objective_sensitivity/statistical_tests.csv
-->

## 5.9 Capability-Aware Padding Analysis

The original capability test enabled padding whenever a flow contained forward payload. Uniform padding, however, changes every modeled forward packet. For a source with `Fwd Packet Length Min = 0`, this necessarily changes at least one empty packet to a positive length. Because the aggregate row cannot identify packet roles, the final method conservatively disables padding on such flows.

Within canonical attacked rosters, 3,199 of 3,200 CICIDS2017 flows and 3,188 of 3,200 CICIDS2018 flows per victim have zero forward minimum. The relaxed rule would have considered approximately 75.1% and 76.8% padding-eligible, whereas the capability-aware rule allows 0.03% and 0.38%. Every canonical valid success uses $p=0$ and positive delay; no success uses padding or fills an empty packet.

The current capability-fix analysis records that the relaxed CICIDS2017 sensitivity result had Valid ASR 11.06%, 36.67%, and 0.50% for MLP, CNN, and FT-Transformer. These values are non-canonical and are reported only to quantify the methodological effect. Post-hoc validation of those fixed outputs leaves approximately 0.15%, 2.31%, and 0.03%. Fresh timing-focused optimization reaches 4.09%, 13.47%, and 0.12%, recovering 126–127, 348–366, and three successes per seed beyond filtering alone. This confirms why a rerun was necessary: deleting invalid padding successes cannot discover timing alternatives.

On CICIDS2018, capability-aware PrimAttack increases untargeted Valid ASR from approximately 0.19%, 0.03%, and 0% to 2.53%, 1.16%, and 0%. The previous search spent evaluations on padding that the dataset-specific mined rule rejected; the revised attack removes that operation before optimization and allocates the budget to timing.

> **[FIGURE 5.8 — Effect of capability-aware padding]**
> Existing artifact: `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A8_relaxed_vs_capability_aware.png`.
> Purpose: Separates relaxed sensitivity, post-hoc filtering, and fresh capability-aware search.

> **[TABLE 5.2 PLACEHOLDER — Empty-forward-packet prevalence by class]**
> Suggested columns: Dataset, source class/label, flows, zero-forward-minimum %, relaxed padding %, capability-aware padding %, timing %.
> Suggested source/artifact: `primattack_empty_packet_fix_report.md` and `FINAL_OUTPUTS/A_primary_baseline_comparison/capability_fix/test_split_padding_eligibility.csv`.
> Purpose: Shows why the restriction changes the effective attack space.

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/capability_fix/eligibility.csv
- FINAL_OUTPUTS/A_primary_baseline_comparison/capability_fix/relaxed_vs_capability_aware.csv
- primattack_empty_packet_fix_report.md
- validation/tests/test_empty_forward_packet_rule.py
-->

## 5.10 Statistical Evaluation

Experiment A's omnibus test is significant on five victims. It is not significant for CICIDS2018 FT-Transformer because all five inferential methods have zero valid success. PrimAttack significantly exceeds PGD, C&W, and C-PGD on the MLP and CNN of both datasets. Against CAPGD-PrimSupport, seed-42 differences are +2.06, +8.28, +2.41, and +0.94 percentage points for those four victim cells, with Holm-adjusted $p\le1.23\times10^{-5}$. None of the four CICIDS2017 FT-Transformer contrasts is significant.

Experiment B supports only one strong optimizer difference: Prim-C&W underperforms the two sign-momentum methods on CICIDS2017 CNN. Experiment C shows that envelope-only timing exceeds p75 on most victims, while the p50-to-p75 increase is statistically clear only on CICIDS2017 MLP and CNN. Experiment D establishes that untargeted success is easier than targeted-to-Benign success for CICIDS2018 MLP and CNN.

In Experiment E, all 24 feature-space baseline Raw-versus-Valid tests show systematic losses, while the 24 PrimAttack optimizer/objective cells have no raw-success-but-invalid discordance. Across the 48 tests, 24 are significant and 24 have no discordant flow. This pattern reflects effect magnitude, not merely sample size: unconstrained methods lose up to 100 percentage points, whereas PrimAttack loses zero.

| Analysis | Dataset/victim example | Comparison | N | A-only / B-only | Difference | Adjusted/raw $p$ |
|---|---|---|---:|---:|---:|---:|
| Exp A | 2017 CNN | PrimAttack vs CAPGD-PrimSupport | 3,200 | 364 / 99 | +8.28 pp | Holm $1.33\times10^{-34}$ |
| Exp A | 2017 FT | PrimAttack vs CAPGD-PrimSupport | 3,200 | 4 / 5 | −0.03 pp | Holm 1.0 |
| Exp B | 2017 CNN | Prim-PGD vs Prim-C&W | 3,200 | 296 / 0 | +9.25 pp | Holm $2.0\times10^{-65}$ |
| Exp C | 2018 MLP | Envelope-only vs p75 | 3,200 | 767 / 0 | +23.97 pp | Holm $4.39\times10^{-168}$ |
| Exp D | 2018 MLP | Targeted vs untargeted | 3,200 | 0 / 56 | −1.75 pp | $1.99\times10^{-13}$ |
| Exp E | 2017 MLP | PGD Raw vs Valid | 3,200 | 3,200 / 0 | −100 pp after validity | $<10^{-300}$ |

Exact complete test rows remain in each experiment's `statistical_tests.csv`. Statistical significance establishes evidence against equal paired marginal success probabilities under this sample and victim. It does not establish practical importance, packet realizability, new-campaign generalization, or equivalence when a comparison is non-significant.

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/statistical_tests.csv
- FINAL_OUTPUTS/B_optimizer_selection/statistical_tests.csv
- FINAL_OUTPUTS/C_budget_sensitivity/statistical_tests.csv
- FINAL_OUTPUTS/D_objective_sensitivity/statistical_tests.csv
- FINAL_OUTPUTS/E_paired_validity_gap/statistical_tests.csv
- FINAL_OUTPUTS/statistics/statistical_summary.md
-->

## 5.11 Validator Evaluation

Experiment F applies validator_v2 to every genuine validation and test flow from both datasets. CICIDS2017 test acceptance is 100%; validation acceptance is 99.9699%, with 94 rejected Benign rows. All 94 violate train-derived URG constant rules. CICIDS2018 general-only acceptance is 100% on both splits. Adding mined rules gives 99.9328% validation acceptance and 99.9360% test acceptance; all 84 and 80 rejections come from `MINED_0001`.

| Dataset | Split | Genuine flows | General-only acceptance | Hybrid acceptance | Hybrid rejections |
|---|---|---:|---:|---:|---:|
| 2017 | Validation | 312,058 | 99.9699% | 99.9699% | 94 |
| 2017 | Test | 312,056 | 100.0000% | 100.0000% | 0 |
| 2018 | Validation | 125,033 | 100.0000% | 99.9328% | 84 |
| 2018 | Test | 125,032 | 100.0000% | 99.9360% | 80 |

No EXTRACTOR or PROTOCOL rule rejects a genuine held-out flow. The source-conditioned transition is ineligible for unperturbed self-comparisons and therefore does not affect these rates. Held-out false-rejection rates below 0.07% support using the validator as a structural gate, but do not prove completeness: an accepted vector may still lack a realizable packet sequence or preserved application behavior.

<!--
Repository evidence:
- FINAL_OUTPUTS/F_validator_evaluation/validator_evaluation.md
- FINAL_OUTPUTS/F_validator_evaluation/validator_acceptance.csv
- FINAL_OUTPUTS/F_validator_evaluation/rule_rejections.csv
-->

## 5.12 Cross-Dataset Analysis

Several conclusions are stable across datasets. Unconstrained PGD and C&W achieve high Raw ASR and zero Valid ASR. C-PGD-PrimSupport also has zero Valid ASR. Capability-aware PrimAttack has zero validity gap, and every successful output is timing-only. FT-Transformer is the least exposed victim under p75 PrimAttack.

The magnitudes are dataset-dependent. CICIDS2017 CNN is substantially more vulnerable to p75 timing than the CICIDS2018 CNN: 13.47% versus 1.16% untargeted Valid ASR. Conversely, envelope-only untargeted PrimAttack is 44.32% on CICIDS2018 MLP versus 22.97% on CICIDS2017 MLP. CICIDS2018's much smaller p75 effective delay cap suppresses calibrated-budget success, but the envelope-only result shows that its victims are not universally insensitive to timing.

Native CAPGD performs especially strongly on CICIDS2018 MLP and CNN, reaching 28.42% and 16.30% Valid ASR, compared with 9.53% and 19.24% on CICIDS2017. This difference cannot be attributed solely to model architecture because preprocessing populations, victim checkpoints, class composition, mined rules, and training envelopes also differ. Cross-dataset percentages therefore support sensitivity analysis, not broad generalization.

| Finding | CICIDS2017 | CICIDS2018 |
|---|---|---|
| p75 PrimAttack Valid ASR, MLP/CNN/FT | 4.09 / 13.47 / 0.12% | 2.53 / 1.16 / 0.00% |
| Envelope-only untargeted PrimAttack | 22.97 / 59.94 / 0.59% | 44.32 / 26.28 / 0.12% |
| Padding-eligible attacked flows | 0.03% | 0.38% |
| Hybrid validator test acceptance | 100.0000% | 99.9360% |
| Main genuine rejection source | None on test | `MINED_0001` |

> **[FIGURE 5.9 PLACEHOLDER — Cross-dataset Valid ASR comparison]**
> Suggested content: paired panels by dataset, victim, and method using the Experiment A table; optionally add p75/envelope PrimAttack markers.
> Suggested source/artifact: Experiment A and C `table_level.csv` files.
> Purpose: Summarizes which conclusions replicate and which depend on dataset or budget.

<!--
Repository evidence:
- FINAL_OUTPUTS/final_experiment_summary.md
- FINAL_OUTPUTS/A_primary_baseline_comparison/table_level.csv
- FINAL_OUTPUTS/C_budget_sensitivity/table_level.csv
- FINAL_OUTPUTS/F_validator_evaluation/validator_acceptance.csv
-->

## 5.13 Synthesis and Discussion

**Validity enforcement changes vulnerability estimates.** The highest raw-success methods are not the highest valid-success methods. For PGD and C&W, 53.59–100 percentage points of apparent success disappear. Raw ASR alone would therefore substantially overstate vulnerability under the implemented domain requirements.

**Feature support is important but insufficient.** Restricting CAPGD and C-PGD to PrimAttack's 23 potential coordinates reduces one confound, yet most successes remain invalid because the coordinates are still optimized independently. Coupled dependencies and source-conditioned capabilities matter beyond a static mask.

**Primitive-domain restriction produces a different reachable set.** PrimAttack reports fewer classifier successes than unconstrained attacks but retains all of its successful outputs under validator_v2. It is higher than CAPGD-PrimSupport on four victims, statistically indistinguishable on CICIDS2017 FT-Transformer, and tied at zero on CICIDS2018 FT-Transformer. The contribution is not that PrimAttack must always win; it is that imposing modeled primitive structure materially changes the measured attack surface.

**The final attack is timing-dominated.** The capability correction removes uniform padding from nearly every canonical source flow. Joint and timing-only outcomes are identical, padding-only fails, and every success uses delay. Thus, the current empirical evidence supports PrimAttack principally as a timing-based flow-level attack under these datasets and source rosters.

**Budget dominates optimizer choice.** Hybrid and Prim-PGD tie under p75, while the envelope-only box produces large gains for MLP and CNN victims. The p75 results should be interpreted as conservative train-calibrated estimates, not upper bounds on timing-based evasion.

**Victim and dataset dependence remain strong.** CICIDS2017 CNN is most exposed at p75, while both FT-Transformers remain near zero. Objective sensitivity is small on CICIDS2017 but meaningful on CICIDS2018 because some untargeted successes remain within malicious classes. Results must therefore be reported per victim and dataset rather than pooled as independent replicates.

<!--
Repository evidence:
- FINAL_OUTPUTS/interpretation.md
- FINAL_OUTPUTS/A_primary_baseline_comparison/interpretation.md
- FINAL_OUTPUTS/B_optimizer_selection/interpretation.md
- FINAL_OUTPUTS/C_budget_sensitivity/interpretation.md
- FINAL_OUTPUTS/D_objective_sensitivity/interpretation.md
-->

## 5.14 Limitations

**Flow-level rather than packet-level realization.** No PCAP is edited, replayed, or passed again through CICFlowMeter. The primitive mapping is an aggregate proxy. Passing its internal checks and validator_v2 does not prove that a concrete packet sequence realizes the vector.

**Aggregate-information limitation.** One flow row does not retain packet identities, order, payload, complete merged timing, scan sequence, authentication attempts, or target response. Several sequence-dependent features are held constant because exact inverse mapping is unavailable.

**Conservative padding capability.** Requiring positive forward minimum sacrifices potential valid opportunities. A mixed flow may contain data packets that could be padded even when another packet is empty, but the aggregate vector cannot identify safe packets. The rule favors defensibility over coverage.

**Primitive coverage.** Only uniform forward packet-length augmentation and added forward delay are modeled. Packet splitting, injection, selective per-packet padding, payload transformation, backward changes, flag changes, endpoint changes, and adaptive target feedback are excluded.

**White-box assumption.** The attacks require victim gradients and knowledge of preprocessing. PrimAttack additionally uses mappings, budgets, and validator access. This is stronger than many operational attacker models.

**Feature-extractor dependence.** PrimAttack's equations and validator identities are tied to the corrected DistriNet CICFlowMeter feature definitions. Other extractors or releases require new audits, profiles, and calibration.

**Validator incompleteness.** Validator_v2 covers its schema, extractor, protocol, and mined grammar. It can miss valid-looking combinations outside that grammar. A low genuine false-rejection rate does not establish completeness against adaptive attacks or malicious-function preservation.

**Dataset and split limitations.** Both datasets are public benchmark traffic. Splits are chronological only within source labels, not globally forward in time or independent campaigns. CICIDS2018 has a controlled, sampled class distribution. CICIDS2017 BruteForce is small. None of these results establishes performance on an operational network.

**Victim replication.** Final attacks use one frozen victim per architecture. The three attack seeds vary optimization, not victim training. CICIDS2017 has no training-seed replication in the final evaluation; CICIDS2018 training replicates exist but only seed 42 is attacked.

**Statistical scope.** Inference uses one reference attack seed and one fixed source roster. Mean ± SD over attack seeds is descriptive. Non-significant results do not prove equivalence. The primitive-mode tests were added post-run. The auxiliary `statistics/` budget request conflicts with the locked p50/p75/envelope family and lacks p25 artifacts.

**Comparability of methods.** Experiment A pairs flows and outcomes but does not equalize native budgets, representations, query counts, or validator access. Native CAPGD has a different 16-feature mask and is descriptive. Matched support still does not imply matched feasibility.

**No distributional-realism claim.** The final policy excludes VAE IDR and True-IDSR. Structural validity must not be described as in-distribution realism.

<!--
Repository evidence:
- docs/full_thesis_methodology/00_OPEN_ISSUES.md
- FINAL_OUTPUTS/00_PROTOCOL.md
- src/attack/realizability/base.py
- src/attack/realizability/cicids2017.py
- FINAL_OUTPUTS/statistics/README.md
-->

# Proposed Figure and Table Inventory

## Figure inventory

| ID | Chapter/Section | Suggested caption | Purpose | Existing artifact path | Needs generation? | Data source | Priority |
|---|---|---|---|---|---|---|---|
| F4.1 | 4.1 | Complete thesis experimental pipeline | Summarize end-to-end workflow and leakage boundary | — | Yes | preprocessing, final-suite driver, analyzer | Essential |
| F4.2 | 4.4 | Dataset preprocessing and leakage-control pipeline | Show ordering of cleaning, split, sampling, fitting, and persistence | — | Yes | preprocessing scripts/manifests | Essential |
| F4.3 | 4.5 | Simplified victim architectures | Clarify model diversity | — | Yes | classifier source | Useful |
| F4.4 | 4.8 | Validator_v2 architecture and rule families | Separate structural validity from plausibility and capability | — | Yes | validator source/rules | Essential |
| F4.5 | 4.10.3 | PrimAttack primitive-to-feature recomputation graph | Show coupled 23-feature support | — | Yes | primitive model | Essential |
| F4.6 | 4.10.8 | Detailed PrimAttack execution pipeline | Locate capability, projection, victim, validator, and incumbent | — | Yes | optimizer and runner | Essential |
| F4.7 | 4.11 | Final A–F experimental design | Show reused paired cells and objectives | — | Yes | locked protocol | Essential |
| F5.1 | 5.2 | Category-victim confusion matrices | Establish clean target quality | Classifier `plots/` directories | No, if existing plots are selected | classifier predictions | Useful |
| F5.2 | 5.3 | Raw ASR by attack and victim | Show classifier-level attack success | `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A1_raw_asr_by_attack.png` | No | Exp A | Essential |
| F5.3 | 5.3 | Valid ASR by attack and victim | Show ranking after validation | `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A2_valid_asr_by_attack.png` | No | Exp A | Essential |
| F5.4 | 5.4 | Raw versus Valid ASR | Visualize validity gap | `FINAL_OUTPUTS/E_paired_validity_gap/plots/E1_raw_vs_valid_asr.png` | No | Exp E | Essential |
| F5.5 | 5.5 | Valid ASR by primitive mode | Demonstrate timing dominance | `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A7_primitive_ablation.png` | No | mode ablation | Essential |
| F5.6 | 5.6 | Valid targeted ASR versus budget | Show budget sensitivity | `FINAL_OUTPUTS/C_budget_sensitivity/plots/C2_valid_asr_vs_budget.png` | No | Exp C | Essential |
| F5.7 | 5.7 | Valid targeted ASR by optimizer | Support optimizer selection | `FINAL_OUTPUTS/B_optimizer_selection/plots/B2_valid_asr_by_optimizer.png` | No | Exp B | Useful |
| F5.8 | 5.9 | Relaxed versus capability-aware PrimAttack | Show effect of capability correction and rerun | `FINAL_OUTPUTS/A_primary_baseline_comparison/plots/A8_relaxed_vs_capability_aware.png` | No | capability-fix outputs | Essential |
| F5.9 | 5.12 | Cross-dataset Valid ASR comparison | Summarize stable and dataset-dependent findings | — | Yes | Exp A/C table-level CSVs | Useful |
| F5.10 | 5.8 | Targeted versus untargeted Valid ASR | Show objective sensitivity | `FINAL_OUTPUTS/D_objective_sensitivity/plots/D2_valid_asr_targeted_vs_untargeted.png` | No | Exp D | Useful |
| F5.11 | 5.4 | Validity gap by attack | Compare percentage-point losses | `FINAL_OUTPUTS/E_paired_validity_gap/plots/E2_validity_gap_by_method.png` | No | Exp E | Useful |

## Table inventory

| ID | Chapter/Section | Suggested caption | Purpose | Existing artifact path | Needs generation? | Data source | Priority |
|---|---|---|---|---|---|---|---|
| T4.1 | 4.2 | Paired experimental design | Define unit, seeds, and pairing key | — | Yes | protocol/selections | Essential |
| T4.2 | 4.3 | Dataset overview | Compare canonical datasets | — | Included in draft; format for thesis | manifests | Essential |
| T4.3 | 4.3 | Source-label to category mapping | Document target construction | — | Included; expand dropped labels if space permits | manifests | Essential |
| T4.4 | 4.4 | Final split distributions | Record experimental populations | — | Included in draft | manifests | Essential |
| T4.5 | 4.4 | Ordered feature set and semantic groups | Freeze model input and attack roles | — | Yes | manifests/schema/primitive support | Essential |
| T4.6 | 4.5 | Victim architecture and training hyperparameters | Enable reproduction | — | Included in draft | source/manifests | Essential |
| T4.7 | 4.7 | Metric definitions | Prevent denominator ambiguity | — | Included in draft | protocol/analyzer | Essential |
| T4.8 | 4.8 | Validator rule inventory and examples | Explain validity layers | `FINAL_OUTPUTS/F_validator_evaluation/rule_inventory.csv` | Included; expand examples | validator artifacts | Essential |
| T4.9 | 4.9 | Baseline configurations | Document threat-model differences | — | Included in draft | final configs | Essential |
| T4.10 | 4.10.3 | Primitive-to-feature dependencies | Document recomputation | — | Included in condensed form | primitive model | Essential |
| T4.11 | 4.10.4 | Primitive capability rules | Explain source-conditioned modes | — | Included in draft | primitive model | Essential |
| T4.12 | 4.10.5 | Dataset/class budget calibration | Document train-only p25/p50/p75 | — | Included in draft | calibration JSON | Essential |
| T4.13 | 4.11 | Complete experimental matrix | Describe A–F | — | Included in draft | protocol | Essential |
| T4.14 | 4.12 | Statistical questions and tests | Explain minimal test plan | — | Included in draft | protocol/analyzer | Essential |
| T5.1 | 5.2 | Victim test performance | Establish target quality | Classifier summary CSVs | Included in draft | classifier outputs | Essential |
| T5.2 | 5.3 | Experiment A—CICIDS2017 | Main comparison | `FINAL_OUTPUTS/A_primary_baseline_comparison/table_level.csv` | Included in draft | Exp A | Essential |
| T5.3 | 5.3 | Experiment A—CICIDS2018 | Main comparison | same | Included in draft | Exp A | Essential |
| T5.4 | 5.3.3 | CAPGD-PrimSupport versus PrimAttack | Explain matched support | `capability_fix/capgd_primsupport_fairness.csv` | Included in draft | Exp A | Essential |
| T5.5 | 5.4 | Invalid successes by validator family | Explain gaps | `E_paired_validity_gap/rejection_categories_of_invalid_successes.csv` | Format subset | Exp E | Useful |
| T5.6 | 5.5 | Primitive eligibility | Contextualize ablation | `capability_fix/eligibility.csv` | Included in draft | Exp A | Essential |
| T5.7 | 5.5 | Primitive-mode ablation | Attribute success to timing/padding | `capability_fix/primitive_ablation.csv` | Included in draft | Exp A | Essential |
| T5.8 | 5.6 | Budget sensitivity | Quantify budget/effectiveness trade-off | `C_budget_sensitivity/table_level.csv` | Included in draft | Exp C | Essential |
| T5.9 | 5.7 | Optimizer comparison | Support selection | `B_optimizer_selection/table_level.csv` | Included in draft | Exp B | Useful |
| T5.10 | 5.8 | Objective sensitivity | Distinguish evasion objectives | `D_objective_sensitivity/table_level.csv` | Included in draft | Exp D | Useful |
| T5.11 | 5.9 | Padding capability before and after | Quantify correction | `capability_fix/relaxed_vs_capability_aware.csv` | Format concise subset | capability fix | Essential |
| T5.12 | 5.10 | Statistical test results | Report N, discordance, effect, p | each experiment's `statistical_tests.csv` | Included representative rows; appendix full table | A–E | Essential |
| T5.13 | 5.11 | Genuine-flow validator acceptance | Evaluate false rejection | `F_validator_evaluation/validator_acceptance.csv` | Included in draft | Exp F | Essential |
| T5.14 | 5.12 | Cross-dataset summary | Synthesize external validity | — | Included in draft | A/C/F | Useful |

<!--
Repository evidence:
- FINAL_OUTPUTS/A_primary_baseline_comparison/plots/
- FINAL_OUTPUTS/B_optimizer_selection/plots/
- FINAL_OUTPUTS/C_budget_sensitivity/plots/
- FINAL_OUTPUTS/D_objective_sensitivity/plots/
- FINAL_OUTPUTS/E_paired_validity_gap/plots/
- classifier output plot directories
-->

# Experiment Completion and Rerun Audit

| Experiment | Dataset | Method | Current artifact | Valid under final methodology? | Rerun required? | Reason | Expected output path |
|---|---|---|---|---|---|---|---|
| A | 2017/2018 | PGD, C&W, CAPGD-PrimSupport, C-PGD-PrimSupport, native CAPGD | `FINAL_OUTPUTS/runs/*/baselines_untargeted/` | Yes | No | Re-run after validator amendment A2; source-conditioned validation included | Existing path |
| A | 2017/2018 | Capability-aware PrimAttack, Prim-PGD p75 | `FINAL_OUTPUTS/runs/*/primattack_untargeted/` | Yes | No | Final padding capability and validator rule enforced before/after search | Existing path |
| A descriptive | 2017/2018 | PrimAttack envelope-only | same | Yes, as envelope-only sensitivity | No | Retains p99 envelope/capability/rate rules; not a named empirical budget | Existing path |
| A2 primitive ablation | 2017/2018 | Joint/timing-only/padding-only | `FINAL_OUTPUTS/runs/*/primattack_untargeted_modes/` plus joint headline cell | Yes | No | Fresh capability-aware mode runs complete | Existing path |
| B | 2017/2018 | Hybrid, Prim-PGD, Prim-C&W | `FINAL_OUTPUTS/runs/*/primattack_targeted_optimizers/` | Yes | No | Complete three-seed p75 paired run; Prim-PGD selected | Existing path |
| C | 2017/2018 | Prim-PGD and Hybrid, p50/p75/envelope-only | optimizer and budget run directories | Yes | No | Matches locked `00_PROTOCOL.md` family | Existing path |
| C alternative | 2017/2018 | Restricted p25 | No attack artifacts | Not available | Only if thesis changes protocol | Auxiliary `statistics/` request assumes p25, but locked final protocol does not | `FINAL_OUTPUTS/runs/<dataset>/primattack_targeted_budgets/` |
| D | 2017/2018 | Prim-PGD targeted vs untargeted | Reused B p75 and A p75 cells | Yes | No | Same paired flows and final capability rule | Existing paths |
| E | 2017/2018 | Raw vs Valid for A/B conditions | `FINAL_OUTPUTS/E_paired_validity_gap/` | Yes | No | Recomputed with current source-conditioned validator | Existing path |
| F | 2017/2018 | validator_v2 on all val/test flows | `FINAL_OUTPUTS/F_validator_evaluation/` | Yes | No | Current profiles and transition rule; descriptive | Existing path |
| Post-run statistics | 2017/2018 | Primitive-mode inference | `FINAL_OUTPUTS/statistics/` | Yes if labelled post-run | No | Not pre-registered; adds inference to descriptive ablation | Existing path |
| Relaxed-padding sensitivity | 2017/2018 | Pre-A2 PrimAttack | Summaries in `A_primary_baseline_comparison/capability_fix/` | No as canonical; yes only as labelled sensitivity | No | Old operation filled empty packets; canonical suite was rerun | Keep only current capability-fix summary outputs |
| All canonical figures/tables | Both | A–F | Experiment report directories | Yes | No | Generated from canonical run artifacts | Existing paths |

The CAPGD-PrimSupport mask is static and derived from PrimAttack's 23-feature write support. It does not dynamically shrink when a source flow loses padding capability. Therefore its existing attack outputs required revalidation under `PROTO_0080`, and the canonical suite did rerun all baseline stages after A2; no further regeneration is required. The final full analyzer audited 1,152 files and 921,600 rows. No canonical capability-aware experiment is pending.

<!--
Repository evidence:
- FINAL_OUTPUTS/00_PROTOCOL.md
- FINAL_OUTPUTS/analysis_audit.json
- FINAL_OUTPUTS/final_experiment_summary.md
- FINAL_OUTPUTS/statistics/README.md
- scripts/run_final_suite.py
- scripts/analyze_final_suite.py
- primattack_empty_packet_fix_report.md
-->
