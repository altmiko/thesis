# Final Experiment A — Primary baseline comparison (untargeted)

Five attacks run on identical paired source flows under one **untargeted** objective (success = prediction ≠ original malicious class). Metrics: Raw ASR = successes / attempted flows; Valid ASR = (success ∧ validator_v2 `hybrid_valid` on the same final flow) / attempted flows; Validity Gap = Raw − Valid (pp). Values are mean ± SD over attack seeds 42/2024/2026. Classes are pooled within a victim. Victims and datasets are never pooled. Protocol: `../00_PROTOCOL.md`.

PrimAttack configuration: optimizer **PrimAttack (Hybrid Search, p75)** (selected by the pre-registered Exp B rule; see `../B_optimizer_selection/`), joint mode, p75 budget. † = descriptive p75-vs-unbounded row, not part of the inferential comparison.

### Fairness guide — what is held constant and what differs

**Held constant for all five attacks:** the canonical source flows (identical sample IDs, clean inputs, labels and clean predictions, asserted), the victim checkpoints (SHA-256 asserted), preprocessing/scaling, the chronological test split, the untargeted objective (success = prediction ≠ true source class), the seeds 42/2024/2026, the success definitions, the independent validator_v2 `hybrid_valid` verdict on the final adversarial flow, and the aggregation rules (one denominator = attempted clean-correct flows).

**Inherently different (not forced to be equal):**

| Attack | Attack space / feature mask | Norm / budget | Iterations / restarts | Loss | Projection / clamping | Stopping rule | Evaluations per flow | Implementation |
|---|---|---|---|---|---|---|---|---|
| PGD | all 79 features, victim RobustScaler space | L∞ ε = 0.5 | 40 steps, α = 0.05, 1 random start | CE (untargeted) | L∞ ball only (no box/type/mask) | fixed 40 steps; final iterate | 40 fwd+bwd (exact) | `attack/input_baselines.py` |
| C&W | all 79 features, victim RobustScaler space | L2 penalty, no hard ε | ≤ 60 Adam steps, lr 0.01, λ = 1, κ = 0; 1 run | max(z_true − max z_other + κ, 0)·λ + ‖δ‖² | none | convergence (Δδ < 1e-5) or 60 steps; lowest-L2 success kept | 2 fwd per step (exact) | `attack/input_baselines.py` |
| CAPGD-PrimSupport | 23-feature `primattack_joint_feature_mask`; all other 56 features bitwise unchanged (asserted) | L2 ε = 0.5 in train min-max space | 10 steps, 2 restarts (TabularBench CAPGD) | CE (untargeted) | L2 ball + train box + mask + integer-type repair | fixed steps | forward-hook batch mean | frozen `external/tabularbench` via `comparisons/capgd_cicids2017.py` |
| C-PGD-PrimSupport | same 23-feature mask (asserted) | L2 ε = 0.5 in train min-max space | 40 steps, step 0.05, 1 random start | CE − 1.0·differentiable relation penalty | L2 ball + train box + mask + integer-type repair | fixed steps | exact per flow | `comparisons/cpgd_prim_support.py` (Simonetto et al., IJCAI 2022) |
| PrimAttack | primitive controls only: padding p (bytes/fwd packet, increase-only), added forward delay (µs) + shape; downstream changes only through the canonical recomputation φ onto the same 23-feature support | train-calibrated per-class p75 box (joint) | per-flow cap of 256 victim evaluations | untargeted margin z_true − max z_other on realized flows | integer bytes/µs projection, capability gates, quantized recomputation | incumbent: success > failure, lowest primitive cost among successes, best margin among failures | exact per flow | `attack/primitive_optimizer.py` |

**Validator in the loop.** PrimAttack's search success predicate includes validator_v2 (it keeps the cheapest *valid* success). PGD, C&W and CAPGD optimize without the validator; C-PGD optimizes a differentiable subset of flow relations (penalty), not the validator. Validator access is part of the PrimAttack threat model, not a shared setting.

**Matched support is not matched feasibility.** CAPGD/C-PGD may move any of the 23 coordinates independently within their norm ball. PrimAttack reaches the same coordinates only through two coupled, increase-only primitives. The mask gives a controlled matched-support comparison. It does not give CAPGD/C-PGD packet-level realizability.

## Main baseline table

| Dataset | Victim | Attack | Budget / configuration | n/seed | Raw ASR | Valid ASR | Validity Gap | Raw per seed (42/2024/2026, %) | Valid per seed (%) | Perturbation / budget metric |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | PrimAttack (Hybrid Search, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 11.06 / 11.06 / 11.06 | 11.06 / 11.06 / 11.06 | median normalized primitive cost of valid successes 1.400 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | mlp | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp | 100.00 / 100.00 / 100.00 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.4 features modified |
| CICIDS2017 | mlp | C&W | L2 penalty (unbounded), 79 features | 3200 | 99.94% ± 0.00% | 0.00% ± 0.00% | 99.94 ± 0.00 pp | 99.94 / 99.94 / 99.94 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.323; 77.7 features modified |
| CICIDS2017 | mlp | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 94.41% ± 0.71% | 2.18% ± 0.08% | 92.23 ± 0.78 pp | 94.59 / 95.00 / 93.62 | 2.19 / 2.09 / 2.25 | mean L2 (train min-max) 2.762; 19.0 of 23 features modified |
| CICIDS2017 | mlp | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 50.80% ± 2.04% | 0.00% ± 0.00% | 50.80 ± 2.04 pp | 51.41 / 52.47 / 48.53 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.189; 20.7 of 23 features modified |
| CICIDS2017 | mlp | PrimAttack (Hybrid Search, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 47.91% ± 0.00% | 47.91% ± 0.00% | 0.00 ± 0.00 pp | 47.91 / 47.91 / 47.91 | 47.91 / 47.91 / 47.91 | median normalized primitive cost of valid successes 0.769 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | cnn | PrimAttack (Hybrid Search, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 36.67% ± 0.04% | 36.67% ± 0.04% | 0.00 ± 0.00 pp | 36.69 / 36.62 / 36.69 | 36.69 / 36.62 / 36.69 | median normalized primitive cost of valid successes 1.608 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | cnn | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 96.12% ± 0.09% | 0.00% ± 0.00% | 96.12 ± 0.09 pp | 96.22 / 96.12 / 96.03 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.4 features modified |
| CICIDS2017 | cnn | C&W | L2 penalty (unbounded), 79 features | 3200 | 95.53% ± 0.00% | 0.00% ± 0.00% | 95.53 ± 0.00 pp | 95.53 / 95.53 / 95.53 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.407; 77.7 features modified |
| CICIDS2017 | cnn | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 96.53% ± 1.07% | 5.21% ± 0.07% | 91.32 ± 1.00 pp | 97.53 / 96.66 / 95.41 | 5.25 / 5.25 / 5.12 | mean L2 (train min-max) 4.460; 19.6 of 23 features modified |
| CICIDS2017 | cnn | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 60.42% ± 2.98% | 0.00% ± 0.00% | 60.42 ± 2.98 pp | 63.50 / 60.19 / 57.56 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.188; 21.0 of 23 features modified |
| CICIDS2017 | cnn | PrimAttack (Hybrid Search, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 73.41% ± 0.00% | 73.41% ± 0.00% | 0.00 ± 0.00 pp | 73.41 / 73.41 / 73.41 | 73.41 / 73.41 / 73.41 | median normalized primitive cost of valid successes 0.675 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | ft_transformer | PrimAttack (Hybrid Search, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 0.50% ± 0.00% | 0.50% ± 0.00% | 0.00 ± 0.00 pp | 0.50 / 0.50 / 0.50 | 0.50 / 0.50 / 0.50 | median normalized primitive cost of valid successes 0.192 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | ft_transformer | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 97.36% ± 0.28% | 0.00% ± 0.00% | 97.36 ± 0.28 pp | 97.34 / 97.66 / 97.09 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.4 features modified |
| CICIDS2017 | ft_transformer | C&W | L2 penalty (unbounded), 79 features | 3200 | 77.16% ± 0.00% | 0.00% ± 0.00% | 77.16 ± 0.00 pp | 77.16 / 77.16 / 77.16 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 0.845; 77.7 features modified |
| CICIDS2017 | ft_transformer | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 52.21% ± 4.33% | 0.18% ± 0.02% | 52.03 ± 4.32 pp | 51.81 / 56.72 / 48.09 | 0.16 / 0.19 / 0.19 | mean L2 (train min-max) 42.850; 20.5 of 23 features modified |
| CICIDS2017 | ft_transformer | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 21.61% ± 0.31% | 0.00% ± 0.00% | 21.61 ± 0.31 pp | 21.44 / 21.97 / 21.44 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.189; 21.2 of 23 features modified |
| CICIDS2017 | ft_transformer | PrimAttack (Hybrid Search, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 0.97% ± 0.00% | 0.97% ± 0.00% | 0.00 ± 0.00 pp | 0.97 / 0.97 / 0.97 | 0.97 / 0.97 / 0.97 | median normalized primitive cost of valid successes 0.200 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | mlp-s42 | PrimAttack (Hybrid Search, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 12.00% ± 0.03% | 0.19% ± 0.03% | 11.81 ± 0.03 pp | 12.03 / 12.00 / 11.97 | 0.19 / 0.22 / 0.16 | median normalized primitive cost of valid successes 0.988 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | mlp-s42 | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 94.34% ± 0.25% | 0.00% ± 0.00% | 94.34 ± 0.25 pp | 94.34 / 94.09 / 94.59 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.8 features modified |
| CICIDS2018 | mlp-s42 | C&W | L2 penalty (unbounded), 79 features | 3200 | 87.63% ± 0.00% | 0.00% ± 0.00% | 87.62 ± 0.00 pp | 87.62 / 87.62 / 87.62 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.565; 78.5 features modified |
| CICIDS2018 | mlp-s42 | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 91.57% ± 0.84% | 0.14% ± 0.02% | 91.44 ± 0.85 pp | 92.53 / 90.97 / 91.22 | 0.12 / 0.12 / 0.16 | mean L2 (train min-max) 1104.296; 18.7 of 23 features modified |
| CICIDS2018 | mlp-s42 | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 28.25% ± 0.51% | 0.00% ± 0.00% | 28.25 ± 0.51 pp | 27.66 / 28.53 / 28.56 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.192; 20.8 of 23 features modified |
| CICIDS2018 | mlp-s42 | PrimAttack (Hybrid Search, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 44.64% ± 0.05% | 23.51% ± 0.10% | 21.12 ± 0.05 pp | 44.69 / 44.62 / 44.59 | 23.59 / 23.53 / 23.41 | median normalized primitive cost of valid successes 0.600 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | cnn-s42 | PrimAttack (Hybrid Search, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 15.31% ± 0.00% | 0.03% ± 0.00% | 15.28 ± 0.00 pp | 15.31 / 15.31 / 15.31 | 0.03 / 0.03 / 0.03 | median normalized primitive cost of valid successes 0.492 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | cnn-s42 | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 99.70% ± 0.02% | 0.00% ± 0.00% | 99.70 ± 0.02 pp | 99.72 / 99.69 / 99.69 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.8 features modified |
| CICIDS2018 | cnn-s42 | C&W | L2 penalty (unbounded), 79 features | 3200 | 99.16% ± 0.00% | 0.00% ± 0.00% | 99.16 ± 0.00 pp | 99.16 / 99.16 / 99.16 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.136; 78.5 features modified |
| CICIDS2018 | cnn-s42 | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 76.01% ± 1.68% | 0.29% ± 0.07% | 75.72 ± 1.69 pp | 76.72 / 74.09 / 77.22 | 0.22 / 0.31 / 0.34 | mean L2 (train min-max) 1577.341; 19.7 of 23 features modified |
| CICIDS2018 | cnn-s42 | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 50.54% ± 3.30% | 0.00% ± 0.00% | 50.54 ± 3.30 pp | 47.84 / 49.56 / 54.22 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.196; 20.9 of 23 features modified |
| CICIDS2018 | cnn-s42 | PrimAttack (Hybrid Search, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 46.08% ± 0.02% | 5.01% ± 0.20% | 41.07 ± 0.21 pp | 46.09 / 46.09 / 46.06 | 5.12 / 4.78 / 5.12 | median normalized primitive cost of valid successes 0.475 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Hybrid Search, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 0.34 / 0.34 / 0.31 | 0.00 / 0.00 / 0.00 | median normalized primitive cost of valid successes — (p/p_hi + delay/delay_hi) |
| CICIDS2018 | ft_transformer-s42 | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 91.70% ± 0.18% | 0.00% ± 0.00% | 91.70 ± 0.18 pp | 91.91 / 91.59 / 91.59 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.8 features modified |
| CICIDS2018 | ft_transformer-s42 | C&W | L2 penalty (unbounded), 79 features | 3200 | 53.59% ± 0.00% | 0.00% ± 0.00% | 53.59 ± 0.00 pp | 53.59 / 53.59 / 53.59 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 0.565; 78.4 features modified |
| CICIDS2018 | ft_transformer-s42 | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 9.74% ± 0.70% | 0.00% ± 0.00% | 9.74 ± 0.70 pp | 10.47 / 9.69 / 9.06 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 2760.851; 20.2 of 23 features modified |
| CICIDS2018 | ft_transformer-s42 | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 1.65% ± 0.42% | 0.00% ± 0.00% | 1.65 ± 0.42 pp | 1.22 / 1.66 / 2.06 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.200; 21.1 of 23 features modified |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Hybrid Search, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 0.81% ± 0.03% | 0.09% ± 0.00% | 0.72 ± 0.03 pp | 0.78 / 0.81 / 0.84 | 0.09 / 0.09 / 0.09 | median normalized primitive cost of valid successes 0.300 (p/p_hi + delay/delay_hi) |

## Constrained-baseline diagnostic table

`Allowed downstream features` = size of the canonical `primattack_joint_feature_mask` (23 of 79). For CAPGD/C-PGD it is the set of directly optimized coordinates. For PrimAttack it is the potential write-support of its recomputation φ. `Max modified outside mask` = 0 confirms that no feature outside the mask changed in any flow of any seed (also asserted at run time and in this analysis). A modified-feature count is **not** a PrimAttack primitive cost. PrimAttack's primitive cost is reported separately below. Evaluation counts: C-PGD and PrimAttack are exact per flow. CAPGD is the forward-hook batch mean.

| Dataset | Victim | Attack | Allowed downstream features | Mean modified features | Max modified outside mask | Raw ASR | Valid ASR | Validity Gap | Validator pass rate | Mean victim evals / flow |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | CAPGD-PrimSupport | 23 | 18.96 | 0 | 94.41% ± 0.71% | 2.18% ± 0.08% | 92.23 ± 0.78 pp | 2.23% ± 0.13% | 24.8 |
| CICIDS2017 | cnn | CAPGD-PrimSupport | 23 | 19.59 | 0 | 96.53% ± 1.07% | 5.21% ± 0.07% | 91.32 ± 1.00 pp | 5.21% ± 0.07% | 23.5 |
| CICIDS2017 | ft_transformer | CAPGD-PrimSupport | 23 | 20.50 | 0 | 52.21% ± 4.33% | 0.18% ± 0.02% | 52.03 ± 4.32 pp | 0.25% ± 0.11% | 24.9 |
| CICIDS2018 | mlp-s42 | CAPGD-PrimSupport | 23 | 18.70 | 0 | 91.57% ± 0.84% | 0.14% ± 0.02% | 91.44 ± 0.85 pp | 0.14% ± 0.02% | 22.7 |
| CICIDS2018 | cnn-s42 | CAPGD-PrimSupport | 23 | 19.75 | 0 | 76.01% ± 1.68% | 0.29% ± 0.07% | 75.72 ± 1.69 pp | 0.31% ± 0.06% | 24.0 |
| CICIDS2018 | ft_transformer-s42 | CAPGD-PrimSupport | 23 | 20.19 | 0 | 9.74% ± 0.70% | 0.00% ± 0.00% | 9.74 ± 0.70 pp | 0.02% ± 0.02% | 24.9 |
| CICIDS2017 | mlp | C-PGD-PrimSupport | 23 | 20.65 | 0 | 50.80% ± 2.04% | 0.00% ± 0.00% | 50.80 ± 2.04 pp | 0.00% ± 0.00% | 40.0 |
| CICIDS2017 | cnn | C-PGD-PrimSupport | 23 | 20.99 | 0 | 60.42% ± 2.98% | 0.00% ± 0.00% | 60.42 ± 2.98 pp | 0.00% ± 0.00% | 40.0 |
| CICIDS2017 | ft_transformer | C-PGD-PrimSupport | 23 | 21.24 | 0 | 21.61% ± 0.31% | 0.00% ± 0.00% | 21.61 ± 0.31 pp | 0.00% ± 0.00% | 40.0 |
| CICIDS2018 | mlp-s42 | C-PGD-PrimSupport | 23 | 20.79 | 0 | 28.25% ± 0.51% | 0.00% ± 0.00% | 28.25 ± 0.51 pp | 0.00% ± 0.00% | 40.0 |
| CICIDS2018 | cnn-s42 | C-PGD-PrimSupport | 23 | 20.87 | 0 | 50.54% ± 3.30% | 0.00% ± 0.00% | 50.54 ± 3.30 pp | 0.00% ± 0.00% | 40.0 |
| CICIDS2018 | ft_transformer-s42 | C-PGD-PrimSupport | 23 | 21.09 | 0 | 1.65% ± 0.42% | 0.00% ± 0.00% | 1.65 ± 0.42 pp | 0.00% ± 0.00% | 40.0 |
| CICIDS2017 | mlp | PrimAttack (Hybrid Search, p75) | 23 | 15.01 | 0 | 11.06% ± 0.00% | 11.06% ± 0.00% | 0.00 ± 0.00 pp | 100.00% ± 0.00% | 188.3 |
| CICIDS2017 | cnn | PrimAttack (Hybrid Search, p75) | 23 | 14.90 | 0 | 36.67% ± 0.04% | 36.67% ± 0.04% | 0.00 ± 0.00 pp | 100.00% ± 0.00% | 188.2 |
| CICIDS2017 | ft_transformer | PrimAttack (Hybrid Search, p75) | 23 | 13.32 | 0 | 0.50% ± 0.00% | 0.50% ± 0.00% | 0.00 ± 0.00 pp | 100.00% ± 0.00% | 188.1 |
| CICIDS2018 | mlp-s42 | PrimAttack (Hybrid Search, p75) | 23 | 8.71 | 0 | 12.00% ± 0.03% | 0.19% ± 0.03% | 11.81 ± 0.03 pp | 74.22% ± 0.08% | 192.9 |
| CICIDS2018 | cnn-s42 | PrimAttack (Hybrid Search, p75) | 23 | 13.57 | 0 | 15.31% ± 0.00% | 0.03% ± 0.00% | 15.28 ± 0.00 pp | 23.75% ± 0.03% | 192.8 |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Hybrid Search, p75) | 23 | 14.73 | 0 | 0.33% ± 0.02% | 0.00% ± 0.00% | 0.33 ± 0.02 pp | 23.81% ± 0.00% | 192.9 |

### PrimAttack primitive-domain cost / budget information

Normalized primitive cost = p/p_hi + delay/delay_hi (the incumbent's cost rule). The values are medians over valid successes, averaged over seeds.

| Dataset | Victim | Configuration | Valid successes / seed | Median p (bytes/pkt) of valid successes | Median added delay (µs) | Median normalized primitive cost | Valid successes using padding / timing | Median per-flow cap p_hi (bytes) / delay_hi (µs) | Flows with no primitive headroom |
|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | PrimAttack (Hybrid Search, p75) | 354.0 | 27.2 | 312979 | 1.400 | 99% / 97% | 10.0 / 892452 | 24.7% |
| CICIDS2017 | cnn | PrimAttack (Hybrid Search, p75) | 1173.3 | 52.0 | 1355380 | 1.608 | 94% / 99% | 10.0 / 908011 | 24.8% |
| CICIDS2017 | ft_transformer | PrimAttack (Hybrid Search, p75) | 16.0 | 14.5 | 0 | 0.192 | 94% / 6% | 10.0 / 843449 | 24.7% |
| CICIDS2018 | mlp-s42 | PrimAttack (Hybrid Search, p75) | 6.0 | 0.0 | 4116087 | 0.988 | 0% / 100% | 44.2 / 30539 | 22.1% |
| CICIDS2018 | cnn-s42 | PrimAttack (Hybrid Search, p75) | 1.0 | 0.0 | 249003 | 0.492 | 0% / 100% | 44.9 / 30532 | 22.1% |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Hybrid Search, p75) | 0.0 | — | — | — | — / — | 44.2 / 30539 | 22.1% |
| CICIDS2017 | mlp | PrimAttack (Hybrid Search, unbounded) | 1533.0 | 52.0 | 658237 | 0.769 | 98% / 57% | 78.0 / 11603098 | 24.7% |
| CICIDS2017 | cnn | PrimAttack (Hybrid Search, unbounded) | 2349.0 | 36.0 | 6383325 | 0.675 | 92% / 65% | 78.0 / 11603098 | 24.8% |
| CICIDS2017 | ft_transformer | PrimAttack (Hybrid Search, unbounded) | 31.0 | 15.0 | 0 | 0.200 | 77% / 42% | 78.0 / 11668959 | 24.7% |
| CICIDS2018 | mlp-s42 | PrimAttack (Hybrid Search, unbounded) | 752.3 | 0.0 | 70099011 | 0.600 | 0% / 100% | 44.2 / 9508830 | 22.1% |
| CICIDS2018 | cnn-s42 | PrimAttack (Hybrid Search, unbounded) | 160.3 | 0.0 | 53526358 | 0.475 | 0% / 100% | 44.9 / 9508830 | 22.1% |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Hybrid Search, unbounded) | 3.0 | 0.0 | 14399175 | 0.300 | 0% / 100% | 44.2 / 9508830 | 22.1% |

## Statistical analysis (Valid Success)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

Cochran's Q across the five paired attacks per (dataset, victim). Only if it is significant: the four planned McNemar comparisons PrimAttack vs each baseline, Holm-corrected over those four. A = PrimAttack, B = baseline, Δ = Valid ASR(A) − Valid ASR(B) in pp at seed 42.

| Dataset | Victim | Family | Test | Comparison | n | A-only | B-only | Δ (pp) | Variant | Statistic | p | Holm p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Hybrid Search, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 1132.94 | 5.48e-244 | — | Valid success differs among the 5 paired conditions (Q = 1132.9, p = 5.48e-244); planned McNemar tests follow. |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs PGD | 3200 | 354 | 0 | +11.06 | χ² (cc) | 352.00 | 1.55e-78 | 6.21e-78 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than PGD by 11.06 pp (354 vs 0 discordant flows; Holm-adjusted p = 6.21e-78). |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C&W | 3200 | 354 | 0 | +11.06 | χ² (cc) | 352.00 | 1.55e-78 | 6.21e-78 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than C&W by 11.06 pp (354 vs 0 discordant flows; Holm-adjusted p = 6.21e-78). |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs CAPGD-PrimSupport | 3200 | 338 | 54 | +8.88 | χ² (cc) | 204.31 | 2.4e-46 | 2.4e-46 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than CAPGD-PrimSupport by 8.88 pp (338 vs 54 discordant flows; Holm-adjusted p = 2.4e-46). |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C-PGD-PrimSupport | 3200 | 354 | 0 | +11.06 | χ² (cc) | 352.00 | 1.55e-78 | 6.21e-78 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than C-PGD-PrimSupport by 11.06 pp (354 vs 0 discordant flows; Holm-adjusted p = 6.21e-78). |
| CICIDS2017 | cnn | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Hybrid Search, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 4032.01 | <1e-300 | — | Valid success differs among the 5 paired conditions (Q = 4032.0, p = <1e-300); planned McNemar tests follow. |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs PGD | 3200 | 1174 | 0 | +36.69 | χ² (cc) | 1172.00 | 7.42e-257 | 2.97e-256 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than PGD by 36.69 pp (1174 vs 0 discordant flows; Holm-adjusted p = 2.97e-256). |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C&W | 3200 | 1174 | 0 | +36.69 | χ² (cc) | 1172.00 | 7.42e-257 | 2.97e-256 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than C&W by 36.69 pp (1174 vs 0 discordant flows; Holm-adjusted p = 2.97e-256). |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs CAPGD-PrimSupport | 3200 | 1085 | 79 | +31.44 | χ² (cc) | 867.72 | 1.02e-190 | 1.02e-190 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than CAPGD-PrimSupport by 31.44 pp (1085 vs 79 discordant flows; Holm-adjusted p = 1.02e-190). |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C-PGD-PrimSupport | 3200 | 1174 | 0 | +36.69 | χ² (cc) | 1172.00 | 7.42e-257 | 2.97e-256 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than C-PGD-PrimSupport by 36.69 pp (1174 vs 0 discordant flows; Holm-adjusted p = 2.97e-256). |
| CICIDS2017 | ft_transformer | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Hybrid Search, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 45.90 | 2.58e-09 | — | Valid success differs among the 5 paired conditions (Q = 45.9, p = 2.58e-09); planned McNemar tests follow. |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs PGD | 3200 | 16 | 0 | +0.50 | exact binomial |  | 3.05e-05 | 0.000122 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than PGD by 0.50 pp (16 vs 0 discordant flows; Holm-adjusted p = 0.000122). |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C&W | 3200 | 16 | 0 | +0.50 | exact binomial |  | 3.05e-05 | 0.000122 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than C&W by 0.50 pp (16 vs 0 discordant flows; Holm-adjusted p = 0.000122). |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs CAPGD-PrimSupport | 3200 | 16 | 5 | +0.34 | exact binomial |  | 0.0266 | 0.0266 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than CAPGD-PrimSupport by 0.34 pp (16 vs 5 discordant flows; Holm-adjusted p = 0.0266). |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C-PGD-PrimSupport | 3200 | 16 | 0 | +0.50 | exact binomial |  | 3.05e-05 | 0.000122 | PrimAttack (Hybrid Search, p75) has higher Valid ASR than C-PGD-PrimSupport by 0.50 pp (16 vs 0 discordant flows; Holm-adjusted p = 0.000122). |
| CICIDS2018 | mlp-s42 | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Hybrid Search, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 16.00 | 0.00302 | — | Valid success differs among the 5 paired conditions (Q = 16.0, p = 0.00302); planned McNemar tests follow. |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs PGD | 3200 | 6 | 0 | +0.19 | exact binomial |  | 0.0312 | 0.125 | No significant difference (Holm-adjusted p = 0.125; Δ = +0.19 pp, 6 vs 0 discordant flows). |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C&W | 3200 | 6 | 0 | +0.19 | exact binomial |  | 0.0312 | 0.125 | No significant difference (Holm-adjusted p = 0.125; Δ = +0.19 pp, 6 vs 0 discordant flows). |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs CAPGD-PrimSupport | 3200 | 6 | 4 | +0.06 | exact binomial |  | 0.754 | 0.754 | No significant difference (Holm-adjusted p = 0.754; Δ = +0.06 pp, 6 vs 4 discordant flows). |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C-PGD-PrimSupport | 3200 | 6 | 0 | +0.19 | exact binomial |  | 0.0312 | 0.125 | No significant difference (Holm-adjusted p = 0.125; Δ = +0.19 pp, 6 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Hybrid Search, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 23.25 | 0.000113 | — | Valid success differs among the 5 paired conditions (Q = 23.2, p = 0.000113); planned McNemar tests follow. |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs PGD | 3200 | 1 | 0 | +0.03 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.03 pp, 1 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C&W | 3200 | 1 | 0 | +0.03 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.03 pp, 1 vs 0 discordant flows). |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs CAPGD-PrimSupport | 3200 | 1 | 7 | -0.19 | exact binomial |  | 0.0703 | 0.281 | No significant difference (Holm-adjusted p = 0.281; Δ = -0.19 pp, 1 vs 7 discordant flows). |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C-PGD-PrimSupport | 3200 | 1 | 0 | +0.03 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = +0.03 pp, 1 vs 0 discordant flows). |
| CICIDS2018 | ft_transformer-s42 | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Hybrid Search, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 0.00 | 1 | — | No evidence that valid success differs among the 5 conditions (Q = 0.00, p = 1); planned McNemar tests not performed. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs CAPGD-PrimSupport |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Hybrid Search, p75) vs C-PGD-PrimSupport |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |

## Class-wise results

| Dataset | Victim | Class | Attack | n/seed | Raw ASR | Valid ASR | Gap |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | PrimAttack (Hybrid Search, p75) | 800 | 28.88% ± 0.00% | 28.88% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | DDoS | PrimAttack (Hybrid Search, p75) | 800 | 13.25% ± 0.00% | 13.25% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | Recon | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | BruteForce | PrimAttack (Hybrid Search, p75) | 800 | 2.12% ± 0.00% | 2.12% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | DoS | PrimAttack (Hybrid Search, p75) | 800 | 49.29% ± 0.14% | 49.29% ± 0.14% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | DDoS | PrimAttack (Hybrid Search, p75) | 800 | 37.75% ± 0.00% | 37.75% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | Recon | PrimAttack (Hybrid Search, p75) | 800 | 0.38% ± 0.00% | 0.38% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | BruteForce | PrimAttack (Hybrid Search, p75) | 800 | 59.25% ± 0.00% | 59.25% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | DoS | PrimAttack (Hybrid Search, p75) | 800 | 0.38% ± 0.00% | 0.38% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | DDoS | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | Recon | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | BruteForce | PrimAttack (Hybrid Search, p75) | 800 | 1.62% ± 0.00% | 1.62% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | DoS | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | DDoS | PrimAttack (Hybrid Search, p75) | 800 | 43.25% ± 0.13% | 0.29% ± 0.14% | 42.96 ± 0.07 pp |
| CICIDS2018 | mlp-s42 | Recon | PrimAttack (Hybrid Search, p75) | 800 | 4.75% ± 0.00% | 0.46% ± 0.07% | 4.29 ± 0.07 pp |
| CICIDS2018 | mlp-s42 | BruteForce | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | DoS | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | DDoS | PrimAttack (Hybrid Search, p75) | 800 | 57.12% ± 0.00% | 0.12% ± 0.00% | 57.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | Recon | PrimAttack (Hybrid Search, p75) | 800 | 4.12% ± 0.00% | 0.00% ± 0.00% | 4.12 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | BruteForce | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | DoS | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | DDoS | PrimAttack (Hybrid Search, p75) | 800 | 0.08% ± 0.07% | 0.00% ± 0.00% | 0.08 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | Recon | PrimAttack (Hybrid Search, p75) | 800 | 1.25% ± 0.00% | 0.00% ± 0.00% | 1.25 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | BruteForce | PrimAttack (Hybrid Search, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | DoS | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | mlp | DDoS | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | mlp | Recon | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | mlp | BruteForce | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | cnn | DoS | PGD | 800 | 98.58% ± 0.19% | 0.00% ± 0.00% | 98.58 ± 0.19 pp |
| CICIDS2017 | cnn | DDoS | PGD | 800 | 85.92% ± 0.29% | 0.00% ± 0.00% | 85.92 ± 0.29 pp |
| CICIDS2017 | cnn | Recon | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | cnn | BruteForce | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | DoS | PGD | 800 | 98.58% ± 0.26% | 0.00% ± 0.00% | 98.58 ± 0.26 pp |
| CICIDS2017 | ft_transformer | DDoS | PGD | 800 | 99.71% ± 0.31% | 0.00% ± 0.00% | 99.71 ± 0.31 pp |
| CICIDS2017 | ft_transformer | Recon | PGD | 800 | 99.92% ± 0.14% | 0.00% ± 0.00% | 99.92 ± 0.14 pp |
| CICIDS2017 | ft_transformer | BruteForce | PGD | 800 | 91.25% ± 0.88% | 0.00% ± 0.00% | 91.25 ± 0.88 pp |
| CICIDS2018 | mlp-s42 | DoS | PGD | 800 | 99.87% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | DDoS | PGD | 800 | 99.87% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | Recon | PGD | 800 | 99.12% ± 0.00% | 0.00% ± 0.00% | 99.12 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | BruteForce | PGD | 800 | 78.50% ± 1.00% | 0.00% ± 0.00% | 78.50 ± 1.00 pp |
| CICIDS2018 | cnn-s42 | DoS | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | DDoS | PGD | 800 | 99.87% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | Recon | PGD | 800 | 98.92% ± 0.07% | 0.00% ± 0.00% | 98.92 ± 0.07 pp |
| CICIDS2018 | cnn-s42 | BruteForce | PGD | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | DoS | PGD | 800 | 99.92% ± 0.07% | 0.00% ± 0.00% | 99.92 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | DDoS | PGD | 800 | 99.87% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | Recon | PGD | 800 | 99.71% ± 0.07% | 0.00% ± 0.00% | 99.71 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | BruteForce | PGD | 800 | 67.29% ± 0.72% | 0.00% ± 0.00% | 67.29 ± 0.72 pp |
| CICIDS2017 | mlp | DoS | C&W | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | mlp | DDoS | C&W | 800 | 99.75% ± 0.00% | 0.00% ± 0.00% | 99.75 ± 0.00 pp |
| CICIDS2017 | mlp | Recon | C&W | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | mlp | BruteForce | C&W | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | cnn | DoS | C&W | 800 | 85.62% ± 0.00% | 0.00% ± 0.00% | 85.62 ± 0.00 pp |
| CICIDS2017 | cnn | DDoS | C&W | 800 | 96.50% ± 0.00% | 0.00% ± 0.00% | 96.50 ± 0.00 pp |
| CICIDS2017 | cnn | Recon | C&W | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | cnn | BruteForce | C&W | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | DoS | C&W | 800 | 80.25% ± 0.00% | 0.00% ± 0.00% | 80.25 ± 0.00 pp |
| CICIDS2017 | ft_transformer | DDoS | C&W | 800 | 98.75% ± 0.00% | 0.00% ± 0.00% | 98.75 ± 0.00 pp |
| CICIDS2017 | ft_transformer | Recon | C&W | 800 | 31.62% ± 0.00% | 0.00% ± 0.00% | 31.62 ± 0.00 pp |
| CICIDS2017 | ft_transformer | BruteForce | C&W | 800 | 98.00% ± 0.00% | 0.00% ± 0.00% | 98.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | DoS | C&W | 800 | 99.87% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | DDoS | C&W | 800 | 99.87% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | Recon | C&W | 800 | 97.25% ± 0.00% | 0.00% ± 0.00% | 97.25 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | BruteForce | C&W | 800 | 53.50% ± 0.00% | 0.00% ± 0.00% | 53.50 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | DoS | C&W | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | DDoS | C&W | 800 | 99.87% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | Recon | C&W | 800 | 96.75% ± 0.00% | 0.00% ± 0.00% | 96.75 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | BruteForce | C&W | 800 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | DoS | C&W | 800 | 36.62% ± 0.00% | 0.00% ± 0.00% | 36.62 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | DDoS | C&W | 800 | 78.62% ± 0.00% | 0.00% ± 0.00% | 78.62 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | Recon | C&W | 800 | 99.12% ± 0.00% | 0.00% ± 0.00% | 99.12 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | BruteForce | C&W | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | DoS | CAPGD-PrimSupport | 800 | 94.29% ± 0.44% | 1.17% ± 0.07% | 93.12 ± 0.38 pp |
| CICIDS2017 | mlp | DDoS | CAPGD-PrimSupport | 800 | 100.00% ± 0.00% | 7.33% ± 0.29% | 92.67 ± 0.29 pp |
| CICIDS2017 | mlp | Recon | CAPGD-PrimSupport | 800 | 100.00% ± 0.00% | 0.21% ± 0.19% | 99.79 ± 0.19 pp |
| CICIDS2017 | mlp | BruteForce | CAPGD-PrimSupport | 800 | 83.33% ± 2.57% | 0.00% ± 0.00% | 83.33 ± 2.57 pp |
| CICIDS2017 | cnn | DoS | CAPGD-PrimSupport | 800 | 95.17% ± 2.46% | 6.00% ± 0.22% | 89.17 ± 2.24 pp |
| CICIDS2017 | cnn | DDoS | CAPGD-PrimSupport | 800 | 98.33% ± 0.63% | 14.50% ± 0.22% | 83.83 ± 0.64 pp |
| CICIDS2017 | cnn | Recon | CAPGD-PrimSupport | 800 | 94.04% ± 0.56% | 0.33% ± 0.14% | 93.71 ± 0.69 pp |
| CICIDS2017 | cnn | BruteForce | CAPGD-PrimSupport | 800 | 98.58% ± 1.01% | 0.00% ± 0.00% | 98.58 ± 1.01 pp |
| CICIDS2017 | ft_transformer | DoS | CAPGD-PrimSupport | 800 | 32.96% ± 6.11% | 0.67% ± 0.07% | 32.29 ± 6.15 pp |
| CICIDS2017 | ft_transformer | DDoS | CAPGD-PrimSupport | 800 | 50.79% ± 3.22% | 0.04% ± 0.07% | 50.75 ± 3.15 pp |
| CICIDS2017 | ft_transformer | Recon | CAPGD-PrimSupport | 800 | 99.33% ± 0.14% | 0.00% ± 0.00% | 99.33 ± 0.14 pp |
| CICIDS2017 | ft_transformer | BruteForce | CAPGD-PrimSupport | 800 | 25.75% ± 9.25% | 0.00% ± 0.00% | 25.75 ± 9.25 pp |
| CICIDS2018 | mlp-s42 | DoS | CAPGD-PrimSupport | 800 | 95.00% ± 0.70% | 0.00% ± 0.00% | 95.00 ± 0.70 pp |
| CICIDS2018 | mlp-s42 | DDoS | CAPGD-PrimSupport | 800 | 97.83% ± 0.94% | 0.00% ± 0.00% | 97.83 ± 0.94 pp |
| CICIDS2018 | mlp-s42 | Recon | CAPGD-PrimSupport | 800 | 97.38% ± 0.54% | 0.54% ± 0.07% | 96.83 ± 0.51 pp |
| CICIDS2018 | mlp-s42 | BruteForce | CAPGD-PrimSupport | 800 | 76.08% ± 2.75% | 0.00% ± 0.00% | 76.08 ± 2.75 pp |
| CICIDS2018 | cnn-s42 | DoS | CAPGD-PrimSupport | 800 | 50.92% ± 4.57% | 0.00% ± 0.00% | 50.92 ± 4.57 pp |
| CICIDS2018 | cnn-s42 | DDoS | CAPGD-PrimSupport | 800 | 91.50% ± 1.62% | 0.79% ± 0.07% | 90.71 ± 1.55 pp |
| CICIDS2018 | cnn-s42 | Recon | CAPGD-PrimSupport | 800 | 62.54% ± 1.82% | 0.38% ± 0.22% | 62.17 ± 1.75 pp |
| CICIDS2018 | cnn-s42 | BruteForce | CAPGD-PrimSupport | 800 | 99.08% ± 0.69% | 0.00% ± 0.00% | 99.08 ± 0.69 pp |
| CICIDS2018 | ft_transformer-s42 | DoS | CAPGD-PrimSupport | 800 | 0.17% ± 0.07% | 0.00% ± 0.00% | 0.17 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | DDoS | CAPGD-PrimSupport | 800 | 0.92% ± 0.40% | 0.00% ± 0.00% | 0.92 ± 0.40 pp |
| CICIDS2018 | ft_transformer-s42 | Recon | CAPGD-PrimSupport | 800 | 4.75% ± 0.54% | 0.00% ± 0.00% | 4.75 ± 0.54 pp |
| CICIDS2018 | ft_transformer-s42 | BruteForce | CAPGD-PrimSupport | 800 | 33.12% ± 2.78% | 0.00% ± 0.00% | 33.12 ± 2.78 pp |
| CICIDS2017 | mlp | DoS | C-PGD-PrimSupport | 800 | 21.96% ± 1.23% | 0.00% ± 0.00% | 21.96 ± 1.23 pp |
| CICIDS2017 | mlp | DDoS | C-PGD-PrimSupport | 800 | 82.92% ± 5.77% | 0.00% ± 0.00% | 82.92 ± 5.77 pp |
| CICIDS2017 | mlp | Recon | C-PGD-PrimSupport | 800 | 95.00% ± 1.54% | 0.00% ± 0.00% | 95.00 ± 1.54 pp |
| CICIDS2017 | mlp | BruteForce | C-PGD-PrimSupport | 800 | 3.33% ± 1.23% | 0.00% ± 0.00% | 3.33 ± 1.23 pp |
| CICIDS2017 | cnn | DoS | C-PGD-PrimSupport | 800 | 48.67% ± 1.71% | 0.00% ± 0.00% | 48.67 ± 1.71 pp |
| CICIDS2017 | cnn | DDoS | C-PGD-PrimSupport | 800 | 72.92% ± 4.50% | 0.00% ± 0.00% | 72.92 ± 4.50 pp |
| CICIDS2017 | cnn | Recon | C-PGD-PrimSupport | 800 | 50.79% ± 6.64% | 0.00% ± 0.00% | 50.79 ± 6.64 pp |
| CICIDS2017 | cnn | BruteForce | C-PGD-PrimSupport | 800 | 69.29% ± 3.14% | 0.00% ± 0.00% | 69.29 ± 3.14 pp |
| CICIDS2017 | ft_transformer | DoS | C-PGD-PrimSupport | 800 | 7.12% ± 2.38% | 0.00% ± 0.00% | 7.12 ± 2.38 pp |
| CICIDS2017 | ft_transformer | DDoS | C-PGD-PrimSupport | 800 | 0.67% ± 0.29% | 0.00% ± 0.00% | 0.67 ± 0.29 pp |
| CICIDS2017 | ft_transformer | Recon | C-PGD-PrimSupport | 800 | 76.46% ± 2.47% | 0.00% ± 0.00% | 76.46 ± 2.47 pp |
| CICIDS2017 | ft_transformer | BruteForce | C-PGD-PrimSupport | 800 | 2.21% ± 1.28% | 0.00% ± 0.00% | 2.21 ± 1.28 pp |
| CICIDS2018 | mlp-s42 | DoS | C-PGD-PrimSupport | 800 | 21.92% ± 1.95% | 0.00% ± 0.00% | 21.92 ± 1.95 pp |
| CICIDS2018 | mlp-s42 | DDoS | C-PGD-PrimSupport | 800 | 29.50% ± 0.88% | 0.00% ± 0.00% | 29.50 ± 0.88 pp |
| CICIDS2018 | mlp-s42 | Recon | C-PGD-PrimSupport | 800 | 29.79% ± 2.56% | 0.00% ± 0.00% | 29.79 ± 2.56 pp |
| CICIDS2018 | mlp-s42 | BruteForce | C-PGD-PrimSupport | 800 | 31.79% ± 0.69% | 0.00% ± 0.00% | 31.79 ± 0.69 pp |
| CICIDS2018 | cnn-s42 | DoS | C-PGD-PrimSupport | 800 | 21.29% ± 2.15% | 0.00% ± 0.00% | 21.29 ± 2.15 pp |
| CICIDS2018 | cnn-s42 | DDoS | C-PGD-PrimSupport | 800 | 77.08% ± 3.50% | 0.00% ± 0.00% | 77.08 ± 3.50 pp |
| CICIDS2018 | cnn-s42 | Recon | C-PGD-PrimSupport | 800 | 18.88% ± 2.62% | 0.00% ± 0.00% | 18.88 ± 2.62 pp |
| CICIDS2018 | cnn-s42 | BruteForce | C-PGD-PrimSupport | 800 | 84.92% ± 5.90% | 0.00% ± 0.00% | 84.92 ± 5.90 pp |
| CICIDS2018 | ft_transformer-s42 | DoS | C-PGD-PrimSupport | 800 | 0.04% ± 0.07% | 0.00% ± 0.00% | 0.04 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | DDoS | C-PGD-PrimSupport | 800 | 0.21% ± 0.07% | 0.00% ± 0.00% | 0.21 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | Recon | C-PGD-PrimSupport | 800 | 1.42% ± 0.07% | 0.00% ± 0.00% | 1.42 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | BruteForce | C-PGD-PrimSupport | 800 | 4.92% ± 1.82% | 0.00% ± 0.00% | 4.92 ± 1.82 pp |

## Plots

- `plots/A1_raw_asr_by_attack.png` — Raw ASR by attack
- `plots/A2_valid_asr_by_attack.png` — Valid ASR by attack
- `plots/A3_classwise_valid_asr.png` — class-wise Valid ASR
- `plots/A4_modelwise_valid_asr.png` — model-wise Valid ASR
- `plots/A5_validity_gap_by_attack.png` — Validity Gap by attack
- `plots/A6_primattack_p75_vs_unbounded.png` — PrimAttack p75 vs unbounded

## Machine-readable outputs

`per_sample.parquet` (every flow × seed × attack, incl. clean/adversarial prediction, raw success, validator pass, valid success, evaluations, primitive controls, attack parameters), `seed_level.csv`, `table_level.csv`, `statistical_tests.csv`, `constrained_baseline_diagnostics.csv`, `primattack_primitive_costs.csv`.

## Interpretation

Numbers below are seed means over the six victims (3 per dataset, n = 3,200 flows per victim
and seed). All tests are McNemar on seed-42 Valid Success, Holm over the four
PrimAttack-vs-baseline comparisons.

**1. Unrestricted feature-space attacks: high raw success, no valid success.** PGD reaches a
Raw ASR of 91.70–100.00% and C&W 53.59–99.94% on all six victims. Every one of these
adversarial flows is rejected by validator_v2 (Valid ASR = 0.00% for all 12 PGD/C&W cells), so
the Validity Gap equals the Raw ASR. Among the rejected seed-42 examples, 100% violate SCHEMA,
EXTRACTOR and PROTOCOL rules and ≥ 98.5% violate MINED rules (Exp E breakdown). Moving all 79
scaled features independently yields vectors that lie outside the feature domain (SCHEMA),
contradict CICFlowMeter's own aggregate identities (EXTRACTOR), and break protocol rules
(PROTOCOL).

**2. Public constrained attacks restricted to PrimAttack's 23 downstream coordinates.**
Limiting CAPGD and C-PGD to the canonical `primattack_joint_feature_mask` lowers raw success
without removing it. CAPGD-PrimSupport reaches 9.74–96.53% and C-PGD-PrimSupport 1.65–60.42%.
No feature outside the mask changed in any flow (max = 0, asserted). Almost none of this success
is valid. C-PGD reaches a Valid ASR of 0.00% on every victim. CAPGD reaches at most 5.21% (CICIDS2017
CNN), 2.18% on CICIDS2017 MLP and ≤ 0.29% elsewhere. The resulting Validity Gaps are 9.74–92.23 pp
(CAPGD) and 1.65–60.42 pp (C-PGD). Their invalid examples mostly break EXTRACTOR identities
(67–100%) and MINED invariants (84–100%). Changing the 23 coordinates independently decouples
features that one packet-level change moves together. For example, padding changes forward
total length, mean, max, min and byte rate jointly. C-PGD's differentiable penalty covers only
11 of those relations (λ = 1) and does not prevent this. Restricting the support therefore
constrains which features move but not whether they stay mutually consistent.

**3. Reaching the same coordinates only through packet-size/timing primitives.** PrimAttack
(Hybrid Search, p75, untargeted) has far lower raw success than any feature-space baseline:
11.06% / 36.67% / 0.50% on CICIDS2017 MLP / CNN / FT-Transformer, and 12.00% / 15.31% / 0.33% on
CICIDS2018. On CICIDS2017 every raw success is also valid (gap 0.00 pp). That makes it the method
with the highest **Valid** ASR on all three CICIDS2017 victims. Cochran's Q is significant for
each victim, and PrimAttack beats every baseline after Holm correction. The margins are +11.06 pp
over PGD, C&W and C-PGD and +8.88 pp over CAPGD on MLP. On CNN they are +36.69 pp and +31.44 pp.
On FT-Transformer they are +0.50 pp and +0.34 pp (Holm p = 0.027 against CAPGD). On
CICIDS2018 at p75, Valid ASR is near zero for every method: PrimAttack 0.19% / 0.03% / 0.00%,
CAPGD 0.14% / 0.29% / 0.00%. Cochran's Q is significant for MLP and CNN, but none of the four planned
PrimAttack comparisons survives Holm correction (all |Δ| ≤ 0.19 pp). FT-Transformer has no valid
success under any attack (Q = 0, tests not performed). The CICIDS2018 PrimAttack raw successes
are rejected by a single dataset-specific rule, `MINED_0001` (`Fwd Packet Length Min ≈ Packet
Length Min`). Every rejected example uses padding. The p75 timing budget is also small
(median per-flow delay cap about 31 ms vs about 0.9 s on CICIDS2017). The descriptive unbounded run
confirms that the budget, not the method, limits CICIDS2018. Valid ASR rises to 23.51% (MLP) and
5.01% (CNN) there, and to 47.91% / 73.41% on CICIDS2017 MLP / CNN (plot A6).

**4. How much Raw ASR disappears under the validator.** For PGD and C&W all of it: 53.59–100 pp
per victim. For CAPGD 9.74–92.23 pp and for C-PGD 1.65–60.42 pp. For PrimAttack 0.00 pp on CICIDS2017
and 0.33–15.28 pp on CICIDS2018. PrimAttack's raw successes survive validation far more
often. PrimAttack is **not** valid by construction, though: when no valid success exists, its
incumbent is the best-margin failure, which can evade and still be invalid.

**How to read these differences.** The attack with the highest Raw ASR (PGD) is not the best
attack. It is the one with the least constrained threat model, and none of its evasions is a valid
flow. The attacks also differ in validator access. PrimAttack's search uses validator_v2 as part
of its success predicate. The baselines never query it, and C-PGD only sees a differentiable
subset. PrimAttack's valid-success advantage therefore reflects both the primitive
parameterization and this validity-aware search. The comparison controls downstream feature
support. It does not give CAPGD/C-PGD packet-level realizability, and it does not give
PrimAttack realizability beyond the feature-space proxy (no PCAP is modified). Valid evasion is
strongly victim-dependent. FT-Transformer resists every validity-preserving attack
(≤ 0.50% Valid ASR at p75 and ≤ 0.97% even unbounded), while the CICIDS2017 CNN is the most
exposed victim.
