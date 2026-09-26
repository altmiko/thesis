# Final Experiment A — Primary baseline comparison (untargeted)

Five attacks run on identical paired source flows under one **untargeted** objective (success = prediction ≠ original malicious class). Metrics: Raw ASR = successes / attempted flows; Valid ASR = (success ∧ validator_v2 `hybrid_valid` on the same final flow) / attempted flows; Validity Gap = Raw − Valid (pp). Values are mean ± SD over attack seeds 42/2024/2026. Classes are pooled within a victim. Victims and datasets are never pooled. Protocol: `../00_PROTOCOL.md`.

PrimAttack configuration: optimizer **PrimAttack (Prim-PGD, p75)** (selected by the pre-registered Exp B rule; see `../B_optimizer_selection/`), joint mode, p75 budget, capability-aware padding. † = descriptive rows (PrimAttack unbounded; native CAPGD, amendment A3), not part of the inferential comparison.

**Capability-aware PrimAttack (amendment A2).** Padding adds `p` bytes to *every* forward packet. A source flow with `Fwd Packet Length Min = 0` contains at least one zero-length forward packet (e.g. a pure ACK), and aggregate features do not say which one, so padding would put bytes into an empty packet (payload insertion, not length augmentation). PrimAttack therefore infers `pad_allowed = payload present ∧ Fwd Packet Length Min > 0` before optimization; such flows are attacked timing-only with the full per-flow budget. validator_v2 independently rejects any attack output that turns a source minimum of 0 into a positive value (source-conditioned PROTOCOL rule `PROTO_0080`, both datasets). The pre-fix run is kept only as the `PrimAttack-relaxed-padding` sensitivity result (`../superseded_relaxed_padding/`).

### Fairness guide — what is held constant and what differs

**Held constant for all five attacks:** the canonical source flows (identical sample IDs, clean inputs, labels and clean predictions, asserted), the victim checkpoints (SHA-256 asserted), preprocessing/scaling, the chronological test split, the untargeted objective (success = prediction ≠ true source class), the seeds 42/2024/2026, the success definitions, the independent validator_v2 `hybrid_valid` verdict on the final adversarial flow, and the aggregation rules (one denominator = attempted clean-correct flows).

**Inherently different (not forced to be equal):**

| Attack | Attack space / feature mask | Norm / budget | Iterations / restarts | Loss | Projection / clamping | Stopping rule | Evaluations per flow | Implementation |
|---|---|---|---|---|---|---|---|---|
| PGD | all 79 features, victim RobustScaler space | L∞ ε = 0.5 | 40 steps, α = 0.05, 1 random start | CE (untargeted) | L∞ ball only (no box/type/mask) | fixed 40 steps; final iterate | 40 fwd+bwd (exact) | `attack/input_baselines.py` |
| C&W | all 79 features, victim RobustScaler space | L2 penalty, no hard ε | ≤ 60 Adam steps, lr 0.01, λ = 1, κ = 0; 1 run | max(z_true − max z_other + κ, 0)·λ + ‖δ‖² | none | convergence (Δδ < 1e-5) or 60 steps; lowest-L2 success kept | 2 fwd per step (exact) | `attack/input_baselines.py` |
| CAPGD-PrimSupport | 23-feature `primattack_joint_feature_mask`; all other 56 features bitwise unchanged (asserted) | L2 ε = 0.5 in train min-max space | 10 steps, 2 restarts (TabularBench CAPGD) | CE (untargeted) | L2 ball + train box + mask + integer-type repair | fixed steps | forward-hook batch mean | frozen `external/tabularbench` via `comparisons/capgd_cicids2017.py` |
| CAPGD (native) † | native CAPGD configuration mask (directly perturbable + repairable derived features; not the PrimAttack mask) | L2 ε = 0.5 in train min-max space | 10 steps, 2 restarts | CE (untargeted) | L2 ball + train box + mask + CAPGD relation repair | fixed steps | forward-hook batch mean | same frozen TabularBench CAPGD |
| C-PGD-PrimSupport | same 23-feature mask (asserted) | L2 ε = 0.5 in train min-max space | 40 steps, step 0.05, 1 random start | CE − 1.0·differentiable relation penalty | L2 ball + train box + mask + integer-type repair | fixed steps | exact per flow | `comparisons/cpgd_prim_support.py` (Simonetto et al., IJCAI 2022) |
| PrimAttack (capability-aware) | primitive controls only: padding p (bytes/fwd packet, increase-only; only for flows with forward payload AND no zero-length forward packet, `Fwd Packet Length Min > 0`), added forward delay (µs) + shape (flows with ≥ 2 forward packets and non-zero forward IAT); downstream changes only through the canonical recomputation φ onto the same 23-feature support | train-calibrated per-class p75 box (joint; per flow: joint / timing-only / padding-only / none by capability) | per-flow cap of 256 victim evaluations, the whole cap spent on timing for timing-only flows | untargeted margin z_true − max z_other on realized flows | integer bytes/µs projection, capability gates, quantized recomputation | incumbent: success > failure, lowest primitive cost among successes, best margin among failures | exact per flow | `attack/primitive_optimizer.py` |

† descriptive row (amendment A3), not part of Cochran's Q / Holm.

**Validator in the loop.** PrimAttack's search success predicate includes validator_v2 (it keeps the cheapest *valid* success). PGD, C&W and CAPGD optimize without the validator; C-PGD optimizes a differentiable subset of flow relations (penalty), not the validator. Validator access is part of the PrimAttack threat model, not a shared setting.

**Matched support is not matched feasibility.** CAPGD/C-PGD may move any of the 23 coordinates independently within their norm ball. PrimAttack reaches the same coordinates only through two coupled, increase-only primitives, and only where the source flow supports the primitive (most attack flows contain an empty forward packet and are timing-only). The mask gives a controlled matched-support comparison of the parameterization; it does not give CAPGD/C-PGD packet-level realizability, and PrimAttack is not expected to reach a higher Valid ASR.

## Main baseline table

| Dataset | Victim | Attack | Budget / configuration | n/seed | Raw ASR | Valid ASR | Validity Gap | Raw per seed (42/2024/2026, %) | Valid per seed (%) | Perturbation / budget metric |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | PrimAttack (Prim-PGD, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 4.09 / 4.09 / 4.09 | 4.09 / 4.09 / 4.09 | median normalized primitive cost of valid successes 0.617 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | mlp | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp | 100.00 / 100.00 / 100.00 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.4 features modified |
| CICIDS2017 | mlp | C&W | L2 penalty (unbounded), 79 features | 3200 | 99.94% ± 0.00% | 0.00% ± 0.00% | 99.94 ± 0.00 pp | 99.94 / 99.94 / 99.94 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.323; 77.7 features modified |
| CICIDS2017 | mlp | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 94.41% ± 0.71% | 2.01% ± 0.10% | 92.40 ± 0.79 pp | 94.59 / 95.00 / 93.62 | 2.03 / 1.91 / 2.09 | mean L2 (train min-max) 2.762; 19.0 of 23 features modified |
| CICIDS2017 | mlp | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 50.80% ± 2.04% | 0.00% ± 0.00% | 50.80 ± 2.04 pp | 51.41 / 52.47 / 48.53 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.189; 20.7 of 23 features modified |
| CICIDS2017 | mlp | CAPGD (native) † | L2 ε=0.5 (train min-max space), native CAPGD configuration mask | 3200 | 94.65% ± 0.88% | 9.53% ± 0.85% | 85.11 ± 0.08 pp | 94.69 / 95.50 / 93.75 | 9.66 / 10.31 / 8.62 | mean L2 (train min-max) 2.799; 14.1 of 16 features modified |
| CICIDS2017 | mlp | PrimAttack (Prim-PGD, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 22.97% ± 0.00% | 22.97% ± 0.00% | 0.00 ± 0.00 pp | 22.97 / 22.97 / 22.97 | 22.97 / 22.97 / 22.97 | median normalized primitive cost of valid successes 0.300 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | cnn | PrimAttack (Prim-PGD, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 13.47% ± 0.00% | 13.47% ± 0.00% | 0.00 ± 0.00 pp | 13.47 / 13.47 / 13.47 | 13.47 / 13.47 / 13.47 | median normalized primitive cost of valid successes 0.640 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | cnn | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 96.12% ± 0.09% | 0.00% ± 0.00% | 96.12 ± 0.09 pp | 96.22 / 96.12 / 96.03 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.4 features modified |
| CICIDS2017 | cnn | C&W | L2 penalty (unbounded), 79 features | 3200 | 95.53% ± 0.00% | 0.00% ± 0.00% | 95.53 ± 0.00 pp | 95.53 / 95.53 / 95.53 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.407; 77.7 features modified |
| CICIDS2017 | cnn | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 96.53% ± 1.07% | 5.15% ± 0.07% | 91.39 ± 1.00 pp | 97.53 / 96.66 / 95.41 | 5.19 / 5.19 / 5.06 | mean L2 (train min-max) 4.460; 19.6 of 23 features modified |
| CICIDS2017 | cnn | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 60.42% ± 2.98% | 0.00% ± 0.00% | 60.42 ± 2.98 pp | 63.50 / 60.19 / 57.56 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.188; 21.0 of 23 features modified |
| CICIDS2017 | cnn | CAPGD (native) † | L2 ε=0.5 (train min-max space), native CAPGD configuration mask | 3200 | 96.25% ± 1.19% | 19.24% ± 0.13% | 77.01 ± 1.31 pp | 97.53 / 96.03 / 95.19 | 19.12 / 19.22 / 19.38 | mean L2 (train min-max) 5.341; 14.3 of 16 features modified |
| CICIDS2017 | cnn | PrimAttack (Prim-PGD, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 59.94% ± 0.00% | 59.94% ± 0.00% | 0.00 ± 0.00 pp | 59.94 / 59.94 / 59.94 | 59.94 / 59.94 / 59.94 | median normalized primitive cost of valid successes 0.397 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | ft_transformer | PrimAttack (Prim-PGD, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | median normalized primitive cost of valid successes 0.265 (p/p_hi + delay/delay_hi) |
| CICIDS2017 | ft_transformer | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 97.36% ± 0.28% | 0.00% ± 0.00% | 97.36 ± 0.28 pp | 97.34 / 97.66 / 97.09 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.4 features modified |
| CICIDS2017 | ft_transformer | C&W | L2 penalty (unbounded), 79 features | 3200 | 77.16% ± 0.00% | 0.00% ± 0.00% | 77.16 ± 0.00 pp | 77.16 / 77.16 / 77.16 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 0.845; 77.7 features modified |
| CICIDS2017 | ft_transformer | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 52.21% ± 4.33% | 0.18% ± 0.02% | 52.03 ± 4.32 pp | 51.81 / 56.72 / 48.09 | 0.16 / 0.19 / 0.19 | mean L2 (train min-max) 42.850; 20.5 of 23 features modified |
| CICIDS2017 | ft_transformer | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 21.61% ± 0.31% | 0.00% ± 0.00% | 21.61 ± 0.31 pp | 21.44 / 21.97 / 21.44 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.189; 21.2 of 23 features modified |
| CICIDS2017 | ft_transformer | CAPGD (native) † | L2 ε=0.5 (train min-max space), native CAPGD configuration mask | 3200 | 50.61% ± 3.39% | 5.71% ± 1.07% | 44.91 ± 3.64 pp | 49.44 / 54.44 / 47.97 | 4.56 / 5.88 / 6.69 | mean L2 (train min-max) 42.216; 14.7 of 16 features modified |
| CICIDS2017 | ft_transformer | PrimAttack (Prim-PGD, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 0.59% ± 0.00% | 0.59% ± 0.00% | 0.00 ± 0.00 pp | 0.59 / 0.59 / 0.59 | 0.59 / 0.59 / 0.59 | median normalized primitive cost of valid successes 0.050 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | mlp-s42 | PrimAttack (Prim-PGD, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 2.53% ± 0.00% | 2.53% ± 0.00% | 0.00 ± 0.00 pp | 2.53 / 2.53 / 2.53 | 2.53 / 2.53 / 2.53 | median normalized primitive cost of valid successes 0.608 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | mlp-s42 | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 94.34% ± 0.25% | 0.00% ± 0.00% | 94.34 ± 0.25 pp | 94.34 / 94.09 / 94.59 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.8 features modified |
| CICIDS2018 | mlp-s42 | C&W | L2 penalty (unbounded), 79 features | 3200 | 87.63% ± 0.00% | 0.00% ± 0.00% | 87.62 ± 0.00 pp | 87.62 / 87.62 / 87.62 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.565; 78.5 features modified |
| CICIDS2018 | mlp-s42 | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 91.57% ± 0.84% | 0.14% ± 0.02% | 91.44 ± 0.85 pp | 92.53 / 90.97 / 91.22 | 0.12 / 0.12 / 0.16 | mean L2 (train min-max) 1104.296; 18.7 of 23 features modified |
| CICIDS2018 | mlp-s42 | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 28.25% ± 0.51% | 0.00% ± 0.00% | 28.25 ± 0.51 pp | 27.66 / 28.53 / 28.56 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.192; 20.8 of 23 features modified |
| CICIDS2018 | mlp-s42 | CAPGD (native) † | L2 ε=0.5 (train min-max space), native CAPGD configuration mask | 3200 | 80.42% ± 0.69% | 28.42% ± 1.13% | 52.00 ± 0.97 pp | 81.00 / 79.66 / 80.59 | 28.16 / 27.44 / 29.66 | mean L2 (train min-max) 1760.581; 13.6 of 16 features modified |
| CICIDS2018 | mlp-s42 | PrimAttack (Prim-PGD, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 44.32% ± 0.02% | 44.32% ± 0.02% | 0.00 ± 0.00 pp | 44.31 / 44.31 / 44.34 | 44.31 / 44.31 / 44.34 | median normalized primitive cost of valid successes 0.400 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | cnn-s42 | PrimAttack (Prim-PGD, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 1.16% ± 0.00% | 1.16% ± 0.00% | 0.00 ± 0.00 pp | 1.16 / 1.16 / 1.16 | 1.16 / 1.16 / 1.16 | median normalized primitive cost of valid successes 0.848 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | cnn-s42 | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 99.70% ± 0.02% | 0.00% ± 0.00% | 99.70 ± 0.02 pp | 99.72 / 99.69 / 99.69 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.8 features modified |
| CICIDS2018 | cnn-s42 | C&W | L2 penalty (unbounded), 79 features | 3200 | 99.16% ± 0.00% | 0.00% ± 0.00% | 99.16 ± 0.00 pp | 99.16 / 99.16 / 99.16 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 1.136; 78.5 features modified |
| CICIDS2018 | cnn-s42 | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 76.01% ± 1.68% | 0.29% ± 0.07% | 75.72 ± 1.69 pp | 76.72 / 74.09 / 77.22 | 0.22 / 0.31 / 0.34 | mean L2 (train min-max) 1577.341; 19.7 of 23 features modified |
| CICIDS2018 | cnn-s42 | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 50.54% ± 3.30% | 0.00% ± 0.00% | 50.54 ± 3.30 pp | 47.84 / 49.56 / 54.22 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.196; 20.9 of 23 features modified |
| CICIDS2018 | cnn-s42 | CAPGD (native) † | L2 ε=0.5 (train min-max space), native CAPGD configuration mask | 3200 | 66.44% ± 1.28% | 16.30% ± 3.22% | 50.14 ± 2.15 pp | 66.66 / 65.06 / 67.59 | 15.16 / 13.81 / 19.94 | mean L2 (train min-max) 1651.342; 14.2 of 16 features modified |
| CICIDS2018 | cnn-s42 | PrimAttack (Prim-PGD, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 26.28% ± 0.00% | 26.28% ± 0.00% | 0.00 ± 0.00 pp | 26.28 / 26.28 / 26.28 | 26.28 / 26.28 / 26.28 | median normalized primitive cost of valid successes 0.100 (p/p_hi + delay/delay_hi) |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Prim-PGD, p75) | primitive box p75 (joint), ≤256 evals/flow | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 | median normalized primitive cost of valid successes — (p/p_hi + delay/delay_hi) |
| CICIDS2018 | ft_transformer-s42 | PGD | L∞ ε=0.5 (RobustScaler space), 79 features | 3200 | 91.70% ± 0.18% | 0.00% ± 0.00% | 91.70 ± 0.18 pp | 91.91 / 91.59 / 91.59 | 0.00 / 0.00 / 0.00 | mean L∞ (RobustScaler) 0.500; 78.8 features modified |
| CICIDS2018 | ft_transformer-s42 | C&W | L2 penalty (unbounded), 79 features | 3200 | 53.59% ± 0.00% | 0.00% ± 0.00% | 53.59 ± 0.00 pp | 53.59 / 53.59 / 53.59 | 0.00 / 0.00 / 0.00 | mean L2 (RobustScaler) 0.565; 78.4 features modified |
| CICIDS2018 | ft_transformer-s42 | CAPGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 9.74% ± 0.70% | 0.00% ± 0.00% | 9.74 ± 0.70 pp | 10.47 / 9.69 / 9.06 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 2760.851; 20.2 of 23 features modified |
| CICIDS2018 | ft_transformer-s42 | C-PGD-PrimSupport | L2 ε=0.5 (train min-max space), 23-feature PrimAttack mask | 3200 | 1.65% ± 0.42% | 0.00% ± 0.00% | 1.65 ± 0.42 pp | 1.22 / 1.66 / 2.06 | 0.00 / 0.00 / 0.00 | mean L2 (train min-max) 0.200; 21.1 of 23 features modified |
| CICIDS2018 | ft_transformer-s42 | CAPGD (native) † | L2 ε=0.5 (train min-max space), native CAPGD configuration mask | 3200 | 5.96% ± 0.53% | 0.45% ± 0.45% | 5.51 ± 0.88 pp | 6.56 / 5.75 / 5.56 | 0.22 / 0.16 / 0.97 | mean L2 (train min-max) 2739.115; 14.3 of 16 features modified |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Prim-PGD, unbounded) † | primitive box unbounded (joint), ≤256 evals/flow | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 0.12 / 0.12 / 0.12 | 0.12 / 0.12 / 0.12 | median normalized primitive cost of valid successes 0.075 (p/p_hi + delay/delay_hi) |

## Constrained-baseline diagnostic table

`Allowed downstream features` = size of the canonical `primattack_joint_feature_mask` (23 of 79). For CAPGD/C-PGD it is the set of directly optimized coordinates. For PrimAttack it is the potential write-support of its recomputation φ. `Max modified outside mask` = 0 confirms that no feature outside the mask changed in any flow of any seed (also asserted at run time and in this analysis). A modified-feature count is **not** a PrimAttack primitive cost. PrimAttack's primitive cost is reported separately below. Evaluation counts: C-PGD and PrimAttack are exact per flow. CAPGD is the forward-hook batch mean.

| Dataset | Victim | Attack | Allowed downstream features | Mean modified features | Max modified outside mask | Raw ASR | Valid ASR | Validity Gap | Validator pass rate | Mean victim evals / flow |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | CAPGD-PrimSupport | 23 | 18.96 | 0 | 94.41% ± 0.71% | 2.01% ± 0.10% | 92.40 ± 0.79 pp | 2.06% ± 0.14% | 24.8 |
| CICIDS2017 | cnn | CAPGD-PrimSupport | 23 | 19.59 | 0 | 96.53% ± 1.07% | 5.15% ± 0.07% | 91.39 ± 1.00 pp | 5.15% ± 0.07% | 23.5 |
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
| CICIDS2017 | mlp | CAPGD (native) | 16 | 14.10 | n/a (own mask) | 94.65% ± 0.88% | 9.53% ± 0.85% | 85.11 ± 0.08 pp | 9.81% ± 0.58% | 24.8 |
| CICIDS2017 | cnn | CAPGD (native) | 16 | 14.33 | n/a (own mask) | 96.25% ± 1.19% | 19.24% ± 0.13% | 77.01 ± 1.31 pp | 19.48% ± 0.25% | 23.5 |
| CICIDS2017 | ft_transformer | CAPGD (native) | 16 | 14.70 | n/a (own mask) | 50.61% ± 3.39% | 5.71% ± 1.07% | 44.91 ± 3.64 pp | 8.07% ± 2.61% | 24.9 |
| CICIDS2018 | mlp-s42 | CAPGD (native) | 16 | 13.55 | n/a (own mask) | 80.42% ± 0.69% | 28.42% ± 1.13% | 52.00 ± 0.97 pp | 29.60% ± 1.83% | 23.3 |
| CICIDS2018 | cnn-s42 | CAPGD (native) | 16 | 14.19 | n/a (own mask) | 66.44% ± 1.28% | 16.30% ± 3.22% | 50.14 ± 2.15 pp | 18.06% ± 4.15% | 24.5 |
| CICIDS2018 | ft_transformer-s42 | CAPGD (native) | 16 | 14.30 | n/a (own mask) | 5.96% ± 0.53% | 0.45% ± 0.45% | 5.51 ± 0.88 pp | 5.07% ± 3.95% | 25.0 |
| CICIDS2017 | mlp | PrimAttack (Prim-PGD, p75) | 23 | 8.13 | 0 | 4.09% ± 0.00% | 4.09% ± 0.00% | 0.00 ± 0.00 pp | 100.00% ± 0.00% | 187.1 |
| CICIDS2017 | cnn | PrimAttack (Prim-PGD, p75) | 23 | 8.20 | 0 | 13.47% ± 0.00% | 13.47% ± 0.00% | 0.00 ± 0.00 pp | 100.00% ± 0.00% | 187.1 |
| CICIDS2017 | ft_transformer | PrimAttack (Prim-PGD, p75) | 23 | 7.17 | 0 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp | 100.00% ± 0.00% | 187.2 |
| CICIDS2018 | mlp-s42 | PrimAttack (Prim-PGD, p75) | 23 | 6.60 | 0 | 2.53% ± 0.00% | 2.53% ± 0.00% | 0.00 ± 0.00 pp | 100.00% ± 0.00% | 189.9 |
| CICIDS2018 | cnn-s42 | PrimAttack (Prim-PGD, p75) | 23 | 6.58 | 0 | 1.16% ± 0.00% | 1.16% ± 0.00% | 0.00 ± 0.00 pp | 99.90% ± 0.02% | 189.8 |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Prim-PGD, p75) | 23 | 7.80 | 0 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp | 99.94% ± 0.00% | 189.9 |

### PrimAttack primitive-domain cost / budget information

Normalized primitive cost = p/p_hi + delay/delay_hi (the incumbent's cost rule). The values are medians over valid successes, averaged over seeds.

| Dataset | Victim | Configuration | Valid successes / seed | Median p (bytes/pkt) of valid successes | Median added delay (µs) | Median normalized primitive cost | Valid successes using padding / timing | Median per-flow cap p_hi (bytes) / delay_hi (µs) | Flows with no primitive headroom |
|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | PrimAttack (Prim-PGD, p75) | 131.0 | 0.0 | 1262353 | 0.617 | 0% / 100% | 0.0 / 892452 | 26.2% |
| CICIDS2017 | cnn | PrimAttack (Prim-PGD, p75) | 431.0 | 0.0 | 1003291 | 0.640 | 0% / 100% | 0.0 / 908011 | 26.2% |
| CICIDS2017 | ft_transformer | PrimAttack (Prim-PGD, p75) | 4.0 | 0.0 | 5360 | 0.265 | 0% / 100% | 0.0 / 843449 | 26.1% |
| CICIDS2018 | mlp-s42 | PrimAttack (Prim-PGD, p75) | 81.0 | 0.0 | 502226 | 0.608 | 0% / 100% | 0.0 / 30539 | 25.0% |
| CICIDS2018 | cnn-s42 | PrimAttack (Prim-PGD, p75) | 37.0 | 0.0 | 124712 | 0.848 | 0% / 100% | 0.0 / 30532 | 25.1% |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Prim-PGD, p75) | 0.0 | — | — | — | — / — | 0.0 / 30539 | 25.0% |
| CICIDS2017 | mlp | PrimAttack (Prim-PGD, unbounded) | 735.0 | 0.0 | 5149682 | 0.300 | 0% / 100% | 0.0 / 11603098 | 26.2% |
| CICIDS2017 | cnn | PrimAttack (Prim-PGD, unbounded) | 1918.0 | 0.0 | 8006382 | 0.397 | 0% / 100% | 0.0 / 11603098 | 26.2% |
| CICIDS2017 | ft_transformer | PrimAttack (Prim-PGD, unbounded) | 19.0 | 0.0 | 2874359 | 0.050 | 0% / 100% | 0.0 / 11668959 | 26.1% |
| CICIDS2018 | mlp-s42 | PrimAttack (Prim-PGD, unbounded) | 1418.3 | 0.0 | 43881383 | 0.400 | 0% / 100% | 0.0 / 9508830 | 25.0% |
| CICIDS2018 | cnn-s42 | PrimAttack (Prim-PGD, unbounded) | 841.0 | 0.0 | 11684448 | 0.100 | 0% / 100% | 0.0 / 9508830 | 25.1% |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Prim-PGD, unbounded) | 4.0 | 0.0 | 3599892 | 0.075 | 0% / 100% | 0.0 / 9508830 | 25.0% |

## Capability-aware PrimAttack: eligibility, primitive use and ablation

**Capability-aware PrimAttack (amendment A2).** Padding adds `p` bytes to *every* forward packet. A source flow with `Fwd Packet Length Min = 0` contains at least one zero-length forward packet (e.g. a pure ACK), and aggregate features do not say which one, so padding would put bytes into an empty packet (payload insertion, not length augmentation). PrimAttack therefore infers `pad_allowed = payload present ∧ Fwd Packet Length Min > 0` before optimization; such flows are attacked timing-only with the full per-flow budget. validator_v2 independently rejects any attack output that turns a source minimum of 0 into a positive value (source-conditioned PROTOCOL rule `PROTO_0080`, both datasets). The pre-fix run is kept only as the `PrimAttack-relaxed-padding` sensitivity result (`../superseded_relaxed_padding/`).

### Primitive eligibility of the attacked source flows

Capability shares are seed-independent (asserted). `Padding eligible, relaxed rule` = the pre-fix rule (payload present only). `p75 box` = effective per-flow search space after capabilities and the p75 budget (≥ 1 integer unit of headroom).

| Dataset | Victim | Class | n | Fwd min = 0 | Timing eligible | Padding eligible | Joint eligible | Neither | Padding eligible, relaxed rule | p75 box: joint / timing-only / padding-only / none |
|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | 800 | 799 (99.9%) | 100.0% | 0.1% | 0.1% | 0.0% | 100.0% | 0.0 / 98.1 / 0.1 / 1.8% |
| CICIDS2017 | mlp | DDoS | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 96.0 / 0.0 / 4.0% |
| CICIDS2017 | mlp | Recon | 800 | 800 (100.0%) | 1.1% | 0.0% | 0.0% | 98.9% | 0.4% | 0.0 / 1.1 / 0.0 / 98.9% |
| CICIDS2017 | mlp | BruteForce | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 100.0 / 0.0 / 0.0% |
| CICIDS2017 | mlp | ALL | 3200 | 3199 (100.0%) | 75.3% | 0.0% | 0.0% | 24.7% | 75.1% | 0.0 / 73.8 / 0.0 / 26.2% |
| CICIDS2017 | cnn | DoS | 800 | 799 (99.9%) | 100.0% | 0.1% | 0.1% | 0.0% | 100.0% | 0.0 / 98.2 / 0.1 / 1.6% |
| CICIDS2017 | cnn | DDoS | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 96.0 / 0.0 / 4.0% |
| CICIDS2017 | cnn | Recon | 800 | 800 (100.0%) | 1.0% | 0.0% | 0.0% | 99.0% | 0.2% | 0.0 / 1.0 / 0.0 / 99.0% |
| CICIDS2017 | cnn | BruteForce | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 100.0 / 0.0 / 0.0% |
| CICIDS2017 | cnn | ALL | 3200 | 3199 (100.0%) | 75.2% | 0.0% | 0.0% | 24.8% | 75.1% | 0.0 / 73.8 / 0.0 / 26.2% |
| CICIDS2017 | ft_transformer | DoS | 800 | 799 (99.9%) | 100.0% | 0.1% | 0.1% | 0.0% | 100.0% | 0.0 / 98.2 / 0.1 / 1.6% |
| CICIDS2017 | ft_transformer | DDoS | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 96.0 / 0.0 / 4.0% |
| CICIDS2017 | ft_transformer | Recon | 800 | 800 (100.0%) | 1.1% | 0.0% | 0.0% | 98.9% | 0.4% | 0.0 / 1.1 / 0.0 / 98.9% |
| CICIDS2017 | ft_transformer | BruteForce | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 100.0 / 0.0 / 0.0% |
| CICIDS2017 | ft_transformer | ALL | 3200 | 3199 (100.0%) | 75.3% | 0.0% | 0.0% | 24.7% | 75.1% | 0.0 / 73.8 / 0.0 / 26.1% |
| CICIDS2018 | mlp-s42 | DoS | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 92.9 / 0.0 / 7.1% |
| CICIDS2018 | mlp-s42 | DDoS | 800 | 799 (99.9%) | 100.0% | 0.1% | 0.1% | 0.0% | 100.0% | 0.0 / 94.9 / 0.0 / 5.1% |
| CICIDS2018 | mlp-s42 | Recon | 800 | 789 (98.6%) | 11.1% | 1.4% | 0.4% | 87.9% | 7.0% | 0.0 / 11.1 / 1.0 / 87.9% |
| CICIDS2018 | mlp-s42 | BruteForce | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 100.0 / 0.0 / 0.0% |
| CICIDS2018 | mlp-s42 | ALL | 3200 | 3188 (99.6%) | 77.8% | 0.4% | 0.1% | 22.0% | 76.8% | 0.0 / 74.7 / 0.2 / 25.0% |
| CICIDS2018 | cnn-s42 | DoS | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 92.8 / 0.0 / 7.2% |
| CICIDS2018 | cnn-s42 | DDoS | 800 | 799 (99.9%) | 100.0% | 0.1% | 0.1% | 0.0% | 100.0% | 0.0 / 94.9 / 0.0 / 5.1% |
| CICIDS2018 | cnn-s42 | Recon | 800 | 789 (98.6%) | 11.1% | 1.4% | 0.4% | 87.9% | 7.0% | 0.0 / 11.1 / 1.0 / 87.9% |
| CICIDS2018 | cnn-s42 | BruteForce | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 100.0 / 0.0 / 0.0% |
| CICIDS2018 | cnn-s42 | ALL | 3200 | 3188 (99.6%) | 77.8% | 0.4% | 0.1% | 22.0% | 76.8% | 0.0 / 74.7 / 0.2 / 25.1% |
| CICIDS2018 | ft_transformer-s42 | DoS | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 92.9 / 0.0 / 7.1% |
| CICIDS2018 | ft_transformer-s42 | DDoS | 800 | 799 (99.9%) | 100.0% | 0.1% | 0.1% | 0.0% | 100.0% | 0.0 / 94.9 / 0.0 / 5.1% |
| CICIDS2018 | ft_transformer-s42 | Recon | 800 | 789 (98.6%) | 11.1% | 1.4% | 0.4% | 87.9% | 7.0% | 0.0 / 11.1 / 1.0 / 87.9% |
| CICIDS2018 | ft_transformer-s42 | BruteForce | 800 | 800 (100.0%) | 100.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0.0 / 100.0 / 0.0 / 0.0% |
| CICIDS2018 | ft_transformer-s42 | ALL | 3200 | 3188 (99.6%) | 77.8% | 0.4% | 0.1% | 22.0% | 76.8% | 0.0 / 74.7 / 0.2 / 25.0% |

### PrimAttack primitive use (victim level)

Counts per seed 42 / 2024 / 2026. Timing-only = p = 0 ∧ delay > 0; padding-only = p > 0 ∧ delay = 0; joint = both. `Filled empty packet` must be 0 (also asserted).

| Dataset | Victim | Class | N | Fwd min = 0 | Padding disabled (empty pkt) | Attacked timing-only | Padding eligible | Valid successes (42/2024/2026) | Valid timing-only | Valid padding-only | Valid joint | Valid involving padding | Filled empty packet (valid / raw) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | ALL | 3200 | 3199 | 3199 (2402) | 2362 | 1 | 131 / 131 / 131 | 131 / 131 / 131 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | cnn | ALL | 3200 | 3199 | 3199 (2401) | 2362 | 1 | 431 / 431 / 431 | 431 / 431 / 431 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | ft_transformer | ALL | 3200 | 3199 | 3199 (2402) | 2363 | 1 | 4 / 4 / 4 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | mlp-s42 | ALL | 3200 | 3188 | 3188 (2444) | 2391 | 12 | 81 / 81 / 81 | 81 / 81 / 81 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | cnn-s42 | ALL | 3200 | 3188 | 3188 (2444) | 2390 | 12 | 37 / 37 / 37 | 37 / 37 / 37 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | ft_transformer-s42 | ALL | 3200 | 3188 | 3188 (2444) | 2391 | 12 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |

Class level:

| Dataset | Victim | Class | N | Fwd min = 0 | Padding disabled (empty pkt) | Attacked timing-only | Padding eligible | Valid successes (42/2024/2026) | Valid timing-only | Valid padding-only | Valid joint | Valid involving padding | Filled empty packet (valid / raw) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | 800 | 799 | 799 (799) | 785 | 1 | 41 / 41 / 41 | 41 / 41 / 41 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | mlp | DDoS | 800 | 800 | 800 (800) | 768 | 0 | 86 / 86 / 86 | 86 / 86 / 86 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | mlp | Recon | 800 | 800 | 800 (3) | 9 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | mlp | BruteForce | 800 | 800 | 800 (800) | 800 | 0 | 4 / 4 / 4 | 4 / 4 / 4 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | cnn | DoS | 800 | 799 | 799 (799) | 786 | 1 | 164 / 164 / 164 | 164 / 164 / 164 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | cnn | DDoS | 800 | 800 | 800 (800) | 768 | 0 | 257 / 257 / 257 | 257 / 257 / 257 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | cnn | Recon | 800 | 800 | 800 (2) | 8 | 0 | 3 / 3 / 3 | 3 / 3 / 3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | cnn | BruteForce | 800 | 800 | 800 (800) | 800 | 0 | 7 / 7 / 7 | 7 / 7 / 7 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | ft_transformer | DoS | 800 | 799 | 799 (799) | 786 | 1 | 1 / 1 / 1 | 1 / 1 / 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | ft_transformer | DDoS | 800 | 800 | 800 (800) | 768 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | ft_transformer | Recon | 800 | 800 | 800 (3) | 9 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2017 | ft_transformer | BruteForce | 800 | 800 | 800 (800) | 800 | 0 | 3 / 3 / 3 | 3 / 3 / 3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | mlp-s42 | DoS | 800 | 800 | 800 (800) | 743 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | mlp-s42 | DDoS | 800 | 799 | 799 (799) | 759 | 1 | 60 / 60 / 60 | 60 / 60 / 60 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | mlp-s42 | Recon | 800 | 789 | 789 (45) | 89 | 11 | 21 / 21 / 21 | 21 / 21 / 21 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | mlp-s42 | BruteForce | 800 | 800 | 800 (800) | 800 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | cnn-s42 | DoS | 800 | 800 | 800 (800) | 742 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | cnn-s42 | DDoS | 800 | 799 | 799 (799) | 759 | 1 | 37 / 37 / 37 | 37 / 37 / 37 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | cnn-s42 | Recon | 800 | 789 | 789 (45) | 89 | 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | cnn-s42 | BruteForce | 800 | 800 | 800 (800) | 800 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | ft_transformer-s42 | DoS | 800 | 800 | 800 (800) | 743 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | ft_transformer-s42 | DDoS | 800 | 799 | 799 (799) | 759 | 1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | ft_transformer-s42 | Recon | 800 | 789 | 789 (45) | 89 | 11 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |
| CICIDS2018 | ft_transformer-s42 | BruteForce | 800 | 800 | 800 (800) | 800 | 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 |

### Primitive ablation (untargeted, p75, selected optimizer)

Joint = the Exp A PrimAttack cell; timing-only / padding-only restrict the box to one primitive. Denominator = all attacked flows; padding-only is also shown on the padding-eligible flows only (n shown).

| Dataset | Victim | Mode | n/seed | Raw ASR | Valid ASR (all attacked flows) | Valid per seed (%) | Valid ASR on padding-eligible flows |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | joint | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 4.09 / 4.09 / 4.09 | — |
| CICIDS2017 | cnn | joint | 3200 | 13.47% ± 0.00% | 13.47% ± 0.00% | 13.47 / 13.47 / 13.47 | — |
| CICIDS2017 | ft_transformer | joint | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.12 / 0.12 / 0.12 | — |
| CICIDS2018 | mlp-s42 | joint | 3200 | 2.53% ± 0.00% | 2.53% ± 0.00% | 2.53 / 2.53 / 2.53 | — |
| CICIDS2018 | cnn-s42 | joint | 3200 | 1.16% ± 0.00% | 1.16% ± 0.00% | 1.16 / 1.16 / 1.16 | — |
| CICIDS2018 | ft_transformer-s42 | joint | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | timing-only | 3200 | 4.09% ± 0.00% | 4.09% ± 0.00% | 4.09 / 4.09 / 4.09 | — |
| CICIDS2017 | cnn | timing-only | 3200 | 13.47% ± 0.00% | 13.47% ± 0.00% | 13.47 / 13.47 / 13.47 | — |
| CICIDS2017 | ft_transformer | timing-only | 3200 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.12 / 0.12 / 0.12 | — |
| CICIDS2018 | mlp-s42 | timing-only | 3200 | 2.53% ± 0.00% | 2.53% ± 0.00% | 2.53 / 2.53 / 2.53 | — |
| CICIDS2018 | cnn-s42 | timing-only | 3200 | 1.16% ± 0.00% | 1.16% ± 0.00% | 1.16 / 1.16 / 1.16 | — |
| CICIDS2018 | ft_transformer-s42 | timing-only | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | — |
| CICIDS2017 | mlp | padding-only | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% (n = 1) |
| CICIDS2017 | cnn | padding-only | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% (n = 1) |
| CICIDS2017 | ft_transformer | padding-only | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% (n = 1) |
| CICIDS2018 | mlp-s42 | padding-only | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% (n = 12) |
| CICIDS2018 | cnn-s42 | padding-only | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% (n = 12) |
| CICIDS2018 | ft_transformer-s42 | padding-only | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 / 0.00 / 0.00 | 0.00% ± 0.00% (n = 12) |

### PrimAttack-relaxed-padding vs capability-aware PrimAttack

Relaxed = the pre-fix Exp A PrimAttack cell as run (old validator). Post-hoc filtered = the same relaxed flows re-judged by the current validator (a lower bound: it cannot find new timing successes). Capability-aware = the fresh re-run. Timing recovery = capability-aware valid successes − post-hoc filtered valid successes.

| Optimizer (relaxed → new) | Objective | Dataset | Victim | Relaxed Valid ASR | Relaxed, post-hoc filtered | Capability-aware Valid ASR | Δ new − relaxed (pp) | Relative Δ | Timing recovery vs filter (successes, 42/2024/2026) | Timing-only valid: relaxed → new (seed mean) | Padding-based valid: relaxed → new | Padding headroom: relaxed → new |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid Search → Prim-PGD | untargeted | CICIDS2017 | mlp | 11.06% ± 0.00% | 0.15% ± 0.02% | 4.09% ± 0.00% | -6.97 | -63.0% | 126 / 127 / 126 | 4.7 → 131.0 | 349.3 → 0.0 | 75.1% → 0.0% |
| Hybrid Search → Prim-PGD | untargeted | CICIDS2017 | cnn | 36.67% ± 0.04% | 2.31% ± 0.28% | 13.47% ± 0.00% | -23.20 | -63.3% | 357 / 366 / 348 | 74.0 → 431.0 | 1099.3 → 0.0 | 75.0% → 0.0% |
| Hybrid Search → Prim-PGD | untargeted | CICIDS2017 | ft_transformer | 0.50% ± 0.00% | 0.03% ± 0.00% | 0.12% ± 0.00% | -0.38 | -75.0% | 3 / 3 / 3 | 1.0 → 4.0 | 15.0 → 0.0 | 75.1% → 0.0% |
| Hybrid Search → Prim-PGD | untargeted | CICIDS2018 | mlp-s42 | 0.19% ± 0.03% | 0.19% ± 0.03% | 2.53% ± 0.00% | +2.34 | +1250.0% | 75 / 74 / 76 | 6.0 → 81.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Hybrid Search → Prim-PGD | untargeted | CICIDS2018 | cnn-s42 | 0.03% ± 0.00% | 0.03% ± 0.00% | 1.16% ± 0.00% | +1.12 | +3600.0% | 36 / 36 / 36 | 1.0 → 37.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Hybrid Search → Prim-PGD | untargeted | CICIDS2018 | ft_transformer-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | — | 0 / 0 / 0 | 0.0 → 0.0 | 0.0 → 0.0 | 76.4% → 0.2% |

The Exp A cell above changes two things at once when the pre-registered selection picks a different optimizer. Like-for-like (same optimizer, targeted→Benign, p75, Exp B cells), relaxed vs capability-aware:

| Optimizer (relaxed → new) | Objective | Dataset | Victim | Relaxed Valid ASR | Relaxed, post-hoc filtered | Capability-aware Valid ASR | Δ new − relaxed (pp) | Relative Δ | Timing recovery vs filter (successes, 42/2024/2026) | Timing-only valid: relaxed → new (seed mean) | Padding-based valid: relaxed → new | Padding headroom: relaxed → new |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Hybrid Search | targeted | CICIDS2017 | mlp | 11.06% ± 0.00% | 0.15% ± 0.02% | 4.09% ± 0.00% | -6.97 | -63.0% | 126 / 127 / 126 | 4.7 → 131.0 | 349.3 → 0.0 | 75.1% → 0.0% |
| Hybrid Search | targeted | CICIDS2017 | cnn | 36.15% ± 0.02% | 2.14% ± 0.28% | 13.25% ± 0.00% | -22.90 | -63.3% | 355 / 365 / 347 | 68.3 → 424.0 | 1088.3 → 0.0 | 75.0% → 0.0% |
| Hybrid Search | targeted | CICIDS2017 | ft_transformer | 0.44% ± 0.00% | 0.03% ± 0.00% | 0.12% ± 0.00% | -0.31 | -71.4% | 3 / 3 / 3 | 1.0 → 4.0 | 13.0 → 0.0 | 75.1% → 0.0% |
| Hybrid Search | targeted | CICIDS2018 | mlp-s42 | 0.29% ± 0.04% | 0.29% ± 0.04% | 0.78% ± 0.00% | +0.49 | +167.9% | 15 / 15 / 17 | 9.3 → 25.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Hybrid Search | targeted | CICIDS2018 | cnn-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | — | 0 / 0 / 0 | 0.0 → 0.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Hybrid Search | targeted | CICIDS2018 | ft_transformer-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | — | 0 / 0 / 0 | 0.0 → 0.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Prim-PGD | targeted | CICIDS2017 | mlp | 11.06% ± 0.00% | 0.18% ± 0.02% | 4.09% ± 0.00% | -6.97 | -63.0% | 126 / 125 / 125 | 5.7 → 131.0 | 348.3 → 0.0 | 75.1% → 0.0% |
| Prim-PGD | targeted | CICIDS2017 | cnn | 36.10% ± 0.13% | 3.36% ± 0.17% | 13.25% ± 0.00% | -22.85 | -63.3% | 322 / 316 / 311 | 107.7 → 424.0 | 1047.7 → 0.0 | 75.0% → 0.0% |
| Prim-PGD | targeted | CICIDS2017 | ft_transformer | 0.44% ± 0.00% | 0.03% ± 0.00% | 0.12% ± 0.00% | -0.31 | -71.4% | 3 / 3 / 3 | 1.0 → 4.0 | 13.0 → 0.0 | 75.1% → 0.0% |
| Prim-PGD | targeted | CICIDS2018 | mlp-s42 | 0.26% ± 0.02% | 0.26% ± 0.02% | 0.78% ± 0.00% | +0.52 | +200.0% | 16 / 17 / 17 | 8.3 → 25.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Prim-PGD | targeted | CICIDS2018 | cnn-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | — | 0 / 0 / 0 | 0.0 → 0.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Prim-PGD | targeted | CICIDS2018 | ft_transformer-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | — | 0 / 0 / 0 | 0.0 → 0.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Prim-C&W | targeted | CICIDS2017 | mlp | 11.00% ± 0.00% | 3.25% ± 0.00% | 4.09% ± 0.00% | -6.91 | -62.8% | 27 / 27 / 27 | 104.0 → 131.0 | 248.0 → 0.0 | 75.1% → 0.0% |
| Prim-C&W | targeted | CICIDS2017 | cnn | 14.31% ± 0.00% | 3.97% ± 0.00% | 4.00% ± 0.00% | -10.31 | -72.1% | 1 / 1 / 1 | 127.0 → 128.0 | 331.0 → 0.0 | 75.0% → 0.0% |
| Prim-C&W | targeted | CICIDS2017 | ft_transformer | 0.44% ± 0.00% | 0.06% ± 0.00% | 0.12% ± 0.00% | -0.31 | -71.4% | 2 / 2 / 2 | 2.0 → 4.0 | 12.0 → 0.0 | 75.1% → 0.0% |
| Prim-C&W | targeted | CICIDS2018 | mlp-s42 | 0.72% ± 0.00% | 0.72% ± 0.00% | 0.75% ± 0.00% | +0.03 | +4.3% | 1 / 1 / 1 | 23.0 → 24.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Prim-C&W | targeted | CICIDS2018 | cnn-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | — | 0 / 0 / 0 | 0.0 → 0.0 | 0.0 → 0.0 | 76.4% → 0.2% |
| Prim-C&W | targeted | CICIDS2018 | ft_transformer-s42 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | — | 0 / 0 / 0 | 0.0 → 0.0 | 0.0 → 0.0 | 76.4% → 0.2% |

### Impact of the empty-forward-packet validator rule on every Exp A attack

`Valid ASR without rule` recomputes validator_v2 with `PROTO_0080` removed on the same stored final flows.

| Dataset | Victim | Attack | Raw successes (seed mean) | …filling an empty fwd packet | Valid ASR with rule | Valid ASR without rule | Δ (pp, seed mean) | Valid successes lost only to the rule (Σ seeds) |
|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | PrimAttack (Prim-PGD, p75) | 131.0 | 0.0 | 4.09% ± 0.00% | 4.09% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | cnn | PrimAttack (Prim-PGD, p75) | 431.0 | 0.0 | 13.47% ± 0.00% | 13.47% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | ft_transformer | PrimAttack (Prim-PGD, p75) | 4.0 | 0.0 | 0.12% ± 0.00% | 0.12% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | mlp-s42 | PrimAttack (Prim-PGD, p75) | 81.0 | 0.0 | 2.53% ± 0.00% | 2.53% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | cnn-s42 | PrimAttack (Prim-PGD, p75) | 37.0 | 0.0 | 1.16% ± 0.00% | 1.16% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | ft_transformer-s42 | PrimAttack (Prim-PGD, p75) | 0.0 | 0.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | mlp | PGD | 3200.0 | 3160.7 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | cnn | PGD | 3076.0 | 1971.7 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | ft_transformer | PGD | 3115.7 | 2251.7 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | mlp-s42 | PGD | 3019.0 | 586.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | cnn-s42 | PGD | 3190.3 | 2194.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | ft_transformer-s42 | PGD | 2934.3 | 1481.7 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | mlp | C&W | 3198.0 | 3179.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | cnn | C&W | 3057.0 | 2200.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | ft_transformer | C&W | 2469.0 | 925.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | mlp-s42 | C&W | 2804.0 | 883.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | cnn-s42 | C&W | 3173.0 | 1905.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | ft_transformer-s42 | C&W | 1715.0 | 834.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | mlp | CAPGD-PrimSupport | 3021.0 | 453.7 | 2.01% ± 0.10% | 2.18% ± 0.08% | -0.167 | 16 |
| CICIDS2017 | cnn | CAPGD-PrimSupport | 3089.0 | 341.3 | 5.15% ± 0.07% | 5.21% ± 0.07% | -0.062 | 6 |
| CICIDS2017 | ft_transformer | CAPGD-PrimSupport | 1670.7 | 287.7 | 0.18% ± 0.02% | 0.18% ± 0.02% | +0.000 | 0 |
| CICIDS2018 | mlp-s42 | CAPGD-PrimSupport | 2930.3 | 460.3 | 0.14% ± 0.02% | 0.14% ± 0.02% | +0.000 | 0 |
| CICIDS2018 | cnn-s42 | CAPGD-PrimSupport | 2432.3 | 610.3 | 0.29% ± 0.07% | 0.29% ± 0.07% | +0.000 | 0 |
| CICIDS2018 | ft_transformer-s42 | CAPGD-PrimSupport | 311.7 | 175.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | mlp | C-PGD-PrimSupport | 1625.7 | 789.3 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | cnn | C-PGD-PrimSupport | 1933.3 | 1007.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | ft_transformer | C-PGD-PrimSupport | 691.7 | 333.7 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | mlp-s42 | C-PGD-PrimSupport | 904.0 | 416.7 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | cnn-s42 | C-PGD-PrimSupport | 1617.3 | 812.0 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2018 | ft_transformer-s42 | C-PGD-PrimSupport | 52.7 | 44.7 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.000 | 0 |
| CICIDS2017 | mlp | CAPGD (native) | 3028.7 | 447.3 | 9.53% ± 0.85% | 10.90% ± 0.76% | -1.365 | 131 |
| CICIDS2017 | cnn | CAPGD (native) | 3080.0 | 346.3 | 19.24% ± 0.13% | 21.95% ± 0.19% | -2.708 | 260 |
| CICIDS2017 | ft_transformer | CAPGD (native) | 1619.7 | 265.3 | 5.71% ± 1.07% | 6.01% ± 1.02% | -0.302 | 29 |
| CICIDS2018 | mlp-s42 | CAPGD (native) | 2573.3 | 561.3 | 28.42% ± 1.13% | 28.42% ± 1.13% | +0.000 | 0 |
| CICIDS2018 | cnn-s42 | CAPGD (native) | 2126.0 | 576.0 | 16.30% ± 3.22% | 16.30% ± 3.22% | +0.000 | 0 |
| CICIDS2018 | ft_transformer-s42 | CAPGD (native) | 190.7 | 119.3 | 0.45% ± 0.45% | 0.45% ± 0.45% | +0.000 | 0 |

### CAPGD-PrimSupport vs capability-aware PrimAttack

Identical source flows, victims, seeds, validator and metrics; matched 23-feature downstream support. Different parameterization: CAPGD-PrimSupport optimizes the allowed feature values directly; PrimAttack optimizes primitives and reaches features only through deterministic recomputation under capability restrictions. The comparison quantifies the cost of the primitive-domain parameterization; PrimAttack is not expected to win. Test = the planned Exp A McNemar (Holm over 4) at seed 42.

| Dataset | Victim | n/seed | PrimAttack Valid ASR | CAPGD-PrimSupport Valid ASR | Δ PrimAttack − CAPGD (pp) | Δ per seed (pp) | Discordant (Prim-only / CAPGD-only, seed 42) | Holm p | Higher |
|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 3200 | 4.09% ± 0.00% | 2.01% ± 0.10% | +2.08 | +2.06 / +2.19 / +2.00 | 125 / 59 | 1.65e-06 | PrimAttack |
| CICIDS2017 | cnn | 3200 | 13.47% ± 0.00% | 5.15% ± 0.07% | +8.32 | +8.28 / +8.28 / +8.41 | 364 / 99 | 1.33e-34 | PrimAttack |
| CICIDS2017 | ft_transformer | 3200 | 0.12% ± 0.00% | 0.18% ± 0.02% | -0.05 | -0.03 / -0.06 / -0.06 | 4 / 5 | 1 | CAPGD-PrimSupport |
| CICIDS2018 | mlp-s42 | 3200 | 2.53% ± 0.00% | 0.14% ± 0.02% | +2.40 | +2.41 / +2.41 / +2.38 | 80 / 3 | 7.3e-17 | PrimAttack |
| CICIDS2018 | cnn-s42 | 3200 | 1.16% ± 0.00% | 0.29% ± 0.07% | +0.86 | +0.94 / +0.84 / +0.81 | 37 / 7 | 1.23e-05 | PrimAttack |
| CICIDS2018 | ft_transformer-s42 | 3200 | 0.00% ± 0.00% | 0.00% ± 0.00% | +0.00 | +0.00 / +0.00 / +0.00 | — | — | tie |

## Statistical analysis (Valid Success)

Paired unit = one source flow. Inference uses the pre-specified reference seed 42 only (one outcome per flow, n = attempted flows of one victim, classes pooled within the victim), so the three seeded runs of a flow are never treated as independent observations. Seeds 2024/2026 contribute mean ± SD and a descriptive per-seed paired difference (columns `diff_pp_seed2024/2026` in `statistical_tests.csv`, no p-values). McNemar: exact binomial if discordant pairs < 25, else continuity-corrected χ² (statistic shown). α = 0.05. Holm correction only within the planned family of one experiment and one (dataset, victim).

Cochran's Q across the five paired attacks per (dataset, victim). Only if it is significant: the four planned McNemar comparisons PrimAttack vs each baseline, Holm-corrected over those four. A = PrimAttack, B = baseline, Δ = Valid ASR(A) − Valid ASR(B) in pp at seed 42.

| Dataset | Victim | Family | Test | Comparison | n | A-only | B-only | Δ (pp) | Variant | Statistic | p | Holm p | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Prim-PGD, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 354.99 | 1.46e-75 | — | Valid success differs among the 5 paired conditions (Q = 355.0, p = 1.46e-75); planned McNemar tests follow. |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs PGD | 3200 | 131 | 0 | +4.09 | χ² (cc) | 129.01 | 6.76e-30 | 2.7e-29 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than PGD by 4.09 pp (131 vs 0 discordant flows; Holm-adjusted p = 2.7e-29). |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C&W | 3200 | 131 | 0 | +4.09 | χ² (cc) | 129.01 | 6.76e-30 | 2.7e-29 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C&W by 4.09 pp (131 vs 0 discordant flows; Holm-adjusted p = 2.7e-29). |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs CAPGD-PrimSupport | 3200 | 125 | 59 | +2.06 | χ² (cc) | 22.96 | 1.65e-06 | 1.65e-06 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than CAPGD-PrimSupport by 2.06 pp (125 vs 59 discordant flows; Holm-adjusted p = 1.65e-06). |
| CICIDS2017 | mlp | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C-PGD-PrimSupport | 3200 | 131 | 0 | +4.09 | χ² (cc) | 129.01 | 6.76e-30 | 2.7e-29 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C-PGD-PrimSupport by 4.09 pp (131 vs 0 discordant flows; Holm-adjusted p = 2.7e-29). |
| CICIDS2017 | cnn | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Prim-PGD, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 1260.29 | 1.35e-271 | — | Valid success differs among the 5 paired conditions (Q = 1260.3, p = 1.35e-271); planned McNemar tests follow. |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs PGD | 3200 | 431 | 0 | +13.47 | χ² (cc) | 429.00 | 2.68e-95 | 1.07e-94 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than PGD by 13.47 pp (431 vs 0 discordant flows; Holm-adjusted p = 1.07e-94). |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C&W | 3200 | 431 | 0 | +13.47 | χ² (cc) | 429.00 | 2.68e-95 | 1.07e-94 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C&W by 13.47 pp (431 vs 0 discordant flows; Holm-adjusted p = 1.07e-94). |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs CAPGD-PrimSupport | 3200 | 364 | 99 | +8.28 | χ² (cc) | 150.53 | 1.33e-34 | 1.33e-34 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than CAPGD-PrimSupport by 8.28 pp (364 vs 99 discordant flows; Holm-adjusted p = 1.33e-34). |
| CICIDS2017 | cnn | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C-PGD-PrimSupport | 3200 | 431 | 0 | +13.47 | χ² (cc) | 429.00 | 2.68e-95 | 1.07e-94 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C-PGD-PrimSupport by 13.47 pp (431 vs 0 discordant flows; Holm-adjusted p = 1.07e-94). |
| CICIDS2017 | ft_transformer | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Prim-PGD, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 13.78 | 0.00804 | — | Valid success differs among the 5 paired conditions (Q = 13.8, p = 0.00804); planned McNemar tests follow. |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs PGD | 3200 | 4 | 0 | +0.12 | exact binomial |  | 0.125 | 0.5 | No significant difference (Holm-adjusted p = 0.5; Δ = +0.12 pp, 4 vs 0 discordant flows). |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C&W | 3200 | 4 | 0 | +0.12 | exact binomial |  | 0.125 | 0.5 | No significant difference (Holm-adjusted p = 0.5; Δ = +0.12 pp, 4 vs 0 discordant flows). |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs CAPGD-PrimSupport | 3200 | 4 | 5 | -0.03 | exact binomial |  | 1 | 1 | No significant difference (Holm-adjusted p = 1; Δ = -0.03 pp, 4 vs 5 discordant flows). |
| CICIDS2017 | ft_transformer | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C-PGD-PrimSupport | 3200 | 4 | 0 | +0.12 | exact binomial |  | 0.125 | 0.5 | No significant difference (Holm-adjusted p = 0.5; Δ = +0.12 pp, 4 vs 0 discordant flows). |
| CICIDS2018 | mlp-s42 | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Prim-PGD, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 303.67 | 1.75e-64 | — | Valid success differs among the 5 paired conditions (Q = 303.7, p = 1.75e-64); planned McNemar tests follow. |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs PGD | 3200 | 81 | 0 | +2.53 | χ² (cc) | 79.01 | 6.17e-19 | 2.47e-18 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than PGD by 2.53 pp (81 vs 0 discordant flows; Holm-adjusted p = 2.47e-18). |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C&W | 3200 | 81 | 0 | +2.53 | χ² (cc) | 79.01 | 6.17e-19 | 2.47e-18 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C&W by 2.53 pp (81 vs 0 discordant flows; Holm-adjusted p = 2.47e-18). |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs CAPGD-PrimSupport | 3200 | 80 | 3 | +2.41 | χ² (cc) | 69.59 | 7.3e-17 | 7.3e-17 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than CAPGD-PrimSupport by 2.41 pp (80 vs 3 discordant flows; Holm-adjusted p = 7.3e-17). |
| CICIDS2018 | mlp-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C-PGD-PrimSupport | 3200 | 81 | 0 | +2.53 | χ² (cc) | 79.01 | 6.17e-19 | 2.47e-18 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C-PGD-PrimSupport by 2.53 pp (81 vs 0 discordant flows; Holm-adjusted p = 2.47e-18). |
| CICIDS2018 | cnn-s42 | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Prim-PGD, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 117.14 | 2.18e-24 | — | Valid success differs among the 5 paired conditions (Q = 117.1, p = 2.18e-24); planned McNemar tests follow. |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs PGD | 3200 | 37 | 0 | +1.16 | χ² (cc) | 35.03 | 3.25e-09 | 1.3e-08 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than PGD by 1.16 pp (37 vs 0 discordant flows; Holm-adjusted p = 1.3e-08). |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C&W | 3200 | 37 | 0 | +1.16 | χ² (cc) | 35.03 | 3.25e-09 | 1.3e-08 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C&W by 1.16 pp (37 vs 0 discordant flows; Holm-adjusted p = 1.3e-08). |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs CAPGD-PrimSupport | 3200 | 37 | 7 | +0.94 | χ² (cc) | 19.11 | 1.23e-05 | 1.23e-05 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than CAPGD-PrimSupport by 0.94 pp (37 vs 7 discordant flows; Holm-adjusted p = 1.23e-05). |
| CICIDS2018 | cnn-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C-PGD-PrimSupport | 3200 | 37 | 0 | +1.16 | χ² (cc) | 35.03 | 3.25e-09 | 1.3e-08 | PrimAttack (Prim-PGD, p75) has higher Valid ASR than C-PGD-PrimSupport by 1.16 pp (37 vs 0 discordant flows; Holm-adjusted p = 1.3e-08). |
| CICIDS2018 | ft_transformer-s42 | A: 5 untargeted attacks | Cochran's Q | PrimAttack (Prim-PGD, p75) / PGD / C&W / CAPGD-PrimSupport / C-PGD-PrimSupport | 3200 |  |  |  | Cochran χ² | 0.00 | 1 | — | No evidence that valid success differs among the 5 conditions (Q = 0.00, p = 1); planned McNemar tests not performed. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs PGD |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C&W |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs CAPGD-PrimSupport |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |
| CICIDS2018 | ft_transformer-s42 | A: PrimAttack vs baselines (Holm over 4) | McNemar | PrimAttack (Prim-PGD, p75) vs C-PGD-PrimSupport |  |  |  |  |  |  | — | — | Not performed: the omnibus Cochran's Q was not significant. |

## Class-wise results

| Dataset | Victim | Class | Attack | n/seed | Raw ASR | Valid ASR | Gap |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | DoS | PrimAttack (Prim-PGD, p75) | 800 | 5.12% ± 0.00% | 5.12% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | DDoS | PrimAttack (Prim-PGD, p75) | 800 | 10.75% ± 0.00% | 10.75% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | Recon | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | mlp | BruteForce | PrimAttack (Prim-PGD, p75) | 800 | 0.50% ± 0.00% | 0.50% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | DoS | PrimAttack (Prim-PGD, p75) | 800 | 20.50% ± 0.00% | 20.50% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | DDoS | PrimAttack (Prim-PGD, p75) | 800 | 32.12% ± 0.00% | 32.12% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | Recon | PrimAttack (Prim-PGD, p75) | 800 | 0.38% ± 0.00% | 0.38% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | cnn | BruteForce | PrimAttack (Prim-PGD, p75) | 800 | 0.88% ± 0.00% | 0.88% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | DoS | PrimAttack (Prim-PGD, p75) | 800 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | DDoS | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | Recon | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | ft_transformer | BruteForce | PrimAttack (Prim-PGD, p75) | 800 | 0.38% ± 0.00% | 0.38% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | DoS | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | DDoS | PrimAttack (Prim-PGD, p75) | 800 | 7.50% ± 0.00% | 7.50% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | Recon | PrimAttack (Prim-PGD, p75) | 800 | 2.62% ± 0.00% | 2.62% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | mlp-s42 | BruteForce | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | DoS | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | DDoS | PrimAttack (Prim-PGD, p75) | 800 | 4.62% ± 0.00% | 4.62% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | Recon | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | cnn-s42 | BruteForce | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | DoS | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | DDoS | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | Recon | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | BruteForce | PrimAttack (Prim-PGD, p75) | 800 | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
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
| CICIDS2017 | mlp | DDoS | CAPGD-PrimSupport | 800 | 100.00% ± 0.00% | 6.67% ± 0.36% | 93.33 ± 0.36 pp |
| CICIDS2017 | mlp | Recon | CAPGD-PrimSupport | 800 | 100.00% ± 0.00% | 0.21% ± 0.19% | 99.79 ± 0.19 pp |
| CICIDS2017 | mlp | BruteForce | CAPGD-PrimSupport | 800 | 83.33% ± 2.57% | 0.00% ± 0.00% | 83.33 ± 2.57 pp |
| CICIDS2017 | cnn | DoS | CAPGD-PrimSupport | 800 | 95.17% ± 2.46% | 6.00% ± 0.22% | 89.17 ± 2.24 pp |
| CICIDS2017 | cnn | DDoS | CAPGD-PrimSupport | 800 | 98.33% ± 0.63% | 14.25% ± 0.22% | 84.08 ± 0.64 pp |
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
| CICIDS2017 | mlp | DoS | CAPGD (native) | 800 | 93.54% ± 0.51% | 4.04% ± 0.47% | 89.50 ± 0.90 pp |
| CICIDS2017 | mlp | DDoS | CAPGD (native) | 800 | 100.00% ± 0.00% | 30.29% ± 3.51% | 69.71 ± 3.51 pp |
| CICIDS2017 | mlp | Recon | CAPGD (native) | 800 | 100.00% ± 0.00% | 3.29% ± 0.72% | 96.71 ± 0.72 pp |
| CICIDS2017 | mlp | BruteForce | CAPGD (native) | 800 | 85.04% ± 3.20% | 0.50% ± 0.54% | 84.54 ± 3.63 pp |
| CICIDS2017 | cnn | DoS | CAPGD (native) | 800 | 95.58% ± 1.70% | 18.29% ± 1.44% | 77.29 ± 0.83 pp |
| CICIDS2017 | cnn | DDoS | CAPGD (native) | 800 | 97.96% ± 1.16% | 36.62% ± 1.87% | 61.33 ± 2.90 pp |
| CICIDS2017 | cnn | Recon | CAPGD (native) | 800 | 93.71% ± 1.00% | 11.21% ± 1.45% | 82.50 ± 1.75 pp |
| CICIDS2017 | cnn | BruteForce | CAPGD (native) | 800 | 97.75% ± 1.82% | 10.83% ± 1.13% | 86.92 ± 2.53 pp |
| CICIDS2017 | ft_transformer | DoS | CAPGD (native) | 800 | 38.67% ± 6.03% | 15.88% ± 3.94% | 22.79 ± 4.19 pp |
| CICIDS2017 | ft_transformer | DDoS | CAPGD (native) | 800 | 54.21% ± 2.60% | 1.50% ± 0.78% | 52.71 ± 3.07 pp |
| CICIDS2017 | ft_transformer | Recon | CAPGD (native) | 800 | 98.92% ± 0.36% | 5.25% ± 2.21% | 93.67 ± 2.50 pp |
| CICIDS2017 | ft_transformer | BruteForce | CAPGD (native) | 800 | 10.67% ± 4.96% | 0.21% ± 0.07% | 10.46 ± 4.90 pp |
| CICIDS2018 | mlp-s42 | DoS | CAPGD (native) | 800 | 88.71% ± 1.39% | 64.04% ± 0.94% | 24.67 ± 2.32 pp |
| CICIDS2018 | mlp-s42 | DDoS | CAPGD (native) | 800 | 92.71% ± 1.84% | 28.67% ± 2.22% | 64.04 ± 0.40 pp |
| CICIDS2018 | mlp-s42 | Recon | CAPGD (native) | 800 | 96.12% ± 0.66% | 17.50% ± 0.33% | 78.62 ± 0.37 pp |
| CICIDS2018 | mlp-s42 | BruteForce | CAPGD (native) | 800 | 44.12% ± 1.74% | 3.46% ± 3.12% | 40.67 ± 4.85 pp |
| CICIDS2018 | cnn-s42 | DoS | CAPGD (native) | 800 | 34.75% ± 2.17% | 2.79% ± 0.14% | 31.96 ± 2.31 pp |
| CICIDS2018 | cnn-s42 | DDoS | CAPGD (native) | 800 | 76.83% ± 1.40% | 14.42% ± 1.53% | 62.42 ± 1.45 pp |
| CICIDS2018 | cnn-s42 | Recon | CAPGD (native) | 800 | 59.21% ± 3.57% | 12.42% ± 2.27% | 46.79 ± 1.42 pp |
| CICIDS2018 | cnn-s42 | BruteForce | CAPGD (native) | 800 | 94.96% ± 0.52% | 35.58% ± 9.78% | 59.38 ± 9.66 pp |
| CICIDS2018 | ft_transformer-s42 | DoS | CAPGD (native) | 800 | 0.04% ± 0.07% | 0.00% ± 0.00% | 0.04 ± 0.07 pp |
| CICIDS2018 | ft_transformer-s42 | DDoS | CAPGD (native) | 800 | 0.12% ± 0.00% | 0.12% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | ft_transformer-s42 | Recon | CAPGD (native) | 800 | 1.75% ± 0.33% | 0.62% ± 0.12% | 1.12 ± 0.33 pp |
| CICIDS2018 | ft_transformer-s42 | BruteForce | CAPGD (native) | 800 | 21.92% ± 1.87% | 1.04% ± 1.70% | 20.88 ± 3.28 pp |

## Plots

- `plots/A1_raw_asr_by_attack.png` — Raw ASR by attack
- `plots/A2_valid_asr_by_attack.png` — Valid ASR by attack
- `plots/A3_classwise_valid_asr.png` — class-wise Valid ASR
- `plots/A4_modelwise_valid_asr.png` — model-wise Valid ASR
- `plots/A5_validity_gap_by_attack.png` — Validity Gap by attack
- `plots/A6_primattack_p75_vs_unbounded.png` — PrimAttack p75 vs unbounded
- `plots/A7_primitive_ablation.png` — PrimAttack joint vs timing-only vs padding-only
- `plots/A8_relaxed_vs_capability_aware.png` — relaxed-padding vs capability-aware PrimAttack

## Machine-readable outputs

`per_sample.parquet` (every flow × seed × attack, incl. clean/adversarial prediction, raw success, validator pass, valid success, evaluations, primitive controls, attack parameters), `seed_level.csv`, `table_level.csv`, `statistical_tests.csv`, `constrained_baseline_diagnostics.csv`, `primattack_primitive_costs.csv`; `capability_fix/{eligibility, primattack_breakdown, primitive_ablation, relaxed_vs_capability_aware, validator_rule_impact, capgd_primsupport_fairness}.csv`.

## Interpretation

Numbers are seed means (n = 3,200 flows per victim and seed). Tests are McNemar on seed-42
Valid Success, Holm over the four PrimAttack-vs-baseline comparisons. PrimAttack is the
capability-aware version (amendment A2): padding only for flows without a zero-length forward
packet, every other flow timing-only.

**1. Unrestricted feature-space attacks: high raw success, no valid success.** PGD reaches a Raw
ASR of 91.70–100.00% and C&W 53.59–99.94% on all six victims. validator_v2 rejects every one of
these flows (Valid ASR 0.00% in all 12 cells), so the Validity Gap equals the Raw ASR. Moving all
79 scaled features independently leaves the feature domain (SCHEMA), contradicts CICFlowMeter's
aggregate identities (EXTRACTOR) and breaks protocol rules (PROTOCOL): 100% of the seed-42
invalid examples fail each of these categories (Exp E).

**2. Public constrained attacks restricted to PrimAttack's 23 downstream coordinates.**
CAPGD-PrimSupport reaches a Raw ASR of 9.74–96.53% and C-PGD-PrimSupport 1.65–60.42%, with no
feature changed outside the mask (asserted). Almost none of it is valid. C-PGD has a Valid ASR of
0.00% on every victim. CAPGD reaches 2.01% / 5.15% / 0.18% on CICIDS2017 MLP / CNN /
FT-Transformer and 0.14% / 0.29% / 0.00% on CICIDS2018. Their invalid examples mostly break
EXTRACTOR identities (67–100%) and MINED invariants (84–100%). Changing the 23 coordinates
independently decouples features that one packet-level change moves together. The new
empty-forward-packet rule `PROTO_0080` removes only 16 (MLP) and 6 (CNN) of CAPGD's CICIDS2017
valid successes over three seeds (−0.17 / −0.06 pp) and none elsewhere.

**3. Capability-aware PrimAttack is a timing attack on almost every attack flow.** 99.97%
(CICIDS2017) and 99.63% (CICIDS2018) of the attacked flows contain a zero-length forward packet,
so only 1 of 3,200 (CICIDS2017) and 12 of 3,200 (CICIDS2018) flows per victim may be padded. At
p75, 73.8–74.7% of the flows are searched timing-only and 25.0–26.2% have no primitive at all
(mostly Recon: single-packet or zero-IAT probes). Every valid success uses timing only
(p = 0, delay > 0); none fills an empty packet (asserted). PrimAttack (Prim-PGD, p75, untargeted)
reaches a Valid ASR of 4.09% / 13.47% / 0.12% on CICIDS2017 and 2.53% / 1.16% / 0.00% on
CICIDS2018, with a Validity Gap of 0.00 pp everywhere. Its successes come from DoS and DDoS (and
21 CICIDS2018 Recon flows on MLP); BruteForce and Recon barely move.

**4. Inference.** Cochran's Q is significant on five victims (not on CICIDS2018 FT-Transformer,
where none of the five attacks has a valid success). PrimAttack has a higher Valid ASR than PGD, C&W and C-PGD on
CICIDS2017 MLP / CNN and CICIDS2018 MLP / CNN (Holm p ≤ 1.3e-8) and than CAPGD-PrimSupport on the
same four victims: +2.06, +8.28, +2.41 and +0.94 pp at seed 42 (364 vs 99 discordant flows on the
CICIDS2017 CNN; Holm p ≤ 1.3e-5). On CICIDS2017 FT-Transformer none of the four comparisons is
significant (4 PrimAttack-only flows; vs CAPGD Δ = −0.03 pp, Holm p = 1).

**5. What the capability fix changed.** The relaxed pre-fix PrimAttack reached 11.06% / 36.67% /
0.50% on CICIDS2017, almost all of it by padding empty packets. Re-judging those flows with the
new rule leaves 0.15% / 2.31% / 0.03%. The fresh timing-focused re-run recovers 126–127, 348–366
and 3 valid successes per seed above that filter and lands at 4.09% / 13.47% / 0.12%: 63–75% below
the relaxed result, but 27×, 5.8× and 4× the post-hoc lower bound. On CICIDS2018 the fix raises
PrimAttack from 0.19% / 0.03% to 2.53% / 1.16%. There the relaxed search spent its evaluations on
padding that `MINED_0001` always rejected; now the whole budget goes to timing. The selected
optimizer also changed (Hybrid → Prim-PGD, a tie broken by evaluations; Exp B). The like-for-like
targeted comparison with a fixed optimizer shows the same direction (e.g. Hybrid on CICIDS2018
MLP 0.29% → 0.78%).

**6. Descriptive rows.** Native CAPGD (†, its own 16-feature configuration mask, not the PrimAttack
support) reaches a Valid ASR of 9.53% / 19.24% / 5.71% (CICIDS2017) and 28.42% / 16.30% / 0.45%
(CICIDS2018): higher than every inferential attack on every victim. It is not a matched-support
comparison. `PROTO_0080` removes 131 / 260 / 29 of its CICIDS2017 valid successes over three seeds
(−1.37 / −2.71 / −0.30 pp). Unbounded PrimAttack reaches 22.97% / 59.94% / 0.59% and 44.32% /
26.28% / 0.12%, so the p75 timing box (median per-flow delay cap about 0.9 s on CICIDS2017 and
31 ms on CICIDS2018), not the search, limits p75 success.

**How to read these differences.** The highest Raw ASR (PGD) belongs to the least constrained
threat model, and none of its evasions is a valid flow. PrimAttack's search includes validator_v2
in its success predicate; the baselines never query it, and C-PGD sees only a differentiable
subset. PrimAttack's advantage over CAPGD-PrimSupport therefore reflects the primitive
parameterization together with this validity-aware search, and it is limited to the victims
where timing alone moves the decision. Matched support does not give CAPGD/C-PGD packet-level
realizability, and PrimAttack's results are flow-level proxies (no PCAP is modified). Valid
evasion is strongly victim-dependent: FT-Transformer resists every validity-preserving attack
(≤ 0.18% Valid ASR at p75 across the inferential attacks).
