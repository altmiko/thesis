# Final Attack-Performance Results

## Protocol and provenance

| Item | Executed configuration |
|---|---|
| Datasets | CICIDS2017-DistriNet and CSE-CIC-IDS-2018-DistriNet. |
| Victims | MLP, CNN, and FT-Transformer category heads; CICIDS2018 uses the frozen training-seed-42 checkpoints. |
| Seeds | 42, 2024, 2026. Each value is the sample mean ± sample SD (ddof = 1) across these three seed-level rates. |
| Source rows | For each dataset, victim, class, and seed, `run_full_adversarial_eval.py --selection random --selection-seed <seed>` permuted all class rows with NumPy RNG seed `seed + class_id`, retained clean-correct rows, selected 800, and restored test order. Draws are independent by seed but can overlap. All attacks compared within one seed reuse that seed’s exact selection. |
| Exact n | 800 clean-correct flows per class, victim, and seed; 3,200 per victim and seed after pooling the four classes. Thus each class-level mean uses three rates of n = 800 (2,400 attacked instances), and each victim-level mean uses three rates of n = 3,200 (9,600 attacked instances). No row was dropped after selection. |
| Source classes | DoS, DDoS, Recon, and BruteForce. |
| Untargeted success | The final prediction differs from the original malicious class. |
| Targeted success | The final prediction is Benign (class 0). |
| Validator | validator_v2 `hybrid_valid = SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED`, including source-conditioned `PROTO_0080`, recomputed from each stored final `adv_raw` and its selected clean source flow. `Valid success = raw success ∧ hybrid_valid`. |
| Seed-specific selection | Yes. Seed 42 reuses the exact locked FINAL artifacts because their selection seed and attack seed are both 42. Seeds 2024 and 2026 were rerun with selection seeds 2024 and 2026, respectively. |
| Main tables | Untargeted locked FINAL attack roster. PrimAttack uses selected optimizer Prim-PGD, joint primitives, unbounded envelope-only box, and 256 victim evaluations per flow. Baselines retain their locked native budgets. |
| Objective comparison | PrimAttack/Prim-PGD, joint mode, unbounded box, 256 evaluations per flow; only the targeted-versus-untargeted objective changes. |
| Budget sensitivity | Targeted-to-Benign PrimAttack/Prim-PGD, joint mode, p50, p75, and unbounded boxes, 256 evaluations per flow. |

The locked baseline roster and exact method settings are:

| Attack | Space and budget | Optimization |
|---|---|---|
| PrimAttack (Prim-PGD) | Capability-aware primitive controls `(p, delay, shape)`; canonical integer/µs realization and full φ recomputation; joint mode; p50, p75, or unbounded train-envelope box as stated. | 3 restarts × 42 steps, step 0.05, momentum 0.75; 256 victim evaluations/flow; success-first incumbent selection. |
| PGD | All 79 RobustScaler features; L∞ ε = 0.5. | Untargeted CE; 40 steps; α = 0.05; one random start. |
| C&W | All 79 RobustScaler features; unbounded L2-penalized search. | Untargeted margin; ≤60 Adam steps; lr = 0.01; λ = 1; κ = 0; convergence = 1e-5. |
| CAPGD-PrimSupport | Train min-max space; 23-feature PrimAttack support; L2 ε = 0.5. | Untargeted CE; 10 steps; 2 restarts; train box, mask, type, and TabularBench constraint repair. |
| C-PGD-PrimSupport | Train min-max space; same 23-feature support; L2 ε = 0.5. | Untargeted CE − constraint penalty; 40 steps; step 0.05; penalty weight = 1. |
| CAPGD (native) | Train min-max space; native 16-feature configuration mask; L2 ε = 0.5. | Untargeted CE; 10 steps; 2 restarts. Descriptive FINAL row. |

Artifact audit covered 648 NPZ files and 518,400 attacked rows. Sample IDs/order, positional indices, clean-input hashes, checkpoint hashes, objectives, seeds, clean-correct denominators, raw success, validator verdicts, and valid success were checked. validator_v2 was rerun on all 518,400 stored final adversarial flows; mismatches: 0.

## CICIDS2017 Attack Performance

Untargeted performance. PrimAttack uses the unbounded box; every other method uses the locked budget in the configuration table. `n = 3,200` per victim and seed.

| Attack | MLP Raw ASR | MLP Valid ASR | MLP Validity Gap | CNN Raw ASR | CNN Valid ASR | CNN Validity Gap | FT-Transformer Raw ASR | FT-Transformer Valid ASR | FT-Transformer Validity Gap |
|---|---|---|---|---|---|---|---|---|---|
| PrimAttack (Prim-PGD) | 23.01% ± 0.59% | 23.01% ± 0.59% | 0.00 ± 0.00 pp | 59.67% ± 0.42% | 59.67% ± 0.42% | 0.00 ± 0.00 pp | 0.60% ± 0.02% | 0.60% ± 0.02% | 0.00 ± 0.00 pp |
| PGD | 99.99% ± 0.02% | 0.00% ± 0.00% | 99.99 ± 0.02 pp | 96.07% ± 0.16% | 0.00% ± 0.00% | 96.07 ± 0.16 pp | 97.41% ± 0.17% | 0.00% ± 0.00% | 97.41 ± 0.17 pp |
| C&W | 99.95% ± 0.02% | 0.00% ± 0.00% | 99.95 ± 0.02 pp | 95.45% ± 0.08% | 0.00% ± 0.00% | 95.45 ± 0.08 pp | 76.90% ± 0.42% | 0.00% ± 0.00% | 76.90 ± 0.42 pp |
| CAPGD-PrimSupport | 94.41% ± 0.83% | 1.98% ± 0.33% | 92.43 ± 1.15 pp | 96.65% ± 1.00% | 5.06% ± 0.12% | 91.58 ± 0.96 pp | 52.89% ± 3.94% | 0.17% ± 0.02% | 52.72 ± 3.95 pp |
| C-PGD-PrimSupport | 50.76% ± 1.90% | 0.00% ± 0.00% | 50.76 ± 1.90 pp | 59.98% ± 3.38% | 0.00% ± 0.00% | 59.98 ± 3.38 pp | 21.69% ± 0.33% | 0.00% ± 0.00% | 21.69 ± 0.33 pp |
| CAPGD (native) | 94.47% ± 0.82% | 10.02% ± 0.50% | 84.45 ± 0.65 pp | 96.50% ± 1.00% | 19.00% ± 0.19% | 77.50 ± 0.99 pp | 50.76% ± 3.37% | 5.72% ± 1.17% | 45.04 ± 3.78 pp |

## CICIDS2018 Attack Performance

Untargeted performance. PrimAttack uses the unbounded box; every other method uses the locked budget in the configuration table. `n = 3,200` per victim and seed.

| Attack | MLP Raw ASR | MLP Valid ASR | MLP Validity Gap | CNN Raw ASR | CNN Valid ASR | CNN Validity Gap | FT-Transformer Raw ASR | FT-Transformer Valid ASR | FT-Transformer Validity Gap |
|---|---|---|---|---|---|---|---|---|---|
| PrimAttack (Prim-PGD) | 44.44% ± 0.27% | 44.44% ± 0.27% | 0.00 ± 0.00 pp | 26.32% ± 0.04% | 26.32% ± 0.04% | 0.00 ± 0.00 pp | 0.26% ± 0.13% | 0.26% ± 0.13% | 0.00 ± 0.00 pp |
| PGD | 94.45% ± 0.18% | 0.00% ± 0.00% | 94.45 ± 0.18 pp | 99.68% ± 0.10% | 0.00% ± 0.00% | 99.68 ± 0.10 pp | 91.85% ± 0.49% | 0.00% ± 0.00% | 91.85 ± 0.49 pp |
| C&W | 87.60% ± 0.75% | 0.00% ± 0.00% | 87.60 ± 0.75 pp | 99.05% ± 0.18% | 0.00% ± 0.00% | 99.05 ± 0.18 pp | 53.66% ± 0.17% | 0.00% ± 0.00% | 53.66 ± 0.17 pp |
| CAPGD-PrimSupport | 91.78% ± 0.66% | 0.10% ± 0.02% | 91.68 ± 0.64 pp | 76.00% ± 1.70% | 0.36% ± 0.18% | 75.64 ± 1.63 pp | 9.72% ± 0.88% | 0.00% ± 0.00% | 9.72 ± 0.88 pp |
| C-PGD-PrimSupport | 28.35% ± 0.63% | 0.00% ± 0.00% | 28.35 ± 0.63 pp | 50.36% ± 2.68% | 0.00% ± 0.00% | 50.36 ± 2.68 pp | 1.70% ± 0.56% | 0.00% ± 0.00% | 1.70 ± 0.56 pp |
| CAPGD (native) | 80.60% ± 0.56% | 28.50% ± 1.79% | 52.10 ± 1.47 pp | 66.67% ± 2.05% | 16.55% ± 3.91% | 50.11 ± 2.06 pp | 6.24% ± 0.28% | 0.57% ± 0.48% | 5.67 ± 0.69 pp |

## Per-Class Attack Performance

Each row uses `n = 800` source flows per seed (`2,400` attacked instances across three runs).

### CICIDS2017

| Dataset | Victim | Attack | Class | n | Raw ASR | Valid ASR | Validity Gap |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | MLP | PrimAttack (Prim-PGD) | DoS | 800/seed | 60.92% ± 1.94% | 60.92% ± 1.94% | 0.00 ± 0.00 pp |
| CICIDS2017 | MLP | PrimAttack (Prim-PGD) | DDoS | 800/seed | 28.50% ± 1.94% | 28.50% ± 1.94% | 0.00 ± 0.00 pp |
| CICIDS2017 | MLP | PrimAttack (Prim-PGD) | Recon | 800/seed | 0.75% ± 0.38% | 0.75% ± 0.38% | 0.00 ± 0.00 pp |
| CICIDS2017 | MLP | PrimAttack (Prim-PGD) | BruteForce | 800/seed | 1.88% ± 0.12% | 1.88% ± 0.12% | 0.00 ± 0.00 pp |
| CICIDS2017 | MLP | PGD | DoS | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | MLP | PGD | DDoS | 800/seed | 99.96% ± 0.07% | 0.00% ± 0.00% | 99.96 ± 0.07 pp |
| CICIDS2017 | MLP | PGD | Recon | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | MLP | PGD | BruteForce | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | MLP | C&W | DoS | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | MLP | C&W | DDoS | 800/seed | 99.79% ± 0.07% | 0.00% ± 0.00% | 99.79 ± 0.07 pp |
| CICIDS2017 | MLP | C&W | Recon | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | MLP | C&W | BruteForce | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | MLP | CAPGD-PrimSupport | DoS | 800/seed | 94.21% ± 0.47% | 1.21% ± 0.07% | 93.00 ± 0.45 pp |
| CICIDS2017 | MLP | CAPGD-PrimSupport | DDoS | 800/seed | 100.00% ± 0.00% | 6.62% ± 1.27% | 93.38 ± 1.27 pp |
| CICIDS2017 | MLP | CAPGD-PrimSupport | Recon | 800/seed | 99.96% ± 0.07% | 0.08% ± 0.14% | 99.88 ± 0.22 pp |
| CICIDS2017 | MLP | CAPGD-PrimSupport | BruteForce | 800/seed | 83.46% ± 3.25% | 0.00% ± 0.00% | 83.46 ± 3.25 pp |
| CICIDS2017 | MLP | C-PGD-PrimSupport | DoS | 800/seed | 22.04% ± 1.26% | 0.00% ± 0.00% | 22.04 ± 1.26 pp |
| CICIDS2017 | MLP | C-PGD-PrimSupport | DDoS | 800/seed | 82.58% ± 4.08% | 0.00% ± 0.00% | 82.58 ± 4.08 pp |
| CICIDS2017 | MLP | C-PGD-PrimSupport | Recon | 800/seed | 94.75% ± 1.77% | 0.00% ± 0.00% | 94.75 ± 1.77 pp |
| CICIDS2017 | MLP | C-PGD-PrimSupport | BruteForce | 800/seed | 3.67% ± 1.70% | 0.00% ± 0.00% | 3.67 ± 1.70 pp |
| CICIDS2017 | MLP | CAPGD (native) | DoS | 800/seed | 93.71% ± 0.31% | 4.83% ± 1.16% | 88.88 ± 1.44 pp |
| CICIDS2017 | MLP | CAPGD (native) | DDoS | 800/seed | 99.96% ± 0.07% | 31.71% ± 0.95% | 68.25 ± 0.88 pp |
| CICIDS2017 | MLP | CAPGD (native) | Recon | 800/seed | 99.96% ± 0.07% | 2.88% ± 1.25% | 97.08 ± 1.19 pp |
| CICIDS2017 | MLP | CAPGD (native) | BruteForce | 800/seed | 84.25% ± 2.91% | 0.67% ± 0.64% | 83.58 ± 3.43 pp |
| CICIDS2017 | CNN | PrimAttack (Prim-PGD) | DoS | 800/seed | 91.46% ± 0.14% | 91.46% ± 0.14% | 0.00 ± 0.00 pp |
| CICIDS2017 | CNN | PrimAttack (Prim-PGD) | DDoS | 800/seed | 46.54% ± 1.58% | 46.54% ± 1.58% | 0.00 ± 0.00 pp |
| CICIDS2017 | CNN | PrimAttack (Prim-PGD) | Recon | 800/seed | 0.67% ± 0.31% | 0.67% ± 0.31% | 0.00 ± 0.00 pp |
| CICIDS2017 | CNN | PrimAttack (Prim-PGD) | BruteForce | 800/seed | 100.00% ± 0.00% | 100.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2017 | CNN | PGD | DoS | 800/seed | 98.42% ± 0.36% | 0.00% ± 0.00% | 98.42 ± 0.36 pp |
| CICIDS2017 | CNN | PGD | DDoS | 800/seed | 85.88% ± 0.76% | 0.00% ± 0.00% | 85.88 ± 0.76 pp |
| CICIDS2017 | CNN | PGD | Recon | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | CNN | PGD | BruteForce | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | CNN | C&W | DoS | 800/seed | 85.75% ± 0.22% | 0.00% ± 0.00% | 85.75 ± 0.22 pp |
| CICIDS2017 | CNN | C&W | DDoS | 800/seed | 96.04% ± 0.40% | 0.00% ± 0.00% | 96.04 ± 0.40 pp |
| CICIDS2017 | CNN | C&W | Recon | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | CNN | C&W | BruteForce | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | CNN | CAPGD-PrimSupport | DoS | 800/seed | 95.88% ± 1.63% | 6.38% ± 0.33% | 89.50 ± 1.96 pp |
| CICIDS2017 | CNN | CAPGD-PrimSupport | DDoS | 800/seed | 98.21% ± 0.81% | 13.79% ± 0.51% | 84.42 ± 0.47 pp |
| CICIDS2017 | CNN | CAPGD-PrimSupport | Recon | 800/seed | 94.08% ± 1.16% | 0.08% ± 0.14% | 94.00 ± 1.11 pp |
| CICIDS2017 | CNN | CAPGD-PrimSupport | BruteForce | 800/seed | 98.42% ± 1.13% | 0.00% ± 0.00% | 98.42 ± 1.13 pp |
| CICIDS2017 | CNN | C-PGD-PrimSupport | DoS | 800/seed | 48.04% ± 1.53% | 0.00% ± 0.00% | 48.04 ± 1.53 pp |
| CICIDS2017 | CNN | C-PGD-PrimSupport | DDoS | 800/seed | 73.17% ± 4.54% | 0.00% ± 0.00% | 73.17 ± 4.54 pp |
| CICIDS2017 | CNN | C-PGD-PrimSupport | Recon | 800/seed | 50.21% ± 7.18% | 0.00% ± 0.00% | 50.21 ± 7.18 pp |
| CICIDS2017 | CNN | C-PGD-PrimSupport | BruteForce | 800/seed | 68.50% ± 3.75% | 0.00% ± 0.00% | 68.50 ± 3.75 pp |
| CICIDS2017 | CNN | CAPGD (native) | DoS | 800/seed | 95.33% ± 1.31% | 20.62% ± 1.41% | 74.71 ± 2.65 pp |
| CICIDS2017 | CNN | CAPGD (native) | DDoS | 800/seed | 98.21% ± 0.95% | 34.25% ± 1.95% | 63.96 ± 2.06 pp |
| CICIDS2017 | CNN | CAPGD (native) | Recon | 800/seed | 94.50% ± 0.43% | 10.29% ± 0.19% | 84.21 ± 0.44 pp |
| CICIDS2017 | CNN | CAPGD (native) | BruteForce | 800/seed | 97.96% ± 1.71% | 10.83% ± 1.13% | 87.13 ± 2.49 pp |
| CICIDS2017 | FT-Transformer | PrimAttack (Prim-PGD) | DoS | 800/seed | 0.08% ± 0.07% | 0.08% ± 0.07% | 0.00 ± 0.00 pp |
| CICIDS2017 | FT-Transformer | PrimAttack (Prim-PGD) | DDoS | 800/seed | 0.08% ± 0.07% | 0.08% ± 0.07% | 0.00 ± 0.00 pp |
| CICIDS2017 | FT-Transformer | PrimAttack (Prim-PGD) | Recon | 800/seed | 0.54% ± 0.26% | 0.54% ± 0.26% | 0.00 ± 0.00 pp |
| CICIDS2017 | FT-Transformer | PrimAttack (Prim-PGD) | BruteForce | 800/seed | 1.71% ± 0.26% | 1.71% ± 0.26% | 0.00 ± 0.00 pp |
| CICIDS2017 | FT-Transformer | PGD | DoS | 800/seed | 98.62% ± 0.33% | 0.00% ± 0.00% | 98.62 ± 0.33 pp |
| CICIDS2017 | FT-Transformer | PGD | DDoS | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2017 | FT-Transformer | PGD | Recon | 800/seed | 99.96% ± 0.07% | 0.00% ± 0.00% | 99.96 ± 0.07 pp |
| CICIDS2017 | FT-Transformer | PGD | BruteForce | 800/seed | 91.04% ± 0.40% | 0.00% ± 0.00% | 91.04 ± 0.40 pp |
| CICIDS2017 | FT-Transformer | C&W | DoS | 800/seed | 81.12% ± 0.82% | 0.00% ± 0.00% | 81.12 ± 0.82 pp |
| CICIDS2017 | FT-Transformer | C&W | DDoS | 800/seed | 99.04% ± 0.51% | 0.00% ± 0.00% | 99.04 ± 0.51 pp |
| CICIDS2017 | FT-Transformer | C&W | Recon | 800/seed | 29.29% ± 2.44% | 0.00% ± 0.00% | 29.29 ± 2.44 pp |
| CICIDS2017 | FT-Transformer | C&W | BruteForce | 800/seed | 98.12% ± 0.13% | 0.00% ± 0.00% | 98.12 ± 0.12 pp |
| CICIDS2017 | FT-Transformer | CAPGD-PrimSupport | DoS | 800/seed | 33.42% ± 7.01% | 0.54% ± 0.14% | 32.88 ± 7.15 pp |
| CICIDS2017 | FT-Transformer | CAPGD-PrimSupport | DDoS | 800/seed | 52.33% ± 2.06% | 0.12% ± 0.12% | 52.21 ± 1.94 pp |
| CICIDS2017 | FT-Transformer | CAPGD-PrimSupport | Recon | 800/seed | 99.50% ± 0.22% | 0.00% ± 0.00% | 99.50 ± 0.22 pp |
| CICIDS2017 | FT-Transformer | CAPGD-PrimSupport | BruteForce | 800/seed | 26.29% ± 8.70% | 0.00% ± 0.00% | 26.29 ± 8.70 pp |
| CICIDS2017 | FT-Transformer | C-PGD-PrimSupport | DoS | 800/seed | 7.62% ± 3.19% | 0.00% ± 0.00% | 7.62 ± 3.19 pp |
| CICIDS2017 | FT-Transformer | C-PGD-PrimSupport | DDoS | 800/seed | 0.62% ± 0.33% | 0.00% ± 0.00% | 0.62 ± 0.33 pp |
| CICIDS2017 | FT-Transformer | C-PGD-PrimSupport | Recon | 800/seed | 76.29% ± 3.33% | 0.00% ± 0.00% | 76.29 ± 3.33 pp |
| CICIDS2017 | FT-Transformer | C-PGD-PrimSupport | BruteForce | 800/seed | 2.21% ± 1.01% | 0.00% ± 0.00% | 2.21 ± 1.01 pp |
| CICIDS2017 | FT-Transformer | CAPGD (native) | DoS | 800/seed | 38.75% ± 5.04% | 15.50% ± 3.47% | 23.25 ± 3.83 pp |
| CICIDS2017 | FT-Transformer | CAPGD (native) | DDoS | 800/seed | 54.42% ± 3.80% | 1.33% ± 0.51% | 53.08 ± 4.03 pp |
| CICIDS2017 | FT-Transformer | CAPGD (native) | Recon | 800/seed | 99.04% ± 0.38% | 5.88% ± 2.26% | 93.17 ± 2.64 pp |
| CICIDS2017 | FT-Transformer | CAPGD (native) | BruteForce | 800/seed | 10.83% ± 4.73% | 0.17% ± 0.07% | 10.67 ± 4.74 pp |

### CICIDS2018

| Dataset | Victim | Attack | Class | n | Raw ASR | Valid ASR | Validity Gap |
|---|---|---|---|---|---|---|---|
| CICIDS2018 | MLP | PrimAttack (Prim-PGD) | DoS | 800/seed | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | MLP | PrimAttack (Prim-PGD) | DDoS | 800/seed | 80.50% ± 1.15% | 80.50% ± 1.15% | 0.00 ± 0.00 pp |
| CICIDS2018 | MLP | PrimAttack (Prim-PGD) | Recon | 800/seed | 5.88% ± 0.43% | 5.88% ± 0.43% | 0.00 ± 0.00 pp |
| CICIDS2018 | MLP | PrimAttack (Prim-PGD) | BruteForce | 800/seed | 91.38% ± 0.66% | 91.38% ± 0.66% | 0.00 ± 0.00 pp |
| CICIDS2018 | MLP | PGD | DoS | 800/seed | 99.75% ± 0.13% | 0.00% ± 0.00% | 99.75 ± 0.12 pp |
| CICIDS2018 | MLP | PGD | DDoS | 800/seed | 99.88% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | MLP | PGD | Recon | 800/seed | 99.29% ± 0.14% | 0.00% ± 0.00% | 99.29 ± 0.14 pp |
| CICIDS2018 | MLP | PGD | BruteForce | 800/seed | 78.88% ± 0.65% | 0.00% ± 0.00% | 78.88 ± 0.65 pp |
| CICIDS2018 | MLP | C&W | DoS | 800/seed | 99.75% ± 0.13% | 0.00% ± 0.00% | 99.75 ± 0.12 pp |
| CICIDS2018 | MLP | C&W | DDoS | 800/seed | 99.88% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | MLP | C&W | Recon | 800/seed | 96.58% ± 0.69% | 0.00% ± 0.00% | 96.58 ± 0.69 pp |
| CICIDS2018 | MLP | C&W | BruteForce | 800/seed | 54.21% ± 2.63% | 0.00% ± 0.00% | 54.21 ± 2.63 pp |
| CICIDS2018 | MLP | CAPGD-PrimSupport | DoS | 800/seed | 94.75% ± 0.90% | 0.00% ± 0.00% | 94.75 ± 0.90 pp |
| CICIDS2018 | MLP | CAPGD-PrimSupport | DDoS | 800/seed | 98.29% ± 0.38% | 0.00% ± 0.00% | 98.29 ± 0.38 pp |
| CICIDS2018 | MLP | CAPGD-PrimSupport | Recon | 800/seed | 97.08% ± 0.47% | 0.42% ± 0.07% | 96.67 ± 0.52 pp |
| CICIDS2018 | MLP | CAPGD-PrimSupport | BruteForce | 800/seed | 77.00% ± 1.95% | 0.00% ± 0.00% | 77.00 ± 1.95 pp |
| CICIDS2018 | MLP | C-PGD-PrimSupport | DoS | 800/seed | 22.62% ± 1.54% | 0.00% ± 0.00% | 22.62 ± 1.54 pp |
| CICIDS2018 | MLP | C-PGD-PrimSupport | DDoS | 800/seed | 28.92% ± 1.18% | 0.00% ± 0.00% | 28.92 ± 1.18 pp |
| CICIDS2018 | MLP | C-PGD-PrimSupport | Recon | 800/seed | 29.79% ± 2.44% | 0.00% ± 0.00% | 29.79 ± 2.44 pp |
| CICIDS2018 | MLP | C-PGD-PrimSupport | BruteForce | 800/seed | 32.08% ± 0.89% | 0.00% ± 0.00% | 32.08 ± 0.89 pp |
| CICIDS2018 | MLP | CAPGD (native) | DoS | 800/seed | 89.25% ± 2.41% | 63.67% ± 1.63% | 25.58 ± 0.95 pp |
| CICIDS2018 | MLP | CAPGD (native) | DDoS | 800/seed | 93.42% ± 1.69% | 29.75% ± 1.19% | 63.67 ± 0.51 pp |
| CICIDS2018 | MLP | CAPGD (native) | Recon | 800/seed | 95.71% ± 0.40% | 16.79% ± 1.25% | 78.92 ± 1.58 pp |
| CICIDS2018 | MLP | CAPGD (native) | BruteForce | 800/seed | 44.04% ± 2.11% | 3.79% ± 3.34% | 40.25 ± 5.45 pp |
| CICIDS2018 | CNN | PrimAttack (Prim-PGD) | DoS | 800/seed | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | CNN | PrimAttack (Prim-PGD) | DDoS | 800/seed | 1.08% ± 0.07% | 1.08% ± 0.07% | 0.00 ± 0.00 pp |
| CICIDS2018 | CNN | PrimAttack (Prim-PGD) | Recon | 800/seed | 4.21% ± 0.07% | 4.21% ± 0.07% | 0.00 ± 0.00 pp |
| CICIDS2018 | CNN | PrimAttack (Prim-PGD) | BruteForce | 800/seed | 100.00% ± 0.00% | 100.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | CNN | PGD | DoS | 800/seed | 99.92% ± 0.07% | 0.00% ± 0.00% | 99.92 ± 0.07 pp |
| CICIDS2018 | CNN | PGD | DDoS | 800/seed | 99.88% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | CNN | PGD | Recon | 800/seed | 98.92% ± 0.38% | 0.00% ± 0.00% | 98.92 ± 0.38 pp |
| CICIDS2018 | CNN | PGD | BruteForce | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2018 | CNN | C&W | DoS | 800/seed | 99.92% ± 0.07% | 0.00% ± 0.00% | 99.92 ± 0.07 pp |
| CICIDS2018 | CNN | C&W | DDoS | 800/seed | 99.92% ± 0.07% | 0.00% ± 0.00% | 99.92 ± 0.07 pp |
| CICIDS2018 | CNN | C&W | Recon | 800/seed | 96.38% ± 0.65% | 0.00% ± 0.00% | 96.38 ± 0.65 pp |
| CICIDS2018 | CNN | C&W | BruteForce | 800/seed | 100.00% ± 0.00% | 0.00% ± 0.00% | 100.00 ± 0.00 pp |
| CICIDS2018 | CNN | CAPGD-PrimSupport | DoS | 800/seed | 52.04% ± 4.35% | 0.00% ± 0.00% | 52.04 ± 4.35 pp |
| CICIDS2018 | CNN | CAPGD-PrimSupport | DDoS | 800/seed | 91.42% ± 1.94% | 0.83% ± 0.26% | 90.58 ± 1.68 pp |
| CICIDS2018 | CNN | CAPGD-PrimSupport | Recon | 800/seed | 61.29% ± 1.13% | 0.62% ± 0.50% | 60.67 ± 1.16 pp |
| CICIDS2018 | CNN | CAPGD-PrimSupport | BruteForce | 800/seed | 99.25% ± 0.22% | 0.00% ± 0.00% | 99.25 ± 0.22 pp |
| CICIDS2018 | CNN | C-PGD-PrimSupport | DoS | 800/seed | 21.33% ± 1.55% | 0.00% ± 0.00% | 21.33 ± 1.55 pp |
| CICIDS2018 | CNN | C-PGD-PrimSupport | DDoS | 800/seed | 77.04% ± 3.43% | 0.00% ± 0.00% | 77.04 ± 3.43 pp |
| CICIDS2018 | CNN | C-PGD-PrimSupport | Recon | 800/seed | 18.29% ± 1.87% | 0.00% ± 0.00% | 18.29 ± 1.87 pp |
| CICIDS2018 | CNN | C-PGD-PrimSupport | BruteForce | 800/seed | 84.79% ± 5.43% | 0.00% ± 0.00% | 84.79 ± 5.43 pp |
| CICIDS2018 | CNN | CAPGD (native) | DoS | 800/seed | 35.54% ± 2.63% | 2.83% ± 0.19% | 32.71 ± 2.78 pp |
| CICIDS2018 | CNN | CAPGD (native) | DDoS | 800/seed | 76.21% ± 0.80% | 14.46% ± 3.20% | 61.75 ± 2.67 pp |
| CICIDS2018 | CNN | CAPGD (native) | Recon | 800/seed | 59.54% ± 5.81% | 12.71% ± 2.77% | 46.83 ± 3.09 pp |
| CICIDS2018 | CNN | CAPGD (native) | BruteForce | 800/seed | 95.38% ± 1.00% | 36.21% ± 10.43% | 59.17 ± 10.47 pp |
| CICIDS2018 | FT-Transformer | PrimAttack (Prim-PGD) | DoS | 800/seed | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | PrimAttack (Prim-PGD) | DDoS | 800/seed | 0.04% ± 0.07% | 0.04% ± 0.07% | 0.00 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | PrimAttack (Prim-PGD) | Recon | 800/seed | 1.00% ± 0.50% | 1.00% ± 0.50% | 0.00 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | PrimAttack (Prim-PGD) | BruteForce | 800/seed | 0.00% ± 0.00% | 0.00% ± 0.00% | 0.00 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | PGD | DoS | 800/seed | 99.71% ± 0.31% | 0.00% ± 0.00% | 99.71 ± 0.31 pp |
| CICIDS2018 | FT-Transformer | PGD | DDoS | 800/seed | 99.88% ± 0.00% | 0.00% ± 0.00% | 99.88 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | PGD | Recon | 800/seed | 99.79% ± 0.14% | 0.00% ± 0.00% | 99.79 ± 0.14 pp |
| CICIDS2018 | FT-Transformer | PGD | BruteForce | 800/seed | 68.04% ± 1.75% | 0.00% ± 0.00% | 68.04 ± 1.75 pp |
| CICIDS2018 | FT-Transformer | C&W | DoS | 800/seed | 36.12% ± 1.09% | 0.00% ± 0.00% | 36.12 ± 1.09 pp |
| CICIDS2018 | FT-Transformer | C&W | DDoS | 800/seed | 79.08% ± 0.90% | 0.00% ± 0.00% | 79.08 ± 0.90 pp |
| CICIDS2018 | FT-Transformer | C&W | Recon | 800/seed | 99.25% ± 0.33% | 0.00% ± 0.00% | 99.25 ± 0.33 pp |
| CICIDS2018 | FT-Transformer | C&W | BruteForce | 800/seed | 0.17% ± 0.19% | 0.00% ± 0.00% | 0.17 ± 0.19 pp |
| CICIDS2018 | FT-Transformer | CAPGD-PrimSupport | DoS | 800/seed | 0.08% ± 0.07% | 0.00% ± 0.00% | 0.08 ± 0.07 pp |
| CICIDS2018 | FT-Transformer | CAPGD-PrimSupport | DDoS | 800/seed | 0.58% ± 0.19% | 0.00% ± 0.00% | 0.58 ± 0.19 pp |
| CICIDS2018 | FT-Transformer | CAPGD-PrimSupport | Recon | 800/seed | 4.96% ± 0.38% | 0.00% ± 0.00% | 4.96 ± 0.38 pp |
| CICIDS2018 | FT-Transformer | CAPGD-PrimSupport | BruteForce | 800/seed | 33.25% ± 3.00% | 0.00% ± 0.00% | 33.25 ± 3.00 pp |
| CICIDS2018 | FT-Transformer | C-PGD-PrimSupport | DoS | 800/seed | 0.04% ± 0.07% | 0.00% ± 0.00% | 0.04 ± 0.07 pp |
| CICIDS2018 | FT-Transformer | C-PGD-PrimSupport | DDoS | 800/seed | 0.25% ± 0.00% | 0.00% ± 0.00% | 0.25 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | C-PGD-PrimSupport | Recon | 800/seed | 1.67% ± 0.29% | 0.00% ± 0.00% | 1.67 ± 0.29 pp |
| CICIDS2018 | FT-Transformer | C-PGD-PrimSupport | BruteForce | 800/seed | 4.83% ± 2.02% | 0.00% ± 0.00% | 4.83 ± 2.02 pp |
| CICIDS2018 | FT-Transformer | CAPGD (native) | DoS | 800/seed | 0.04% ± 0.07% | 0.00% ± 0.00% | 0.04 ± 0.07 pp |
| CICIDS2018 | FT-Transformer | CAPGD (native) | DDoS | 800/seed | 0.17% ± 0.07% | 0.17% ± 0.07% | 0.00 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | CAPGD (native) | Recon | 800/seed | 2.38% ± 0.25% | 1.00% ± 0.38% | 1.38 ± 0.13 pp |
| CICIDS2018 | FT-Transformer | CAPGD (native) | BruteForce | 800/seed | 22.38% ± 1.41% | 1.12% ± 1.62% | 21.25 ± 2.69 pp |

## Per-Model Attack Performance

Victim-pooled-over-class rates; `n = 3,200` per seed. Victims are never pooled together.

### MLP

| Dataset | Attack | n | Raw ASR | Valid ASR | Validity Gap |
|---|---|---|---|---|---|
| CICIDS2017 | PrimAttack (Prim-PGD) | 3,200/seed | 23.01% ± 0.59% | 23.01% ± 0.59% | 0.00 ± 0.00 pp |
| CICIDS2017 | PGD | 3,200/seed | 99.99% ± 0.02% | 0.00% ± 0.00% | 99.99 ± 0.02 pp |
| CICIDS2017 | C&W | 3,200/seed | 99.95% ± 0.02% | 0.00% ± 0.00% | 99.95 ± 0.02 pp |
| CICIDS2017 | CAPGD-PrimSupport | 3,200/seed | 94.41% ± 0.83% | 1.98% ± 0.33% | 92.43 ± 1.15 pp |
| CICIDS2017 | C-PGD-PrimSupport | 3,200/seed | 50.76% ± 1.90% | 0.00% ± 0.00% | 50.76 ± 1.90 pp |
| CICIDS2017 | CAPGD (native) | 3,200/seed | 94.47% ± 0.82% | 10.02% ± 0.50% | 84.45 ± 0.65 pp |
| CICIDS2018 | PrimAttack (Prim-PGD) | 3,200/seed | 44.44% ± 0.27% | 44.44% ± 0.27% | 0.00 ± 0.00 pp |
| CICIDS2018 | PGD | 3,200/seed | 94.45% ± 0.18% | 0.00% ± 0.00% | 94.45 ± 0.18 pp |
| CICIDS2018 | C&W | 3,200/seed | 87.60% ± 0.75% | 0.00% ± 0.00% | 87.60 ± 0.75 pp |
| CICIDS2018 | CAPGD-PrimSupport | 3,200/seed | 91.78% ± 0.66% | 0.10% ± 0.02% | 91.68 ± 0.64 pp |
| CICIDS2018 | C-PGD-PrimSupport | 3,200/seed | 28.35% ± 0.63% | 0.00% ± 0.00% | 28.35 ± 0.63 pp |
| CICIDS2018 | CAPGD (native) | 3,200/seed | 80.60% ± 0.56% | 28.50% ± 1.79% | 52.10 ± 1.47 pp |

### CNN

| Dataset | Attack | n | Raw ASR | Valid ASR | Validity Gap |
|---|---|---|---|---|---|
| CICIDS2017 | PrimAttack (Prim-PGD) | 3,200/seed | 59.67% ± 0.42% | 59.67% ± 0.42% | 0.00 ± 0.00 pp |
| CICIDS2017 | PGD | 3,200/seed | 96.07% ± 0.16% | 0.00% ± 0.00% | 96.07 ± 0.16 pp |
| CICIDS2017 | C&W | 3,200/seed | 95.45% ± 0.08% | 0.00% ± 0.00% | 95.45 ± 0.08 pp |
| CICIDS2017 | CAPGD-PrimSupport | 3,200/seed | 96.65% ± 1.00% | 5.06% ± 0.12% | 91.58 ± 0.96 pp |
| CICIDS2017 | C-PGD-PrimSupport | 3,200/seed | 59.98% ± 3.38% | 0.00% ± 0.00% | 59.98 ± 3.38 pp |
| CICIDS2017 | CAPGD (native) | 3,200/seed | 96.50% ± 1.00% | 19.00% ± 0.19% | 77.50 ± 0.99 pp |
| CICIDS2018 | PrimAttack (Prim-PGD) | 3,200/seed | 26.32% ± 0.04% | 26.32% ± 0.04% | 0.00 ± 0.00 pp |
| CICIDS2018 | PGD | 3,200/seed | 99.68% ± 0.10% | 0.00% ± 0.00% | 99.68 ± 0.10 pp |
| CICIDS2018 | C&W | 3,200/seed | 99.05% ± 0.18% | 0.00% ± 0.00% | 99.05 ± 0.18 pp |
| CICIDS2018 | CAPGD-PrimSupport | 3,200/seed | 76.00% ± 1.70% | 0.36% ± 0.18% | 75.64 ± 1.63 pp |
| CICIDS2018 | C-PGD-PrimSupport | 3,200/seed | 50.36% ± 2.68% | 0.00% ± 0.00% | 50.36 ± 2.68 pp |
| CICIDS2018 | CAPGD (native) | 3,200/seed | 66.67% ± 2.05% | 16.55% ± 3.91% | 50.11 ± 2.06 pp |

### FT-Transformer

| Dataset | Attack | n | Raw ASR | Valid ASR | Validity Gap |
|---|---|---|---|---|---|
| CICIDS2017 | PrimAttack (Prim-PGD) | 3,200/seed | 0.60% ± 0.02% | 0.60% ± 0.02% | 0.00 ± 0.00 pp |
| CICIDS2017 | PGD | 3,200/seed | 97.41% ± 0.17% | 0.00% ± 0.00% | 97.41 ± 0.17 pp |
| CICIDS2017 | C&W | 3,200/seed | 76.90% ± 0.42% | 0.00% ± 0.00% | 76.90 ± 0.42 pp |
| CICIDS2017 | CAPGD-PrimSupport | 3,200/seed | 52.89% ± 3.94% | 0.17% ± 0.02% | 52.72 ± 3.95 pp |
| CICIDS2017 | C-PGD-PrimSupport | 3,200/seed | 21.69% ± 0.33% | 0.00% ± 0.00% | 21.69 ± 0.33 pp |
| CICIDS2017 | CAPGD (native) | 3,200/seed | 50.76% ± 3.37% | 5.72% ± 1.17% | 45.04 ± 3.78 pp |
| CICIDS2018 | PrimAttack (Prim-PGD) | 3,200/seed | 0.26% ± 0.13% | 0.26% ± 0.13% | 0.00 ± 0.00 pp |
| CICIDS2018 | PGD | 3,200/seed | 91.85% ± 0.49% | 0.00% ± 0.00% | 91.85 ± 0.49 pp |
| CICIDS2018 | C&W | 3,200/seed | 53.66% ± 0.17% | 0.00% ± 0.00% | 53.66 ± 0.17 pp |
| CICIDS2018 | CAPGD-PrimSupport | 3,200/seed | 9.72% ± 0.88% | 0.00% ± 0.00% | 9.72 ± 0.88 pp |
| CICIDS2018 | C-PGD-PrimSupport | 3,200/seed | 1.70% ± 0.56% | 0.00% ± 0.00% | 1.70 ± 0.56 pp |
| CICIDS2018 | CAPGD (native) | 3,200/seed | 6.24% ± 0.28% | 0.57% ± 0.48% | 5.67 ± 0.69 pp |

## Targeted vs Untargeted Attacks

The two rows per victim differ only in the PrimAttack objective. Both use Prim-PGD, joint mode, the unbounded box, and the 256-evaluation cap; `n = 3,200` per seed.

| Dataset | Victim | Objective | n | Raw ASR | Valid ASR | Validity Gap |
|---|---|---|---|---|---|---|
| CICIDS2017 | MLP | Targeted → Benign | 3,200/seed | 22.98% ± 0.59% | 22.98% ± 0.59% | 0.00 ± 0.00 pp |
| CICIDS2017 | MLP | Untargeted → any non-source class | 3,200/seed | 23.01% ± 0.59% | 23.01% ± 0.59% | 0.00 ± 0.00 pp |
| CICIDS2017 | CNN | Targeted → Benign | 3,200/seed | 59.39% ± 0.42% | 59.39% ± 0.42% | 0.00 ± 0.00 pp |
| CICIDS2017 | CNN | Untargeted → any non-source class | 3,200/seed | 59.67% ± 0.42% | 59.67% ± 0.42% | 0.00 ± 0.00 pp |
| CICIDS2017 | FT-Transformer | Targeted → Benign | 3,200/seed | 0.60% ± 0.02% | 0.60% ± 0.02% | 0.00 ± 0.00 pp |
| CICIDS2017 | FT-Transformer | Untargeted → any non-source class | 3,200/seed | 0.60% ± 0.02% | 0.60% ± 0.02% | 0.00 ± 0.00 pp |
| CICIDS2018 | MLP | Targeted → Benign | 3,200/seed | 24.86% ± 0.11% | 24.86% ± 0.11% | 0.00 ± 0.00 pp |
| CICIDS2018 | MLP | Untargeted → any non-source class | 3,200/seed | 44.44% ± 0.27% | 44.44% ± 0.27% | 0.00 ± 0.00 pp |
| CICIDS2018 | CNN | Targeted → Benign | 3,200/seed | 26.08% ± 0.02% | 26.08% ± 0.02% | 0.00 ± 0.00 pp |
| CICIDS2018 | CNN | Untargeted → any non-source class | 3,200/seed | 26.32% ± 0.04% | 26.32% ± 0.04% | 0.00 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | Targeted → Benign | 3,200/seed | 0.26% ± 0.13% | 0.26% ± 0.13% | 0.00 ± 0.00 pp |
| CICIDS2018 | FT-Transformer | Untargeted → any non-source class | 3,200/seed | 0.26% ± 0.13% | 0.26% ± 0.13% | 0.00 ± 0.00 pp |

## Perturbation Budget Sensitivity

Targeted-to-Benign PrimAttack/Prim-PGD in joint mode. p50 and p75 are train-only per-class calibrations of padding and relative-duration headroom. Unbounded removes those percentile caps while retaining the train-p99 feature envelope and DoS/DDoS minimum-rate floor. `Median normalized primitive cost` is the FINAL-pipeline cost `p / p_hi + delay / delay_hi`, computed among valid successes per seed and then reported as mean ± SD; `—` means no valid success in at least one seed.

| Dataset | Victim | Budget | n | Raw ASR | Valid ASR | Validity Gap | Median normalized primitive cost |
|---|---|---|---|---|---|---|---|
| CICIDS2017 | MLP | p50 | 3,200/seed | 2.45% ± 0.14% | 2.45% ± 0.14% | 0.00 ± 0.00 pp | 0.695 ± 0.048 |
| CICIDS2017 | MLP | p75 | 3,200/seed | 4.11% ± 0.07% | 4.11% ± 0.07% | 0.00 ± 0.00 pp | 0.623 ± 0.025 |
| CICIDS2017 | MLP | unbounded | 3,200/seed | 22.98% ± 0.59% | 22.98% ± 0.59% | 0.00 ± 0.00 pp | 0.301 ± 0.002 |
| CICIDS2017 | CNN | p50 | 3,200/seed | 8.53% ± 0.73% | 8.53% ± 0.73% | 0.00 ± 0.00 pp | 0.619 ± 0.018 |
| CICIDS2017 | CNN | p75 | 3,200/seed | 12.74% ± 0.52% | 12.74% ± 0.52% | 0.00 ± 0.00 pp | 0.641 ± 0.008 |
| CICIDS2017 | CNN | unbounded | 3,200/seed | 59.39% ± 0.42% | 59.39% ± 0.42% | 0.00 ± 0.00 pp | 0.392 ± 0.014 |
| CICIDS2017 | FT-Transformer | p50 | 3,200/seed | 0.15% ± 0.02% | 0.15% ± 0.02% | 0.00 ± 0.00 pp | 0.319 ± 0.046 |
| CICIDS2017 | FT-Transformer | p75 | 3,200/seed | 0.15% ± 0.02% | 0.15% ± 0.02% | 0.00 ± 0.00 pp | 0.241 ± 0.037 |
| CICIDS2017 | FT-Transformer | unbounded | 3,200/seed | 0.60% ± 0.02% | 0.60% ± 0.02% | 0.00 ± 0.00 pp | 0.050 ± 0.000 |
| CICIDS2018 | MLP | p50 | 3,200/seed | 0.47% ± 0.19% | 0.47% ± 0.19% | 0.00 ± 0.00 pp | 0.483 ± 0.041 |
| CICIDS2018 | MLP | p75 | 3,200/seed | 0.53% ± 0.22% | 0.53% ± 0.22% | 0.00 ± 0.00 pp | 0.558 ± 0.125 |
| CICIDS2018 | MLP | unbounded | 3,200/seed | 24.86% ± 0.11% | 24.86% ± 0.11% | 0.00 ± 0.00 pp | 0.600 ± 0.000 |
| CICIDS2018 | CNN | p50 | 3,200/seed | 0.01% ± 0.02% | 0.01% ± 0.02% | 0.00 ± 0.00 pp | — |
| CICIDS2018 | CNN | p75 | 3,200/seed | 0.01% ± 0.02% | 0.01% ± 0.02% | 0.00 ± 0.00 pp | — |
| CICIDS2018 | CNN | unbounded | 3,200/seed | 26.08% ± 0.02% | 26.08% ± 0.02% | 0.00 ± 0.00 pp | 0.100 ± 0.000 |
| CICIDS2018 | FT-Transformer | p50 | 3,200/seed | 0.04% ± 0.05% | 0.04% ± 0.05% | 0.00 ± 0.00 pp | — |
| CICIDS2018 | FT-Transformer | p75 | 3,200/seed | 0.04% ± 0.05% | 0.04% ± 0.05% | 0.00 ± 0.00 pp | — |
| CICIDS2018 | FT-Transformer | unbounded | 3,200/seed | 0.26% ± 0.13% | 0.26% ± 0.13% | 0.00 ± 0.00 pp | 0.096 ± 0.019 |

Artifacts used for seed 42 are under `FINAL_OUTPUTS/runs/`. Independently sampled seed-2024 and seed-2026 artifacts are under `FINAL_OUTPUTS/independent_seed_runs/`. All reported values are computed from the audited NPZ outcomes; none was entered manually.
