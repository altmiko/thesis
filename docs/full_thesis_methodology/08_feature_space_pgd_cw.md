# 8. Feature-Space PGD / C&W Baselines (MODERATE DETAIL)

Unconstrained feature-space baselines that perturb the 79-dim (scaled) vector directly.
They exist to show the *contrast*: unconstrained attacks trivially evade but produce
domain-invalid vectors.

### Kernels (`src/attack/input_baselines.py`)
- **`input_pgd_attack`** (`:8-65`): **untargeted L∞ PGD**. `eps=0.5`, `alpha=0.05`,
  `steps=40`, `random_start=True`; cross-entropy **ascent** on the true class,
  `grad.sign()` step, projection back into the L∞ ε-ball around x₀ (box-project; no [0,1]
  clamp — operates in RobustScaler space).
- **`input_cw_attack`** (`:68-177`): **untargeted C&W**. Adam on `delta`, `lr=0.01`,
  `iters=200`, `lambda=1.0`, `kappa=0.0`, convergence `1e-5`; loss
  `λ·max(0, (f_true − f_other) + κ) + ‖delta‖₂²`, tracking the best successful low-L₂
  perturbation.

Both are **untargeted** (source-class CE / logit margin), not targeted→Benign.

### What actually ran (CICIDS2017)
- **`scripts/run_baseline_attacks_nids.py`** → `outputs/cicids2017_baseline_pgd_cw/`
  (`baseline_pgd_cw_results.json` + `.md`, **aggregated only, no per-sample npz**).
  6 cells = victims {mlp, cnn, ft_transformer} × {input-pgd, input-cw}, classes
  DoS/DDoS/Recon/BruteForce, 5 seeds, 200 samples/class. 79 features, RobustScaler space,
  `y_true = source class` (untargeted).
- **Result**: **raw ASR ≈ 1.0**, but `domain_valid = realizable = valid_real_evasion =
  0.0`. Unconstrained perturbations evade every victim yet **never** pass validator_v2 —
  exactly the intended baseline story motivating PrimAttack's constrained design.

### FAB (AutoAttack) — `fab` family of `scripts/run_full_adversarial_eval.py`
- **Implementation**: upstream `autoattack.fab_pt.FABAttack_PT` from the frozen clone
  `external/auto-attack` (commit `a39220048b3c9f2cca9a4d3a54604793c68eca7e`, installed
  editable into the `thesis` env); adapter `src/comparisons/fab_autoattack.py`
  (`load_fab_class` asserts the import resolves to the clone).
- **Attack space**: FAB hard-codes `[0,1]^d`, so `u = (raw − train_min)/span` with the
  **train-fitted** pristine min-max box (same box as CAPGD native). The 3 train-constant
  features (`Fwd/Bwd URG Flags`, `URG Flag Count`) are pinned; clean rows outside the box
  are rejected (none of the CICIDS2017 attack-class test rows are). Output
  `raw + (u_adv − u0)·span`, so rows FAB fails on are returned bit-identical.
- **Defaults**: untargeted, `L2`, `eps=0.5` (acceptance radius in the unit box),
  `n_iter=100`, `n_restarts=1`.
- **Demo run** (seed 42, 800 clean-correct rows/class, all 3 victims, `--families fab`) →
  `outputs/adv_campaign/fab_demo_cicids2017/`: raw untargeted ASR mlp 0.894, cnn 0.857,
  FT 0.918 (class-mean); **valid ASR (evasion ∧ hybrid_valid) = 0.0 in all 12 cells**;
  0 of 8,541 evading rows even pass `hard_structural_valid`. Single seed; untargeted.

### Status classification of the related files
| File | Status |
|---|---|
| `src/attack/input_baselines.py` | LIVE (kernels) |
| `scripts/run_baseline_attacks_nids.py` + `outputs/cicids2017_baseline_pgd_cw/` | LIVE (aggregated json/md; **no npz**) |
| `src/attack/run_cicids2017_input_baseline.py` | **CODE-ONLY** — targeted→Benign PGD variant; `outputs/cicids2017_input_baseline/` **absent** (never run) |
| `src/attack/adversarial_attacks.py` | **CICIoT** (torchattacks FGSM/PGD/CW; `load_model` reused as helper) |
| `src/attack/constrained_input_baselines.py` | **CICIoT** (39-feature schema, VAE-constrained) |
| `src/attack/latent_pgd.py`, `latent_cw.py` | VAE **latent-space** (not feature-space), CICIoT/legacy |

### Assumptions · Limitations · Claims
- **Assumptions**: L∞/L₂ balls in RobustScaler space are the standard baseline threat.
- **Limitations**: untargeted (not comparable class-target to PrimAttack's targeted→Benign);
  no per-sample artifacts (cannot re-audit individual flows or run paired raw-vs-valid on
  these); the targeted feature-PGD variant was never executed.
- **Can claim**: unconstrained feature-space PGD/C&W achieve ≈100% raw evasion but 0%
  domain-valid evasion on CICIDS2017 — motivating constrained/primitive attacks.
- **Must NOT claim**: that these are a completed targeted→Benign comparison to PrimAttack,
  or that any produced a realizable/valid adversarial flow. Do not present
  `run_cicids2017_input_baseline.py` or the CICIoT attacks as CICIDS2017 results.
