# 7. Attack Metrics & Evaluation (MODERATE DETAIL)

All PrimAttack metrics share **one denominator**: eligible clean-correct malicious rows.
Defined in `run_cicids2017_primitive_attack.py` (`evaluate_cell`, `run`) and
`validation/metrics.py` (`targeted_asr_suite`). Let, per (class, victim, seed) cell:

- `E` = **clean-correct eligibility**: `victim(scale(x₀)).argmax == class_id` (`:164`).
- `N_elig = Σ E`. `_rate(mask, E) = Σ(mask ∧ E)/Σ E` (`:190-192`).
- **targeted** `T`: `adv_pred == 0` (Benign) (`:166`).
- **evasion** (untargeted) `U`: `adv_pred ≠ class_id` (`:165`) — any misclassification.
  Note `T ⊆ U`.
- `V` = domain validity = validator_v2 `hybrid_valid` (`:167`).
- `P` = primitive feasibility = `semantic.primitive_feasible ∧
  primitive_transform_consistent` (`:368-374`).
- `S` = semantic status; `[S=PASS]` (doc 6).

## 7.1 ASR family (all over `N_elig`)
$$
\text{raw untargeted ASR}=\frac{\sum(U)}{N_{elig}}\quad
\text{raw targeted ASR}=\frac{\sum(T)}{N_{elig}}
$$
$$
\text{valid targeted ASR}=\frac{\sum(T\wedge V)}{N_{elig}}\;
\subseteq\;\text{primitive-feasible ASR}=\frac{\sum(T\wedge V\wedge P)}{N_{elig}}\;
\subseteq\;\text{SP-ASR}=\frac{\sum(T\wedge V\wedge P\wedge[S{=}PASS])}{N_{elig}}
$$
Numerators are **strictly nested** (`run :511-519`): raw ⊇ valid ⊇ feasible ⊇ SP. In the
committed sweep raw=valid=feasible (validator/feasibility rejected no classifier success),
and SP=0.

`validation/metrics.py:targeted_asr_suite` mirrors this for the generic attack adapter:
`raw = e/denom`, `hard_valid = (e∧hard)/denom`, `hybrid_valid = (e∧hybrid)/denom`,
`valid_in_distribution = (e∧hybrid∧in_dist)/denom` — one shared denom.

## 7.2 Plausibility / in-distribution (two distinct notions)
- **validator_v2 plausibility** (`plausibility.py`): all 79 features inside the train
  p0.1–p99.9 band; **train-fit**; used by `targeted_asr_suite.valid_in_distribution` and
  the attack-artifact scorer. Separate from structural validity.
- **VAE Mahalanobis IDR** (`run_cicids2017_vae_attacks.py:_idr_mask`): per-class
  generator-relative Mahalanobis distance ≤ a **val-anchored** p95 threshold; saved into
  the PrimAttack npz as `in_distribution`. **PrimAttack's SP-ASR uses neither** — it is
  reported but not gated.

## 7.3 Per-class / per-victim / pooling
- **PrimAttack runner**: computes metrics **per (class, victim, seed) cell**, per-cell
  denominator; no cross-victim pooling at write time (each cell → one npz + one JSON
  `cell`).
- **Sweep statistics** (`analyze_primattack_experiments.py`): pools victims + classes for
  the paired tests (doc 4) — the pseudoreplication concern.
- **`validation/evaluation/attack_artifact_validity.py`**: sample-weighted pooling
  (overall = Σ successes / Σ clean-correct across cells; by_class pools victims+seeds).
  **STALE** — expects fields the runner doesn't save; do not use (see
  `00_OPEN_ISSUES.md#A3`).

## 7.4 Denominator discipline (why it matters)
Using clean-correct rows as the shared denominator prevents inflating ASR by counting
already-misclassified flows, and keeps raw/valid/feasible/SP directly comparable (same
base). Dropping `NOT_FULLY_TESTABLE` rows from the denominator would inflate SP-ASR and
is explicitly prohibited (`docs/primattack/06_RESULTS_GUIDE.md`).

## 7.5 Claims
- **Can claim**: a transparent, nested ASR hierarchy with one clean-correct denominator,
  separating classifier evasion from validity, feasibility, and semantic preservation.
- **Must NOT claim**: that raw ASR (or feature-space PGD/C&W raw ASR ≈ 1.0) is a
  meaningful evasion result without the validity/feasibility/semantic gates; that
  in-distribution (either notion) implies realism.
