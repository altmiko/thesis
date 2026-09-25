# 7. Attack Metrics & Evaluation (MODERATE DETAIL)

All PrimAttack metrics share **one denominator**: eligible clean-correct malicious rows.
The canonical paired driver selects and freezes this set once per `(victim,class)`;
every attack and seed reuses the same row IDs and order. The standalone runner computes
clean-correct eligibility inside its class-sampled batch. Let, per cell:

- `E` = **clean-correct eligibility**: `victim(scale(x0)).argmax == class_id`.
- `N_elig = sum(E)` and every rate divides by `N_elig`.
- **targeted** `T`: `adv_pred == 0` (Benign).
- **evasion** (untargeted) `U`: `adv_pred != class_id`; `T` is a subset of `U`.
- `V` = domain validity = validator_v2 `hybrid_valid`.
- `P` = `semantic.primitive_feasible & primitive_transform_consistent`.
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
Numerators are **strictly nested**: raw ⊇ valid ⊇ feasible ⊇ SP. In the current
v2 artifacts, domain validity and primitive feasibility are 100% for every PrimAttack
cell, so raw targeted = valid targeted = primitive-feasible targeted. SP-ASR is lower
where the semantic proxy is not `PASS`; it is not zero in the current campaign.

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
- **Standalone runner**: computes one cell per `(class,victim,seed)`.
- **Full paired driver**: freezes at most 800 clean-correct rows per
  `(victim,class)` and writes one NPZ per `(victim,class,attack,seed)`.
- **Current v2 optimizer comparison**: concatenates the four classes only within each
  victim for paired tests; victims are never pooled as independent rows. Seed 42 is the
  paired reference and seeds 42/123/2024 are summarized descriptively.
- **Historical sweep statistics** (`analyze_primattack_experiments.py`): pool
  victims/classes and therefore carry the pseudoreplication concern in doc 4.
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
