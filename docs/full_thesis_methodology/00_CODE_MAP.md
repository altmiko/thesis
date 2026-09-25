# Code Map — file → responsibility, with liveness classification

Legend: **LIVE** = current CICIDS2017 path with real artifacts · **CODE-ONLY** =
implemented but no committed artifacts · **STALE** = broken/superseded/expects
missing inputs · **CICIoT** = belongs to the older CICIoT2023 study · **ARCHIVE** =
retired snapshot.

## Preprocessing (doc 1)

| Path | Role | Status |
|------|------|--------|
| `src/preprocessing/preprocess_cicids2017_distrinet.py` | Full CSV→arrays pipeline (single file, no hidden deps) | LIVE |
| `data/raw/CICIDS_2017_Distrinet/*.csv` | 5 corrected DistriNet day files | input |
| `data/processed/CICIDS_2017_Distrinet/` | `X_*.npy`, `X_*_pristine.npy`, `y_*_{cat,bin}.npy`, `*.parquet`, `scaler.pkl`, `class_weights_*.npy`, `*_manifest.json`, `leakage_audit.json`, `duplicate_audit.json` | LIVE artifacts |
| `src/datasets/cicids2017.py` | `CICIDS2017Adapter` (loads splits, scaler, manifest, class map) | LIVE |
| `src/datasets/feature_manifest.py` | `FeatureManifest` (feature order + index lookup) | LIVE |
| `scripts/merge_cicids2017.py` | one-off raw merge helper | CODE-ONLY |

## PrimAttack (doc 2)

| Path | Role | Status |
|------|------|--------|
| `src/attack/realizability/base.py` | `FeatureRole`, `PrimitiveSpec`, `PrimitiveCapabilities`, `NullPacketBackend`, protocols | LIVE |
| `src/attack/realizability/cicids2017.py` | `CICIDS2017PrimitiveModel` — the φ(x₀,p,delay,shape) transform, capabilities, bounds, projection | LIVE (core) |
| `src/attack/primitive_optimizer.py` | `optimize_primitive_candidates` — exact integer-padding enumeration + adaptive projected refinement, success-first candidate selection | LIVE (core) |
| `src/attack/realizability/validator.py` | `RealizabilityValidator` — internal primitive-consistency checks | LIVE |
| `src/attack/primattack_budget.py` | Train-only budget calibration + loading | LIVE |
| `src/attack/flow_semantics.py` | `FlowSemanticValidator` — SP proxy | LIVE |
| `src/attack/run_cicids2017_primitive_attack.py` | Attack runner + per-cell metrics + npz artifacts | LIVE |
| `scripts/run_full_adversarial_eval.py` | Clean-correct frozen-roster paired evaluation: input PGD/C&W, VAE A1–A6, PrimAttack search/random, native/primitive CAPGD, FAB; random/head selection → `outputs/adv_campaign/<dataset>/` by default | LIVE |
| `outputs/full_adv_eval_primattack_v2/` | Completed CICIDS2017 paired PrimAttack-v2 campaign: 3 victims × 4 classes × p50/p75/envelope-only × 3 modes × search/random × 3 attack seeds | LIVE artifacts |
| `outputs/full_adv_eval/` | Paired evaluation with the replaced Adam/sigmoid `(p, α)` optimizer (`prim_opt_*`); frozen selection source and "before" condition | LIVE (historical) |
| `scripts/budget_sweep_primitive.py` | Drives the 3-budget × 3-mode sweep | LIVE |
| `artifacts/primattack/budget_calibration.json` | Frozen train-fit calibration | LIVE |
| `outputs/primattack_budget_sensitivity_full/` | Sweep from the replaced `(p, α)` Adam optimizer (mlp,cnn × 4 classes × 3 budgets × 3 modes × seed 42 = 72 npz) | historical |
| `outputs/primattack_random_control/` | Random-feasible control (older commit, MLP-only) | CODE-ONLY/older |
| `outputs/primattack_smoke/` | Smoke run | ARCHIVE-ish |
| `src/attack/vae_latent_primitive.py` | VAE-latent → primitive attack | CODE-ONLY (no live results) |

## Validator v2 (doc 3)

| Path | Role | Status |
|------|------|--------|
| `validation/validator/{engine,rule,tolerance,result,plausibility,report}.py` | Rule engine, rule dataclass, tolerance, result aggregation, plausibility, reporting | LIVE |
| `validation/schema/cicids2017_distrinet.yaml` | Inferred per-feature schema profile (79 features) | LIVE artifact |
| `validation/rules/cicids2017_distrinet/{extractor_rules.yaml,protocol_rules.yaml,mined_rules.json,plausibility_profile.json}` | Rule sets (7 EXTRACTOR, 79 PROTOCOL, 16 MINED) + plausibility | LIVE artifacts |
| `validation/mining/{run_mining,candidate_templates,mine_pairwise,mine_arithmetic,mine_implications,evaluate_candidates,prune_rules,infer_schema,feature_registry,data_access}.py` | Mining pipeline | LIVE |
| `validation/attack_interface.py` | `structural_masks`, `evaluate_attack` (attack-side adapter) | LIVE |
| `validation/metrics.py` | `targeted_asr_suite` | LIVE |
| `validation/evaluation/{clean_acceptance,synthetic_violations,legacy_vs_v2,attack_artifact_validity}.py` | Validator self-tests + attack-artifact scorer | mixed (attack_artifact_validity = STALE) |
| `validation/reports/cicids2017_distrinet/{clean_acceptance_report,synthetic_violation_report}.md` | Test reports | LIVE artifacts |
| `src/attack/validator.py`, `src/attack/latent_infra.py` | old hand-coded validator/mask | CICIoT/legacy |

## Statistical evaluation (doc 4)

| Path | Role | Status |
|------|------|--------|
| `scripts/analyze_primattack_experiments.py` | Cochran's Q + exact McNemar + Friedman + Wilcoxon(Pratt) + Holm over the sweep | LIVE |
| `src/evaluation/paired_validity_gap.py` | `mcnemar_test`, `holm_adjust`, `contingency`, `newcombe_paired_ci`, targeted-validity-gap analysis | LIVE |
| `scripts/build_primattack_budget_report.py` | Renders `docs/primattack_budget_results.md` from artifacts + stats | LIVE |
| `outputs/primattack_budget_sensitivity_full/paired_statistics.json` | Committed test output | LIVE artifact |
| `src/attack/statistical_analysis.py` | Bootstrap + statsmodels McNemar (latent PGD vs C&W) | CICIoT/legacy |

## Victim classifiers (doc 5)

| Path | Role | Status |
|------|------|--------|
| `src/classifiers/models.py` | `SimpleMLP`, `CNNOnly` | LIVE |
| `src/classifiers/ft_transformer.py` | `FTTransformer` (local, rtdl-free) | LIVE |
| `src/classifiers/cicids2017d_experiments.py` | Training driver (loss, optim, sched, early-stop) | LIVE |
| `src/classifiers/cicids2017d_victims.py` | `load_category_victim` (sha256/identity guards) | LIVE |
| `outputs/cicids2017distrinet/models/{mlp,cnn,ft_transformer}_{binary,category}.pt` | Trained checkpoints | LIVE artifacts |
| `outputs/cicids2017distrinet_ft/`, `outputs/_ft_smoke/` | FT dedicated + smoke | LIVE / smoke |
| `src/classifiers/review_baselines.py` | RF/XGB baselines (mlp,cnn only) | LIVE (eval) |
| `src/classifiers/prior_corrected_evaluation.py` | CICIoT2023 (39 feats) | CICIoT |

## Attacks — feature-space & VAE (docs 8, 9)

| Path | Role | Status |
|------|------|--------|
| `src/attack/input_baselines.py` | `input_pgd_attack`, `input_cw_attack` kernels | LIVE |
| `scripts/run_baseline_attacks_nids.py` | Runs PGD/C&W on CICIDS2017 → `outputs/cicids2017_baseline_pgd_cw/` | LIVE (aggregated json/md only, no npz) |
| `src/attack/run_cicids2017_input_baseline.py` | targeted→Benign PGD variant | CODE-ONLY (no artifacts) |
| `src/attack/adversarial_attacks.py` | torchattacks FGSM/PGD/CW | CICIoT |
| `src/attack/constrained_input_baselines.py` | VAE-constrained input attack (39 feats) | CICIoT |
| `src/attack/{latent_pgd,latent_cw,latent_gmm,latent_restarts}.py` | VAE latent-space attacks | CICIoT/legacy |
| `src/vae/model.py` (`MixedInputBetaVAE`), `src/vae/cicids2017_stage_a.py` | β-VAE + per-class generator training | LIVE (generators) |
| `src/attack/run_cicids2017_vae_latent_attack.py`, `vae_latent_variants.py`, `run_cicids2017_latent_variants.py` | VAE latent attacks (CICIDS2017) | CODE-ONLY (only smoke npz) |
| `outputs/cicids2017_vae_stage_a/` | 4 `vae_*.pt`, 4 `idr_*.npz`, summary | LIVE artifacts |
| `outputs/cicids2017_vae_attacks_{masked,pave_run}/` | 1 smoke npz each | SMOKE |

## Archived / retired

- `old_root_files/` — retired docs, `retired_lstm_cnn_lstm_2026-09-24/` (57 snapshots of the retired `LSTMOnly`/`SerialCNNLSTM` victims and their attack results), older reports.
- `old_constraints/` — pre-validator_v2 hand-coded constraints (CICIoT + CICIDS).
- `src/attack/run_phase{0..4}_*.py`, `run_new_vae_attack_rerun.py`, etc. — superseded CICIoT runners.
