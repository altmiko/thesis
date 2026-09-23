# Independent code and artifact audit — CICIDS2017-DistriNet

## Audit verdict

**Current post-fix status: CONDITIONALLY TRUSTWORTHY for the audited CPU experiment configuration.**

The original audit findings remain below as the immutable “before” record. Section 22 records the
implemented fixes and post-fix reruns. The main attack is genuinely VAE-latent, every success and
validity flag is computed from the same final projected vector, fitted objects remain train/val
only as documented, and the independent metric auditor reports zero runner disagreements.

The former blockers are closed for the new `*_postfix` artifacts: each artifact stores row IDs,
Git state, a source-tree hash, run ID, complete config, checkpoint hashes, predictions, logits,
validity masks, costs, and controls; mask-rule provenance is train-only; timing dilation now uses
`exp(relu(mean_log_ratio))`; loaders enforce victim architecture and VAE class identity; and
per-sample summed losses remove material batch-composition coupling.

Post-fix CPU headline results are Primitive-Direct 78.4406% three-seed strict-valid ASR,
VAE-Latent-Primitive 8.6291%, Input PGD 99.63% targeted but 0% strict-valid, Raw 68.6036%
targeted but 0% strict-valid, and Masked 2.2677% strict-valid (seed42). These supersede the old
7.7% latent, 80.7% raw, and 6.9% masked tables. Final thesis claims should use only Section 22
and the `*_postfix` result directories.

## 1. Repository and experiment architecture map

| Stage | Canonical implementation/artifact | Audited behavior |
|---|---|---|
| Raw preprocessing | `scripts/preprocess_cicids2017_distrinet.py` | Five corrected DistriNet CSVs; mapping/filtering, global exact feature+label deduplication, within-source-label chronological split, train-only RobustScaler |
| Dataset contract | `src/datasets/cicids2017.py`; `preprocessing_manifest.json` | 79 ordered features; class order `Benign, DoS, DDoS, Recon, BruteForce`; scaler wrapper |
| Victims | `src/classifiers/cicids2017d_experiments.py`; `cicids2017d_victims.py` | MLP, CNN, LSTM, serial CNN-LSTM; 5-class category head used by attacks |
| VAE | `src/vae/cicids2017_stage_a.py`; `model.py`; `decoder.py` | Per-malicious-class typed beta-VAE; train-only fit; validation checkpoint selection and IDR calibration |
| Direct primitive baseline | `src/attack/run_cicids2017_primitive_attack.py` | Optimizes only two unconstrained leaves `(u,v)`, mapped to `(p, alpha)`; no VAE in gradient path |
| VAE latent primitive | `src/attack/vae_latent_primitive.py`; `run_cicids2017_vae_latent_attack.py` | Optimizes only `z_adv`; decoder -> inferred `(p,alpha)` -> dependency recomputation -> victim |
| VAE latent raw/masked | `src/attack/vae_latent_variants.py`; `run_cicids2017_latent_variants.py` | Raw diagnostic moves all features; masked moves 9 perturbable features and recomputes 7 exact-derived features |
| Independent validators | `evaluation/pave_style_validator.py`; `constraints/*`; `attack/realizability/validator.py` | PAVE Level A, mined density, and internal Level B/discreteness/frozen checks are separate from optimization |
| Saved results | `outputs/cicids2017_{primitive_attack,vae_latent_attack,input_baseline,...}` | Per-cell NPZ plus runner JSON; metrics independently recomputed by new `audit_results.py` |

## 2. Data-flow diagram

```mermaid
flowchart LR
    A[5 public DistriNet CSVs] --> B[Map/filter labels; parse timestamp]
    B --> C[Global exact feature+label dedup]
    C --> D[70/15/15 chronological within source label]
    D --> E[TRAIN]
    D --> F[VAL]
    D --> G[TEST]
    E --> H[Fit RobustScaler]
    E --> I[Train victims]
    F --> I
    E --> J[Train per-class beta-VAEs]
    F --> J
    F --> K[Calibrate latent Mahalanobis IDR]
    E --> L[Fit PAVE/ranges/CFF/mined constraints]
    G --> M[Fixed malicious test row IDs]
    H --> M
    I --> N[Clean prediction and clean-correct pool]
    J --> O[z0 = encoder(x0).mu]
    O --> P[Optimize z_adv only]
    P --> Q[Decoder raw proposal]
    Q --> R[Infer p, alpha or apply mask]
    R --> S[Project/quantize controls]
    S --> T[Recompute dependencies; FINAL x_adv]
    T --> U[Classify FINAL x_adv]
    T --> V[Validate SAME FINAL x_adv]
    U --> W[Per-sample artifact]
    V --> W
    W --> X[audit_results.py independent metrics]
    X --> Y[Thesis tables]
```

## 3. Frozen experiment state

### 3.1 Code state

- Git HEAD: `4147b03860b4993af8106e118b81c2f6003a1d1e` (`main`).
- The working tree was dirty. Tracked modifications included `src/classifiers/cicids2017d_experiments.py`, `src/vae/{diagnostics,losses,model}.py`; the CICIDS2017 attack, realizability, mask, dataset-adapter, reports, scripts, tests, and result directories were mostly untracked.
- Neither canonical `attack_results.json` nor per-cell NPZ files store a Git commit/source-tree hash.
- Consequence: `4147b03` identifies only the repository base, **not** the code that produced the attack outputs.

### 3.2 Python environments

| Environment | Python | NumPy | PyTorch | scikit-learn | Notes |
|---|---:|---:|---:|---:|---|
| Victim training manifest | 3.11.15 | 2.4.4 | 2.5.1+cu121 | 1.9.0 | CUDA, RTX 4070 Ti SUPER |
| `environment.yml` | 3.11 | 2.4.4 | 2.5.1+cu121 | 1.9.0 | Declared environment |
| Audit runtime | 3.12.3 | 2.4.4 | 2.5.1 CPU build | 1.9.0 | pandas 3.0.3, pytest 9.1.1; `matplotlib` missing |

The active runtime cannot start the classifier training script because `matplotlib` is absent. VAE/attack/audit commands do run. This is environment drift, not evidence that saved classifier outputs are wrong.

### 3.3 Dataset and split hashes

Raw input SHA-256 values are embedded in `preprocessing_manifest.json`: Monday `580bc5...71bc`, Tuesday `59d60e...19a`, Wednesday `820446...583`, Thursday `3db32a...365`, Friday `8bf5a7...9df`.

| Artifact | SHA-256 |
|---|---|
| `preprocessing_manifest.json` | `df93d07c2cd0978fab18f6171fffc7617dfb6dab829b192f2e69ec8cdcec8269` |
| `scaler.pkl` | `c04ed7034c7e735d326acfceebaea405246a556af543eca88e2dc5784c150c09` |
| `X_train.npy` | `a63cc5d15c9ad230ce63fc69dfd2dec3c17786503b37184aa73f8eb8d1d98118` |
| `X_val.npy` | `eadfc9649500e16c4deef127e965010869df0ae2bf4b9f288fe7995c2734381e` |
| `X_test.npy` | `35bf20bcbc5512cbce2096501265d06e6d36a01ec3bbc29a00e4146f7d4ef8de` |
| `X_train_pristine.npy` | `f77bcdd79645bd44d107ba9b38bdeffc505cd0b6e33d5f0686bba5ca487ff375` |
| `X_val_pristine.npy` | `77055e3039dfa0161ee52369f28e2d182c6f3b80cb099970dfc53d91ed3db3e1` |
| `X_test_pristine.npy` | `aa81135b10d03766803370cb49b1d1064e8076cf614479016c7cec7a3d3fe569` |
| `y_train_cat.npy` | `28dbc680357730bcf1d37e13a3d173432ea320b3aad7aaf02469d00a296000ef` |
| `y_val_cat.npy` | `092d1130023b3a48c723cffdb8a90ec390f0fd3eb23d2b319ca150c5147b9309` |
| `y_test_cat.npy` | `dc7f8f3c86d4e4c0304904f1037c7773e6abdf244fd2259389f3174a5542c345` |

### 3.4 Feature order

Source of truth: `preprocessing_manifest.json:modelling_feature_names` (79). Exact order:

```text
Src Port, Dst Port, Protocol, Flow Duration, Total Fwd Packet, Total Bwd packets,
Total Length of Fwd Packet, Total Length of Bwd Packet, Fwd Packet Length Max,
Fwd Packet Length Min, Fwd Packet Length Mean, Fwd Packet Length Std,
Bwd Packet Length Max, Bwd Packet Length Min, Bwd Packet Length Mean,
Bwd Packet Length Std, Flow Bytes/s, Flow Packets/s, Flow IAT Mean, Flow IAT Std,
Flow IAT Max, Flow IAT Min, Fwd IAT Total, Fwd IAT Mean, Fwd IAT Std, Fwd IAT Max,
Fwd IAT Min, Bwd IAT Total, Bwd IAT Mean, Bwd IAT Std, Bwd IAT Max, Bwd IAT Min,
Fwd PSH Flags, Bwd PSH Flags, Fwd URG Flags, Bwd URG Flags, Fwd Header Length,
Bwd Header Length, Fwd Packets/s, Bwd Packets/s, Packet Length Min, Packet Length Max,
Packet Length Mean, Packet Length Std, Packet Length Variance, FIN Flag Count,
SYN Flag Count, RST Flag Count, PSH Flag Count, ACK Flag Count, URG Flag Count,
CWR Flag Count, ECE Flag Count, Down/Up Ratio, Average Packet Size,
Fwd Segment Size Avg, Bwd Segment Size Avg, Fwd Bytes/Bulk Avg,
Fwd Packet/Bulk Avg, Fwd Bulk Rate Avg, Bwd Bytes/Bulk Avg, Bwd Packet/Bulk Avg,
Bwd Bulk Rate Avg, Subflow Fwd Packets, Subflow Fwd Bytes, Subflow Bwd Packets,
Subflow Bwd Bytes, FWD Init Win Bytes, Bwd Init Win Bytes, Fwd Act Data Pkts,
Fwd Seg Size Min, Active Mean, Active Std, Active Max, Active Min, Idle Mean,
Idle Std, Idle Max, Idle Min
```

`Flow ID`, source/destination IP, timestamp, sample/record IDs, source metadata, labels, and attempted/binary flags are excluded. Ports and protocol intentionally remain as model features.

### 3.5 Checkpoint hashes

| Checkpoint | SHA-256 |
|---|---|
| victim CNN | `51849974ae7bcd53bba49fa7c89ffc6b178f3e7ff9e6677b80df8f863cfd2a3f` |
| victim LSTM | `c3333fb3022dce3243a2b6f4c315438f9cc15f69d7f9cffb8517378e0ba670a5` |
| victim MLP | `a896ffccf88bac84f962c618f4fe7ee2e5bfb4a7066d93f32b175eaae987ff52` |
| victim Serial | `43f981a2e37fc72f46d12f45b455d0246a4d3e52a448d66b99ddea0424493095` |
| VAE DoS | `e2db90c27c61d821d73b9bcbbf379023b5498f78a83561738f381511eeae5d20` |
| VAE DDoS | `ccbb2da3b2e17bcdd88e0e33f486f6601651e76dd3e9f8b65310cb9e50f3bd0b` |
| VAE Recon | `d9ac6a38bea762b8c647844316ee701826a8e81d4ebe3cacb6e9dec883cd47bd` |
| VAE BruteForce | `e24bb3f50db450d25a7e860f125ff6becd5306e93fad113ce126431d08f231b3f` |

All current VAE checkpoint `class_name`, `class_id`, manifest hash, latent dimension, and beta fields pass the new identity test. Production loaders still do not enforce every identity.

### 3.6 Canonical VAE-Latent-Primitive attack and validator config

```json
{
  "steps": 120,
  "learning_rate": 0.08,
  "objective": "cw",
  "kappa": 0.0,
  "lambda_cls": 1.0,
  "lambda_latent": 0.005,
  "lambda_cost": 0.05,
  "lambda_realism": 0.001,
  "lambda_recon": 0.0,
  "epsilon_z": 10.0,
  "init_noise": 0.3,
  "restarts": 1,
  "p_max": 1460.0,
  "alpha_max": 100.0,
  "mtu_cap": 0.0,
  "test_limit_per_class": 1024,
  "seeds": [42, 43, 44],
  "target_class": 0
}
```

Validators: PAVE integer/range tolerance `1e-6`, fitted on pristine train; mined A4 Layer-1 fit on first 200,000 pristine train rows plus `constraints/cicids2017_distrinet/mined.json`; internal realizability defaults `atol=1e-3`, `rtol=1e-4`, frozen `atol=1e-6`, integer `atol=1e-3`. Strict validity is `PAVE & mined & realizable` for primitive methods and `PAVE & mined & mask_valid` for masked methods.

## 4. Split and leakage audit

- Split: 70/15/15 chronological **within each retained source attack label**, not global chronological and not random (`preprocess...:373-471`). Rows: train 1,456,265; val 312,058; test 312,056.
- Split membership is computed before fitted preprocessing.
- Exact float32 feature+category duplicates are removed globally before the split: 15,754/2,096,133 (0.752%).
- Train/val/test sample IDs are pairwise disjoint and exhaustive.
- Train-vs-val and train-vs-test have zero feature-only and zero feature+label fingerprints in common.
- Val-vs-test has one identical 79-feature vector with different labels (Benign vs Recon); no feature+label collision and no shared sample ID. This is label ambiguity, not direct train/test leakage.
- RobustScaler, class weights, constant-column audit, primitive envelopes, CFF, PAVE, and mined constraints are train-only.
- Victim fitting uses train; validation selects checkpoints/early stopping; test is scored afterward.
- Per-class VAE fitting uses train; validation selects checkpoint and fits the p95 Mahalanobis/IDR gate; test is not loaded during Stage A.
- The split supports closed-set within-source-label forward holdout. It does **not** support a global future-campaign generalization claim; the manifest states this.
- **Failure:** `attack/masks/cicids2017_distrinet.py:6-12,85-129` describes train+test cross-checks as provenance for promoted formulas. Recreate the exact rule-selection log using train only; use val only for confirmation/calibration; reserve test for final evaluation.

## 5. Model and checkpoint audit

### Victims

Class order is `Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`. Loaders set `eval()`, freeze parameters, enforce 79 input features and 5 classes, and use strict state-dict loading. Checkpoints contain model type/kwargs and dimensions, but no feature-order hash, label-map hash, dataset ID, or training-split hash.

Clean test metrics independently recomputed from saved prediction arrays:

| Victim | Accuracy | Benign | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 98.448% | 98.151% | 99.417% | 99.846% | 99.660% | 98.081% |
| CNN | 98.433% | 98.155% | 99.250% | 99.853% | 99.614% | 97.697% |
| LSTM | 98.387% | 98.214% | 98.306% | 99.853% | 99.413% | 97.697% |
| Serial | 98.426% | 98.148% | 99.339% | 99.853% | 99.497% | 97.697% |

### VAEs

Four VAEs: DoS, DDoS, Recon, BruteForce. Each uses only train rows of its class; validation selects the best ELBO checkpoint and calibrates IDR. `latent_dim=16`, beta target `0.5` after 10-epoch warmup, free bits `0.1`, Laplace reconstruction likelihood, typed raw-space decoder. `encode()` returns `(mu, logvar)`; the attack explicitly takes `mu`. `decode()['continuous_mu_raw']` is in raw feature units.

The checkpoint stores class and manifest metadata, but `load_stage_a()` verifies only the manifest hash. A wrong-class VAE with the same schema can load. New tests verify the current checkpoint set, but production loading must accept an expected class and assert it.

## 6. Scaler audit

The canonical arrays remain separate: pristine raw `X_*_pristine.npy` and RobustScaler-space `X_*.npy`. Full scan across all three splits (164,349,941 values):

| Statistic | Absolute raw->scale->inverse error |
|---|---:|
| finite | 100% |
| mean | 0.0087269 |
| median | 0 |
| p99 | 0.0009765625 |
| p99.9 | approximately 4.0 (deterministic 1.63M-value sample) |
| maximum | 8.0 |

Per-feature maxima of 8 occur on large timing/rate features, including Flow Duration, forward/backward IAT fields, Flow IAT fields, and Flow Bytes/s. A large raw-space tolerance would hide real errors. The attack correctly avoids this: pristine raw test rows are loaded directly, frozen fields are copied from those rows, and only the final adversarial vector is scaled for the victim. `SCALER_ATOL=1e-6` is therefore a frozen-copy tolerance, not a roundtrip tolerance.

## 7. VAE gradient-path audit

`vae_latent_primitive.py:162-206` creates exactly one optimizer leaf, `z_adv`; VAE and victim parameters are frozen. Graph:

```text
targeted loss -> victim(final continuous raw scaled) -> primitive.generate
-> inferred p/alpha -> decoder raw output -> z_adv
```

Evidence:

- `optimizer_parameters(z_adv) == [z_adv]`; `p` and `alpha` are non-leaf tensors.
- Classifier gradient to `z_adv` is finite and non-zero.
- Detaching decoder output makes logits independent of `z_adv`.
- Decoder movement changes controls and the generated vector.
- `z_adv=z0` maps to `p=0`, `alpha=1`.
- 34 core attack tests pass.

The method is genuinely VAE-latent. The direct primitive baseline separately optimizes only `(u,v)->(p,alpha)` and uses the VAE only for the post-hoc IDR gate.

## 8. Attack-method and final-projection audit

### Final order

All canonical primitive/latent-primitive paths perform:

```text
optimize continuous representation
-> infer continuous controls
-> round p and enforce alpha activity
-> quantize timing to integer microseconds
-> recompute dependencies
-> FINAL x_adv_raw
-> classify FINAL vector
-> validate SAME FINAL vector
-> save predictions/masks
```

`run_cicids2017_vae_latent_attack.py:111-143` passes `res.x_adv_realized_raw` to both `evaluate_cell()` and the artifact writer. Restart selection also uses the final realized target margin. There is no audited path where ASR is pre-projection and validity post-projection.

### VAE-Latent-Masked

Only these 9 features receive direct decoder movement: Flow Duration; Total Length of Fwd Packet; Fwd Packet Length Max/Min/Std; Fwd IAT Total/Std/Max/Min. Seven exact-derived targets are recomputed: Fwd Packet Length Mean, Fwd Segment Size Avg, Fwd IAT Mean, Fwd Packets/s, Bwd Packets/s, Flow Packets/s, Flow Bytes/s. The remaining 63 features are structurally preserved from pristine raw. Tests confirm changed features are within perturbable+dependency closure.

### Primitive mapping and units

- `p` is bytes added per forward packet. Signals are forward mean/max/min changes and total-forward-length change divided by raw forward count. This is unit-consistent.
- `alpha` is a ratio. The signal is the mean log ratio of decoded Fwd IAT Total, Fwd IAT Mean, and Flow Duration.
- **Issue:** code maps that signal as `alpha = 1 + relu(mean_log_ratio)`. Exact inversion of a log ratio is `exp(relu(mean_log_ratio))`. The implemented map is an explicit heuristic only if documented as such; current reports present it as a dilation derived from log ratios without justifying the linearization.
- For `N_fwd<2`, alpha is forced to 1 inside the graph. New regression coverage confirms zero alpha gradient and zero timing-feature change.

## 9. Realizability/dependency audit

The primitive implementation recomputes forward totals/min/max/mean, forward timing, conservative duration and Flow-IAT max, combined packet min/max/mean/std/variance, Average Packet Size, Fwd Segment Size Avg, and all four rates. Integer length/timing outputs are quantized. Frozen values come from pristine raw.

Classification:

- Primitive-derived length: Total Length of Fwd Packet, Fwd min/max/mean, Fwd Segment Size Avg.
- Primitive-derived timing: Fwd IAT total/mean/std/max/min.
- Exact/conditional combined: packet min/max/mean/std/variance, Average Packet Size, Flow Duration, Flow IAT mean/max.
- Rate-derived: Flow Bytes/s, Flow/Fwd/Bwd Packets/s.
- Proven invariant: Fwd Packet Length Std under uniform length shift.
- Level-C unresolved and held constant: Subflow Fwd Bytes, bulk fields, Flow IAT Std/Min, Fwd Act Data Pkts, Active/Idle fields.

No PCAP is edited or re-extracted. The supportable wording is **realizability-aware, feature-space dependency-consistent, Level A+B validated**. Level C is unproven. Current CICIDS2017 reports correctly state this limitation.

## 10. Validator independence audit

The generator embeds only structural projection/recomputation needed by the method. It does not call the complete PAVE, mined, or internal validator and repair until success. External evaluation is separate:

- PAVE: domain/type/range (Level A), train-fit.
- Mined A4 engine: learned density/constraints, train-fit.
- Internal realizability validator: algebraic, packet summary, timing, rates, discreteness, and frozen consistency (Level B).
- IDR: validation-calibrated Mahalanobis in the same VAE latent space.

IDR is **generator-relative**, not independent realism evidence. PAVE/mined/internal checks remain independent.

## 11. Metric and denominator audit

New `audit_results.py` imports no attack runner metric helper. It loads NPZ arrays, rejects NaN/Inf, reconstructs strict masks, recomputes cell/class/victim/micro/macro counts and costs, and compares with runner JSON at float32 tolerance. It found zero disagreements across 144 canonical cells.

| Method | Seed | Clean-correct N | Targeted-Benign ASR | Targeted strict-valid ASR | Mean cost |
|---|---:|---:|---:|---:|---:|
| Primitive-Direct | 42 | 16,228 | 78.4632% | 78.4632% | 2.66679 |
| VAE-Latent-Primitive | 42 | 16,228 | 7.7335% | 7.7274% | 0.07177 |
| Input PGD | 42 | 16,228 | 99.6056% | 0.0000% | 0.41630 |

Three-seed means: Primitive 78.4426% strict-valid (population std 0.1017 percentage points); latent primitive 7.7253% (0.0780 pp); input PGD 0%. Untargeted and targeted labels are stored separately. Misclassification into another attack is not counted as targeted-Benign success.

The latent ladder independently reproduces: Raw 80.655% targeted but 0% strict-valid; Masked 6.908% strict-valid; strengthened Primitive 10.294% strict-valid.

## 12. Same-sample and budget fairness

Canonical input PGD, direct primitive, and proposed latent primitive use the same dataset, victims, class IDs, target 0, scaler, selected 1,024 rows/class, and 16,228 clean-correct pooled denominator. Seeds vary optimization initialization while rows stay fixed.

The main 78.5% vs 7.7% comparison is not a matched-cost comparison (mean costs 2.67 vs 0.072). The later bottleneck report adds post-hoc cost-capped success at shared caps. That is useful but is not the same as optimizing both attacks under an identical hard budget. Claims should be “higher ASR at the reported cost/configuration” and separately show the cost-capped curve; avoid an unconditional “stronger under equal budget” claim.

## 13. Reproducibility and sanity controls

### Same-seed rerun

Canonical DoS/LSTM/seed42, 1,024 rows, CPU, run twice:

- Every saved A/B array was bit-identical: clean predictions, `z0`, `z_adv`, controls, final vectors, predictions, validity masks, success, and costs.
- Against original CUDA artifact: `z0` max difference `5.36e-7`; `z_adv` `2.00e-5`; raw final vector max difference 4 (large-valued feature); alpha `1.91e-6`; p/predictions/validity/success exact. Both had 578 target successes and 577 strict-valid target successes.
- Attack runners seed Python, NumPy, and PyTorch, but do not explicitly enable deterministic algorithms/cuDNN deterministic mode. Classifier/VAE training code does set deterministic cuDNN mode.

### Controls

| Control | Observation |
|---|---|
| zero steps, no init noise | `z_adv=z0`, `p=0`, `alpha=1`, no feature/prediction change |
| epsilon_z=0 | same exact identity behavior |
| decoder detached | no gradient path to z |
| no classifier loss | target-Benign 0% vs optimized 46.81% on fixed 256-row DoS/LSTM control |
| matched-norm random latent | 0.43% target-Benign vs optimized 46.81% |
| wrong target DDoS | 0% DDoS target success; incidental 3.40% Benign was not counted as target success |
| clean validators | 100% internal and PAVE validity on control cohort |
| single-forward-packet | alpha=1, alpha gradient=0, timing change=0 |

### Batch-size sensitivity

A 50-row rerun did not reproduce the same 50 rows from the 1,024-row optimization; replaying the full 1,024 rows did. Loss terms are globally averaged before Adam, so gradient scaling and Adam epsilon make results batch-size dependent. `test_limit_per_class=1024` therefore acts as an optimization batch-size parameter and must remain frozen/recorded.

## 14. Statistical and paired-validity audit

- Seeds 42/43/44 produce different adversarial-vector hashes in all 16 cells for each canonical method; they are independent seeded attack initializations over identical rows, not duplicated outputs.
- The reported mean/std is descriptive optimization-seed variability. It is not evidence of independent dataset replication.
- Primitive bootstrap intervals resample clean-correct rows from seed42; they do not concatenate three versions of each row.
- For primitive, latent primitive, and input PGD artifacts, independently checked `valid_success = raw_success & strict_valid`; the impossible McNemar cell `c = count(~raw_success & valid_success)` is exactly zero.
- Any paired method comparison must retain row ID/victim/class alignment. Current NPZ files omit row IDs, so alignment is reconstructed from runner selection logic rather than artifact-local identity.

## 15. Manual sample audit

Fifteen seed42 VAE-Latent-Primitive rows were checked. `eq residual` is maximum absolute residual over audited length/timing/rate equations; large-valued rates are also subject to relative tolerance. Every row had zero changes outside the primitive dependency closure.

| Type | Row ID | Class/victim | Clean->final | p | alpha | Cost | PAVE/mined/internal | Frozen changed | Max eq residual |
|---|---|---|---:|---:|---:|---:|---|---:|---:|
| success-valid | Tuesday:168031 | BruteForce/CNN | 4->0 | 12 | 1.9853 | .0969 | T/T/T | 0 | 1e-6 |
| success-valid | Tuesday:313502 | BruteForce/LSTM | 4->0 | 3 | 1.4989 | .1227 | T/T/T | 0 | .00893 |
| success-valid | Tuesday:144478 | BruteForce/Serial | 4->0 | 13 | 1.8550 | .1413 | T/T/T | 0 | 2e-6 |
| success-valid | Friday:196533 | DDoS/CNN | 2->0 | 13 | 2.1330 | .2810 | T/T/T | 0 | .0714 |
| success-valid | Wednesday:468691 | DoS/CNN | 1->0 | 102 | 1.9598 | .2730 | T/T/T | 0 | .000823 |
| unsuccessful | Tuesday:302016 | BruteForce/CNN | 4->4 | 13 | 1.8706 | .1281 | T/T/T | 0 | .05 |
| unsuccessful | Tuesday:302016 | BruteForce/LSTM | 4->4 | 0 | 1 | 0 | T/T/T | 0 | .025 |
| unsuccessful | Tuesday:302016 | BruteForce/MLP | 4->4 | 0 | 1 | 0 | T/T/T | 0 | .025 |
| unsuccessful | Tuesday:302016 | BruteForce/Serial | 4->4 | 13 | 1.8714 | .1282 | T/T/T | 0 | .05 |
| unsuccessful | Friday:302346 | DDoS/CNN | 2->2 | 0 | 1 | 0 | T/T/T | 0 | 6.4e-5 |
| invalid | Wednesday:15812 | DoS/CNN | 1->1 | 0 | 1 | 0 | T/F/T | 0 | ~0 |
| invalid | Wednesday:28737 | DoS/LSTM | 1->1 | 106 | 1.1364 | .4729 | F/T/T | 0 | 3.0 |
| invalid | Wednesday:28737 | DoS/MLP | 1->1 | 115 | 1.1364 | .4937 | F/T/T | 0 | 3.0 |
| invalid | Wednesday:28737 | DoS/Serial | 1->1 | 109 | 1.1364 | .4798 | F/T/T | 0 | 3.0 |
| invalid | Wednesday:15830 | DoS/CNN | 1->1 | 0 | 1 | 0 | T/F/T | 0 | ~0 |

Invalid cases show the validators are not tautological repairs: internal dependency consistency can pass while independent PAVE or mined density rejects the same final vector.

## 16. Classified findings

| ID/severity | File/function | Problem and why it matters | Affected experiments | Recommended fix | Rerun? |
|---|---|---|---|---|---|
| H1 HIGH | Git state; all CICIDS attack outputs | Attack implementation/results are untracked/dirty and artifacts contain no source hash. Exact generating code is unknowable. | All current CICIDS attack conclusions | Commit code; store commit plus dirty diff hash in every run manifest/artifact | **Yes**, final publication runs |
| H2 HIGH | `attack/masks/cicids2017_distrinet.py:6-12,85-129` | Rule provenance cites TEST cross-check. Test-informed rule promotion can bias mask/validity and ASR. | VAE-Latent-Masked and comparisons involving its mask; possibly architecture claims | Reconstruct/freeze formulas from train only; validate on val; never consult test until final | If frozen rules differ: **yes** |
| H3 HIGH | `realizability/cicids2017.py:322-335` | `alpha=1+relu(mean log ratio)` is not exact conversion from log-time movement; `exp()` would be. No mathematical justification for linearization. | VAE-Latent-Primitive, strong/sweep/ablation results | Decide/document heuristic or change to exponential and recheck bounds/gradients | If changed: **yes** |
| H4 HIGH | `CICIDS2017_vae_latent_attack_report.md:172-175,228-231` vs `CICIDS2017_latent_bottleneck_report.md:212-237` | One report concludes “manifold-limited”; later report says explicitly “not manifold/latent,” but mask/constraint bottleneck. Thesis interpretation is internally contradictory. | Narrative/conclusion | Mark older report superseded; use ladder/gradient evidence and cautious causal wording | No numerical rerun |
| H5 HIGH | `cicids2017d_victims.py:38-77`; `cicids2017_stage_a.py:311-332` | Victim loader checks only dimensions; VAE loader checks manifest but not expected class. Wrong same-shape victim/wrong-class VAE can load silently. | Any rerun with misnamed/copied checkpoint | Store/assert dataset, class order, feature hash/order, split hash, class name, architecture, checkpoint SHA | No for current checked set; yes if mismatch found |
| M1 MEDIUM | Canonical NPZ/JSON writers | NPZ lacks row ID, dataset/method/victim/seed, Git/config/checkpoint IDs. JSON has some context but no Git/checkpoint hashes. | Traceability and paired analysis | Add self-contained provenance or immutable sidecar with content hashes | Regenerate final artifacts |
| M2 MEDIUM | `vae_latent_primitive.py:179-195`; variants equivalent | Global mean reduction makes per-sample Adam outcomes batch-size sensitive. | All latent results and golden comparisons | Freeze/store attack batch size; preferably optimize independent per-sample objectives or prove invariance | If implementation changes: yes |
| M3 MEDIUM | `environment.yml`; active runtime | Declared/original environment is Python 3.11 CUDA; audit runtime is Python 3.12 CPU and lacks matplotlib, blocking classifier CLI. | Full reproduction | Create lock file; test clean environment; record CUDA/cuDNN/driver | Re-run smoke; final runs in locked env |
| M4 MEDIUM | Attack `_seed()` functions | No explicit `torch.use_deterministic_algorithms`/cuDNN flags in attack runners. CPU is deterministic; GPU guarantee is incomplete. | GPU attack reproducibility | Enable/report deterministic mode or document allowed tolerance | Re-run final 3 seeds |
| M5 MEDIUM | `CICIDS2017_vae_latent_attack_report.md:198-200` | Report says decoded vectors and logits are saved; canonical NPZ keys do not contain them. | Gradient/instrumentation provenance | Save claimed arrays or correct report | No metric rerun |
| M6 MEDIUM | `run_cicids2017_latent_variants.py:141-142` vs primitive strict gate | “Strict validity” is method-specific: mask closure for masked, primitive realizability for primitive. Rates are meaningful but not one identical Level-B checker. | Cross-method validity comparisons | Name gates explicitly and report common PAVE+mined plus method-specific structural gate | No |
| M7 MEDIUM | Multiple `outputs/cicids2017_*` dirs | Proposed config (7.7%, 1024 rows, 3 seeds) and strong config (10.3%, 512 rows, 1 seed) coexist with similar method IDs. Easy to mix. | Result-table provenance | Unique immutable run ID from config+code+data hashes; `LATEST` only as explicit pointer | No |
| L1 LOW | `run_cicids2017_primitive_attack.py` JSON | `method_id` is absent/null while other runners set it. | Tooling/provenance | Add `method_id="primitive_direct"` | No |
| L2 LOW | `leakage_audit.json:33-63` | One val/test feature-identical row has conflicting labels. | Validation interpretation only | Document; investigate source-label ambiguity | No |
| L3 LOW | legacy CICIoT runners/EDA (`eda_figures_part1.py`, `latent_infra.py`, old reports) | Hardcoded `D:/thesis_final`, TODO mask, duplicate validators/metric pipelines, stale compatibility branches remain near current code. | Operator error, not canonical CICIDS calculations | Archive or clearly namespace legacy paths; never use them for CICIDS tables | No |

**CRITICAL findings: none observed.** This means the saved headline arithmetic is supported, not that the final trust gate passes.

## 17. Fixes applied during this audit

Method-neutral only; no old result file was overwritten:

1. Added `audit_results.py`, an independent NPZ metric auditor. Output: `outputs/audit_results.json`.
2. Allowed `steps=0` and `epsilon_z=0` in `LatentAttackConfig`; defaults and positive configurations are unchanged.
3. Added zero-step, zero-radius, and single-packet no-gradient/no-timing-cost regression checks.
4. Added `tests/test_experiment_identity.py` for schema/scaler width, classifier manifest hash, all victim dimensions, and all VAE class/schema/beta identities.
5. Added immutable 50-row golden fixture `tests/data/golden_cicids2017_dos_lstm_seed42.npz` and `tests/test_golden_attack_regression.py`. The test replays the original 1,024-row batch and reports exact/numeric field drift.
6. Generated this audit report. No attack formula, checkpoint, scaler, split, or reported result was changed.

Verification performed:

```text
34 passed — core attack/mask/realizability tests
4 passed  — experiment identity tests
2 passed  — full-batch golden regression
0 disagreements — independent audit over 144 canonical cells
0 NaN/Inf fields — 192 checked canonical/variant NPZ files
```

## 18. Experiments requiring rerun

1. **Mandatory final thesis rerun:** after committing/fingerprinting the exact code and adding artifact provenance. Preserve old outputs in a read-only dated directory.
2. **Conditional masked rerun:** if train-only reconstruction of mask/rule selection differs from current rules.
3. **Conditional latent-primitive rerun:** if alpha mapping is changed from linearized log-ratio to exponential or otherwise redefined.
4. **Mandatory final reproducibility run:** three seeds in a locked Python/CUDA environment with deterministic settings and explicit batch size.
5. No rerun is needed merely for `audit_results.py`, identity tests, zero-control support, or report corrections; these do not alter positive-step attack behavior.

## 19. Trusted versus untrusted existing results

### Supported by current artifacts

- Current preprocessing manifest, split membership claims, and train-only scaler fitting.
- Current victim test predictions and clean metrics.
- Canonical per-sample arithmetic for:
  - `outputs/cicids2017_primitive_attack`;
  - `outputs/cicids2017_vae_latent_attack`;
  - `outputs/cicids2017_input_baseline`.
- Latent ladder artifact arithmetic for Raw, Masked, and strengthened Primitive.
- Genuine VAE gradient path, final post-projection evaluation, clean-correct denominator, and Level A+B terminology.
- Three-seed optimization variability for canonical methods.

### Not yet trustworthy as final thesis evidence

- Any claim that the artifacts came from Git commit `4147b03`; they did not record a code identity and the generating code is untracked.
- The older “manifold-limited” causal conclusion; it conflicts with the newer ladder audit.
- Independent realism claims from IDR; generator and evaluator share the same VAE.
- Packet-level/Level-C realizability; no PCAP edit/re-extraction exists.
- Masked-method results as leakage-free until train-only rule-selection provenance is demonstrated.
- Strong-run 10.3% as a three-seed headline; it is seed42 only and uses 512 rows/class.
- Any old CICIoT/legacy report or hard-coded Markdown number not tied to an explicit artifact directory and configuration.

## 20. Exact reproduction and verification commands

Run from repository root. On POSIX/Git Bash, `PYTHONPATH=src` is required because the project is not installed as a package.

```bash
# Preprocess (destructive to the named output directory; use a fresh directory for verification)
python scripts/preprocess_cicids2017_distrinet.py \
  --input-dir data/raw/CICIDS_2017_Distrinet \
  --output-dir data/processed/CICIDS_2017_Distrinet_repro

# Victims — requires the declared environment, including matplotlib and CUDA for original parity
PYTHONPATH=src python src/classifiers/cicids2017d_experiments.py \
  --processed-dir data/processed/CICIDS_2017_Distrinet \
  --output-dir outputs/cicids2017distrinet_repro \
  --models all --device cuda --epochs 10 --batch-size 2048 \
  --learning-rate 0.001 --patience 3 --class-weighting balanced --seed 42

# Per-class beta-VAEs
PYTHONPATH=src python src/vae/cicids2017_stage_a.py \
  --classes DoS,DDoS,Recon,BruteForce \
  --output-dir outputs/cicids2017_vae_attacks_repro/stage_a \
  --device cuda --epochs 30 --batch-size 2048 --patience 5

# Canonical proposed VAE-Latent-Primitive
PYTHONPATH=src python -m attack.run_cicids2017_vae_latent_attack \
  --classes DoS,DDoS,Recon,BruteForce --victims mlp,cnn,lstm,serial \
  --device cuda --test-limit 1024 --steps 120 --learning-rate 0.08 \
  --objective cw --epsilon-z 10 --kappa 0 --restarts 1 \
  --lambda-latent 0.005 --lambda-cost 0.05 --lambda-realism 0.001 \
  --p-max 1460 --alpha-max 100 --mtu-cap 0 --seeds 42,43,44 \
  --stage-a-dir outputs/cicids2017_vae_stage_a \
  --output-dir outputs/cicids2017_vae_latent_attack_postfix

# Independent artifact audit
python audit_results.py \
  outputs/cicids2017_primitive_attack_postfix \
  outputs/cicids2017_input_baseline_postfix \
  outputs/cicids2017_vae_latent_attack_postfix \
  outputs/cicids2017_latent_raw_postfix \
  outputs/cicids2017_latent_masked_postfix \
  --json outputs/audit_results_postfix.json

# Verification
python -m pytest \
  src/attack/tests/test_vae_latent_primitive.py \
  src/attack/tests/test_latent_variants.py \
  src/attack/tests/test_primitive_controls.py \
  tests/test_experiment_identity.py \
  tests/test_golden_attack_regression.py -q
```

## 21. Final verification gate

| Gate | Post-fix status |
|---|---|
| same-seed reproducibility | PASS; deterministic CPU reruns and golden replay |
| no train/test leakage | PASS; mask rule provenance is explicitly train-only |
| schema consistency | PASS |
| scaler audit | PASS |
| checkpoint identity checks | PASS; victim architecture and VAE class enforced |
| zero-radius sanity | PASS |
| zero-step sanity | PASS |
| decoder-gradient test | PASS |
| batch-composition stability | PASS within numerical tolerance |
| final post-projection evaluation | PASS |
| independent metric recomputation | PASS; zero disagreements for all post-fix cells |
| clean-correct denominator match | PASS |
| artifact provenance | PASS; all 176 artifacts contain every required field |
| NaN/Inf audit | PASS; zero non-finite fields in all 176 artifacts |
| manual sample audit | PASS; 5 success-valid, 5 unsuccessful, 5 invalid post-fix rows |
| 3-seed stability | PASS for Primitive, Input PGD, and Latent-Primitive |

**Decision:** the `*_postfix` CPU results pass the requested code/artifact verification gates.
They are traceable to exact source-tree hashes despite the working tree being dirty during the
runs. Commit the code and report without regenerating or silently replacing these artifacts.
The older non-postfix result directories remain historical only.

## 22. Post-fix closure and rerun results

### 22.1 Implemented corrections

| Original finding | Closure |
|---|---|
| H1 untraceable source/artifacts | Closed: immutable run manifests and per-artifact Git/source/data/scaler/checkpoint/config identities |
| H2 TEST cited in mask-rule promotion | Closed: rule provenance is train-only; test cannot promote/remove/tune rules |
| H3 linearized log-time mapping | Closed: `alpha = exp(relu(mean_log_ratio))`; all affected latent results rerun |
| H4 contradictory causal reports | Closed for current report: old tables are historical; this section is authoritative |
| H5 weak checkpoint loading | Closed: expected victim model type and VAE class/class ID are mandatory |
| M1 incomplete artifact fields | Closed: required provenance and per-sample fields present in every artifact |
| M2 batch-size coupling | Closed: independent per-sample losses are summed before Adam; batch-composition regressions pass |
| M4 incomplete deterministic setup | Closed: attack runners seed Python/NumPy/Torch and enable deterministic Torch/cuDNN policy |
| M5 missing decoded/logit arrays | Closed for latent postfix artifacts |
| L1 missing primitive method ID | Closed: `primitive_direct` is explicit |

### 22.2 Independent post-fix results

| Method | Seeds | Clean-correct N/seed | Targeted ASR | Targeted strict-valid ASR | Mean normalized cost |
|---|---:|---:|---:|---:|---:|
| Primitive-Direct | 42/43/44 | 16,228 | 78.4406% mean | **78.4406% ± 0.0994 pp** | 2.6669–2.7493 |
| Input PGD | 42/43/44 | 16,228 | **99.6323% mean** | **0.0000%** | 0.4163–0.4166 |
| VAE-Latent-Primitive | 42/43/44 | 16,228 | 8.6332% mean | **8.6291% ± 0.0947 pp** | 0.1025–0.1035 |
| VAE-Latent-Raw | 42 | 16,228 | **68.6036%** | **0.0000%** | 0.8071 |
| VAE-Latent-Masked | 42 | 16,228 | 2.2738% | **2.2677%** | 0.2264 |

`audit_results.py` independently reconstructed every denominator, success count, strict mask,
and cost from NPZ arrays. Runner disagreements: zero.

### 22.3 Provenance identities

| Output directory | Run ID | Source-tree SHA-256 prefix | Artifacts |
|---|---|---|---:|
| `cicids2017_primitive_attack_postfix` | `d41b35fcaabf4a86cc49` | `b8c85e93fce2f492` | 48 |
| `cicids2017_input_baseline_postfix` | `4690c0b7f1fb8ff4d3c6` | `d7ea91ad1136f57a` | 48 |
| `cicids2017_vae_latent_attack_postfix` | `ffbdef9093e634e82e3d` | `d7ea91ad1136f57a` | 48 |
| `cicids2017_latent_raw_postfix` | `027628559931010a1a14` | `d7ea91ad1136f57a` | 16 |
| `cicids2017_latent_masked_postfix` | `b6a0af02623577766335` | `d7ea91ad1136f57a` | 16 |

All runs record Git `7f2a781881aafa4de26da08636e98e35fdad8535` plus dirty-state and exact
source-tree hashes. All ran on Python 3.12.3 / PyTorch 2.5.1 CPU. This is a reproducible CPU
configuration; it is not represented as a CUDA rerun.

### 22.4 Before/after interpretation

- Exponential timing conversion raises canonical latent strict-valid ASR from 7.73% to 8.63%
  and mean cost from about 0.072 to 0.103.
- Removing batch coupling lowers Masked seed42 strict-valid ASR from 6.91% to 2.27%.
- Raw remains highly evasive but entirely invalid, now 68.60% targeted rather than 80.66%.
- Direct Primitive and Input PGD remain effectively unchanged.
- The corrected evidence still supports the same qualitative conclusion: the valid latent methods
  are much weaker than direct primitive optimization, while unconstrained Raw/Input success does
  not survive strict validity. Do not reuse the old numerical tables.

### 22.5 Final regression evidence

```text
171 passed, 1 non-failing PyTorch same-padding warning, 25.83 s
```

The initial collection attempt exposed that the active Python lacked Matplotlib even though
`environment.yml` declares `matplotlib==3.11.0`. Installing that declared pin into the active
interpreter resolved collection; no test or source suppression was used.

### 22.6 Post-fix manual sample audit

`outputs/manual_sample_audit_postfix.json` records 15 inspected seed42 latent-primitive rows:
5 successful-valid, 5 unsuccessful, and 5 invalid across BruteForce, DDoS, DoS and multiple
victims. Every sample had zero changes outside the primitive/dependency closure. Successful
samples classified to Benign while passing PAVE, mined, and internal realizability; unsuccessful
samples preserved their non-Benign prediction; invalid examples were independently rejected by
PAVE or the mined gate while internal dependency consistency still passed. This confirms the
external validators are not generator-embedded tautologies.
