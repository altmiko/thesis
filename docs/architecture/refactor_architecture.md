# Generic Constrained-Adversarial VAE Platform — Refactor Report

Status: Phases A–F implemented and verified in the `thesis` conda env
(torch 2.5.1+cu121). All new tests green: **49 passed**
(`datasets` 19, `vae` 10, `constraints` 9, `attack` 8, `experiments` 3).

Jacobian / JSMA / tangent-space / saliency guidance is deliberately **not**
implemented (deferred, per spec).

---

## 1. Architecture before vs after

**Before**
```
CICIoT arrays ─► MixedInputBetaVAE (n==39 hard-gated, scaler baked in as buffers,
                 literal indices ttl=2/min=31/…, silent center=0 scale=1 fallback)
              ─► single continuous structured decoder
              ─► attack: decode_to_39 + hand-listed PerturbationMask + protocol reimpose
              ─► post-hoc validator (G1–G8) + physics + Mahalanobis (mixed gen/eval)
```
Feature semantics were tied to positions; constraints were imposed both structurally
and as (often vacuous) losses; diagnostics crashed under the continuous schema.

**After**
```
DatasetAdapter ─► FeatureManifest ──► FeatureTransform (train-only, fail-loud)
                        │
                        ▼
              Generic typed VAE (dynamic dims, per-value-type heads)
                        │  base manifold reconstruction
                        ▼
              Masked residual adversarial head  x_cand = x + M⊙Δ
                        ▼
              Layer 0 hard projector (P0) + immutable copy + exact derived recompute
                        ▼
              Layer 1 generic soft (C1) + Layer 2 dataset/mined soft (C2)
                        ▼
              x_adv ─► Victim classifier (injected)   ─► Stage-B training
                    └► Independent validators V0/V1/V2 (hierarchical ASR)
```

---

## 2. Files changed / added

| File | Purpose | Main changes |
|---|---|---|
| `src/datasets/feature_manifest.py` | **new** | `FeatureSpec`/`FeatureManifest`; 8 value types; order contract; `content_hash`; fail-loud `assert_matches_*` |
| `src/datasets/transforms.py` | **new** | `FeatureTransform` train-only fit; `TransformNotFittedError` (no silent identity); `from_sklearn_scaler` wraps existing scaler |
| `src/datasets/base.py` | **new** | `DatasetAdapter`, `ClassMapping`, `Split` |
| `src/datasets/ciciot2023.py` | **new** | manifest built from `preprocessing.schema` (no re-authoring); flags/services typed `probability` |
| `src/datasets/cicids2017.py` | **completed** | 79-feature DistriNet manifest; five-category labels; saved train-fit RobustScaler; train/val/test arrays |
| `src/datasets/__init__.py` | **new** | public API + `get_adapter` registry |
| `src/vae/decoder.py` | **new** | `TypedDecoder` per-value-type heads (softplus/sigmoid/bounded), manifest-driven |
| `src/vae/model.py` | **rewritten** | dynamic `n_features`; manifest+typed or legacy path; injected transform; optional `asinh` encoder stabilization; fail-loud decode |
| `src/vae/losses.py` | **extended** | `compute_manifold_elbo` (recon+βKL+C0 pre-proj+C1+C2); legacy `compute_elbo` retained for A0 |
| `src/vae/diagnostics.py` | **fixed** | guarded empty protocol/binary heads and `None` `protocol_idx_batch` (the audit crashes) |
| `src/constraints/base.py` | **new** | `Constraint` ABC (penalty + validate), `ConstraintReport` |
| `src/constraints/layer0.py` | **new** | domain clamps, immutable copy, manifest-driven `identity`/`square`/`product` recompute, `soft_penalty` |
| `src/constraints/layer1.py` | **new** | `RobustTailBound` (train-fit IQR/std fallback + joint 99% threshold), product/order/half-range rules |
| `src/constraints/layer2.py` | **new** | JSON rule-set loader/dumper (dataset-guarded) |
| `src/constraints/engine.py` | **new** | `ConstraintEngine`: `project`, `penalty` (C1/C2), hierarchical `validate` (rate_l0 / rate_l0_l1 / rate_l0_l1_l2) |
| `src/constraints/registry.py` | **new** | serializable type registry |
| `old_constraints/ciciot2023/mined.json` | **new** | Layer-2 rule set (order, half-range, Tot=N·AVG, Var=Std²) |
| `old_constraints/cicids2017_distrinet/mined.json` | **new** | 14 train-mined order/product rules; zero train violations at the retained tolerance |
| `src/vae/cicids2017_stage_a.py` | **new** | four per-attack-class typed β-VAEs; train-only loss weights; val checkpoint selection and IDR calibration |
| `src/classifiers/cicids2017d_victims.py` | **new** | safe loading of the four existing category victims with attack input gradients |
| `src/attack/run_cicids2017_vae_attacks.py` | **new** | CICIDS A1–A6 runner; CFF masks; Stage-B; hierarchical ASR/IDR/True-IDSR |
| `src/attack/residual_head.py` | **new** | `ResidualHead` + `ResidualAttackGenerator` (latent/residual/both), mask+P0+derived |
| `src/attack/train_attack_head.py` | **new** | `VictimGuidedTrainer` + `StageBConfig` (frozen VAE, normalized cost, injected victim) |
| `src/experiments/ablations.py` | **new** | `AblationConfig` + `PRESETS` A0–A6 + `build_ablation` |
| tests under `*/tests/` | **new** | 49 tests across all packages |

Not modified: existing attack runners, `train.py`, `train_all.py` constructors still
work (the model kept a backward-compatible `partition=` path and `decode_to_39`).

---

## 3. Dataset portability (add CICIDS2017)

To support a new flow dataset you write **one adapter file** only:

1. Implement `feature_manifest()` — one `FeatureSpec` per column with a `value_type`
   chosen from extractor semantics (not the name), hard bounds, and `scaling`.
2. Implement `class_mapping()` from the dataset's label encoder.
3. Implement `feature_transform()` — `FeatureTransform.fit(X_train)` or
   `from_sklearn_scaler(scaler, manifest)`.
4. Implement `load_split()`.
5. *Optional*: drop `old_constraints/<dataset>/mined.json` for Layer 2.

You do **not** edit dataset-specific logic into the decoder, constraint engine, attack
head, Stage-B trainer, or validators. `src/datasets/cicids2017.py` now implements this
contract for the 79-column DistriNet artifacts. Its VAE consumes the saved
RobustScaler-space arrays; raw `_pristine` arrays are used only for train-fit
constraints and validation.
 
CICIDS uses the preprocessing pipeline's chronological-within-source-label 70/15/15
split. VAE weights, Layer-1/2 parameters, CFF masks, and Stage-B heads use TRAIN;
validation selects Stage-A checkpoints and calibrates each class's empirical 95th
percentile Mahalanobis gate; test is attack/evaluation only.

---

## 4. Constraint architecture

| Layer | Nature | Examples | Where |
|---|---|---|---|
| **0** | hard, structural / inviolable | non-negativity, [0,1] aggregates, bounded [l,u], immutable copy, exact derived recompute | `constraints/layer0.py` (P0 projector, differentiable) |
| **1** | soft, generic algorithm + train-fit params | robust tail bound, product equality, monotone order, half-range | `constraints/layer1.py` |
| **2** | soft, dataset/extractor-specific or mined | per-dataset `old_constraints/<dataset>/mined.json` | `constraints/layer2.py` + registry |

**Generation vs evaluation are kept separate.** Generation uses `P0` (project) and
soft penalties `C1`/`C2` in the objective. Evaluation uses the engine's independent
`validate()` returning per-sample masks and hierarchical rates
(`rate_l0`, `rate_l0_l1`, `rate_l0_l1_l2`) — it re-checks samples rather than trusting
the generator's own projection. Layer 0 remains structural (not a soft penalty);
`Layer0Projector.soft_penalty` is a separate *pre-projection* regularizer used only if
a config enables it.

Empirical: on real CICIoT val, Layer-0 pass rate > 0.99 (test asserts). CICIoT
Layer-2 exact identities remain soft pending its clean-data exactness verification.
For CICIDS, six identities with zero train violations and 100% validation pass were
promoted to manifest-declared Layer-0 recomputation; Layer 2 independently rechecks
them.

**Selecting active layers per run.** The active layer set is a runtime knob, so any
combination can be toggled without editing code:

```python
from constraints import build_engine          # build only the layers you want
eng = build_engine(manifest, "01",            # "0" | "01" | "012" | "0,2" | {0,1}
                   layer1_fit_x_raw=X_train_raw)          # required iff layer 1
# or build once and flip per run:
eng.set_active_layers("012"); eng.set_active_layers({0, 1})
eng.penalty(x_raw)        # C1 iff 1 active, C2 iff 2 active
eng.project(x_raw)        # Layer-0 projection iff 0 active (else identity)
eng.validate(x_raw)       # rate_l0 / rate_l0_l1 / rate_l0_l1_l2 over active layers

from experiments.ablations import build_ablation
bundle = build_ablation("A6", adapter, active_layers="01",   # overrides the preset
                        layer1_fit_x_raw=X_train_raw)
# active_layers also sets generator.apply_layer0 (=0 in the set) so generation and
# evaluation use the same layers.
```

---

## 5. Training methodology

**Stage A — class manifold (no victim).** Per attack class `c`, train the typed β-VAE
on TRAIN samples of class `c` only. Objective = reconstruction + βKL (+ optional
C0/C1/C2). Free bits + KL warmup + posterior-collapse diagnostics retained. Verified:
`compute_manifold_elbo` reduces reconstruction on real DDoS; free-bits floor holds.

**Stage B — victim-guided residual (base frozen).** Freeze the VAE; train only the
`ResidualHead` with an **injected** victim:
```
L = λ_attack·CE(victim(x_adv), y_benign) + λ_delta·Σ w_i|Δraw_i|/(IQR_i+ε) + λ1·C1 + λ2·C2
x_adv = P0(x + M⊙Δ)   (immutable copied, derived recomputed)
```
Perturbation cost is feature-normalized by the TRAIN robust scale (RobustScaler IQR),
not raw L1/L2. Verified: Stage-B raises the victim's benign-target rate while keeping
Layer-0 validity > 0.99 and finite bounded cost; cost is exactly 0 when nothing is
mutable; swapping the victim changes results (true dependency).

**Train/val/test discipline.** VAE weights, `FeatureTransform`, `RobustTailBound`
params, and Stage-B head all fit on TRAIN only (`build_ablation` raises if Layer 1 is
requested without a train sample). Val is used for calibration/diagnostics; test only
for final evaluation. Encoding/attacking a test sample at eval time is not training on
test.

---

## 6. Backward compatibility

| Item | Status |
|---|---|
| `MixedInputBetaVAE(partition=…)` constructor | **active** (legacy path, A0) |
| `decode_to_39(z, scaler, mode)` | **compatibility alias** over `decode` (all attack callers keep working) |
| `register_protocol_references(scaler)` | **compatibility** shim → sets transform buffers |
| legacy `_structure_raw` (CICIoT structured decoder) | **active** for A0 / `decoder_kind="legacy"` |
| `compute_elbo` (hardcoded-index constraint loss) | **compatibility** (used by legacy `train.py`) |
| `protocol_embed_dim`, `binary_logits`, `PROTOCOL_ALLOWLIST` | **dormant/compat** (empty), unchanged |
| existing checkpoints | none exist (`models/vae` absent) → **fresh retraining** for the typed path; spec permits |

No obsolete concept leaked into the new APIs; legacy is isolated behind the
`partition=`/`decoder_kind="legacy"` path.

---

## 7. Remaining technical debt (intentionally deferred)

- CICIoT exact identities (Var=Std², Tot size≡AVG) remain Layer-2 soft checks pending
  CICIoT-specific clean-data exactness verification. CICIDS promotion is complete.
- `PerturbabilityScorer` is not yet part of the manifest pipeline (`mutable=None`).
  The CICIDS runner consumes the existing class-conditional, train-only CFF masks
  (`top10`/`top25`/`top50`/`eligible`) with an exact feature-order check. Other generic
  callers still default to all-mutable when no explicit mask is provided.
- The dedicated CICIDS runner consumes the manifest/engine path. Older CICIoT runners
  still import `attack.validator` / `PerturbationMask` / `vae.schema.get_partition`;
  migrating the canonical `run_all_models_attack_rerun.py` remains a follow-up.
- Legacy `losses.compute_elbo` constraint terms remain vacuous under the structured
  decoder; superseded by `compute_manifold_elbo` for the typed path.
- Jacobian/JSMA/tangent guidance: not started (as instructed).

---

## 8. Tests performed

Env: `C:/Users/user6/.local/share/mamba/envs/thesis/python.exe`, `PYTHONPATH=src`,
`CUDA_VISIBLE_DEVICES=""`.

```
python -m pytest src/datasets/tests src/vae/tests src/constraints/tests \
    src/attack/tests src/experiments/tests -q -p no:faulthandler
# 49 passed
```
Coverage highlights: manifest order/width/scaler fail-loud; transform train-only +
exact wrap of existing scaler; typed VAE ranges + dynamic dims + fail-loud + trains on
real DDoS; Layer-0 project/validate + immutable + real-data >0.99; Layer-1/2 penalties
+ registry roundtrip + engine hierarchy; manifold ELBO trains + free-bits + diagnostics
no-crash; residual head mask/immutable/derived/differentiable attack step; Stage-B
target-rate↑ with validity held; all A0–A6 build and run.

---

## 9. Recommended next experiment (minimum ablation sequence)

Run A0→A6 on the same victims/test split and report, per step, the hierarchical
metrics with explicit denominators:
```
ASR_raw = evasion / clean_correct
ASR_L0, ASR_L0_L1, ASR_L0_L1_L2   (evasion ∧ pass_l*)
IDR (Mahalanobis, val-fit), True-IDSR = evasion ∧ joint_valid ∧ in_distribution
mean/median normalized perturbation cost, per-constraint violation rate
```
Expected reads:
- **A1 vs A0**: typed decoder should match/improve reconstruction and raw validity.
- **A2/A3/A4**: each constraint layer should raise `ASR_L*`/valid-ASR and per-constraint
  pass rates at some cost to `ASR_raw`; keep a layer only if valid-ASR or plausibility
  improves.
- **A5 vs latent-only**: residual head should lower perturbation cost at equal evasion.
- **A6 vs A5**: Stage-B should raise valid evasion and/or cut cost further, transferring
  across victims.

Do **not** claim superiority of any layer/head until these numbers show it.

---

## 10. The VAE in detail

This section documents the VAE exactly as it exists after the refactor
(`src/vae/model.py`, `src/vae/decoder.py`, `src/vae/losses.py`).

### 10.1 Role and the two decoder paths

The VAE is a per-class β-VAE that learns the manifold of one attack class in Stage A.
`MixedInputBetaVAE` supports two decoder paths chosen at construction:

- **typed** (default when a `FeatureManifest` is passed) — the generic, dataset-agnostic
  head in `vae/decoder.py`. Dimensionality and per-feature activations come from the
  manifest. This is the A1+ path.
- **legacy** (default when only a `partition` dict is passed) — the original CICIoT
  structured continuous decoder, retained unchanged for the A0 ablation and for callers
  not yet migrated to the manifest.

`decoder_kind` overrides the default; `decoder_kind="typed"` requires a manifest.

### 10.2 Input/output contract

- Input `x`: `(N, F)` in **model (scaled)** space, `F = manifest.n_features` (39 for
  CICIoT, 79 for CICIDS2017; `encode` checks the width dynamically).
- The decoder produces values in **raw** feature space (`continuous_mu_raw`) and their
  scaled image (`continuous_mu`); reconstruction is scored in scaled space.
- CICIoT value-type census (from the manifest): 22 `probability` (flag/service/protocol
  window means in [0,1]), 2 `bounded_continuous` (`Time_To_Live`, `Protocol Type`, both
  [0,255]), 15 `positive_continuous` (counts/sizes/rates). None are `binary` or
  `categorical`.

### 10.3 Encoder

```
x (N,F) ─► optional asinh ─► Linear(F,128) ReLU ─► Linear(128,64) ReLU
          ─► Linear(64, 2·latent) ─► mu (N,16), logvar (N,16)
```
`encoder_hidden = (128,64)`, `latent_dim = 16`, `latent_logvar_bounds = (-6, 6)`.
The encoder transform defaults to `none`; CICIDS selects `asinh` because rare
zero-IQR rate columns can otherwise produce unstable latent activations. Decoder
targets and outputs remain in the authoritative RobustScaler space.

### 10.4 Reparameterization

$$z = \mu + \varepsilon \odot e^{0.5\,\log\sigma^2},\quad \varepsilon\sim\mathcal N(0,I)\ \ (\text{train});\qquad z=\mu\ \ (\text{eval, deterministic}).$$

### 10.5 Shared decoder trunk

```
z (N,16) ─► Linear(16,64) ReLU ─► Linear(64,128) ReLU ─► hidden (N,128)
```
`decoder_hidden = (64,128)`. Both decoder paths sit on top of this trunk.

### 10.6 Typed decoder (`TypedDecoder`, A1+)

Two linear heads over the trunk: `pre: Linear(128,F)` (pre-activation `a`, i.e. the
*unstructured* raw prediction) and `logvar_head: Linear(128,F)` (clamped to `(-7,2)`).
Per-feature activation is selected by the manifest `value_type` and applied in **raw
space**:

| value_type | activation | guarantees |
|---|---|---|
| `real` | $x_i = a_i$ | unrestricted |
| `positive_continuous` | $x_i = \mathrm{softplus}(a_i)$ | $x_i \ge 0$ |
| `integer_count` | $x_i = \mathrm{softplus}(a_i)$ | $\ge 0$; rounding is a separate eval-time STE, **not** applied in the decoder |
| `probability` | $x_i = \sigma(a_i)$ | $x_i \in (0,1)$ |
| `binary` | $x_i = \sigma(a_i)$ | $(0,1)$; hard threshold is a downstream STE |
| `bounded_continuous` | $x_i = l_i + (u_i - l_i)\,\sigma(a_i)$ | $x_i \in (l_i,u_i)$ |

`derived` and `categorical` are **rejected** by `TypedDecoder` (raises): exact derived
features are recomputed by Layer 0 from their parents, and categorical logit/Gumbel
decoding is a later phase. This keeps typing honest rather than silently mis-decoding.

Because the typed activations already satisfy the Layer-0 domain, the typed decoder is
itself a structural `P0` for value-type domains — the pre-projection penalty `C0` on its
output is ~0 by construction (verified).

### 10.7 Legacy structured decoder (A0)

`head_continuous_mu`/`head_continuous_logvar: Linear(128,F)`; `_structure_raw` reimposes
CICIoT structure with literal indices (retained only on this legacy path): `sigmoid` on
the 22 bounded indicators, `clamp_min(0)` on the rest, `TTL = 255·σ(a/32)`, softplus
ordering `Min ≤ AVG ≤ Max`, `Number = softplus`, `Std = floor + softplus`,
`Variance = Std²`, and (if the physics decoder is on) `Tot size = AVG`,
`Tot sum = Number·AVG`. This path is what A0 evaluates.

### 10.8 Raw ↔ scaled transform (injected, fail-loud)

The VAE no longer owns preprocessing. Affine scaling is registered from a
`FeatureTransform` (preferred) or an sklearn scaler:

```python
vae.register_feature_transform(transform)      # sets feature_center/feature_scale buffers
vae.register_protocol_references(scaler)        # compat shim, same effect
```
with $\text{raw} = \text{scaled}\cdot s + c$ and $\text{scaled} = (\text{raw}-c)/s$
(`continuous_scaled_to_raw` / `continuous_raw_to_scaled`). If neither is registered,
`decode_internal` raises **`TransformNotRegisteredError`** — there is no silent
`center=0 / scale=1` fallback. Buffer lengths must equal `n_features` (validated).

### 10.9 Decode pipeline and outputs

`decode_internal(z)` (typed path):
```
hidden = trunk(z)
a, x_typed, logvar = TypedDecoder(hidden)        # a = pre-activation, x_typed = raw
scaled = raw_to_scaled(x_typed)
```
Returned dict:

| key | space | meaning |
|---|---|---|
| `continuous_mu` | scaled | reconstruction / generation output (loss + victim input) |
| `continuous_mu_raw` | raw | typed raw output (fed to constraints/validators) |
| `continuous_mu_raw_unstructured` | raw-ish | pre-activation `a` (pre-projection) |
| `continuous_logvar` | scaled | heteroscedastic recon log-variance |
| `binary_logits`, `protocol_logits` | — | empty tensors (legacy compat) |

`decode(z)` returns this dict; `decode_to_39(z, scaler=None, mode="soft")` is the
backward-compatible alias for existing attack/diagnostic callers — `scaler` is ignored
(registered transform is authoritative) and `soft`/`hard` are identical (aggregates stay
continuous). `forward(x)` = encode → reparameterize → `decode_internal`, returning
`{mu, logvar, **decoded}`.

### 10.10 Objective — manifold ELBO (`compute_manifold_elbo`)

$$\mathcal L = \underbrace{\text{recon}}_{\text{NLL}} + \beta\,\mathrm{KL} + \lambda_{\text{pre}}\,C_0 + \lambda_1 C_1 + \lambda_2 C_2$$

- **Reconstruction** on `continuous_mu` vs the scaled target, per feature, then summed
  and mean-batched:
  - Gaussian: $\tfrac12\big(\log\sigma^2 + (t-\hat t)^2 e^{-\log\sigma^2} + \log 2\pi\big)$
  - Laplace: $\log 2 + \log\sigma^2 + |t-\hat t|\,e^{-\log\sigma^2}$
  ($\log\sigma^2$ = `continuous_logvar`, clamped to `(-7,2)`; optional per-feature weights.)
- **KL with free bits**: $\mathrm{KL} = \sum_j \max\!\big(\text{kl}_j,\ \lambda_{fb}\big)$,
  $\text{kl}_j = -\tfrac12(1+\log\sigma_j^2-\mu_j^2-e^{\log\sigma_j^2})$, giving each latent
  dim a $\lambda_{fb}$-nat floor (anti-collapse). `per_dim_kl_mean` is also returned for the
  collapse diagnostic.
- **$C_0$** = `engine.layer0.soft_penalty(raw)` — pre-projection domain regularizer
  (default weight 0; ~0 for the typed decoder anyway).
- **$C_1$, $C_2$** = `engine.penalty(raw)` — Layer-1 / Layer-2 soft penalties, and they
  respect the engine's active-layer set (so ELBO constraint terms follow the same on/off
  toggle as attacks/eval).

The legacy `compute_elbo` (hard-coded-index constraint loss) is retained for the A0
`train.py` path but is superseded by `compute_manifold_elbo` for the typed VAE.

### 10.11 β schedule

`BetaScheduler` linearly warms β from 0 to `beta_target` over `warmup_steps`
(`beta_warmup_epochs · steps_per_epoch`, default 10 epochs), then holds. With CICIoT
defaults `beta_target = 0.5`, `free_bits_lambda = 0.1` (the anti-collapse pair).

### 10.12 Key hyperparameters

`latent_dim=16`, `encoder_hidden=(128,64)`, `decoder_hidden=(64,128)`,
`beta_target=0.5`, `free_bits_lambda=0.1`, `continuous_likelihood∈{gaussian,laplace}`,
`latent_logvar_bounds=(-6,6)`, `continuous_logvar_bounds=(-7,2)`, AdamW `lr=1e-3`,
`weight_decay=1e-5`, `grad_clip=5.0`, early stop on val recon/loss.

### 10.13 Diagnostics (post-fix)

`vae/diagnostics.py` runs posterior-collapse (per-dim KL, 0.01 threshold), per-feature
reconstruction error, and unconditional/conditional validity. The audit crashes are
fixed: the legacy protocol/binary sections are guarded on empty heads
(`protocol_top1_accuracy` → `None`) and the `None` `protocol_idx_batch` yields vacuous
`protocol_binary_consistency = 1.0`. Empirically confirmed: `argmax` on the empty
protocol head raised `IndexError`; the guarded code no longer does.

### 10.14 Determinism and leakage discipline

Seeds fixed (`torch`/`numpy` = 42; `cudnn.deterministic`). The VAE weights, the
`FeatureTransform`, reconstruction feature weights, and Layer-1/2 statistics are fit
on **TRAIN only**; the decoder refuses to run without a registered transform. Val is
used for early stopping and to fit/calibrate the Mahalanobis IDR gate (empirical 95th
percentile per attack class); test only for final attack evaluation. Encoding or
attacking a test sample at evaluation time is not training on test.

### 10.15 How the VAE connects downstream

`continuous_mu_raw` feeds the `ConstraintEngine` (Layer 0 projection + Layer 1/2 soft
penalties and independent validators). `decode`/`decode_to_39` is the differentiable
generator used by latent PGD/C&W and by the `ResidualAttackGenerator`
($x_{cand}=x+M\odot\Delta \to P_0 \to$ derived recompute). In Stage B the VAE is frozen
and only the residual head trains against an injected victim. Because the VAE consumes
**only** the manifest + transform, the same design ports to any dataset by writing an
adapter (see §3 and `docs/dataset_portability.md`).
