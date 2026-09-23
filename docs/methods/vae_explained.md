# Per-Class β-VAE (`src/vae/`): Implementation Knowledgebase

## 1. Purpose and scope

`src/vae/` implements the generative half of this thesis: **eight per-class
β-VAEs** (one per coarse category) that model the distribution of CICIoT2023
network-flow feature vectors. Their role in the adversarial pipeline is to
provide a **smooth, class-conditional latent manifold** the attack code steers
(latent PGD / C&W) to synthesize adversarial flows that stay close to the
real data manifold, plus a **decoder** that maps latent points back to the
39-feature raw space. Validity/realism of what the decoder emits is then judged
by the domain validator (G1–G8), the post-hoc physics validator, and the
in-distribution (Mahalanobis) gate.

The current model is `MixedInputBetaVAE` (the "Path C" design named in
`CLAUDE.md`). Despite the name, **every one of the 39 CSV features is
reconstructed continuously** — there is no Bernoulli head, no protocol
embedding/cross-entropy, no integer rounding, and no derived one-hot. The
"mixed input" name and several constructor arguments (`protocol_embed_dim`,
`n_pseudo_binary`, binary/protocol loss weights) are **backward-compatibility
vestiges** retained so old checkpoints/configs still load; they are inert.

This document describes the source as it exists now: model architecture, the
ELBO, config, dataset, training loop, orchestration + gates, diagnostics, the
physics validator, and how the attack code consumes the VAE. It also flags the
dormant/vestigial code paths so they are not mistaken for active behavior.

## 2. Source of truth and module boundaries

| Concern | Source |
|---|---|
| Feature names, order, roles, bounded-indicator indices | `src/preprocessing/schema.py` |
| VAE feature partition (`continuous_idx`, `bounded_continuous_idx`, …) | `src/vae/schema.py:get_partition` |
| Model architecture (`MixedInputBetaVAE`) | `src/vae/model.py` |
| ELBO / losses / β schedule | `src/vae/losses.py` |
| Default hyperparameters (all 8 classes) | `src/vae/config.py` |
| Per-class dataset | `src/vae/dataset.py` |
| Single-class training loop | `src/vae/train.py` |
| 8-model orchestrator + Phase-10 gates | `src/vae/train_all.py` |
| Post-training diagnostics | `src/vae/diagnostics.py` |
| Post-hoc physics plausibility checks | `src/vae/physics_validator.py` |
| Physics-check calibration on clean train data | `src/vae/calibrate_physics_validator.py` |
| Processed arrays / scaler | `data/processed/X_*.npy`, `y_*.npy`, `scaler.pkl` |
| Checkpoints | `models/vae/vae_class_{id}_{name}.pt` |
| Curves / diagnostics JSON | `results/vae/` |

**Feature order is a hard contract.** All 39 columns follow the frozen order in
`preprocessing/schema.py:FEATURE_NAMES`. Every saved array, scaler, and
checkpoint is meaningless if the order changes (`CLAUDE.md` locked decision).

### 2.1 The 39 features (frozen index → name)

```
 0 Header_Length      10 cwr_flag_number   20 SSH        30 Tot sum
 1 Protocol Type      11 ack_count         21 IRC        31 Min
 2 Time_To_Live       12 syn_count         22 TCP        32 Max
 3 Rate               13 fin_count         23 UDP        33 AVG
 4 fin_flag_number    14 rst_count         24 DHCP       34 Std
 5 syn_flag_number    15 HTTP              25 ARP        35 Tot size
 6 rst_flag_number    16 HTTPS             26 ICMP       36 IAT
 7 psh_flag_number    17 DNS               27 IGMP       37 Number
 8 ack_flag_number    18 Telnet           28 IPv        38 Variance
 9 ece_flag_number    19 SMTP             29 LLC
```

**Bounded aggregated indicators (22 columns)** = `BOUNDED_AGGREGATED_IDX`:
the 7 TCP-flag averages (idx 4–10) + 15 service/protocol averages (idx 15–29).
These are window means of per-packet 0/1 indicators, so their raw values live in
[0,1]; the decoder sigmoid-bounds exactly these columns. The other 17 columns
are non-negative continuous aggregates/statistics.

Semantics called out in code: **`Tot size` (35) is an exact duplicate of `AVG`
(33)** in this processed schema (physics rule P3 is therefore omitted), and
`Protocol Type` (1) is treated as a **continuous averaged code**, not a
categorical — `raw_to_protocol_index`/`protocol_index_to_raw` raise on purpose.

### 2.2 The partition dict (`vae/schema.py:get_partition`)

```python
{
  "continuous_idx":            [0..38],          # ALL 39 features
  "bounded_continuous_idx":    BOUNDED_AGGREGATED_IDX,   # 22 sigmoid columns
  "unbounded_continuous_idx":  the other 17,
  "protocol_idx":              [1],
  "independent_binary_idx":    [],   # deliberately empty (compat)
  "derived_binary_idx":        [],   # deliberately empty (compat)
  "pseudo_binary_idx":         [],   # deliberately empty (compat)
}
```

The empty compat keys are why the model reports `n_independent_binary = 0` and
emits empty `binary_logits`/`protocol_logits`.

## 3. Model: `MixedInputBetaVAE` (`model.py`)

A plain fully-connected Gaussian VAE with a **structured continuous decoder**
that reimposes domain structure in raw feature space.

### 3.1 Topology

```
x (N,39, scaled)
   └─ encoder_body: Linear(39→128) ReLU  Linear(128→64) ReLU      (encoder_hidden=[128,64])
        └─ encoder_out: Linear(64 → 2*latent_dim) → split → mu, logvar
             logvar clamped to latent_logvar_bounds (default [-6, 6])
   reparameterize: train → mu + eps*exp(0.5*logvar);  eval → mu (deterministic)
z (N,16)
   └─ decoder_body: Linear(16→64) ReLU  Linear(64→128) ReLU        (decoder_hidden=[64,128])
        ├─ head_continuous_mu:     Linear(128→39)   → raw_head (in SCALED space)
        └─ head_continuous_logvar: Linear(128→39)   → clamp(-7, 2)
```

`latent_dim = 16` for every class (`config.py`). Encoder input width and both
heads are hard-wired to 39; the constructor **raises** if `continuous_idx` is
not length 39.

### 3.2 Scaler buffers and raw↔scaled conversion

The model carries the fitted `RobustScaler` as two non-persistent-value buffers,
registered via the (historically named) `register_protocol_references(scaler)`:

- `feature_center` ← `scaler.center_` (39,)
- `feature_scale`  ← `scaler.scale_` (39,)

Helpers: `continuous_scaled_to_raw(v) = v*scale + center` and
`continuous_raw_to_scaled(v) = (v - center)/scale`. These let the decoder move
into raw space, apply physical structure, and move back — so the returned
`continuous_mu` is in **scaled space** (matching the training targets) while
`continuous_mu_raw` is the physically-structured raw view.

### 3.3 The structured decoder (`_structure_raw`) — the core of Path C

`decode_internal` does: `raw_head` (scaled) → `continuous_scaled_to_raw` →
`_structure_raw` → `continuous_raw_to_scaled`. `_structure_raw` reimposes
domain structure in raw space (all indices per §2.1):

- **Bounded indicators (22 cols):** `sigmoid(raw)` → forced into [0,1]. This is
  applied unconditionally (even without the structured continuous decoder).

When `use_structured_continuous_decoder = True` (the default), additionally:

- **Unbounded cols:** `clamp_min(0.0)` (non-negativity).
- **TTL (2):** `sigmoid(raw/32) * 255` → forced into [0,255].
- **Min/AVG/Max ordering (31,33,32):** built with softplus deltas so the
  invariant **Min ≤ AVG ≤ Max** holds by construction:
  `Min = softplus(r31)`, `AVG = Min + softplus(r33−r31)`,
  `Max = AVG + softplus(r32−r33)`.
- **Number (37):** `softplus` → non-negative.
- **Std (34):** `structured_std_floor + softplus(r34)`;
  **Variance (38) = Std²** — enforced exactly, not learned independently.

When `use_structured_physics_decoder = True` (default **False**, dormant):

- `Tot size (35) = AVG`, and `Tot sum (30) = Number * AVG`.

Because averaged indicators stay continuous, **`decode_to_39` "soft" and "hard"
modes are identical** — there is no thresholding. `decode_to_39(z, scaler,
mode)` returns `(continuous_mu_scaled, metadata)`; the attack code uses this as
the differentiable generator.

### 3.4 `decode_internal` output dict

`continuous_mu` (scaled), `continuous_mu_raw` (structured raw),
`continuous_mu_raw_unstructured` (pre-structure raw), `continuous_logvar`
(clamped [-7,2]), and empty `binary_logits`/`protocol_logits` +
`pseudo_binary_sigmoid=None` (compat). `forward(x)` returns
`{mu, logvar, **decode_internal(reparameterize(mu,logvar))}`.

## 4. Loss / ELBO (`losses.py`)

`compute_elbo(...)` returns a dict whose `loss` is:

```
loss = recon
     + beta * kl
     + constraint_loss_weight   * constraint_total
     + physics_constraint_loss_weight * physics_total     # weight 0.0 → inactive
     + raw_relative_continuous_loss_weight * raw_relative  # 0.0 by default
     + raw_relative_tail_focus_weight * raw_relative_tail  # 0.0 by default
```

### 4.1 Reconstruction term

Computed against `target = batch[:, continuous_idx]` (all 39, scaled) using
`continuous_mu` and the clamped `continuous_logvar`:

- **`gaussian`** (default): per-feature `0.5*(logvar + (t-r)²/exp(logvar) + log2π)`
  — heteroscedastic Gaussian NLL.
- **`laplace`**: `log2 + logvar + |t-r|/exp(logvar)` — L1 NLL, heavier-tail
  robust (selectable via `continuous_likelihood`).

Optionally scaled per feature by `continuous_feature_weights`, summed over
features, optionally capped per sample (`continuous_nll_per_sample_cap`), then
mean over the batch.

### 4.2 KL term + free bits (anti-collapse)

`per_dim_kl = -0.5*(1 + logvar - mu² - exp(logvar))`. With
`free_bits_lambda = 0.1`, each dim is `clamp(min=λ)` **before** summing, giving
every latent dim a 0.1-nat "free" allowance so the KL term stops squeezing it to
zero. `per_dim_kl_mean` (unclamped) is also returned for the collapse
diagnostic. This pairs with **β = 0.5** (see §5) to prevent posterior collapse.

### 4.3 Constraint loss (`constraint_loss_weight = 0.1`, active)

Soft penalties on the **structured raw** output, all `mean()`ed and summed:

| Term | Expression | Meaning |
|---|---|---|
| `nonneg` | `relu(-raw)` (all cols) | features ≥ 0 |
| `ttl` | `relu(r2 − 255)` | TTL ≤ 255 |
| `ordering` | `relu(r31−r32)+relu(r31−r33)+relu(r33−r32)` | Min ≤ AVG ≤ Max |
| `variance` | `|r38 − r34²| / (r34²+1)` | Variance ≈ Std² |
| `packet_positive` | `relu(-r37)` | Number ≥ 0 |

(Several of these are already guaranteed by the structured decoder; the loss is
belt-and-suspenders and also disciplines the pre-structure head.)

### 4.4 Physics constraint loss (`physics_constraint_loss_weight = 0.0`, DORMANT)

`_compute_physics_constraint_loss`: P2 `|Tot sum − Number*AVG|/(|Tot sum|+1)`
and P4 `relu(Std − 0.5*(Max−Min))`. Weight is 0.0 by default, so it never
contributes — present but inactive (a `CLAUDE.md` code smell).

### 4.5 Raw-relative terms (optional, default off)

If a raw target and positive weight are supplied, adds relative L1 error in raw
space `|raw − target_raw|/(|target_raw|+ε)`, with an optional tail-focus term
(mean over the worst quantile of per-sample rel-error). Both weights default to
0.0.

### 4.6 β schedule (`BetaScheduler`)

Linear warmup from 0 to `beta_target` over `warmup_steps`. `train.py` derives
`warmup_steps = beta_warmup_epochs * steps_per_epoch` (default 10 epochs), which
overrides the older `warmup_frac * total_steps` schedule so β actually reaches
target before early stopping fires.

## 5. Config (`config.py`)

Keyed dict `DEFAULT_CONFIG` shared by all 8 classes. 8-class label IDs are
sklearn-alphabetical: `Benign=0, BruteForce=1, DDoS=2, DoS=3, Mirai=4, Recon=5,
Spoofing=6, Web=7`.

Salient values:

| Key | Value | Note |
|---|---|---|
| `latent_dim` | 16 (all classes) | |
| `beta_target` | 0.5 (all classes) | anti-collapse (with free bits) |
| `free_bits_lambda` | 0.1 | 0.1-nat allowance per dim |
| `encoder_hidden` / `decoder_hidden` | [128,64] / [64,128] | |
| `max_epochs` | 200 | with early stopping |
| `batch_size` | 512 | val loader uses 2× |
| `lr` / `weight_decay` | 1e-3 / 1e-5 | AdamW + CosineAnnealingLR |
| `warmup_frac` / `beta_warmup_epochs` | 0.3 / 10 | β warmup in epochs wins |
| `continuous_likelihood` | `gaussian` | or `laplace` |
| `early_stop_patience` / `grad_clip` | 10 / 5.0 | |
| `constraint_loss_weight` | 0.1 | active |
| `physics_constraint_loss_weight` | 0.0 | dormant |
| `use_structured_continuous_decoder` | True | Path-C structure on |
| `use_structured_physics_decoder` | False | dormant |
| `structured_std_floor` | 0.01 | |
| `latent_logvar_floor/ceiling` | -6 / 6 | encoder logvar clamp |
| `continuous_logvar_floor/ceiling` | -7 / 2 | decoder logvar clamp |
| `protocol_embed_dim`, `protocol_loss_weight`, `binary_feature_loss_weights` | 0 / 0.0 / {} | vestigial |

**Anti-collapse rationale (from the config comment):** β=0.5 lowers KL pressure
and free-bits gives each dim a 0.1-nat floor, so information spreads across all
16 dims instead of packing into a few. The collapse *diagnostic* still uses the
real measured per-dim KL with a 0.01 threshold (so a dim at KL≈0.057 counts as
non-collapsed even though it sits below the 0.1 allowance).

## 6. Dataset (`dataset.py`)

`PerClassDataset(X_split, y_split, class_id, scaler, partition)`:

- Masks rows where `y_split == class_id`; **raises** if none.
- Validates the slice is finite, 2-D, width 39; stores `x_scaled` as float32.
- `target_independent_binary` = empty (N,0), `target_protocol_index` = zeros —
  compat tensors carried through `__getitem__` (`x_scaled`, `target_ind_binary`,
  `target_proto_idx`) but never used to build BCE/CE targets.

## 7. Single-class training (`train.py:train_one_vae`)

Steps:

1. **Determinism:** `torch.manual_seed(42)`, `np.random.seed(42)`,
   `cudnn.deterministic=True`.
2. **Load data:** from `data/processed/{X,y}_{train,val}.npy` + `scaler.pkl`, or
   from an in-memory `shared_arrays` dict (used by the orchestrator to load once
   and reuse across all 8 classes).
3. **8-class labels:** `_load_8class_labels` maps the stored 34-class labels to
   the coarse 8-class category, or reuses `shared_arrays["y_*_8"]`.
4. **Datasets/loaders:** per-class train/val datasets. Optional
   `WeightedRandomSampler` when `train_sample_weight_rules` is set; otherwise
   `shuffle=True`. Val loader is `batch_size*2`, no shuffle.
5. **Feature weights:** builds continuous loss-weight tensors (historical
   `binary_feature_loss_weights` are merged into the continuous map — never
   re-materialized as BCE), optionally normalized.
6. **Model:** resolves per-class `latent_dim`/`beta_target` (dict or scalar),
   constructs `MixedInputBetaVAE`, calls `register_protocol_references(scaler)`.
7. **Optim/sched:** AdamW + `CosineAnnealingLR(T_max=max_epochs)` +
   `BetaScheduler`.
8. **Loop:** per epoch, train then val. Per batch it computes the raw target via
   `continuous_scaled_to_raw(...).detach()`, steps β, calls `compute_elbo`,
   **skips non-finite losses**, backprops, clips grads to 5.0, steps optimizer.
   `beta` counts only on train; `recon_metric` computed on val.
9. **Reconstruction metric** (`_compute_reconstruction_metric`): deterministic
   (eval mode → mu, no sampling) error used for early stopping; supports a
   quantile summary.
10. **Early stopping / checkpoint:** monitors `val_loss` (or `recon_metric` if
    `early_stop_metric="recon_metric"`). On improvement saves the checkpoint;
    stops after `patience` (10) non-improving epochs.
11. **Curves:** 6-panel PNG (total loss, continuous recon, binary+protocol recon
    [zero], KL, β, recon metric) to `results/vae/curves_{name}.png`.
12. **Returns** a metrics dict (best losses, epochs, KL, histories, checkpoint
    path).

### 7.1 Checkpoint contents (`models/vae/vae_class_{id}_{name}.pt`)

`state_dict`, full `config`, `partition`, `pseudo_binary_columns=[]`,
`protocol_allowlist` (historical `[0,1,2,6,17,47]`), `val_history`,
`best_val_loss`, `best_monitor_value`, `best_epoch`, `early_stop_metric`,
`epoch`, `class_id`, `class_name`.

## 8. Orchestration (`train_all.py`)

Trains all 8 VAEs, runs diagnostics, maintains a JSON manifest, and evaluates
Phase-10 gates.

- `_config_for_class` deep-copies `DEFAULT_CONFIG` and applies per-class
  overrides + checkpoint/curve name templates.
- `_load_manifest`/`_save_manifest` with `_deep_update` track checkpoint paths +
  SHA-256 (`compute_sha256`) and diagnostics paths under
  `schema_version`, `checkpoints`, `diagnostics`.
- A summary table (`_SUMMARY_COLS`) is written as markdown + CSV.

### 8.1 Phase-10 gates (`_check_gates`)

| Gate | Condition |
|---|---|
| Gate1 | all 8 checkpoints exist |
| Gate2 | no VAE has > 50% collapsed dims |
| Gate3 | raw **conditional** validity ≥ 60% every class |
| Gate4 | raw **unconditional** validity ≥ 30% every class |
| Gate4b | **postprocessed** unconditional validity ≥ 30% every class |
| Gate5 | protocol accuracy ≥ 95% every class |
| Gate6 | no NaN/Inf in per-feature recon errors |
| Gate7 | `protocol_binary_consistency == 1.0` every class |
| Gate8 | `summary.md` exists |

(Gate5/Gate7 are legacy protocol/binary gates; under continuous protocol
semantics they read fields that diagnostics still populate for compatibility.)

## 9. Diagnostics (`diagnostics.py:run_diagnostics`)

Per class, writes `results/vae/diagnostics_{name}.json` with four blocks. It
lazily imports `attack.validator.validate_batch` (`_load_validator`) as the
domain (G1–G8) checker; validity numbers are gated by that validator.

1. **Posterior collapse** (`_diag_posterior_collapse`): accumulates per-dim KL
   over val; dims with mean KL < **0.01** are "collapsed". Returns
   `per_dim_kl`, `collapsed_dim_count`, indices.
2. **Per-feature reconstruction** (`_diag_per_feature_recon`): per-feature recon
   error on val (likelihood-aware).
3. **Unconditional validity** (`_diag_unconditional_validity`): sample
   `z ~ N(0,I)` (n=1000, seed `1000+class_id`), `decode_to_39(mode="hard")`,
   inverse-transform to raw, `raw_postprocess`, and validate **pre and post**
   postprocess (Gate4 vs Gate4b).
4. **Conditional validity** (`_diag_conditional_validity`): encode→reparam→
   decode real val samples (seed `2000+class_id`), validate raw output (Gate3).

`raw_postprocess` (`vae/schema.py`) applies **only structural continuous bounds
and never rounds** aggregate values.

## 10. Post-hoc physics validator (`physics_validator.py`)

Secondary, raw-space cross-feature plausibility — **separate from** the G1–G8
domain validator. `validate_batch(x_raw)` runs only the implemented rules and
returns per-rule + per-sample (`all_rules_pass_rate`) results.

**Active rules:**

- **P2** `check_p2_total_bytes`: `|Tot sum − Number*AVG| / |Tot sum| < rtol`.
- **P4** `check_p4_std_magnitude`: `Std ≤ ((Max−Min)/2)*(1+rtol)`.
- **P5** `check_p5_single_packet`: for singleton flows (`Number≈1`) require
  near-zero Std/Variance and Min≈Max; non-singletons pass vacuously.

**Omitted (return all-True), with recorded reasons:** P1 (rate/IAT), P3 (Tot
size ≡ AVG duplicate), P6 (header vs `PROTOCOL_MIN_HEADER`), P7 (flag
indicator↔count), P8 (non-TCP flags) — prototyped/calibrated but dropped because
clean-data pass rates were too low to serve as fidelity metrics.

`calibrate_physics_validator.py` samples ≤5000 clean train rows per category
(seed 42), inverse-transforms, runs the validator, and writes per-rule clean
pass rates to `physics_calibration.json` — the empirical baseline used to decide
which rules are trustworthy.

## 11. How the attack code consumes the VAE

Per `CLAUDE.md` (attack module owns these): the canonical runner is
`attack/run_all_models_attack_rerun.py`. Latent attacks (PGD / C&W) perturb a
class VAE's **latent z**, decode via `decode_to_39` (differentiable, soft≡hard),
reimpose mask + protocol each step, and push the decoded flow toward
misclassification by a target IDS. There are also unconstrained input PGD/C&W
and VAE-constrained input variants. Success is scored as:

- `ASR_raw = evasion / clean_correct`
- `ASR_valid` gated on protocol ∧ mask ∧ raw G1–G8
- `IDR` = Mahalanobis in-distribution rate (fit on **VAL**, not train)
- **True-IDSR** = mean(evasion ∧ joint_valid ∧ in_distribution)

Downstream code should import tiers/rules from the planned FeatureManifest, but
today still imports `attack.validator`, `attack.latent_infra.PerturbationMask`,
and `vae.schema.get_partition` directly (a `CLAUDE.md` refactor target).

## 12. Vestigial / dormant / dead code — do not mistake for active

- **Vestigial (inert but load-bearing for compat):** `protocol_embed_dim`,
  `n_pseudo_binary`, `binary_logits`/`protocol_logits` (empty),
  `PROTOCOL_ALLOWLIST` (checkpoint field only), `PerClassDataset` binary/proto
  target tensors, `binary_feature_loss_weights` (merged into continuous),
  `raw_to_protocol_index`/`protocol_index_to_raw` (raise by design).
- **Dormant (present, weight 0 / flag off):** physics constraint loss
  (`physics_constraint_loss_weight=0.0`), structured physics decoder
  (`use_structured_physics_decoder=False`), raw-relative loss terms (weights
  0.0).
- **Dead / scratch / superseded (per `CLAUDE.md`):** `vae/_rediag.py` (broken
  `D:/thesis_final` path, no `__main__`); `retrain_phase1_fix{,_v3,_v4,_v5}.py`
  are superseded by `_v6`. `analyze_collapsed_perturbable.py`,
  `inspect_benign_reconstruction.py`, `reconstruction_accuracy.py`,
  `latent_geometry.py`, `experiment_runner.py` are auxiliary analysis
  one-offs, not part of the train→diagnose→gate path.

## 13. Reproduce / operate

- **Train all 8:** `python -m src.vae.train_all` (see its `argparse` for
  overrides); loads `data/processed/*`, writes `models/vae/*.pt`,
  `results/vae/*`, updates the manifest, evaluates gates.
- **Diagnostics only:** invoked automatically by the orchestrator; core entry is
  `vae.diagnostics.run_diagnostics`.
- **Calibrate physics rules:** `python -m src.vae.calibrate_physics_validator`
  → `physics_calibration.json`.
- **Determinism:** global `SEED=42`; per-class diagnostic seeds are offset
  (`1000+id`, `2000+id`) so unconditional/conditional draws differ per class.

## 14. Correctness-sensitive invariants

- Encoder input and both decoder heads are fixed at 39; partition must be
  length-39 or the constructor raises.
- The scaler must be registered (`register_protocol_references`) before decoding,
  or raw↔scaled conversion uses the identity buffers (center 0, scale 1) and
  silently corrupts raw-space structure/validity.
- `continuous_mu` is in **scaled** space (training target space);
  `continuous_mu_raw` is the structured raw view. Consumers must not confuse
  them.
- Min ≤ AVG ≤ Max and Variance = Std² hold by construction only when
  `use_structured_continuous_decoder=True` (the default). The constraint loss
  softly enforces them regardless.
- Everything the VAE fits (scaler, targets) is **train-only**; val/test are
  transformed, never fitted (leakage rule, `CLAUDE.md`).
