# 9. VAE Adversarial Method (MODERATE DETAIL)

A per-class β-VAE generator plus a **latent-space attack** that steers the latent code,
decodes to a traffic-modification proposal, collapses that proposal into the three
realizable primitive controls (p, delay, shape), and re-uses PrimAttack's φ. Status: **generators are
complete and current; the final aggregated attack run against the current victim roster
is missing.**

### Source files
`src/vae/model.py` (`MixedInputBetaVAE`), `src/vae/cicids2017_stage_a.py` (per-class
training + IDR fit), `src/attack/vae_latent_primitive.py` (`LatentPrimitiveAttack`),
`vae_latent_variants.py` (`LatentRawAttack`, `LatentMaskedAttack`), runners
`run_cicids2017_vae_latent_attack.py`, `run_cicids2017_latent_variants.py`,
`run_cicids2017_vae_attacks.py`.

## 9.1 VAE architecture & training
- `MixedInputBetaVAE` (`model.py:48-277`): typed decoder; **latent_dim = 16**; encoder
  hidden (128, 64), decoder (64, 128); `asinh` input; for CICIDS2017 all 79 features are
  treated as continuous (the binary/protocol decoder heads are inert here).
- **Per-class β-VAEs** (`cicids2017_stage_a.py`) for DoS, DDoS, Recon, BruteForce:
  `β=0.5` + **free-bits 0.1** (anti-collapse), Laplace likelihood, AdamW `lr=1e-3`, 30
  epochs, patience 5, train-fit RMS loss weights. Fitted **on train only**.
- **IDR gate** (in-distribution rate): per-class **Mahalanobis** distance in a
  generator-relative space, threshold = **val-anchored p95** (val, not train — call this
  out). Saved as `idr_*.npz`.

## 9.2 Latent attack (`LatentPrimitiveAttack`, `vae_latent_primitive.py:90-313`)
- Optimize **only `z_adv`** (the latent code), not features.
- Decoder proposes a *direction*; `CICIDS2017PrimitiveModel.infer_primitives_from_decoded`
  reads that direction through the reconstructable forward-length / forward-timing
  signals and collapses it into `p ≥ 0` (relu of length movement), `delay ≥ 0` (relu of
  mean Fwd IAT Total / Flow Duration movement), and `shape ∈ [0,1]` (one minus the
  proportional share implied by the Fwd IAT Std movement), each clamped to the per-flow
  feasible caps. Movement is measured relative to `decode(z₀)` so the VAE
  reconstruction bias cancels.
- **Loss**: C&W-style targeted→Benign + latent-, cost-, and realism-regularizers;
  **latent L2-ball projection** (`eps_z`). Classifier feedback via the victim logits, same
  as PrimAttack.
- **Masks/constraints**: same primitive contract, capabilities, and φ as PrimAttack;
  frozen features copied; derived features recomputed. Success re-evaluated **after**
  discrete primitive projection.
- Runner CLI (`run_cicids2017_vae_latent_attack.py`): `steps=120, lr=0.08, eps_z=10,
  seeds=42,43,44, victims=mlp,cnn,ft_transformer`.

## 9.3 Variants
- `LatentRawAttack` (`vae_latent_variants.py`): perturbs all 79 decoded features, Layer-0
  clamp only — **diagnostic** (not validity-preserving).
- `LatentMaskedAttack`: perturbs perturbable features only, copies frozen, recomputes
  derived.
- Older CFF residual ladder (A1–A6) in `run_cicids2017_vae_attacks.py`.

## 9.4 Experiment status (be precise)
- **LIVE**: `outputs/cicids2017_vae_stage_a/` — 4 `vae_*.pt` (~210 KB each), 4
  `idr_*.npz` (6138 B), `stage_a_summary.json`. Generators complete/current.
- **SMOKE only**: two npz — `cicids2017_vae_attacks_masked/DoS_mlp_A1.npz`,
  `cicids2017_vae_attacks_pave_run/DoS_mlp_A4.npz`. **No live `attack_results.json` for
  any VAE attack.**
- **ARCHIVED / stale**: all aggregated latent-attack results are under `old_root_files/`
  and were built against the **retired** LSTM/serial victims (48-cell primitive-latent,
  16-cell masked/raw). Those showed near-zero evasion of MLP/CNN but high evasion of the
  (now-retired) LSTM/serial. **No VAE attack has been run against the current
  ft_transformer roster.**

## 9.5 Claims
- **Can claim**: four train-only per-class β-VAE generators with a val-anchored
  Mahalanobis IDR gate; a fully-implemented latent→primitive attack that inherits
  PrimAttack's validity/feasibility guarantees.
- **Must NOT claim**: any completed VAE-attack result on the current victims — the final
  aggregated run is **missing**; archived numbers used retired victims and must not be
  presented as current. Distinguish generator training (done) from attack evaluation
  (not done) in the thesis.
