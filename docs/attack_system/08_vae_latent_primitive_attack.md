# 08 — The VAE Latent-Space Primitive-Constrained Attack (proposed method)

> **Superseded control contract.** This page still describes the former `(p, alpha)`
> collapse. The active primitive layer uses `(p, delay, shape)`. See
> [`../full_thesis_methodology/09_vae_adversarial_method.md`](../full_thesis_methodology/09_vae_adversarial_method.md)
> for current VAE status and
> [`../full_thesis_methodology/02_primattack.md`](../full_thesis_methodology/02_primattack.md)
> for the primitive contract.

**Sources:** `src/attack/vae_latent_primitive.py` (algorithm),
`src/attack/run_cicids2017_vae_latent_attack.py` (runner).

This is the thesis's **proposed** attack. Instead of optimizing `(p, alpha)` directly, it
optimizes the **latent code `z_adv`** of the per-class β-VAE. The decoder proposes a *direction*
of traffic modification; that direction is collapsed into the two realizable primitives and
pushed through the **same realizability layer** the direct baseline uses. The point is to test
whether searching in a learned traffic manifold produces more *realistic* adversarials than
searching in primitive space directly.

---

## 1. The computational graph (classifier-gradient path)

```
z_adv  (the ONLY optimizer variable)
  -> attack VAE decoder                         decode(z_adv)["continuous_mu_raw"]
  -> infer_primitives_from_decoded              -> continuous (p, alpha)  [differentiable]
  -> primitive_model.generate(quantize=False)   -> x_adv_raw
  -> affine scaling  (x_adv_raw - center)/scale
  -> victim classifier
  -> targeted-Benign loss
```

The decoder output is **never classified directly** — it only *proposes movement*, which is
converted to true controls, and the realizability layer regenerates the final vector. Final
success is re-evaluated **after discrete primitive projection** (`project_controls` +
`generate(quantize=True)`), exactly like the direct attack.

Movement is measured relative to `decode(z0)` (the reconstruction of the source), not `raw0`,
so the VAE's constant reconstruction bias cancels and the primitive is driven purely by the
latent displacement `z_adv - z0` (doc 05 §5).

---

## 2. The objective (`_optimize_once`)

`z_adv` starts at `z0 + init_noise·randn` (PGD-style random start). Per step:

```python
decoded  = vae.decode(z_adv)["continuous_mu_raw"]
controls = infer_primitives_from_decoded(raw0, decoded, decoded_base, bounds)
x_adv    = generate(raw0, controls, quantize=False)
logits   = victim((x_adv - center)/scale)

loss = lambda_cls     * targeted_loss(logits, Benign)      # CW margin or CE
     + lambda_latent  * ||z_adv - z0||^2                    # stay near source posterior
     + lambda_cost    * mean(|x_adv - raw0| / scale)        # small realized edit
     + lambda_realism * relu(mahalanobis(z_adv)/threshold - 1)   # stay in-distribution
     + lambda_recon   * mean(((decoded - decoded_base)/scale)^2) # (default 0)
loss.sum().backward(); optimizer.step()
```

- **Targeted loss** (`_targeted_loss`): `objective="cw"` uses the Carlini–Wagner margin
  `relu(strongest_other - target_logit + kappa)`; `objective="ce"` uses cross-entropy.
- **ε-ball projection**: after each step, if `epsilon_z` is set, `z_adv` is projected back into
  the L2 ball of radius `epsilon_z` around `z0`.
- Optional **cosine LR schedule** (`lr_schedule="cosine"`).

## 3. Restarts and selection

With `restarts > 1`, `_optimize_once` runs multiple random starts and `_select` keeps, **per
sample**, the restart with the smaller **realized** targeted margin
(`_targeted_margin` on the *quantized* logits; ≤ 0 iff the realized flow is classified Benign).
So a realized success always beats a failure, and among successes the strongest margin wins.
Selection is done on the final projected result, not the continuous one — you never keep a
restart that only "wins" before quantization.

## 4. Final realization & instrumentation (`attack`)

VAE and victim are frozen (`requires_grad_(False)`) but the decoder stays in the autograd graph
w.r.t. `z_adv`. The returned `LatentAttackResult` carries both the **continuous** and
**realized** vectors, controls, logits, `latent_l2 = ||z_adv - z0||`, `primitive_cost`,
`reconstruction_error`, `latent_distance_sq` (Mahalanobis), and optionally `grad_norms`
(mean per-sample L2 norm of the classifier-loss gradient at `x_adv`, `p`, `alpha`, decoder
output, and `z_adv` — a diagnostic for where the gradient bottleneck is).

The runner then applies the same hard frozen-feature assertion and the same `evaluate_cell`
used by the direct attack (docs 06–07), so the two methods are measured identically.

## 5. Realism caveat (important for the report)

The attack VAE and the IDR/Mahalanobis gate are the **same** per-class VAE. Therefore the IDR
here is **generator-relative**, and the runner reports it as `IDR_generator_relative`, not as
independent realism evidence. The independent evaluators remain PAVE (Level-A), the mined
density engine, and the internal realizability validator. This is stated in the runner
docstring (lines 8–10) and is the honest framing to use in the thesis.

## 6. CLI defaults (`run_cicids2017_vae_latent_attack.py::main`)

| Flag | Default | | Flag | Default |
|------|--------:|-|------|--------:|
| `--steps` | 120 | | `--lambda-latent` | 0.005 |
| `--learning-rate` | 0.08 | | `--lambda-cost` | 0.05 |
| `--objective` | `cw` | | `--lambda-realism` | 0.001 |
| `--epsilon-z` | 10.0 | | `--p-max` | 1460.0 |
| `--kappa` | 0.0 | | `--alpha-max` | 100.0 |
| `--restarts` | 1 | | `--mtu-cap` | 0.0 |
| `--lr-schedule` | `constant` | | `--seeds` | `42,43,44` |
| `--test-limit` | 1024 | | `--variant` | `full` |

(`LatentAttackConfig` internal defaults differ slightly — `steps=60`, `lr=0.05`,
`epsilon_z=5.0`, `init_noise=0.3` — but the runner's CLI overrides them to the values above for
the reported experiments.)

## 7. Ablation variants (`VARIANTS`)

Run with `--variant`:

| Variant | Override | Tests |
|---------|----------|-------|
| `full` | — | the complete proposed method |
| `no_latent_reg` | `lambda_latent=0` | does staying near `z0` matter? |
| `no_realism` | `lambda_realism=0` | does the in-distribution penalty matter? |
| `ce` | `objective="ce"` | CW margin vs cross-entropy |

Each variant is a different `method_id` and writes a **separate output tree** (never mixed with
the primitive baseline). Row selection, seeds, victims, validators, and metrics are identical
to the direct attack, so the two are directly comparable.

## 8. Why "primitive-constrained" latent (not a plain latent attack)

A naive latent attack would classify the raw decoder output — which is not guaranteed to be a
realizable flow. Here the decoder output is *only advisory*: it is projected onto the two
realizable primitives and regenerated through the physics map. So every reported adversarial —
whether found in primitive space (doc 07) or latent space (this doc) — satisfies the *same*
realizability contract. The latent search only changes *how* the `(p, alpha)` proposal is
generated, not what counts as valid.
