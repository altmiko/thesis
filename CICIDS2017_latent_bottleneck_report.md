# Where is the VAE-latent attack capability lost? — CICIDS2017-DistriNet bottleneck study

Offline research on the public CICIDS2017-DistriNet dataset. This report answers one question:

> Is the low VAE-latent ASR caused by the VAE/manifold itself, or by collapsing the decoder's
> 79-dimensional movement into only two primitives `(p, alpha)`?

It does so by placing three latent variants on a **nested ladder of decoder expressiveness** and
comparing them, under a matched search budget, against the direct-primitive control and the
unconstrained input-PGD ceiling. **The numbers were not tuned to look impressive**; where a method
is weak, that is reported and used as evidence about *where* capability is lost.

Existing implementations and results are preserved; this adds experiments only. All artifacts:
`outputs/cicids2017_latent_raw/`, `outputs/cicids2017_latent_masked/`,
`outputs/cicids2017_vae_latent_primitive_strong/`, `outputs/latent_strength_sweep/`,
`outputs/latent_bottleneck/`. Regenerate with:

```
PYTHONPATH="src;." python -m attack.run_cicids2017_latent_variants --variant raw    ... # VAE-Latent-Raw
PYTHONPATH="src;." python -m attack.run_cicids2017_latent_variants --variant masked ... # VAE-Latent-Masked
PYTHONPATH="src;." python -m attack.run_cicids2017_vae_latent_attack --restarts 2 --lr-schedule cosine ... # VAE-Latent-Primitive
PYTHONPATH="src;." python scripts/latent_strength_sweep.py            # ceiling + gradient norms
PYTHONPATH="src;." python scripts/compare_latent_variants.py          # tables below
```

---

## 1. The four attack representations (nested expressiveness ladder)

All latent variants optimise **only `z_adv`** with Adam; the VAE decoder stays in the classifier
gradient path; feature values are never optimised directly. Movement is anchored at `decode(z0)`
so `z_adv = z0` reproduces the pristine input (verified by the sanity checks, §6).

| Representation | What the decoder movement may change | Constraint applied | Proposed? |
|---|---|---|---|
| **VAE-Latent-Raw** | all 79 features (`x0 + (decode(z_adv)-decode(z0))`) | Layer-0 domain clamp only | **No** — diagnostic ceiling before any mask/realizability |
| **VAE-Latent-Masked** | only the 9 PERTURBABLE mask features; FROZEN copied from `x0`; DERIVED_EXACT recomputed | mask + Layer-0 domain clamp | Yes — *masked feature-space* attack (not Level-C) |
| **VAE-Latent-Primitive** | movement collapsed into `(p, alpha)` → realizability layer | full dependency recompute, integer/µs projection | Yes — the proposed thesis method |
| **Primitive-Direct** (control) | `(p, alpha)` optimised **directly** (no VAE in graph) | same realizability layer | Control |

`Input PGD` (unconstrained L∞ feature edits) is the classical upper bound on raw evasion.

Perturbation mask (`attack/masks/cicids2017_distrinet.py`): **PERTURBABLE (9)** = Flow Duration,
Total Length of Fwd Packet, Fwd Packet Length Max/Min/Std, Fwd IAT Total/Std/Max/Min;
**DERIVED_EXACT (7)** = Fwd Packet Length Mean, Fwd Segment Size Avg, Fwd IAT Mean, and the four
rate features; **FROZEN (63)** = everything else (backward-direction stats, packet counts,
ports/protocol, TCP flags, window/header fields, bulk/subflow/active-idle statistics).

---

## 2. Main comparison (pooled over 4 classes × 4 victims, seed 42)

Matched search config for the three latent variants: steps 150, lr 0.1 (cosine), ε_z 30,
2 restarts, λ_latent 0, λ_cost 0.02, λ_realism 0, CW objective. Denominator = clean-correct
malicious rows. Full table: `outputs/latent_bottleneck/comparison.md`.

| Method | Targeted-Benign ASR | Targeted Strict-Valid ASR | Validity | Feature/primitive cost | Latent distance |
|---|--:|--:|--:|--:|--:|
| Input PGD | 99.6 | **0.0** | 0.0 | 0.416 | N/A |
| Primitive-Direct | 78.5 | **78.5** | 100.0 | 2.667 | N/A |
| VAE-Latent-Raw | 80.7 | **0.0** | 0.0 | 1.295 | 6.83 |
| VAE-Latent-Masked | 6.9 | **6.9** | 95.3 | 0.316 | 20.06 |
| VAE-Latent-Primitive | 10.3 | **10.3** | 99.9 | 0.110 | 7.85 |
| VAE-Latent-Primitive (proposed cfg) | 7.7 | 7.7 | 99.9 | 0.072 | 2.51 |

**Reading of the ladder** (Input/Raw show high raw evasion but ~0 strict-valid; the two
constrained latent variants are close):

* **Raw achieves 80.7% targeted-Benign but 0.0% strict-valid** (PAVE 0.0%, mined 0.1%). The
  decoder *can* move the victim strongly — but essentially all of that movement is in FROZEN /
  out-of-domain features, so nothing survives validity. The manifold is not the limiter.
* **Masked (6.9%) ≈ Primitive (10.3%)**, with Primitive **slightly higher**, not lower. Giving
  the decoder unrestricted movement over every perturbable feature does **not** beat compressing
  it into `(p, alpha)`.
* **Primitive-Direct (78.5%)** dominates every VAE-latent variant: direct optimisation drives the
  same `(p, alpha)` to large realizable values the per-class decoder will not propose.

---

## 3. Per-class × victim targeted strict-valid ASR (%)

Both constrained latent variants concentrate all their success on the **LSTM / serial** victims of
the **DoS / Recon / BruteForce** classes; both are ≈0 on every **mlp / cnn** cell and across DDoS.

VAE-Latent-Masked:

| class \ victim | mlp | cnn | lstm | serial |
|---|--:|--:|--:|--:|
| DoS | 0.2 | 1.2 | 39.6 | 40.6 |
| DDoS | 0.0 | 0.0 | 0.0 | 0.0 |
| Recon | 0.8 | 0.0 | 18.6 | 8.6 |
| BruteForce | 0.0 | 0.4 | 0.0 | 0.2 |

VAE-Latent-Primitive:

| class \ victim | mlp | cnn | lstm | serial |
|---|--:|--:|--:|--:|
| DoS | 0.4 | 0.0 | 57.8 | 50.6 |
| DDoS | 0.0 | 1.8 | 6.5 | 8.0 |
| Recon | 0.0 | 0.0 | 0.2 | 0.0 |
| BruteForce | 0.2 | 0.0 | 0.0 | 39.7 |

Primitive-Direct (control) succeeds broadly (DoS/lstm 98, DDoS/lstm 100, Recon 97–100, …) because
it can push `p`/`alpha` to large realizable magnitudes; see comparison.md.

**Interpretation:** where the discriminative signal is in the *perturbable* forward length/timing
features (the recurrent LSTM/serial victims), realizable latent attacks genuinely work
(40–58%). Where the victim (mlp/cnn) keys on *frozen* features, every constrained latent attack
collapses — independent of whether movement goes through the mask or through `(p, alpha)`.

---

## 4. Strengthened-search sweep — the current latent attack has a ~10% ceiling

Pooled targeted strict-valid ASR of VAE-Latent-Primitive vs each search axis (seed 42,
256 rows/class; full table `outputs/latent_strength_sweep/sweep.md`). Baseline pooled = **8.1%**.

| axis | values → pooled TSV-ASR (%) |
|---|---|
| ε_z | 5→1.9, 10→8.1, 20→9.7, 40→9.7 (**saturates**) |
| restarts | 1→8.1, 4→9.9 |
| steps | 60→8.2, 120→8.1, 240→8.1 (**flat**) |
| λ_latent | 0→6.1, 0.005→8.1, 0.05→5.3 |
| λ_cost | 0→8.0, 0.05→8.1, 0.2→8.0 (**λ_cost=0 does not help**) |
| κ (CW margin) | 0→8.1, 5→8.5, 15→8.5 |
| lr schedule | constant→8.1, cosine→8.1 |

**Decisive:** ASR plateaus at ~8–10% under every axis. The **λ_cost=0 diagnostic** raises neither
ASR nor cost meaningfully — the attack is **not budget-limited and not search-limited**. Extra
budget cannot be spent usefully because the decoder will not propose the moves that would help.

---

## 5. Matched-budget comparison

Cost-capped targeted strict-valid ASR (fraction of clean-correct rows that are strict-valid Benign
AND whose realized feature cost ≤ cap; same denominator per method):

| Method | ≤0.1 | ≤0.25 | ≤0.5 | ≤1.0 | ≤2.0 | ≤∞ |
|---|--:|--:|--:|--:|--:|--:|
| Primitive-Direct | 0.1 | 16.4 | 33.0 | 43.9 | 62.3 | 78.4 |
| VAE-Latent-Masked | 0.1 | 2.9 | 5.2 | 6.1 | 6.7 | 6.9 |
| VAE-Latent-Primitive | 0.0 | 3.6 | 9.6 | 10.1 | 10.3 | 10.3 |

At **every** shared budget Primitive-Direct dominates, and the two VAE-latent methods **saturate**
(their curves are flat by cost ≤1.0). This is not an unmatched-cost artefact: even given unlimited
budget the latent methods stop at 6.9 / 10.3. Masked and Primitive are within a few points of each
other at all budgets — no representation advantage for unrestricted perturbable movement.

---

## 6. Gradient diagnostics — the constrained channels are the ones the victim ignores

Mean per-sample ‖∂L/∂·‖₂ at representative cells (`outputs/latent_bottleneck/comparison.json`,
`outputs/latent_strength_sweep/grad_norms.json`):

| cell | variant | TSV % | ‖dL/dx‖ | ‖dL/ddecoder‖ | ‖dL/dz‖ |
|---|---|--:|--:|--:|--:|
| DoS/lstm (high) | Raw | 0.0 (invalid) | 6.3e-3 | **5.1e-3** | 4.6e-3 |
| DoS/lstm (high) | Masked | 39.6 | 6.0e-3 | **2.9e-5** | 5.8e-4 |
| DoS/lstm (high) | Primitive | 57.8 | 5.9e-3 | 2.2e-5 | 9.3e-4 |
| DDoS/cnn (zero) | Masked | 0.0 | 1.3e-2 | 1.4e-4 | 3.5e-4 |
| DDoS/mlp (zero) | Primitive | 0.0 | 1.9e-2 | 1.9e-4 | 6.7e-4 |

For the primitive path, ‖dL/dα‖ (≈1–6e-3) ≫ ‖dL/dp‖ (≈1e-4): the timing dilation carries almost
all the usable gradient; padding barely moves the victim. Critically, **‖dL/ddecoder‖ collapses by
~100× the moment movement is routed through the mask or the primitives** (5.1e-3 raw → 2.9e-5
masked/primitive on DoS/lstm). The gradient that reaches `z` (‖dL/dz‖ ~1e-3) is small everywhere,
and *smaller* for the constrained variants — the victim is highly sensitive to features the
constraints hold fixed and nearly insensitive to the features the decoder is allowed to move.
(`nan` entries for Recon/primitive `dL/dp` arise because Recon flows are single-forward-packet, so
`p` is inactive by construction.)

---

## 7. Validity breakdown (pooled, %)

| Method | PAVE (Level-A domain) | mined density | strict (all gates) |
|---|--:|--:|--:|
| Input PGD | 0.0 | 0.0 | 0.0 |
| Primitive-Direct | 100.0 | 100.0 | 100.0 |
| VAE-Latent-Raw | 0.0 | 0.1 | 0.0 |
| VAE-Latent-Masked | 100.0 | 95.3 | 95.3 (PAVE ∧ mined ∧ mask frozen/derived) |
| VAE-Latent-Primitive | 100.0 | 99.9 | 99.9 (PAVE ∧ mined ∧ primitive realizability) |

Masked/Primitive are fully domain-valid; Raw and Input PGD are essentially never valid. The masked
strict gate is the *mask* realizability (frozen unchanged ∧ DERIVED_EXACT consistent), since it is
not a `(p, alpha)` attack; PAVE and mined density are the same independent validators as elsewhere.

---

## 8. Sanity checks (all pass — `src/attack/tests/test_latent_variants.py`, 13 cases)

For every latent variant: `z_adv` is the only optimizer leaf; detaching the decoder output kills
all gradient to `z_adv`; `z_adv = z0` reproduces the pristine input (no adversarial change); the
same seed reproduces identical realized vectors; the reported prediction is computed on the
projected/realized vector; the clean-correct denominator depends only on `(raw, victim)`, identical
across variants. Masked-specific: **no FROZEN feature changes** (structural + asserted at runtime),
**no DERIVED_EXACT feature is independently optimised** (recomputed from parents), and **every
changed feature lies in the perturbation/dependency closure**. Data hygiene: PAVE and the mined
engine are fit on `X_train_pristine`, the per-class VAEs on the train split, the IDR gate on the
val split — **test data is never used** for any generator/validator/scaler fitting. The primitive
attack's own tests (`test_vae_latent_primitive.py`, 12) still pass — the restart/schedule/grad
additions are backward compatible (restarts=1, constant LR reproduce the prior numbers bit-for-bit).

---

## 9. Conclusion — where is the capability lost?

Mapping the ladder onto the diagnostic rubric:

* **Raw is high (80.7% targeted-Benign) but 0% strict-valid; Masked ≈ Primitive and both low.**
  This is the *perturbability / constraint* regime, **not** the p/alpha-compression regime and
  **not** a manifold/latent-optimisation failure.

Concretely, the low VAE-latent ASR is **primarily a perturbation-mask / constraint bottleneck**:

1. **Not the `(p, alpha)` compression.** VAE-Latent-Masked, which lets the decoder move every
   perturbable feature freely, does **not** exceed VAE-Latent-Primitive (6.9% vs 10.3% pooled;
   comparable per-cell). Since masked is not materially greater than primitive, collapsing decoder
   movement to two primitives is **not** what suppresses the attack.
2. **Not the VAE manifold or latent optimisation.** The decoder can drive the victim hard
   (Raw = 80.7% targeted-Benign; ‖dL/ddecoder‖ and ‖dL/dz‖ are non-zero), and `z_adv` reaches the
   ε_z boundary; the strength sweep saturates at ~10% and λ_cost=0 does not help. The manifold and
   the optimiser are not the limiter.
3. **It is the mask / realizable feature set.** Raw's evasion power lives almost entirely in the
   FROZEN and out-of-domain features (its 80.7% is 0% valid). Once movement is restricted to the
   realizable/perturbable features, ‖dL/ddecoder‖ drops ~100× — the victim is sensitive to features
   that are, correctly, frozen (backward stats, flags, ports, packet counts, active/idle), and
   nearly insensitive to the forward length/timing features an attacker can actually change.
4. **It is victim-dependent.** Where the discriminative signal *is* in the perturbable forward
   features (recurrent LSTM/serial victims), realizable latent attacks reach 40–58%; where the
   victim (mlp/cnn) relies on frozen features, every constrained latent attack is ≈0.

**Answer to the thesis question:** the low VAE-latent ASR is **not** caused by the two-primitive
bottleneck and **not** by a VAE/manifold or latent-optimisation limitation. It is caused by the
**perturbation mask / realizable feature set**: the features a valid network-flow attacker may edit
carry little of the victims' decision signal, so any constraint-respecting latent (or direct)
attack that stays valid is inherently limited — Primitive-Direct only exceeds the latent methods by
spending large realizable `(p, alpha)` magnitudes the per-class decoder will not generate.

## 10. Limitations

* **Level-C not demonstrated** for any method; Masked is a *masked feature-space* attack and does
  **not** claim packet-level realizability. Raw is a diagnostic, not a proposed attack.
* **Generator-relative IDR** — the attack VAE and the IDR gate share the same per-class VAE, so IDR
  is not independent realism evidence (PAVE and mined density are independent).
* The matched-config strong runs are seed 42, 512 rows/class; the proposed-config primitive and the
  Primitive-Direct / Input-PGD controls retain their published 3-seed / 1024-row runs. Per-cell
  seed variance on the headline metrics was ≤0.7pp in the earlier 3-seed study.
* **Optional learned `g_φ(z)→(p, alpha)` head (§5 of the brief) was not implemented**: the
  experiments show the `(p, alpha)` compression is not the bottleneck (Masked ≤ Primitive), so a
  learned latent→primitive head cannot recover the lost capability — the missing signal is in the
  frozen features, outside any primitive or perturbable channel. Documented as unnecessary rather
  than pending.
