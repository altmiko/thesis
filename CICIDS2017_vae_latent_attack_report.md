# VAE Latent-Space Primitive-Constrained Attack — implementation & audit (CICIDS2017-DistriNet)

Offline, controlled research on the public CICIDS2017-DistriNet dataset. This report covers
the correction of the primitive baseline and the implementation of the genuine VAE
latent-space attack (the proposed thesis method), organized by the 17 requested deliverables.
Guiding rule honored: **the numbers were not tuned to preserve the previous 78.5%**; where the
latent method is weaker than direct primitive optimization, that is reported truthfully.

---

## 1. Previous architecture and why it was not a VAE attack

The prior method optimized the two primitives `(p, alpha)` **directly** with Adam; the
classifier gradient flowed `p,alpha → primitive→feature map → classifier`. The VAE never
appeared in the gradient path — it was loaded only for the Mahalanobis IDR realism metric.
Removing the VAE left the attack mathematically unchanged. It is therefore a *differentiable
primitive-domain attack*, now retained under the identifier **`primitive_direct`** as a
baseline (Part B), not the proposed VAE method.

## 2. New VAE latent attack architecture (`vae_latent_primitive`)

The per-class β-VAE decoder is placed **inside** the attack gradient path. The attacker
optimizes a latent displacement; the decoder proposes a traffic-modification direction that is
collapsed into the realizable primitives and pushed through the same realizability layer as the
baseline. The decoder cannot independently change arbitrary aggregate features — only the two
primitives reach the final vector.

## 3. Computational graph (gradient flow)

```
x0 (pristine raw)
  └─ (−center)/scale ─► x0_scaled ─► VAE encoder ─► z0 = μ(x0)         [detached baseline]
                                                     │
        z_adv = z0 + δz     ◄── ONLY optimizer variable (random start; ε_z ball)
             │
             ▼
        VAE decoder ─► x_decoded_raw(z_adv)         decode(z0) = baseline (constant)
             │
             ▼   infer_primitives_from_decoded  (relu projection of decoder movement)
        p̂ = relu(mean Δ fwd-length signals)         α̂ = 1 + relu(mean log fwd-timing ratios)
        (clamped to per-flow feasible caps; α̂≡1 when Total Fwd Packet < 2)
             │
             ▼
        realizability layer  primitive_model.generate(raw0, {p̂, α̂})   ► x_adv_raw
             │
             ▼   (−center)/scale ─► victim classifier ─► targeted-Benign loss
             ▲──────────────────── ∂L/∂z_adv flows back through the decoder ──────────────────
```

The gradient path `classifier → realizability → latent→primitive → decoder → z_adv` is
proven by tests (§9). Detaching the decoder output removes all gradient to `z_adv`.

## 4. Exact optimization objective

Per-sample latent optimization (Adam over `δz` only):

```
minimise  L(z_adv) =  λ_cls · L_target(victim(x_adv), Benign)
                    + λ_latent · ‖z_adv − z0‖²
                    + λ_cost   · mean_f |x_adv − x0|_f / scale_f
                    + λ_realism· relu( D²_maha(z_adv)/τ² − 1 )
                    + λ_recon  · ‖decoded_adv − decoded_base‖²/scale   (optional, default 0)
s.t.  ‖z_adv − z0‖₂ ≤ ε_z   (projected each step)
```

`L_target` is a targeted CW margin (default) `max(max_{k≠B} logit_k − logit_B + κ, 0)` or CE.
Defaults: λ_cls 1, λ_latent 5e-3, λ_cost 5e-2, λ_realism 1e-3, ε_z 10, steps 120, lr 0.08,
random start σ 0.3, target = Benign.

## 5. Decoder → primitive mapping (deterministic, differentiable; Part E)

Movement is read relative to `decode(z0)` (cancels the constant reconstruction bias):

```
p̂  = relu( mean[ Δ Fwd Pkt Len Mean, Δ Fwd Pkt Len Max, Δ Fwd Pkt Len Min,
                 Δ Total Length of Fwd Packet / Nf ] )            # bytes, ≥ 0
α̂  = 1 + relu( mean[ ln(Fwd IAT Total ratio), ln(Fwd IAT Mean ratio),
                     ln(Flow Duration ratio) ] )                  # ≥ 1, delay-only
```

`relu` is the differentiable projection onto the feasible primitive direction: identity
decoder movement maps exactly to `p=0, α=1`; padding cannot shorten packets and dilation
cannot compress a flow, so clamping the negative direction is correct, not merely convenient.
A random latent start escapes the `relu(0)` dead point. Both primitives are clamped to the
per-flow feasible caps; `α̂≡1` for single-forward-packet flows *inside the graph*. No learned
primitive head is used (Part F path not needed; documented as the fallback).

## 6. Realizability layer (Part M/N)

Identical to the baseline: `primitive_model.generate` recomputes every feature-space
reconstructable dependency; frozen/invariant/Level-C features are copied from pristine raw.
During optimization the continuous map is used; for the **final** result the primitives are
projected (round `p` to integer bytes; µs-quantize timing) and every dependency is recomputed,
then the realized vector is **re-classified** and re-validated. All reported success is on the
realized vector.

## 7. Feature-role map (Part A1/A3)

Precise taxonomy replacing the old "primitive-controlled" label (no CICFlowMeter feature is an
attack variable; only `p, alpha` are):

| tag | meaning | examples |
|---|---|---|
| Dp | exactly derived from packet-length primitive p | Total Length of Fwd Packet, Fwd Packet Length Min/Max/Mean, Fwd Segment Size Avg, Packet Length Mean/Std/Variance, Average Packet Size |
| Dt | exactly derived from timing primitive α | Fwd IAT Total/Mean/Std/Max/Min, Flow IAT Mean |
| C | conditionally / conservatively reconstructed | Packet Length Max/Min, Flow Duration, Flow IAT Max |
| R | rate-derived | Flow Bytes/s, Flow/Fwd/Bwd Packets/s |
| I | **proven invariant** under the relevant primitive | Fwd Packet Length Std (uniform shift) |
| F | genuinely unaffected / frozen | ports, protocol, all bwd stats, flags, headers, Down/Up, window bytes, Fwd Seg Size Min |
| Fᶜ | **UNRESOLVED, held constant** (would change under real packet edits; not aggregate-reconstructable) | Fwd Act Data Pkts, Subflow Fwd Bytes, Fwd Bytes/Packet/Bulk Avg, Fwd Bulk Rate Avg, Flow IAT Std/Min, **Active/Idle Mean/Std/Max/Min** |

`I` (proven invariant) is explicitly distinguished from `Fᶜ` (held constant, not claimed
physically invariant). Active/Idle are `Fᶜ`: they depend on packet-level burst timing and
cannot be reconstructed from aggregates, so they are held — not claimed invariant.

## 8. Level A/B/C validity (Part U)

- **Level A** — feature domain/type/range (PAVE, independent).
- **Level B** — feature dependency/algebraic consistency (internal realizability validator +
  mined density engine).
- **Level C** — actual packet modification + CICFlowMeter re-extraction. **NOT** demonstrated.

Both attacks demonstrate **Level A+B** only. Wording is "realizability-aware /
primitive-consistent / dependency-consistent"; "packet-realizable" is reserved for a future
PCAP→edit→re-extract backend (`NullPacketBackend` interface is in place). Fᶜ features remain
explicitly marked.

## 9. Tests proving the VAE lies in the attack path (Part W)

`src/attack/tests/test_vae_latent_primitive.py` (all passing):

- `test_classifier_gradient_flows_through_decoder_to_z` — `z.grad` finite, non-zero.
- `test_detaching_decoder_breaks_attack_gradient` — detached decoder ⇒ loss has no z path.
- `test_decoder_movement_changes_controls_and_adversarial_vector` — moving z changes p̂/α̂ and x_adv.
- `test_optimizer_parameter_list_contains_only_z_adv` — only `z_adv` is a leaf; `p`/`alpha` are non-leaf.
- `test_single_packet_timing_disabled_in_latent_graph` — `Total Fwd Packet<2 ⇒ α≡1` in-graph.
- `test_final_result_reclassified_after_projection` — realized logits match victim on projected vector; `p` integral.

Baseline realizability tests (`test_primitive_controls.py`, 13) also pass (roles/invariants updated).

---

<!--RESULTS-->
## 10 & 11. Results — baseline and proposed (pooled/micro, seed 42; full tables in `comparison.md`)

| Method | Targeted-Benign ASR | Targeted Strict-Valid ASR | Primitive cost | PAVE | Mined | Realizability | IDR/realism |
|---|--:|--:|--:|--:|--:|--:|--:|
| Input PGD (unconstrained) | 99.6 | **0.0** | 0.416 | 0.0 | 0.0 | 0.0 | 24.5 |
| Primitive-Direct (baseline) | 78.5 | **78.5** | 2.673 | 100.0 | 100.0 | 100.0 | 7.6 |
| VAE-Latent-Primitive (proposed) | 7.7 | **7.7** | 0.072 | 99.9 | 100.0 | 100.0 | 63.5 (gen-rel.) |

Denominator = clean-correct malicious test rows. Input PGD fools the victim (99.6%) but is
**0% strict-valid** — free feature edits are unrealizable. Both realizability-aware methods are
~100% valid. The proposed latent method is **much weaker (7.7%) but far cheaper (0.072 vs 2.67)
and stays in-distribution** (IDR 63.5% generator-relative vs 7.6% for direct).

### Proposed method is class/victim-dependent (mean±std over 3 seeds)

Strong where small manifold-plausible moves suffice, near-zero elsewhere:
DoS/lstm **57.7±0.1**, DoS/serial **50.0±0.6**, BruteForce/serial **15.2±0.7**; ≈0 on every
mlp/cnn cell and all DDoS/Recon cells. Seed std ≤0.7pp. Mean latent Δ ranges 0.13–6.8.

## 12. Matched-budget (targeted strict-valid ASR vs mean physical cost)

| point | Direct cost | Direct TSV-ASR | Latent cost | Latent TSV-ASR |
|---|--:|--:|--:|--:|
| 1 | 1.72 | 35.3 | 0.066 | 7.2 |
| 2 | 1.89 | 46.3 | 0.071 | 7.6 |
| 3 | 2.53 | 60.8 | 0.073 | 7.5 |
| 4 | 4.32 | 77.9 | 0.805 | 7.6 |

**Decisive finding:** the latent method's ASR is flat (~7.6%) even with the cost penalty removed
(λ_cost=0 only raises cost 0.07→0.81). It is **manifold-limited, not budget-limited** — the
per-class decoder will not propose the large forward paddings direct optimization uses, so extra
budget cannot be spent usefully. Direct dominates at every comparable budget.

## 13. VAE ablation (pooled/micro, seed 42)

| Variant | Targeted Strict-Valid ASR | Latent Δ | Primitive cost | IDR (gen-rel.) |
|---|--:|--:|--:|--:|
| direct primitive | 78.5 | N/A | 2.673 | 7.6 |
| latent, no latent regularizer | 5.9 | 3.02 | 0.066 | 60.7 |
| latent, no realism term | 7.1 | 3.40 | 0.086 | 53.5 |
| full proposed method | **7.7** | 2.52 | 0.072 | **63.5** |

Both the latent regularizer and the realism term help: full > no-realism > no-latent-reg on
ASR, and the realism term measurably raises in-distribution rate (63.5 vs 53.5).

## 14. Latent-distance & primitive-cost statistics (successful strict-valid, pooled seed 42)

1254 successful samples: ‖z_adv−z0‖ mean 7.00 / median 7.07 / p95 9.93; primitive cost mean
0.249 / median 0.239; generator-relative Mahalanobis² mean 749. Successful latent attacks use
larger latent moves (≈7) than the pooled average (≈2.5) — evasion needs a real latent
excursion, but one bounded by ε_z and the manifold.

## 15. Statistical robustness

3 seeds (fixed eval rows, random latent start); per-cell std ≤0.7pp on all headline metrics
(see per-class table). Full per-cell tables, bootstrap-ready masks, and instrumentation arrays
(z0, z_adv, decoded, controls, logits) are saved under `outputs/cicids2017_vae_latent_attack/`.

## Part R — what the VAE contributes

The decoder converts a bounded latent excursion into small, in-distribution forward
padding/delay proposals; the realizability layer turns those into valid flows. Its poor
reconstruction of tiny-scale Fᶜ features (Idle/Active) never reaches x_adv because those are
held constant — evidence supporting the Fᶜ classification. The VAE's contribution is
*plausibility and low cost*, not raw evasion power.

**Answer to the thesis question:** optimizing in the learned VAE latent space *does* produce
effective, constrained, distributionally-plausible adversarial flows on some victims (LSTM/serial
DoS) at very low cost and full Level-A/B validity — but it is **less effective and manifold-
limited compared with direct primitive-space optimization** under the identical realizability
layer. The contribution is the framework and the honest comparison, not a higher ASR.

*(Full tables incl. per class×victim and the 10 audit answers: `comparison.md`.)*


---

## 16. Remaining limitations

- **Level-C not demonstrated** — no PCAP edit / CICFlowMeter re-extraction; Fᶜ features held.
- **Realism gate is generator-relative** — the attack VAE and the IDR gate are the same
  per-class VAE, so IDR for the latent method is not independent evidence (Part L, Option 2).
  PAVE, mined density, and the internal realizability validator remain independent. A
  separately-trained evaluation VAE is future work (Option 1).
- **Latent attack magnitude is manifold-limited** — the decoder, trained to reconstruct the
  malicious class, cannot propose the large forward paddings that direct optimization uses, so
  the proposed method reaches lower ASR (see §10). This is a genuine property of latent-space
  generation, reported honestly, not a bug.
- **Deterministic decoder→primitive inference** (relu projection) is used; a learned
  self-supervised control head (Part F) was not required and is documented as the fallback.

## 17. Defense-safe methodological interpretation

**What the proposed method shows.** Optimizing inside a learned per-class VAE latent space,
with the decoder in the gradient path and the output constrained through a realizability layer,
produces targeted Attack→Benign adversarial flows that are Level-A/B valid and stay close to
the learned malicious-class manifold at low perturbation cost. It answers the thesis question
directly: latent-space generation yields *constrained, distributionally plausible* adversarial
network-flow examples.

**What it does not show.** It does not prove packet-level realizability (Level C), and — because
the realism gate shares the generator VAE — the IDR score is not independent realism evidence.
On raw effectiveness the **direct primitive baseline is stronger** (it can push primitives to
large realizable values the manifold decoder will not propose); the honest thesis contribution
is the *framework* and the *comparison*, not a headline ASR. The proposed method's value is
lower-cost, manifold-plausible perturbations under identical realizability constraints.
