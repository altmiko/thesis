# 07 — The Direct Primitive-Domain Attack

**Source:** `src/attack/run_cicids2017_primitive_attack.py`.

This is the **baseline realizability-aware attack**. It optimizes the two primitives
`(p, alpha)` directly with Adam; the classifier gradient flows through the differentiable
realizability map (doc 05) but **not** through any VAE. The VAE appears only as the realism
gate. The runner docstring names it precisely: *a differentiable primitive-domain attack with
a VAE realism gate*, not a latent-VAE attack.

---

## 1. What is optimized

Per flow, two scalars `(p, alpha)`, parameterized so the hard bounds act on the **control**
space (not derived features) via a sigmoid:

```python
u, v = -2.0 + init_noise * randn(n)        # requires_grad
p     = p_hi     * sigmoid(u) * pad_active            # in [0, p_hi]
alpha = 1.0 + (alpha_hi - 1.0) * sigmoid(v) * timing_active   # in [1, alpha_hi]
```

- `p_hi`, `alpha_hi` are the **per-flow feasible caps** from `model.per_flow_bounds` — the
  train envelope caps **after the semantic gate** (doc 05 §4), so `p_hi = 0` where padding is
  inadmissible and `alpha_hi = 1` where timing dilation is inadmissible.
- `pad_active`, `timing_active` come from `model.infer_capabilities(raw)` (`caps.pad_allowed` /
  `caps.timing_allowed`) — forward padding is disabled for flows with no forward payload and
  timing dilation for single-forward-packet / zero-IAT flows, so the optimizer wastes no effort.
  Admissibility is enforced **twice** (control range *and* cap) so the artifacts agree.
- Init `u = v ≈ -2.0` ⇒ `sigmoid ≈ 0.12`, so both primitives start near identity; `init_noise`
  (default 0.5) adds a random PGD-style start that escapes the `relu(0)`/dead region.

## 1.1 Semantic primitive capabilities (`infer_capabilities`)

Numerical feasibility (inside the train envelope) is **necessary but not sufficient**: a
primitive can be numerically applicable yet unsupported by the source flow. Before any
optimization, `model.infer_capabilities(raw)` computes a per-flow, **conservative (fail-closed)**
admissibility gate (`PrimitiveCapabilities`: `pad_allowed`, `timing_allowed` bool tensors + one
machine-readable reason code per flow):

```python
has_fwd_packets = Total Fwd Packet >= MIN_FWD_PACKETS_FOR_PADDING   # default 1
has_fwd_payload = (Total Length of Fwd Packet > 0) & (Fwd Packet Length Mean > 0)
pad_allowed     = has_fwd_packets & has_fwd_payload                 # else NO_FORWARD_PAYLOAD
timing_allowed  = (Total Fwd Packet >= 2) & (Fwd IAT Total > 0)     # else SINGLE_FWD_PACKET / ZERO_TIMING_HEADROOM
```

- **Why:** aggregate CICFlowMeter features never reveal *which* forward packets carry
  modifiable application data, so the gate keys on the strongest evidence they do — forward
  payload presence for `p`, a non-zero forward IAT sequence for `alpha`. It is a **feature-level
  model of primitive realizability**, not a packet-semantics claim; ambiguous flows fall to the
  identity. The rule is **class-agnostic** — a DoS, DDoS, Recon, or BruteForce flow with the
  same structure is treated identically (no per-class special cases).
- **The Recon pathology it fixes:** a single-SYN probe (`Nf=1`, `TL_fwd=0`, `Fwd mean=0`)
  previously admitted `p=28` because `pad_active` only asked whether padding was *numerically*
  applicable. It now returns `pad_allowed=False` (`NO_FORWARD_PAYLOAD`) and `timing_allowed=False`
  (`SINGLE_FWD_PACKET`), so `A(x) = {identity}` and the flow is correctly left unmodified.
- **Enforced twice.** `per_flow_bounds` folds the gate into the caps
  (`p_hi := p_hi·pad_allowed`, `alpha_hi := 1` where `~timing_allowed`) — so
  `p_hi^semantic(x) = p_hi^envelope(x)·m_p(x)`, `m_p ∈ {0,1}` — *and* `optimize_primitives`
  multiplies the control range by the same masks. Belt-and-suspenders: the saved artifacts
  agree instead of one mechanism silently disabling a numerically-`>0` cap. The pre-gate caps
  are retained as `p_numeric` / `alpha_numeric` for auditing.
- **Terminology.** A "successful" sample here is **semantically-admissible primitive-realizable**
  (Levels A+B *and* every applied primitive supported by source-flow evidence), **not**
  functionality-verified — establishing that the traffic still performs the malicious behaviour
  needs packet-level replay (Level C, doc 05 §7).

**Attackable-source rate.** A flow with no admissible primitive can never be perturbed under
this threat model. Such flows stay in the clean-correct denominator (ASR is *not* inflated by
dropping hard/unmodifiable examples); each cell additionally reports `attackable_rate`
(`pad ∨ timing` over clean-correct), `pad_admissible_rate`, and `timing_admissible_rate`, which
explain category-level ASR variation without blaming the optimizer.

## 2. The objective (`optimize_primitives`)

```python
logits = victim( (model.generate(raw, controls) - center) / scale )
loss   = CE(logits, target=Benign)  +  cost_weight * ( sigmoid(u)*pad_active + sigmoid(v)*timing_active )
loss.sum().backward(); opt.step()
```

- Cross-entropy toward the **Benign** target (targeted). `loss.sum()` keeps each row
  batch-size invariant.
- The additive **cost term** penalizes how far each primitive is pushed toward its cap
  (`cost_weight = 0.01`), i.e. a soft budget — cheaper adversarials preferred.
- Optimizer: `Adam([u, v], lr)`, `steps` iterations. No LR schedule here.
- `center`/`scale` are the train-fit RobustScaler affine params (`transform.center/scale`); the
  victim always sees model-space input.

### 2.1 This is the Carlini–Wagner optimization scheme

The optimization here **is the Carlini–Wagner (C&W) attack scheme**, adapted to the primitive
domain:

- **Change-of-variables box handling.** C&W removes a box constraint by optimizing an
  unconstrained variable through a bounded squashing function (C&W use `½(tanh(w)+1)`; we use
  `p = p_hi·σ(u)`, `alpha = 1+(alpha_hi−1)·σ(v)`). The optimizer variables `u, v` are
  unconstrained; the primitives are always inside their feasible box for any real `u, v`.
- **Adam optimizer.** C&W solve the resulting unconstrained problem with Adam — exactly what
  `optimize_primitives` does.
- **Distance term traded against a classification loss.** C&W minimize `c·‖δ‖ + loss`; here
  the `cost_weight·(σ(u)+σ(v))` term is the perturbation-cost analogue traded against the
  targeted classification loss.

One deliberate difference: the **classification term is cross-entropy toward Benign**, not the
C&W logit margin `relu(max_{i≠t} logit_i − logit_t + κ)`. The *latent* attack
([`08_vae_latent_primitive_attack.md`](08_vae_latent_primitive_attack.md)) uses the true C&W
margin (`objective="cw"`). So: direct primitive attack = C&W change-of-variables + Adam + cost,
with a CE classification objective; latent attack = full C&W margin. Cite Carlini & Wagner
2017 (arXiv 1608.04644) for the optimization scheme in the report.

## 3. Discrete realization (the honest part)

After optimization the continuous controls are **projected and quantized** before anything is
measured:

```python
proj    = model.project_controls(raw, ctl)          # round p to int bytes; keep alpha
adv_raw = model.generate(raw, proj, quantize=True)  # integer-us timing; recompute ALL deps
# hard guarantee: every frozen feature byte-identical to pristine (atol=SCALER_ATOL)
assert allclose(adv_raw[:, frozen], raw[:, frozen], atol=1e-6, rtol=1e-4)
```

**All metrics are computed on `adv_raw`, i.e. after discrete projection** — never on the
continuous optimizer output. This is what makes the reported success realizable rather than a
floating-point artifact.

## 4. Evaluation (`evaluate_cell`)

Produces the per-sample masks and the additive normalized cost decomposition. The masks
(`clean_correct, evasion, benign, pave_valid, mined_valid, in_dist, realizable`, plus the six
realizability categories) and the metric composition (`strict_valid`, `true_idsr`) are exactly
as in [`06_validators.md`](06_validators.md).

The **cost decomposition** (`_decompose_cost`) reports the mean summed normalized-L1 movement
`|adv - raw| / scale`, split into `padding` (11 length columns), `timing` (8 IAT/duration
columns), `rate` (4 rate columns), and `total` (all 79, divided by F). This tells you *which*
kind of edit the attack spent its budget on.

## 5. The loop structure

```
for class_name in classes:                 # DoS, DDoS, Recon, BruteForce
    idx = _class_rows(test.y, cid, test_limit, 42 + cid)   # fixed rows (see doc 09)
    raw = X_test_pristine[idx]
    load base VAE + IDR for this class
    engine = build_ablation("A4", ...)     # Layer 0/1/2 mined validator
    bounds = model.per_flow_bounds(raw, envelope)
    for vname in victims:                   # mlp, cnn, lstm, serial
        victim = load_category_victim(...)
        for seed in seeds:                  # 42, 43, 44
            deterministic_runtime(seed)
            ctl     = optimize_primitives(...)
            adv_raw = generate(project_controls(ctl), quantize=True)
            masks, cost, yc, ya = evaluate_cell(...)
            save NPZ artifact + append metrics cell
```

Note the ordering: **rows are fixed per class before the victim/seed loops**, so the three
seeds attack the *same* rows across all four victims — this isolates optimization variance
from sampling variance.

## 6. CLI defaults (`main`)

| Flag | Default | Meaning |
|------|--------:|---------|
| `--classes` | `DoS,DDoS,Recon,BruteForce` | attack classes |
| `--victims` | `mlp,cnn,lstm,serial` | victim models |
| `--test-limit` | 1024 | rows per class |
| `--steps` | 40 | Adam iterations |
| `--learning-rate` | 0.1 | Adam LR |
| `--p-max` | 1460.0 | absolute forward-padding ceiling (bytes; ~1 MTU) |
| `--alpha-max` | 100.0 | absolute timing-dilation ceiling |
| `--mtu-cap` | 0.0 | optional per-packet resulting-length cap (0 = disabled) |
| `--cost-weight` | 0.01 | budget penalty weight |
| `--seeds` | `42,43,44` | optimization seeds |
| `--init-noise` | 0.5 | random-start magnitude |
| `--stage-a-dir` | `outputs/cicids2017_vae_stage_a` | per-class VAE + IDR |
| `--output-dir` | `outputs/cicids2017_primitive_attack` | results |

## 7. Artifacts written

`ensure_fresh_output_dir` refuses to overwrite a non-empty dir (no silent result
replacement). It writes `run_manifest.json` (full provenance, doc 09), `attack_results.json`
(all cells), and per-cell `attack_artifacts/<class>_<victim>_seed<seed>.npz` containing clean
& adversarial raw/scaled vectors, the realized and continuous `(p, alpha)`, the per-flow caps
`(p_hi, alpha_hi)`, clean/adv predictions and logits, the cost decomposition, every per-sample
mask, and full provenance arrays (row IDs, checkpoint SHA-256s, run id). It also records the
**semantic-capability provenance** per flow: `pad_semantic_allowed` / `timing_semantic_allowed`
(bool), `pad_disable_reason` / `timing_disable_reason` (reason codes), `attackable`
(`pad ∨ timing`), and both the numeric and semantic caps (`p_hi_numeric`/`p_hi_semantic`,
`alpha_hi_numeric`/`alpha_hi_semantic`) — so you can report e.g. "padding available for 62% of
DDoS sources vs 3% of Recon sources". Everything needed to recompute every rate and confidence
interval offline is in the NPZ — the printed JSON is only a convenience summary.

**Attackable-source rate.** Because a flow with no admissible primitive (`A(x) = {identity}`)
can never be perturbed under this threat model, each cell reports `attackable_rate` (and
`pad_admissible_rate` / `timing_admissible_rate`) alongside ASR. Such flows stay in the
clean-correct denominator (ASR is not inflated by dropping hard/unmodifiable examples); the
separate attackable-source rate explains category-level ASR variation without blaming the
optimizer.

## 7.1 Results & plausibility (semantic-gated rerun)

Canonical run: 4 classes × 4 victims (`mlp/cnn/lstm/serial`) × 3 seeds (42,43,44),
`test-limit=1024`, CPU, `outputs/cicids2017_primitive_attack/`. All rates are the mean over the
12 (victim × seed) cells per class, in **percent of clean-correct** malicious test rows.

| class | Untgt ASR | Tgt-Benign ASR | Tgt strict-valid ASR | strict-valid | IDR | **true-IDSR** | **Attackable** | pad-adm | timing-adm |
|---|---|---|---|---|---|---|---|---|---|
| DoS | 65.0 | 65.0 | 65.0 | 99.8 | 30.1 | **25.4** | 100.0 | 100.0 | 99.9 |
| DDoS | 98.0 | 72.8 | 72.8 | 100.0 | 0.0 | **0.0** | 100.0 | 100.0 | 100.0 |
| Recon | 0.7 | 0.6 | 0.6 | 100.0 | 16.1 | **0.0** | **0.7** | **0.1** | 0.7 |
| BruteForce | 77.0 | 77.0 | 77.0 | 100.0 | 0.0 | **0.0** | 100.0 | 100.0 | 100.0 |

Best per-victim targeted-benign ASR: DDoS/lstm 100.0, BruteForce/lstm 99.5, DDoS/serial 99.3,
DoS/lstm 98.2, DoS/serial 98.1; the MLP victim is hardest everywhere (DoS 16.2, DDoS 16.1,
BruteForce 42.8). Realizability + strict validity are ~100% across all 48 cells and every cell
passes the frozen-feature byte-identity assertion.

**These numbers are plausible and internally consistent** — the three surprising figures each
have a concrete, evidence-backed cause (not a bug):

| class | clean in-dist | adv in-dist | p median | α median | untargeted-success routing |
|---|---|---|---|---|---|
| DoS | 95.9% | 30.1% | 743 B | 19.5 | Benign 100% |
| DDoS | 95.5% | **0.0%** | 333 B | 1.79 | Benign 74%, **DoS 26%** |
| BruteForce | 88.9% | **0.0%** | 492 B | 2.87 | Benign 100% |
| Recon | 16.1% | 16.1% | 0 B | 1.00 | Benign 88%, DoS 11%, BF 1% |

- **IDR = 0% for DDoS/BruteForce is real off-manifold movement, not a broken gate.** Clean
  malicious flows pass their *own* class VAE gate ~90–96%, so the gate is well-calibrated; adv
  IDR collapses to 0 because the successful attack adds hundreds of bytes/forward-packet
  (`p` median 333–492 B), pushing flows far off the class manifold in latent space. The
  adversarials are **primitive-realizable but not realistic** — precisely the tension the IDR
  gate exists to catch. Consequently the **trustworthy "realistic evasion" metric is true-IDSR,
  and only DoS delivers it (25.4%)**; DDoS/BruteForce true-IDSR are 0.
- **Untgt ASR > Tgt-Benign ASR for DDoS is class routing, not noise.** 26% of DDoS evasions land
  in the adjacent **DoS** class rather than Benign; DoS/BruteForce route 100% to Benign, so their
  `Untgt == Tgt-Benign` exactly — a consistency signal.
- **Recon** clean and adv in-dist are identical (16.1%) because ~99% of Recon rows are the
  identity (unchanged), so realism status cannot change — matching the capability story (§1.1).

**Honest caveats.** (1) High DDoS/BruteForce ASR is *feature-realizable* evasion only; report
true-IDSR for realistic success. (2) These are **not cheap** attacks — sub-MTU paddings of
hundreds of bytes/packet (within the mined `p_max=1460` envelope) drive `cost_total_mean` ≈ 2–3.
(3) DoS is timing-heavy (`α` median ≈ 19.5), the others padding-heavy — consistent with the
per-class flow structure.

## 8. Relationship to the other primitive-domain methods

| Runner | What it optimizes | VAE in gradient path? | Realizability layer? |
|--------|-------------------|-----------------------|----------------------|
| `run_cicids2017_primitive_attack.py` (this) | `(p, alpha)` directly | no | yes |
| `run_cicids2017_vae_latent_attack.py` (doc 08) | `z_adv` in latent space | **yes** | yes |
| `run_cicids2017_input_baseline.py` | all 79 scaled features (L∞ PGD) | no | **no** (pure PGD) |
| `run_cicids2017_vae_attacks.py` | masked latent/residual (CFF or config mask) | yes | via `ResolvedMask` |

The input baseline is the classical "edit everything freely" attack; it exists to show what
success costs in validity (its frozen/dependency/discreteness validity is expected to be low).
All four share the same victims, test split, clean-correct denominator, validators, and target
class, so their metrics are directly comparable.

## 9. What role does each component play here? (and why masking is unnecessary)

In the **direct primitive attack**, three components that look central elsewhere play very
specific, limited roles — and one of them plays *no* role at all.

### VAE — realism gate only (NOT in the gradient path)

The per-class Stage-A β-VAE is used **only to score realism after the fact**. Concretely, the
attack computes the val-anchored Mahalanobis in-distribution mask
(`_idr_mask(base_vae, x_adv, idr_path)`) to produce the `in_dist` mask and the `IDR` /
`true_idsr` metrics. The classifier gradient **never flows through the VAE encoder or
decoder** — it flows `victim(generate(raw, controls))`, i.e. through the realizability map, not
the VAE. So here the VAE is a *realism evaluator*, not a generator. (This is the opposite of
the latent attack in doc 08, where the VAE decoder is inside the gradient path and the IDR is
therefore generator-relative.)

### CFF — plays NO role in this attack

The direct primitive attack does **not** load or use any CFF mask. CFF's job — ranking *which
of the 79 features* an attacker may perturb — is only meaningful when the attacker directly
edits many features. Here the attacker edits **two primitives**, and *which* features move is
decided by the realizability model's roles (Dp/Dt/D/C/R/I/F/Fᶜ, doc 05), not by a learned
selection. CFF is consumed **only** by the masked VAE ladder
(`run_cicids2017_vae_attacks.py`, `--mask-source cff`).

### Mined rules (and Layer 0/1) — independent validator only

The A4 `ConstraintEngine` (Layer 0 + Layer 1 + Layer 2/mined) is built and called **only as an
evaluator**: `evaluate_cell` runs `engine.validate(adv_raw)["pass_l0_l1_l2"]` to produce the
`mined_valid` mask, which is one of the three gates in `strict_valid = pave_valid ∧ mined_valid
∧ realizable`. The mined rules are **not** a generation penalty and are **not** projected onto
during optimization — the flow is made consistent by `generate()`'s exact recompute, and the
mined engine independently *checks* whether it also lands in the train density envelope. This
is the deliberate generation-vs-evaluation separation (docs 02–03, 06).

### Is masking necessary? No — the two-primitive parameterization subsumes it.

Correct: **feature masking is unnecessary in the primitive attacks.** A mask (CFF or the config
`DatasetMask`) does two jobs in the masked ladder — (1) declare which of the 79 features are
perturbable vs derived vs frozen, and (2) hand a dependency-recompute stage the parent/child
structure. Optimizing only `(p, alpha)` **already achieves both, more strongly**:

- **Degrees of freedom are 2, by construction** — there is nothing to mask down from 79. The
  attacker cannot touch a frozen feature because `generate()` never writes it (and the runner
  *asserts* every frozen/invariant/Level-C feature is byte-identical to the pristine row).
- **Dependencies are recomputed exactly** — every derived feature is regenerated from the
  projected primitives, so there is no "which features must be updated" bookkeeping for a mask
  to carry.

So a CFF/`DatasetMask` mask would be **redundant** here: the primitive parameterization is a
*tighter* constraint than any feature-selection mask (2 DOF + guaranteed dependency
consistency, vs. "these k features may move, now remember to recompute their children").
Masking is needed *only* in the masked ladder, whose attacker perturbs the feature vector
directly and therefore must be told what may move and what to recompute.

## 10. Novelty — what can and cannot be claimed

This method should be described as a **rigorous synthesis / instantiation and empirical study**,
**not** as a novel adversarial algorithm. Every major building block has close, named prior
art:

| Component | Established prior art |
|-----------|-----------------------|
| Change-of-variables box + Adam + cost/loss trade-off | **Carlini & Wagner 2017** (arXiv 1608.04644) — this *is* the C&W scheme |
| Perturb a few "primitive" features and recompute dependents | **FENCE** (ACM TOPS 2022); **Constrained Network Adversarial Attacks** (arXiv 2505.01328): "changing a primary feature affects a subset of others, which must be updated correspondingly" |
| Mine constraints on train, then check/project | **CaFA** (IEEE S&P 2024, arXiv 2501.10013) |
| Problem-space / realizable framing, side-effect features | **Pierazzi et al.** (IEEE S&P 2020, arXiv 1911.02142) |
| CFF perturbability scoring | **"A novel perturb-ability score … flow-based ML-NIDS"** (arXiv 2409.07448) — the **PAVE** score this repo is modeled on |
| Latent/VAE-based NIDS evasion | widely done (e.g. NIDSGAN, arXiv 2203.06694) |

**Safe to claim** (phrase precisely, and back with a related-work comparison table):
- a **novel integrated system** — a per-class β-VAE latent search collapsed *differentiably*
  into two physical primitives `(p, alpha)` through an exact CICFlowMeter recompute layer for
  CICIDS2017-DistriNet, scored under a *joint* validity + realism gate (True-IDSR);
- the **engineering artifact** (the differentiable decoder→primitive bridge);
- the **empirical contribution** — controlled comparison of feature-space PGD vs. direct
  primitive vs. latent-primitive under a strict realizability + PAVE + mined-density +
  Mahalanobis-realism gate.

**Do NOT claim:** a novel optimization method (it is C&W), the first realizable/problem-space
NIDS attack (Pierazzi/FENCE/NIDSGAN precede it), or a novel constraint-mining/perturbability
method (CaFA and the PAVE perturb-ability paper precede it — CFF is especially close to the
latter). Scope every claim as: *feature-space (Level A/B) realizability-aware evasion under a
conservative padding+timing threat model, white-box, not packet-verified.*
