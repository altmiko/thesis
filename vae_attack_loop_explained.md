# VAE + Attack Loop, Explained — with Literature-Grounded Novelty Plan

*Scope: this document (1) reverse-documents the VAE (`src/vae/`) and the adversarial
attack machinery (`src/attack/`) as they exist in this repo, (2) situates them against
the source paper (Zhipeng He, *VAE-TabAttack*) and the adjacent constrained-NIDS-attack
literature (NetDiffuser, CPAD/CAA, problem-space attacks), and (3) proposes concrete,
literature-grounded, ASR-raising changes to the VAE architecture and attack loop that make
this VAE **meaningfully different** from He's, so novelty can be claimed and defended.*

*All code claims below were read from source. Forward-looking claims (expected ASR/IDSR
effects) are marked `[INFERENCE]`.*

---

## 0. TL;DR

- The repo's VAE (`MixedInputBetaVAE`, "Path C") is a **heavily domain-specialised fork** of
  He's mixed-input tabular VAE. It already diverges from He in five ways: (a) **8 per-class
  β-VAEs** instead of one global VAE; (b) a **protocol-embedding encoder** with a
  nearest-reference decode; (c) **four type-specific decoder heads** (continuous Gaussian,
  independent-binary BCE, protocol softmax, pseudo-binary); (d) a **structured continuous
  decoder** that enforces domain rules *by construction* (softplus ordering, TTL range,
  integer packet count via straight-through, variance = std², optional physics P2/P4/P5);
  (e) **differentiable domain-constraint losses** + free-bits/β=0.5 anti-collapse.
- **The single most consequential divergence is a deletion, not an addition:** He's VAE has an
  **auxiliary latent classifier head `h_ω(z)`** that shapes the latent into a *discriminative,
  attack-friendly* manifold. The repo has **no** such head, and worse, splits the model into
  per-class VAEs — so each VAE's latent knows nothing about the IDS decision boundary or the
  other 7 classes. This is the structural reason latent attacks here need decoder-residual
  anchoring, masks, GMM restarts, and still under-perform. **This is the biggest, best-grounded
  place to add novelty that raises ASR.**
- The attack loop is a competent **latent-PGD / latent-C&W** engine (adaptive step, GMM
  restarts, ε-ball projection, mask + protocol reimposition, anchored decoder residual) plus
  input-space and VAE-constrained-input baselines. It optimises **only** the classifier loss;
  the in-distribution (Mahalanobis/IDR) test that defines success is applied **post-hoc**, never
  inside the optimiser. NetDiffuser's core lesson (perturb only weakly-dependent features) and
  CAA's lesson (discrete/constrained features need special handling) are **absent from the
  attack objective** — but note CAA solves the discrete part with *slow* evolutionary search
  (MOEVA), which this repo should avoid if speed is a claim (see Part S).
- Proposed novelty is a coherent bundle (Part G): **(N1)** a single **conditional,
  adversarially-structured VAE** with a surrogate-IDS latent head + supervised-contrastive latent
  geometry (recovers and surpasses He's `h_ω`); **(N2)** a **relative-feature-group decoder** that
  models NetDiffuser "relative" groups jointly (low-rank/autoregressive) instead of hand-coding
  every rule; **(N3)** a **manifold-in-the-loop attack objective** (differentiable Mahalanobis/IDR
  penalty inside PGD/C&W) that directly optimises IDSR; **(N4)** **fast differentiable discrete
  heads** (Gumbel-Softmax/Concrete relaxation for protocol/binary) that get CAA's discrete-flip
  benefit *without* its evolutionary search; **(N5)** a **learned "toward-benign" latent
  direction**; **(N7)** an **amortised one-shot perturbation generator**. N1–N2 are the
  architectural novelty; N3–N5+N7 are the attack-loop novelty.
- **Speed is a first-class claim (Part S).** A shallow VAE decode is a *single* forward pass, so
  latent attacks here are inherently far cheaper than (a) diffusion generation (NetDiffuser runs a
  T-step reverse chain per sample) and (b) evolutionary constrained search (CAA/MOEVA runs a
  population × generations). N4 (differentiable discrete, no search) and N7 (amortised generator,
  O(1) inference) are chosen specifically to *keep* that speed advantage while raising ASR.

---

# PART A — The VAE architecture (`src/vae/model.py`)

## A.1 Data contract (39-dim CICIoT2023 "Schema A")

Fixed 39-feature order (`preprocessing/schema.py:FEATURE_NAMES`); a train-only `RobustScaler`
transforms everything. `vae/schema.py:get_partition()` splits the 39 columns into four
**hard-coded, domain-defined** index groups (not data-derived):

| Partition key | Indices | Count | Meaning |
|---|---|--:|---|
| `protocol_idx` | `[1]` | 1 | `Protocol Type` (categorical, raw ∈ `{0,1,2,6,17,47}`) |
| `continuous_idx` | `[0,2..14,30..38]` | 23 | rates, sizes, IAT, TTL, statistical aggregates |
| `independent_binary_idx` | `[15..21,24,25,28,29]` | 11 | app/flag indicators trained with BCE |
| `derived_binary_idx` | `[22,23,26,27]` | 4 | TCP/UDP/ICMP/IGMP — **derived from protocol argmax, never free** |

`PROTOCOL_ALLOWLIST = [0,1,2,6,17,47]` → 6 protocol classes. `PROTOCOL_TO_BINARY` maps
`{6:TCP, 17:UDP, 1:ICMP, 2:IGMP}`; IGMP/`0`/`47` have no derived bit column.

## A.2 Encoder (`encode`)

- Input = 39-dim scaled vector, but the **protocol column is replaced by an embedding**:
  `protocol_embed = nn.Embedding(6, protocol_embed_dim=4)`.
- Protocol index is recovered by a **nearest-reference lookup**: the scaled value of each of the
  6 allowlist protocols is precomputed into buffer `ref_proto_scaled`; the encoder takes
  `argmin |x_scaled[:,1] − ref|`. (`register_protocol_references(scaler)` must be called before
  `encode`, else it raises — the placeholder buffer is all-zeros.)
- Encoder input dim = `n_continuous(23) + n_independent_binary(11) + n_pseudo(0) + embed(4) = 38`.
- Body = `Linear→BatchNorm→ReLU` stack over `encoder_hidden=[128,64]`, then
  `encoder_out: Linear(64, 2·latent_dim)` → split into `mu`, `logvar`.
- `logvar` clamped to `[-6, 6]` (`latent_logvar_bounds`). `latent_dim = 16` per class.
- `reparameterize`: sampling **only in `training` mode**; at eval `z = mu` (deterministic —
  same choice He makes for stable attacks).

## A.3 Decoder heads (`decode_internal`, `decode_to_39`)

Decoder body: `Linear→BatchNorm→ReLU` over `decoder_hidden=[64,128]`. Then **four heads**:

1. `head_continuous_mu: Linear(128, 23)` → continuous means (scaled space).
2. `head_continuous_logvar: Linear(128, 23)` → per-feature log-variance, clamped `[-7, 2]`
   (heteroscedastic; reinterpreted as log-scale when `continuous_likelihood='laplace'`).
3. `head_binary: Linear(128, 11)` → independent-binary logits (BCE targets are raw {0,1}).
4. `head_protocol: Linear(128, 6)` → protocol logits (softmax over allowlist).
5. `head_pseudo` — present in code, unused (`n_pseudo_binary=0`).

### A.3.1 Structured continuous decoder (`_structure_continuous_raw`) — the domain "physics"
When `use_structured_continuous_decoder=True` (it is, in `config.py`), continuous means are
mapped to **raw space** and repaired *by construction* so the decoder can only emit
domain-consistent rows:

- `mode='full'`: `clamp_min(0)` on all continuous (non-negativity).
- Ordering: `min = softplus(·)`, `avg = min + softplus(·)`, `max = avg + softplus(·)`
  → guarantees `Min ≤ AVG ≤ Max`.
- Packet `Number`: `1 + softplus(·)`, then **straight-through rounding**
  (`round(n) − n` detached) → positive integer count with usable gradient.
- `TTL = sigmoid(·/32)·255` → range `[0,255]`.
- `Std = std_floor + softplus(·)`; `Variance = Std²` (exact G6 by construction).
- **Optional physics decoder** (`use_structured_physics_decoder`, **OFF** by default):
  P4 `Std ≤ 0.5·(Max−Min)`; P5 singleton flows (`Number≤1`) → zero variance and collapsed
  min/avg/max; P2 `Tot sum = Number·AVG`, `Tot size = AVG`.

The repaired raw vector is mapped back to scaled space (`continuous_raw_to_scaled`). Derived
binaries come from `derive_binaries_from_protocol_index(argmax logits)`; protocol scaled value
is gathered from `ref_proto_scaled`. `decode_to_39` supports `mode='soft'` (sigmoid binaries,
for gradients) and `mode='hard'` (thresholded binaries, straight-through in some paths, for the
final adversarial sample). Output is a full 39-dim **scaled** vector.

## A.4 Persistent buffers
`register_protocol_references(scaler)` fills 9 buffers (`ref_proto_scaled`, per-group
`center`/`scale` for continuous / independent-binary / derived-binary / protocol). These let
`decode_to_39` do all scaled↔raw conversion **without a Python-side scaler call**, which is what
makes the inner PGD/C&W loop GPU-resident (no per-step CUDA sync). Loaders tolerate these as the
only legitimately-missing keys.

---

# PART B — Training objective (`src/vae/losses.py`, `config.py`)

## B.1 ELBO (`compute_elbo`)
Loss = sum of:
- **Continuous reconstruction**: Gaussian NLL (default) or Laplace NLL (`continuous_likelihood`).
  Per-feature, summed over features, mean over batch; optional per-feature weights and per-sample
  NLL cap. Laplace (L1) is offered as the heavy-tail-robust alternative for flow features
  (Rate/IAT/Tot sum).
- **Independent-binary reconstruction**: BCE-with-logits vs raw {0,1} targets (11 dims).
- **Protocol reconstruction**: cross-entropy over 6 classes, ×`protocol_loss_weight=2.0`,
  optional class weights (`use_protocol_class_weights`, power 0.5).
- **Pseudo-binary**: MSE (inactive).
- **KL**: `−0.5·Σ(1+logvar−μ²−e^logvar)`, with **free-bits**: `per_dim_kl.clamp(min=λ=0.1)`
  before summing → each latent dim keeps ≥0.1 nats, preventing posterior collapse. ×`β`.
- **Constraint loss** (`_compute_constraint_loss`, ×0.1): differentiable raw-space penalties
  mirroring the validator — non-negativity, TTL≤255, `Min≤AVG≤Max` ordering, `Variance=Std²`,
  `Number≥1`, integer-`Number` (squared fractional-distance-to-round; the old `sin²(πN)` was
  replaced because it loses float32 precision for large counts).
- **Physics constraint loss** (`_compute_physics_constraint_loss`, weight **0.0** = dormant):
  P2/P4/P5 raw penalties.
- **Raw-relative continuous loss** (optional, off by default): `|x_raw−x̂_raw|/(|x_raw|+ε)` with
  optional tail-focus on a high-quantile subset.

## B.2 Anti-collapse config
`β_target=0.5`, `free_bits_λ=0.1`, `beta_warmup_epochs=10` (linear `BetaScheduler`). The comment
block in `config.py` documents that β=0.5 + free-bits was the fix for classes previously
collapsing >8 of 16 latent dims. Collapse diagnostic uses **measured** per-dim KL with a 0.01
threshold.

## B.3 What the loss does *not* contain (important for novelty)
- **No classification / discriminative term.** He's ELBO has `+α·L_cls(h_ω(z), y)`. Here there is
  none — the latent is shaped purely by reconstruction + KL + domain constraints. Combined with
  per-class training, the latent has **no knowledge of the IDS decision boundary**.
- **No adversarial / robustness term**, no latent smoothness/consistency term.
- **No inter-feature covariance term** — feature dependencies are enforced only through the
  hand-coded structured decoder, not learned.

---

# PART C — The attack loop (`src/attack/`)

## C.1 Shared infrastructure (`latent_infra.py`)

- **`PerturbationMask`** — 3-tier feature policy built from artifacts
  (`from_preprocessing_artifacts`): `near_zero_iqr_features.json` +
  `netdiffuser_categorization.json` + static mutable/override sets. Decision order per feature:
  auto-freeze → full-override → capped-partial (near-zero) → NetDiffuser `discrete`→full →
  NetDiffuser `relative`→partial(δ=0.3) → else frozen. `apply()` clamps partial-feature deltas to
  ±0.3 and hard-restores frozen features to original. (This is the "3-month-old" hand-listed
  taxonomy flagged for rebuild in `CLAUDE.md`.)
- **`ProtocolValidator`** — checks scaled protocol column against allowlist.
- **`reimpose_protocol_features`** — every decode step overwrites `Protocol Type` + the 4 derived
  bits with the original (detached): **protocol is frozen by design** during attacks.
- **`apply_decoder_residual`** — the key trick. Because per-class VAEs are not identity maps,
  the attack does **not** use `decode(z_adv)` directly. It computes
  `x_delta = decode(z_adv) − decode(z_orig).detach()`, then `x_candidate = x_original + x_delta`,
  then masks and reimposes protocol. This cancels reconstruction bias and keeps only the *latent
  direction* the attack found, applied around the real sample.
- **`MahalanobisOutlierDetector`** — per-class latent Mahalanobis gate (fit on VAL, not train),
  used **post-hoc** for the in-distribution rate (IDR). Not in the optimiser.
- **`LatentGMMPrior` / `fit_or_load_latent_gmm`** (`latent_gmm.py`) — Bayesian GMM fit on latent
  μ of VAL samples, used to seed attack restarts from high-density latent regions.
- **`latent_restarts.py`** — restart initialisers `encoded+jitter+gmm`, per-class ε overrides
  (DDoS/DoS/Mirai=0.8, Web=1.0, Recon/Spoofing=0.5, Benign/BruteForce=0.3).

## C.2 Latent PGD (`latent_pgd.py:latent_pgd_attack`)
Per restart, per step:
1. `z_adv.requires_grad`; `x_soft = apply_decoder_residual(decode_to_39(z_adv,'soft'), anchor, x_orig, mask)`.
2. `logits = classifier(x_soft)`; `loss = _objective_loss` (untargeted CE, or targeted CE, or
   **C&W margin** `max(Z_y − max_{i≠y}Z_i + κ, 0)` + `λ·‖z−z_orig‖²`).
3. `grad = ∂loss/∂z_adv`; `z_adv += α·sign(grad)`; project to `‖z_adv−z_orig‖_∞ ≤ ε`.
4. **Adaptive step**: if no improvement for `ρ·checkpoint_interval` steps, halve α (down to
   `min_alpha`). (Auto-PGD-style schedule.)
Final: decode `mode='hard'`, recompute success (`argmax ≠ y_true` untargeted), keep best across
restarts by lexicographic rule (success > lower input-L2 > higher objective).

## C.3 Latent C&W (`latent_cw.py:latent_cw_attack`)
Adam over `delta = z_start − z_orig`. Objective = `λ_conf·max(margin+κ,0) + ‖·‖`; per-restart
best tracked by (success ∧ min latent-L2). Multi-restart via GMM/jitter seeds; convergence when
δ shift < threshold. `λ_conf ≤ 0` → pass-through (decode & mask only).

## C.4 Input-space baselines
- **Unconstrained** (`adversarial_attacks.py`, `input_baselines.py`): `TabularFGSM/PGD/CW`
  (torchattacks subclasses with the image `[0,1]` clamp removed) directly on scaled features.
  These get high raw ASR but near-zero **valid** ASR (they violate G1–G8 / protocol / mask).
- **VAE-constrained input** (`constrained_input_baselines.py`): PGD/C&W in input space but every
  step passes through `VAEConstraintProjection.project` — the **same structured-decoder repair as
  the VAE decoder**, applied directly to scaled inputs (unscale → `_structure_raw` → rescale →
  mask → reimpose protocol). This is the "decoder physics without the decoder" ablation.

## C.5 Orchestration (`run_all_models_attack_rerun.py`, canonical)
For each neural classifier (`SimpleMLP, CNNOnly, LSTMOnly, SerialCNNLSTM`) × each source class
(all except Benign), select 100 correctly-classified test samples, run
`latent-pgd, latent-cw, input-pgd, input-cw`, and score:
`ASR_raw`, protocol validity, mask compliance, raw G1–G8 validity, joint validity, IDR
(Mahalanobis in-distribution), and (per `CLAUDE.md`) the intended **True-IDSR =
mean(evasion ∧ joint_valid ∧ in_distribution)**. (`CLAUDE.md` flags the currently-shipped IDSR
definitions as defective — `he_idsr = E·(1−O)` omits validity, and one plot uses `1−ASR`.)

---

# PART D — Validation & metrics

- **Domain validator G1–G8** (`attack/validator.py`, raw space after inverse transform):
  G1 non-negativity, G2 protocol ∈ allowlist & integer, G3 binaries ∈ {0,1}, G4 protocol-indicator
  consistency (indicator⇒protocol), G5 `Min≤AVG≤Max`, G6 `Variance=Std²`, G7 TTL∈[0,255],
  G8 `Number` positive integer. Tolerances `FLOAT_TOL=0.01`.
- **Physics validator P1–P8** (`vae/physics_validator.py`): active set is P2 (`Tot sum≈Number·AVG`),
  P4 (`Std ≤ ½ range`), P5 (singleton ⇒ zero variance); P1/P6/P7/P8 prototyped but excluded
  (clean pass rates too low); P3 omitted (`Tot size`≡`AVG` ambiguity).
- **He-style IDR/IDSR**: Mahalanobis latent distance vs per-class latent distribution; IDSR is
  meant to be ASR gated on in-distribution ∧ validity.

---

# PART E — Literature

## E.1 Zhipeng He — *Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data* (VAE-TabAttack, arXiv 2507.10998)
The **direct source** of this VAE. Key facts (read from the paper):

- **Problem framing.** ℓ_p-norm input-space attacks (FGSM/PGD/C&W) are ill-suited to tabular data:
  one-hot flips cost fixed large ℓ_2 while numeric moves are cheap, and perturbations leave the
  data distribution (become detectable outliers). Solution = perturb in a VAE **latent manifold**.
- **VAE architecture.** *Single global* mixed-input VAE: categorical → embeddings, numerical → FC,
  **shared encoder** → Gaussian `(μ,σ)`; decoder = softmax per categorical + Gaussian (MSE) per
  numerical; **plus an auxiliary classification head `h_ω(z)` on the latent**, trained jointly.
- **Loss.** `L_VAE = recon(MSE_num + CE_cat) + β·KL + α·L_cls(h_ω(z), y)`. The `α·L_cls` term is
  the load-bearing design choice: it forces the latent to be **discriminative / task-aware** so
  latent perturbations efficiently cross the classifier's decision boundary. Their **ablation
  (α=0) degrades both reconstruction and class separability** on all 6 datasets.
- **Attack.** Freeze encoder+decoder, `z=μ(x)` (deterministic), `δ←0`, minimise the **C&W margin**
  `λ·max(Z_y − max_{i≠y}Z_i + κ, 0) + ‖δ‖_2` by Adam; decode `x̃ = p_ψ(z+δ*)`. **Untargeted**,
  κ=0 (smallest manifold-preserving flip). They explicitly *forgo a conditional VAE* because the
  attack is untargeted.
- **Metrics.** ASR; **Outlier Rate O_r** via latent Mahalanobis + χ²₀.₉₅ test;
  **IDSR = ASR·(1−O_r)**. General tabular domains (Adult, Phishing, Pendigits, German,
  Electricity, Covertype) — **not** NIDS, **no** protocol/domain constraints, **no** structured
  decoder, **no** per-class split.

**So what this repo already changed vs He:** per-class split, protocol embedding + nearest-ref
decode, 4 type-specific heads, structured/physics decoder, differentiable domain-constraint loss,
free-bits/β anti-collapse, GMM restarts + adaptive PGD + mask + protocol reimposition + decoder
residual, and the whole G1–G8/physics validity apparatus. **What it removed:** the latent
classifier head `h_ω(z)` (and, by splitting per class, any inter-class latent geometry).

## E.2 NetDiffuser (arXiv 2603.08901) — diffusion-generated adversarial traffic for NIDS
Two components:
1. **Feature categorization (Algorithm 1).** Pearson correlation `r_ij` → distance
   `d_ij = √(2(1−r_ij))` → agglomerative clustering → **discrete** features (weakly dependent →
   safe to perturb, minimal collateral change) vs **relative** features (strongly dependent groups,
   e.g. Flow IAT mean/std/max — perturbing one in isolation is unrealistic). Only discrete features
   are perturbed. *This is exactly the algorithm this repo already vendored into
   `netdiffuser_categorization.py` and consumes to build the 3-tier mask* (full = discrete,
   partial = relative).
2. **Diffusion infusion.** A DDPM injects semantically-consistent perturbation into the discrete
   features at each denoising step → **Natural Adversarial Examples (NAEs)** on the data manifold.
- **Results.** Up to **+29.93% ASR** vs FGSM/PGD/ACG, and it **evades AE detectors** (MANDA,
  Artifact) — reducing AUC by 0.267–0.534. Goal is not just misclassification but *undetectability*
  by manifold/statistical detectors. White-box, flow-level.

**Takeaway for this thesis:** NetDiffuser's *feature-partition idea* is already used for the mask,
but (a) it is used only as a **hard mask**, not built into the **VAE decoder** (which instead
hand-codes rules for *every* continuous feature), and (b) NetDiffuser's *undetectability* objective
(fool the classifier **and** the manifold detector) is exactly the IDSR objective this repo
measures but never optimises.

## E.3 CPAD / constrained tabular & problem-space attacks
"CPAD" is not a standard term in this literature; the closest well-defined bodies of work are:
- **CAA — Constrained Adaptive Attack** (Simonetto et al., arXiv 2406.00775 / 2311.04503).
  Combines a gradient attack (**CAPGD**, an Auto-PGD variant with a *repair/projection* operator
  that enforces feature constraints each step) with a **search-based** attack (**MOEVA**, a
  multi-objective evolutionary search). Key lesson: gradients alone fail on tabular data because of
  **non-differentiable feasibility constraints and discrete features**; a **gradient + search**
  hybrid dominates. Benchmarked on credit / phishing / **botnet (NIDS-adjacent)**.
- **FENCE** (Chernikova & Oprea) and **Sheatsley et al.** — feasible/constrained evasion that
  encodes domain "feature-dependency" constraints (mathematical relations between features) into
  the attack and projects onto the feasible set. Sheatsley notably found network constraints do
  *not* buy much robustness — a useful adversarial framing for the thesis.
- **Problem-space attacks for NIDS** (Apruzzese et al.; GNN problem-space, arXiv 2403.11830).
  Distinguish **feature-space** perturbations (what this repo does) from **problem-space**
  (actually craftable packets/flows via a traffic-control utility). They formalise the
  **flow-level vs packet-level** constraint hierarchy: what an attacker can physically change and
  the mapping back to features.

## E.4 Flow-level vs packet-level constrained attacks (framing)
- **Feature/flow-level** (this repo, He, CAA, NetDiffuser): perturb the extracted flow-feature
  vector, then check domain validity (G1–G8, protocol, ordering, variance=std²). Mutability is
  encoded by the mask.
- **Packet-level / problem-space** (Apruzzese, Teuffenbach): perturb the raw packets/timing;
  features are a *consequence*. Stronger realism claim, weaker attack strength, harder to
  differentiate. This repo lives entirely at flow/feature level — a stated limitation and a place
  to make the *mask/constraints* more defensible (map each mutable feature to a packet-level action
  the attacker actually controls).

---

# PART F — What is done vs what is missing

## F.1 Done (solid)
- Mixed-input VAE with protocol embedding and type-specific heads.
- By-construction domain repair in the decoder + differentiable constraint loss + G1–G8/physics
  validation harness (this repo's genuine contribution over He: **validity is measured, not
  assumed**, and repair is *inside* the generator).
- Competent latent PGD/C&W with adaptive step, GMM restarts, ε-ball, mask, protocol reimposition,
  decoder-residual anchoring.
- Input-space and VAE-constrained-input baselines (the ablation that isolates "decoder physics").
- Per-class latent Mahalanobis IDR gate; the (intended) validity-gated IDSR metric.

## F.2 Missing / weak (opportunities)
1. **No task-aware latent** (deleted He's `h_ω`, plus per-class split) → latent moves don't align
   with class boundaries → **low latent ASR**, over-reliance on decoder residual + large ε. *(Root
   cause of weak results; highest-value fix.)*
2. **No inter-class / conditional latent geometry** → cannot express or exploit a "toward-benign"
   direction, which is the *actual* IDS threat (malicious→benign).
3. **Hand-coded structured decoder** duplicates rules the mined NetDiffuser partition already
   knows; feature **covariance within relative groups is not learned** → reconstructions can be
   marginally valid but jointly implausible (higher outlier rate → lower IDSR).
4. **Attack optimises classifier loss only**; the Mahalanobis/IDR in-distribution test that
   *defines* success is post-hoc. NetDiffuser's whole point (evade the manifold detector too) is
   unoptimised.
5. **No search component** for the discrete heads (protocol frozen; binaries via
   sigmoid/STE) → gradient can't explore binary/categorical flips CAA shows are decisive.
6. **Physics decoder + physics loss are dormant** (`weight=0.0`, decoder flag off) — realism
   headroom left on the table.
7. **Reconstruction likelihood** is Gaussian/Laplace point-estimate → heavy-tailed flow features
   (Rate/IAT/Tot sum) are the dominant source of outliers.
8. **Threat model is flow/feature level only**; no packet-level grounding for the mask.

---

# PART G — Novelty plan (grounded, motivated, ASR-oriented)

Each item states: **the change**, **why (literature)**, **why it raises ASR/IDSR** `[INFERENCE]`,
and **how it differs from He**. N1–N2 are the *VAE architecture* novelty (the claimable core);
N3–N5 are *attack-loop* novelty. They compose.

## N1 — Conditional, adversarially-structured VAE (recover + surpass He's latent head)
**Change.** Replace the 8 per-class β-VAEs with **one class-conditional VAE** (condition on the
8-way label via FiLM/concat on encoder & decoder) whose latent is shaped by two auxiliary terms on
top of the existing ELBO:
- a **surrogate-IDS latent head** `h_ω(z)` trained with cross-entropy (He's term), and
- a **supervised-contrastive loss** (Khosla et al., 2020) on `z`: pull same-class latents together,
  push different classes apart.

Attack still perturbs `z`; conditioning + contrastive geometry give a latent where class regions
are convex-ish and separated, so a bounded latent move reliably crosses a boundary.

**Why (literature).** He shows `α·L_cls` is what makes latent attacks work and that removing it
hurts *both* reconstruction and separability. Supervised contrastive learning is the standard way
to get well-separated, smooth class manifolds. Stutz et al. (on-manifold robustness) show
class-aligned manifolds are where on-manifold adversarials live.

**Why it raises ASR/IDSR `[INFERENCE]`.** Per-class VAEs currently force the attack to fight a
decoder that always reconstructs *the source class*; that is why huge ε (0.8–1.0) and decoder
residual are needed. A conditional, discriminative latent lets small, in-distribution latent moves
change the predicted class → higher ASR at **smaller** perturbation → lower outlier rate →
higher IDSR.

**Differs from He.** He uses a *single unconditional* VAE and *explicitly forgoes* a conditional
one (untargeted). Yours is **conditional + contrastive + NIDS-domain-structured** and supports
**targeted-benign** evasion (the real IDS threat). This is a defensible architectural novelty:
"conditional supervised-contrastive mixed-input VAE for constrained NIDS evasion."

> *Migration note:* this reverses the "8 per-class β-VAEs" locked decision in `CLAUDE.md`. If that
> decision must hold, the fallback is **N1′**: keep per-class VAEs but add a **shared cross-class
> latent alignment** — train the 8 encoders into one **shared latent coordinate frame** with an
> inter-class contrastive/anchor loss on a held-out mixed batch, plus a small shared `h_ω(z)`
> surrogate head. Weaker than N1 but preserves the per-class decoders.

## N2 — Relative-feature-group decoder (learned covariance, not hand-coded rules)
**Change.** Stop hand-coding a rule for every continuous feature. Use the **NetDiffuser
discrete/relative partition** (already computed) as decoder structure:
- **Discrete** features → independent heads (as now).
- **Relative** groups → a **joint head** that models within-group covariance: a **low-rank +
  diagonal Gaussian** (`Σ = LLᵀ + D`) or a small **autoregressive/coupling** head per group, so a
  reconstructed group (e.g. {IAT mean/std/max}, {Min/AVG/Max/Std/Variance}) is **internally
  consistent by construction** rather than by six separate softplus tricks.

**Why (literature).** NetDiffuser's categorization exists precisely because relative features move
together; He et al.'s seven imperceptibility properties include **feature interdependencies**. CAA
enforces feature relations as constraints. Modeling covariance is the generative way to honour
those relations.

**Why it raises ASR/IDSR `[INFERENCE]`.** Lower **outlier rate** (jointly-plausible groups) →
higher `(1−O_r)` → higher IDSR at equal ASR; and the attacker gets **clean independent knobs**
(discrete features) that don't drag correlated features off-manifold, so more perturbation budget
converts to misclassification. Also removes brittle hand-coded physics duplicated across
`model.py` and `constrained_input_baselines.py`.

**Differs from He.** He's decoder is plain per-feature Gaussian/softmax with **no** covariance and
**no** feature-group structure. A **correlation-partition-structured decoder** is novel and
NIDS-specific.

## N3 — Manifold-in-the-loop attack objective (optimise IDSR directly)
**Change.** Add a **differentiable in-distribution penalty** to the latent PGD/C&W objective:
`L = L_classifier + γ·MahalanobisÂ²(z_adv | class)` (Mahalanobis on the *current* latent, using the
per-class μ,Σ already fit for IDR; differentiable in `z`). Optionally add a **GMM log-density**
term (the GMM is already fit). Keep the ε-ball as a hard backstop.

**Why (literature).** He's IDSR = ASR·(1−O_r) and NetDiffuser's goal is evading the **manifold
detector** (MANDA/Artifact) — both are exactly "stay in-distribution while misclassifying." The
repo *measures* Mahalanobis but never *optimises* it.

**Why it raises ASR/IDSR `[INFERENCE]`.** Directly trades a little raw ASR for a large drop in
outlier rate → **higher IDSR**, the metric the thesis reports and the He paper introduces. Makes
adversarials survive the realism gate instead of being discarded post-hoc.

**Differs from He.** He's attack objective is `margin + ‖δ‖`. Adding an explicit **manifold-density
regulariser inside the optimiser** (not just as a metric) is a novel attack formulation and
directly answers "does your attack beat a manifold detector?"

## N4 — Fast differentiable discrete heads (replaces CAA's slow search)
**Conceded: a gradient+search hybrid is *not* novel (CAA already did it) and is *slow*** — MOEVA is
population-based evolutionary search, the opposite of a speed claim. So drop the search and instead
make the discrete heads **differentiable and fast**: relax the protocol softmax and the 11
independent-binary logits with the **Gumbel-Softmax / Concrete** trick (Jang et al. 2017; Maddison
et al. 2017) under a temperature anneal, so the attack optimises discrete flips **by gradient, one
pass per step** — no search, no population.

**Change.** In `decode_to_39`'s attack path, replace the sigmoid/STE binary path (and the frozen
protocol, *only if* the threat model permits protocol change) with Gumbel-Softmax samples; anneal
τ→0 across attack steps so the relaxation converges to a valid one-hot / {0,1} choice.

**Why (literature).** CAA (CAPGD+MOEVA) shows discrete/constrained features are *why* gradient-only
tabular attacks under-perform — but pays for it with slow search. Gumbel-Softmax is the standard way
to get **differentiable discrete choices at gradient speed**, and is exactly what the current
straight-through binary path crudely approximates. This keeps the *benefit* CAA identified
(exploring discrete flips) without the *cost* (search), preserving the speed claim.

**Why it raises ASR `[INFERENCE]`.** Recovers misclassifications reachable only by a discrete flip
that the crude STE misses; the temperature anneal gives a lower-bias gradient than STE.

**Differs from He / CAA.** He has no discrete-flip mechanism; CAA uses slow evolutionary search. A
**Gumbel-Softmax discrete-head latent attack** is both faster than CAA and a novel formulation in
the VAE-latent NIDS setting.

## N5 — Learned "toward-benign" latent direction (targeted-benign evasion)
**Change.** Using N1's conditional/contrastive latent, learn (or estimate from class centroids /
an LDA direction in latent space) a **per-source-class direction `d_c` toward the Benign region**,
and initialise / bias the attack along `d_c` (as an extra restart seed alongside GMM).

**Why (literature).** The operative IDS threat is malicious→benign (false negative), the exact case
NetDiffuser and problem-space papers target. He's attack is untargeted (any wrong class).

**Why it raises ASR `[INFERENCE]`.** A benign-oriented warm start needs far less perturbation to
reach the benign side of the boundary than a random/GMM start → higher targeted-benign success at
lower ε → higher IDSR.

**Differs from He.** He explicitly avoids conditioning/targeting. Targeted-benign latent traversal
on a class-conditional manifold is novel and threat-model-aligned.

## N6 (supporting) — Activate physics + heavy-tail likelihood
Turn on the dormant **physics decoder/loss** (P2/P4/P5) and replace the continuous head's
point-Gaussian with a **2–3 component mixture density** (or keep Laplace) for the heavy-tailed
features. Motivation: outlier rate is dominated by Rate/IAT/Tot-sum tails; better tail modeling →
lower `O_r` → higher IDSR. Grounded in He's own finding that VAE-attack quality is bounded by
**reconstruction quality**, and in tabular-generative practice.

## N7 — Amortised one-shot perturbation generator (headline speed novelty)
**Change.** Instead of optimising `δ` per sample, train a small **conditional generator**
`g_θ(z, c_src→benign) → δ` (an advGAN-style amortised attacker; Xiao et al. 2018, *Generating
Adversarial Examples with Adversarial Networks*; cf. Zhao et al. 2018, *Generating Natural
Adversarial Examples*). At inference the attack is **one forward pass**: encode → `g_θ` → decode →
mask/reimpose. Train `g_θ` against the (surrogate) IDS with the same objective family as the
iterative attack (margin + N3 manifold-density penalty + ε bound) over the whole training set.

**Why (literature).** Amortised adversarial generators trade a one-time training cost for **O(1)
inference** — orders of magnitude faster than per-sample iterative, search, or diffusion methods —
the established route to real-time adversarial generation.

**Why it raises ASR *and* speed `[INFERENCE]`.** (a) Speed: single pass, no per-sample loop.
(b) ASR: `g_θ` learns a *global* attack policy on the conditional/contrastive manifold (N1) and its
output is a near-optimal **warm start**; a **1–3 step gradient refinement** on top recovers
per-sample slack — "amortised speed with iterative-quality ASR". This learned-initialisation +
few-step-refine hybrid is itself a clean, defensible contribution.

**Differs from He / NetDiffuser.** Both are **per-sample iterative** (C&W / reverse diffusion). An
**amortised conditional latent perturbation generator for constrained NIDS evasion** is
architecturally distinct and is the basis of the speed claim.

## Speed as a first-class claim (why this beats diffusion and search) — Part S
He's attack is ~300 Adam iterations per sample; NetDiffuser runs a full **T-step reverse diffusion
chain** per sample (T typically 50–1000 denoiser evals); CAA runs **CAPGD + MOEVA** (an evolutionary
population × generations of constraint-checked evaluations). All three are expensive. A VAE latent
attack's per-step cost is one pass through a **shallow 2-layer decoder** (`[64,128]`) plus the
classifier, over a 16-dim latent — no generative Markov chain, no search population.

| Method | Per-sample cost (order) | Why |
|---|---|---|
| NetDiffuser (diffusion) | `T × denoiser eval` | full reverse chain per sample |
| CAA (CAPGD+MOEVA) | `iters + (pop × gens × constraint eval)` | gradient **plus** evolutionary search |
| He C&W (latent) | `~300 × (decode + classifier)` | per-sample iterative optimisation |
| **This VAE, iterative (N3/N4)** | `k × (shallow decode + classifier)`, small `k` | on-manifold ⇒ few steps; no chain/search |
| **This VAE, amortised (N7)** | **`1 × (generator + decode + classifier)`** | single forward pass, no per-sample loop |

**Claim (defensible):** *"Our constrained on-manifold attack reaches comparable or higher
valid-ASR/IDSR than diffusion- and search-based constrained attacks at a fraction of the per-sample
generation cost, because perturbation is a single shallow-decoder pass rather than a reverse
diffusion chain or an evolutionary search."* Quantify it: report **wall-clock and forward-eval count
per adversarial sample** alongside ASR/IDSR for {input-PGD, He-C&W-latent, NetDiffuser-style
diffusion if reproduced, CAA if reproduced, ours-iterative, ours-amortised}. Speed then becomes a
measured axis of the contribution, not an assertion.

## G.1 Suggested ablation ladder (for the thesis defense)
1. Baseline = current per-class VAE + latent PGD/C&W (reproduce He-style IDSR).
2. +N1 (conditional + `h_ω` + contrastive) — expect the largest ASR jump.
3. +N2 (relative-group decoder) — expect O_r ↓, IDSR ↑.
4. +N3 (manifold-in-loop) — expect IDSR ↑ at small ASR cost; report detector-evasion (MANDA-style).
5. +N4 (Gumbel-Softmax discrete heads) — expect ASR ↑ from discrete flips, **no speed loss**.
6. +N5 (toward-benign) / +N6 (physics+MDN) — targeted-benign IDSR and realism.
7. +N7 (amortised generator, + 1–3 step refine) — expect ~equal ASR/IDSR at **O(1) inference**.
Report **per-sample wall-clock + forward-eval count** on every rung so the speed claim is quantified.
Report each as ΔASR, ΔO_r, ΔIDSR vs the previous rung so novelty is *causally attributed*.

## G.2 One-paragraph novelty claim (defensible)
> We extend He's unconditional mixed-input tabular VAE into a **class-conditional,
> supervised-contrastive, NIDS-domain-structured VAE**: (i) a conditional encoder/decoder with a
> surrogate-IDS latent head and contrastive latent geometry that restores and surpasses the
> discriminative latent He shows is essential, while enabling *targeted-benign* evasion; (ii) a
> **relative-feature-group decoder** that learns within-group covariance from the NetDiffuser
> correlation partition instead of hand-coding per-feature rules; and (iii) a **manifold-in-the-loop
> latent attack** with **fast differentiable (Gumbel-Softmax) discrete heads** and an **amortised
> one-shot perturbation generator**, optimising in-distribution success (Mahalanobis/GMM) while
> resolving discrete-feature flips *without* the evolutionary search CAA relies on. This is
> architecturally and objectively distinct from VAE-TabAttack, is grounded in NetDiffuser (feature
> partition, NAE/undetectability), CAA (the discrete-feature problem), and amortised adversarial
> generation (advGAN), and is expected to raise valid-ASR/IDSR **at a fraction of the per-sample
> generation cost** of diffusion- and search-based constrained attacks.

---

# PART E2 — Second-pass literature (things the first pass missed)

A broader search turned up a substantial **constrained-tabular / constrained-NIDS attack family**
that the first draft under-covered. It does not overturn the plan, but it (a) shows the space is
crowded, so the novelty must be positioned precisely, and (b) hands us stronger, better-grounded
mechanisms than the hand-coded structured decoder.

## E2.1 Constrained tabular attack family (beyond CAA)
- **CaFA — Cost-aware, Feasible Attacks with Database Constraints** (Nyffenegger et al., arXiv
  2501.10013, 2025). **TabPGD** in feature space, then **projection onto mined *denial
  constraints*** (database integrity constraints automatically discovered from data) so examples
  are realizable; achieves higher *feasible* success while perturbing fewer features. **This is the
  most important thing the first pass missed:** it *mines* feature-relationship constraints instead
  of hand-coding them — exactly the `ConstraintMiner` target already written into `CLAUDE.md`'s
  refactor principle — and enforces them by **projection**, not soft penalty.
- **Kireev, Kulynych, Troncoso — Cost/Utility-Aware Robustness** (NDSS 2024 / ICML 2023, arXiv
  2208.13058). Argues tabular threat models should not be ℓ_p-imperceptibility but **per-feature
  cost** and attacker **utility** (profit = gain − cost). Mutability is a special case (cost=∞).
- **Mathov et al.** — formalise **mutability, type, boundary, and distribution** constraints for
  heterogeneous tabular data. Mutability constraints appear in 33/53 tabular-attack studies
  (systematic review arXiv 2506.15506).
- **TabularBench** (NeurIPS 2024, arXiv) and **TabAttackBench** (He et al.) — robustness/attack
  benchmarks; TabAttackBench's four imperceptibility axes = Proximity, Sparsity, Deviation,
  Sensitivity (the metric family this thesis should report).

## E2.2 Constraint taxonomy & the "constraints don't guarantee robustness" result
- **SoK: Realistic Adversarial Attacks and Defenses for NIDS** (Apruzzese et al., C&S 2023, arXiv
  2308.06819). Canonical **problem-space vs feature-space** split; the **inverse feature-mapping
  problem** (a valid feature vector may not correspond to any craftable flow); constraints =
  {available transformations, preserved semantics, absent artifacts, plausibility}. Cite this for
  the thesis threat-model section (which `IMPROVEMENTS.md` flags as missing).
- **Sheatsley et al. — *Adversarial Examples in Constrained Domains* (JCS 2022) & *On the
  Robustness of Domain Constraints* (CCS 2021).** Constraints = feature *relationships* (e.g. TCP
  flags only valid on TCP flows — your G4). Key finding to **not overclaim against**: domain
  constraints do **not** confer much robustness; ≥95% misclassification is reachable with as few as
  5 mutable features. Adaptive-JSMA obeys constraints.
- **Pierazzi et al. — Intriguing Properties of AML in the Problem Space** (IEEE S&P 2020) and
  **A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space** (arXiv
  2112.01156). Formal constrained-feature-space framing.
- **Constrained Network Adversarial Attacks: Validity, Robustness, Transferability** (2025) —
  recent, directly on-topic; worth citing as the closest contemporaneous NIDS work.

## E2.3 Generative constrained NIDS attacks — the GAN lineage (must differentiate)
- **IDSGAN** (Lin et al., 2018, arXiv 1809.02077): generator turns malicious records adversarial;
  **restricted-modification mechanism** freezes functional features (= your protocol/mask freeze);
  black-box vs a surrogate discriminator.
- **attackGAN** (2021): new loss to preserve traffic **functionality** while evading a black-box IDS.
- **Attack-GAN** (SeqGAN, packet-level, arXiv 2103.04794): generates packets under packet-format
  constraints (e.g. Windows TTL ∈ {32,128}); a concrete **packet-level** example.
- **CWVAEGAN**, VAE-GAN hybrids: mostly for **class-imbalance augmentation**, not attacks.
  → **Differentiation for the thesis:** the whole GAN-NIDS line is (i) mostly **black-box**, (ii)
  **GAN** (no explicit density → cannot report He-style IDSR/Mahalanobis directly), (iii) on older
  data (NSL-KDD/CICIDS2017). Your VAE gives an **explicit density manifold** (IDSR-native),
  **white-box on-manifold** perturbation, **per-class** structure, on **CICIoT2023**, with a
  **speed** story. State this explicitly — "generative constrained NIDS attack" alone is not novel;
  the *VAE-latent + density-native validity + speed* combination is.

## E2.4 Constraint-enforcement machinery worth adopting
- **Differentiable projection layers** (Chen et al., arXiv 2111.10785; energy: arXiv 2105.08881).
  Enforce **hard** constraints in the forward pass while passing gradients — a principled
  replacement for both the hand-coded softplus repairs *and* the soft constraint-penalty loss.
- **Projected / constrained diffusion** (CTDF *Constrained Tabular Diffusion for Finance*, arXiv
  2606.28674; *Constraint-Guided Prediction Refinement*, arXiv 2506.12911). Diffusion competitors
  that project onto feasible sets each denoising step — **slow**, so a good speed-comparison target.
- **AE detectors** (MANDA manifold test; Artifact KDE+BNN uncertainty; NIDS-DA autoencoder;
  RADNN density) — the detectors N3 must beat; motivates the manifold-in-the-loop objective.

# PART G2 — Refined & added novelty (incorporating the second pass)

## G2.1 Upgrade N2 → differentiable projection onto *mined* domain constraints
**Change.** Replace the hand-coded structured decoder rules (and reduce reliance on the soft
constraint loss) with a **differentiable projection layer** as the decoder's final stage that
projects the raw output onto the feasible set defined by **mined constraints** (denial-constraint /
relationship mining à la CaFA, plus the NetDiffuser relative-group covariance from N2). Hard
validity by construction, gradients still flow.
**Why (literature).** CaFA (mine + project) and differentiable-projection-layer work show this is
both realizable and trainable; it also fulfils the repo's own `ConstraintMiner→MinedValidator`
refactor principle (constraints *mined on train*, not hand-authored).
**Why it raises ASR/IDSR `[INFERENCE]`.** Guarantees G1–G8/relationship validity → the validity
gate no longer discards examples; and mined constraints are tighter/more complete than the 8
hand-coded rules → lower outlier rate → higher IDSR.
**Differs from He / CaFA.** He has no constraints; CaFA projects in **input space** as a post-hoc
step on a general tabular classifier. Doing the projection **inside the VAE decoder**, over **mined
NIDS constraints**, feeding a **latent** attack, is the novel combination.

## G2.2 New — N8: cost/utility-weighted latent perturbation (replace the binary mask)
**Change.** Replace (or augment) the 3-tier mutable/partial/frozen mask with a **per-feature cost
vector** `c_f` (Kireev/Mathov): the attack minimises `L_classifier + Σ_f c_f·|Δ_f|` (or a
cost-weighted latent penalty), with `c_f=∞` recovering immutability. Derive `c_f` from how hard a
feature is to change in a real flow (e.g. IAT/rate cheap; protocol/service expensive).
**Why (literature).** Kireev et al. (NDSS 2024) show cost/utility threat models are the right
abstraction for tabular; Mathov formalises boundary/type/distribution costs. This also fills the
**missing formal threat model** (`IMPROVEMENTS.md`).
**Why it raises ASR/realism `[INFERENCE]`.** Budget flows to cheap, realistic features → higher
*realistic* ASR at lower attacker cost, and a defensible economic threat model instead of a
hand-listed mask.
**Differs from He.** He uses ℓ_2 only; no cost model, no mutability. A cost-weighted **latent**
attack on a NIDS VAE is novel.

## G2.3 Positioning correction (so novelty survives scrutiny)
"Constrained adversarial NIDS generation" is a crowded field (CAA, CaFA, NetDiffuser, IDSGAN,
attackGAN, Sheatsley). The **defensible, non-overlapping** contribution is the *conjunction*:
> a **conditional, supervised-contrastive, mixed-input VAE** for **IoT (CICIoT2023)** whose decoder
> enforces **mined** domain constraints by **differentiable projection**, attacked in **latent
> space** with a **manifold-density-in-the-loop, cost-weighted** objective and an **amortised
> one-shot generator** — evaluated by **validity-gated IDSR** and **per-sample cost/latency**.
No single prior work occupies that intersection: He (no constraints/NIDS), NetDiffuser (diffusion,
slow, no density), CaFA/CAA (input-space, non-VAE, finance/phishing/botnet), IDSGAN/attackGAN
(black-box GAN, no density, older data). Also **do not claim constraints add robustness** — cite
Sheatsley and frame the contribution as *realistic, efficient, on-manifold evasion*, not defense.

## G2.4 Honest caveats
- **Projection over non-convex constraint sets** (G5 ordering ∧ G6 variance=std² ∧ mined rules) can
  be non-convex; the differentiable-projection literature notes this. Fallback: keep the
  by-construction softplus repairs for the non-convex subset, project only the convex/box part.
- **CaFA already does mine+project** — so lead with *VAE-latent + mined-projection-decoder*, not
  "mining constraints" per se, to avoid an incremental-over-CaFA critique.

## Appendix — File map used
- VAE: `src/vae/{model,losses,config,schema,physics_validator}.py`.
- Attack: `src/attack/{latent_infra,latent_pgd,latent_cw,latent_gmm,latent_restarts,
  constrained_input_baselines,adversarial_attacks,validator,run_all_models_attack_rerun}.py`.
- Preprocessing partition/mask: `netdiffuser_categorization.json`, `near_zero_iqr_features.json`,
  `feature_groups_and_netdiffuser_audit.md`.
- Literature: He arXiv 2507.10998 (VAE-TabAttack); NetDiffuser arXiv 2603.08901;
  CAA arXiv 2406.00775 / 2311.04503; problem-space NIDS arXiv 2403.11830;
  advGAN (Xiao et al. 2018, arXiv 1801.02610); Natural AEs (Zhao et al. 2018, arXiv 1710.11342);
  Gumbel-Softmax (Jang et al. 2017, arXiv 1611.01144; Maddison et al. 2017, arXiv 1611.00712);
  supervised contrastive (Khosla et al. 2020, arXiv 2004.11362).
- Second-pass literature: CaFA arXiv 2501.10013; Kireev cost/utility arXiv 2208.13058 (NDSS 2024);
  systematic review arXiv 2506.15506; TabularBench (NeurIPS 2024); SoK realistic NIDS arXiv
  2308.06819; Sheatsley constrained domains arXiv 2011.01183 / domain-constraint robustness arXiv
  2105.08619; Pierazzi problem-space IEEE S&P 2020; unified constrained feature space arXiv
  2112.01156; IDSGAN arXiv 1809.02077; Attack-GAN (packet-level SeqGAN) arXiv 2103.04794;
  differentiable projection arXiv 2111.10785 / 2105.08881; constrained tabular diffusion arXiv
  2606.28674 / 2506.12911; MANDA / Artifact / NIDS-DA AE detectors.
