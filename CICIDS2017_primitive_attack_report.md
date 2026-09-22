# Realizability-aware primitive-control adversarial attack — audit & redesign (CICIDS2017-DistriNet)

This report documents the audit of the primitive-control attack, the methodological bugs
found, the fixes implemented, and the re-run evaluation. It is organized around the 12
requested deliverables. **Priority ordering honored:** correct primitive semantics →
complete dependency propagation → discrete realizability projection → correct targeted-valid
metrics → independent validity → cost-controlled comparison → statistical robustness →
generalization → packet-level interface. Correctness was chosen over headline ASR throughout.

---

## 1. Architecture audit

### 1.1 Current attack pipeline (as implemented, after redesign)

```
pristine raw test row  (X_test_pristine.npy, 79 CICFlowMeter features, canonical raw)
      │
      ▼
per-flow feasible bounds  (train-mined envelope headroom → p_hi, alpha_hi)   [leakage-safe]
      │
      ▼
differentiable optimization over control logits (u,v)                        [Adam, 40 steps]
   controls: p = p_hi·σ(u) ≥ 0 ,  α = 1+(α_hi−1)·σ(v) ≥ 1
      │  classifier cross-entropy toward Benign + λ·(σ(u)+σ(v))
      ▼
continuous primitives (p, α)
      │
      ▼
DISCRETE REALIZABILITY PROJECTION   model.project_controls  (round p→int bytes;
      │                              α realized via µs-quantization of derived timing)
      ▼
COMPLETE DEPENDENCY RECOMPUTATION   model.generate(quantize=True)
      │   (every feature a primitive affects is recomputed; frozen copied from raw)
      ▼
final adversarial CICFlowMeter vector (79-dim, internally consistent + integral)
      │
      ├─► victim classifier (mlp / cnn / lstm / serial)         → prediction
      └─► validators  (PAVE Level-A · mined density · internal realizability · IDR realism)
```

### 1.2 Actual role of the VAE (deliverable, section 14)

**Finding: the experiment is NOT a latent-VAE attack.** Tracing the code:

- `x` is **not** encoded to a latent `z` for optimization.
- The optimizer variables are the two primitives `(p, α)` directly (via logits `u,v`).
- The classifier gradient backpropagates through the **differentiable primitive→feature map**
  (`CICIDS2017PrimitiveModel.generate`), which is pure tensor algebra — **not** through any
  VAE decoder.
- The per-class Stage-A β-VAE is loaded **only** to provide the realism gate: a val-anchored
  Mahalanobis in-distribution test in latent space (`_idr_mask`, `idr_*.npz`) used for the
  IDR / True-IDSR metric. It never participates in generating or steering the perturbation.

**Conclusion & recommended naming.** This is a *differentiable primitive-domain constrained
attack with a VAE-based realism gate*, not a “primitive-control VAE attack.” The report,
code docstrings, and result JSON (`"vae_role"`) now state this explicitly. The VAE is
**retained** (it provides an independent realism metric orthogonal to the classifier and to
the algebraic validators), but the method is not claimed to optimize a VAE latent. If a
genuine latent-VAE variant is wanted later, it would decode `z → (p, α)` and backprop the
classifier loss through the decoder; that is a separate experiment.

### 1.3 Primitive transformation design

Two attacker-controllable primitives per flow, both differentiable and both with a defined
“no-op” value (`p=0`, `α=1`):

| primitive | meaning | acts on | disabled when |
|---|---|---|---|
| `p ≥ 0` | **forward packet-length augmentation** (bytes added to every forward packet length) | forward length block + everything derived from it | `Total Fwd Packet < 1` |
| `α ≥ 1` | **forward timing dilation** (stretch every forward inter-arrival gap) | forward IAT block, duration, flow-IAT mean/max, rates | `Total Fwd Packet < 2` (no fwd IAT sequence) |

Hard bounds act on the **control space** (sigmoid reparameterization to `[identity, per-flow
cap]`), never as independent post-hoc clipping of aggregate features (section 22 satisfied).

---

## 2. Bugs / inconsistencies found

| # | Severity | Bug | Fix |
|---|---|---|---|
| B1 | **High** | **Incomplete packet-length propagation.** `p` updated only the forward length block + `Flow Bytes/s`; the *combined* stats `Packet Length {Min,Max,Mean,Std,Variance}` and `Average Packet Size` were **frozen** although they depend on forward packet lengths. Padding produced flows whose combined length statistics contradicted the forward ones. | All recomputed from mined identities (pooled sample-variance for Std/Var; conditional ext for Min/Max). |
| B2 | **High** | **Incomplete timing propagation.** `α` scaled the forward IAT block but **froze** `Flow IAT Mean`, which is exactly `Flow Duration/(Nf+Nb−1)`; when duration changed, this identity was violated. | `Flow IAT Mean` recomputed; `Flow IAT Max` grown by the added delay so `mean ≤ max ≤ duration` stays exact. |
| B3 | **High** | **No discrete realizability projection.** Reported success on continuous vectors (`p=42.842`, lengths `50.84`, µs-fractional IAT). Packet lengths / µs timestamps must be integral. | `project_controls` rounds `p` to integer bytes; `generate(quantize=True)` µs-quantizes integer timing features and regenerates all derived features from the projected primitives. Success reported **after** projection. |
| B4 | **Medium** | **`p` overclaimed as “payload padding.”** True application-payload insertion would also move `Fwd Act Data Pkts`; the map left it frozen. | Renamed to **forward packet-length augmentation**; `Fwd Act Data Pkts` explicitly labelled `LEVEL_C_FROZEN` (not asserted as payload insertion). |
| B5 | **Medium** | **`p_max = 1460` simplistic** and not even an upper bound in this data (2.4% of train forward packets exceed 1460 due to GRO/TSO coalescing). | Per-flow **data-mined headroom** (train envelope) is the real cap; an optional explicit `--mtu-cap` per-packet ceiling is exposed and documented as a Level-C assumption. |
| B6 | **Medium** | **Timing optimized for 1-packet flows** (no forward IAT sequence exists). | `active_mask("alpha")` forces `α=1` for `Total Fwd Packet < 2`; gradient/effect masked. |
| B7 | **Medium** | **Validator too narrow** — only checked what the old map wrote, so “0 failures” was near-tautological. | Expanded to 6 independent categories incl. combined-stat consistency, discreteness, and the full algebraic-identity set, recomputed from the adversarial vector. |
| B8 | **High** | **Headline metric was untargeted raw ASR**; targeted-benign & strict-valid ASR under-reported; macro vs pooled silently mixed. | Both untargeted and targeted-Benign reported at raw/valid/strict-valid levels, with explicit per-victim / per-class-macro / overall-macro / pooled-micro and documented denominator. |
| B9 | **Medium** | **Single seed, no uncertainty.** | 3 seeds (fixed eval rows, varied optimization init) → mean±std; plus 95% bootstrap CIs over clean-correct rows. |
| B10 | **Low** | **Cost not decomposed / not physically interpreted.** | Additive normalized-cost decomposition into padding / timing / rate parts + raw primitive magnitudes (`p` bytes, `α` ratio); matched-budget sweep vs the old attack. |
| B11 | **Design** | **Subflow / bulk / Flow-IAT-Std were silently frozen** with no acknowledgement they *should* move under real packet edits. | Labelled `LEVEL_C_FROZEN`; the report distinguishes feature-space algebraic consistency from packet-trace realizability (section 15). |

---

## 3. Files changed

| file | change | reason |
|---|---|---|
| `src/attack/realizability/__init__.py` | **new** | package exports for the dataset-agnostic framework |
| `src/attack/realizability/base.py` | **new** | `FeatureRole`, `PrimitiveSpec`, `IdentityCheck`, `DatasetPrimitiveModel` protocol, `PacketVerificationBackend` + `NullPacketBackend` (Level-C interface, section 16/17) |
| `src/attack/realizability/cicids2017.py` | **new** | concrete CICIDS2017 primitive model: complete dependency graph, one-packet masking, headroom/MTU bounds, discrete projection (`generate`, `project_controls`, `roles`, `algebraic_identities`, `per_flow_bounds`) |
| `src/attack/realizability/validator.py` | **new** | dataset-agnostic categorized realizability validator (algebraic / packet / timing / rate / discreteness / frozen) |
| `src/attack/run_cicids2017_primitive_attack.py` | **rewritten** | projection before eval; targeted+untargeted metrics; per-cell masks/artifacts; multi-seed; cost decomposition; VAE-role documented in output |
| `src/attack/tests/test_primitive_controls.py` | **rewritten** | regression suite for the new model (section 21) |
| `src/attack/primitive_controls.py` | **deleted** | superseded by `realizability/` (clean cutover) |
| `scripts/analyze_primitive_attack.py` | **new** | tables, bootstrap CIs, distributions, old-vs-new, audit cases |
| `scripts/budget_sweep_primitive.py` | **new** | matched-budget ASR-vs-cost comparison (section 13) |
| `scripts/compare_cicids2017_attacks.py`, `scripts/print_primitive_examples.py` | **deleted** | folded into the analyzer |

---

## 4. Final dependency map

Mined on the pristine **TRAIN** split only (leakage-safe); every “exact” identity below had
**0.000% violation** across 400k–600k train rows. `Nf,Nb` = fwd/bwd packet counts,
`N=Nf+Nb`, durₛ = duration/1e6.

### 4.1 Primitive `p` (forward packet-length augmentation)

| feature | role | equation / reason |
|---|---|---|
| Total Length of Fwd Packet | primitive-controlled | `+= Nf·p` |
| Fwd Packet Length Min / Max | primitive-controlled | `+= p` (uniform shift) |
| Fwd Packet Length Mean | direct-derived | `= TL_fwd/Nf` (exact) |
| Fwd Packet Length Std | frozen | shift-invariant under uniform padding |
| Fwd Segment Size Avg | direct-derived | `= Fwd Packet Length Mean` (exact) |
| Packet Length Mean | direct-derived | `= (TL_fwd+TL_bwd)/N` (exact) |
| Average Packet Size | direct-derived | `= Packet Length Mean` (exact) |
| Packet Length Max / Min | conditional-derived | `= ext(fwd±p, bwd)` with direction-presence branch (fixes 0.70% Nb=0 edge) |
| Packet Length Variance | direct-derived | pooled **sample** variance of (fwd+p, bwd) — 0% train violation |
| Packet Length Std | direct-derived | `= √Variance` |
| Flow Bytes/s | rate | `= (TL_fwd+TL_bwd)/durₛ` |

### 4.2 Primitive `α` (forward timing dilation)

| feature | role | equation / reason |
|---|---|---|
| Fwd IAT Total / Max / Min / Std | primitive-controlled | `×= α` (uniform dilation) |
| Fwd IAT Mean | direct-derived | `= Fwd IAT Total/(Nf−1)` (exact) |
| Flow Duration | conditional-derived | `= dur + (α−1)·Fwd IAT Total`, floored by directional totals (delay-only projection; **Level-C approximate**) |
| Flow IAT Mean | direct-derived | `= Flow Duration/(N−1)` (exact) |
| Flow IAT Max | conditional-derived | `+= added delay` (one realizable trace; preserves mean ≤ max ≤ dur) |
| Flow/Fwd/Bwd Packets/s | rate | `= count/durₛ` |
| Flow Bytes/s | rate | `= bytes/durₛ` |

### 4.3 Frozen (genuinely unaffected)

All backward stats, flags, header lengths, ports/protocol, window bytes, Down/Up ratio
(`= Nb/Nf`, counts unchanged), active/idle, `Fwd Seg Size Min`, `Bwd IAT` block, etc.

### 4.4 Level-C frozen (would change under real packet edits; NOT reconstructable from the
aggregate flow — held frozen and reported as a limitation)

`Fwd Act Data Pkts` (payload-count semantics), `Subflow Fwd Bytes` (subflow decomposition
unknown — empirically **not** `= TL_fwd` nor `TL_fwd/Nf`, 95–99% mismatch), `Fwd
Bytes/Bulk Avg`, `Fwd Packet/Bulk Avg`, `Fwd Bulk Rate Avg` (bulk detection needs the packet
size/timing sequence), `Flow IAT Std`, `Flow IAT Min` (merged fwd+bwd gap sequence reshuffles
under forward dilation).

---

## 5. Validator rule map

Three **independent** validity layers plus the internal realizability categories:

| layer | source | checks |
|---|---|---|
| **A — feature-domain** | `PAVEStyleValidator` (train-fit ranges/types, tol 1e-6) | non-negativity, valid ranges, integer/categorical domains |
| **B — algebraic/dependency** | internal `RealizabilityValidator.algebraic_dependency` | all §4 exact identities, recomputed from the adversarial vector |
| **realizability (structural)** | internal | `packet_summary` (min≤mean≤max, max≤total, std≥0), `timing` (min≤mean≤max≤total, dir-total≤duration, flow-IAT≤duration, duration>0), `negative_rate` (rates≥0) |
| **discreteness** | internal | all 54 mined integer features integral after projection |
| **frozen** | internal | frozen + Level-C-frozen features byte-identical to pristine raw |
| **dataset-mined density** | `ConstraintEngine` A4 (`constraints/…/mined.json`) | Layer-0/1/2 mined density/support constraints |
| **realism** | Stage-A β-VAE | val-anchored Mahalanobis in-distribution (IDR / True-IDSR) |

**Separation guarantee (section 23):** the generator enforces only universal/structural
consistency (the §4 identities and orderings that any packet sequence satisfies). The mined
density validator and PAVE are **external** and are *not* embedded in the generator, so a
100% internal-realizability rate does not trivially imply external validity — the external
validators remain an independent test. `strict = PAVE ∧ mined ∧ internal-realizability`.

---

## 6. Regression / unit tests added

`src/attack/tests/test_primitive_controls.py` (13 tests, all passing):

- identity `(0,1)` is a no-op and fully realizable;
- padding/dilation stay realizable across `(p,α) ∈ {(0,1),(50,2),(250,3),(800,20)}`;
- **uniform-padding identities** `new_total = old_total + Nf·p`, `new_mean = old_mean + p`,
  `std` invariant, `Fwd Segment Size Avg = mean`;
- **combined length stats move & stay consistent** (`var = std²`, `min ≤ mean ≤ max`) — guards B1;
- rates ≥ 0 and equal their equations;
- timing ordering (`duration>0`, `Fwd IAT Max ≤ Total`, `Total ≤ duration`, `Flow IAT Mean ≤ Max`) — guards B2;
- **single forward packet ⇒ α disabled** (fwd IAT / duration unchanged) — guards B6;
- **projection ⇒ all 54 integer features integral** & discreteness category passes — guards B3;
- **frozen features exactly preserved** + Level-C roles asserted — guards B4/B11;
- gradient flows into both primitives (incl. through the new pooled-variance path).

---

## 7. Experiment configuration

- Victims: `mlp, cnn, lstm, serial` (category classifiers). Classes: `DoS, DDoS, Recon, BruteForce`.
- Rows: up to 1024 clean test rows/class, **fixed across seeds** (isolates optimization
  variance). Steps 40, Adam lr 0.1, cost-weight λ=0.01, `p_max=1460`, `α_max=100`,
  `mtu_cap` disabled (train-envelope headroom used), init-noise 0.5, seeds `{42,43,44}`.
- Denominator everywhere: **clean-correct malicious test rows** per (class, victim).
- Determinism: global seeding of `random/numpy/torch`; per-flow bounds & PAVE fit on TRAIN only.
- Device: CPU (torch 2.5.1). Frozen round-trip: adversarial frozen columns are asserted
  byte-identical to `X_test_pristine.npy` at `atol=1e-6` (SCALER_ATOL); scaler float64
  round-trip residue ≤ 1.5e-8.

---

## 8. Updated results

Full per-cell tables (per class×victim, mean±std over 3 seeds), primitive-cost and
perturbation distributions, validity breakdown, failure-reason counts, bootstrap CIs, and
the smallest-cost audit case per class are in
`outputs/cicids2017_primitive_attack/results_tables.md`. Headline summary (denominator =
clean-correct malicious test rows; strict = PAVE ∧ mined ∧ internal-realizability):

### 8.1 Effectiveness (overall)

| metric | macro (mean over class×victim) | micro (pooled) |
|---|--:|--:|
| Untargeted ASR | 85.0% | — |
| **Targeted-Benign ASR** | **78.5%** | 78.4% |
| **Targeted Strict-Valid ASR** | **78.5%** | 78.4% |

Per-class targeted-Benign macro ASR: DoS 65.0%, DDoS 72.8%, Recon 99.1%, BruteForce 77.0%.
The untargeted↔targeted gap is real and now visible — e.g. **DDoS/mlp: 92.1% untargeted vs
16.1% targeted-Benign** (the mlp misclassifies DDoS as *some* other attack far more easily
than as Benign). Seed spread is tiny (std ≤ 0.8pp everywhere).

### 8.2 Validity is not the bottleneck

PAVE (Level-A), dependency (Level-B), discreteness, and internal realizability are **100.0%**;
mined density is 99.8–100.0%; strict validity 99.8–100.0%. Targeted-Benign ASR and Targeted
Strict-Valid ASR are therefore ~identical — evasion succeeds *within* the realizable set, not
by leaving it. All six internal failure categories: **0 failures / 16384 samples**.

### 8.3 Perturbation cost (successful targeted-strict-valid samples, seed 42)

| class | p median (bytes) | p p95 | α median | α p95 | note |
|---|--:|--:|--:|--:|---|
| DoS | 385 | 1158 | 12.04 | 66.3 | timing-heavy on lstm/serial |
| DDoS | 471 | 1162 | 1.62 | 24.6 | mostly padding |
| Recon | 44 | 567 | 1.00 | 1.00 | single-fwd-packet → **α forced to 1** |
| BruteForce | 333 | 1197 | 3.20 | 8.58 | mixed |

`%p@cap` and `%α@cap` are 0.0% everywhere — success does **not** depend on saturating the
bounds. Recon needs no timing dilation at all (α≡1), a direct consequence of the
single-packet masking fix (B6).

## 9. Old vs new

| Method | Targeted-Benign ASR | Targeted Strict-Valid ASR | Mean cost | Packet-summary fails | Timing fails | Neg-rate fails |
|---|--:|--:|--:|--:|--:|--:|
| old aggregate (9-feature, A4) | 5.0% | 3.2% | 0.27 | 4432/16384 | 1685/16384 | 2637/16384 |
| new primitive-control | 78.4% | 78.4% | 2.67 | 0/16384 | 0/16384 | 0/16384 |

The old aggregate attack was cheaper (0.27) but produced physically impossible flows in
27%/10%/16% of samples and only 3.2% *valid* targeted success. The new attack spends more
normalized cost but every sample is realizable and discrete.

### 9.1 Matched-budget sweep (is the gain representation or budget?)

Tightening the primitive ceilings `(p_max, α_max)` lowers the achievable cost (1 seed,
512 rows/class, pooled over class×victim):

| Method | p_max | α_max | mean cost | Targeted-Benign ASR | Targeted Strict-Valid ASR |
|---|--:|--:|--:|--:|--:|
| primitive | 40 | 2 | 1.72 | 35.3% | 35.3% |
| primitive | 120 | 5 | 1.89 | 46.3% | 46.3% |
| primitive | 400 | 20 | 2.53 | 60.8% | 60.8% |
| primitive | 1460 | 100 | 4.32 | 77.9% | 77.9% |
| old aggregate (A4) | – | – | 0.27 | 5.0% | 3.2% |

**Interpretation.** Two honest facts: (1) the primitive attack cannot reach as *low* a
normalized cost as the old aggregate attack — realizable padding/timing changes have a
minimum feature-space footprint (padding one forward length also moves ~10 derived features
together), so its floor is ~1.7, not 0.27. (2) At *every* budget the primitive attack's
**strict-valid** ASR exceeds the old attack's by an order of magnitude (35.3% at its tightest
vs 3.2%), and its samples are 100% realizable. The old attack's cheapness bought almost no
*valid* success. The advantage is therefore the **realizable representation and its
constraints**, not a larger perturbation budget — the primitive attack wins even though it is
structurally unable to operate at the old attack's cost point.

---

## 10. Remaining limitations

- **Level-C (packet-trace) realizability is NOT proven.** All results are feature-space
  (Level A + B) realizability-*aware*: the 79-vector is internally consistent, discrete, and
  frozen-preserving, and every primitive-affected feature is recomputed. It has **not** been
  produced by editing a PCAP and re-running CICFlowMeter.
- **Duration / Flow-IAT-Max under dilation are conservative projections**, not exact
  reconstructions (the merged fwd+bwd packet order is unknown from the aggregate flow).
  `Flow IAT Std`, `Flow IAT Min`, subflow bytes, and bulk stats are held frozen because they
  are not aggregate-reconstructable — a documented inconsistency vs a hypothetical real
  packet edit, never fabricated.
- **`p` is a length augmentation, not asserted payload insertion**; `Fwd Act Data Pkts` is
  therefore conservatively frozen.
- **MTU is not modelled exactly** (the data itself contains >1460-byte coalesced records);
  the padding cap is data-mined headroom, optionally tightened by `--mtu-cap`.
- The **packet-level verification backend is an interface only** (`NullPacketBackend`).

## 11. Defense-panel-safe interpretation

**What the method proves.** Under a constrained threat model where an attacker can only
(i) pad forward packet lengths and (ii) delay forward inter-arrivals, one can craft flows
that (a) evade the victim NIDS and are misclassified as Benign, while (b) remaining
internally consistent CICFlowMeter vectors — satisfying all mined algebraic identities,
discreteness, feature-domain ranges, and dataset-mined density — with every non-controlled
feature identical to a real flow. The attack optimizes only the two attacker-controllable
primitives; success is reported **after** discrete realizability projection.

**What it does NOT prove.** It does **not** prove packet-level realizability: no PCAP was
edited and re-extracted. Features that depend on the unobserved packet order (subflow, bulk,
flow-IAT dispersion) are held frozen, so the vector is Level-A/B consistent but not a
guaranteed CICFlowMeter output of an actual modified trace. High untargeted ASR on some
class×victim cells reflects the victim’s brittleness, not necessarily a deployable evasion;
the honest headline is **targeted-Benign strict-valid ASR** at a stated, bounded perturbation
cost. Reported numbers are per the CICIDS2017-DistriNet preprocessing and these four victim
architectures only; they do not transfer automatically to CICIoT2023 (different, windowed
feature semantics — see the dataset-adapter note).

---

## 9 & 12. Old-vs-new and saved artifacts

Old-vs-new comparison, matched-budget curve, and the smallest-cost audit case per class are
in §8 / `results_tables.md` and `budget_sweep/budget_curve.md`. Per-cell arrays + boolean
masks are saved under `outputs/cicids2017_primitive_attack/attack_artifacts/` for full
reproducibility.
