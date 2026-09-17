## 1. Core Research Question

Given a VAE (+ PGD/C&W) attack that finds classifier-evading perturbations in NIDS flow-feature space, **how much of that perturbation survives the transition from feature-space fiction to genuine host-space reality** - i.e., can a real attacker, using real tools with real command-line parameters, actually produce traffic that realizes the attack, and does the classifier still get fooled once the perturbation is filtered through that physical constraint?

This reframes the thesis from "build a working host-space attack" (binary, risky) to "measure how realizable feature-space attacks are, and under what conditions" (empirical, defensible regardless of outcome).

---

## 2. Grounding in Literature (Why This Is Novel)

| Paper | Contribution | Gap it leaves open |
|---|---|---|
| Apruzzese et al., "What is the Problem Space? Defining Host-Space Perturbations" (ASIA CCS 2026) | Formally defines host-space perturbation (HsP); systematic review of 316→201 papers finds **zero** prior work does genuine HsP; shows one patator flag change (`--persistent=1`→`0`) collapses classifier TPR from ~0.998 to ~0.000 | Only one hand-picked anecdote; no systematic, multi-sample, multi-feature measurement; explicitly critiques Scapy-style packet editing as "not an HsP" |
| Apruzzese et al., SoK - Reshaping Research on NIDS (ASIA CCS 2026) | Three assertions: (1) feature-space attacks implicitly assume pipeline compromise, (2) testbeds must match claimed deployment scale, (3) NetFlow-level TPR/FPR can be practically meaningless vs. host-level alerting | Confirms feature-space attacks (like your VAE) rest on an unrealistic threat model - must be acknowledged explicitly |
| elShehaby & Matrawy, Perturb-ability Score (PS) | Hand-designed rubric (header type, cardinality, correlation, accessibility) scoring which features are "hard to perturb"; used defensively for feature selection/training | Static, heuristic, not empirically measured from real tool behavior; not used as a generative constraint inside an attack |
| Grini et al. | Shows up to **80.3%** of unconstrained adversarial examples are structurally invalid (protocol/categorical constraints violated) | Structural validity only - doesn't address whether *valid* examples are *host-realizable* |
| PLAA (RL), PANDA (image-encoding + AML), NIDSGAN (GAN), CSU 2025 (pure Scapy problem-space, 72% recon evasion) | Packet/problem-space attacks with reported high evasion rates | All stop at packet crafting; none execute real host commands or empirically validate via containerized real-tool traffic |
| ConCap (SaTML 2026) | Kubernetes-based containerized attacker/target harness; runs real tools (Nmap, Patator, Slowloris, wfuzz); auto-captures + auto-labels NetFlow; validated against bare-metal traffic (e.g., slowloris: 21,300 vs 21,305 packets) | Built for dataset generation, not yet used as an adversarial-attack realizability testbed |

**Your positioning:** PS asks *"which features are inherently hard to perturb"* using fixed heuristics. Apruzzese shows the feature→host gap *matters* but only with one example. **This thesis asks "which features can actually be produced by real tools," measured empirically via tool-parameter sweeps, and uses that as a generative constraint inside the attack itself** - a combination that does not exist in the surveyed 2022–2026 literature.

**Important framing constraint:** Since your VAE operates in feature space, it inherits the SoK paper's Assertion-1 critique (implicitly assumes pipeline compromise). Explicitly state this as a limitation; your contribution is *measuring* how much of that fiction survives contact with reality, not claiming the base attack itself is realistic.

---

## 3. Existing Assets (Already Built)

- Per-class VAEs (trained per attack-class pair) with PGD and C&W in the attack loop.
- Real inverse-transformed feature-delta outputs, e.g.:
  - `cPGD-TB, Recon→Benign, CNN`: `Header Len +6.04, IAT +0.000464, Min +2.003, fin count -0.3, ack flag -0.3`
  - `cC&W, DoS→DDoS, MLP`: `psh flag +0.086, Min -0.138, Header Len -0.165, Number +1, syn flag +0.009`
- Confirmed dataset: CICIDS2017 (5-day capture; PortScan/Nmap, Brute Force/Patator, DoS/Slowloris+Hulk+GoldenEye, DDoS/LOIC, Web Attacks, Infiltration, Botnet/Ares, Heartbleed).

### Known data issues already identified in your own outputs
1. **Sub-quantum deltas**: flag/count changes under ~0.5 magnitude (`syn flag: 0.009`, `psh flag: 0.086`, `ack flag: -0.3`) round to zero at the packet level - mathematically real, physically non-existent.
2. **Derived-feature contamination**: `Rate`, `Std`, `Variance` are computed from other features, not independently packet-controllable; treating them as independent optimization targets is invalid without consistency-checking against co-listed deltas.
3. Not all attack-class labels in your data (e.g., "Spoofing") map cleanly to CICIDS2017's 5 standard categories or to ConCap's tool coverage - verify label provenance before committing tool mappings.

---

## 4. Tool Coverage: What's Directly Usable

| Attack class | CICIDS2017 tool | ConCap scenario exists? | 3-week usability |
|---|---|---|---|
| Recon / Port scan | Nmap | Yes (validated) | **Primary case - full coverage** |
| Brute force (SSH/FTP) | Patator | Yes (validated) | Usable if needed |
| DoS (slow-rate) | Slowloris | Yes (validated) | **Secondary case**, only if DoS samples are Slowloris-sourced (not Hulk/GoldenEye) |
| DoS/DDoS (flood) | Hulk, GoldenEye, LOIC | No | Gap - substitute with **hping3** (custom container) instead of porting LOIC |
| Spoofing | N/A (not a standard CICIDS2017 category) | No | Verify label source; likely out of scope for Tier-3 in 3 weeks |

**Recommendation:** Lead with **Recon (Nmap)** as the fully-validated case. Add **hping3** (lighter than LOIC/MHDDoS to containerize) to cover flag-based/flood features for DoS/DDoS/Spoofing-adjacent classes. Treat DDoS-via-LOIC and Spoofing explicitly as scoped-out / future work, citing the tooling gap honestly rather than forcing a weak mapping.

---

## 5. The Realizability Ladder

Rename tiers to match Apruzzese et al.'s own vocabulary for maximum defensibility:

- **Tier 1 - Feature space**: VAE/PGD/C&W output; passes structural validity (one-hot sums, protocol/flag legality, categorical dependency checks per Grini et al.). *Explicitly flagged as resting on an unrealistic "pipeline compromise" assumption (SoK Assertion 1).*
- **Tier 2 - Packet space** *(diagnostic only, not the headline result)*: Scapy-based seed-flow editing (pad payload, shift timestamps, toggle flags, repair checksums) → re-extract → compare gap. Explicitly acknowledged as critiqued by the HsP paper ("not an HsP" - edited packets aren't guaranteed transmittable in that form).
- **Tier 3 - Host space** *(the actual novel contribution)*: real tool + real parameters, executed in ConCap containers, packets emerge as a natural side effect, captured via `tcpdump`, auto-extracted and labeled. This is what "counts" as genuine realizability.

Report a **per-sample, per-feature partial realizability score**, not binary pass/fail:

```
Realizability Score = |{features with achieved gap < τ}| / |{total perturbed features}|
```

---

## 6. Step-by-Step Implementation Plan

### Step A - Perturbability Pre-Audit (1–2 days)
Classify every recurring feature in your existing VAE outputs into:
1. Independently modifiable (packet-controllable directly)
2. Derived/computed (must be recomputed from group 1, never targeted directly)
3. Backward/victim-controlled (attacker cannot influence)
4. Non-decomposable structural (depends on full packet sequence)

Flag sub-quantum deltas (|Δ| < ~0.5 on flag/count features) as Tier-1 "unrealizable by construction" - a reportable finding, not a bug.

### Step B - Tool Parameter Sweep / Response-Surface Study (4–5 days)
Build once, reuse for every sample - the core empirical artifact of the thesis.

1. Pick 2–3 tools: **Nmap** (Recon), **Slowloris** (DoS, if applicable), **hping3** (DoS/DDoS/flag-based).
2. Define parameter grids per tool (e.g., Nmap: `-T0`–`-T5`, port count; Slowloris: `interval_sec`, `num_sockets`; hping3: `-d`, `-i`, flag flags).
3. Run every combination through ConCap **3+ times each** (repeats capture natural jitter → range, not point estimate).
4. Re-extract NetFlow per run; record mean/std per (tool, parameter, feature) triple.
5. Fit a response curve per feature (linear/polynomial); report R² - flags which parameters have clean, predictable effects vs. noisy/flat (uncontrollable) ones.
6. Explicitly test for **interaction effects** (factorial design) when multiple features must be hit simultaneously by one tool.
7. **Deliverable:** a standalone table/heatmap per tool - parameter vs. feature response, range, noise band, controllability rating. Valuable even independent of downstream results.

### Step C - Co-Controllability Clustering (1 day)
Group sweep-validated features by which single tool can jointly produce them (e.g., Cluster A = slowloris-achievable: IAT, packet count, duration; Cluster B = hping3-achievable: packet length, flag counts, IAT). A real attack session draws from one cluster at a time - this becomes a hard constraint on which multi-feature VAE outputs are even attemptable.

### Step D - Controllability-Masked VAE Attack (2–3 days)
Modify the existing PGD/C&W loop to zero out gradient updates on non-sweep-validated feature dimensions:

```python
controllable_mask = torch.zeros(num_features)
controllable_mask[controllable_indices] = 1.0

def masked_pgd_step(x_adv, grad, epsilon, mask):
    return x_adv + epsilon * grad.sign() * mask
```

Run once per co-controllability cluster. Compare **unconstrained vs. constrained attack success rate** per attack class - report both outcomes as findings regardless of direction (small drop = alarming/realistic; large drop = quantifies inflation in prior literature's reported success rates).

### Step E - Target Mapping via Lookup/Interpolation (1–2 days)
For each constrained-VAE adversarial sample, map its feature deltas to tool parameters via direct lookup/interpolation from Step B's response surface - no live search needed for most single-tool, single/few-feature cases. Reserve automated search (below) only for stubborn multi-feature interaction cases.

### Step F - Automated Search Fallback (optional, 2–3 days)
For samples the lookup can't resolve cleanly (parameter interactions, discrete/categorical tool settings):
- **Bayesian optimization** (`scikit-optimize`/`Optuna`) as primary - most sample-efficient given expensive (container-spin-up) evaluations, typically 15–30 evaluations to converge.
- **Small genetic search** only for discrete/interacting-parameter cases BO handles poorly.
- Skip RL (PLAA-style) - too costly to build/train in remaining time; explicitly note this as a deliberate engineering trade-off in the thesis.

Cap search budget per sample (~20–25 evaluations); log best-achieved gap and pass/fail vs. tolerance.

### Step G - Pipeline Automation (2–3 days)
- Orchestration script: read target deltas → search/lookup → render ConCap scenario YAML → trigger run → parse NetFlow → score gap → log.
- Cluster lifecycle: use **k3s/k3d** (lightweight, not full cloud K8s) wrapped in a context manager for clean per-batch state.
- Batch samples within one cluster lifecycle where safe to amortize spin-up cost.
- Parallelize independent sample searches via `ProcessPoolExecutor` (test with 2 workers before scaling).
- **Run a small pilot (5–10 samples, single-threaded) before any large unattended batch** - catch container/network failure modes early.

### Step H - Classifier Re-Evaluation on Realized Traffic (1 day - high value, low cost)
This is the most important verification step. Run ConCap's actual NetFlow output back through your **original trained classifiers** (exact checkpoints used for the original VAE attack, not retrained copies):

```python
def evaluate_realized_sample(netflow_row, classifier, scaler, feature_columns):
    x = scaler.transform(netflow_row[feature_columns].values.reshape(1, -1))
    return classifier.predict(x), (classifier.predict_proba(x) if hasattr(classifier, 'predict_proba') else None)
```

Classify every sample into a 2×2 outcome table:

| | Still evades | No longer evades |
|---|---|---|
| **Small feature gap** | Full transfer (strongest positive result) | Failed transfer (classifier sensitive to real-world noise) |
| **Large feature gap** | Lucky evasion (boundary region broader than VAE's exact point) | Total failure (host-unrealizable) |

Report proportions per attack class and per feature - this table does not exist elsewhere in the literature.

### Step I - (Optional) Extend ConCap Tooling (2–3 days, only if DDoS coverage is a priority)
Use an AI coding agent to: build a Dockerfile for a CLI-based flood tool (hping3 preferred over LOIC - lighter, more scriptable, avoids .NET/GUI portability issues), write a scenario YAML matching ConCap's existing pattern, validate against bare-metal traffic (same comparison method ConCap's authors used), confirm auto-labeling propagates correctly. Deprioritize if time is tight - name as future work instead.

---

## 7. Handling a Full or Partial Null Result

A null result ("VAE attacks don't survive host-space realization") is a **legitimate, strong, citable finding**, not a failure - provided failures are attributed correctly:

| Failure type | Cause | What it supports |
|---|---|---|
| Sub-quantum / derived-feature failure | VAE exploits continuous-relaxation artifacts on discrete/derived features | Methodological finding about attack generation, not about host-space infeasibility |
| Tool-coverage gap | No ConCap/tool scenario exists for that class (e.g., DDoS/LOIC, Spoofing) | Tooling limitation - says nothing about realizability itself |
| Masked VAE + adequate search budget + full tool coverage (e.g., Recon/Nmap) still fails | Genuine host-space infeasibility | **Your strongest, most defensible negative result** |

Report the full outcome distribution (Section 6, Step H table) rather than a single success/fail number. Reframe the thesis question with your advisor now as **"how realizable, and under what conditions"**, not **"can we build a working exploit"** - this protects the thesis regardless of outcome.

---

## 8. Standalone Deliverables (Valuable Even If Downstream Steps Stall)

1. Tool-parameter → NetFlow-feature response-surface tables/heatmaps (Step B) - citable independent of attack results.
2. Perturbability audit of VAE outputs (Step A) - quantifies sub-quantum/derived-feature artifacts empirically, extending PS (elShehaby & Matrawy) from static heuristic to measured evidence.
3. Realizability ladder + partial-realizability scores per sample (Section 5).
4. Classifier re-evaluation 2×2 outcome table (Step H) - novel, not present in surveyed literature.
5. Explicit "unconstrained vs. constrained VAE attack success rate" comparison (Step D).