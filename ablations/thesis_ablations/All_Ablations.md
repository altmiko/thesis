# PrimAttack ablations D6, A2, P1 and P2

This folder holds four ablations of PrimAttack. Each removes or replaces one design element and
reruns the attack on the same flows, victims, seeds and budgets as the FINAL suite. All four are
added analyses outside the locked FINAL protocol (`FINAL_OUTPUTS/00_PROTOCOL.md`).

| folder | element ablated | ablated arm(s) | objective | detailed report |
|---|---|---|---|---|
| `D6_capability_inference/` | source-dependent capability mask M(x) | `no_capability`: M(x) = 1 for every flow | untargeted | `README.md`, `results/report.md` |
| `A2_shape_allocation/` | learned delay allocation `shape` | `shape_fixed_0`, `shape_fixed_0p5`, `shape_fixed_1` | untargeted | `README.md`, `results/report.md` |
| `P1_realization_aware_search/` | scoring and selecting realized integer states during the search | `no_realized_search`: search on the continuous state, one realization at the end | targeted → Benign | `P1_REALIZATION_AWARE_SEARCH.md` |
| `P2_coupled_phi/` | coupled recomputation of dependent features in φ | `direct_only`: search on a map that moves only each primitive's direct statistics | targeted → Benign | `results/P2_COUPLED_PHI_ABLATION.md` |

## Shared evaluation protocol

All four use the harness in `ablations/common/` (`hybrid.py`, `runner.py`, `analysis.py`). With
its default configuration the harness reproduces the FINAL Hybrid cells flow for flow.

| item | setting |
|---|---|
| datasets | CICIDS2017-DistriNet, CICIDS2018-DistriNet |
| victims | `mlp`, `cnn`, `ft_transformer` (2017 canonical checkpoints); `mlp-s42`, `cnn-s42`, `ft_transformer-s42` (2018) |
| source flows | the frozen 800 clean-correct test flows per (dataset, victim, class) from `FINAL_OUTPUTS/runs/<dataset>/baselines_untargeted/selection.json`, SHA-256 verified; classes DoS, DDoS, Recon, BruteForce (3,200 flows per cell and seed) |
| attack seeds | 42, 2024, 2026 |
| budgets | p75 (`maximum-evaluated`, train-fit calibration) and envelope-only `unbounded` |
| cells | 2 datasets × 3 victims × 2 budgets = 12 cells |
| attack | Hybrid Search with the FINAL `PRIM_ARGS`: exact integer padding enumeration, then 40-step adaptive projected refinement of delay and shape (lr 0.1, momentum 0.75, stall halving), restarts until 256 victim evaluations per flow are spent; joint mode; capability-aware padding/timing |
| final validity | validator_v2 `hybrid_valid` = SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED, conditioned on the source flow; the same validator scores every arm's final flow and each layer is stored per row |

The objective follows the comparison each ablation needs. D6 and A2 belong to the untargeted
ablation suite and compare against the shared untargeted reference arm (`ablations/reference`,
equal to FINAL `primattack_hybrid_objective_untargeted` on all 57,600 p75 flows). P1 and P2 were
specified as targeted → Benign. Each runs its own targeted reference arm, which reproduces the
FINAL targeted Hybrid cells (`primattack_hybrid_objective_targeted` at p75,
`primattack_targeted_budgets` unbounded) on all 115,200 flows: identical adversarial flow,
prediction, validity and valid success. Reference Valid ASR therefore differs slightly between
the two pairs. For example, 2017 CNN p75 is 13.47% untargeted (D6, A2) and 13.25% targeted (P1,
P2).

### Metrics

- **Raw ASR**: share of the 3,200 flows whose final realized flow meets the objective
  (untargeted: prediction ≠ source class; targeted: prediction = Benign).
- **Valid ASR**: share whose final realized flow meets the objective and passes validator_v2.
  This is the outcome every ablation is judged on.
- Per cell, ASR is the mean over the three attack seeds; the seed range or SD is shown in the
  detailed reports.
- Additional per-ablation diagnostics: capability-violation counts (D6), learned-shape
  distribution (A2), continuous-to-realized changes (P1), reduced-space vs full-φ outcomes and
  feature-group substitutions (P2).

### Statistics

The test is a paired McNemar on per-flow valid success, ablated arm vs reference, at attack
seed 42. It uses the exact binomial version below 25 discordant pairs, with a Newcombe 95% CI
for the difference. Holm corrects within one ablation: 12 comparisons for D6, P1 and P2; 36 for
A2 (3 arms × 12 cells). Victims and datasets are never pooled, since pooling victims would be
pseudoreplication. Seeds 2024 and 2026 are reported descriptively (P2 also gives them separate
Holm families).

### Sanity checks common to all four

- Both arms of a cell use identical sample IDs in identical order.
- The per-flow budget box and the capability mask are identical across arms (except D6, whose
  ablated arm sets M(x) = 1 by design).
- The final validator_v2 verdict is recomputed from the stored flows.
- The reference arm reproduces the corresponding FINAL cells flow for flow.

## D6: capability inference removed

**Question.** Is the source-dependent capability mask M(x) needed, or does validator_v2 alone
keep the attack feasible?

**Design.** The reference uses u = M(x) ⊙ (p, D, s). M(x) allows padding only on flows with
forward packets, forward payload and no zero-length forward packet, and timing only on flows with
at least two forward packets and a positive `Fwd IAT Total`. `no_capability` sets M(x) = 1, so
the box comes from the train-envelope/budget headroom alone, and φ may pad or delay any flow.
Outside the capability set φ is not differentiable everywhere, so the ablated arm zeroes
non-finite gradient coordinates and counts them: 69 flows per (victim, budget) on CICIDS2018,
none on CICIDS2017.

**Results** (Valid ASR, mean of seeds; Holm over 12):

| dataset | victim | budget | reference | no_capability | Δ (pp) | Raw ASR, no_capability | p (Holm) |
|---|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09% | 0.19% | −3.91 | 13.97% | 6.7e-28 |
| 2017 | mlp | unb | 22.97% | 0.71% | −22.26 | 72.62% | 3.6e-155 |
| 2017 | cnn | p75 | 13.47% | 2.70% | −10.77 | 49.07% | 2.1e-75 |
| 2017 | cnn | unb | 59.94% | 7.25% | −52.69 | 98.16% | <1e-300 |
| 2017 | ft_transformer | p75 | 0.12% | 0.03% | −0.09 | 25.25% | 0.5 |
| 2017 | ft_transformer | unb | 0.55% | 0.03% | −0.52 | 25.69% | 6.1e-05 |
| 2018 | mlp-s42 | p75 | 2.53% | 0.22% | −2.31 | 31.69% | 9.0e-17 |
| 2018 | mlp-s42 | unb | 44.36% | 38.52% | −5.84 | 66.61% | 4.5e-06 |
| 2018 | cnn-s42 | p75 | 1.16% | 0.03% | −1.12 | 17.72% | 3.3e-08 |
| 2018 | cnn-s42 | unb | 26.19% | 4.87% | −21.31 | 68.11% | 5.2e-144 |
| 2018 | ft_transformer-s42 | p75 | 0.00% | 0.00% | 0.00 | 0.36% | 1 |
| 2018 | ft_transformer-s42 | unb | 0.12% | 0.34% | +0.22 | 1.79% | 0.12 |

**Findings.**

1. Removing M(x) lowers Valid ASR in 10 of 12 cells while Raw ASR rises to as much as 98%.
   The gradient pushes padding into flows that cannot carry it. validator_v2 rejects those
   flows: PROTOCOL rejects every capability-violating objective hit on CICIDS2017, MINED most of
   them on CICIDS2018. The search spends its evaluations in infeasible space. On 2017 CNN
   unbounded, valid successes among timing-capable flows drop from 81.20% to 9.82%, although
   every timing-capable flow still received refinement steps.
2. validator_v2 misses one timing violation. On CICIDS2018, 1,512 ablated-arm valid successes add
   forward delay to flows with a single forward packet, which have no forward inter-arrival gap
   (e.g. `Fwd IAT Total` 0 → 1.2e7 µs with `Total Fwd Packet = 1`). No validator_v2 layer
   encodes `Total Fwd Packet = 1 ⇒ Fwd IAT Total = 0`. The 2018 MLP unbounded value (38.52%)
   therefore overstates feasible evasion; only 2,237 of its 3,698 valid successes respect the
   source's capability. The FT-Transformer 2018 unbounded increase (0.12% → 0.34%) is the same
   artifact (24 of 33 successes).
3. M(x) contributes both search efficiency and soundness. It keeps the search in the feasible
   subspace, and on single-forward-packet flows it is the only guard against an impossible
   manipulation. `../V1_single_fwd_packet_rule/` tests the missing rule as a toggle: it accepts
   all 2.91 M genuine flows and never binds for capability-aware PrimAttack.

## A2: delay allocation `shape`

**Question.** Does it matter how the added forward delay is spread over the forward gaps, or
only how much is added?

**Design.** The total delay and its budget are as in the reference; only the allocation changes.
`shape_fixed_0` dilates the existing gaps proportionally (g′ = a·g), `shape_fixed_1` adds the same
delay to every gap (g′ = g + b), `shape_fixed_0p5` mixes the two equally. The reference optimizes
`shape` jointly with padding and delay. A pinned shape applies to every candidate and receives no
gradient. The design follows Nasr et al. (USENIX Security 2021, Sec. 7.2), who vary mean and
spread of added delay separately, and FRONT (USENIX Security 2020, Sec. 5.4).

**Results** (Valid ASR %, mean of seeds; `*` = significant seed-42 McNemar after Holm over 36):

| dataset | victim | budget | reference (learned) | fixed 0 | fixed 0.5 | fixed 1 |
|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09 | 0.00* | 2.12* | 4.09 |
| 2017 | mlp | unb | 22.97 | 0.91* | 19.25* | 22.75 |
| 2017 | cnn | p75 | 13.47 | 0.06* | 7.44* | 13.41 |
| 2017 | cnn | unb | 59.94 | 1.28* | 53.66* | 59.94 |
| 2017 | ft_transformer | p75 | 0.12 | 0.03 | 0.12 | 0.12 |
| 2017 | ft_transformer | unb | 0.55 | 0.25* | 0.53 | 0.59 |
| 2018 | mlp-s42 | p75 | 2.53 | 0.00* | 1.50* | 2.53 |
| 2018 | mlp-s42 | unb | 44.36 | 39.09* | 42.59* | 35.50* |
| 2018 | cnn-s42 | p75 | 1.16 | 0.00* | 0.16* | 1.16 |
| 2018 | cnn-s42 | unb | 26.19 | 25.62* | 26.01 | 26.27 |
| 2018 | ft_transformer-s42 | p75 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2018 | ft_transformer-s42 | unb | 0.12 | 0.12 | 0.09 | 0.09 |

**Findings.**

1. Proportional dilation (shape 0) almost never evades at p75: Valid ASR ≤ 0.06% on every
   victim, against 1.16–13.47% for the reference.
2. Uniform delay (shape 1) matches the reference within 0.06 pp on every p75 cell, none
   significant. The reference's own p75 timing successes have median shape 0.94–1.00.
3. Uniform delay is not best everywhere. On 2018 MLP unbounded it loses 8.9 pp (35.50% vs
   44.36%; 1 vs 284 discordant flows at seed 42), while shape 0 loses 5.3 pp. There the learned
   shape is mostly 0 (56% of successes below 0.05).
4. No fixed shape significantly beats the learned one (≤ 5 fixed-only flows at seed 42).
   Optimizing `shape` lets one search cover both the uniform regime (p75) and the proportional
   regime (2018 MLP unbounded).

## P1: realization-aware search

**Question.** Does PrimAttack need to score and select realized integer primitive states during
the search, or does a continuous-state search realized once at the end reach the same Valid ASR?

**Design.** The reference projects every candidate to integer bytes / µs inside the box and
M(x), maps it through φ with quantization, and lets the success test, stall checkpoints, restart
bests and incumbent read these realized flows. `no_realized_search`
(`HybridConfig(realization_aware_search=False)`) runs the same optimizer and the same gradient,
but scores every candidate on the continuous state (continuous box `[0, bounds]`, M(x), φ
without quantization). The returned candidate is realized once through the FINAL code path, and
only that flow is counted. The continuous search gets 255 evaluations and the final realization
is the 256th, so both arms have the same 256-evaluation cap. validator_v2 is not in the
continuous search's success test, because its SCHEMA layer requires integer µs; amendment A6
showed the in-search gate never changed a FINAL flow.

**Results.** Valid ASR is identical in all 12 cells and all 3 seeds:

| dataset | victim | p75 ref / P1 | unb ref / P1 |
|---|---|---|---|
| 2017 | mlp | 4.09 / 4.09 | 22.94 / 22.94 |
| 2017 | cnn | 13.25 / 13.25 | 59.69 / 59.69 |
| 2017 | ft_transformer | 0.12 / 0.12 | 0.55 / 0.55 |
| 2018 | mlp-s42 | 0.78 / 0.78 | 24.80 / 24.80 |
| 2018 | cnn-s42 | 0.00 / 0.00 | 26.09 / 26.09 |
| 2018 | ft_transformer-s42 | 0.00 / 0.00 | 0.12 / 0.12 |

There are 0 discordant flows at every seed (McNemar p = 1, Holm p = 1). Median primitive cost of
valid successes is identical per cell and seed.

**Findings.**

1. 0 of 14,635 continuous hits lose the Benign prediction after the final rounding, and no flow
   changes its predicted class. Rounding moves the delay by < 1 µs. The smallest delay among
   valid successes is 1,305 µs, and padding is reached only by integer enumeration (no joint
   rows exist under capability-aware PrimAttack), so padding is never rounded.
2. The final realization is what makes flows valid. Before it, 94.3% of continuous hits fail
   validator_v2, all on SCHEMA integer checks (fractional Fwd IAT Min/Max, Flow Duration, Flow
   IAT Max). After it, all pass.
3. Both arms share the gradient path until a margin-based decision differs, and rounding moves
   the margin by ≤ 6.5 × 10⁻³. The final flow is bit-identical to the reference's in 99.0% of
   flows and in 14,630 of 14,635 valid successes.
4. At integer-µs granularity, search-time realization is empirically unnecessary. Coarser
   timestamps or joint padding+timing refinement could change this.

## P2: coupled feature recomputation in φ

**Question.** Is φ's recomputation of dependent features needed to reach valid adversarial
states, or does moving only the statistics each primitive sets directly give the same result?

**Design.** `ablations/common/phi_mapping.py` traces φ's 23 writes from the source code: 7 are
direct (Total Length of Fwd Packet, Fwd Packet Length Min/Max, Fwd IAT Total/Std/Max/Min), 16
derived (e.g. Fwd IAT Mean, Flow Duration, Flow IAT Mean/Max, packet-length aggregates, rates).
`direct_only` searches on a map that keeps the direct writes and holds the derived features at
the source values. Its returned primitives are realized again through full φ, and only that
full-φ flow is scored and compared. validator_v2 is not in its search success test (it would
judge the inconsistent reduced flow; A6 shows the gate is inert in the reference).

**Results** (Valid ASR %, mean ± SD over seeds; `*` = significant after Holm over 12):

| dataset | victim | budget | ref Valid | P2 full-φ Valid | Δ (pp) | P2 reduced-space Raw | seed-42 P2-only / ref-only |
|---|---|---|---|---|---|---|---|
| 2017 | mlp | p75 | 4.09 | 3.08 ± 0.09 | −1.01* | 4.16 | 0 / 34 |
| 2017 | mlp | unb | 22.94 | 14.90 ± 0.02 | −8.04* | 28.91 | 0 / 257 |
| 2017 | cnn | p75 | 13.25 | 12.07 ± 0.15 | −1.18* | 13.41 | 0 / 34 |
| 2017 | cnn | unb | 59.69 | 53.75 ± 0.29 | −5.94* | 60.81 | 0 / 188 |
| 2017 | ft_transformer | p75 | 0.12 | 0.12 | 0.00 | 0.12 | 0 / 0 |
| 2017 | ft_transformer | unb | 0.55 | 0.45 ± 0.02 | −0.10 | 0.41 | 0 / 4 |
| 2018 | mlp-s42 | p75 | 0.78 | 0.69 | −0.09 | 0.69 | 0 / 3 |
| 2018 | mlp-s42 | unb | 24.80 | 24.02 ± 0.04 | −0.78* | 1.61 | 0 / 23 |
| 2018 | cnn-s42 | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0 / 0 |
| 2018 | cnn-s42 | unb | 26.09 | 25.64 ± 0.02 | −0.45* | 20.03 | 0 / 14 |
| 2018 | ft_transformer-s42 | p75 | 0.00 | 0.00 | 0.00 | 0.00 | 0 / 0 |
| 2018 | ft_transformer-s42 | unb | 0.12 | 0.12 | 0.00 | 0.00 | 0 / 0 |

**Findings.**

1. Coupled recomputation raises Valid ASR in 6 of 12 cells: by 1.0–8.0 pp on 2017 MLP and CNN,
   and by 0.45–0.78 pp on 2018 MLP and CNN unbounded. P2 never finds a valid success the
   reference lacks (0 P2-only flows in all 36 seed × cell comparisons). FT-Transformer and the
   2018 p75 cells show no difference.
2. The reduced search misjudges evasion in both directions. It overstates it on 2017 MLP
   unbounded (28.91% seen vs 14.90% under φ) and misses most of it on 2018 MLP unbounded (1.61%
   seen vs 24.02% under φ). 76.6% (2017) and 93.8% (2018) of its apparent successes survive full
   φ; every loss comes from a changed victim prediction.
3. The derived timing features of φ's delay block (Fwd IAT Mean, Flow Duration, Flow IAT Mean,
   Flow IAT Max) account for the lost successes: their full-φ values alone end 96.9% (2017) and
   100% (2018) of them. Padding-derived features play no part, since no valid success pads.
4. validator_v2 accepts 8.3–78.5% of the reduced flows, although ≥ 99.6% of the accepted ones
   break `Fwd IAT Mean = Fwd IAT Total / (Nf − 1)`. No EXTRACTOR rule covers forward IAT means,
   IAT totals or flow duration. This is a second validator_v2 gap after D6. Capability-aware
   PrimAttack is unaffected, because its φ flows pass the Level-B identity and timing checks on
   every row.

## What the four ablations show together

| design element | effect on Valid ASR when removed | where it matters |
|---|---|---|
| capability mask M(x) (D6) | −0.09 to −52.69 pp in 10/12 cells; 2 cells show an artifactual gain | MLP and CNN on both datasets; also guards single-forward-packet flows that validator_v2 accepts |
| learned `shape` (A2) | shape 0 loses almost everything at p75; shape 1 loses 8.9 pp on 2018 MLP unbounded | p75 favours uniform delay, 2018 MLP unbounded proportional delay |
| coupled φ recomputation (P2) | −0.45 to −8.04 pp in 6/12 cells | 2017 MLP/CNN, 2018 MLP/CNN unbounded |
| search-time realization (P1) | 0.00 pp in 12/12 cells | nowhere at integer-µs granularity; the single final realization suffices |

- Three of the four elements carry measurable Valid ASR: the capability mask, the learned delay
  allocation and the coupled recomputation. The fourth, realizing candidates inside the search,
  can be replaced by one realization at the end without loss under the tested granularity.
- Two ablations (D6, P2) expose flows that validator_v2 accepts although they are physically or
  arithmetically inconsistent: delay on single-forward-packet flows, and IAT/duration
  statistics that disagree with the added delay. Neither gap affects capability-aware PrimAttack
  with full φ. Both mean validator_v2 alone is not sufficient evidence of feasibility for other
  attacks.
- On FT-Transformer every ablation moves Valid ASR by at most 0.52 pp, consistent with its
  near-zero Valid ASR in all arms. The ablations do not explain FT-Transformer's robustness.

## Limits

- Results are feature-space proxies on CICFlowMeter aggregates. No PCAP was edited or replayed,
  and packet-level realizability is not established (`CLAUDE.md` claim boundary).
- One frozen cohort of 800 clean-correct flows per class; the three seeds vary only the attack's
  random restarts. Tests are at seed 42 only, so claims of seed robustness are not supported.
- D6 and A2 are untargeted, P1 and P2 targeted → Benign. Their reference numbers are not
  interchangeable.
- P1 covers one primitive granularity (integer bytes, integer µs).
- P2's feature-group substitutions describe these victims' decisions one group at a time. They
  are not a causal decomposition.

## Reproduction

From the repo root in the `thesis` env:

```
python ablations/reference/run.py --device cuda                                   # untargeted reference (D6, A2)
python ablations/thesis_ablations/D6_capability_inference/run.py --device cuda
python ablations/thesis_ablations/A2_shape_allocation/run.py --device cuda
python ablations/thesis_ablations/P1_realization_aware_search/run.py --device cuda  # own targeted reference
python ablations/thesis_ablations/P2_coupled_phi/run.py --device cuda               # own targeted reference
```

`--skip-run` re-analyzes existing cells. Each experiment writes `results/` (tables, tests,
sanity checks); per-row npz artifacts under `results/<dataset>/artifacts/` are git-ignored but
kept locally.
