# Final experiment protocol (locked before any final run)

This file fixes the design of the final thesis suite defined in `master_experiments.md`.
It was written before the first final attack run. Nothing below was changed after results
were seen. Driver: `scripts/run_final_suite.py`. Analysis: `scripts/analyze_final_suite.py`.

## 1. Scope

| Item | Locked value |
|---|---|
| Datasets | CICIDS2017-DistriNet (`cicids2017_distrinet`), CSE-CIC-IDS-2018-DistriNet (`cicids2018_distrinet`) |
| Victims | 2017: `mlp`, `cnn`, `ft_transformer` (the single canonical category checkpoints). 2018: `mlp-s42`, `cnn-s42`, `ft_transformer-s42` (training-seed-42 replicates) |
| Source classes | DoS, DDoS, Recon, BruteForce (malicious categories) |
| Seeds | 42, 2024, 2026 for every run on both datasets |
| Source samples | One canonical list per (dataset, victim, class): 800 clean-correct test flows, seeded uniform rule (selection seed 42), created once by the baseline stage (`runs/<dataset>/baselines_untargeted/selection.json`) and re-verified by SHA-256 in every PrimAttack stage |
| Split / preprocessing | Existing leakage-safe chronological within-label 70/15/15 split; train-only RobustScaler; unchanged |
| Independent validator | validator_v2 `hybrid_valid` = SCHEMA ∧ EXTRACTOR ∧ PROTOCOL ∧ MINED, dataset-specific profile |

### Why one victim per architecture and what the seeds mean

The master rules require one canonical source list per dataset/model/class and the same three
seeds across both datasets. A clean-correct list depends on the victim, so each (dataset,
architecture) uses exactly one frozen checkpoint. CICIDS2017 has one checkpoint per architecture.
For CICIDS2018 the seed-42 training replicate is used; replicates for seed 2026 do not exist.
Seeds 42/2024/2026 are therefore **attack seeds**. They control random starts and restarts.
They do not control victim training. Seed variability is attack-run variability only.

## 2. Locked metrics and success definitions

* Denominator: attempted source samples = the 800 clean-correct flows per (victim, class).
  Already-misclassified flows are never attempted, and a failed attack stays in the
  denominator as a failure.
* Untargeted success: `adv_pred != true source class`.
* Targeted success: `adv_pred == Benign`.
* `validator_pass`: validator_v2 `hybrid_valid` on the same final adversarial raw flow.
* `valid_success = raw_success AND validator_pass`.
* Raw ASR = raw successes / attempted. Valid ASR = valid successes / attempted.
* Validity Gap = Raw ASR − Valid ASR, in percentage points.
* Tables report mean ± SD (sample SD, ddof = 1) over the three seed-level rates. Classes are
  pooled within a victim by attempts: 3,200 flows per victim and seed. Victims and datasets
  are never pooled in tables or tests.

## 3. Statistical plan

* Toolkit: Cochran's Q (3+ paired conditions), McNemar (planned pairs), Holm (within one
  planned family only). alpha = 0.05. Primary inferential outcome: Valid Success.
* **Handling of repeated seeded observations.** The paired unit is one source flow. Each flow
  is attacked once per seed, so the three seeded outcomes of a flow are not independent
  observations. Every inferential test uses exactly one outcome per flow: the pre-specified
  **reference seed 42** (n = 3,200 flows per dataset × victim; classes pooled within victim).
  Seeds 2024 and 2026 enter only through mean ± SD and a descriptive per-seed paired
  difference (pp) with discordant counts, without p-values. The three seed means are never
  used as n = 3 observations.
* **Test unit / family.** One analysis per (dataset, victim). A Holm family = the planned
  McNemar comparisons of one experiment within one (dataset, victim). No cross-thesis or
  cross-victim correction.
* McNemar variant: exact two-sided binomial when discordant pairs b + c < 25, otherwise the
  continuity-corrected χ² statistic (reported). Planned McNemar tests run only when the
  preceding Cochran's Q is significant (Exp A, B, C). Otherwise they are reported as
  "not performed (Q n.s.)".

| Exp | Conditions | Omnibus | Planned McNemar (Holm family) |
|---|---|---|---|
| A | PrimAttack, PGD, C&W, CAPGD-PrimSupport, C-PGD-PrimSupport (all untargeted) | Cochran's Q (5) | PrimAttack vs each of the 4 baselines (Holm over 4) |
| B | Hybrid, Prim-PGD, Prim-C&W (targeted, p75) | Cochran's Q (3) | H vs PGD, H vs C&W, PGD vs C&W (Holm over 3) |
| C | p50, p75, unbounded, per optimizer (targeted) | Cochran's Q (3) | p50 vs p75, p75 vs unbounded (Holm over 2) |
| D | targeted vs untargeted (selected optimizer, p75) | none | one McNemar, no Holm |
| E | raw vs valid success of the same adversarial examples, per main condition | none | one McNemar per condition, no Holm |
| F | validator acceptance of held-out genuine flows | none (descriptive) | none |

## 4. Attack configurations (locked)

| Method | Space / mask | Budget | Iterations / restarts | Loss | Projection | Stopping / selection | Source |
|---|---|---|---|---|---|---|---|
| PGD | victim RobustScaler space, all 79 features | L∞ ε = 0.5 | 40 steps, α = 0.05, 1 random start | CE (untargeted) | L∞ ball only | fixed steps, final iterate | `attack/input_baselines.py:input_pgd_attack` |
| C&W | victim RobustScaler space, all 79 features | L2 penalty (unbounded) | ≤ 60 Adam steps, lr 0.01, λ = 1, κ = 0 | margin + ‖δ‖² | none | convergence 1e-5; lowest-L2 success kept | `attack/input_baselines.py:input_cw_attack` |
| CAPGD-PrimSupport | train min-max space; 23-feature `primattack_joint_feature_mask` | L2 ε = 0.5 | 10 steps, 2 restarts (TabularBench CAPGD) | CE (untargeted) + TabularBench constraint repair | norm ball + train box + mask + type repair | fixed steps | `external/tabularbench`, `comparisons/capgd_cicids2017.py` |
| C-PGD-PrimSupport | same 23-feature mask | L2 ε = 0.5 | 40 steps, step 0.05, 1 random start | CE − λ·constraint penalty, λ = 1 | norm ball + train box + mask + type repair | fixed steps | `comparisons/cpgd_prim_support.py` (Simonetto et al., IJCAI 2022) |
| PrimAttack (all optimizers) | primitive controls (p, delay, shape), joint mode | train-calibrated per-class p50 / p75 / unbounded (envelope) box | per-flow cap B = 256 victim evaluations | objective margin (targeted or untargeted) | integer bytes / µs projection, canonical recomputation φ, quantized realized flow | incumbent: success > failure; lowest normalized primitive cost among successes; best margin among failures | `attack/primitive_optimizer.py` |

PrimAttack optimizers: **Hybrid** (40 steps/restart, lr 0.1, restarts until budget), **Prim-PGD**
(3 restarts × 42 steps, α = 0.05, momentum 0.75), **Prim-C&W** (3 binary-search stages × 42 Adam
steps, lr 0.5, c₀ = 1, κ = 0). Prim-PGD and Prim-C&W hyperparameters were frozen on the
CICIDS2017 **validation** split (`primattack_optimizer_ablation.md` §3.4). The same values are
reused on CICIDS2018 without re-tuning. All PrimAttack variants share `RealizedSearch`: the same
parameterization, budgets, recomputation, rounding, victim, validator gate and success predicate.
PrimAttack's search success predicate includes validator_v2 (validity-aware search). The
baselines optimize without the validator. This is part of the threat-model difference and is
documented in every fairness guide.

## 5. Experiment design decisions not fixed by the master prompt

* **Exp A PrimAttack configuration:** selected optimizer (Exp B rule), untargeted, **p75**
  budget, joint mode. The unbounded-budget untargeted run is a descriptive p75-vs-unbounded
  comparison. It is not a sixth condition in Cochran's Q.
* **Exp B budget:** p75 (the `maximum-evaluated` headline budget).
* **Optimizer-selection rule (pre-registered):** the highest aggregate Valid Targeted ASR at
  p75, pooled over both datasets, all victims, classes and seeds (sum of valid successes / sum
  of attempts). Ties are broken by fewer mean victim evaluations per flow, then by the order
  Hybrid, Prim-PGD, Prim-C&W. p-values play no role.
* **Exp C optimizers:** the two top-ranked optimizers under that rule. Their p75 cells are the
  Exp B cells: same rows, seeds and configuration.
* **Exp D:** targeted arm = the selected optimizer's Exp B p75 cells; untargeted arm = the
  Exp A PrimAttack cells.
* **Exp E conditions:** the five Exp A methods (untargeted) and the Exp B optimizers
  (targeted, p75), each on its own generated adversarial examples.
* **Exp F:** validator_v2 on every genuine flow of the held-out validation and test splits of
  both datasets, all classes including Benign. Not used for mining. General-only =
  SCHEMA ∧ EXTRACTOR ∧ PROTOCOL; general + dataset-specific = additionally MINED.

## 6. Fail-loud rules implemented

* Runners raise instead of dropping a cell on non-finite output or on non-clean-correct
  eligible rows.
* The analyzer refuses to aggregate unless every expected (dataset, victim, class, seed,
  condition) cell exists, sample IDs are identical in order without duplicates, the clean-input
  hash, labels, clean predictions and victim checkpoint hash are identical, the seed sets
  match and the denominators are equal.
* Stored `validator_pass` is recomputed from the stored final adversarial flow. A mismatch
  aborts the analysis.

## 7. Amendment log (methodological fixes and requested additions)

* **A1 — non-finite surrogate gradient at pinned PrimAttack controls.** The first Exp B run
  crashed on CICIDS2018 (`mlp-s42`/Recon/p75, Prim-C&W): `project_controls` received NaN
  controls. Cause: for flows with no padding headroom (`p_hi < 1`) and no backward packets, the
  partial derivative of the canonical map φ with respect to padding at p = 0 is NaN. The
  padding coordinate is pinned (multiplied by `p_hi = 0`), yet 0 · NaN = NaN leaked into the
  step. Prim-C&W crashed. Hybrid and Prim-PGD would silently freeze the affected rows, because
  `sign(NaN) = 0` in torch, so those rows would never move. Fix (`attack/primitive_optimizer.py`):
  pinned coordinates are detached in `RealizedSearch.surrogate_logits`, which gives them an
  exactly-zero gradient, and every optimizer step goes through `objective_gradient`, which
  raises on any non-finite gradient. Regression test:
  `test_pinned_padding_control_cannot_poison_timing_steps`. The fix only changes steps of rows
  that have a pinned control, and a pinned control can still carry a small finite gradient that
  enters the step normalization. The already finished CICIDS2017 Exp B stage was therefore
  discarded, and Exp B was re-run from scratch on both
  datasets before the optimizer selection. The discarded logs are kept in
  `runs/_superseded_logs/`. The baseline stage does not use this code and was not re-run.
  (These logs moved with the whole pre-A2 run to `superseded_relaxed_padding/runs/`.)
* **A2 — empty-forward-packet padding capability (PrimAttack) and validator rule.** Padding
  adds `p` bytes to every forward packet. The pre-A2 capability rule allowed it whenever the
  flow had forward payload, including flows with `Fwd Packet Length Min = 0`, i.e. with at
  least one zero-length forward packet; padding then filled an empty packet (payload
  insertion). On CICIDS2017 1,464 of the 1,544 seed-42 valid PrimAttack successes (Exp A,
  p75) did this; CICIDS2018 already rejected it through `MINED_0001`. Changes:
  (1) `CICIDS2017PrimitiveModel.infer_capabilities` requires `Fwd Packet Length Min > 0` for
  padding (reason `EMPTY_FWD_PACKET`), before optimization, so such flows are attacked
  timing-only with the full per-flow budget (per-row search space recorded; Hybrid/Prim-PGD
  step normalization uses free coordinates only);
  (2) validator_v2 gains the source-conditioned PROTOCOL rule `PROTO_0080`
  (`zero_preserved::Fwd Packet Length Min`: source 0 ⇒ perturbed 0) on **both** datasets,
  applied to every attack's output; genuine flows are their own source, so Exp F acceptance
  is unaffected;
  (3) new stage `primattack_untargeted_modes`: the Exp A PrimAttack configuration restricted
  to timing-only and padding-only (joint = the Exp A cell), reported descriptively (no new
  test family).
  Everything else is unchanged (source lists, victims, budgets and their train-only
  calibration, the 23-feature support mask, seeds, objectives, metrics, statistical plan,
  optimizer-selection rule). Because the validator changed, **every stage was re-run from
  scratch** (baselines included) on both datasets, and the optimizer selection was re-applied
  with the frozen rule. The pre-A2 run and reports are kept, non-canonical, in
  `superseded_relaxed_padding/` and reported only as the `PrimAttack-relaxed-padding`
  sensitivity result. Audit: `../primattack_empty_packet_fix_report.md`.
* **A3 — native CAPGD as a descriptive Exp A row (requested).** `capgd_native` (TabularBench
  CAPGD with its own configuration mask, L2 ε = 0.5, 10 steps, 2 restarts) runs in the
  baseline stage on the same flows and seeds and is judged by the same validator. It is
  reported (†) but is not part of Exp A's Cochran's Q or Holm family, which stay as locked.
* **Realized sample counts.** Every (dataset, victim, class) had ≥ 800 clean-correct test flows,
  so every cell has exactly 800 flows (3,200 per victim and seed).
