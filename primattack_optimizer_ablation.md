# PrimAttack optimizer ablation — Hybrid Search vs Prim-PGD vs Prim-C&W (CICIDS2017-DistriNet)

Scope as run: **three optimizers** (Hybrid Search = the proposed/default PrimAttack search,
Prim-PGD, Prim-C&W) on **CICIDS2017-DistriNet only**. Random Search, Pattern/Coordinate Search,
and the CICIDS2018 run from the original brief were dropped at the user's request. All results
are feature-space proxies on aggregate CICFlowMeter statistics (CLAUDE.md global claim boundary):
no PCAP is edited or replayed.

| Artifact | Path |
|---|---|
| Shared search core + 3 optimizers | `src/attack/primitive_optimizer.py` |
| Ablation runner | `scripts/run_primattack_optimizer_ablation.py` |
| Validation-split tuning | `scripts/tune_primattack_ablation_baselines.py` → `outputs/primattack_optimizer_ablation/val_tuning/` |
| Analysis (tables, tests, plots) | `scripts/analyze_primattack_optimizer_ablation.py` → `outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/` |
| Test-run artifacts (324 per-row npz + `cells.json`, `config.json`, `selection.json`) | `outputs/primattack_optimizer_ablation/cicids2017_distrinet/` |
| Run log | `outputs/primattack_optimizer_ablation/run_cicids2017.log` |
| Unit tests | `src/attack/tests/test_primitive_optimizer.py` (12 passed) |

No earlier result directory was touched. The ablation writes to a new root, and the runner
refuses to write into a non-empty output directory unless `--resume` is passed.

## Summary tables

### S1. Headline results (valid ASR = targeted→Benign ∧ validator_v2 `hybrid_valid`)

ASR and median cost are means over seeds 42/123/2024, with n = 3,200 flows per seed (4 classes ×
800). Δ columns are the primary paired tests at seed 42: McNemar with Newcombe 95% CI, Holm over
27 tests. ASR@64 is the valid ASR reached within 64 victim evaluations per flow. The budget is
B = 256 evaluations per flow for every method.

| Victim | Budget | Hybrid | Prim-PGD | Prim-C&W | Hybrid − PGD (pp) [95% CI], Holm p | Hybrid − C&W (pp) [95% CI], Holm p | ASR@64 H / P / C | median cost H / P / C |
|---|---|---|---|---|---|---|---|---|
| mlp | p50 | 6.31% | 6.31% | 6.31% | +0.00 [-0.12, +0.12], 1.0e+00 | +0.00 [-0.12, +0.12], 1.0e+00 | 4.1 / 6.3 / 6.1% | 1.50 / 1.49 / 1.25 |
| mlp | p75 | 11.06% | 11.06% | 11.00% | +0.00 [-0.11, +0.11], 1.0e+00 | +0.06 [-0.08, +0.21], 1.0e+00 | 4.4 / 11.0 / 11.0% | 1.40 / 1.32 / 1.06 |
| mlp | unb | 41.90% | 42.04% | 41.09% | -0.03 [-0.18, +0.12], 1.0e+00 | +0.91 [+0.56, +1.25], 1.0e-05 | 0.5 / 41.8 / 40.3% | 0.79 / 0.59 / 0.42 |
| cnn | p50 | 13.11% | 13.04% | 2.84% | +0.00 [-0.18, +0.18], 1.0e+00 | +10.22 [+9.20, +11.31], 2.7e-71 | 2.7 / 2.8 / 2.3% | 1.39 / 1.34 / 0.48 |
| cnn | p75 | 36.20% | 36.09% | 14.31% | +0.13 [-0.10, +0.35], 1.0e+00 | +21.84 [+20.41, +23.28], 3.4e-152 | 3.2 / 13.3 / 12.8% | 1.61 / 1.58 / 1.48 |
| cnn | unb | 70.49% | 70.51% | 43.38% | +0.00 [-0.12, +0.12], 1.0e+00 | +27.12 [+25.57, +28.65], 1.8e-188 | 1.8 / 46.2 / 42.3% | 0.98 / 0.59 / 0.37 |
| ft_transformer | p50 | 0.16% | 0.16% | 0.16% | +0.00 [-0.13, +0.13], 1.0e+00 | +0.00 [-0.13, +0.13], 1.0e+00 | 0.2 / 0.2 / 0.2% | 0.20 / 0.32 / 0.17 |
| ft_transformer | p75 | 0.44% | 0.44% | 0.44% | +0.00 [-0.13, +0.13], 1.0e+00 | +0.00 [-0.13, +0.13], 1.0e+00 | 0.4 / 0.4 / 0.4% | 0.20 / 0.21 / 0.20 |
| ft_transformer | unb | 0.88% | 0.88% | 0.38% | +0.00 [-0.13, +0.13], 1.0e+00 | +0.50 [+0.26, +0.81], 5.8e-04 | 0.6 / 0.9 / 0.3% | 0.22 / 0.20 / 0.67 |

### S2. Key findings

| # | Finding | Evidence |
|---|---|---|
| 1 | Hybrid ≡ Prim-PGD in effectiveness | No Holm-significant difference in 9/9 victim×budget cells; seed means within 0.14 pp; all CIs within ±0.35 pp; ≤ 8 vs 4 discordant flows of 3,200 (§6.1, §6.3) |
| 2 | Hybrid and Prim-PGD beat Prim-C&W | CNN +10.2 / +21.8 / +27.1 pp (p50 / p75 / unb); MLP unb +0.9 pp; FT unb +0.5 pp; Holm p ≤ 5.8e−4 |
| 3 | Victim dominates optimizer | FT-Transformer ≤ 0.88% for every method/budget; mlp up to 42%, cnn up to 70% (unb) |
| 4 | Recon unattackable in this primitive space | 99% of Recon flows have no integer headroom → 24.75% `no_headroom` per victim |
| 5 | Hybrid is least query-efficient | Up to ⌊p_hi⌋ (≤ 78) padding evaluations before any timing move; cnn unb ASR@64 1.8% vs 46.2% (PGD) vs 42.3% (C&W); parity only near 256 |
| 6 | Hybrid finds the most expensive successes | Median normalized cost, cnn unb: 0.98 vs 0.60 vs 0.37 |
| 7 | Exact padding enumeration adds few successes | Winning source `exact-padding`: 5/415, 10/1,149, 66/2,256 CNN successes (seed 42) |
| 8 | Fixed Hybrid > pre-fix Hybrid (same flows) | Up to +1.1 pp; Holm-significant on cnn p50/p75 and mlp unb (§9.1) |
| 9 | Directed search ≫ one-draw random control | Up to +30.8 pp (cnn p75) (§9.1) |
| 10 | Directed search ≫ replaced `(p, α)` Adam optimizer | e.g. cnn p75 36.2% here vs 1.14% historically (different selection, unpaired); paired v2-vs-old: 34.91% vs 1.19% (§9.2) |
| 11 | Runtime comparable | 2.7–3.3 ms/flow (MLP/CNN), 10.9–13.4 ms/flow (FT); Hybrid mean 185–189 evaluations vs 190.6 (PGD/C&W) |

### S3. Methodological bugs fixed before the run

| ID | Problem | Evidence | Fix |
|---|---|---|---|
| F1 | Optimizer selected on "classified Benign" only; reported success also requires validator validity | CICIDS2018 cnn-s123 unb: 41.5% raw vs 21.2% valid; 491/491 targeted DDoS flows fail `MINED_0001` under padding | Success = targeted ∧ `hybrid_valid`, owned by the shared `RealizedSearch`; applied to all methods |
| F2 | Zero gradient at zero controls (φ copies the flow when `p = 0` / `delay = 0`) | Gradient `(0, 0, 0)` at `q = 0`; first smoke run: C&W 0% everywhere; Hybrid restart 0 could never move timing; PGD clean start never moved | Surrogate evaluated at `max(q, 1e−3)` with straight-through gradient; realized scoring unchanged |
| F3 | Query counts were batched calls, not per-flow evaluations | One padding chunk of up to 4,096 candidates counted as 1 | Per-flow realized / surrogate / backward counters, first-success index; shared cap B = 256 enforced |
| F4 | Existing random-feasible control uses 1 evaluation vs hundreds for search | Canonical driver design | Used only as historical comparator, never as budget-matched baseline |
| F5 | Baseline hyperparameters must not be tuned on test | — | PGD α = 0.05; C&W c₀ = 1, lr = 0.5 selected on the validation split (C&W lr grid extended until plateau) |
| F6 | Incumbent vs final re-evaluation cross-check | 2 / 259,200 flows disagree, \|margin\| ≤ 5.4e−4 logits (float noise) | Reported ASR uses the conservative final re-evaluation |

### S4. Changes affecting other runs

| Change | Impact |
|---|---|
| F1 + F2 now in `scripts/run_full_adversarial_eval.py` and `src/attack/run_cicids2017_primitive_attack.py` | Future canonical campaigns will differ slightly from `outputs/adv_campaign_noidr` (not re-run, not overwritten) |
| Hybrid defaults were designed after inspecting the test roster (historical) | Possible advantage to Hybrid; listed in §10 |
| New scripts: `run_primattack_optimizer_ablation.py`, `tune_primattack_ablation_baselines.py`, `analyze_primattack_optimizer_ablation.py` | Reproduce the ablation (§4) |
| Docs updated: `docs/full_thesis_methodology/02_primattack.md`, `00_CODE_MAP.md` | Optimizer section and code map reflect the shared core and fixes |

---

## 0. Pre-run audit and methodological fixes

The existing PrimAttack stack was audited before any ablation run. The audit covered leakage,
inconsistent budgets, mismatched success definitions, continuous-vs-rounded scoring,
final-iterate-only selection, and unfair query budgets. The code was fixed before running, and
each fix below is checked in a test or a run.

### F1 — Mismatched success definition (fixed)

**Finding.** `optimize_primitive_candidates` ranked candidates on `argmax(logits) == Benign`
only. The reported headline success is `targeted ∧ validator_v2 hybrid_valid`. A
validator-invalid "success" therefore (a) beat every valid failure, (b) stopped the exact
padding sweep for that row, and (c) kept the row out of timing refinement.
**Evidence.** In the canonical no-IDR campaign on CICIDS2018, `prim_search_joint_unb` on
`cnn-s123` gave a raw targeted rate of 41.5% but only 21.2% valid. Regenerating the stored
controls showed 491/491 targeted DDoS flows failing mined rule `MINED_0001`
(`Fwd Packet Length Min == Packet Length Min`), which padding breaks. On CICIDS2017 the gap is
small (e.g. ft_transformer unb 0.9% raw vs 0.8% valid).
**Fix.** A single `RealizedSearch` object now owns the success predicate:
success = `argmax == Benign` on the realized flow **∧** an injected validity gate. Every runner
passes `hybrid_valid_gate(dataset)` (validator_v2 `hybrid_valid`). The gate is also wired into
the canonical `scripts/run_full_adversarial_eval.py` and into
`src/attack/run_cicids2017_primitive_attack.py`. Incumbent ordering: a success beats a failure;
successes are ranked by normalized cost `p/p_hi + delay/delay_hi`, then margin; failures by
margin `max(non-Benign logits) − Benign logit`.

### F2 — Zero-gradient dead zone in the continuous relaxation (fixed)

**Finding.** The canonical map φ copies a flow verbatim wherever `p == 0` (resp. `delay == 0`),
via `write_when(..., p != 0)` and `torch.where(identity_rows, raw, x)`. The relaxation's gradient
is therefore **exactly zero at any zero control**.
**Evidence.** A probe on mlp/DoS/unbounded gave a median gradient w.r.t. `(p, delay, shape)` of
`0, 0, 0` at `q = 0`, and `0.10, −166.6, −146.8` at `q = 0.01`. In the first smoke run Prim-C&W,
which restarts every binary-search stage at `q = 0`, got **0% on every cell**. The same defect
hit the existing Hybrid: its restart 0 starts at `delay = 0`, so it could never move timing, and
Prim-PGD's clean restart could never leave the identity.
**Fix.** Surrogate evaluations use `max(q, SURROGATE_FLOOR = 1e−3)` on coordinates with integer
headroom, with a straight-through gradient to `q` (`RealizedSearch.surrogate_logits`). Realized
scoring never sees the floor. The fix sits in the shared core, so it applies identically to all
three optimizers.

### F3 — Query accounting counted batched calls, not per-flow evaluations (fixed)

**Finding.** `forward_evaluations` / `backward_evaluations` were scalar counts of *batched victim
calls*. For example, one exact-padding chunk of up to 4,096 flow-candidates counted as 1.
**Fix.** Per-flow int64 counters of realized forwards, surrogate forwards, and backwards, plus
the 1-based evaluation index of the first success, the first target-classified candidate, and
the restart/stage of the first success. Batched padding values after a row's first success are
discarded and not charged, so the count equals sequential scoring. An optional per-flow cap
`eval_budget` masks rows that cannot afford another query (a gradient step costs 2: one surrogate
and one realized forward).

### F4 — Unfair query budget in the existing random control (documented, not reused)

The canonical driver's `random-feasible` control is **one** uniform draw per flow, against
hundreds of evaluations for search. It appears in this report only as a historical comparator
(§9) and never as a budget-matched baseline. All three ablation methods share one per-flow cap
(§2).

### F5 — Baseline hyperparameters must not be chosen on test (fixed by protocol)

Prim-PGD and Prim-C&W hyperparameters were selected on a fresh clean-correct **validation**
selection (§3.4). Test rows were never used for selection. Hybrid keeps the canonical settings.

### F6 — Incumbent vs. final re-evaluation cross-check (added)

Each cell re-evaluates the final incumbents in one 800-row batch (`evaluate_cell`, validator_v2,
realizability, semantics) and records `incumbent_final_mismatch`. Over 324 cells × 800 rows there
are **2 mismatches**, both Hybrid cnn/DDoS/unbounded (seeds 42 and 2024), with incumbent margins
of −5.4e−4 and −1.2e−4 logits. That is batch-composition float noise at the decision boundary.
All reported ASRs use the **final re-evaluation**, the conservative choice.

### Audited and found consistent

- **Leakage.** The budget calibration is train-only (`fit_split == "train"` is asserted by the
  runner). The test selection is the frozen canonical one: SHA-256 of the sample ids is
  re-verified, and every row is re-checked as clean-correct. Tuning reads only the validation
  split. Caveat: the Hybrid defaults (lr 0.1, 40 steps, 2 restarts) were designed after
  diagnostic inspection of the frozen test roster (post-hoc caveat in
  `PRIMATTACK_V2_OPTIMIZER_COMPARISON.md`). If anything, this favours Hybrid.
- **Budgets.** All methods receive the *same* `bounds` tensor per (victim, class, budget), built
  once from `class_calibration`/`unbounded_calibration` → `per_flow_bounds` →
  `_apply_primitive_mode(joint)`.
- **Rounding.** Every committed candidate is `project_controls` (integer bytes/µs) →
  `generate(quantize=True)` → victim. The tests assert that each result equals the realized
  projection of its request, including `success == targeted ∧ gate`.
- **Final-iterate selection.** None. All three methods keep a per-flow incumbent throughout.
- **Success definition.** One predicate object is shared by all methods (F1).

---

## 1. Optimizer implementations (exact)

All three optimizers run on `attack.primitive_optimizer.RealizedSearch`. Normalized controls are
`q ∈ [0,1]³` ↦ `(p, delay, shape) = (p_hi·q_p, delay_hi·q_d, shape_hi·q_s)`. A coordinate is
pinned to 0 when its integer headroom is < 1 (`p_hi < 1`; `delay_hi < 1` pins delay and shape).
Every step's realized candidate is committed to the incumbent. The targeted margin is
`m = max_{k≠Benign} z_k − z_Benign`.

**Identity (shared).** Every method starts with one realized evaluation of `(0,0,0)`; it counts
toward the budget. Rows with no integer headroom in either control stop there (1 evaluation) for
every method.

**Hybrid Search** — `optimize_primitive_candidates` (proposed/default; unchanged algorithm apart
from F1–F3):
1. *Exact padding enumeration*: `p = 1, 2, …, ⌊p_hi⌋` with `delay = shape = 0`, in increasing
   cost order, batched ≤ 4,096 candidates per call. A row leaves after its first (valid,
   targeted) success or when its cap is exhausted.
2. *Adaptive projected refinement* for rows still unresolved that have `delay_hi ≥ 1 µs`.
   Restart 0 starts from the best identity/padding control (`adaptive-clean`); later restarts
   start uniformly in the box (`adaptive-random`). Per step:
   `g = ∇_q Σ m(φ(x₀, max(q, 1e−3)))`, `v ← 0.75·v + g / mean|g|`,
   `q ← clip(q − η·sign(v), 0, 1)`, then realize and commit. Every `max(5, T/4)` steps, rows
   whose restart-best realized margin has not improved by 1e−6 halve η, return to their
   restart-best `q`, and zero `v`.
3. In the ablation, `restarts=None`: restarts continue until each refined row has spent the
   per-flow budget, and the last restart is truncated. The canonical 2-restart schedule is the
   exact prefix: 1 + ⌊p_hi⌋ (≤ 78) + 2·40·2 ≤ 239 ≤ 256. Its ASR is reported as
   "Hybrid R=2 prefix", computed from `first_success_phase < 2`.

**Prim-PGD** — `optimize_primitive_pgd`: fixed-step projected sign-momentum descent on `m` over
normalized controls. There is no padding enumeration, no step adaptation, and no reset to a
restart's best point. Restart 0 starts at `q = 0` (clean flow; relaxation evaluated at the floor);
restarts 1–2 start uniformly in the box. Same update as Hybrid with constant `α` and momentum `μ`.

**Prim-C&W** — `optimize_primitive_cw`: projected Adam on the C&W-style objective
`L(q) = cost(q) + c · max(m(q) + κ, 0)`, with `cost(q) = q_p·[p_hi≥1] + q_d·[delay_hi≥1]`. This is
the same normalized cost the incumbent ranks successes by; shape is free. `q` is clipped to
`[0,1]³` after each Adam step. A tanh reparameterization is impossible because the clean point is
the box corner `q = 0`. Each binary-search stage restarts from `q = 0` with fresh Adam state.
Per-flow `c` update after each stage, as in Carlini & Wagner:
- stage with a realized success: `upper = min(upper, c)`;
- otherwise: `lower = max(lower, c)`;
- then `c = (lower + upper)/2` if `upper < 1e9`, else `c ← 10c`.

The best realized candidate is retained continuously through the incumbent.

## 2. Fairness and evaluation protocol

- **Identical attack space.** Same frozen source rows, victims, primitive controls
  `(p, delay, shape)` in **joint** mode, calibrated per-flow hard box per budget, canonical feature
  recomputation φ with integer quantization, validator_v2 `hybrid_valid` gate, target class
  Benign (id 0), and incumbent ordering. All of it comes from one code path (`RealizedSearch`).
- **Matched query budget.** Every method has the same per-flow cap of **B = 256 victim forward
  evaluations** (realized + surrogate, including the shared identity evaluation). One gradient
  iteration costs 2 (1 surrogate forward+backward, 1 realized forward). Backward passes are
  reported separately.
  - PGD and C&W use 1 + 3·42·2 = **253** evaluations per movable flow.
  - Hybrid uses 1 + (padding values actually scored) + 2·(refinement steps) ≤ 256; it spends
    less when padding succeeds early.
  - Realized evaluations alone: PGD/C&W 127 per movable flow, Hybrid up to 1 + ⌊p_hi⌋ + 127.
- **Anytime reporting.** The per-flow first-success index turns every run into an ASR-vs-budget
  curve. Incumbents never lose a success, so `ASR@Q = P(first_success ≤ Q)` is exactly the ASR a
  method would have had with budget Q. This is exact for PGD's and C&W's fixed schedules;
  Hybrid's restart scheduling does not depend on Q.
- **Transparency.** Per flow: realized, surrogate, and backward counts, first-success index,
  restart/stage of first success, winning candidate source. Per cell: optimizer iterations,
  restarts started, and wall-clock runtime (CUDA-synchronized, excluding final evaluation).
- **Outcomes** (nested, one clean-correct denominator):
  - *valid ASR* (primary) = targeted ∧ `hybrid_valid`;
  - *raw targeted* = victim predicts Benign on the final flow;
  - *SP-ASR* = valid ∧ primitive-feasible ∧ flow-semantic PASS. Recon/BruteForce are
    `NOT_FULLY_TESTABLE` by design and never PASS.
- **Failure classes** for unsuccessful flows:
  - `no_headroom`: no legal non-identity control;
  - `invalid`: a Benign-classified candidate was found but none was validator-valid;
  - `exhausted`: the victim never predicted Benign within the budget.
- **Unique successes**: flows only that method solved, within the same (victim, class, budget,
  seed).

## 3. Hyperparameters

### 3.1 Shared

| Item | Value |
|---|---|
| Per-flow evaluation budget B | 256 (realized + surrogate forwards) |
| Primitive mode | joint (`p`, `delay`, `shape`) |
| Budgets | intermediate = class train p50; maximum-evaluated = class train p75; unbounded = envelope-only (`p_max = ∞`, relative duration change ∞; train-p99 feature envelope + DoS/DDoS min-rate floor) — `artifacts/primattack/budget_calibration.json` (train-fit) |
| Surrogate floor | 1e−3 (normalized, straight-through) |
| Success gate | validator_v2 `hybrid_valid` (`validation/rules/cicids2017_distrinet`) |
| Victim batch / padding chunk | 8,192 / 4,096 candidate rows |

### 3.2 Per method (frozen before the test run)

| Method | Hyperparameters |
|---|---|
| Hybrid | T = 40 steps/restart, η₀ = 0.1, momentum 0.75, stall check every 10 steps (tolerance 1e−6), step ×0.5 + reset to restart-best + zero momentum on stall, restart 0 from best padding control, later restarts uniform, restarts until budget exhausted (canonical 2 = prefix) |
| Prim-PGD | R = 3 restarts (clean + 2 uniform), T = 42 steps/restart, α = 0.05 fixed, momentum 0.75, gradient normalized by mean \|g\| |
| Prim-C&W | S = 3 binary-search stages, T = 42 Adam steps/stage, lr = 0.5, β = (0.9, 0.999), ε = 1e−8, c₀ = 1, κ = 0, restart from `q = 0` each stage |

### 3.3 Hybrid defaults

These are the canonical PrimAttack defaults (`--prim-steps 40 --prim-lr 0.1 --prim-restarts 2`).
They were not re-tuned.

### 3.4 Validation-split selection of baseline hyperparameters

Selection used a fresh clean-correct selection on `X_val_pristine`: the canonical random rule
with seed 42, 200 flows per victim×class, 3 victims × 4 classes × 3 budgets, attack seed 42,
B = 256. Criterion: maximum pooled valid ASR, ties broken by lower median cost. The Prim-C&W lr
grid was extended to 0.2 and 0.5 after the first grid's best value sat on its edge (0.1); the
extended grid shows a plateau.

| Config | Pooled valid ASR (val, n = 7,200) | Median cost |
|---|---|---|
| PGD α = 0.02 | 18.83% | 0.861 |
| **PGD α = 0.05 (chosen)** | **18.97%** | 0.929 |
| PGD α = 0.1 | 18.96% | 0.995 |
| PGD α = 0.2 | 18.85% | 1.100 |
| C&W c₀ = 1, lr 0.02 / 0.05 / 0.1 / 0.2 | 7.46% / 10.33% / 11.57% / 11.72% | 0.477 / 0.479 / 0.425 / 0.546 |
| **C&W c₀ = 1, lr 0.5 (chosen)** | **11.92%** | 0.456 |
| C&W c₀ = 10, lr 0.02 / 0.05 / 0.1 / 0.2 / 0.5 | 7.46% / 10.33% / 11.47% / 11.68% / 11.81% | 0.495 / 0.634 / 0.763 / 0.779 / 0.997 |
| C&W c₀ = 100, lr 0.02 / 0.05 / 0.1 / 0.2 / 0.5 | 7.43% / 10.32% / 11.46% / 11.65% / 11.78% | 0.496 / 0.637 / 0.766 / 0.796 / 1.000 |

Prim-PGD is insensitive to α (18.8–19.0%). Prim-C&W is insensitive to c₀ (≤ 0.3 pp) and limited
by step size and restart diversity, not by the cost/margin trade-off.

## 4. Experimental setup and seeds

- **Dataset.** CICIDS2017-DistriNet, test split. Rows are the frozen clean-correct selection of
  the canonical no-IDR campaign (`outputs/adv_campaign_noidr/cicids2017_distrinet/selection.json`;
  random rule, selection seed 42): **800 flows per victim × class**, identical across methods,
  budgets, and seeds. The SHA-256 of the sample ids was re-verified at load.
- **Victims.** `mlp`, `cnn` (`outputs/cicids2017distrinet/models/*_category.pt`) and
  `ft_transformer` (`outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt`),
  category heads, frozen.
- **Classes.** DoS, DDoS, Recon, BruteForce.
- **Budgets.** p50, p75, unbounded (envelope).
- **Attack seeds.** 42, 123, 2024 (the existing protocol). `deterministic_runtime(seed)` is called
  per cell. Seeds affect only random restarts. Prim-C&W is deterministic by construction, so its
  seed SD is 0.
- **Grid.** 3 victims × 4 classes × 3 budgets × 3 seeds × 3 methods = **324 cells**, 259,200
  attacked flow instances. No seed or sample reduction.
- **Environment.** Conda env `thesis`: Python 3.11.15, torch 2.5.1+cu121, one RTX 4070 Ti SUPER.
  The test run took 1,641 s wall-clock. `CUBLAS_WORKSPACE_CONFIG` was unset, so cuBLAS kernels
  are not bit-deterministic (warned). Observed seed SDs are ≤ 0.14 pp.
- **Statistics.** The paired unit is one source flow. Primary tests use reference seed 42 and pool
  the 4 classes within a victim (n = 3,200); victims and budgets are never pooled. Seeds 123/2024
  are reported as replications.
  - McNemar: exact binomial if b+c < 25, else continuity-corrected χ².
  - Newcombe square-and-add paired 95% CI.
  - Holm correction within each family: 27 primary tests, 108 per-class tests, 9 omnibus tests.
  - Cochran's Q omnibus across the 3 methods.


## 5. Results

Rates are over the clean-correct eligible set (800 flows per victim × class). "valid ASR" is the primary outcome. The complete per-cell raw table (every model/class/budget/seed/method, with attempted samples, successes, ASR, validator-valid ASR, cost, padding/delay/shape statistics, evaluations, median evaluations to first success, failure classes, unique successes, iterations, restarts, runtime) is Appendix A.

### 5.1 Aggregated results (classes pooled within victim; mean ± SD over attack seeds)

| Victim | Budget | Method | n/seed | seeds | valid ASR | raw targeted | SP-ASR | Hybrid R=2 prefix | cost mean/median | evals mean/median | median evals→1st success | fail invalid | fail exhausted | no headroom | runtime s (sum/seed) | ms/flow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | p50 | Hybrid | 3200 | 3 | 6.31% ± 0.00% | 6.31% | 6.09% | 6.31% | 1.48/1.5 | 188.8/255 | 58 | 0.00% | 68.94% | 24.75% | 9.7 | 3.02 |
| mlp | p50 | Prim-PGD | 3200 | 3 | 6.31% ± 0.00% | 6.31% | 6.09% | — | 1.44/1.49 | 190.6/253 | 35 | 0.00% | 68.94% | 24.75% | 9.9 | 3.08 |
| mlp | p50 | Prim-C&W | 3200 | 3 | 6.31% ± 0.00% | 6.31% | 6.09% | — | 1.16/1.25 | 190.6/253 | 9 | 0.00% | 68.94% | 24.75% | 10.5 | 3.28 |
| mlp | p75 | Hybrid | 3200 | 3 | 11.06% ± 0.00% | 11.06% | 10.53% | 11.06% | 1.38/1.4 | 188.3/255 | 69 | 0.00% | 64.19% | 24.75% | 9.0 | 2.80 |
| mlp | p75 | Prim-PGD | 3200 | 3 | 11.06% ± 0.00% | 11.06% | 10.53% | — | 1.32/1.32 | 190.6/253 | 33 | 0.00% | 64.19% | 24.75% | 9.7 | 3.03 |
| mlp | p75 | Prim-C&W | 3200 | 3 | 11.00% ± 0.00% | 11.00% | 10.53% | — | 1.06/1.06 | 190.6/253 | 7 | 0.00% | 64.25% | 24.75% | 10.3 | 3.21 |
| mlp | unb | Hybrid | 3200 | 3 | 41.90% ± 0.11% | 41.90% | 40.19% | 41.83% | 0.805/0.79 | 188.5/255 | 85 | 0.00% | 33.35% | 24.75% | 9.3 | 2.91 |
| mlp | unb | Prim-PGD | 3200 | 3 | 42.04% ± 0.02% | 42.04% | 40.90% | — | 0.665/0.595 | 190.6/253 | 15 | 0.00% | 33.21% | 24.75% | 9.9 | 3.08 |
| mlp | unb | Prim-C&W | 3200 | 3 | 41.09% ± 0.00% | 41.09% | 40.19% | — | 0.474/0.416 | 190.6/253 | 3 | 0.00% | 34.16% | 24.75% | 10.6 | 3.32 |
| cnn | p50 | Hybrid | 3200 | 3 | 13.11% ± 0.05% | 13.11% | 12.58% | 12.81% | 1.34/1.39 | 188.7/255 | 88.33 | 0.00% | 62.10% | 24.78% | 9.3 | 2.90 |
| cnn | p50 | Prim-PGD | 3200 | 3 | 13.04% ± 0.04% | 13.04% | 12.61% | — | 1.3/1.34 | 190.6/253 | 91 | 0.00% | 62.18% | 24.78% | 9.2 | 2.88 |
| cnn | p50 | Prim-C&W | 3200 | 3 | 2.84% ± 0.00% | 2.84% | 2.66% | — | 0.638/0.476 | 190.6/253 | 5 | 0.00% | 72.38% | 24.78% | 9.6 | 3.00 |
| cnn | p75 | Hybrid | 3200 | 3 | 36.20% ± 0.10% | 36.20% | 20.79% | 35.52% | 1.48/1.61 | 188.4/255 | 95 | 0.00% | 39.02% | 24.78% | 9.0 | 2.81 |
| cnn | p75 | Prim-PGD | 3200 | 3 | 36.09% ± 0.14% | 36.09% | 21.02% | — | 1.44/1.58 | 190.6/253 | 87 | 0.00% | 39.13% | 24.78% | 9.7 | 3.04 |
| cnn | p75 | Prim-C&W | 3200 | 3 | 14.31% ± 0.00% | 14.31% | 4.72% | — | 1.22/1.48 | 190.6/253 | 5 | 0.00% | 60.91% | 24.78% | 10.1 | 3.15 |
| cnn | unb | Hybrid | 3200 | 3 | 70.49% ± 0.05% | 70.49% | 43.42% | 70.45% | 0.95/0.983 | 185.5/255 | 91 | 0.02% | 4.71% | 24.78% | 8.6 | 2.68 |
| cnn | unb | Prim-PGD | 3200 | 3 | 70.51% ± 0.02% | 70.51% | 44.19% | — | 0.675/0.595 | 190.6/253 | 25 | 0.00% | 4.71% | 24.78% | 9.5 | 2.98 |
| cnn | unb | Prim-C&W | 3200 | 3 | 43.38% ± 0.00% | 43.38% | 28.06% | — | 0.44/0.367 | 190.6/253 | 3 | 0.00% | 31.84% | 24.78% | 10.1 | 3.16 |
| ft_transformer | p50 | Hybrid | 3200 | 3 | 0.16% ± 0.00% | 0.16% | 0.03% | 0.16% | 0.224/0.2 | 188.8/255 | 5 | 0.00% | 75.09% | 24.75% | 40.2 | 12.56 |
| ft_transformer | p50 | Prim-PGD | 3200 | 3 | 0.16% ± 0.00% | 0.16% | 0.03% | — | 0.286/0.316 | 190.6/253 | 9 | 0.00% | 75.09% | 24.75% | 42.3 | 13.23 |
| ft_transformer | p50 | Prim-C&W | 3200 | 3 | 0.16% ± 0.00% | 0.16% | 0.03% | — | 0.14/0.167 | 190.6/253 | 3 | 0.00% | 75.09% | 24.75% | 42.8 | 13.39 |
| ft_transformer | p75 | Hybrid | 3200 | 3 | 0.44% ± 0.00% | 0.44% | 0.03% | 0.44% | 0.237/0.196 | 188.2/255 | 17.5 | 0.00% | 74.81% | 24.75% | 37.5 | 11.73 |
| ft_transformer | p75 | Prim-PGD | 3200 | 3 | 0.44% ± 0.00% | 0.44% | 0.03% | — | 0.292/0.205 | 190.6/253 | 9 | 0.00% | 74.81% | 24.75% | 42.2 | 13.20 |
| ft_transformer | p75 | Prim-C&W | 3200 | 3 | 0.44% ± 0.00% | 0.44% | 0.03% | — | 0.232/0.199 | 190.6/253 | 3 | 0.00% | 74.81% | 24.75% | 42.6 | 13.32 |
| ft_transformer | unb | Hybrid | 3200 | 3 | 0.88% ± 0.00% | 0.88% | 0.06% | 0.88% | 0.476/0.215 | 188.8/255 | 19 | 0.00% | 74.38% | 24.75% | 34.8 | 10.86 |
| ft_transformer | unb | Prim-PGD | 3200 | 3 | 0.88% ± 0.00% | 0.88% | 0.06% | — | 0.401/0.204 | 190.6/253 | 7 | 0.00% | 74.38% | 24.75% | 42.4 | 13.24 |
| ft_transformer | unb | Prim-C&W | 3200 | 3 | 0.38% ± 0.00% | 0.38% | 0.06% | — | 0.633/0.674 | 190.6/253 | 5 | 0.00% | 74.87% | 24.75% | 43.0 | 13.43 |

### 5.2 Per-class results (mean over attack seeds)

| Victim | Class | Budget | Method | n | movable | valid ASR | SD | raw | SP-ASR | median cost | mean evals | unique succ. | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | DoS | p50 | Hybrid | 800 | 799 | 14.50% | 0.00% | 14.50% | 14.50% | 1.59 | 252 | 0.0 | 2.81 |
| mlp | DoS | p50 | Prim-PGD | 800 | 799 | 14.50% | 0.00% | 14.50% | 14.50% | 1.55 | 252.7 | 0.0 | 3.03 |
| mlp | DoS | p50 | Prim-C&W | 800 | 799 | 14.50% | 0.00% | 14.50% | 14.50% | 1.48 | 252.7 | 0.0 | 3.27 |
| mlp | DoS | p75 | Hybrid | 800 | 799 | 28.88% | 0.00% | 28.88% | 28.88% | 1.46 | 251.2 | 0.0 | 2.51 |
| mlp | DoS | p75 | Prim-PGD | 800 | 799 | 28.88% | 0.00% | 28.88% | 28.88% | 1.42 | 252.7 | 0.0 | 2.78 |
| mlp | DoS | p75 | Prim-C&W | 800 | 799 | 28.88% | 0.00% | 28.88% | 28.88% | 1.3 | 252.7 | 0.0 | 2.94 |
| mlp | DoS | unb | Hybrid | 800 | 799 | 89.12% | 0.00% | 89.12% | 89.00% | 0.695 | 251.6 | 0.0 | 2.30 |
| mlp | DoS | unb | Prim-PGD | 800 | 799 | 89.12% | 0.00% | 89.12% | 89.08% | 0.516 | 252.7 | 0.0 | 2.72 |
| mlp | DoS | unb | Prim-C&W | 800 | 799 | 89.00% | 0.00% | 89.00% | 88.88% | 0.275 | 252.7 | 0.0 | 2.82 |
| mlp | DDoS | p50 | Hybrid | 800 | 800 | 9.88% | 0.00% | 9.88% | 9.88% | 1.42 | 244.9 | 0.0 | 2.43 |
| mlp | DDoS | p50 | Prim-PGD | 800 | 800 | 9.88% | 0.00% | 9.88% | 9.88% | 1.37 | 253 | 0.0 | 2.42 |
| mlp | DDoS | p50 | Prim-C&W | 800 | 800 | 9.88% | 0.00% | 9.88% | 9.88% | 0.639 | 253 | 0.0 | 2.66 |
| mlp | DDoS | p75 | Hybrid | 800 | 800 | 13.25% | 0.00% | 13.25% | 13.25% | 1.28 | 245.9 | 0.0 | 2.37 |
| mlp | DDoS | p75 | Prim-PGD | 800 | 800 | 13.25% | 0.00% | 13.25% | 13.25% | 1.25 | 253 | 0.0 | 2.37 |
| mlp | DDoS | p75 | Prim-C&W | 800 | 800 | 13.25% | 0.00% | 13.25% | 13.25% | 0.53 | 253 | 0.0 | 2.56 |
| mlp | DDoS | unb | Hybrid | 800 | 800 | 74.96% | 0.44% | 74.96% | 71.75% | 0.861 | 246.3 | 1.0 | 2.29 |
| mlp | DDoS | unb | Prim-PGD | 800 | 800 | 75.54% | 0.07% | 75.54% | 74.50% | 0.709 | 253 | 3.7 | 2.50 |
| mlp | DDoS | unb | Prim-C&W | 800 | 800 | 73.00% | 0.00% | 73.00% | 71.88% | 0.546 | 253 | 0.7 | 2.73 |
| mlp | Recon | p50 | Hybrid | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.857 | 0.0 | 2.10 |
| mlp | Recon | p50 | Prim-PGD | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.06 |
| mlp | Recon | p50 | Prim-C&W | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.13 |
| mlp | Recon | p75 | Hybrid | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.857 | 0.0 | 2.13 |
| mlp | Recon | p75 | Prim-PGD | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.08 |
| mlp | Recon | p75 | Prim-C&W | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.17 |
| mlp | Recon | unb | Hybrid | 800 | 9 | 1.00% | 0.00% | 1.00% | 0.00% | 0.1 | 3.857 | 0.0 | 2.11 |
| mlp | Recon | unb | Prim-PGD | 800 | 9 | 1.00% | 0.00% | 1.00% | 0.00% | 0.05 | 3.835 | 0.0 | 2.01 |
| mlp | Recon | unb | Prim-C&W | 800 | 9 | 0.50% | 0.00% | 0.50% | 0.00% | 0.5 | 3.835 | 0.0 | 2.19 |
| mlp | BruteForce | p50 | Hybrid | 800 | 800 | 0.88% | 0.00% | 0.88% | 0.00% | 0.806 | 254.4 | 0.0 | 2.32 |
| mlp | BruteForce | p50 | Prim-PGD | 800 | 800 | 0.88% | 0.00% | 0.88% | 0.00% | 0.699 | 253 | 0.0 | 2.35 |
| mlp | BruteForce | p50 | Prim-C&W | 800 | 800 | 0.88% | 0.00% | 0.88% | 0.00% | 0.37 | 253 | 0.0 | 2.43 |
| mlp | BruteForce | p75 | Hybrid | 800 | 800 | 2.12% | 0.00% | 2.12% | 0.00% | 0.846 | 252.3 | 0.0 | 1.96 |
| mlp | BruteForce | p75 | Prim-PGD | 800 | 800 | 2.12% | 0.00% | 2.12% | 0.00% | 0.712 | 253 | 0.0 | 2.46 |
| mlp | BruteForce | p75 | Prim-C&W | 800 | 800 | 1.88% | 0.00% | 1.88% | 0.00% | 0.521 | 253 | 0.0 | 2.61 |
| mlp | BruteForce | unb | Hybrid | 800 | 800 | 2.50% | 0.00% | 2.50% | 0.00% | 0.835 | 252.3 | 0.0 | 2.63 |
| mlp | BruteForce | unb | Prim-PGD | 800 | 800 | 2.50% | 0.00% | 2.50% | 0.00% | 0.127 | 253 | 0.0 | 2.65 |
| mlp | BruteForce | unb | Prim-C&W | 800 | 800 | 1.88% | 0.00% | 1.88% | 0.00% | 0.973 | 253 | 0.0 | 2.89 |
| cnn | DoS | p50 | Hybrid | 800 | 799 | 23.46% | 0.19% | 23.46% | 23.17% | 1.65 | 252 | 3.7 | 2.24 |
| cnn | DoS | p50 | Prim-PGD | 800 | 799 | 23.17% | 0.14% | 23.17% | 22.83% | 1.64 | 252.7 | 1.3 | 2.27 |
| cnn | DoS | p50 | Prim-C&W | 800 | 799 | 4.50% | 0.00% | 4.50% | 4.37% | 0.37 | 252.7 | 0.0 | 2.37 |
| cnn | DoS | p75 | Hybrid | 800 | 799 | 49.54% | 0.40% | 49.54% | 49.21% | 1.55 | 251.1 | 4.7 | 2.03 |
| cnn | DoS | p75 | Prim-PGD | 800 | 799 | 49.38% | 0.37% | 49.38% | 49.04% | 1.48 | 252.7 | 3.3 | 2.31 |
| cnn | DoS | p75 | Prim-C&W | 800 | 799 | 10.00% | 0.00% | 10.00% | 9.88% | 0.722 | 252.7 | 0.0 | 2.39 |
| cnn | DoS | unb | Hybrid | 800 | 799 | 92.62% | 0.00% | 92.62% | 91.25% | 0.951 | 251.4 | 0.0 | 1.86 |
| cnn | DoS | unb | Prim-PGD | 800 | 799 | 92.62% | 0.00% | 92.62% | 91.88% | 0.3 | 252.7 | 0.0 | 2.29 |
| cnn | DoS | unb | Prim-C&W | 800 | 799 | 63.38% | 0.00% | 63.38% | 63.25% | 0.351 | 252.7 | 0.0 | 2.37 |
| cnn | DDoS | p50 | Hybrid | 800 | 800 | 28.12% | 0.00% | 28.12% | 27.17% | 1.24 | 244.9 | 0.0 | 2.23 |
| cnn | DDoS | p50 | Prim-PGD | 800 | 800 | 28.12% | 0.00% | 28.12% | 27.63% | 1.23 | 253 | 0.0 | 2.23 |
| cnn | DDoS | p50 | Prim-C&W | 800 | 800 | 6.25% | 0.00% | 6.25% | 6.25% | 0.495 | 253 | 0.0 | 2.34 |
| cnn | DDoS | p75 | Hybrid | 800 | 800 | 35.87% | 0.00% | 35.87% | 33.96% | 1.22 | 245.9 | 0.0 | 2.41 |
| cnn | DDoS | p75 | Prim-PGD | 800 | 800 | 35.87% | 0.00% | 35.87% | 35.04% | 1.23 | 253 | 0.0 | 2.58 |
| cnn | DDoS | p75 | Prim-C&W | 800 | 800 | 9.50% | 0.00% | 9.50% | 9.00% | 0.627 | 253 | 0.0 | 2.67 |
| cnn | DDoS | unb | Hybrid | 800 | 800 | 88.33% | 0.19% | 88.33% | 82.42% | 0.922 | 234 | 0.0 | 2.38 |
| cnn | DDoS | unb | Prim-PGD | 800 | 800 | 88.42% | 0.07% | 88.42% | 84.88% | 0.729 | 253 | 0.3 | 2.70 |
| cnn | DDoS | unb | Prim-C&W | 800 | 800 | 52.00% | 0.00% | 52.00% | 49.00% | 0.901 | 253 | 0.0 | 2.92 |
| cnn | Recon | p50 | Hybrid | 800 | 8 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.54 | 0.0 | 2.34 |
| cnn | Recon | p50 | Prim-PGD | 800 | 8 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.52 | 0.0 | 2.27 |
| cnn | Recon | p50 | Prim-C&W | 800 | 8 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.52 | 0.0 | 2.34 |
| cnn | Recon | p75 | Hybrid | 800 | 8 | 0.12% | 0.00% | 0.12% | 0.00% | 0.908 | 3.54 | 0.0 | 2.39 |
| cnn | Recon | p75 | Prim-PGD | 800 | 8 | 0.12% | 0.00% | 0.12% | 0.00% | 0.912 | 3.52 | 0.0 | 2.37 |
| cnn | Recon | p75 | Prim-C&W | 800 | 8 | 0.12% | 0.00% | 0.12% | 0.00% | 0.782 | 3.52 | 0.0 | 2.45 |
| cnn | Recon | unb | Hybrid | 800 | 8 | 1.00% | 0.00% | 1.00% | 0.00% | 0.05 | 3.54 | 0.0 | 2.42 |
| cnn | Recon | unb | Prim-PGD | 800 | 8 | 1.00% | 0.00% | 1.00% | 0.00% | 0.0575 | 3.52 | 0.0 | 2.27 |
| cnn | Recon | unb | Prim-C&W | 800 | 8 | 0.25% | 0.00% | 0.25% | 0.00% | 0.997 | 3.52 | 0.0 | 2.45 |
| cnn | BruteForce | p50 | Hybrid | 800 | 800 | 0.88% | 0.00% | 0.88% | 0.00% | 0.808 | 254.4 | 0.0 | 2.47 |
| cnn | BruteForce | p50 | Prim-PGD | 800 | 800 | 0.88% | 0.00% | 0.88% | 0.00% | 0.791 | 253 | 0.0 | 2.45 |
| cnn | BruteForce | p50 | Prim-C&W | 800 | 800 | 0.63% | 0.00% | 0.63% | 0.00% | 0.652 | 253 | 0.0 | 2.56 |
| cnn | BruteForce | p75 | Hybrid | 800 | 800 | 59.25% | 0.00% | 59.25% | 0.00% | 1.67 | 252.9 | 1.3 | 2.14 |
| cnn | BruteForce | p75 | Prim-PGD | 800 | 800 | 59.00% | 0.33% | 59.00% | 0.00% | 1.64 | 253 | 0.0 | 2.48 |
| cnn | BruteForce | p75 | Prim-C&W | 800 | 800 | 37.62% | 0.00% | 37.62% | 0.00% | 1.5 | 253 | 0.0 | 2.57 |
| cnn | BruteForce | unb | Hybrid | 800 | 800 | 100.00% | 0.00% | 100.00% | 0.00% | 1.06 | 252.9 | 0.0 | 1.91 |
| cnn | BruteForce | unb | Prim-PGD | 800 | 800 | 100.00% | 0.00% | 100.00% | 0.00% | 0.65 | 253 | 0.0 | 2.29 |
| cnn | BruteForce | unb | Prim-C&W | 800 | 800 | 57.88% | 0.00% | 57.88% | 0.00% | 0.244 | 253 | 0.0 | 2.38 |
| ft_transformer | DoS | p50 | Hybrid | 800 | 799 | 0.25% | 0.00% | 0.25% | 0.12% | 0.185 | 252.3 | 0.0 | 11.73 |
| ft_transformer | DoS | p50 | Prim-PGD | 800 | 799 | 0.25% | 0.00% | 0.25% | 0.12% | 0.167 | 252.7 | 0.0 | 13.33 |
| ft_transformer | DoS | p50 | Prim-C&W | 800 | 799 | 0.25% | 0.00% | 0.25% | 0.12% | 0.0961 | 252.7 | 0.0 | 13.39 |
| ft_transformer | DoS | p75 | Hybrid | 800 | 799 | 0.38% | 0.00% | 0.38% | 0.12% | 0.2 | 251.4 | 0.0 | 11.54 |
| ft_transformer | DoS | p75 | Prim-PGD | 800 | 799 | 0.38% | 0.00% | 0.38% | 0.12% | 0.15 | 252.7 | 0.0 | 13.35 |
| ft_transformer | DoS | p75 | Prim-C&W | 800 | 799 | 0.38% | 0.00% | 0.38% | 0.12% | 0.148 | 252.7 | 0.0 | 13.32 |
| ft_transformer | DoS | unb | Hybrid | 800 | 799 | 0.50% | 0.00% | 0.50% | 0.25% | 0.446 | 251.7 | 0.0 | 10.87 |
| ft_transformer | DoS | unb | Prim-PGD | 800 | 799 | 0.50% | 0.00% | 0.50% | 0.25% | 0.293 | 252.7 | 0.0 | 13.33 |
| ft_transformer | DoS | unb | Prim-C&W | 800 | 799 | 0.50% | 0.00% | 0.50% | 0.25% | 0.242 | 252.7 | 0.0 | 13.47 |
| ft_transformer | DDoS | p50 | Hybrid | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 244.9 | 0.0 | 12.95 |
| ft_transformer | DDoS | p50 | Prim-PGD | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 253 | 0.0 | 13.36 |
| ft_transformer | DDoS | p50 | Prim-C&W | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 253 | 0.0 | 13.62 |
| ft_transformer | DDoS | p75 | Hybrid | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 245.9 | 0.0 | 12.88 |
| ft_transformer | DDoS | p75 | Prim-PGD | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 253 | 0.0 | 13.27 |
| ft_transformer | DDoS | p75 | Prim-C&W | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 253 | 0.0 | 13.42 |
| ft_transformer | DDoS | unb | Hybrid | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 248 | 0.0 | 10.65 |
| ft_transformer | DDoS | unb | Prim-PGD | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 253 | 0.0 | 13.39 |
| ft_transformer | DDoS | unb | Prim-C&W | 800 | 800 | 0.00% | 0.00% | 0.00% | 0.00% | — | 253 | 0.0 | 13.64 |
| ft_transformer | Recon | p50 | Hybrid | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.857 | 0.0 | 2.55 |
| ft_transformer | Recon | p50 | Prim-PGD | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.42 |
| ft_transformer | Recon | p50 | Prim-C&W | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.50 |
| ft_transformer | Recon | p75 | Hybrid | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.857 | 0.0 | 2.43 |
| ft_transformer | Recon | p75 | Prim-PGD | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.48 |
| ft_transformer | Recon | p75 | Prim-C&W | 800 | 9 | 0.00% | 0.00% | 0.00% | 0.00% | — | 3.835 | 0.0 | 2.51 |
| ft_transformer | Recon | unb | Hybrid | 800 | 9 | 0.75% | 0.00% | 0.75% | 0.00% | 0.0455 | 3.857 | 0.0 | 2.46 |
| ft_transformer | Recon | unb | Prim-PGD | 800 | 9 | 0.75% | 0.00% | 0.75% | 0.00% | 0.048 | 3.835 | 0.0 | 2.43 |
| ft_transformer | Recon | unb | Prim-C&W | 800 | 9 | 0.12% | 0.00% | 0.12% | 0.00% | 0.706 | 3.835 | 0.0 | 2.51 |
| ft_transformer | BruteForce | p50 | Hybrid | 800 | 800 | 0.38% | 0.00% | 0.38% | 0.00% | 0.25 | 254.1 | 0.0 | 12.98 |
| ft_transformer | BruteForce | p50 | Prim-PGD | 800 | 800 | 0.38% | 0.00% | 0.38% | 0.00% | 0.35 | 253 | 0.0 | 13.22 |
| ft_transformer | BruteForce | p50 | Prim-C&W | 800 | 800 | 0.38% | 0.00% | 0.38% | 0.00% | 0.167 | 253 | 0.0 | 13.33 |
| ft_transformer | BruteForce | p75 | Hybrid | 800 | 800 | 1.38% | 0.00% | 1.38% | 0.00% | 0.192 | 251.7 | 0.0 | 10.69 |
| ft_transformer | BruteForce | p75 | Prim-PGD | 800 | 800 | 1.38% | 0.00% | 1.38% | 0.00% | 0.205 | 253 | 0.0 | 13.13 |
| ft_transformer | BruteForce | p75 | Prim-C&W | 800 | 800 | 1.38% | 0.00% | 1.38% | 0.00% | 0.205 | 253 | 0.0 | 13.37 |
| ft_transformer | BruteForce | unb | Hybrid | 800 | 800 | 2.25% | 0.00% | 2.25% | 0.00% | 0.237 | 251.7 | 0.0 | 10.78 |
| ft_transformer | BruteForce | unb | Prim-PGD | 800 | 800 | 2.25% | 0.00% | 2.25% | 0.00% | 0.204 | 253 | 0.0 | 13.21 |
| ft_transformer | BruteForce | unb | Prim-C&W | 800 | 800 | 0.88% | 0.00% | 0.88% | 0.00% | 0.726 | 253 | 0.0 | 13.36 |

### 5.3 ASR at matched evaluation budgets (from per-flow first-success index, seed mean)

| Victim | Budget | Method | ASR@1 | ASR@8 | ASR@16 | ASR@32 | ASR@64 | ASR@128 | ASR@256 |
|---|---|---|---|---|---|---|---|---|---|
| mlp | p50 | Hybrid | 0.00% | 0.03% | 0.59% | 2.62% | 4.09% | 6.31% | 6.31% |
| mlp | p50 | Prim-PGD | 0.00% | 0.00% | 0.12% | 2.41% | 6.31% | 6.31% | 6.31% |
| mlp | p50 | Prim-C&W | 0.00% | 3.09% | 5.16% | 6.09% | 6.12% | 6.31% | 6.31% |
| mlp | p75 | Hybrid | 0.00% | 0.03% | 1.62% | 3.47% | 4.44% | 11.06% | 11.06% |
| mlp | p75 | Prim-PGD | 0.00% | 0.09% | 0.44% | 5.25% | 11.00% | 11.05% | 11.06% |
| mlp | p75 | Prim-C&W | 0.00% | 6.88% | 10.56% | 10.97% | 11.00% | 11.00% | 11.00% |
| mlp | unb | Hybrid | 0.00% | 0.22% | 0.28% | 0.34% | 0.50% | 41.13% | 41.90% |
| mlp | unb | Prim-PGD | 0.00% | 6.41% | 23.59% | 39.81% | 41.84% | 41.96% | 42.04% |
| mlp | unb | Prim-C&W | 0.00% | 34.22% | 38.19% | 39.03% | 40.31% | 40.88% | 41.09% |
| cnn | p50 | Hybrid | 0.00% | 0.06% | 0.69% | 1.78% | 2.66% | 8.84% | 13.11% |
| cnn | p50 | Prim-PGD | 0.00% | 0.06% | 0.25% | 1.75% | 2.75% | 12.52% | 13.04% |
| cnn | p50 | Prim-C&W | 0.00% | 2.19% | 2.31% | 2.31% | 2.31% | 2.75% | 2.84% |
| cnn | p75 | Hybrid | 0.00% | 0.06% | 1.03% | 2.53% | 3.22% | 21.73% | 36.20% |
| cnn | p75 | Prim-PGD | 0.00% | 0.19% | 0.78% | 2.50% | 13.34% | 34.72% | 36.09% |
| cnn | p75 | Prim-C&W | 0.00% | 12.22% | 12.59% | 12.78% | 12.84% | 14.09% | 14.31% |
| cnn | unb | Hybrid | 0.00% | 0.12% | 0.28% | 0.75% | 1.78% | 39.75% | 70.51% |
| cnn | unb | Prim-PGD | 0.00% | 7.75% | 28.62% | 43.59% | 46.16% | 70.44% | 70.51% |
| cnn | unb | Prim-C&W | 0.00% | 40.34% | 41.78% | 42.22% | 42.31% | 43.03% | 43.38% |
| ft_transformer | p50 | Hybrid | 0.00% | 0.09% | 0.12% | 0.12% | 0.16% | 0.16% | 0.16% |
| ft_transformer | p50 | Prim-PGD | 0.00% | 0.03% | 0.16% | 0.16% | 0.16% | 0.16% | 0.16% |
| ft_transformer | p50 | Prim-C&W | 0.00% | 0.12% | 0.16% | 0.16% | 0.16% | 0.16% | 0.16% |
| ft_transformer | p75 | Hybrid | 0.00% | 0.09% | 0.22% | 0.38% | 0.44% | 0.44% | 0.44% |
| ft_transformer | p75 | Prim-PGD | 0.00% | 0.12% | 0.41% | 0.41% | 0.44% | 0.44% | 0.44% |
| ft_transformer | p75 | Prim-C&W | 0.00% | 0.41% | 0.44% | 0.44% | 0.44% | 0.44% | 0.44% |
| ft_transformer | unb | Hybrid | 0.00% | 0.25% | 0.41% | 0.56% | 0.59% | 0.88% | 0.88% |
| ft_transformer | unb | Prim-PGD | 0.00% | 0.50% | 0.59% | 0.78% | 0.88% | 0.88% | 0.88% |
| ft_transformer | unb | Prim-C&W | 0.00% | 0.25% | 0.34% | 0.34% | 0.34% | 0.38% | 0.38% |

## 6. Paired statistical tests

### 6.1 Paired tests, primary family (reference seed, classes pooled within victim, Holm over 27 tests)

| Victim | Budget | A vs B | n | A | B | A only | B only | Δ [95% CI] (pp) | p | Holm p | replications Δpp (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | p50 | Hybrid vs Prim-PGD | 3200 | 6.31% | 6.31% | 0 | 0 | +0.00 [-0.12, +0.12] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| mlp | p50 | Hybrid vs Prim-C&W | 3200 | 6.31% | 6.31% | 0 | 0 | +0.00 [-0.12, +0.12] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| mlp | p50 | Prim-PGD vs Prim-C&W | 3200 | 6.31% | 6.31% | 0 | 0 | +0.00 [-0.12, +0.12] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| mlp | p75 | Hybrid vs Prim-PGD | 3200 | 11.06% | 11.06% | 0 | 0 | +0.00 [-0.11, +0.11] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| mlp | p75 | Hybrid vs Prim-C&W | 3200 | 11.06% | 11.00% | 2 | 0 | +0.06 [-0.08, +0.21] | 5.00e-01 | 1.00e+00 | s123: +0.06 (5.0e-01); s2024: +0.06 (5.0e-01) |
| mlp | p75 | Prim-PGD vs Prim-C&W | 3200 | 11.06% | 11.00% | 2 | 0 | +0.06 [-0.08, +0.21] | 5.00e-01 | 1.00e+00 | s123: +0.06 (5.0e-01); s2024: +0.06 (5.0e-01) |
| mlp | unb | Hybrid vs Prim-PGD | 3200 | 42.00% | 42.03% | 2 | 3 | -0.03 [-0.18, +0.12] | 1.00e+00 | 1.00e+00 | s123: -0.16 (1.2e-01); s2024: -0.25 (3.9e-02) |
| mlp | unb | Hybrid vs Prim-C&W | 3200 | 42.00% | 41.09% | 30 | 1 | +0.91 [+0.56, +1.25] | 4.93e-07 | 1.04e-05 | s123: +0.81 (1.8e-05); s2024: +0.69 (2.1e-04) |
| mlp | unb | Prim-PGD vs Prim-C&W | 3200 | 42.03% | 41.09% | 32 | 2 | +0.94 [+0.58, +1.30] | 6.58e-07 | 1.32e-05 | s123: +0.97 (1.8e-07); s2024: +0.94 (3.0e-07) |
| cnn | p50 | Hybrid vs Prim-PGD | 3200 | 13.06% | 13.06% | 3 | 3 | +0.00 [-0.18, +0.18] | 1.00e+00 | 1.00e+00 | s123: +0.16 (6.2e-02); s2024: +0.06 (6.2e-01) |
| cnn | p50 | Hybrid vs Prim-C&W | 3200 | 13.06% | 2.84% | 327 | 0 | +10.22 [+9.20, +11.31] | 1.18e-72 | 2.71e-71 | s123: +10.31 (2.6e-73); s2024: +10.28 (4.3e-73) |
| cnn | p50 | Prim-PGD vs Prim-C&W | 3200 | 13.06% | 2.84% | 327 | 0 | +10.22 [+9.20, +11.31] | 1.18e-72 | 2.71e-71 | s123: +10.16 (3.2e-72); s2024: +10.22 (1.2e-72) |
| cnn | p75 | Hybrid vs Prim-PGD | 3200 | 36.16% | 36.03% | 8 | 4 | +0.13 [-0.10, +0.35] | 3.88e-01 | 1.00e+00 | s123: +0.31 (2.0e-03); s2024: -0.12 (2.9e-01) |
| cnn | p75 | Hybrid vs Prim-C&W | 3200 | 36.16% | 14.31% | 699 | 0 | +21.84 [+20.41, +23.28] | 1.34e-153 | 3.36e-152 | s123: +22.00 (1.1e-154); s2024: +21.81 (2.2e-153) |
| cnn | p75 | Prim-PGD vs Prim-C&W | 3200 | 36.03% | 14.31% | 696 | 1 | +21.72 [+20.29, +23.15] | 2.69e-152 | 6.46e-151 | s123: +21.69 (4.4e-152); s2024: +21.94 (3.0e-154) |
| cnn | unb | Hybrid vs Prim-PGD | 3200 | 70.50% | 70.50% | 1 | 1 | +0.00 [-0.12, +0.12] | 1.00e+00 | 1.00e+00 | s123: +0.03 (1.0e+00); s2024: -0.09 (2.5e-01) |
| cnn | unb | Hybrid vs Prim-C&W | 3200 | 70.50% | 43.38% | 869 | 1 | +27.12 [+25.57, +28.65] | 6.54e-190 | 1.77e-188 | s123: +27.16 (1.5e-190); s2024: +27.06 (4.8e-189) |
| cnn | unb | Prim-PGD vs Prim-C&W | 3200 | 70.50% | 43.38% | 869 | 1 | +27.12 [+25.57, +28.65] | 6.54e-190 | 1.77e-188 | s123: +27.12 (6.5e-190); s2024: +27.16 (1.5e-190) |
| ft_transformer | p50 | Hybrid vs Prim-PGD | 3200 | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| ft_transformer | p50 | Hybrid vs Prim-C&W | 3200 | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| ft_transformer | p50 | Prim-PGD vs Prim-C&W | 3200 | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| ft_transformer | p75 | Hybrid vs Prim-PGD | 3200 | 0.44% | 0.44% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| ft_transformer | p75 | Hybrid vs Prim-C&W | 3200 | 0.44% | 0.44% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| ft_transformer | p75 | Prim-PGD vs Prim-C&W | 3200 | 0.44% | 0.44% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| ft_transformer | unb | Hybrid vs Prim-PGD | 3200 | 0.88% | 0.88% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 | s123: +0.00 (1.0e+00); s2024: +0.00 (1.0e+00) |
| ft_transformer | unb | Hybrid vs Prim-C&W | 3200 | 0.88% | 0.38% | 16 | 0 | +0.50 [+0.26, +0.81] | 3.05e-05 | 5.80e-04 | s123: +0.50 (3.1e-05); s2024: +0.50 (3.1e-05) |
| ft_transformer | unb | Prim-PGD vs Prim-C&W | 3200 | 0.88% | 0.38% | 16 | 0 | +0.50 [+0.26, +0.81] | 3.05e-05 | 5.80e-04 | s123: +0.50 (3.1e-05); s2024: +0.50 (3.1e-05) |

### 6.2 Cochran's Q omnibus (3 methods, reference seed, Holm-adjusted)

| Victim | Budget | Q | df | p | Holm p |
|---|---|---|---|---|---|
| mlp | p50 | 0.00 | 2 | 1.00e+00 | 1.00e+00 |
| mlp | p75 | 4.00 | 2 | 1.35e-01 | 5.41e-01 |
| mlp | unb | 49.77 | 2 | 1.56e-11 | 9.34e-11 |
| cnn | p50 | 648.05 | 2 | 1.89e-141 | 1.32e-140 |
| cnn | p75 | 1380.17 | 2 | 1.99e-300 | 1.59e-299 |
| cnn | unb | 1730.02 | 2 | 0.00e+00 | 0.00e+00 |
| ft_transformer | p50 | 0.00 | 2 | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | 0.00 | 2 | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | 32.00 | 2 | 1.13e-07 | 5.63e-07 |

### 6.3 Success overlap (reference seed, valid successes, classes pooled)

| Victim | Budget | n | all 3 | H∧P only | H∧C only | P∧C only | Hybrid only | PGD only | C&W only | none | H&P | H&C | P&C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp | p50 | 3200 | 202 | 0 | 0 | 0 | 0 | 0 | 0 | 2998 | 202 | 202 | 202 |
| mlp | p75 | 3200 | 352 | 2 | 0 | 0 | 0 | 0 | 0 | 2846 | 354 | 352 | 352 |
| mlp | unb | 3200 | 1312 | 30 | 2 | 1 | 0 | 2 | 0 | 1853 | 1342 | 1314 | 1313 |
| cnn | p50 | 3200 | 91 | 324 | 0 | 0 | 3 | 3 | 0 | 2779 | 415 | 91 | 91 |
| cnn | p75 | 3200 | 457 | 692 | 1 | 0 | 7 | 4 | 0 | 2039 | 1149 | 458 | 457 |
| cnn | unb | 3200 | 1386 | 869 | 1 | 1 | 0 | 0 | 0 | 943 | 2255 | 1387 | 1387 |
| ft_transformer | p50 | 3200 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 3195 | 5 | 5 | 5 |
| ft_transformer | p75 | 3200 | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 3186 | 14 | 14 | 14 |
| ft_transformer | unb | 3200 | 12 | 16 | 0 | 0 | 0 | 0 | 0 | 3172 | 28 | 12 | 12 |

### 6.4 Per-class paired tests (reference seed, Holm over 108 tests)

| Victim | Budget | Class | A vs B | A | B | A only | B only | Δ [95% CI] (pp) | p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp | p50 | DoS | Hybrid vs Prim-PGD | 14.50% | 14.50% | 0 | 0 | +0.00 [-0.42, +0.42] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | DDoS | Hybrid vs Prim-PGD | 9.88% | 9.88% | 0 | 0 | +0.00 [-0.45, +0.45] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | Recon | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | BruteForce | Hybrid vs Prim-PGD | 0.88% | 0.88% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | DoS | Hybrid vs Prim-C&W | 14.50% | 14.50% | 0 | 0 | +0.00 [-0.42, +0.42] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | DDoS | Hybrid vs Prim-C&W | 9.88% | 9.88% | 0 | 0 | +0.00 [-0.45, +0.45] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | Recon | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | BruteForce | Hybrid vs Prim-C&W | 0.88% | 0.88% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | DoS | Prim-PGD vs Prim-C&W | 14.50% | 14.50% | 0 | 0 | +0.00 [-0.42, +0.42] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | DDoS | Prim-PGD vs Prim-C&W | 9.88% | 9.88% | 0 | 0 | +0.00 [-0.45, +0.45] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | Recon | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| mlp | p50 | BruteForce | Prim-PGD vs Prim-C&W | 0.88% | 0.88% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | DoS | Hybrid vs Prim-PGD | 28.88% | 28.88% | 0 | 0 | +0.00 [-0.32, +0.32] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | DDoS | Hybrid vs Prim-PGD | 13.25% | 13.25% | 0 | 0 | +0.00 [-0.43, +0.43] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | Recon | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | BruteForce | Hybrid vs Prim-PGD | 2.12% | 2.12% | 0 | 0 | +0.00 [-0.52, +0.52] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | DoS | Hybrid vs Prim-C&W | 28.88% | 28.88% | 0 | 0 | +0.00 [-0.32, +0.32] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | DDoS | Hybrid vs Prim-C&W | 13.25% | 13.25% | 0 | 0 | +0.00 [-0.43, +0.43] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | Recon | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | BruteForce | Hybrid vs Prim-C&W | 2.12% | 1.88% | 2 | 0 | +0.25 [-0.33, +0.91] | 5.00e-01 | 1.00e+00 |
| mlp | p75 | DoS | Prim-PGD vs Prim-C&W | 28.88% | 28.88% | 0 | 0 | +0.00 [-0.32, +0.32] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | DDoS | Prim-PGD vs Prim-C&W | 13.25% | 13.25% | 0 | 0 | +0.00 [-0.43, +0.43] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | Recon | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | BruteForce | Prim-PGD vs Prim-C&W | 2.12% | 1.88% | 2 | 0 | +0.25 [-0.33, +0.91] | 5.00e-01 | 1.00e+00 |
| mlp | unb | DoS | Hybrid vs Prim-PGD | 89.12% | 89.12% | 0 | 0 | +0.00 [-0.45, +0.45] | 1.00e+00 | 1.00e+00 |
| mlp | unb | DDoS | Hybrid vs Prim-PGD | 75.38% | 75.50% | 2 | 3 | -0.12 [-0.77, +0.52] | 1.00e+00 | 1.00e+00 |
| mlp | unb | Recon | Hybrid vs Prim-PGD | 1.00% | 1.00% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| mlp | unb | BruteForce | Hybrid vs Prim-PGD | 2.50% | 2.50% | 0 | 0 | +0.00 [-0.52, +0.52] | 1.00e+00 | 1.00e+00 |
| mlp | unb | DoS | Hybrid vs Prim-C&W | 89.12% | 89.00% | 1 | 0 | +0.12 [-0.38, +0.64] | 1.00e+00 | 1.00e+00 |
| mlp | unb | DDoS | Hybrid vs Prim-C&W | 75.38% | 73.00% | 20 | 1 | +2.38 [+1.23, +3.55] | 2.10e-05 | 1.93e-03 |
| mlp | unb | Recon | Hybrid vs Prim-C&W | 1.00% | 0.50% | 4 | 0 | +0.50 [-0.11, +1.31] | 1.25e-01 | 1.00e+00 |
| mlp | unb | BruteForce | Hybrid vs Prim-C&W | 2.50% | 1.88% | 5 | 0 | +0.63 [-0.05, +1.45] | 6.25e-02 | 1.00e+00 |
| mlp | unb | DoS | Prim-PGD vs Prim-C&W | 89.12% | 89.00% | 1 | 0 | +0.12 [-0.38, +0.64] | 1.00e+00 | 1.00e+00 |
| mlp | unb | DDoS | Prim-PGD vs Prim-C&W | 75.50% | 73.00% | 22 | 2 | +2.50 [+1.28, +3.75] | 3.59e-05 | 3.27e-03 |
| mlp | unb | Recon | Prim-PGD vs Prim-C&W | 1.00% | 0.50% | 4 | 0 | +0.50 [-0.11, +1.31] | 1.25e-01 | 1.00e+00 |
| mlp | unb | BruteForce | Prim-PGD vs Prim-C&W | 2.50% | 1.88% | 5 | 0 | +0.63 [-0.05, +1.45] | 6.25e-02 | 1.00e+00 |
| cnn | p50 | DoS | Hybrid vs Prim-PGD | 23.25% | 23.25% | 3 | 3 | +0.00 [-0.70, +0.70] | 1.00e+00 | 1.00e+00 |
| cnn | p50 | DDoS | Hybrid vs Prim-PGD | 28.12% | 28.12% | 0 | 0 | +0.00 [-0.32, +0.32] | 1.00e+00 | 1.00e+00 |
| cnn | p50 | Recon | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| cnn | p50 | BruteForce | Hybrid vs Prim-PGD | 0.88% | 0.88% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| cnn | p50 | DoS | Hybrid vs Prim-C&W | 23.25% | 4.50% | 150 | 0 | +18.75 [+16.11, +21.56] | 4.73e-34 | 4.44e-32 |
| cnn | p50 | DDoS | Hybrid vs Prim-C&W | 28.12% | 6.25% | 175 | 0 | +21.88 [+19.05, +24.82] | 1.63e-39 | 1.60e-37 |
| cnn | p50 | Recon | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| cnn | p50 | BruteForce | Hybrid vs Prim-C&W | 0.88% | 0.62% | 2 | 0 | +0.25 [-0.32, +0.94] | 5.00e-01 | 1.00e+00 |
| cnn | p50 | DoS | Prim-PGD vs Prim-C&W | 23.25% | 4.50% | 150 | 0 | +18.75 [+16.11, +21.56] | 4.73e-34 | 4.44e-32 |
| cnn | p50 | DDoS | Prim-PGD vs Prim-C&W | 28.12% | 6.25% | 175 | 0 | +21.88 [+19.05, +24.82] | 1.63e-39 | 1.60e-37 |
| cnn | p50 | Recon | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| cnn | p50 | BruteForce | Prim-PGD vs Prim-C&W | 0.88% | 0.62% | 2 | 0 | +0.25 [-0.32, +0.94] | 5.00e-01 | 1.00e+00 |
| cnn | p75 | DoS | Hybrid vs Prim-PGD | 49.38% | 49.00% | 7 | 4 | +0.38 [-0.47, +1.22] | 5.49e-01 | 1.00e+00 |
| cnn | p75 | DDoS | Hybrid vs Prim-PGD | 35.88% | 35.88% | 0 | 0 | +0.00 [-0.28, +0.28] | 1.00e+00 | 1.00e+00 |
| cnn | p75 | Recon | Hybrid vs Prim-PGD | 0.12% | 0.12% | 0 | 0 | +0.00 [-0.54, +0.54] | 1.00e+00 | 1.00e+00 |
| cnn | p75 | BruteForce | Hybrid vs Prim-PGD | 59.25% | 59.13% | 1 | 0 | +0.12 [-0.23, +0.48] | 1.00e+00 | 1.00e+00 |
| cnn | p75 | DoS | Hybrid vs Prim-C&W | 49.38% | 10.00% | 315 | 0 | +39.38 [+35.93, +42.72] | 4.84e-70 | 5.13e-68 |
| cnn | p75 | DDoS | Hybrid vs Prim-C&W | 35.88% | 9.50% | 211 | 0 | +26.38 [+23.32, +29.46] | 2.26e-47 | 2.26e-45 |
| cnn | p75 | Recon | Hybrid vs Prim-C&W | 0.12% | 0.12% | 0 | 0 | +0.00 [-0.54, +0.54] | 1.00e+00 | 1.00e+00 |
| cnn | p75 | BruteForce | Hybrid vs Prim-C&W | 59.25% | 37.62% | 173 | 0 | +21.63 [+18.72, +24.44] | 4.46e-39 | 4.28e-37 |
| cnn | p75 | DoS | Prim-PGD vs Prim-C&W | 49.00% | 10.00% | 312 | 0 | +39.00 [+35.56, +42.34] | 2.18e-69 | 2.29e-67 |
| cnn | p75 | DDoS | Prim-PGD vs Prim-C&W | 35.88% | 9.50% | 211 | 0 | +26.38 [+23.32, +29.46] | 2.26e-47 | 2.26e-45 |
| cnn | p75 | Recon | Prim-PGD vs Prim-C&W | 0.12% | 0.12% | 0 | 0 | +0.00 [-0.54, +0.54] | 1.00e+00 | 1.00e+00 |
| cnn | p75 | BruteForce | Prim-PGD vs Prim-C&W | 59.13% | 37.62% | 173 | 1 | +21.50 [+18.58, +24.33] | 1.97e-38 | 1.87e-36 |
| cnn | unb | DoS | Hybrid vs Prim-PGD | 92.62% | 92.62% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| cnn | unb | DDoS | Hybrid vs Prim-PGD | 88.38% | 88.38% | 1 | 1 | +0.00 [-0.56, +0.56] | 1.00e+00 | 1.00e+00 |
| cnn | unb | Recon | Hybrid vs Prim-PGD | 1.00% | 1.00% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| cnn | unb | BruteForce | Hybrid vs Prim-PGD | 100.00% | 100.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| cnn | unb | DoS | Hybrid vs Prim-C&W | 92.62% | 63.38% | 234 | 0 | +29.25 [+26.10, +32.43] | 2.18e-52 | 2.22e-50 |
| cnn | unb | DDoS | Hybrid vs Prim-C&W | 88.38% | 52.00% | 292 | 1 | +36.38 [+32.97, +39.69] | 2.21e-64 | 2.29e-62 |
| cnn | unb | Recon | Hybrid vs Prim-C&W | 1.00% | 0.25% | 6 | 0 | +0.75 [+0.09, +1.66] | 3.12e-02 | 1.00e+00 |
| cnn | unb | BruteForce | Hybrid vs Prim-C&W | 100.00% | 57.88% | 337 | 0 | +42.12 [+38.72, +45.58] | 7.82e-75 | 8.44e-73 |
| cnn | unb | DoS | Prim-PGD vs Prim-C&W | 92.62% | 63.38% | 234 | 0 | +29.25 [+26.10, +32.43] | 2.18e-52 | 2.22e-50 |
| cnn | unb | DDoS | Prim-PGD vs Prim-C&W | 88.38% | 52.00% | 292 | 1 | +36.38 [+32.97, +39.69] | 2.21e-64 | 2.29e-62 |
| cnn | unb | Recon | Prim-PGD vs Prim-C&W | 1.00% | 0.25% | 6 | 0 | +0.75 [+0.09, +1.66] | 3.12e-02 | 1.00e+00 |
| cnn | unb | BruteForce | Prim-PGD vs Prim-C&W | 100.00% | 57.88% | 337 | 0 | +42.12 [+38.72, +45.58] | 7.82e-75 | 8.44e-73 |
| ft_transformer | p50 | DoS | Hybrid vs Prim-PGD | 0.25% | 0.25% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | DDoS | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | Recon | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | BruteForce | Hybrid vs Prim-PGD | 0.38% | 0.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | DoS | Hybrid vs Prim-C&W | 0.25% | 0.25% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | DDoS | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | Recon | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | BruteForce | Hybrid vs Prim-C&W | 0.38% | 0.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | DoS | Prim-PGD vs Prim-C&W | 0.25% | 0.25% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | DDoS | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | Recon | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | BruteForce | Prim-PGD vs Prim-C&W | 0.38% | 0.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | DoS | Hybrid vs Prim-PGD | 0.38% | 0.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | DDoS | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | Recon | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | BruteForce | Hybrid vs Prim-PGD | 1.38% | 1.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | DoS | Hybrid vs Prim-C&W | 0.38% | 0.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | DDoS | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | Recon | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | BruteForce | Hybrid vs Prim-C&W | 1.38% | 1.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | DoS | Prim-PGD vs Prim-C&W | 0.38% | 0.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | DDoS | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | Recon | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | BruteForce | Prim-PGD vs Prim-C&W | 1.38% | 1.38% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | DoS | Hybrid vs Prim-PGD | 0.50% | 0.50% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | DDoS | Hybrid vs Prim-PGD | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | Recon | Hybrid vs Prim-PGD | 0.75% | 0.75% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | BruteForce | Hybrid vs Prim-PGD | 2.25% | 2.25% | 0 | 0 | +0.00 [-0.52, +0.52] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | DoS | Hybrid vs Prim-C&W | 0.50% | 0.50% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | DDoS | Hybrid vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | Recon | Hybrid vs Prim-C&W | 0.75% | 0.12% | 5 | 0 | +0.62 [-0.01, +1.49] | 6.25e-02 | 1.00e+00 |
| ft_transformer | unb | BruteForce | Hybrid vs Prim-C&W | 2.25% | 0.88% | 11 | 0 | +1.37 [+0.57, +2.46] | 9.77e-04 | 8.79e-02 |
| ft_transformer | unb | DoS | Prim-PGD vs Prim-C&W | 0.50% | 0.50% | 0 | 0 | +0.00 [-0.53, +0.53] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | DDoS | Prim-PGD vs Prim-C&W | 0.00% | 0.00% | 0 | 0 | +0.00 [-0.48, +0.48] | 1.00e+00 | 1.00e+00 |
| ft_transformer | unb | Recon | Prim-PGD vs Prim-C&W | 0.75% | 0.12% | 5 | 0 | +0.62 [-0.01, +1.49] | 6.25e-02 | 1.00e+00 |
| ft_transformer | unb | BruteForce | Prim-PGD vs Prim-C&W | 2.25% | 0.88% | 11 | 0 | +1.37 [+0.57, +2.46] | 9.77e-04 | 8.79e-02 |

## 7. Plots

All plots are under `outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/`.

| Plot | Path |
|---|---|
| ASR by optimizer / model / class / budget (seed mean ± SD) | `asr_by_optimizer_model_class_budget.png` |
| ASR vs classifier-evaluation budget (anytime curves, log-x) | `asr_vs_evaluation_budget.png` |
| Perturbation cost vs ASR | `cost_vs_asr.png` |
| Query cost vs ASR | `evals_vs_asr.png` |
| Runtime vs ASR | `runtime_vs_asr.png` |
| Pairwise success overlap (Venn regions, seed 42) | `success_overlap.png` |
| Per-class results (heatmaps) | `per_class_asr_heatmap.png` |

![ASR by optimizer](outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/asr_by_optimizer_model_class_budget.png)
![ASR vs evaluations](outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/asr_vs_evaluation_budget.png)
![Cost vs ASR](outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/cost_vs_asr.png)
![Evaluations vs ASR](outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/evals_vs_asr.png)
![Runtime vs ASR](outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/runtime_vs_asr.png)
![Success overlap](outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/success_overlap.png)
![Per-class heatmap](outputs/primattack_optimizer_ablation/cicids2017_distrinet/analysis/plots/per_class_asr_heatmap.png)

## 8. Interpretation — why methods succeed or fail

**1. Hybrid and Prim-PGD solve essentially the same flows at B = 256.**
- Across the 9 victim×budget cells (seed 42), the Hybrid-vs-PGD discordant pairs are at most 8
  vs 4 out of 3,200 (cnn p75).
- No Hybrid-vs-PGD contrast is significant after Holm; every CI lies within ±0.35 pp.
- Seed replications give one nominal difference in each direction and no consistent sign:
  cnn p75 s123 +0.31 pp (p = 0.002, Hybrid better); mlp unb s2024 −0.25 pp (p = 0.039, PGD
  better).
- The overlap tables show the "Hybrid ∧ PGD only" region carries almost all of Hybrid's advantage
  over C&W. "Hybrid only" is ≤ 7 flows per cell.

Exact padding enumeration is Hybrid's distinctive component, and it contributes few final
successes. Seed-42 winning sources on cnn: `exact-padding` 5/415 (p50), 10/1,149 (p75),
66/2,256 (unb). The integer padding set is tiny (⌊p_hi⌋ ≤ 78). PGD's continuous `q_p` coordinate,
realized with rounding at every step, covers it anyway. The exception is ft_transformer: 13/14
p75 successes come from padding, but PGD finds the same 14.

**2. Hybrid is the least query-efficient of the three.**
- Hybrid must spend up to ⌊p_hi⌋ evaluations (78 in the unbounded box) on padding before its
  first timing move. At matched small budgets it therefore lags badly (§5.3):
  - cnn unb ASR@64: Hybrid 1.78%, PGD 46.16%, C&W 42.31%;
  - cnn unb ASR@128: Hybrid 39.75%, PGD 70.44%;
  - mlp unb ASR@64: Hybrid 0.50%, PGD 41.84%.
- Parity arrives only near 256.
- The anytime curves jump at about 1 + ⌊p_hi⌋ and again about 80 evaluations later. That is
  where Hybrid's first random restart begins. For CNN, restart 1 produces more first successes
  than restart 0 (seed 42, cnn p75: 660 vs 470). The "adaptive-clean" start from the best padding
  point is a weaker initialization than a uniform draw.
- Median evaluations to first success (aggregate table): Hybrid 58–95 on mlp/cnn, PGD 15–91,
  C&W 3–9.
- Runtime per flow is nearly identical for all three: 2.7–3.3 ms (MLP/CNN), 10.9–13.4 ms (FT).
  Hybrid is marginally cheaper because early padding exits lower its mean evaluations to
  185–189, vs 190.6 for PGD/C&W.

**3. Hybrid finds the most expensive successes.** Median normalized cost of successes, unbounded:
- cnn: Hybrid 0.98, PGD 0.60, C&W 0.37;
- mlp: 0.79, 0.60, 0.42.

Neither Hybrid nor PGD has a cost term in its gradient objective. Hybrid's restart 0 starts from
the best (often maximal) padding value, and sign steps do not walk padding back down. PGD starts
from the clean flow. C&W's explicit `cost + c·hinge` objective gives the cheapest attacks but
fewer of them.

**4. Why Prim-C&W loses.** C&W is significantly worse on every CNN budget and on mlp/ft
unbounded:
- cnn p50 2.84% vs 13.06% (−10.2 pp);
- cnn p75 14.31% vs 36.16% (−21.8 pp);
- cnn unb 43.38% vs 70.50% (−27.1 pp);
- Holm p ≤ 1e−71.

Its successes arrive early (median first success 3–9 evaluations), on flows an Adam step from
the clean point already solves. Afterwards the search stalls:
- every stage restarts deterministically from `q = 0`, so the only diversity across stages is `c`;
- validation shows `c` barely matters (Adam normalizes the gradient scale);
- the hinge switches off as soon as the relaxed margin crosses −κ = 0, and the cost gradient pulls
  the iterate back toward the boundary, where integer rounding flips outcomes.

The CNN's piecewise relaxation makes a single deterministic trajectory especially brittle.
Examples: cnn DoS unb 63.4% vs 92.6%, cnn BruteForce unb 57.9% vs 100%. On the MLP, which is
nearly monotone in the controls, the three methods tie except at unbounded.

**5. The victim dominates the optimizer.**
- ft_transformer stays ≤ 0.88% valid ASR for every method and budget; mlp reaches 42% and cnn 70%
  in the unbounded box.
- Recon is untouchable: 99% of Recon flows have no integer headroom in either control, so the
  24.75% `no_headroom` share per victim is almost entirely Recon.
- Failures are otherwise `exhausted`. `invalid` is ≈ 0 on CICIDS2017, where
  `hybrid_valid ≈ targeted` for primitive-realized flows (the F1 gap is a CICIDS2018 phenomenon).

**6. Seeds.** Seed SDs are 0.00–0.14 pp: seeds only reseed random restarts, and C&W is
deterministic.

**7. SP-ASR** follows valid ASR for DoS/DDoS. It is lower on CNN at p75/unbounded because
BruteForce successes, which are 59–100% there, are `NOT_FULLY_TESTABLE` by design.


## 9. Comparison against historical PrimAttack optimizers

**Pre-fix Hybrid (canonical campaign, same flows, same seeds; §6/§9 tables).** The ablation
Hybrid (F1 + F2 + budget-filled restarts) is ≥ the canonical campaign's `prim_search_joint_*`
on every cell, with only one comparator-only flow (cnn unb). Holm-significant gains at seed 42:
- cnn p50: +0.44 pp;
- cnn p75: +0.88 pp;
- mlp unb: +0.56 pp.

The "Hybrid R=2 prefix" column isolates the fixes at the canonical 2-restart schedule. cnn p75:
35.21% (campaign seed mean) → 35.52% (fixes only) → 36.20% (budget-filled). So the fixes add
≈ +0.3 pp and the extra restarts ≈ +0.7 pp. [INFERENCE: on CICIDS2017 the validity gate rarely
binds, so most of the +0.3 pp is attributed to the F2 surrogate floor; the two fixes were not run
separately.]

**One-draw random-feasible control (campaign, same flows).** Hybrid is higher by:
- mlp: +5.6 / +8.8 / +17.7 pp (p50 / p75 / unb);
- cnn: +9.8 / +30.8 / +20.4 pp;
- p ≤ 1e−39.

This control has a 1-evaluation budget (F4), so it bounds what directed search adds; it is not a
budget-matched random search.

**Replaced Adam/sigmoid `(p, α)` optimizer** (`outputs/full_adv_eval`, `prim_opt_joint_*`). It
was run on the legacy **head** selection, so it is not paired with this ablation's rows. Its joint
valid ASR was:

| Victim | p50 | p75 | unb |
|---|---|---|---|
| mlp | 0.09% | 0.15% | 0.51% |
| cnn | 1.04% | 1.14% | 2.59% |
| ft_transformer | 4.47% | 4.67% | 5.19% |

The paired old-vs-v2 comparison on those head rows (`PRIMATTACK_V2_OPTIMIZER_COMPARISON.md`)
found v2 search ≥ old everywhere, e.g. cnn p75 joint 34.91% vs 1.19% and mlp unb 23.22% vs 0.50%.
All three methods here, including Prim-C&W, far exceed the old optimizer on MLP/CNN.

The FT gap (≈ 5% historical vs ≤ 0.9% here) is a **selection** effect. The head rows are a
different, time-ordered subset; the paired historical comparison shows v2 ≥ old on FT as well.
It is not an optimizer regression.


### 9.1 Historical: ablation Hybrid (validity-aware, budget-filled) vs canonical campaign rows on the same flows

| Victim | Budget | Seed | Comparator | Hybrid | Comparator ASR | Hybrid only | Comp. only | Δ [95% CI] (pp) | p | Holm p (primary seed) |
|---|---|---|---|---|---|---|---|---|---|---|
| mlp | p50 | 42 | Pre-fix Hybrid (campaign) | 6.31% | 6.22% | 3 | 0 | +0.09 [-0.06, +0.26] | 2.50e-01 | 1.00e+00 |
| mlp | p50 | 42 | Random-feasible (1 draw, campaign) | 6.31% | 0.75% | 178 | 0 | +5.56 [+4.81, +6.41] | 3.61e-40 | 4.69e-39 |
| mlp | p50 | 123 | Pre-fix Hybrid (campaign) | 6.31% | 6.31% | 0 | 0 | +0.00 [-0.12, +0.12] | 1.00e+00 | — |
| mlp | p50 | 123 | Random-feasible (1 draw, campaign) | 6.31% | 0.75% | 178 | 0 | +5.56 [+4.81, +6.41] | 3.61e-40 | — |
| mlp | p50 | 2024 | Pre-fix Hybrid (campaign) | 6.31% | 6.25% | 2 | 0 | +0.06 [-0.08, +0.21] | 5.00e-01 | — |
| mlp | p50 | 2024 | Random-feasible (1 draw, campaign) | 6.31% | 0.69% | 180 | 0 | +5.62 [+4.87, +6.48] | 1.32e-40 | — |
| mlp | p75 | 42 | Pre-fix Hybrid (campaign) | 11.06% | 11.03% | 1 | 0 | +0.03 [-0.10, +0.16] | 1.00e+00 | 1.00e+00 |
| mlp | p75 | 42 | Random-feasible (1 draw, campaign) | 11.06% | 2.25% | 282 | 0 | +8.81 [+7.86, +9.84] | 7.50e-63 | 1.05e-61 |
| mlp | p75 | 123 | Pre-fix Hybrid (campaign) | 11.06% | 11.00% | 2 | 0 | +0.06 [-0.08, +0.21] | 5.00e-01 | — |
| mlp | p75 | 123 | Random-feasible (1 draw, campaign) | 11.06% | 2.22% | 283 | 0 | +8.84 [+7.89, +9.87] | 4.54e-63 | — |
| mlp | p75 | 2024 | Pre-fix Hybrid (campaign) | 11.06% | 11.06% | 0 | 0 | +0.00 [-0.11, +0.11] | 1.00e+00 | — |
| mlp | p75 | 2024 | Random-feasible (1 draw, campaign) | 11.06% | 2.22% | 283 | 0 | +8.84 [+7.89, +9.87] | 4.54e-63 | — |
| mlp | unb | 42 | Pre-fix Hybrid (campaign) | 42.00% | 41.44% | 18 | 0 | +0.56 [+0.30, +0.83] | 7.63e-06 | 8.39e-05 |
| mlp | unb | 42 | Random-feasible (1 draw, campaign) | 42.00% | 24.28% | 567 | 0 | +17.72 [+16.39, +19.04] | 6.86e-125 | 1.10e-123 |
| mlp | unb | 123 | Pre-fix Hybrid (campaign) | 41.91% | 41.56% | 11 | 0 | +0.34 [+0.13, +0.56] | 9.77e-04 | — |
| mlp | unb | 123 | Random-feasible (1 draw, campaign) | 41.91% | 25.25% | 533 | 0 | +16.66 [+15.36, +17.95] | 1.71e-117 | — |
| mlp | unb | 2024 | Pre-fix Hybrid (campaign) | 41.78% | 41.41% | 12 | 0 | +0.37 [+0.15, +0.60] | 4.88e-04 | — |
| mlp | unb | 2024 | Random-feasible (1 draw, campaign) | 41.78% | 24.94% | 539 | 0 | +16.84 [+15.54, +18.14] | 8.47e-119 | — |
| cnn | p50 | 42 | Pre-fix Hybrid (campaign) | 13.06% | 12.62% | 14 | 0 | +0.44 [+0.19, +0.70] | 1.22e-04 | 1.22e-03 |
| cnn | p50 | 42 | Random-feasible (1 draw, campaign) | 13.06% | 3.25% | 314 | 0 | +9.81 [+8.81, +10.88] | 8.00e-70 | 1.20e-68 |
| cnn | p50 | 123 | Pre-fix Hybrid (campaign) | 13.16% | 12.81% | 11 | 0 | +0.34 [+0.12, +0.58] | 9.77e-04 | — |
| cnn | p50 | 123 | Random-feasible (1 draw, campaign) | 13.16% | 3.34% | 314 | 0 | +9.81 [+8.81, +10.88] | 8.00e-70 | — |
| cnn | p50 | 2024 | Pre-fix Hybrid (campaign) | 13.12% | 12.62% | 16 | 0 | +0.50 [+0.24, +0.77] | 3.05e-05 | — |
| cnn | p50 | 2024 | Random-feasible (1 draw, campaign) | 13.12% | 3.19% | 318 | 0 | +9.94 [+8.93, +11.01] | 1.08e-70 | — |
| cnn | p75 | 42 | Pre-fix Hybrid (campaign) | 36.16% | 35.28% | 28 | 0 | +0.88 [+0.55, +1.21] | 3.35e-07 | 4.02e-06 |
| cnn | p75 | 42 | Random-feasible (1 draw, campaign) | 36.16% | 5.31% | 987 | 0 | +30.84 [+29.25, +32.45] | 3.27e-216 | 5.89e-215 |
| cnn | p75 | 123 | Pre-fix Hybrid (campaign) | 36.31% | 35.34% | 31 | 0 | +0.97 [+0.62, +1.32] | 7.12e-08 | — |
| cnn | p75 | 123 | Random-feasible (1 draw, campaign) | 36.31% | 5.78% | 977 | 0 | +30.53 [+28.94, +32.13] | 4.88e-214 | — |
| cnn | p75 | 2024 | Pre-fix Hybrid (campaign) | 36.12% | 35.00% | 36 | 0 | +1.13 [+0.75, +1.50] | 5.43e-09 | — |
| cnn | p75 | 2024 | Random-feasible (1 draw, campaign) | 36.12% | 5.53% | 979 | 0 | +30.59 [+29.00, +32.20] | 1.79e-214 | — |
| cnn | unb | 42 | Pre-fix Hybrid (campaign) | 70.50% | 70.47% | 2 | 1 | +0.03 [-0.10, +0.16] | 1.00e+00 | 1.00e+00 |
| cnn | unb | 42 | Random-feasible (1 draw, campaign) | 70.50% | 50.12% | 652 | 0 | +20.38 [+18.97, +21.76] | 2.23e-143 | 3.80e-142 |
| cnn | unb | 123 | Pre-fix Hybrid (campaign) | 70.53% | 70.44% | 3 | 0 | +0.09 [-0.04, +0.23] | 2.50e-01 | — |
| cnn | unb | 123 | Random-feasible (1 draw, campaign) | 70.53% | 51.03% | 624 | 0 | +19.50 [+18.12, +20.87] | 2.74e-137 | — |
| cnn | unb | 2024 | Pre-fix Hybrid (campaign) | 70.44% | 70.31% | 4 | 0 | +0.12 [-0.02, +0.27] | 1.25e-01 | — |
| cnn | unb | 2024 | Random-feasible (1 draw, campaign) | 70.44% | 49.97% | 655 | 0 | +20.47 [+19.06, +21.86] | 4.97e-144 | — |
| ft_transformer | p50 | 42 | Pre-fix Hybrid (campaign) | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | 42 | Random-feasible (1 draw, campaign) | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p50 | 123 | Pre-fix Hybrid (campaign) | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | — |
| ft_transformer | p50 | 123 | Random-feasible (1 draw, campaign) | 0.16% | 0.12% | 1 | 0 | +0.03 [-0.11, +0.19] | 1.00e+00 | — |
| ft_transformer | p50 | 2024 | Pre-fix Hybrid (campaign) | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | — |
| ft_transformer | p50 | 2024 | Random-feasible (1 draw, campaign) | 0.16% | 0.16% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | — |
| ft_transformer | p75 | 42 | Pre-fix Hybrid (campaign) | 0.44% | 0.44% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | 1.00e+00 |
| ft_transformer | p75 | 42 | Random-feasible (1 draw, campaign) | 0.44% | 0.31% | 4 | 0 | +0.13 [-0.03, +0.33] | 1.25e-01 | 1.00e+00 |
| ft_transformer | p75 | 123 | Pre-fix Hybrid (campaign) | 0.44% | 0.44% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | — |
| ft_transformer | p75 | 123 | Random-feasible (1 draw, campaign) | 0.44% | 0.34% | 3 | 0 | +0.09 [-0.06, +0.28] | 2.50e-01 | — |
| ft_transformer | p75 | 2024 | Pre-fix Hybrid (campaign) | 0.44% | 0.44% | 0 | 0 | +0.00 [-0.13, +0.13] | 1.00e+00 | — |
| ft_transformer | p75 | 2024 | Random-feasible (1 draw, campaign) | 0.44% | 0.34% | 3 | 0 | +0.09 [-0.06, +0.28] | 2.50e-01 | — |
| ft_transformer | unb | 42 | Pre-fix Hybrid (campaign) | 0.88% | 0.81% | 2 | 0 | +0.06 [-0.09, +0.23] | 5.00e-01 | 1.00e+00 |
| ft_transformer | unb | 42 | Random-feasible (1 draw, campaign) | 0.88% | 0.59% | 9 | 0 | +0.28 [+0.08, +0.53] | 3.91e-03 | 3.52e-02 |
| ft_transformer | unb | 123 | Pre-fix Hybrid (campaign) | 0.88% | 0.84% | 1 | 0 | +0.03 [-0.11, +0.18] | 1.00e+00 | — |
| ft_transformer | unb | 123 | Random-feasible (1 draw, campaign) | 0.88% | 0.59% | 9 | 0 | +0.28 [+0.08, +0.53] | 3.91e-03 | — |
| ft_transformer | unb | 2024 | Pre-fix Hybrid (campaign) | 0.88% | 0.84% | 1 | 0 | +0.03 [-0.11, +0.18] | 1.00e+00 | — |
| ft_transformer | unb | 2024 | Random-feasible (1 draw, campaign) | 0.88% | 0.59% | 9 | 0 | +0.28 [+0.08, +0.53] | 3.91e-03 | — |

### 9.2 Historical: replaced (p, α) Adam optimizer, joint mode (head selection — NOT the same flows; descriptive only)

| Victim | Budget | valid ASR mean | SD over seeds |
|---|---|---|---|
| cnn | p50 | 1.04% | 0.02% |
| cnn | p75 | 1.14% | 0.05% |
| cnn | unb | 2.59% | 0.06% |
| ft_transformer | p50 | 4.47% | 0.03% |
| ft_transformer | p75 | 4.67% | 0.02% |
| ft_transformer | unb | 5.19% | 0.03% |
| mlp | p50 | 0.09% | 0.03% |
| mlp | p75 | 0.15% | 0.02% |
| mlp | unb | 0.51% | 0.02% |

## 10. Methodological limitations

- **Scope.** CICIDS2017 only, joint mode only, one budget B = 256, three optimizers (random
  search, pattern search and CICIDS2018 were dropped by request). Conclusions about
  query-efficiency are specific to B = 256. The anytime curves cover smaller budgets but not
  larger ones.
- **Hyperparameter asymmetry.**
  - Hybrid's defaults were designed after diagnostic inspection of the test roster (historical
    post-hoc caveat).
  - The baselines were tuned only on validation, on a small grid: 4 PGD step sizes, 15 C&W
    (c₀, lr) settings.
  - Other C&W variants were not explored: more stages, randomized stage starts, κ > 0.
  - Prim-C&W is a projected-Adam adaptation (no tanh; the clean point is a box corner).
- **Budget accounting.** A surrogate forward is charged the same as a realized forward. Backward
  passes are reported but not charged. Other charging conventions (e.g. a gradient step = 3
  forwards) would shift the PGD/C&W anytime curves right.
- **The surrogate floor (F2) is a design choice.** It changes where the relaxation is
  differentiated, not what is scored. The historical canonical runs predate it.
- **Canonical driver changed.** It now uses the validity gate and the floor. Future canonical
  campaigns will differ slightly from `outputs/adv_campaign_noidr` (which was not re-run or
  overwritten).
- **Determinism.** cuBLAS/attention kernels are not bit-deterministic; 2 borderline flows flipped
  between incumbent and final evaluation (F6).
- **Statistics.**
  - Primary tests use one reference seed; seeds vary only restarts, and victims are single
    checkpoints, so there is no training-seed robustness claim.
  - Classes are pooled within a victim for the primary family, and per-class tests are reported
    separately.
  - Recon is structurally untestable (no headroom).
- **Validity.** The success predicate is `targeted ∧ hybrid_valid`. SP-ASR is reported but not
  optimized. Primitive feasibility is 1.0 by construction (hard box).
- **Global boundary.** Feature-space proxies on aggregate CICFlowMeter statistics. No PCAP is
  edited or replayed, and there is no packet realizability, malicious-functionality, or
  in-distribution claim (no VAE/IDR metric is computed, per the run policy).

## 11. Conclusion (thesis-ready)

Under an identical realized-flow attack space and an identical success predicate
(targeted→Benign ∧ validator_v2 `hybrid_valid`, scored on quantized, recomputed flows), with a
matched budget of 256 victim evaluations per flow, the proposed **Hybrid Search and Prim-PGD
jointly attain the highest validity-constrained ASR on every victim and budget**. Their seed
means differ by ≤ 0.14 pp, with no Holm-significant difference in any of 9 victim×budget cells
and all 95% CIs within ±0.35 pp. It significantly outperforms Prim-C&W wherever the victim is
attackable (CNN: +10.2, +21.8 and +27.1 pp at p50/p75/unbounded; MLP and FT-Transformer
unbounded: +0.9 and +0.5 pp; Holm p ≤ 6e−4).

All three realized-flow searches exceed the historical baselines by large margins: one random
feasible draw by up to +30.8 pp, and the replaced `(p, α)` Adam optimizer by roughly an order of
magnitude on MLP/CNN.

The ablation therefore supports the claim that PrimAttack's effectiveness comes from
**realized-flow incumbent search with projected sign-momentum steps and random restarts**.

The ablation does **not** support attributing extra evasion power to Hybrid's exact padding
enumeration. That stage yields almost no unique successes. It also makes Hybrid the least
query-efficient method at small budgets (e.g. CNN unbounded ASR@64: 1.8% vs 46.2% for Prim-PGD)
and the one with the most expensive successes. Its merit is exactness and a lower bound on
padding-only attacks, not higher ASR.

Finally, the audit shows that two latent defects affected earlier PrimAttack runs and are now
fixed:
- validator-blind candidate selection;
- a zero-gradient dead zone at zero controls.

Their measured effect on CICIDS2017 is small (≤ +1.1 pp) but significant.


---

## Appendix A — Complete raw per-cell results

Columns: `movable` = flows with a legal non-identity control; `fail inv/exh/noh` = failures due to invalidity / exhausted search / no headroom; `unique` = successes no other optimizer found on the same (victim, class, budget, seed). Cost = normalized primitive cost `p/p_hi + delay/delay_hi` over successes; p in bytes/forward packet, delay in µs.

| victim | class | budget | seed | method | attempted | movable | succ. | valid ASR | raw ASR | SP-ASR | cost mean/med | p mean/med | delay µs mean/med | shape mean/med | evals mean/med | med evals→1st | fail inv/exh/noh | unique | iters | restarts | runtime s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn | BruteForce | p50 | 42 | Prim-C&W | 800 | 800 | 5 | 0.62% | 0.62% | 0.00% | 0.503/0.652 | 4.4/0 | 5.34e+04/2.22e+03 | 0.6/1 | 253/253 | 5 | 0/795/0 | 0 | 126 | 3 | 2.60 |
| cnn | BruteForce | p50 | 42 | Hybrid | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.839/0.917 | 6.14/9 | 1.552e+05/9.55e+04 | 0.395/0.377 | 254.4/255 | 17 | 0/793/0 | 0 | 121 | 4 | 2.71 |
| cnn | BruteForce | p50 | 42 | Prim-PGD | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.792/0.917 | 6.14/8 | 1.396e+05/4.45e+04 | 0.344/0.15 | 253/253 | 37 | 0/793/0 | 0 | 126 | 3 | 2.45 |
| cnn | BruteForce | p50 | 123 | Prim-C&W | 800 | 800 | 5 | 0.62% | 0.62% | 0.00% | 0.503/0.652 | 4.4/0 | 5.34e+04/2.22e+03 | 0.6/1 | 253/253 | 5 | 0/795/0 | 0 | 126 | 3 | 2.83 |
| cnn | BruteForce | p50 | 123 | Hybrid | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.941/0.917 | 7.86/10 | 1.376e+05/5.93e+04 | 0.477/0.384 | 254.4/255 | 17 | 0/793/0 | 0 | 121 | 4 | 2.57 |
| cnn | BruteForce | p50 | 123 | Prim-PGD | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.791/0.917 | 6/7 | 1.426e+05/4.45e+04 | 0.41/0.15 | 253/253 | 37 | 0/793/0 | 0 | 126 | 3 | 2.60 |
| cnn | BruteForce | p50 | 2024 | Prim-C&W | 800 | 800 | 5 | 0.62% | 0.62% | 0.00% | 0.503/0.652 | 4.4/0 | 5.34e+04/2.22e+03 | 0.6/1 | 253/253 | 5 | 0/795/0 | 0 | 126 | 3 | 2.26 |
| cnn | BruteForce | p50 | 2024 | Hybrid | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.664/0.589 | 4.86/3 | 1.145e+05/1.024e+05 | 0.493/0.525 | 254.4/255 | 17 | 0/793/0 | 0 | 121 | 4 | 2.13 |
| cnn | BruteForce | p50 | 2024 | Prim-PGD | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.576/0.539 | 4.57/3 | 9.1e+04/4.45e+04 | 0.41/0.15 | 253/253 | 37 | 0/793/0 | 0 | 126 | 3 | 2.29 |
| cnn | BruteForce | p75 | 42 | Prim-C&W | 800 | 800 | 301 | 37.62% | 37.62% | 0.00% | 1.46/1.5 | 66.4/68 | 1.266e+06/1.293e+06 | 0.983/1 | 253/253 | 5 | 0/499/0 | 0 | 126 | 3 | 2.51 |
| cnn | BruteForce | p75 | 42 | Hybrid | 800 | 800 | 474 | 59.25% | 59.25% | 0.00% | 1.65/1.67 | 66.6/67 | 1.646e+06/1.674e+06 | 0.926/1 | 252.9/255 | 95 | 0/326/0 | 0 | 88 | 3 | 1.99 |
| cnn | BruteForce | p75 | 42 | Prim-PGD | 800 | 800 | 473 | 59.13% | 59.13% | 0.00% | 1.62/1.63 | 65.8/68 | 1.605e+06/1.632e+06 | 0.907/0.992 | 253/253 | 57 | 0/327/0 | 0 | 126 | 3 | 2.50 |
| cnn | BruteForce | p75 | 123 | Prim-C&W | 800 | 800 | 301 | 37.62% | 37.62% | 0.00% | 1.46/1.5 | 66.4/68 | 1.266e+06/1.293e+06 | 0.983/1 | 253/253 | 5 | 0/499/0 | 0 | 126 | 3 | 2.48 |
| cnn | BruteForce | p75 | 123 | Hybrid | 800 | 800 | 474 | 59.25% | 59.25% | 0.00% | 1.65/1.67 | 66.8/68 | 1.649e+06/1.660e+06 | 0.919/1 | 252.9/255 | 95 | 0/326/0 | 4 | 88 | 3 | 2.22 |
| cnn | BruteForce | p75 | 123 | Prim-PGD | 800 | 800 | 469 | 58.63% | 58.63% | 0.00% | 1.62/1.65 | 65.9/69 | 1.609e+06/1.608e+06 | 0.906/0.987 | 253/253 | 57 | 0/331/0 | 0 | 126 | 3 | 2.33 |
| cnn | BruteForce | p75 | 2024 | Prim-C&W | 800 | 800 | 301 | 37.62% | 37.62% | 0.00% | 1.46/1.5 | 66.4/68 | 1.266e+06/1.293e+06 | 0.983/1 | 253/253 | 5 | 0/499/0 | 0 | 126 | 3 | 2.72 |
| cnn | BruteForce | p75 | 2024 | Hybrid | 800 | 800 | 474 | 59.25% | 59.25% | 0.00% | 1.64/1.66 | 66.1/67 | 1.640e+06/1.649e+06 | 0.93/1 | 252.9/255 | 95 | 0/326/0 | 0 | 88 | 3 | 2.21 |
| cnn | BruteForce | p75 | 2024 | Prim-PGD | 800 | 800 | 474 | 59.25% | 59.25% | 0.00% | 1.61/1.64 | 65.4/68 | 1.604e+06/1.625e+06 | 0.911/1 | 253/253 | 57 | 0/326/0 | 0 | 126 | 3 | 2.60 |
| cnn | BruteForce | unb | 42 | Prim-C&W | 800 | 800 | 463 | 57.88% | 57.88% | 0.00% | 0.299/0.244 | 0.793/0 | 3.105e+07/2.651e+07 | 0.992/1 | 253/253 | 3 | 0/337/0 | 0 | 126 | 3 | 2.38 |
| cnn | BruteForce | unb | 42 | Hybrid | 800 | 800 | 800 | 100.00% | 100.00% | 0.00% | 1.01/1.07 | 43.1/39.5 | 4.964e+07/4.802e+07 | 0.6/0.627 | 252.9/255 | 91 | 0/0/0 | 0 | 88 | 3 | 1.90 |
| cnn | BruteForce | unb | 42 | Prim-PGD | 800 | 800 | 800 | 100.00% | 100.00% | 0.00% | 0.77/0.65 | 27.8/20 | 4.430e+07/3.370e+07 | 0.507/0.504 | 253/253 | 25 | 0/0/0 | 0 | 126 | 3 | 2.31 |
| cnn | BruteForce | unb | 123 | Prim-C&W | 800 | 800 | 463 | 57.88% | 57.88% | 0.00% | 0.299/0.244 | 0.793/0 | 3.105e+07/2.651e+07 | 0.992/1 | 253/253 | 3 | 0/337/0 | 0 | 126 | 3 | 2.36 |
| cnn | BruteForce | unb | 123 | Hybrid | 800 | 800 | 800 | 100.00% | 100.00% | 0.00% | 1.01/1.06 | 43/39 | 4.937e+07/4.825e+07 | 0.597/0.64 | 252.9/255 | 91 | 0/0/0 | 0 | 88 | 3 | 1.82 |
| cnn | BruteForce | unb | 123 | Prim-PGD | 800 | 800 | 800 | 100.00% | 100.00% | 0.00% | 0.763/0.65 | 27.5/20 | 4.408e+07/3.466e+07 | 0.507/0.533 | 253/253 | 25 | 0/0/0 | 0 | 126 | 3 | 2.30 |
| cnn | BruteForce | unb | 2024 | Prim-C&W | 800 | 800 | 463 | 57.88% | 57.88% | 0.00% | 0.299/0.244 | 0.793/0 | 3.105e+07/2.651e+07 | 0.992/1 | 253/253 | 3 | 0/337/0 | 0 | 126 | 3 | 2.39 |
| cnn | BruteForce | unb | 2024 | Hybrid | 800 | 800 | 800 | 100.00% | 100.00% | 0.00% | 1.01/1.05 | 43/39 | 4.948e+07/4.762e+07 | 0.619/0.657 | 252.9/255 | 91 | 0/0/0 | 0 | 88 | 3 | 2.02 |
| cnn | BruteForce | unb | 2024 | Prim-PGD | 800 | 800 | 800 | 100.00% | 100.00% | 0.00% | 0.77/0.65 | 28.2/20 | 4.380e+07/3.161e+07 | 0.514/0.519 | 253/253 | 25 | 0/0/0 | 0 | 126 | 3 | 2.24 |
| cnn | DDoS | p50 | 42 | Prim-C&W | 800 | 800 | 50 | 6.25% | 6.25% | 6.25% | 0.621/0.495 | 0.14/0 | 1.626e+06/1.448e+06 | 0.996/1 | 253/253 | 5 | 0/750/0 | 0 | 126 | 3 | 2.37 |
| cnn | DDoS | p50 | 42 | Hybrid | 800 | 800 | 225 | 28.12% | 28.12% | 27.00% | 1.25/1.25 | 1.16/1 | 1.612e+06/1.415e+06 | 0.798/0.879 | 244.9/255 | 85 | 0/575/0 | 0 | 126 | 4 | 2.29 |
| cnn | DDoS | p50 | 42 | Prim-PGD | 800 | 800 | 225 | 28.12% | 28.12% | 27.25% | 1.25/1.25 | 1.14/1 | 1.633e+06/1.484e+06 | 0.764/0.821 | 253/253 | 87 | 0/575/0 | 0 | 126 | 3 | 2.14 |
| cnn | DDoS | p50 | 123 | Prim-C&W | 800 | 800 | 50 | 6.25% | 6.25% | 6.25% | 0.621/0.495 | 0.14/0 | 1.626e+06/1.448e+06 | 0.996/1 | 253/253 | 5 | 0/750/0 | 0 | 126 | 3 | 2.28 |
| cnn | DDoS | p50 | 123 | Hybrid | 800 | 800 | 225 | 28.12% | 28.12% | 27.75% | 1.18/1.17 | 0.996/1 | 1.664e+06/1.407e+06 | 0.82/0.912 | 244.9/255 | 85 | 0/575/0 | 0 | 126 | 4 | 2.12 |
| cnn | DDoS | p50 | 123 | Prim-PGD | 800 | 800 | 225 | 28.12% | 28.12% | 27.88% | 1.2/1.2 | 1.09/1 | 1.588e+06/1.439e+06 | 0.793/0.848 | 253/253 | 87 | 0/575/0 | 0 | 126 | 3 | 2.33 |
| cnn | DDoS | p50 | 2024 | Prim-C&W | 800 | 800 | 50 | 6.25% | 6.25% | 6.25% | 0.621/0.495 | 0.14/0 | 1.626e+06/1.448e+06 | 0.996/1 | 253/253 | 5 | 0/750/0 | 0 | 126 | 3 | 2.37 |
| cnn | DDoS | p50 | 2024 | Hybrid | 800 | 800 | 225 | 28.12% | 28.12% | 26.75% | 1.24/1.29 | 1.09/1 | 1.676e+06/1.565e+06 | 0.81/0.873 | 244.9/255 | 85 | 0/575/0 | 0 | 126 | 4 | 2.27 |
| cnn | DDoS | p50 | 2024 | Prim-PGD | 800 | 800 | 225 | 28.12% | 28.12% | 27.75% | 1.25/1.25 | 1.17/1 | 1.592e+06/1.407e+06 | 0.806/0.864 | 253/253 | 87 | 0/575/0 | 0 | 126 | 3 | 2.20 |
| cnn | DDoS | p75 | 42 | Prim-C&W | 800 | 800 | 76 | 9.50% | 9.50% | 9.00% | 0.754/0.627 | 0.434/0 | 2.192e+06/1.720e+06 | 0.999/1 | 253/253 | 5 | 0/724/0 | 0 | 126 | 3 | 2.33 |
| cnn | DDoS | p75 | 42 | Hybrid | 800 | 800 | 287 | 35.88% | 35.88% | 33.88% | 1.25/1.22 | 1.77/2 | 2.096e+06/1.781e+06 | 0.812/0.901 | 245.9/256 | 86 | 0/513/0 | 0 | 126 | 4 | 2.20 |
| cnn | DDoS | p75 | 42 | Prim-PGD | 800 | 800 | 287 | 35.88% | 35.88% | 35.25% | 1.26/1.24 | 1.77/2 | 2.117e+06/1.859e+06 | 0.767/0.821 | 253/253 | 87 | 0/513/0 | 0 | 126 | 3 | 2.27 |
| cnn | DDoS | p75 | 123 | Prim-C&W | 800 | 800 | 76 | 9.50% | 9.50% | 9.00% | 0.754/0.627 | 0.434/0 | 2.192e+06/1.720e+06 | 0.999/1 | 253/253 | 5 | 0/724/0 | 0 | 126 | 3 | 2.49 |
| cnn | DDoS | p75 | 123 | Hybrid | 800 | 800 | 287 | 35.88% | 35.88% | 34.00% | 1.21/1.19 | 1.62/2 | 2.147e+06/1.839e+06 | 0.812/0.898 | 245.9/256 | 86 | 0/513/0 | 0 | 126 | 4 | 2.26 |
| cnn | DDoS | p75 | 123 | Prim-PGD | 800 | 800 | 287 | 35.88% | 35.88% | 35.00% | 1.23/1.22 | 1.76/2 | 2.053e+06/1.770e+06 | 0.795/0.861 | 253/253 | 87 | 0/513/0 | 0 | 126 | 3 | 2.22 |
| cnn | DDoS | p75 | 2024 | Prim-C&W | 800 | 800 | 76 | 9.50% | 9.50% | 9.00% | 0.754/0.627 | 0.434/0 | 2.192e+06/1.720e+06 | 0.999/1 | 253/253 | 5 | 0/724/0 | 0 | 126 | 3 | 3.20 |
| cnn | DDoS | p75 | 2024 | Hybrid | 800 | 800 | 287 | 35.88% | 35.88% | 34.00% | 1.25/1.25 | 1.73/2 | 2.135e+06/1.929e+06 | 0.816/0.916 | 245.9/256 | 86 | 0/513/0 | 0 | 126 | 4 | 2.77 |
| cnn | DDoS | p75 | 2024 | Prim-PGD | 800 | 800 | 287 | 35.88% | 35.88% | 34.88% | 1.26/1.25 | 1.8/2 | 2.112e+06/1.719e+06 | 0.794/0.873 | 253/253 | 87 | 0/513/0 | 0 | 126 | 3 | 3.23 |
| cnn | DDoS | unb | 42 | Prim-C&W | 800 | 800 | 416 | 52.00% | 52.00% | 49.00% | 0.811/0.901 | 33.2/38 | 2.756e+06/1.872e+06 | 0.791/1 | 253/253 | 5 | 0/384/0 | 0 | 126 | 3 | 3.17 |
| cnn | DDoS | unb | 42 | Hybrid | 800 | 800 | 707 | 88.38% | 88.38% | 81.75% | 0.934/0.929 | 37.6/34 | 2.820e+06/2.375e+06 | 0.571/0.582 | 234/255 | 161 | 1/92/0 | 0 | 88 | 3 | 2.24 |
| cnn | DDoS | unb | 42 | Prim-PGD | 800 | 800 | 707 | 88.38% | 88.38% | 85.62% | 0.808/0.747 | 32.2/27 | 2.400e+06/2.028e+06 | 0.519/0.473 | 253/253 | 29 | 0/93/0 | 0 | 126 | 3 | 2.84 |
| cnn | DDoS | unb | 123 | Prim-C&W | 800 | 800 | 416 | 52.00% | 52.00% | 49.00% | 0.811/0.901 | 33.2/38 | 2.756e+06/1.872e+06 | 0.791/1 | 253/253 | 5 | 0/384/0 | 0 | 126 | 3 | 2.78 |
| cnn | DDoS | unb | 123 | Hybrid | 800 | 800 | 708 | 88.50% | 88.50% | 83.38% | 0.939/0.936 | 38.2/34 | 2.825e+06/2.259e+06 | 0.555/0.567 | 234/255 | 161 | 0/92/0 | 0 | 88 | 3 | 2.63 |
| cnn | DDoS | unb | 123 | Prim-PGD | 800 | 800 | 707 | 88.38% | 88.38% | 84.75% | 0.807/0.731 | 31.4/26 | 2.506e+06/2.098e+06 | 0.521/0.473 | 253/253 | 29 | 0/93/0 | 0 | 126 | 3 | 2.83 |
| cnn | DDoS | unb | 2024 | Prim-C&W | 800 | 800 | 416 | 52.00% | 52.00% | 49.00% | 0.811/0.901 | 33.2/38 | 2.756e+06/1.872e+06 | 0.791/1 | 253/253 | 5 | 0/384/0 | 0 | 126 | 3 | 2.81 |
| cnn | DDoS | unb | 2024 | Hybrid | 800 | 800 | 705 | 88.12% | 88.12% | 82.12% | 0.936/0.901 | 38.4/35 | 2.774e+06/2.334e+06 | 0.588/0.628 | 234/255 | 161 | 1/94/0 | 0 | 88 | 3 | 2.26 |
| cnn | DDoS | unb | 2024 | Prim-PGD | 800 | 800 | 708 | 88.50% | 88.50% | 84.25% | 0.812/0.709 | 32.5/27 | 2.415e+06/2.019e+06 | 0.525/0.479 | 253/253 | 29 | 0/92/0 | 1 | 126 | 3 | 2.43 |
| cnn | DoS | p50 | 42 | Prim-C&W | 800 | 799 | 36 | 4.50% | 4.50% | 4.38% | 0.68/0.37 | 8.92/0 | 2.421e+05/1.599e+05 | 0.956/1 | 252.7/253 | 5 | 0/763/1 | 0 | 126 | 3 | 2.41 |
| cnn | DoS | p50 | 42 | Hybrid | 800 | 799 | 186 | 23.25% | 23.25% | 23.00% | 1.47/1.65 | 32.1/41.5 | 4.668e+05/1.080e+05 | 0.87/1 | 252/256 | 134 | 0/613/1 | 3 | 104 | 3 | 2.39 |
| cnn | DoS | p50 | 42 | Prim-PGD | 800 | 799 | 186 | 23.25% | 23.25% | 23.00% | 1.43/1.65 | 31.4/41 | 4.341e+05/1.165e+05 | 0.85/0.964 | 252.7/253 | 99 | 0/613/1 | 3 | 126 | 3 | 2.25 |
| cnn | DoS | p50 | 123 | Prim-C&W | 800 | 799 | 36 | 4.50% | 4.50% | 4.38% | 0.68/0.37 | 8.92/0 | 2.421e+05/1.599e+05 | 0.956/1 | 252.7/253 | 5 | 0/763/1 | 0 | 126 | 3 | 2.32 |
| cnn | DoS | p50 | 123 | Hybrid | 800 | 799 | 189 | 23.62% | 23.62% | 23.25% | 1.5/1.65 | 34.1/43 | 5.079e+05/1.021e+05 | 0.876/1 | 252/256 | 134 | 0/610/1 | 5 | 104 | 3 | 2.11 |
| cnn | DoS | p50 | 123 | Prim-PGD | 800 | 799 | 184 | 23.00% | 23.00% | 22.62% | 1.4/1.62 | 29/39 | 4.707e+05/1.058e+05 | 0.871/1 | 252.7/253 | 101 | 0/615/1 | 0 | 126 | 3 | 2.31 |
| cnn | DoS | p50 | 2024 | Prim-C&W | 800 | 799 | 36 | 4.50% | 4.50% | 4.38% | 0.68/0.37 | 8.92/0 | 2.421e+05/1.599e+05 | 0.956/1 | 252.7/253 | 5 | 0/763/1 | 0 | 126 | 3 | 2.38 |
| cnn | DoS | p50 | 2024 | Hybrid | 800 | 799 | 188 | 23.50% | 23.50% | 23.25% | 1.51/1.64 | 33.9/41.5 | 4.580e+05/1.064e+05 | 0.869/1 | 252/256 | 134 | 0/611/1 | 3 | 104 | 3 | 2.21 |
| cnn | DoS | p50 | 2024 | Prim-PGD | 800 | 799 | 186 | 23.25% | 23.25% | 22.88% | 1.41/1.64 | 29.8/40 | 4.617e+05/1.069e+05 | 0.841/1 | 252.7/253 | 99 | 0/613/1 | 1 | 126 | 3 | 2.23 |
| cnn | DoS | p75 | 42 | Prim-C&W | 800 | 799 | 80 | 10.00% | 10.00% | 9.88% | 0.771/0.722 | 9.15/0 | 3.742e+05/2.037e+05 | 0.976/1 | 252.7/253 | 9 | 0/719/1 | 0 | 126 | 3 | 2.43 |
| cnn | DoS | p75 | 42 | Hybrid | 800 | 799 | 395 | 49.38% | 49.38% | 49.12% | 1.46/1.54 | 38/45 | 4.395e+05/2.029e+05 | 0.859/1 | 251.1/255 | 139 | 0/404/1 | 7 | 100 | 3 | 2.01 |
| cnn | DoS | p75 | 42 | Prim-PGD | 800 | 799 | 392 | 49.00% | 49.00% | 48.75% | 1.38/1.5 | 34.3/42 | 4.153e+05/2.029e+05 | 0.827/0.921 | 252.7/253 | 97 | 0/407/1 | 4 | 126 | 3 | 2.26 |
| cnn | DoS | p75 | 123 | Prim-C&W | 800 | 799 | 80 | 10.00% | 10.00% | 9.88% | 0.771/0.722 | 9.15/0 | 3.742e+05/2.037e+05 | 0.976/1 | 252.7/253 | 9 | 0/719/1 | 0 | 126 | 3 | 2.39 |
| cnn | DoS | p75 | 123 | Hybrid | 800 | 799 | 400 | 50.00% | 50.00% | 49.50% | 1.47/1.56 | 38.9/46 | 4.413e+05/2.027e+05 | 0.866/0.986 | 251.1/255 | 141 | 0/399/1 | 5 | 100 | 3 | 1.98 |
| cnn | DoS | p75 | 123 | Prim-PGD | 800 | 799 | 395 | 49.38% | 49.38% | 49.12% | 1.36/1.5 | 33/41 | 4.195e+05/2.016e+05 | 0.836/0.936 | 252.7/253 | 97 | 0/404/1 | 0 | 126 | 3 | 2.34 |
| cnn | DoS | p75 | 2024 | Prim-C&W | 800 | 799 | 80 | 10.00% | 10.00% | 9.88% | 0.771/0.722 | 9.15/0 | 3.742e+05/2.037e+05 | 0.976/1 | 252.7/253 | 9 | 0/719/1 | 0 | 126 | 3 | 2.33 |
| cnn | DoS | p75 | 2024 | Hybrid | 800 | 799 | 394 | 49.25% | 49.25% | 49.00% | 1.46/1.56 | 38.9/46.5 | 4.312e+05/2.017e+05 | 0.858/0.999 | 251.1/255 | 139 | 0/405/1 | 2 | 100 | 3 | 2.11 |
| cnn | DoS | p75 | 2024 | Prim-PGD | 800 | 799 | 398 | 49.75% | 49.75% | 49.25% | 1.37/1.46 | 33.8/41.5 | 4.350e+05/1.953e+05 | 0.845/0.971 | 252.7/253 | 95 | 0/401/1 | 6 | 126 | 3 | 2.33 |
| cnn | DoS | unb | 42 | Prim-C&W | 800 | 799 | 507 | 63.38% | 63.38% | 63.25% | 0.262/0.351 | 0.44/0 | 5.671e+06/6.447e+06 | 0.902/1 | 252.7/253 | 5 | 0/292/1 | 0 | 126 | 3 | 2.46 |
| cnn | DoS | unb | 42 | Hybrid | 800 | 799 | 741 | 92.62% | 92.62% | 91.12% | 0.908/0.963 | 40.5/37 | 8.485e+06/7.043e+06 | 0.516/0.474 | 251.4/255 | 83 | 0/58/1 | 0 | 88 | 3 | 1.84 |
| cnn | DoS | unb | 42 | Prim-PGD | 800 | 799 | 741 | 92.62% | 92.62% | 91.75% | 0.457/0.3 | 13.5/0 | 6.097e+06/4.479e+06 | 0.34/0.25 | 252.7/253 | 11 | 0/58/1 | 0 | 126 | 3 | 2.16 |
| cnn | DoS | unb | 123 | Prim-C&W | 800 | 799 | 507 | 63.38% | 63.38% | 63.25% | 0.262/0.351 | 0.44/0 | 5.671e+06/6.447e+06 | 0.902/1 | 252.7/253 | 5 | 0/292/1 | 0 | 126 | 3 | 2.33 |
| cnn | DoS | unb | 123 | Hybrid | 800 | 799 | 741 | 92.62% | 92.62% | 91.00% | 0.909/0.956 | 40.8/38 | 8.398e+06/6.581e+06 | 0.52/0.501 | 251.4/255 | 83 | 0/58/1 | 0 | 88 | 3 | 1.92 |
| cnn | DoS | unb | 123 | Prim-PGD | 800 | 799 | 741 | 92.62% | 92.62% | 92.00% | 0.453/0.3 | 13.9/0 | 5.862e+06/4.448e+06 | 0.329/0.217 | 252.7/253 | 11 | 0/58/1 | 0 | 126 | 3 | 2.30 |
| cnn | DoS | unb | 2024 | Prim-C&W | 800 | 799 | 507 | 63.38% | 63.38% | 63.25% | 0.262/0.351 | 0.44/0 | 5.671e+06/6.447e+06 | 0.902/1 | 252.7/253 | 5 | 0/292/1 | 0 | 126 | 3 | 2.31 |
| cnn | DoS | unb | 2024 | Hybrid | 800 | 799 | 741 | 92.62% | 92.62% | 91.62% | 0.9/0.935 | 40.1/37 | 8.399e+06/6.900e+06 | 0.535/0.514 | 251.4/255 | 83 | 0/58/1 | 0 | 88 | 3 | 1.83 |
| cnn | DoS | unb | 2024 | Prim-PGD | 800 | 799 | 741 | 92.62% | 92.62% | 91.88% | 0.45/0.3 | 13/0 | 6.068e+06/4.449e+06 | 0.351/0.25 | 252.7/253 | 11 | 0/58/1 | 0 | 126 | 3 | 2.41 |
| cnn | Recon | p50 | 42 | Prim-C&W | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.52/1 | — | 0/8/792 | 0 | 126 | 3 | 2.28 |
| cnn | Recon | p50 | 42 | Hybrid | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.54/1 | — | 0/8/792 | 0 | 127 | 4 | 2.34 |
| cnn | Recon | p50 | 42 | Prim-PGD | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.52/1 | — | 0/8/792 | 0 | 126 | 3 | 2.30 |
| cnn | Recon | p50 | 123 | Prim-C&W | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.52/1 | — | 0/8/792 | 0 | 126 | 3 | 2.42 |
| cnn | Recon | p50 | 123 | Hybrid | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.54/1 | — | 0/8/792 | 0 | 127 | 4 | 2.28 |
| cnn | Recon | p50 | 123 | Prim-PGD | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.52/1 | — | 0/8/792 | 0 | 126 | 3 | 2.27 |
| cnn | Recon | p50 | 2024 | Prim-C&W | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.52/1 | — | 0/8/792 | 0 | 126 | 3 | 2.30 |
| cnn | Recon | p50 | 2024 | Hybrid | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.54/1 | — | 0/8/792 | 0 | 127 | 4 | 2.42 |
| cnn | Recon | p50 | 2024 | Prim-PGD | 800 | 8 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.52/1 | — | 0/8/792 | 0 | 126 | 3 | 2.25 |
| cnn | Recon | p75 | 42 | Prim-C&W | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.782/0.782 | 0/0 | 6.22e+03/6.22e+03 | 1/1 | 3.52/1 | 89 | 0/7/792 | 0 | 126 | 3 | 2.48 |
| cnn | Recon | p75 | 42 | Hybrid | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.779/0.779 | 0/0 | 6.2e+03/6.2e+03 | 1/1 | 3.54/1 | 173 | 0/7/792 | 0 | 127 | 4 | 2.35 |
| cnn | Recon | p75 | 42 | Prim-PGD | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.805/0.805 | 0/0 | 6.41e+03/6.41e+03 | 1/1 | 3.52/1 | 191 | 0/7/792 | 0 | 126 | 3 | 2.41 |
| cnn | Recon | p75 | 123 | Prim-C&W | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.782/0.782 | 0/0 | 6.22e+03/6.22e+03 | 1/1 | 3.52/1 | 89 | 0/7/792 | 0 | 126 | 3 | 2.31 |
| cnn | Recon | p75 | 123 | Hybrid | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.946/0.946 | 0/0 | 7.53e+03/7.53e+03 | 0.886/0.886 | 3.54/1 | 89 | 0/7/792 | 0 | 127 | 4 | 2.36 |
| cnn | Recon | p75 | 123 | Prim-PGD | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.946/0.946 | 0/0 | 7.53e+03/7.53e+03 | 0.886/0.886 | 3.52/1 | 101 | 0/7/792 | 0 | 126 | 3 | 2.27 |
| cnn | Recon | p75 | 2024 | Prim-C&W | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.782/0.782 | 0/0 | 6.22e+03/6.22e+03 | 1/1 | 3.52/1 | 89 | 0/7/792 | 0 | 126 | 3 | 2.55 |
| cnn | Recon | p75 | 2024 | Hybrid | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 1/1 | 0/0 | 7.95e+03/7.95e+03 | 1/1 | 3.54/1 | 91 | 0/7/792 | 0 | 127 | 4 | 2.47 |
| cnn | Recon | p75 | 2024 | Prim-PGD | 800 | 8 | 1 | 0.12% | 0.12% | 0.00% | 0.985/0.985 | 0/0 | 7.83e+03/7.83e+03 | 0.779/0.779 | 3.52/1 | 103 | 0/7/792 | 0 | 126 | 3 | 2.44 |
| cnn | Recon | unb | 42 | Prim-C&W | 800 | 8 | 2 | 0.25% | 0.25% | 0.00% | 0.997/0.997 | 0/0 | 1.128e+08/1.128e+08 | 1/1 | 3.52/1 | 3 | 0/6/792 | 0 | 126 | 3 | 2.46 |
| cnn | Recon | unb | 42 | Hybrid | 800 | 8 | 8 | 1.00% | 1.00% | 0.00% | 0.107/0.05 | 2.25/0 | 7.181e+06/2.699e+06 | 0.62/0.771 | 3.54/1 | 42 | 0/0/792 | 0 | 127 | 4 | 2.38 |
| cnn | Recon | unb | 42 | Prim-PGD | 800 | 8 | 8 | 1.00% | 1.00% | 0.00% | 0.0684/0.0671 | 1/0 | 3.399e+06/3.575e+06 | 0.512/0.453 | 3.52/1 | 3 | 0/0/792 | 0 | 126 | 3 | 2.29 |
| cnn | Recon | unb | 123 | Prim-C&W | 800 | 8 | 2 | 0.25% | 0.25% | 0.00% | 0.997/0.997 | 0/0 | 1.128e+08/1.128e+08 | 1/1 | 3.52/1 | 3 | 0/6/792 | 0 | 126 | 3 | 2.38 |
| cnn | Recon | unb | 123 | Hybrid | 800 | 8 | 8 | 1.00% | 1.00% | 0.00% | 0.116/0.0465 | 2.25/0 | 7.769e+06/2.493e+06 | 0.608/0.732 | 3.54/1 | 42 | 0/0/792 | 0 | 127 | 4 | 2.49 |
| cnn | Recon | unb | 123 | Prim-PGD | 800 | 8 | 8 | 1.00% | 1.00% | 0.00% | 0.0626/0.0539 | 1/0 | 3.383e+06/2.997e+06 | 0.421/0.3 | 3.52/1 | 3 | 0/0/792 | 0 | 126 | 3 | 2.42 |
| cnn | Recon | unb | 2024 | Prim-C&W | 800 | 8 | 2 | 0.25% | 0.25% | 0.00% | 0.997/0.997 | 0/0 | 1.128e+08/1.128e+08 | 1/1 | 3.52/1 | 3 | 0/6/792 | 0 | 126 | 3 | 2.52 |
| cnn | Recon | unb | 2024 | Hybrid | 800 | 8 | 8 | 1.00% | 1.00% | 0.00% | 0.112/0.0536 | 2.25/0 | 7.515e+06/2.779e+06 | 0.724/0.89 | 3.54/1 | 42 | 0/0/792 | 0 | 127 | 4 | 2.40 |
| cnn | Recon | unb | 2024 | Prim-PGD | 800 | 8 | 8 | 1.00% | 1.00% | 0.00% | 0.0631/0.0514 | 1/0 | 3.313e+06/2.796e+06 | 0.628/0.73 | 3.52/1 | 3 | 0/0/792 | 0 | 126 | 3 | 2.10 |
| ft_transformer | BruteForce | p50 | 42 | Prim-C&W | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.17/0.167 | 1.67/2 | 552/0 | 0.333/0 | 253/253 | 3 | 0/797/0 | 0 | 126 | 3 | 13.57 |
| ft_transformer | BruteForce | p50 | 42 | Hybrid | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.25/0.25 | 3/3 | 0/0 | 0/0 | 254.1/255 | 4 | 0/797/0 | 0 | 121 | 4 | 12.71 |
| ft_transformer | BruteForce | p50 | 42 | Prim-PGD | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.368/0.367 | 2/2 | 1.58e+03/758 | 0.319/0.2 | 253/253 | 9 | 0/797/0 | 0 | 126 | 3 | 13.12 |
| ft_transformer | BruteForce | p50 | 123 | Prim-C&W | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.17/0.167 | 1.67/2 | 552/0 | 0.333/0 | 253/253 | 3 | 0/797/0 | 0 | 126 | 3 | 13.21 |
| ft_transformer | BruteForce | p50 | 123 | Hybrid | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.25/0.25 | 3/3 | 0/0 | 0/0 | 254.1/255 | 4 | 0/797/0 | 0 | 121 | 4 | 13.14 |
| ft_transformer | BruteForce | p50 | 123 | Prim-PGD | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.393/0.367 | 1.67/2 | 1.74e+03/1.23e+03 | 0.416/0.2 | 253/253 | 9 | 0/797/0 | 0 | 126 | 3 | 13.13 |
| ft_transformer | BruteForce | p50 | 2024 | Prim-C&W | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.17/0.167 | 1.67/2 | 552/0 | 0.333/0 | 253/253 | 3 | 0/797/0 | 0 | 126 | 3 | 13.21 |
| ft_transformer | BruteForce | p50 | 2024 | Hybrid | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.25/0.25 | 3/3 | 0/0 | 0/0 | 254.1/255 | 4 | 0/797/0 | 0 | 121 | 4 | 13.09 |
| ft_transformer | BruteForce | p50 | 2024 | Prim-PGD | 800 | 800 | 3 | 0.38% | 0.38% | 0.00% | 0.332/0.317 | 2/2 | 1.47e+03/439 | 0.387/0.2 | 253/253 | 9 | 0/797/0 | 0 | 126 | 3 | 13.40 |
| ft_transformer | BruteForce | p75 | 42 | Prim-C&W | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.189/0.205 | 14.5/16 | 143/0 | 0.0909/0 | 253/253 | 3 | 0/789/0 | 0 | 126 | 3 | 13.69 |
| ft_transformer | BruteForce | p75 | 42 | Hybrid | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.179/0.192 | 14/15 | 0/0 | 0/0 | 251.7/255 | 16 | 0/789/0 | 0 | 88 | 3 | 10.53 |
| ft_transformer | BruteForce | p75 | 42 | Prim-PGD | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.254/0.205 | 14.8/16 | 2.13e+04/0 | 0.0636/0 | 253/253 | 9 | 0/789/0 | 0 | 126 | 3 | 13.01 |
| ft_transformer | BruteForce | p75 | 123 | Prim-C&W | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.189/0.205 | 14.5/16 | 143/0 | 0.0909/0 | 253/253 | 3 | 0/789/0 | 0 | 126 | 3 | 13.17 |
| ft_transformer | BruteForce | p75 | 123 | Hybrid | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.179/0.192 | 14/15 | 0/0 | 0/0 | 251.7/255 | 16 | 0/789/0 | 0 | 88 | 3 | 10.91 |
| ft_transformer | BruteForce | p75 | 123 | Prim-PGD | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.254/0.205 | 14.8/16 | 2.13e+04/0 | 0.0636/0 | 253/253 | 9 | 0/789/0 | 0 | 126 | 3 | 13.18 |
| ft_transformer | BruteForce | p75 | 2024 | Prim-C&W | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.189/0.205 | 14.5/16 | 143/0 | 0.0909/0 | 253/253 | 3 | 0/789/0 | 0 | 126 | 3 | 13.27 |
| ft_transformer | BruteForce | p75 | 2024 | Hybrid | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.179/0.192 | 14/15 | 0/0 | 0/0 | 251.7/255 | 16 | 0/789/0 | 0 | 88 | 3 | 10.63 |
| ft_transformer | BruteForce | p75 | 2024 | Prim-PGD | 800 | 800 | 11 | 1.38% | 1.38% | 0.00% | 0.254/0.205 | 14.8/16 | 2.13e+04/0 | 0.0636/0 | 253/253 | 9 | 0/789/0 | 0 | 126 | 3 | 13.20 |
| ft_transformer | BruteForce | unb | 42 | Prim-C&W | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.813/0.726 | 33.1/33 | 4.045e+07/5.365e+06 | 0.747/1 | 253/253 | 5 | 0/793/0 | 0 | 126 | 3 | 13.44 |
| ft_transformer | BruteForce | unb | 42 | Hybrid | 800 | 800 | 18 | 2.25% | 2.25% | 0.00% | 0.625/0.237 | 29.9/18 | 2.615e+07/0 | 0.234/0 | 251.7/255 | 19.5 | 0/782/0 | 0 | 88 | 3 | 10.64 |
| ft_transformer | BruteForce | unb | 42 | Prim-PGD | 800 | 800 | 18 | 2.25% | 2.25% | 0.00% | 0.517/0.204 | 20.6/12 | 2.716e+07/5.720e+06 | 0.412/0.15 | 253/253 | 7 | 0/782/0 | 0 | 126 | 3 | 13.01 |
| ft_transformer | BruteForce | unb | 123 | Prim-C&W | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.813/0.726 | 33.1/33 | 4.045e+07/5.365e+06 | 0.747/1 | 253/253 | 5 | 0/793/0 | 0 | 126 | 3 | 13.31 |
| ft_transformer | BruteForce | unb | 123 | Hybrid | 800 | 800 | 18 | 2.25% | 2.25% | 0.00% | 0.632/0.237 | 32.2/18.5 | 2.367e+07/0 | 0.193/0 | 251.7/255 | 19.5 | 0/782/0 | 0 | 88 | 3 | 10.98 |
| ft_transformer | BruteForce | unb | 123 | Prim-PGD | 800 | 800 | 18 | 2.25% | 2.25% | 0.00% | 0.533/0.204 | 21.1/12 | 2.793e+07/5.720e+06 | 0.366/0.15 | 253/253 | 7 | 0/782/0 | 0 | 126 | 3 | 13.47 |
| ft_transformer | BruteForce | unb | 2024 | Prim-C&W | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.813/0.726 | 33.1/33 | 4.045e+07/5.365e+06 | 0.747/1 | 253/253 | 5 | 0/793/0 | 0 | 126 | 3 | 13.32 |
| ft_transformer | BruteForce | unb | 2024 | Hybrid | 800 | 800 | 18 | 2.25% | 2.25% | 0.00% | 0.606/0.237 | 28.9/18.5 | 2.551e+07/0 | 0.218/0 | 251.7/255 | 19.5 | 0/782/0 | 0 | 88 | 3 | 10.71 |
| ft_transformer | BruteForce | unb | 2024 | Prim-PGD | 800 | 800 | 18 | 2.25% | 2.25% | 0.00% | 0.547/0.204 | 22.3/12 | 2.790e+07/5.720e+06 | 0.352/0.15 | 253/253 | 7 | 0/782/0 | 0 | 126 | 3 | 13.14 |
| ft_transformer | DDoS | p50 | 42 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.67 |
| ft_transformer | DDoS | p50 | 42 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 244.9/255 | — | 0/800/0 | 0 | 126 | 4 | 12.72 |
| ft_transformer | DDoS | p50 | 42 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.13 |
| ft_transformer | DDoS | p50 | 123 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.64 |
| ft_transformer | DDoS | p50 | 123 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 244.9/255 | — | 0/800/0 | 0 | 126 | 4 | 13.05 |
| ft_transformer | DDoS | p50 | 123 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.51 |
| ft_transformer | DDoS | p50 | 2024 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.55 |
| ft_transformer | DDoS | p50 | 2024 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 244.9/255 | — | 0/800/0 | 0 | 126 | 4 | 13.08 |
| ft_transformer | DDoS | p50 | 2024 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.44 |
| ft_transformer | DDoS | p75 | 42 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.59 |
| ft_transformer | DDoS | p75 | 42 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 245.9/256 | — | 0/800/0 | 0 | 126 | 4 | 12.77 |
| ft_transformer | DDoS | p75 | 42 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.21 |
| ft_transformer | DDoS | p75 | 123 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.12 |
| ft_transformer | DDoS | p75 | 123 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 245.9/256 | — | 0/800/0 | 0 | 126 | 4 | 13.04 |
| ft_transformer | DDoS | p75 | 123 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.08 |
| ft_transformer | DDoS | p75 | 2024 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.55 |
| ft_transformer | DDoS | p75 | 2024 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 245.9/256 | — | 0/800/0 | 0 | 126 | 4 | 12.83 |
| ft_transformer | DDoS | p75 | 2024 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.54 |
| ft_transformer | DDoS | unb | 42 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.56 |
| ft_transformer | DDoS | unb | 42 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 248/255 | — | 0/800/0 | 0 | 88 | 3 | 10.39 |
| ft_transformer | DDoS | unb | 42 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.04 |
| ft_transformer | DDoS | unb | 123 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.64 |
| ft_transformer | DDoS | unb | 123 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 248/255 | — | 0/800/0 | 0 | 88 | 3 | 10.79 |
| ft_transformer | DDoS | unb | 123 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.54 |
| ft_transformer | DDoS | unb | 2024 | Prim-C&W | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.72 |
| ft_transformer | DDoS | unb | 2024 | Hybrid | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 248/255 | — | 0/800/0 | 0 | 88 | 3 | 10.77 |
| ft_transformer | DDoS | unb | 2024 | Prim-PGD | 800 | 800 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 253/253 | — | 0/800/0 | 0 | 126 | 3 | 13.57 |
| ft_transformer | DoS | p50 | 42 | Prim-C&W | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.0961/0.0961 | 4/4 | 1.48e+03/1.48e+03 | 0.5/0.5 | 252.7/253 | 6 | 0/797/1 | 0 | 126 | 3 | 13.57 |
| ft_transformer | DoS | p50 | 42 | Hybrid | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.185/0.185 | 4/4 | 1.35e+04/1.35e+04 | 0.15/0.15 | 252.3/256 | 31.5 | 0/797/1 | 0 | 104 | 3 | 11.71 |
| ft_transformer | DoS | p50 | 42 | Prim-PGD | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.171/0.171 | 4.5/4.5 | 1.01e+04/1.01e+04 | 0.1/0.1 | 252.7/253 | 9 | 0/797/1 | 0 | 126 | 3 | 13.46 |
| ft_transformer | DoS | p50 | 123 | Prim-C&W | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.0961/0.0961 | 4/4 | 1.48e+03/1.48e+03 | 0.5/0.5 | 252.7/253 | 6 | 0/797/1 | 0 | 126 | 3 | 13.30 |
| ft_transformer | DoS | p50 | 123 | Hybrid | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.185/0.185 | 4/4 | 1.35e+04/1.35e+04 | 0.15/0.15 | 252.3/256 | 31.5 | 0/797/1 | 0 | 104 | 3 | 11.54 |
| ft_transformer | DoS | p50 | 123 | Prim-PGD | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.16/0.16 | 4/4 | 1.01e+04/1.01e+04 | 0.1/0.1 | 252.7/253 | 9 | 0/797/1 | 0 | 126 | 3 | 13.01 |
| ft_transformer | DoS | p50 | 2024 | Prim-C&W | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.0961/0.0961 | 4/4 | 1.48e+03/1.48e+03 | 0.5/0.5 | 252.7/253 | 6 | 0/797/1 | 0 | 126 | 3 | 13.31 |
| ft_transformer | DoS | p50 | 2024 | Hybrid | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.185/0.185 | 4/4 | 1.35e+04/1.35e+04 | 0.15/0.15 | 252.3/256 | 31.5 | 0/797/1 | 0 | 104 | 3 | 11.94 |
| ft_transformer | DoS | p50 | 2024 | Prim-PGD | 800 | 799 | 2 | 0.25% | 0.25% | 0.12% | 0.171/0.171 | 4.5/4.5 | 1.01e+04/1.01e+04 | 0.1/0.1 | 252.7/253 | 9 | 0/797/1 | 0 | 126 | 3 | 13.53 |
| ft_transformer | DoS | p75 | 42 | Prim-C&W | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.39/0.148 | 20.7/8 | 995/0 | 0.333/0 | 252.7/253 | 7 | 0/796/1 | 0 | 126 | 3 | 13.58 |
| ft_transformer | DoS | p75 | 42 | Hybrid | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.449/0.2 | 20.7/8 | 8.98e+03/0 | 0.1/0 | 251.4/255 | 55 | 0/796/1 | 0 | 100 | 3 | 11.42 |
| ft_transformer | DoS | p75 | 42 | Prim-PGD | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.433/0.15 | 20.7/8 | 6.73e+03/0 | 0.0667/0 | 252.7/253 | 9 | 0/796/1 | 0 | 126 | 3 | 13.45 |
| ft_transformer | DoS | p75 | 123 | Prim-C&W | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.39/0.148 | 20.7/8 | 995/0 | 0.333/0 | 252.7/253 | 7 | 0/796/1 | 0 | 126 | 3 | 13.13 |
| ft_transformer | DoS | p75 | 123 | Hybrid | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.449/0.2 | 20.7/8 | 8.98e+03/0 | 0.1/0 | 251.4/255 | 55 | 0/796/1 | 0 | 100 | 3 | 11.45 |
| ft_transformer | DoS | p75 | 123 | Prim-PGD | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.433/0.15 | 20.7/8 | 6.73e+03/0 | 0.0667/0 | 252.7/253 | 9 | 0/796/1 | 0 | 126 | 3 | 13.00 |
| ft_transformer | DoS | p75 | 2024 | Prim-C&W | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.39/0.148 | 20.7/8 | 995/0 | 0.333/0 | 252.7/253 | 7 | 0/796/1 | 0 | 126 | 3 | 13.24 |
| ft_transformer | DoS | p75 | 2024 | Hybrid | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.449/0.2 | 20.7/8 | 8.98e+03/0 | 0.1/0 | 251.4/255 | 55 | 0/796/1 | 0 | 100 | 3 | 11.74 |
| ft_transformer | DoS | p75 | 2024 | Prim-PGD | 800 | 799 | 3 | 0.38% | 0.38% | 0.12% | 0.433/0.15 | 20.7/8 | 6.73e+03/0 | 0.0667/0 | 252.7/253 | 9 | 0/796/1 | 0 | 126 | 3 | 13.60 |
| ft_transformer | DoS | unb | 42 | Prim-C&W | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.3/0.242 | 20.5/14 | 8.043e+05/1.49e+03 | 0.5/0.5 | 252.7/253 | 4 | 0/795/1 | 0 | 126 | 3 | 13.66 |
| ft_transformer | DoS | unb | 42 | Hybrid | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.453/0.446 | 25.5/24 | 1.955e+06/1.35e+04 | 0.198/0.15 | 251.7/255 | 68 | 0/795/1 | 0 | 88 | 3 | 10.65 |
| ft_transformer | DoS | unb | 42 | Prim-PGD | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.366/0.328 | 20.8/14 | 1.603e+06/1.01e+04 | 0.113/0.1 | 252.7/253 | 10 | 0/795/1 | 0 | 126 | 3 | 13.29 |
| ft_transformer | DoS | unb | 123 | Prim-C&W | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.3/0.242 | 20.5/14 | 8.043e+05/1.49e+03 | 0.5/0.5 | 252.7/253 | 4 | 0/795/1 | 0 | 126 | 3 | 13.15 |
| ft_transformer | DoS | unb | 123 | Hybrid | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.502/0.446 | 19.5/12 | 5.159e+06/1.35e+04 | 0.154/0.15 | 251.7/255 | 68 | 0/795/1 | 0 | 88 | 3 | 11.05 |
| ft_transformer | DoS | unb | 123 | Prim-PGD | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.366/0.328 | 20.8/14 | 1.603e+06/1.01e+04 | 0.113/0.1 | 252.7/253 | 10 | 0/795/1 | 0 | 126 | 3 | 13.06 |
| ft_transformer | DoS | unb | 2024 | Prim-C&W | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.3/0.242 | 20.5/14 | 8.043e+05/1.49e+03 | 0.5/0.5 | 252.7/253 | 4 | 0/795/1 | 0 | 126 | 3 | 13.59 |
| ft_transformer | DoS | unb | 2024 | Hybrid | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.488/0.446 | 33.2/31 | 3.135e+05/1.35e+04 | 0.325/0.15 | 251.7/255 | 68 | 0/795/1 | 0 | 88 | 3 | 10.92 |
| ft_transformer | DoS | unb | 2024 | Prim-PGD | 800 | 799 | 4 | 0.50% | 0.50% | 0.25% | 0.314/0.224 | 19/10.5 | 8.437e+05/1.01e+04 | 0.254/0.1 | 252.7/253 | 10 | 0/795/1 | 0 | 126 | 3 | 13.64 |
| ft_transformer | Recon | p50 | 42 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.43 |
| ft_transformer | Recon | p50 | 42 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.69 |
| ft_transformer | Recon | p50 | 42 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.41 |
| ft_transformer | Recon | p50 | 123 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.52 |
| ft_transformer | Recon | p50 | 123 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.46 |
| ft_transformer | Recon | p50 | 123 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.41 |
| ft_transformer | Recon | p50 | 2024 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.55 |
| ft_transformer | Recon | p50 | 2024 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.49 |
| ft_transformer | Recon | p50 | 2024 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.42 |
| ft_transformer | Recon | p75 | 42 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.50 |
| ft_transformer | Recon | p75 | 42 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.38 |
| ft_transformer | Recon | p75 | 42 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.50 |
| ft_transformer | Recon | p75 | 123 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.56 |
| ft_transformer | Recon | p75 | 123 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.48 |
| ft_transformer | Recon | p75 | 123 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.42 |
| ft_transformer | Recon | p75 | 2024 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.47 |
| ft_transformer | Recon | p75 | 2024 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.45 |
| ft_transformer | Recon | p75 | 2024 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.53 |
| ft_transformer | Recon | unb | 42 | Prim-C&W | 800 | 9 | 1 | 0.12% | 0.12% | 0.00% | 0.706/0.706 | 0/0 | 3.430e+07/3.430e+07 | 1/1 | 3.835/1 | 7 | 0/8/791 | 0 | 126 | 3 | 2.46 |
| ft_transformer | Recon | unb | 42 | Hybrid | 800 | 9 | 6 | 0.75% | 0.75% | 0.00% | 0.0377/0.0429 | 0/0 | 2.078e+06/2.122e+06 | 0.737/0.961 | 3.857/1 | 3 | 0/3/791 | 0 | 127 | 4 | 2.42 |
| ft_transformer | Recon | unb | 42 | Prim-PGD | 800 | 9 | 6 | 0.75% | 0.75% | 0.00% | 0.0347/0.0458 | 0/0 | 2.073e+06/2.565e+06 | 0.683/1 | 3.835/1 | 3 | 0/3/791 | 0 | 126 | 3 | 2.49 |
| ft_transformer | Recon | unb | 123 | Prim-C&W | 800 | 9 | 1 | 0.12% | 0.12% | 0.00% | 0.706/0.706 | 0/0 | 3.430e+07/3.430e+07 | 1/1 | 3.835/1 | 7 | 0/8/791 | 0 | 126 | 3 | 2.52 |
| ft_transformer | Recon | unb | 123 | Hybrid | 800 | 9 | 6 | 0.75% | 0.75% | 0.00% | 0.0378/0.0436 | 0/0 | 2.272e+06/2.482e+06 | 0.568/0.617 | 3.857/1 | 3 | 0/3/791 | 0 | 127 | 4 | 2.48 |
| ft_transformer | Recon | unb | 123 | Prim-PGD | 800 | 9 | 6 | 0.75% | 0.75% | 0.00% | 0.0448/0.0488 | 0/0 | 2.489e+06/2.482e+06 | 0.526/0.567 | 3.835/1 | 3 | 0/3/791 | 0 | 126 | 3 | 2.40 |
| ft_transformer | Recon | unb | 2024 | Prim-C&W | 800 | 9 | 1 | 0.12% | 0.12% | 0.00% | 0.706/0.706 | 0/0 | 3.430e+07/3.430e+07 | 1/1 | 3.835/1 | 7 | 0/8/791 | 0 | 126 | 3 | 2.54 |
| ft_transformer | Recon | unb | 2024 | Hybrid | 800 | 9 | 6 | 0.75% | 0.75% | 0.00% | 0.04/0.05 | 0/0 | 2.144e+06/2.142e+06 | 0.792/0.939 | 3.857/1 | 3 | 0/3/791 | 0 | 127 | 4 | 2.49 |
| ft_transformer | Recon | unb | 2024 | Prim-PGD | 800 | 9 | 6 | 0.75% | 0.75% | 0.00% | 0.043/0.0494 | 0/0 | 2.346e+06/2.745e+06 | 0.577/0.621 | 3.835/1 | 3 | 0/3/791 | 0 | 126 | 3 | 2.41 |
| mlp | BruteForce | p50 | 42 | Prim-C&W | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.766/0.37 | 4.29/3 | 2.653e+05/1.410e+05 | 0.714/1 | 253/253 | 5 | 0/793/0 | 0 | 126 | 3 | 2.56 |
| mlp | BruteForce | p50 | 42 | Hybrid | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.976/0.737 | 7.14/8 | 2.527e+05/1.654e+05 | 0.647/0.89 | 254.4/255 | 23 | 0/793/0 | 0 | 121 | 4 | 2.35 |
| mlp | BruteForce | p50 | 42 | Prim-PGD | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.996/0.737 | 6.86/6 | 2.733e+05/1.822e+05 | 0.492/0.5 | 253/253 | 21 | 0/793/0 | 0 | 126 | 3 | 2.32 |
| mlp | BruteForce | p50 | 123 | Prim-C&W | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.766/0.37 | 4.29/3 | 2.653e+05/1.410e+05 | 0.714/1 | 253/253 | 5 | 0/793/0 | 0 | 126 | 3 | 2.27 |
| mlp | BruteForce | p50 | 123 | Hybrid | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 1/0.729 | 6.86/8 | 2.753e+05/1.747e+05 | 0.624/0.834 | 254.4/255 | 23 | 0/793/0 | 0 | 121 | 4 | 2.35 |
| mlp | BruteForce | p50 | 123 | Prim-PGD | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.97/0.667 | 6.57/6 | 2.687e+05/1.549e+05 | 0.583/0.784 | 253/253 | 21 | 0/793/0 | 0 | 126 | 3 | 2.36 |
| mlp | BruteForce | p50 | 2024 | Prim-C&W | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.766/0.37 | 4.29/3 | 2.653e+05/1.410e+05 | 0.714/1 | 253/253 | 5 | 0/793/0 | 0 | 126 | 3 | 2.46 |
| mlp | BruteForce | p50 | 2024 | Hybrid | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 1.13/0.953 | 7.71/8 | 3.106e+05/2.274e+05 | 0.535/0.5 | 254.4/255 | 23 | 0/793/0 | 0 | 121 | 4 | 2.26 |
| mlp | BruteForce | p50 | 2024 | Prim-PGD | 800 | 800 | 7 | 0.88% | 0.88% | 0.00% | 0.973/0.694 | 6/6 | 3.024e+05/2.558e+05 | 0.461/0.429 | 253/253 | 21 | 0/793/0 | 0 | 126 | 3 | 2.37 |
| mlp | BruteForce | p75 | 42 | Prim-C&W | 800 | 800 | 15 | 1.88% | 1.88% | 0.00% | 0.615/0.521 | 35.3/28 | 2.623e+05/1.644e+05 | 0.599/0.998 | 253/253 | 5 | 0/785/0 | 0 | 126 | 3 | 2.49 |
| mlp | BruteForce | p75 | 42 | Hybrid | 800 | 800 | 17 | 2.12% | 2.12% | 0.00% | 0.869/0.872 | 49.2/60 | 4.099e+05/0 | 0.282/0 | 252.3/255 | 69 | 0/783/0 | 0 | 88 | 3 | 1.99 |
| mlp | BruteForce | p75 | 42 | Prim-PGD | 800 | 800 | 17 | 2.12% | 2.12% | 0.00% | 0.75/0.705 | 40.4/35 | 3.749e+05/1.544e+05 | 0.309/0.231 | 253/253 | 19 | 0/783/0 | 0 | 126 | 3 | 2.33 |
| mlp | BruteForce | p75 | 123 | Prim-C&W | 800 | 800 | 15 | 1.88% | 1.88% | 0.00% | 0.615/0.521 | 35.3/28 | 2.623e+05/1.644e+05 | 0.599/0.998 | 253/253 | 5 | 0/785/0 | 0 | 126 | 3 | 2.54 |
| mlp | BruteForce | p75 | 123 | Hybrid | 800 | 800 | 17 | 2.12% | 2.12% | 0.00% | 0.819/0.795 | 50.9/62 | 2.929e+05/0 | 0.272/0 | 252.3/255 | 69 | 0/783/0 | 0 | 88 | 3 | 1.94 |
| mlp | BruteForce | p75 | 123 | Prim-PGD | 800 | 800 | 17 | 2.12% | 2.12% | 0.00% | 0.759/0.636 | 38.8/35 | 4.156e+05/1.544e+05 | 0.359/0.25 | 253/253 | 19 | 0/783/0 | 0 | 126 | 3 | 2.33 |
| mlp | BruteForce | p75 | 2024 | Prim-C&W | 800 | 800 | 15 | 1.88% | 1.88% | 0.00% | 0.615/0.521 | 35.3/28 | 2.623e+05/1.644e+05 | 0.599/0.998 | 253/253 | 5 | 0/785/0 | 0 | 126 | 3 | 2.80 |
| mlp | BruteForce | p75 | 2024 | Hybrid | 800 | 800 | 17 | 2.12% | 2.12% | 0.00% | 0.846/0.872 | 49.5/60 | 3.727e+05/0 | 0.262/0 | 252.3/255 | 69 | 0/783/0 | 0 | 88 | 3 | 1.95 |
| mlp | BruteForce | p75 | 2024 | Prim-PGD | 800 | 800 | 17 | 2.12% | 2.12% | 0.00% | 0.772/0.795 | 42.1/35 | 3.630e+05/1.544e+05 | 0.352/0.25 | 253/253 | 19 | 0/783/0 | 0 | 126 | 3 | 2.72 |
| mlp | BruteForce | unb | 42 | Prim-C&W | 800 | 800 | 15 | 1.88% | 1.88% | 0.00% | 0.842/0.973 | 26.8/0 | 4.983e+07/3.984e+07 | 0.862/0.922 | 253/253 | 5 | 0/785/0 | 0 | 126 | 3 | 2.69 |
| mlp | BruteForce | unb | 42 | Hybrid | 800 | 800 | 20 | 2.50% | 2.50% | 0.00% | 0.747/0.865 | 51.6/62 | 8.499e+06/9.761e+05 | 0.432/0.05 | 252.3/255 | 77 | 0/780/0 | 0 | 88 | 3 | 2.68 |
| mlp | BruteForce | unb | 42 | Prim-PGD | 800 | 800 | 20 | 2.50% | 2.50% | 0.00% | 0.363/0.127 | 21.6/6 | 8.335e+06/5.668e+06 | 0.296/0.075 | 253/253 | 4 | 0/780/0 | 0 | 126 | 3 | 2.83 |
| mlp | BruteForce | unb | 123 | Prim-C&W | 800 | 800 | 15 | 1.88% | 1.88% | 0.00% | 0.842/0.973 | 26.8/0 | 4.983e+07/3.984e+07 | 0.862/0.922 | 253/253 | 5 | 0/785/0 | 0 | 126 | 3 | 3.29 |
| mlp | BruteForce | unb | 123 | Hybrid | 800 | 800 | 20 | 2.50% | 2.50% | 0.00% | 0.7/0.772 | 46.4/53 | 9.633e+06/1.941e+05 | 0.392/0 | 252.3/255 | 77 | 0/780/0 | 0 | 88 | 3 | 2.48 |
| mlp | BruteForce | unb | 123 | Prim-PGD | 800 | 800 | 20 | 2.50% | 2.50% | 0.00% | 0.362/0.127 | 20.9/6 | 8.985e+06/5.668e+06 | 0.305/0.075 | 253/253 | 4 | 0/780/0 | 0 | 126 | 3 | 2.48 |
| mlp | BruteForce | unb | 2024 | Prim-C&W | 800 | 800 | 15 | 1.88% | 1.88% | 0.00% | 0.842/0.973 | 26.8/0 | 4.983e+07/3.984e+07 | 0.862/0.922 | 253/253 | 5 | 0/785/0 | 0 | 126 | 3 | 2.68 |
| mlp | BruteForce | unb | 2024 | Hybrid | 800 | 800 | 20 | 2.50% | 2.50% | 0.00% | 0.721/0.87 | 50/59.5 | 7.380e+06/7.059e+05 | 0.373/0.05 | 252.3/255 | 77 | 0/780/0 | 0 | 88 | 3 | 2.73 |
| mlp | BruteForce | unb | 2024 | Prim-PGD | 800 | 800 | 20 | 2.50% | 2.50% | 0.00% | 0.359/0.127 | 22.2/6 | 7.028e+06/5.666e+06 | 0.333/0.075 | 253/253 | 4 | 0/780/0 | 0 | 126 | 3 | 2.63 |
| mlp | DDoS | p50 | 42 | Prim-C&W | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 0.824/0.639 | 0.304/0 | 1.078e+06/9.779e+05 | 1/1 | 253/253 | 5 | 0/721/0 | 0 | 126 | 3 | 2.54 |
| mlp | DDoS | p50 | 42 | Hybrid | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 1.42/1.45 | 1.18/1 | 1.388e+06/1.292e+06 | 0.878/0.961 | 244.9/255 | 19 | 0/721/0 | 0 | 126 | 4 | 2.36 |
| mlp | DDoS | p50 | 42 | Prim-PGD | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 1.39/1.37 | 1.32/1 | 1.210e+06/1.114e+06 | 0.861/0.95 | 253/253 | 31 | 0/721/0 | 0 | 126 | 3 | 2.36 |
| mlp | DDoS | p50 | 123 | Prim-C&W | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 0.824/0.639 | 0.304/0 | 1.078e+06/9.779e+05 | 1/1 | 253/253 | 5 | 0/721/0 | 0 | 126 | 3 | 2.99 |
| mlp | DDoS | p50 | 123 | Hybrid | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 1.41/1.37 | 1.32/1 | 1.234e+06/1.163e+06 | 0.912/1 | 244.9/255 | 19 | 0/721/0 | 0 | 126 | 4 | 2.36 |
| mlp | DDoS | p50 | 123 | Prim-PGD | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 1.41/1.35 | 1.32/1 | 1.239e+06/1.192e+06 | 0.84/0.88 | 253/253 | 31 | 0/721/0 | 0 | 126 | 3 | 2.75 |
| mlp | DDoS | p50 | 2024 | Prim-C&W | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 0.824/0.639 | 0.304/0 | 1.078e+06/9.779e+05 | 1/1 | 253/253 | 5 | 0/721/0 | 0 | 126 | 3 | 2.46 |
| mlp | DDoS | p50 | 2024 | Hybrid | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 1.42/1.43 | 1.33/1 | 1.244e+06/1.136e+06 | 0.902/0.988 | 244.9/255 | 19 | 0/721/0 | 0 | 126 | 4 | 2.56 |
| mlp | DDoS | p50 | 2024 | Prim-PGD | 800 | 800 | 79 | 9.88% | 9.88% | 9.88% | 1.36/1.4 | 1.2/1 | 1.244e+06/1.135e+06 | 0.85/0.907 | 253/253 | 31 | 0/721/0 | 0 | 126 | 3 | 2.16 |
| mlp | DDoS | p75 | 42 | Prim-C&W | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 0.711/0.53 | 0.396/0 | 1.323e+06/1.118e+06 | 1/1 | 253/253 | 5 | 0/694/0 | 0 | 126 | 3 | 2.56 |
| mlp | DDoS | p75 | 42 | Hybrid | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 1.33/1.27 | 1.77/2 | 1.799e+06/1.664e+06 | 0.845/0.947 | 245.9/256 | 18 | 0/694/0 | 0 | 126 | 4 | 2.37 |
| mlp | DDoS | p75 | 42 | Prim-PGD | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 1.25/1.22 | 1.75/2 | 1.573e+06/1.402e+06 | 0.8/0.829 | 253/253 | 29 | 0/694/0 | 0 | 126 | 3 | 2.38 |
| mlp | DDoS | p75 | 123 | Prim-C&W | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 0.711/0.53 | 0.396/0 | 1.323e+06/1.118e+06 | 1/1 | 253/253 | 5 | 0/694/0 | 0 | 126 | 3 | 2.56 |
| mlp | DDoS | p75 | 123 | Hybrid | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 1.28/1.24 | 1.82/2 | 1.625e+06/1.474e+06 | 0.872/0.96 | 245.9/256 | 18 | 0/694/0 | 0 | 126 | 4 | 2.38 |
| mlp | DDoS | p75 | 123 | Prim-PGD | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 1.27/1.27 | 1.75/2 | 1.655e+06/1.449e+06 | 0.768/0.807 | 253/253 | 29 | 0/694/0 | 0 | 126 | 3 | 2.33 |
| mlp | DDoS | p75 | 2024 | Prim-C&W | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 0.711/0.53 | 0.396/0 | 1.323e+06/1.118e+06 | 1/1 | 253/253 | 5 | 0/694/0 | 0 | 126 | 3 | 2.55 |
| mlp | DDoS | p75 | 2024 | Hybrid | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 1.32/1.33 | 1.92/2 | 1.628e+06/1.479e+06 | 0.863/0.953 | 245.9/256 | 18 | 0/694/0 | 0 | 126 | 4 | 2.36 |
| mlp | DDoS | p75 | 2024 | Prim-PGD | 800 | 800 | 106 | 13.25% | 13.25% | 13.25% | 1.26/1.27 | 1.75/2 | 1.592e+06/1.402e+06 | 0.793/0.824 | 253/253 | 29 | 0/694/0 | 0 | 126 | 3 | 2.39 |
| mlp | DDoS | unb | 42 | Prim-C&W | 800 | 800 | 584 | 73.00% | 73.00% | 71.88% | 0.569/0.546 | 28.9/29 | 1.387e+06/1.121e+06 | 0.994/1 | 253/253 | 3 | 0/216/0 | 0 | 126 | 3 | 2.90 |
| mlp | DDoS | unb | 42 | Hybrid | 800 | 800 | 603 | 75.38% | 75.38% | 71.50% | 0.888/0.867 | 35/36 | 3.048e+06/2.429e+06 | 0.554/0.5 | 246.3/255 | 87 | 0/197/0 | 0 | 88 | 3 | 2.05 |
| mlp | DDoS | unb | 42 | Prim-PGD | 800 | 800 | 604 | 75.50% | 75.50% | 74.25% | 0.779/0.717 | 30.1/27 | 2.495e+06/2.301e+06 | 0.514/0.45 | 253/253 | 17 | 0/196/0 | 2 | 126 | 3 | 2.40 |
| mlp | DDoS | unb | 123 | Prim-C&W | 800 | 800 | 584 | 73.00% | 73.00% | 71.88% | 0.569/0.546 | 28.9/29 | 1.387e+06/1.121e+06 | 0.994/1 | 253/253 | 3 | 0/216/0 | 1 | 126 | 3 | 2.71 |
| mlp | DDoS | unb | 123 | Hybrid | 800 | 800 | 600 | 75.00% | 75.00% | 72.00% | 0.878/0.863 | 34.6/35 | 3.043e+06/2.397e+06 | 0.555/0.5 | 246.3/255 | 85 | 0/200/0 | 1 | 88 | 3 | 2.68 |
| mlp | DDoS | unb | 123 | Prim-PGD | 800 | 800 | 605 | 75.62% | 75.62% | 74.50% | 0.775/0.705 | 30/27 | 2.489e+06/2.308e+06 | 0.527/0.5 | 253/253 | 17 | 0/195/0 | 3 | 126 | 3 | 2.62 |
| mlp | DDoS | unb | 2024 | Prim-C&W | 800 | 800 | 584 | 73.00% | 73.00% | 71.88% | 0.569/0.546 | 28.9/29 | 1.387e+06/1.121e+06 | 0.994/1 | 253/253 | 3 | 0/216/0 | 1 | 126 | 3 | 2.58 |
| mlp | DDoS | unb | 2024 | Hybrid | 800 | 800 | 596 | 74.50% | 74.50% | 71.75% | 0.879/0.853 | 34.9/36 | 2.986e+06/2.395e+06 | 0.551/0.5 | 246.3/255 | 85 | 0/204/0 | 2 | 88 | 3 | 2.13 |
| mlp | DDoS | unb | 2024 | Prim-PGD | 800 | 800 | 604 | 75.50% | 75.50% | 74.75% | 0.772/0.705 | 30/27 | 2.485e+06/2.305e+06 | 0.517/0.487 | 253/253 | 17 | 0/196/0 | 6 | 126 | 3 | 2.49 |
| mlp | DoS | p50 | 42 | Prim-C&W | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.41/1.48 | 29/32 | 2.365e+05/1.027e+05 | 0.999/1 | 252.7/253 | 11 | 0/683/1 | 0 | 126 | 3 | 3.08 |
| mlp | DoS | p50 | 42 | Hybrid | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.56/1.59 | 35.3/36.5 | 2.629e+05/1.049e+05 | 0.912/1 | 252/256 | 66 | 0/683/1 | 0 | 104 | 3 | 2.88 |
| mlp | DoS | p50 | 42 | Prim-PGD | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.51/1.55 | 33.8/33 | 2.437e+05/1.038e+05 | 0.908/1 | 252.7/253 | 37 | 0/683/1 | 0 | 126 | 3 | 2.72 |
| mlp | DoS | p50 | 123 | Prim-C&W | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.41/1.48 | 29/32 | 2.365e+05/1.027e+05 | 0.999/1 | 252.7/253 | 11 | 0/683/1 | 0 | 126 | 3 | 3.26 |
| mlp | DoS | p50 | 123 | Hybrid | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.55/1.6 | 34.4/34 | 2.432e+05/1.060e+05 | 0.931/1 | 252/256 | 66 | 0/683/1 | 0 | 104 | 3 | 2.80 |
| mlp | DoS | p50 | 123 | Prim-PGD | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.5/1.55 | 34.1/35 | 2.421e+05/1.061e+05 | 0.905/1 | 252.7/253 | 37 | 0/683/1 | 0 | 126 | 3 | 3.19 |
| mlp | DoS | p50 | 2024 | Prim-C&W | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.41/1.48 | 29/32 | 2.365e+05/1.027e+05 | 0.999/1 | 252.7/253 | 11 | 0/683/1 | 0 | 126 | 3 | 3.48 |
| mlp | DoS | p50 | 2024 | Hybrid | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.55/1.6 | 34.6/34.5 | 2.484e+05/1.069e+05 | 0.925/1 | 252/256 | 66 | 0/683/1 | 0 | 104 | 3 | 2.76 |
| mlp | DoS | p50 | 2024 | Prim-PGD | 800 | 799 | 116 | 14.50% | 14.50% | 14.50% | 1.5/1.54 | 33.1/31.5 | 2.467e+05/1.035e+05 | 0.913/1 | 252.7/253 | 37 | 0/683/1 | 0 | 126 | 3 | 3.18 |
| mlp | DoS | p75 | 42 | Prim-C&W | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.25/1.3 | 32.4/35 | 3.036e+05/1.762e+05 | 1/1 | 252.7/253 | 9 | 0/568/1 | 0 | 126 | 3 | 2.99 |
| mlp | DoS | p75 | 42 | Hybrid | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.45/1.46 | 37.6/38 | 3.909e+05/2.042e+05 | 0.871/0.952 | 251.2/255 | 73 | 0/568/1 | 0 | 100 | 3 | 2.72 |
| mlp | DoS | p75 | 42 | Prim-PGD | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.39/1.41 | 35.6/35 | 3.603e+05/1.978e+05 | 0.857/0.958 | 252.7/253 | 35 | 0/568/1 | 0 | 126 | 3 | 2.85 |
| mlp | DoS | p75 | 123 | Prim-C&W | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.25/1.3 | 32.4/35 | 3.036e+05/1.762e+05 | 1/1 | 252.7/253 | 9 | 0/568/1 | 0 | 126 | 3 | 3.00 |
| mlp | DoS | p75 | 123 | Hybrid | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.44/1.43 | 38/38 | 3.503e+05/1.953e+05 | 0.886/1 | 251.2/255 | 73 | 0/568/1 | 0 | 100 | 3 | 2.34 |
| mlp | DoS | p75 | 123 | Prim-PGD | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.39/1.42 | 36.2/35 | 3.503e+05/1.936e+05 | 0.86/0.95 | 252.7/253 | 35 | 0/568/1 | 0 | 126 | 3 | 2.71 |
| mlp | DoS | p75 | 2024 | Prim-C&W | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.25/1.3 | 32.4/35 | 3.036e+05/1.762e+05 | 1/1 | 252.7/253 | 9 | 0/568/1 | 0 | 126 | 3 | 2.84 |
| mlp | DoS | p75 | 2024 | Hybrid | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.45/1.5 | 37.8/38 | 3.762e+05/1.998e+05 | 0.874/1 | 251.2/255 | 73 | 0/568/1 | 0 | 100 | 3 | 2.46 |
| mlp | DoS | p75 | 2024 | Prim-PGD | 800 | 799 | 231 | 28.88% | 28.88% | 28.88% | 1.38/1.44 | 36.1/35 | 3.495e+05/1.915e+05 | 0.863/0.958 | 252.7/253 | 35 | 0/568/1 | 0 | 126 | 3 | 2.77 |
| mlp | DoS | unb | 42 | Prim-C&W | 800 | 799 | 712 | 89.00% | 89.00% | 88.88% | 0.388/0.275 | 11.3/0 | 5.343e+06/4.042e+06 | 0.974/1 | 252.7/253 | 3 | 0/87/1 | 0 | 126 | 3 | 2.93 |
| mlp | DoS | unb | 42 | Hybrid | 800 | 799 | 713 | 89.12% | 89.12% | 89.12% | 0.75/0.695 | 33.3/25 | 7.123e+06/6.672e+06 | 0.524/0.42 | 251.6/255 | 85 | 0/86/1 | 0 | 88 | 3 | 2.29 |
| mlp | DoS | unb | 42 | Prim-PGD | 800 | 799 | 713 | 89.12% | 89.12% | 89.12% | 0.588/0.514 | 23.2/20 | 6.400e+06/5.597e+06 | 0.405/0.3 | 252.7/253 | 13 | 0/86/1 | 0 | 126 | 3 | 2.77 |
| mlp | DoS | unb | 123 | Prim-C&W | 800 | 799 | 712 | 89.00% | 89.00% | 88.88% | 0.388/0.275 | 11.3/0 | 5.343e+06/4.042e+06 | 0.974/1 | 252.7/253 | 3 | 0/87/1 | 0 | 126 | 3 | 2.97 |
| mlp | DoS | unb | 123 | Hybrid | 800 | 799 | 713 | 89.12% | 89.12% | 89.00% | 0.751/0.695 | 33.8/27 | 7.026e+06/6.643e+06 | 0.525/0.418 | 251.6/255 | 85 | 0/86/1 | 0 | 88 | 3 | 2.28 |
| mlp | DoS | unb | 123 | Prim-PGD | 800 | 799 | 713 | 89.12% | 89.12% | 89.00% | 0.588/0.528 | 22.6/20 | 6.541e+06/5.889e+06 | 0.395/0.3 | 252.7/253 | 13 | 0/86/1 | 0 | 126 | 3 | 2.75 |
| mlp | DoS | unb | 2024 | Prim-C&W | 800 | 799 | 712 | 89.00% | 89.00% | 88.88% | 0.388/0.275 | 11.3/0 | 5.343e+06/4.042e+06 | 0.974/1 | 252.7/253 | 3 | 0/87/1 | 0 | 126 | 3 | 2.55 |
| mlp | DoS | unb | 2024 | Hybrid | 800 | 799 | 713 | 89.12% | 89.12% | 88.88% | 0.747/0.695 | 32.9/25 | 7.174e+06/6.704e+06 | 0.52/0.405 | 251.6/255 | 85 | 0/86/1 | 0 | 88 | 3 | 2.33 |
| mlp | DoS | unb | 2024 | Prim-PGD | 800 | 799 | 713 | 89.12% | 89.12% | 89.12% | 0.584/0.506 | 23/20 | 6.374e+06/5.718e+06 | 0.402/0.3 | 252.7/253 | 13 | 0/86/1 | 0 | 126 | 3 | 2.63 |
| mlp | Recon | p50 | 42 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.13 |
| mlp | Recon | p50 | 42 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.12 |
| mlp | Recon | p50 | 42 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.06 |
| mlp | Recon | p50 | 123 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.14 |
| mlp | Recon | p50 | 123 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.11 |
| mlp | Recon | p50 | 123 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.08 |
| mlp | Recon | p50 | 2024 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.12 |
| mlp | Recon | p50 | 2024 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.09 |
| mlp | Recon | p50 | 2024 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.05 |
| mlp | Recon | p75 | 42 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.14 |
| mlp | Recon | p75 | 42 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.14 |
| mlp | Recon | p75 | 42 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.10 |
| mlp | Recon | p75 | 123 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.15 |
| mlp | Recon | p75 | 123 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.12 |
| mlp | Recon | p75 | 123 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.08 |
| mlp | Recon | p75 | 2024 | Prim-C&W | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.22 |
| mlp | Recon | p75 | 2024 | Hybrid | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.857/1 | — | 0/9/791 | 0 | 127 | 4 | 2.11 |
| mlp | Recon | p75 | 2024 | Prim-PGD | 800 | 9 | 0 | 0.00% | 0.00% | 0.00% | —/— | —/— | —/— | —/— | 3.835/1 | — | 0/9/791 | 0 | 126 | 3 | 2.07 |
| mlp | Recon | unb | 42 | Prim-C&W | 800 | 9 | 4 | 0.50% | 0.50% | 0.00% | 0.539/0.5 | 11.5/0 | 2.595e+07/2.307e+07 | 0.71/0.669 | 3.835/1 | 3 | 0/5/791 | 0 | 126 | 3 | 2.19 |
| mlp | Recon | unb | 42 | Hybrid | 800 | 9 | 8 | 1.00% | 1.00% | 0.00% | 0.139/0.1 | 1.12/0 | 9.850e+06/6.500e+06 | 0.0625/0.1 | 3.857/1 | 3 | 0/1/791 | 0 | 127 | 4 | 2.08 |
| mlp | Recon | unb | 42 | Prim-PGD | 800 | 9 | 8 | 1.00% | 1.00% | 0.00% | 0.0688/0.05 | 0/0 | 5.628e+06/3.250e+06 | 0.0563/0.05 | 3.835/1 | 3 | 0/1/791 | 0 | 126 | 3 | 1.99 |
| mlp | Recon | unb | 123 | Prim-C&W | 800 | 9 | 4 | 0.50% | 0.50% | 0.00% | 0.539/0.5 | 11.5/0 | 2.595e+07/2.307e+07 | 0.71/0.669 | 3.835/1 | 3 | 0/5/791 | 0 | 126 | 3 | 2.19 |
| mlp | Recon | unb | 123 | Hybrid | 800 | 9 | 8 | 1.00% | 1.00% | 0.00% | 0.139/0.1 | 1.12/0 | 9.850e+06/6.500e+06 | 0.0625/0.1 | 3.857/1 | 3 | 0/1/791 | 0 | 127 | 4 | 2.14 |
| mlp | Recon | unb | 123 | Prim-PGD | 800 | 9 | 8 | 1.00% | 1.00% | 0.00% | 0.0688/0.05 | 0/0 | 5.628e+06/3.250e+06 | 0.0563/0.05 | 3.835/1 | 3 | 0/1/791 | 0 | 126 | 3 | 2.02 |
| mlp | Recon | unb | 2024 | Prim-C&W | 800 | 9 | 4 | 0.50% | 0.50% | 0.00% | 0.539/0.5 | 11.5/0 | 2.595e+07/2.307e+07 | 0.71/0.669 | 3.835/1 | 3 | 0/5/791 | 0 | 126 | 3 | 2.19 |
| mlp | Recon | unb | 2024 | Hybrid | 800 | 9 | 8 | 1.00% | 1.00% | 0.00% | 0.139/0.1 | 1.12/0 | 9.850e+06/6.500e+06 | 0.0625/0.1 | 3.857/1 | 3 | 0/1/791 | 0 | 127 | 4 | 2.11 |
| mlp | Recon | unb | 2024 | Prim-PGD | 800 | 9 | 8 | 1.00% | 1.00% | 0.00% | 0.0688/0.05 | 0/0 | 5.628e+06/3.250e+06 | 0.0563/0.05 | 3.835/1 | 3 | 0/1/791 | 0 | 126 | 3 | 2.01 |
