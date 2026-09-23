# Plan — Black-Box Primitive Attack & Real-World Cost Curves

Two concrete extensions to the CICIDS2017-DistriNet primitive attack that add *genuine*
novelty without re-inventing the optimizer or the constraints:

- **Part A — Black-box (query-only) primitive attack.** Answers the "white-box is
  unrealistic" critique. The search space is only two dimensions per flow (`p`, `alpha`), so
  gradient-free search is cheap and realistic.
- **Part B — Real-world cost curves.** Report evasion as a function of *physical* attacker
  budget (bytes of padding, milliseconds of delay), not an abstract `L_p` norm.

Both reuse the existing realizability layer, validators, victims, rows, seeds, and
True-IDSR metric — only the *inner optimizer* (Part A) or the *sweep + analysis* (Part B)
is new. Detailed background is in `docs/attack_system/`.

---

## 0. Shared scaffolding (reused unchanged)

| Component | Source | Reused as-is |
|-----------|--------|--------------|
| Realizability map `(p, alpha) -> 79 features` | `attack/realizability/cicids2017.py::CICIDS2017PrimitiveModel` | `per_flow_bounds`, `active_mask`, `project_controls`, `generate(quantize=True)` |
| Per-flow feasible caps | `model.per_flow_bounds(raw, bounds_cfg)` | `p_hi`, `alpha_hi` |
| Row selection | `run_cicids2017_primitive_attack.py::_class_rows` | `_class_rows(test.y, cid, limit, 42 + cid)` |
| Victims | `classifiers/cicids2017d_victims.py::load_category_victim` | `mlp, cnn, lstm, serial` |
| Validators | realizability + PAVE + A4 mined engine + IDR | `evaluate_cell` (identical) |
| Metrics | `strict_valid`, `true_idsr`, cost decomposition | `evaluate_cell`, `_decompose_cost` |
| Seeds / provenance | `experiments/provenance.py` | `deterministic_runtime`, `build_provenance`, NPZ arrays |

**Invariant for both parts:** success is always measured on the **realized** vector
(`project_controls` -> `generate(quantize=True)`), never on a continuous candidate.

---

# Part A — Black-Box (Query-Only) Primitive Attack

## A.0 Executive summary — why this is exceptionally query-efficient

Generic black-box attacks burn thousands of queries fighting high-dimensional spaces. This
problem has five structural advantages that collapse that to *tens* of queries per flow:

1. **Only 2 dimensions per flow** (`p`, `alpha`) — no high-dimensional gradient estimation.
2. **Batching is free** — all 1024 flows are queried in one forward pass, each with its own
   candidate, so per-flow query count = number of forward passes, independent of batch size.
3. **Known anchors** — `(0, 1)` is the clean (malicious, correctly-classified) flow and
   `(p_hi, alpha_hi)` is the max-effort corner; two informative points for free.
4. **Approximate monotone structure** — more padding/dilation pushes toward Benign, so the
   success set is roughly an "upper-right" region → bisection converges in `O(log)` queries.
5. **Score signal** — score-based queries expose the Benign margin, so bisection runs on a
   continuous signal, not just a label flip.

**Primary method:** *corner-prune + bisection* — query the max corner once (1 query rejects an
infeasible flow), else bisect the success boundary to the minimal-cost point (~10–15 queries).
**Stack:** corner feasibility prune, a shared coarse-grid prior + local refine, early-stop /
freeze solved flows, surrogate/seed warm-starts, and prefer score- over decision-based.
**Report:** median queries-to-first-success, fraction solved at `Q ≤ 10`, infeasible-prune
rate, and area under the True-IDSR-vs-`Q` curve. Full detail in §A.3.1.

The expected finding: black-box reaches near white-box True-IDSR at a *tiny* query budget, and
that small white-box-vs-black-box gap in a 2-D realizable space is itself a result.

## A.1 Threat model

The victim is a **query oracle**; the attacker has no gradients and no weights. Two settings:

1. **Score-based** (primary): a query returns the class scores (logits or softmax), so the
   attacker sees the Benign-class score. Realistic for scored APIs / probes.
2. **Decision-based** (harder, secondary): a query returns only the predicted label.

The two-primitive box `[0, p_hi] × [1, alpha_hi]` per flow is unchanged from the white-box
attack; only *how* it is searched changes.

## A.2 Query accounting (define this precisely — it is the headline axis)

All 1024 rows of a `(class, victim)` cell are attacked in one batched forward pass. So:

> **1 query per flow = 1 batched victim forward pass** over the cell.

A population/step method that evaluates `K` candidates per flow per step for `T` steps costs
`K·T` queries per flow. Report success vs. **queries-per-flow budget**
`Q ∈ {10, 25, 50, 100, 250, 500}`.

## A.3 Search methods to implement and compare

All operate on the per-flow controls, either directly in the box `(p, alpha)` or through the
same sigmoid reparameterization used white-box (`p = p_hi·σ(u)`, `alpha = 1+(alpha_hi−1)·σ(v)`).
Keep, per flow, the **best realized** candidate found so far (targeted-Benign success first,
then smallest Benign-margin / lowest cost).

1. **Random search (baseline).** Sample `(u, v)` uniformly per step; evaluate realized; keep
   best. Trivial, sets the floor.
2. **Coordinate / grid search.** Because the space is 2-D, a coarse grid (e.g. 6×6 over the
   box) plus local refinement is extremely query-efficient. Good "no-optimizer" baseline.
3. **NES / SPSA (score-based zeroth-order).** Estimate the gradient of the Benign-score
   objective w.r.t. `(u, v)` from antithetic samples, then take an Adam step — a gradient-free
   mirror of the white-box C&W loop. `K = 2` (forward-diff) or `4` (central) queries/step.
4. **Bayesian optimization (score-based, sample-efficient).** Per flow, a GP/TPE surrogate
   over the 2-D box maximizing the Benign score minus a cost penalty (skopt/BoTorch or a
   lightweight TPE). Best success at very low `Q`; highest implementation cost.
5. **Decision-based boundary search.** Exploit monotonicity: search along the ray from
   identity `(0, 1)` toward the cap `(p_hi, alpha_hi)` by bisection to find the smallest
   control that flips the label. Cheap; the natural decision-only method for a 2-D box.
6. **Transfer attack (optional, cross-model).** Run the *white-box* primitive attack on a
   surrogate victim and apply its realized `(p, alpha)` to the real victim (0 queries on the
   target). Reports transferability across `{mlp, cnn, lstm, serial}`.

## A.3.1 Query-efficiency-first design (PRIMARY — minimize queries)

Query efficiency is the goal, not an afterthought. This problem is far more query-friendly
than generic black-box (image) attacks, and the design should exploit five structural facts:

1. **Only 2 dimensions per flow.** No high-dimensional gradient estimation is needed; a smart
   2-D search converges in *tens* of queries, not thousands.
2. **Batching is free.** All 1024 flows are queried in one forward pass, each with its *own*
   candidate. So one query evaluates one candidate *per flow simultaneously* — per-flow query
   count = number of forward passes, independent of batch size.
3. **Known anchors.** `(p=0, alpha=1)` is the clean flow (known malicious, since the
   denominator is clean-correct) and `(p_hi, alpha_hi)` is the max-effort corner. We start
   with two informative points for free.
4. **Approximate monotone structure.** More padding pushes the length features and more
   dilation pushes the timing/rate features toward the Benign region, so the success set is
   (roughly) an "upper-right" region → **bisection converges in `O(log)` queries.**
5. **Score signal.** Score-based queries expose the Benign margin, enabling bisection on a
   continuous signal rather than only a label flip.

### The primary method: corner-prune + bisection (fewest queries)

```
Query 1  — evaluate the max-effort corner (p_hi, alpha_hi), realized.
           If it does NOT flip to Benign -> this flow is INFEASIBLE within budget.
           Stop. Cost = 1 query, no success. (Prunes hopeless flows immediately.)
Else     — the flow is feasible; find the MINIMAL-cost success by bisection:
           (a) diagonal ray identity->corner, bisect the success boundary  (~7-10 queries), or
           (b) axis-wise: min alpha at p=p_hi, then min p at that alpha     (~2 x log queries).
```

This yields a near-minimal-cost adversarial in **~10–15 queries per feasible flow and exactly
1 query per infeasible flow** — dramatically below any population method. It is also the
*natural* method for the decision-based setting (needs only the label at each probe).

### Efficiency levers (stack these on top of any method)

- **Corner feasibility prune (biggest win).** The 1-query max-corner test removes every flow
  that cannot be evaded within budget from all further querying.
- **Shared coarse grid as a prior, then per-flow local refine.** One batched coarse grid
  (e.g. 4×4 = 16 queries) evaluated for *all* flows at once gives each flow a good starting
  cell; a handful of local bisection steps finish it. Amortized across the batch, the grid is
  nearly free.
- **Early stopping / freeze solved flows.** Once a flow reaches targeted-Benign success at
  acceptable cost, record `queries_used` and stop probing it; keep querying only the unsolved
  subset. (They still ride the batched pass for free, or shrink the batch to unsolved.)
- **Surrogate warm-start (near-zero target queries).** Run the *white-box* attack on a cheap
  surrogate once, use its realized `(p, alpha)` as query 1 on the real victim; usually 1–3
  queries to confirm/refine. Doubles as the transfer experiment (#6).
- **Warm-start across seeds.** Seeds 42/43/44 attack the same rows; seed 42's solution
  warm-starts 43/44 to near-zero extra queries (note this when reporting per-seed variance).
- **Prefer score-based over decision-based** when available — the margin makes bisection
  converge faster and avoids flat-region stalls.

### Efficiency metrics (report these, not just final ASR)

- **Median queries-to-first-success** and **queries-to-min-cost** per flow.
- **Fraction solved within `Q ≤ 10`** (and 25, 50).
- **Infeasible-prune rate:** flows rejected in exactly 1 query.
- **Area under the True-IDSR-vs-`Q` curve** (higher = more query-efficient).
- **Queries saved vs. NES/random** at matched True-IDSR.

### Method ranking by query efficiency (implement in this order)

| Rank | Method | Typical queries/flow | Notes |
|-----:|--------|---------------------:|-------|
| 1 | Corner-prune + bisection | 1 (infeasible) / ~10–15 (feasible) | primary; works score- or decision-based |
| 2 | Coarse grid prior + local refine | ~20–30 (amortized) | robust to non-monotonicity |
| 3 | Bayesian optimization (TPE/GP) | ~20–50 | best on noisy/non-monotone response; higher code cost |
| 4 | NES / SPSA | ~100+ | gradient mirror; least efficient smart method |
| 5 | Random search | 100–500 | floor/baseline only |

Use **#1 as the headline black-box result**, #3 as the sample-efficient comparison, and #4/#5
only as baselines to quantify how many queries the structure-aware methods save.

## A.4 Objective (score-based)

Same targeted intent as white-box, but computed from queried scores (no autograd through the
victim — wrap victim calls in `torch.no_grad()`):

```
maximize   Benign_score(realize(p, alpha))            # targeted -> Benign
minus      cost_weight · normalized_control_magnitude # same soft budget as white-box
```

Decision-based methods use only `argmax == Benign` as the success test.

## A.5 Runner design

New file `src/attack/run_cicids2017_blackbox_primitive_attack.py`, mirroring the white-box
runner. It shares `run(...)`, `evaluate_cell`, `_class_rows`, `_decompose_cost`, provenance,
and artifact schema; it only replaces the inner optimizer:

```python
def optimize_primitives_blackbox(model, victim, raw, center, scale, bounds, *,
                                 method, queries, seed, cost_weight):
    """Query-only search over per-flow (p, alpha); NO gradient through victim.
    method in {random, grid, nes, spsa, bayes, boundary}. Returns realized-best controls."""
    ...
```

CLI additions to the white-box flags: `--method`, `--query-budget`, `--score-mode
{score,decision}`. Everything else (classes, victims, seeds `42,43,44`, `--test-limit 1024`,
`p_max=1460`, `alpha_max=100`, `cost_weight=0.01`) is inherited so results are directly
comparable to the white-box cell. `method_id = f"primitive_blackbox_{method}"`; separate
`--output-dir` (never mixed with the white-box tree).

## A.6 Determinism, provenance, artifacts

- `deterministic_runtime(seed)` per seed; all sampling uses a seeded `torch.Generator` /
  `np.random.default_rng(seed)`.
- Same `build_provenance` + `ensure_fresh_output_dir` + per-cell NPZ, plus new arrays:
  `method`, `query_budget`, `queries_used` (per flow), and `score_mode`.

## A.7 Metrics & comparison

Per `(class, victim, method, Q)` report the standard set from `evaluate_cell`
(`untargeted_asr`, `targeted_benign_asr`, `targeted_strict_valid_asr`, `strict_validity`,
`IDR`, `true_idsr`, cost decomposition) **plus**:

- **Success vs. query-budget curves** (the headline): True-IDSR as a function of `Q`, per
  method, per victim.
- **Query efficiency:** median queries-to-first-success per flow.
- **White-box gap:** black-box True-IDSR at `Q=500` vs. white-box True-IDSR — quantifies how
  much the gradient was actually worth in a 2-D realizable space (the interesting result:
  probably *small*, which is itself a finding).
- **Transferability matrix** (if #6 done): 4×4 surrogate→target True-IDSR.

## A.8 Deliverables

- `run_cicids2017_blackbox_primitive_attack.py` + `attack_results.json` + NPZ artifacts.
- `scripts/plot_blackbox_query_curves.py` (True-IDSR vs. `Q`).
- A results table: white-box vs. each black-box method at matched budgets.

## A.9 Effort / risks

- **Effort:** low–medium. The realizability + evaluation scaffolding already exists; #1–#3, #5
  are short; #4 (BayesOpt) is the only nontrivial piece; #6 reuses the white-box runner.
- **Risk:** per-flow BayesOpt across 1024 flows can be slow — vectorize with a batched TPE or
  cap to NES/grid if needed. Decision-based on flat regions may need more queries — report it
  honestly rather than tuning it away.

---

# Part B — Real-World Cost Curves

## B.1 Why

An `L_p` norm is not what a network attacker pays. The attacker pays **bytes of padding** and
**milliseconds of delay**. Reporting evasion vs. those physical units (Kireev et al.,
cost-aware robustness) is more interpretable and less generic than any norm — and your
artifacts already contain everything needed.

## B.2 Physical-unit definitions (exact, from `generate()`)

For each flow, from the realized vector and the pristine source:

| Physical cost | Formula | Source columns |
|---------------|---------|----------------|
| Per-packet padding | `p` (bytes) | `p` (NPZ) |
| **Total forward bytes added** | `Nf · p` | `Total Fwd Packet` (clean) × `p` |
| Byte overhead ratio | `Nf·p / Total Length of Fwd Packet₀` | clean `TL_fwd` |
| Timing dilation factor | `alpha` | `alpha` (NPZ) |
| **Added forward delay** | `(alpha − 1) · Fwd IAT Total₀` (µs) → /1000 = ms | clean `Fwd IAT Total` |
| **Added flow duration** | `Flow Duration_adv − Flow Duration₀` (µs) → ms | `X_adv_raw`, `X_clean_raw` |
| Duration overhead ratio | `(dur_adv − dur₀) / dur₀` | as above |

These are computed from existing NPZ fields (`p`, `alpha`, `X_clean_raw`, `X_adv_raw`,
`Total Fwd Packet`, `Fwd IAT Total`, `Flow Duration`) — **no re-run needed** for artifacts
already produced; the sweep below only varies the caps.

## B.3 Sweep design

Trace the trade-off by varying the attacker's capability. Run the (white-box and/or
black-box) primitive attack over:

- **Padding-only sweep:** `alpha_max = 1` (timing disabled), `p_max ∈ {0, 64, 128, 256, 512,
  1024, 1460}` bytes.
- **Timing-only sweep:** `p_max = 0` (padding disabled), `alpha_max ∈ {1, 2, 5, 10, 25, 50,
  100}`.
- **Joint sweep:** a coarse 2-D grid of `(p_max, alpha_max)`.
- **Budget-weight sweep (optional):** fix caps, vary `cost_weight ∈ {0, 0.003, 0.01, 0.03,
  0.1, 0.3}` to trace the cost/success frontier from the *penalty* side.

Each sweep point is one attack run (fresh output dir); the existing
`scripts/budget_sweep_primitive.py` is the scaffold to extend.

## B.4 Curves and headline metrics

Per `(class, victim)`:

1. **Cost–success curves.** x-axis = *realized* mean physical cost among evaluated flows
   (bytes added, or ms delay); y-axis = `targeted_benign_asr`, `targeted_strict_valid_asr`,
   and **`true_idsr`**. Prefer realized cost over the cap on the x-axis — it shows what
   successes actually cost, not what was allowed.
2. **Pareto frontier** of (physical cost, True-IDSR): the lower-left envelope across sweep
   points — the cheapest budget achieving each success level.
3. **Headline number:** *minimum bytes / ms for X% True-IDSR* (e.g. "50% True-IDSR needs a
   median of N bytes padding and M ms delay against the LSTM victim").
4. **Padding-vs-timing attribution:** compare padding-only, timing-only, and joint curves —
   which physical axis buys evasion more cheaply, per victim/class.

## B.5 Analysis script

New `scripts/cost_curve_from_artifacts.py`:
- input: one or more attack output dirs (sweep points);
- reads every `attack_artifacts/*.npz`, computes the B.2 physical costs per flow, aggregates
  medians/means over `clean_correct` (and separately over successful) flows;
- emits a tidy CSV (`class, victim, method, p_max, alpha_max, cost_weight, bytes_added_median,
  delay_ms_median, targeted_benign_asr, targeted_strict_valid_asr, true_idsr`) and the plots
  (cost–success curves + Pareto frontier per victim).

Build on existing `scripts/analyze_primitive_attack.py` for artifact loading conventions.

## B.6 Deliverables

- Extended `scripts/budget_sweep_primitive.py` (parameter grids above).
- `scripts/cost_curve_from_artifacts.py` + the tidy CSV.
- Figures: cost–success curves (bytes and ms) and the Pareto frontier, per victim/class.
- One summary table: minimum physical budget for 25/50/75% True-IDSR per victim.

## B.7 Effort / risks

- **Effort:** low. Mostly analysis over existing NPZ + a cap grid; no new attack math.
- **Risk:** realized-cost x-axis is noisier than the cap; report both (cap on x, realized cost
  as annotation) if the frontier is jumpy. Ensure padding-only / timing-only runs actually
  disable the other primitive (`p_max=0` / `alpha_max=1`) rather than relying on the penalty.

---

## Milestones (suggested order)

1. **B first (cheapest, immediate payoff):** implement `cost_curve_from_artifacts.py` on the
   *already-produced* white-box artifacts → physical-cost curves with zero new attack runs.
2. Extend the budget sweep (padding-only / timing-only / joint) → full Pareto frontiers.
3. **A next (efficiency-first, §A.3.1):** implement `optimize_primitives_blackbox` starting
   with the **corner-prune + bisection** primary method (fewest queries), then `grid` prior +
   local refine, then `random`/`nes` as baselines; produce True-IDSR-vs-`Q` curves, the
   efficiency metrics (median queries-to-success, `Q≤10` solved fraction, infeasible-prune
   rate), and the white-box gap.
4. Add BayesOpt (#4) and the transfer matrix (#6) if time allows.
5. Re-run the cost-curve analysis on the black-box artifacts → cost curves *and* query curves
   together.

## Acceptance criteria

- Black-box runner reproduces the white-box **row set, victims, validators, and True-IDSR**
  exactly, differing only in the optimizer; results in a separate output tree with full
  provenance.
- Query-budget curves for ≥3 methods vs. the white-box reference, per victim.
- Efficiency metrics reported: median queries-to-first-success, fraction solved at `Q≤10`,
  infeasible-prune rate, and area under the True-IDSR-vs-`Q` curve — with the corner-prune +
  bisection method beating random/NES at matched True-IDSR.
- Physical cost–success curves + Pareto frontier in **bytes and milliseconds**, with the
  "minimum budget for X% True-IDSR" table.
- Every reported number is recomputable from the NPZ artifacts (no metric computed only in
  memory).
