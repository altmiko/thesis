# PrimAttack Optimizer Documentation

## 1. Purpose and scope

PrimAttack searches a small primitive-domain control space and realizes every retained candidate
through `CICIDS2017PrimitiveModel`. The current optimizer family contains Hybrid Search,
Prim-PGD, Prim-C&W, and the Prim-Random null control. All four use the shared
`RealizedSearch` contract in `src/attack/primitive_optimizer.py`, including the same per-flow
box, canonical transform $\phi$, projection, quantized victim scoring, incumbent ordering, and
query accounting.

The attack operates on aggregate CICFlowMeter feature vectors. Its outputs are feature-space
proxies. The implementation does not edit or replay a PCAP, establish packet-level realizability,
observe target state, or verify preservation of malicious functionality. Results apply to the
evaluated frozen victims, source flows, primitive boxes, and aggregate representation.

## 2. Shared primitive search space

For source flow $x_i$, the requested control is

$$
u_i=(p_i,D_i,s_i),
$$

where $p$ is an increase-only number of bytes added uniformly to every represented forward
packet, $D$ (`delay`) is the total added forward inter-arrival delay in microseconds, and $s$
(`shape`) allocates that fixed delay between proportional and uniform gap changes. The projected
$p$ and $D$ are non-negative integers. The projected $s$ lies in $[0,s_i^{\mathrm{hi}}]$, normally
$[0,1]$, and is forced to zero when $D=0$.

Let $G=\max(N_f-1,1)$ be the represented number of forward gaps, let $\Delta_j$ be gap $j$, and
let $T_f$ be their original total. Delay allocation uses

$$
a=1+(1-s)\frac{D}{T_f},
\qquad
b=s\frac{D}{G},
\qquad
\Delta_j'=a\Delta_j+b.
$$

Thus $\sum_j\Delta_j'=T_f+D$. Changing $s$ redistributes a selected delay; it does not add delay
and has zero direct primitive cost.

Each flow receives a hard box

$$
0\le p_i\le p_i^{\mathrm{hi}},\qquad
0\le D_i\le D_i^{\mathrm{hi}},\qquad
0\le s_i\le s_i^{\mathrm{hi}}.
$$

`CICIDS2017PrimitiveModel.per_flow_bounds` intersects the selected train-calibrated class budget,
train-p99 feature-envelope headroom, capability gates, and the DoS/DDoS rate floor. The
`unbounded` condition removes the class-level padding and relative-duration caps while retaining
the envelope, capabilities, integer projection, and rate floor.

Padding capability requires represented forward payload and no zero-length forward packet:

$$
\chi_p=[N_f\ge1]\land[L_f>0]\land[\bar l_f>0]\land[\min l_f>0].
$$

Timing capability requires a forward IAT sequence:

$$
\chi_D=[N_f\ge2]\land[T_f>0].
$$

The runner applies `joint`, `timing-only`, or `padding-only` by zeroing the excluded primitive's
upper bound. `row_primitive_modes` then records the effective row space after capabilities,
budget headroom, and this global restriction have been folded into the box:

| Row space | Integer headroom |
|---|---|
| `joint` | $p_i^{\mathrm{hi}}\ge1$ and $D_i^{\mathrm{hi}}\ge1$ |
| `timing-only` | $p_i^{\mathrm{hi}}<1$ and $D_i^{\mathrm{hi}}\ge1$ |
| `padding-only` | $p_i^{\mathrm{hi}}\ge1$ and $D_i^{\mathrm{hi}}<1$ |
| `no-primitive` | both upper bounds are below one integer unit |

A coordinate with less than one integer unit of headroom is pinned to zero. Timing-only rows
devote gradient steps and random samples to timing coordinates without carrying a dead padding
axis. No-primitive rows retain the identity.

Optimization uses normalized controls

$$
q_i=(q_{p,i},q_{D,i},q_{s,i})\in[0,1]^3,
$$

with the decoding

$$
u_i(q_i)=
\left(p_i^{\mathrm{hi}}q_{p,i},
D_i^{\mathrm{hi}}q_{D,i},
s_i^{\mathrm{hi}}q_{s,i}\right).
$$

Pinned coordinates are zeroed in $q_i$ before starts and updates.

## 3. Objectives, realization, and incumbent selection

Let $z_k(x)$ be victim logit $k$. For a targeted attack toward class $t$, normally Benign with
$t=0$, the minimized margin is

$$
\mathcal M_{\mathrm{tgt}}(x;t)=\max_{k\ne t}z_k(x)-z_t(x).
$$

For an untargeted attack leaving source class $y$, it is

$$
\mathcal M_{\mathrm{untgt}}(x;y)=z_y(x)-\max_{k\ne y}z_k(x).
$$

A negative margin implies that the objective is met. At an exact zero-margin tie,
`torch.argmax` resolves the hit according to class-index order. The authoritative hit tests are
therefore $\arg\max_k z_k(x_i')=t$ for targeted attacks and
$\arg\max_k z_k(x_i')\ne y$ for untargeted attacks; the margin sign is not used as the hit test.

Every requested candidate $u_i$ is projected and regenerated before it can update the incumbent:

$$
\hat u_i=\Pi_{\mathcal B_i}(u_i),
\qquad
x_i'=\phi(x_i,\hat u_i;\texttt{quantize=True}).
$$

The normalized cost of the projected control is

$$
C_i=\begin{cases}\hat p_i/p_i^{\mathrm{hi}},&p_i^{\mathrm{hi}}>0\\0,&\text{otherwise}
\end{cases}
+
\begin{cases}\hat D_i/D_i^{\mathrm{hi}},&D_i^{\mathrm{hi}}>0\\0,&\text{otherwise}.
\end{cases}
$$

`shape` is absent from $C_i$ because it only reallocates $D_i$.

The code ranks candidates by its internal `success` flag. In the executed FINAL runs,

$$
\texttt{success}=\texttt{hit}\land\texttt{hybrid_valid};
$$

with `validity_fn=None`,

$$
\texttt{success}=\texttt{hit}.
$$

The per-row order is lexicographic: internal success replaces failure; successful candidates are
ranked by lower $C_i$ and then lower objective margin; failed candidates are ranked by lower
objective margin. Hybrid's padding early stop and Prim-C&W's stage update use the same internal
success flag. Amendment A6 found that the gated and hit-only configurations returned identical
final flows, so report-level descriptions may state the equivalent hit-only order. Section 11
sets out that interpretation and its boundary.

## 4. Continuous surrogate and realized scoring

Gradients use a continuous, unquantized surrogate. The canonical map copies the source value on
a coordinate whose control is exactly zero, which gives zero gradient at the identity. For every
free normalized coordinate, surrogate evaluation uses the straight-through floor $f=10^{-3}$:

$$
q_{i,j}^{\mathrm{eval}}
=q_{i,j}+\operatorname{stopgrad}\!\left(\max(f-q_{i,j},0)\right).
$$

Its forward value is $\max(q_{i,j},f)$ and its gradient with respect to $q_{i,j}$ is one. The
surrogate control and flow are

$$
u_i^{\mathrm{sur}}=u_i(q_i^{\mathrm{eval}}),
\qquad
\tilde x_i=\phi(x_i,u_i^{\mathrm{sur}};\texttt{quantize=False}).
$$

Incumbents are updated only from the projected, quantized $x_i'$. A hit that exists only in the
continuous relaxation cannot be returned. Realized scoring never receives the floor.

Coordinates without integer headroom are detached and pinned at zero, giving an exactly zero
gradient. `objective_gradient` raises on any remaining non-finite gradient. Momentum
normalization uses the mean absolute gradient over free coordinates only, so a timing-only row is
unaffected by a pinned padding coordinate.

## 5. Per-flow query accounting

`RealizedSearch` stores three counters for every source flow:

- `realized_evaluations` counts forward evaluations of projected, quantized candidates;
- `surrogate_evaluations` counts forward evaluations of the continuous relaxation;
- `backward_evaluations` counts gradient evaluations.

The derived forward-evaluation count is

$$
E_i^{\mathrm{total}}
=E_i^{\mathrm{realized}}+E_i^{\mathrm{surrogate}},
$$

and the FINAL cap is

$$
E_i^{\mathrm{total}}\le256.
$$

Backward evaluations are recorded separately and excluded from this cap. The identity consumes
one realized forward evaluation. A gradient iteration consumes one surrogate forward, one
backward evaluation, and one realized forward. A row may start a gradient step only when at
least two forward evaluations remain. Batched victim calls do not alter per-row charging.

`first_objective_hit_evaluation` records the one-based forward-evaluation index of the first
realized hit. `first_success_evaluation` records the first internal success under the supplied
`validity_fn`. `first_success_phase` records its restart or binary-search stage, using the
implementation's sentinel values for identity, exact-padding, and no success.

## 6. Hybrid Search

`optimize_primitive_candidates` has three stages.

1. It scores the exact identity $(0,0,0)$.
2. It enumerates $p=1,\ldots,\lfloor p_i^{\mathrm{hi}}\rfloor$ with $D=s=0$. Values are ordered
   by increasing cost. A row stops this sweep at its first internal success or when padding
   headroom or query budget is exhausted. Batched candidates after the first success are
   discarded and uncharged, preserving sequential semantics.
3. Rows still unresolved with $D_i^{\mathrm{hi}}\ge1$ enter adaptive timing/shape refinement.

Let $F_i$ be the free-coordinate set and let
$\gamma_i=\nabla_{q_i}\mathcal M(\tilde x_i)$ be the surrogate margin gradient. Refinement uses

$$
v_i\leftarrow0.75v_i+
\frac{\gamma_i}
{\max\left(|F_i|^{-1}\sum_{j\in F_i}|\gamma_{i,j}|,10^{-12}\right)},
$$

$$
q_i\leftarrow\operatorname{mask}_{F_i}
\left(\operatorname{clip}_{[0,1]}
(q_i-\eta_i\operatorname{sign}(v_i))\right).
$$

Restart 0 begins at the best identity or exact-padding control. Later restarts begin from seeded
uniform samples in the normalized box. Every `max(5, steps // 4)` iterations, a stalled row
halves its step size, returns to its best realized point within that restart, and clears momentum.
With `restarts=None`, the runner starts further restarts until refined rows can no longer afford
a full gradient step under the query cap.

```text
score identity
for p in increasing legal integers:
    score realized (p, 0, 0)
for each clean or random restart while budget permits:
    decode floored q and differentiate the unquantized flow
    apply the projected sign-momentum update
    score the projected and quantized candidate
    adapt the step size when realized margin stalls
return the lexicographic incumbent
```

## 7. Prim-PGD

`optimize_primitive_pgd` performs fixed-step projected sign-momentum descent on the objective
margin. Restart 0 starts from $q=0$; later restarts use seeded uniform normalized controls. Each
iteration applies the shared free-coordinate gradient scaling and

$$
v_i\leftarrow\mu v_i+
\frac{\gamma_i}
{\max\left(|F_i|^{-1}\sum_{j\in F_i}|\gamma_{i,j}|,10^{-12}\right)},
$$

$$
q_i\leftarrow\operatorname{mask}_{F_i}
\left(\operatorname{clip}_{[0,1]}
(q_i-\alpha\operatorname{sign}(v_i))\right).
$$

Prim-PGD has no exact padding sweep, stall-triggered step change, or reset to a restart-best
point. Realized candidates still pass through the shared projection, quantized transform, and
incumbent logic.

## 8. Prim-C&W

`optimize_primitive_cw` uses projected Adam on

$$
L(q_i)=q_{p,i}\mathbf 1[p_i^{\mathrm{hi}}\ge1]
+q_{D,i}\mathbf 1[D_i^{\mathrm{hi}}\ge1]
+c_i\max(\mathcal M(\tilde x_i)+\kappa,0).
$$

Each binary-search stage restarts from the clean point. A stage with an internal success updates
the upper bracket for $c_i$; a stage without one updates the lower bracket. The next $c_i$ is the
bracket midpoint when an upper bracket exists and $10c_i$ otherwise. The returned flow is the
shared realized incumbent, independent of the final Adam iterate.

## 9. Prim-Random null control

`optimize_primitive_random` scores the identity and then draws independent normalized controls
uniformly from the continuous box. Pinned coordinates remain zero. Every draw is projected and
quantized before scoring. Under the 256 cap, a movable row receives 255 random realized
candidates, zero surrogate evaluations, and zero backward evaluations.

Uniform sampling occurs before integer projection. The resulting discrete $p$ and $D$ values are
not uniformly distributed over legal integers because rounding gives different pre-image widths
at boundary and interior values.

Prim-Random is a multi-query null control. `random_feasible_primitives` in
`src/attack/run_cicids2017_primitive_attack.py` draws one random feasible control per flow in
older/campaign controls. The two names must not be used interchangeably.

## 10. Optimizer comparison and locked FINAL settings

| Method | Gradient search | Starts or stages | Distinctive behavior |
|---|---|---|---|
| Hybrid Search | Sign-momentum margin descent | Clean refinement plus random restarts until budget | Exact padding enumeration and adaptive stall recovery |
| Prim-PGD | Sign-momentum margin descent | Three restarts | Fixed step and no padding enumeration |
| Prim-C&W | Adam on cost-plus-margin | Three $c$ stages | Per-flow binary search over the trade-off coefficient |
| Prim-Random | None | 255 draws after identity | Multi-query uniform continuous-box null control |

For $B=256$, `method_configs` computes the PGD and C&W step count as
$\lfloor(B-1)/(2\times3)\rfloor=42$.

| Method | Exact FINAL hyperparameters |
|---|---|
| Hybrid Search | `steps=40`, `learning_rate=0.1`, momentum `0.75`, checkpoint interval `10`, stall factor `0.5`, `restarts=None`, `eval_budget=256` |
| Prim-PGD | `restarts=3`, `steps=42` per restart, `step_size=0.05`, momentum `0.75`, `eval_budget=256` |
| Prim-C&W | `stages=3`, `steps=42` per stage, `learning_rate=0.5`, `c_init=1.0`, `kappa=0.0`, Adam betas `(0.9, 0.999)`, `eval_budget=256` |
| Prim-Random | 255 realized draws after identity for movable rows, uniform normalized continuous box, `eval_budget=256` |

`scripts/tune_primattack_ablation_baselines.py` tuned only Prim-PGD and Prim-C&W on the
CICIDS2017 validation split. Prim-PGD tested step sizes $\{0.02,0.05,0.1,0.2\}$. Prim-C&W
tested $c_0\in\{1,10,100\}$ crossed with learning rates
$\{0.02,0.05,0.1,0.2,0.5\}$. Every setting used the same per-flow evaluation budget; selection
maximized pooled valid ASR, with lower median normalized cost as the tie-break. Hybrid retained
its canonical settings. The chosen values were frozen and transferred to CICIDS2018 without
re-tuning.

The canonical rule selected the highest aggregate Valid Targeted ASR at p75 across both datasets,
all victims, classes, and attack seeds, followed by lower mean victim evaluations and then the
fixed order Hybrid, Prim-PGD, Prim-C&W. `FINAL_OUTPUTS/runs/optimizer_selection.json` records
1,752 valid targeted successes from 57,600 attempts for both Prim-PGD and Hybrid. Their aggregate
rate was $0.0304166667$. Prim-PGD used 188.50375 mean evaluations per flow, compared with
189.644375 for Hybrid, so Prim-PGD was selected. Prim-C&W recorded 861 successes from 57,600
attempts and ranked third.

## 11. Amendment A6 and validator_v2

The implementation can receive a validity gate, and the executed FINAL runs used the default
`hybrid_valid_gate(dataset)`. The A6 ablation reran every executed FINAL PrimAttack configuration
with `--no-validity-gate`, making internal search success equal to the realized objective hit
alone. The 252 victim × seed × configuration cells, comprising 806,400 flow attacks, were drawn
from the executed combinations of Hybrid, Prim-PGD, Prim-C&W, and Prim-Random; targeted and
untargeted objectives; p50, p75, and unbounded boxes; joint, timing-only, and padding-only modes;
both datasets; and attack seeds 42, 2024, and 2026. This was a rerun of the executed stage-specific
cells, rather than a full factorial crossing of every listed factor.

The no-gate rerun produced bit-identical final adversarial flows for all 806,400 attacks, kept
zero invalid hits, left every Raw and Valid ASR unchanged, and produced zero discordant flows in
every victim-level McNemar comparison. The report therefore uses the equivalent hit-only
incumbent description and treats validator_v2 as a post-attack assessment performed by
`evaluate_cell`, matching its effective use for the baselines. PrimAttack must not be described
as a validity-aware search, as receiving a validator access advantage, or as creating a
validator-related threat-model asymmetry.

A6 isolates validator access only. It does not equalize parameterization, the capability-aware
box, objective, representation, or query allocation, and it does not identify any one of these
as the cause of differences between methods. The code default remains gated; the hit-only wording
is justified by the complete A6 equivalence result.

## 12. Outputs, diagnostics, and implementation map

`scripts/run_primattack_optimizer_ablation.py` writes `config.json`, the frozen
`selection.json`, `cells.json`, and one compressed NPZ per
`(victim, class, budget, seed, method, mode)` under `artifacts/`. The NPZ stores the projected
`p`, `delay`, and `shape`; the `p_hi` and `delay_hi` bounds; `adv_raw`; predictions; raw,
validator, and valid-success masks; row capability and mode fields; normalized cost and margin;
candidate source; `realized_evaluations`, `surrogate_evaluations`, and
`backward_evaluations`; first-hit indices; and primitive-feasibility and semantic diagnostics.
Requested controls and a `shape_hi` field are not written to the NPZ. `cells.json` supplies
cell-level counts, rates, cost summaries, query summaries, failure categories, and consistency
checks.

| Responsibility | Source |
|---|---|
| Shared objectives, realization, incumbent, accounting, and four optimizers | `src/attack/primitive_optimizer.py` |
| Controlled runner, exact method configs, artifacts, diagnostics, and `--no-validity-gate` | `scripts/run_primattack_optimizer_ablation.py` |
| Locked FINAL stages, arguments, and optimizer-selection rule | `scripts/run_final_suite.py` |
| Validation-only PGD/C&W grid search | `scripts/tune_primattack_ablation_baselines.py` |
| Locked protocol and amendment A6 | `FINAL_OUTPUTS/00_PROTOCOL.md` |
| Canonical selection result | `FINAL_OUTPUTS/runs/optimizer_selection.json` |
| Primitive model, projection, and canonical transform | `src/attack/realizability/cicids2017.py` |
| Post-attack predictions, validator masks, and consistency masks | `src/attack/run_cicids2017_primitive_attack.py:evaluate_cell` |
| Optimizer regression invariants | `src/attack/tests/test_primitive_optimizer.py` |

## 13. Verified invariants and report-safe limitations

`src/attack/tests/test_primitive_optimizer.py` checks regeneration of the returned flow from its
requested and projected controls, integer hard-box compliance, targeted and untargeted realized
hit semantics, per-flow budget compliance, padding exclusion on flows with an empty forward
packet, active timing search on timing-only rows, full-cap realized sampling by Prim-Random,
Hybrid dominance over identity and every enumerated padding candidate, and finite gradients with
an exactly zero pinned-padding derivative. The broader methodology records tests for the
canonical transform's dependency sets, timing and padding equations, frozen columns, and
train-only budget artifact.

These invariants establish behavior within the aggregate primitive model. Sequence-dependent
fields unavailable from one flow row remain held constant, and conservative reconstructions such
as duration and merged-flow timing are not packet-trace re-extractions. The p50 and p75 boxes are
empirical class-conditional budgets; the envelope-only `unbounded` box still has hard limits.
Victim results must be reported separately, and attack-seed variation does not establish victim
training-seed robustness. Claims of PCAP validity, packet-level realizability, preserved malicious
functionality, deployment behavior, target-state feedback, or universal generalization exceed the
implemented and evaluated contract.
