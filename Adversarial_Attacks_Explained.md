# Adversarial Attacks Explained: PGD, C&W, C-PGD and CAPGD

This document explains how the four comparison attacks in the FINAL thesis suite work. It is
written for the thesis report and follows the code, not the literature defaults. Where the
implementation differs from the original paper, the difference is stated. The configurations
are locked in `FINAL_OUTPUTS/00_PROTOCOL.md` §4 and run by `scripts/run_final_suite.py` →
`scripts/run_full_adversarial_eval.py`.

| Report name | Code attack id | Implementation |
|---|---|---|
| PGD (raw / unconstrained) | `pgd_untargeted` | `src/attack/input_baselines.py:input_pgd_attack` |
| C&W (raw / unconstrained) | `cw_untargeted` | `src/attack/input_baselines.py:input_cw_attack` |
| C-PGD-PrimSupport | `cpgd_prim_support` | `src/comparisons/cpgd_prim_support.py:CPGDPrimSupportAttack` |
| CAPGD-PrimSupport | `capgd_prim_support` | vendored `external/tabularbench` CAPGD via `src/comparisons/capgd_cicids2017.py` |
| CAPGD (native), descriptive only (amendment A3) | `capgd_native` | same as above, different feature mask |

---

## 0. Shared setting

### 0.1 Notation

- $x \in \mathbb{R}^{79}$: a raw (unscaled) CICFlowMeter flow vector in the frozen feature order
  (`preprocessing_manifest.json:modelling_feature_names`).
- $y$: its true category, one of Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4.
- $f$: the frozen victim (MLP, CNN or FT-Transformer, category head). It returns logits
  $z = f(\cdot) \in \mathbb{R}^5$.
- $\mathcal{L}_{\text{CE}}(z, y)$: cross-entropy.
- $x'$: the adversarial flow that an attack returns.

### 0.2 Three coordinate systems

The attacks do not all operate in the same space. This matters when reading $\varepsilon$ values.

1. **Raw space.** CICFlowMeter units: bytes, microseconds, counts, rates. validator_v2 judges
   flows here.
2. **Victim (RobustScaler) space.**
   $\tilde{x} = (x - \mathrm{median}_{\text{train}}) / \mathrm{IQR}_{\text{train}}$.
   The victims take $\tilde{x}$ as input and apply $\operatorname{asinh}$ internally
   (`input_transform="asinh"` in `classifiers/models.py` and `ft_transformer.py`).
   **PGD and C&W attack in this space.**
3. **Train min–max space.**
   $\hat{x} = (x - x_{\min}) / (x_{\max} - x_{\min})$, where the per-feature bounds are
   fitted on the pristine TRAIN split only (`capgd_cicids2017.py:fit_train_minmax`). The train
   data therefore lie in $[0,1]^{79}$. **CAPGD and C-PGD attack in this space.** Their victim
   wrapper `RawCICIDSVictim` maps $\hat{x} \to x \to \tilde{x} \to f$.

An $\varepsilon = 0.5$ therefore means different things for different attacks:

- For PGD it is half an interquartile range, applied independently to every feature ($L_\infty$).
- For CAPGD and C-PGD it is an $L_2$ radius of 0.5 in units of the train range.

### 0.3 Common protocol

- **Same inputs.** All attacks are run on the same frozen list of 800 clean-correct test flows
  per (dataset, victim, class), with attack seeds 42/2024/2026.
- **Untargeted in the FINAL suite (Exp A).** Every attack here maximises the true-class loss.
  Success means $f(x') \neq y$.
- **Validity is checked afterwards.** None of these four attacks sees validator_v2 while
  optimising. Every final $x'$ is gated afterwards by
  $V = \texttt{hybrid\_valid}(x' \mid x)$. This asymmetry is part of the threat-model
  difference to PrimAttack, whose search success predicate does include validator_v2
  (`00_PROTOCOL.md` §4).
- **Metrics.** $\mathrm{ASR}_{\text{raw}} = \frac{1}{N}\sum_i S_i$ and
  $\mathrm{ASR}_{\text{valid}} = \frac{1}{N}\sum_i S_i V_i$, both over the $N$ attempted
  clean-correct flows.

### 0.4 What each attack is allowed to change

| Attack | Mutable features | Box | Integer types | Relations between features |
|---|---|---|---|---|
| PGD | all 79 | none | ignored | ignored |
| C&W | all 79 | none | ignored | ignored |
| C-PGD-PrimSupport | 23 (PrimAttack write support) | train $[0,1]$ | repaired once at the end | penalty in the loss only |
| CAPGD-PrimSupport | 23 (PrimAttack write support) | train $[0,1]$ | repaired | penalty in the loss **and** equality repair every iteration and at the end |
| CAPGD (native) | 16 (9 perturbable + 7 exactly recomputed) | train $[0,1]$ | repaired | penalty + repair, then exact recompute of derived features |

**The 23-feature "PrimSupport" mask** is `primattack_joint_feature_mask`
(`attack/realizability/cicids2017.py`). It contains exactly the classifier coordinates that
PrimAttack's canonical transform $\varphi$ can write. PrimAttack reaches these features only
through its two controls: padding $p$ and forward delay (delay, shape). The baselines instead
move the same features directly and independently:

- 12 padding-driven features: Total Length of Fwd Packet; Fwd Packet Length Min, Max and Mean;
  Fwd Segment Size Avg; Packet Length Min, Max, Mean, Variance and Std; Average Packet Size;
  Flow Bytes/s.
- 12 timing-driven features: Fwd IAT Total, Mean, Std, Max and Min; Flow Duration;
  Flow IAT Mean and Max; Flow Bytes/s; Flow Packets/s; Fwd Packets/s; Bwd Packets/s.
- Flow Bytes/s is in both lists, which gives $12 + 12 - 1 = 23$.

Matching the *feature support* is not the same as matching the *feasible set*. A baseline can
put these 23 coordinates in combinations that no packet-level padding or delay could produce.

---

## 1. Unconstrained PGD (raw PGD)

**Idea.** Projected Gradient Descent (Madry et al., 2018) is the standard first-order
white-box attack. Starting near the clean point, it repeatedly takes a signed-gradient step
that increases the victim's loss on the true class. After each step it projects back into an
$L_\infty$ ball around the original input. "Raw" or "unconstrained" means that nothing except
the ball limits the attack: there is no feature mask, no box, no integer typing and no
protocol constraint.

**Formulation.** In victim space, with $\tilde{x}_0$ the scaled clean flow:

$$
\max_{\tilde{x}'} \; \mathcal{L}_{\text{CE}}\big(f(\tilde{x}'), y\big)
\quad \text{s.t.} \quad \|\tilde{x}' - \tilde{x}_0\|_\infty \le \varepsilon .
$$

**Algorithm (as implemented).**

1. Random start: $\tilde{x}^{(0)} = \tilde{x}_0 + u$ with $u \sim \mathcal{U}[-\varepsilon, \varepsilon]^{79}$,
   clipped to the ball.
2. For $t = 0, \dots, T-1$:
   $$
   \tilde{x}^{(t+1)} = \Pi_{B_\infty(\tilde{x}_0,\varepsilon)}\!\Big(\tilde{x}^{(t)} + \alpha \cdot \operatorname{sign}\big(\nabla_{\tilde{x}} \mathcal{L}_{\text{CE}}(f(\tilde{x}^{(t)}), y)\big)\Big)
   $$
   Here $\Pi$ is element-wise clipping to $[\tilde{x}_0 - \varepsilon, \tilde{x}_0 + \varepsilon]$.
3. Return the **final iterate** (not the best one seen). Map back to raw space:
   $x' = \tilde{x}^{(T)} \cdot \mathrm{IQR} + \mathrm{median}$.

**Locked configuration.** $L_\infty$, $\varepsilon = 0.5$, $\alpha = 0.05$, $T = 40$ steps,
1 random start, untargeted CE, fixed number of steps. This costs 40 victim evaluations per flow.

**Why it breaks domain validity.** Every one of the 79 features moves by up to half an IQR,
in whatever direction the gradient points. Examples of what can happen:

- Packet counts become fractional or negative.
- Frozen fields such as ports, protocol, TCP flags and backward statistics change.
- Means stop equalling totals divided by counts.
- Rates stop equalling counts divided by duration.
- Min > mean or mean > max can occur.

Nothing pulls the sample back towards a realisable flow, so validator_v2 rejects essentially
all outputs. PGD therefore marks the upper bound: "the classifier is not robust in feature
space". It says nothing about evasion with realisable traffic.

---

## 2. Unconstrained C&W (raw C&W)

**Idea.** Carlini & Wagner (2017) pose the attack as an optimisation problem. The goal is the
smallest perturbation that makes the true class lose its lead in the logits. There is no hard
$\varepsilon$ ball. Instead, a margin term pushes towards misclassification and an $L_2$
penalty keeps the perturbation small.

**Formulation (as implemented).** In victim space, optimising a perturbation $\delta$:

$$
\min_{\delta} \; \lambda \cdot \max\!\Big( z_y(\tilde{x}_0 + \delta) - \max_{j \neq y} z_j(\tilde{x}_0 + \delta) + \kappa,\; 0 \Big) \;+\; \|\delta\|_2^2 .
$$

- The margin term is positive while the true-class logit $z_y$ still beats the best other
  class. It becomes 0 once some other class leads by at least $\kappa$.
- $\|\delta\|_2^2$ penalises the size of the change.
- $\lambda$ trades off the two terms.

**Algorithm (as implemented).**

1. $\delta^{(0)} = 0$. Optimiser: Adam, learning rate 0.01.
2. At each iteration: compute the loss, take an Adam step, then re-evaluate the victim.
3. A flow counts as fooled when $z_y - \max_{j\neq y} z_j \le 0$. Among fooled iterates, keep
   the one with the **smallest $\|\delta\|_2$** seen so far.
4. Stop after the iteration budget, or earlier when every row's $\delta$ moved by less than
   $10^{-5}$ in one iteration.
5. Return the lowest-$L_2$ successful $\delta$ if one exists, otherwise the final $\delta$.
   Map back to raw space as for PGD.

**Locked configuration.** $\lambda = 1$, $\kappa = 0$, at most 60 Adam steps, lr 0.01,
convergence threshold $10^{-5}$, 1 run. This costs 2 victim evaluations per iteration.

**Differences from the original C&W** (state these in the thesis):

- There is no binary search over the trade-off constant ($\lambda$ is fixed at 1).
- There is no $\tanh$ change of variables for a box constraint, because there is no box at all.
- The number of iterations is small (60).

This is a fixed-constant C&W-$L_2$ variant used as an unconstrained minimum-perturbation
baseline, not the full original procedure.

**Why it breaks domain validity.** The reasons are the same as for PGD. Every feature is free,
types are ignored, and inter-feature identities are not modelled. The $L_2$ penalty makes the
perturbation small, but small is not the same as consistent. Changing a mean without changing
the corresponding total still produces an impossible flow.

---

## 3. C-PGD on PrimAttack's support (C-PGD-PrimSupport)

**Idea.** Constrained PGD (Simonetto et al., IJCAI 2022, "A Unified Framework for Adversarial
Attack and Defense in Constrained Feature Space", Eq. 2) adds domain knowledge to PGD in two
ways:

- **Hard limits** on what may change: a mutable-feature mask, a value box and integer types.
- **Soft relations** between features, as a differentiable penalty subtracted from the attack
  objective.

The attack is thus pushed towards inputs that both fool the model and satisfy the encoded
relations. Here it is restricted to PrimAttack's 23-feature write support, so that it moves
the same features PrimAttack can influence.

**Formulation.** In train min–max space, with mutable mask $M$ (the 23 features) and clean
point $\hat{x}_0$:

$$
\max_{\hat{x}'} \; \mathcal{L}_{\text{CE}}\big(f(\hat{x}'), y\big) \;-\; \lambda \cdot \frac{1}{B}\sum_{b}\,\mathrm{viol}\big(x'_b\big)
\quad \text{s.t.} \quad
\|\hat{x}' - \hat{x}_0\|_2 \le \varepsilon,\;\;
\hat{x}'_{\neg M} = \hat{x}_{0,\neg M},\;\;
\hat{x}' \in [0,1] .
$$

$\mathrm{viol}(x')$ is the sum, evaluated in **raw** units, of the violation magnitudes of
the encoded relations (`capgd_cicids2017.py:_relations`):

- **Equalities**, with violation $\max(|a - b| - \text{tol}, 0)$:
  - $\text{Fwd Packet Length Mean} = \text{Total Length of Fwd Packet} / \text{Total Fwd Packet}$
  - $\text{Fwd Segment Size Avg} = \text{Fwd Packet Length Mean}$
  - $\text{Fwd IAT Mean} = \text{Fwd IAT Total} / (\text{Total Fwd Packet} - 1)$
  - $\text{Fwd Packets/s}$, $\text{Bwd Packets/s}$, $\text{Flow Packets/s}$ and
    $\text{Flow Bytes/s}$ each equal the relevant count or byte total $\times 10^6 / \text{Flow Duration}$
- **Inequalities**, with violation $\max(a - b, 0)$:
  - Fwd Packet Length Min $\le$ Mean $\le$ Max
  - Fwd IAT Min $\le$ Mean $\le$ Max

**Algorithm (as implemented, `CPGDPrimSupportAttack.run`).**

1. **Random start inside the masked $L_2$ ball.** Draw Gaussian noise on the mutable features,
   normalise it, and scale it to radius $r \cdot \varepsilon$ with $r \sim \mathcal{U}[0,1]$.
   Then project.
2. For 40 iterations:
   1. Compute the objective $\mathcal{L}_{\text{CE}} - \lambda \cdot \overline{\mathrm{viol}}$
      and its gradient with respect to $\hat{x}'$.
   2. Zero the gradient on frozen features and $L_2$-normalise it per row.
   3. Step: $\hat{x}' \leftarrow \hat{x}' + \eta \cdot g / \|g\|_2$.
   4. **Project.** Restrict the change to $M$ and rescale it onto the $L_2$ ball of radius
      $\varepsilon$. Clip to the train box $[0,1]$; if a held-out clean value already lies
      outside the box, the clean value is kept as the endpoint rather than being forced into
      the box. Restore frozen features.
3. **Type repair, once at the end.**
   1. Map back to raw space. For integer features (`int` in the validator_v2 schema profile),
      round the perturbation *towards zero* (TabularBench `fix_types`, using `torch.fix`).
      Restore frozen features.
   2. If the rounding pushed a row outside the $\varepsilon$ ball, reset that row's integer
      features to their clean values and re-project only its continuous features.
   3. Hard asserts check that $\varepsilon$ holds and that nothing outside the 23-feature
      support changed.

**Locked configuration.** $L_2$, $\varepsilon = 0.5$, step size $\eta = 0.05$, 40 iterations,
$\lambda = 1$, 1 random start, CE loss, fixed number of steps. This costs 40 victim evaluations
per flow.

**Key property: the constraints are only encouraged.** C-PGD never *enforces* the equality
relations. The penalty only discourages violations, and $\lambda = 1$ fixes how much weight
they get relative to the classification loss. There is no repair step. In addition:

- The encoded relations cover only the forward-length, forward-IAT and rate identities listed
  above.
- The other coupled features in the support (Packet Length Mean, Variance and Std; Average
  Packet Size; Flow IAT Mean and Max; Fwd IAT Std; and so on) are moved independently, with
  no relation tying them to their parents.

In the FINAL suite this leaves Valid ASR at 0% on every victim, even though Raw ASR stays
substantial (see §6).

---

## 4. CAPGD (Constrained Adaptive PGD)

**Idea.** CAPGD (Simonetto et al., "Towards Adaptive Attacks on Constrained Tabular Machine
Learning", the gradient component of CAA) is to C-PGD what Auto-PGD (Croce & Hein, 2020) is to
PGD. It keeps the constrained objective and adds four things:

1. an **adaptive step size**, which starts large and is halved per sample whenever progress
   stalls;
2. a **momentum** term;
3. **repair of equality constraints at every iteration**, by overwriting each relation's
   left-hand feature with the value its right-hand expression implies;
4. **multiple restarts**, with success checked on the repaired, type-fixed flow.

The code uses the frozen upstream TabularBench implementation (commit `bfb75415`) unchanged.
`capgd_cicids2017.py` supplies only the dataset pieces: train min–max scaler, feature types,
mutable mask, relations and the raw-space victim wrapper.

**Objective.** The objective is the same per-sample ascent objective as C-PGD, with penalty
weight 1:

$$
\ell_i(\hat{x}') = \mathcal{L}_{\text{CE}}\big(f(\hat{x}'_i), y_i\big) - \mathrm{viol}\big(x'_i\big).
$$

It is maximised in the train min–max space inside a masked $L_2$ ball. The internal radius is
$\varepsilon (1 - \text{eps\_margin}) = 0.5 \times 0.99 = 0.495$. Success is judged against
the full $\varepsilon = 0.5$.

**Algorithm (as implemented, `CAPGD.attack_single_run` and `perturb`).**

1. **Starting points over 2 restarts.** Restart 0 starts at the clean point
   (`init_start=True`). Restart 1 starts at a random point on the masked $L_2$ sphere of
   radius $\varepsilon$. The start is clipped to $[0,1]$.
2. **Step size.** Every sample starts with $\eta_0 = 2\varepsilon$.
3. For each of the 10 steps:
   1. **Normalised gradient step on the mutable features:**
      $u = x_t + M \odot \eta \, g / \|M \odot g\|_2$, projected onto the $L_2$ ball and
      clipped to $[0,1]$.
   2. **Momentum:** $x_{t+1} = x_t + a\,M\odot(u - x_t) + (1-a)\,M\odot(x_t - x_{t-1})$,
      with $a = 0.75$ after the first step. Project and clip again.
   3. **Equality repair.** Inverse-scale to raw space and recompute every equality relation's
      left-hand feature from its right-hand expression (e.g. set Fwd Packet Length Mean to
      Total Length of Fwd Packet / Total Fwd Packet). Re-scale.
   4. **Loss and gradient.** Recompute $\ell$ and its gradient at the repaired point. Any
      iterate that is already misclassified is remembered as the current adversarial candidate.
   5. **Step-size adaptation (Auto-PGD rule, $\rho = 0.75$).** At checkpoints (first after
      $\max(\lfloor 0.22\,T \rfloor, 1)$ steps, then at shrinking intervals), halve $\eta$
      for every sample where either:
      - the loss increased in fewer than $\rho$ of the steps since the last checkpoint, or
      - the best loss has not improved since the last checkpoint and $\eta$ was not already
        halved there.

      `best_restart=False`, so the iterate is *not* reset to the best point when the step is
      halved.
4. **After each restart, repair and check.**
   1. Map to raw space, then fix integer types, restore immutable features and repair the
      equalities.
   2. Keep a sample's repaired candidate if it is simultaneously:
      - misclassified,
      - within $\varepsilon$ in min–max $L_2$ distance, and
      - satisfies **all** encoded relations with tolerance 0.

      These three together are TabularBench's `mdc` criterion.
   3. The next restart attacks only the samples that are not yet successful.
5. **Final output.** Apply type fix, immutable-feature restore and equality repair once more
   (`fix_equality_constraints_end=True`).
6. **Adapter post-processing** (`finalize_capgd_output`):
   - *PrimSupport:* copy every feature outside the 23-feature support back from the clean flow.
   - *Native:* apply the dataset mask, which recomputes the 7 derived features with the exact
     CICFlowMeter formulas and copies all frozen features.

   `evaluate_capgd_output` then independently records the internal constraint check, the
   distance check and the validator_v2 verdict.

**Locked configuration.** $L_2$, $\varepsilon = 0.5$, 10 steps, 2 restarts, CE loss,
$\rho = 0.75$, adaptive step size on, equality repair every iteration and at the end,
eps_margin 0.01, batch size 64.

**Two mask configurations.**

- **CAPGD-PrimSupport** (inferential baseline in Exp A). The same 23-feature write support as
  C-PGD, and the same relations. All 23 features are directly mutable.
- **CAPGD (native)** (descriptive row only, amendment A3). A 16-feature mask:
  - 9 perturbable features: Flow Duration; Total Length of Fwd Packet; Fwd Packet Length
    Max, Min and Std; Fwd IAT Total, Std, Max and Min.
  - 7 derived features, recomputed exactly: Fwd Packet Length Mean; Fwd Segment Size Avg;
    Fwd IAT Mean; Fwd, Bwd and Flow Packets/s; Flow Bytes/s.
  - All other features are frozen (`attack/masks/cicids2017_distrinet.py`).

  Because the derived features are recomputed from their parents rather than moved
  independently, its outputs satisfy more validator identities than the PrimSupport variant.
  Native CAPGD has the highest Valid ASR of any method on all six victims (0.45–28.42%). This
  shows that the choice of support and parameterisation defines the threat model.

**How CAPGD differs from C-PGD.** The objectives are the same, but CAPGD adds adaptive steps,
momentum, restarts, per-iteration equality repair and a success check on the *repaired*
candidate. Repair turns the encoded equalities from a soft penalty into a property the output
satisfies. That is why CAPGD keeps some valid successes where C-PGD keeps none. The repair only
covers relations that were explicitly encoded, however. Every other coupled feature in the
23-feature support is still moved independently, so most outputs remain inconsistent in ways
validator_v2 detects.

---

## 5. Side-by-side summary

| | PGD | C&W | C-PGD-PrimSupport | CAPGD-PrimSupport |
|---|---|---|---|---|
| Origin | Madry et al. 2018 | Carlini & Wagner 2017 (fixed-$\lambda$ variant) | Simonetto et al. IJCAI 2022 | Simonetto et al. (CAA gradient component), TabularBench |
| Attack space | RobustScaler | RobustScaler | train min–max $[0,1]$ | train min–max $[0,1]$ |
| Threat set | $L_\infty \le 0.5$, all 79 features | unbounded ($L_2$ penalty), all 79 | $L_2 \le 0.5$, 23 features, box | $L_2 \le 0.5$ (0.495 internal), 23 features, box |
| Objective | CE ascent | margin + $\|\delta\|_2^2$ | CE − $\lambda \cdot$ violation | CE − violation |
| Update | sign-gradient, fixed $\alpha$ | Adam | normalised gradient, fixed $\eta$ | normalised gradient, adaptive $\eta$, momentum |
| Constraint handling | none | none | penalty; types at end | penalty + equality repair every step + types/immutables/equalities at end |
| Restarts / steps | 1 × 40 | 1 × ≤60 | 1 × 40 | 2 × 10 |
| Output rule | final iterate | lowest-$L_2$ success, else final | final iterate (typed) | repaired success if found; else the latest misclassified iterate, or the restart's starting point if the flow was never misclassified (all repaired at the end) |
| Sees validator_v2? | no | no | no | no |

---

## 6. Results context (FINAL suite, Exp A, untargeted, Raw → Valid ASR, mean ± SD over attack seeds)

Values are copied from `FINAL_OUTPUTS/final_experiment_summary.md`. Victims are reported
separately and never pooled.

| Dataset | Victim | PGD | C&W | CAPGD-PrimSupport | C-PGD-PrimSupport | CAPGD (native) † |
|---|---|---|---|---|---|---|
| CICIDS2017 | mlp | 100.00% → 0.00% | 99.94% → 0.00% | 94.41% → 2.01% | 50.80% → 0.00% | 94.65% → 9.53% |
| CICIDS2017 | cnn | 96.12% → 0.00% | 95.53% → 0.00% | 96.53% → 5.15% | 60.42% → 0.00% | 96.25% → 19.24% |
| CICIDS2017 | ft_transformer | 97.36% → 0.00% | 77.16% → 0.00% | 52.21% → 0.18% | 21.61% → 0.00% | 50.61% → 5.71% |
| CICIDS2018 | mlp-s42 | 94.34% → 0.00% | 87.63% → 0.00% | 91.57% → 0.14% | 28.25% → 0.00% | 80.42% → 28.42% |
| CICIDS2018 | cnn-s42 | 99.70% → 0.00% | 99.16% → 0.00% | 76.01% → 0.29% | 50.54% → 0.00% | 66.44% → 16.30% |
| CICIDS2018 | ft_transformer-s42 | 91.70% → 0.00% | 53.59% → 0.00% | 9.74% → 0.00% | 1.65% → 0.00% | 5.96% → 0.45% |

† Descriptive row (amendment A3), not part of the inferential family.

**Reading the table.** The more an attack respects how flow features depend on each other,
the smaller the gap between Raw and Valid ASR:

- Unconstrained PGD and C&W get almost every flow misclassified but produce no valid flow.
- Penalty-only C-PGD produces no valid flow either.
- Repair-based CAPGD keeps a few valid successes.
- CAPGD with exact recomputation of derived features (native) keeps the most.

PrimAttack, which changes only padding and delay and recomputes all dependent features through
$\varphi$, has Raw ASR = Valid ASR in every cell. It pays for this with a much lower Raw ASR.

---

## 7. Claim boundaries for the write-up

- All four attacks are **feature-space** attacks on aggregate CICFlowMeter statistics. No
  packet capture is edited or replayed. A valid success means validator_v2 accepts the flow
  as structurally consistent. It does not mean the flow is realisable as packets.
- None of the four baselines optimises against validator_v2. Their Valid ASR measures how
  often their own constraint handling happens to produce validator-consistent flows, not the
  best a validator-aware attacker could achieve.
- The $\varepsilon$ values are not comparable across the attack spaces (§0.2). Compare Valid
  ASR at the locked configurations, not perturbation sizes across methods.
- The C&W implementation is a simplified fixed-constant variant (§2). The CAPGD-PrimSupport
  comparison matches PrimAttack's feature support, not its primitive-feasible set (§0.4).
