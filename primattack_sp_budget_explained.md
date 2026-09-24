# PrimAttack semantic preservation and budget evaluation — complete explanation

## 1. Purpose of this document

This document explains the complete **budget**, **primitive-feasibility**, and
**flow-level semantic-preservation (SP)** parts of PrimAttack. It is intended to provide enough
technical and methodological detail to:

1. understand exactly what the implementation does;
2. reproduce the calibration and attack experiments;
3. interpret every reported metric correctly;
4. write the methodology, results, discussion, and limitations sections of the thesis;
5. avoid claims that are stronger than the available CICIDS2017 evidence.

The implementation operates offline on the corrected DistriNet CICIDS2017 flow-feature data. It
does not modify PCAP files, construct packets, replay attacks, or verify complete application
behavior.

The primary source files are:

| Concern | Source |
|---|---|
| Primitive contract and feature transformation | `src/attack/realizability/{base,cicids2017}.py` |
| Train-only budget calibration | `src/attack/primattack_budget.py` |
| Semantic-preservation proxy | `src/attack/flow_semantics.py` |
| Attack runner and per-sample records | `src/attack/run_cicids2017_primitive_attack.py` |
| Budget/ablation runner | `scripts/budget_sweep_primitive.py` |
| Paired statistics | `scripts/analyze_primattack_experiments.py` |
| Results report builder | `scripts/build_primattack_budget_report.py` |
| Frozen calibration artifact | `artifacts/primattack/budget_calibration.json` |
| Full numerical results | `docs/primattack_budget_results.md` |
| Full experiment outputs | `outputs/primattack_budget_sensitivity_full/` |

---

## 2. The central distinction: four separate evaluation questions

PrimAttack does not reduce evaluation to a single attack-success number. It distinguishes four
questions.

### Level 0 — classifier evasion

Did the final projected adversarial flow receive the target class, Benign?

This is a classifier result only. It does not by itself establish that the feature vector is
valid, that the primitive budget was respected, or that measurable attack semantics remain.

### Level 1 — domain validity

Does the final flow-feature vector pass the existing CICIDS2017 validator?

The implementation uses validator_v2 `hybrid_valid`. This gate checks feature-domain,
extractor, protocol, mined, and plausibility rules. It answers whether the representation is
acceptable under the project's domain model. It does not answer whether the primitive operation
was within budget or whether attack-related behavior was retained.

### Level 2 — primitive feasibility

Did the attack remain inside the declared primitive threat model and hard calibrated budget?

A sample is primitive-feasible only if:

- projected $p$ and $\alpha$ are finite;
- $p$ is non-negative and integral;
- $p\le\lfloor p_{hi}\rfloor$;
- $1\le\alpha\le\alpha_{hi}$;
- actual relative duration change does not exceed the named timing budget;
- represented traffic volume does not decrease;
- the internally generated feature vector passes the independent primitive consistency checks.

### Level 3 — flow-level semantic-preservation proxy

Do the measurable attack-related flow properties available in CICIDS2017 pass the predefined
semantic checks?

This is the SP layer. It is deliberately separate from domain validity. A vector can be
network/domain-valid while failing a class-specific attack-intensity rule. Conversely, an
attack can preserve measurable flow semantics but fail an unrelated validator rule.

### What is not established

CICIDS2017 contains aggregate flow statistics, not the packet sequence, payload, target state, or
application outcome. The SP layer therefore does not establish complete malicious functionality
or packet-trace behavior. Those questions would require packet realization, CICFlowMeter
re-extraction, and isolated replay against an appropriate target/service.

---

## 3. Primitive-domain threat model

The canonical transformation is

$$
 x_{adv}=\phi(x_0,p,\alpha),
$$

where:

- $x_0$ is the pristine 79-feature CICFlowMeter flow;
- $p$ is forward packet-length augmentation;
- $\alpha$ is forward inter-arrival-time dilation;
- $\phi$ is the only normal path from primitive controls to victim-model input features.

No optimizer variable independently edits an arbitrary CICFlowMeter feature.

### 3.1 Padding/size primitive $p$

| Property | Definition |
|---|---|
| Meaning | Uniform forward packet-length augmentation in the declared flow-level model |
| Units | Bytes per forward packet |
| Identity | $p=0$ |
| Direction | Increase only |
| Optimization type | Continuous relaxation |
| Final type | Discrete integer bytes |
| Capability condition | Forward packet count $\ge1$, positive forward total length, positive forward mean length |

Known forward-length equations are:

$$
L_f' = L_f + N_f p,
$$

$$
l_{f,min}'=l_{f,min}+p,
\qquad
l_{f,max}'=l_{f,max}+p,
$$

$$
\bar l_f' = \frac{L_f'}{\max(N_f,1)}.
$$

A uniform shift preserves forward packet-length standard deviation:

$$
s_f'=s_f.
$$

The implementation also recomputes:

- `Fwd Segment Size Avg`;
- combined packet min/max/mean;
- `Average Packet Size`;
- combined packet variance and standard deviation;
- `Flow Bytes/s`.

For combined forward/backward variance, the implementation uses pooled sample variance. Let
$N=N_f+N_b$, $m_f,m_b$ be directional means, and $s_f,s_b$ be directional sample standard
deviations. Then

$$
m_c = \frac{N_fm_f+N_bm_b}{\max(N,1)},
$$

$$
SS_f=\max(N_f-1,0)s_f^2,
\qquad
SS_b=\max(N_b-1,0)s_b^2,
$$

$$
SS_{between}=N_f(m_f-m_c)^2+N_b(m_b-m_c)^2,
$$

$$
s_c^2=\frac{SS_f+SS_b+SS_{between}}{\max(N-1,1)}
\quad\text{for }N\ge2.
$$

The primitive is a declared flow-level model. Aggregate data cannot establish where bytes would
be inserted in a real packet sequence or whether payload/application meaning would remain.

### 3.2 Timing primitive $\alpha$

| Property | Definition |
|---|---|
| Meaning | Uniform dilation of forward inter-arrival gaps |
| Units | Dimensionless ratio |
| Identity | $\alpha=1$ |
| Direction | Delay only; no timing compression |
| Optimization/final type | Continuous; integer dependent timing fields are microsecond-quantized |
| Capability condition | At least two forward packets and positive `Fwd IAT Total` |

Forward-IAT features transform as:

$$
T_f'=\alpha T_f,
$$

$$
IAT_{f,max}'=\alpha IAT_{f,max},
\quad
IAT_{f,min}'=\alpha IAT_{f,min},
\quad
s_{IAT,f}'=\alpha s_{IAT,f},
$$

$$
\overline{IAT}_f'=
\frac{T_f'}{\max(N_f-1,1)}.
$$

The flow-duration projection is

$$
D'=\max\left(
D+(T_f'-T_f),
T_f',
T_b,
1\ \mu s
\right).
$$

Flow-IAT mean is then

$$
\overline{IAT}_{flow}'=
\frac{D'}{\max(N_f+N_b-1,1)}.
$$

`Flow IAT Max` is conservatively increased by the non-negative duration increase. The exact
merged forward/backward packet order is not present in the aggregate row, so this is a declared
conservative flow-level projection rather than a unique packet reconstruction.

### 3.3 Rate recomputation

Let $D_s'=D'/10^6$ seconds. Timing changes recompute:

$$
R_{fwd}'=\frac{N_f}{D_s'},
\qquad
R_{bwd}'=\frac{N_b}{D_s'},
$$

$$
R_{packets}'=\frac{N_f+N_b}{D_s'}.
$$

Padding and/or timing recompute flow byte rate:

$$
R_{bytes}'=\frac{L_f'+L_b}{D_s'}.
$$

### 3.4 Unknown sequence-dependent properties

The aggregate representation cannot reconstruct:

- merged `Flow IAT Std` and `Flow IAT Min`;
- active/idle burst summaries;
- bulk statistics;
- subflow byte decomposition;
- packet payload meaning;
- the exact packet carrying each added byte.

These fields are explicitly marked `LEVEL_C`/UNKNOWN and held constant. Their constancy must not
be described as proof that a packet extractor would reproduce the same values.

---

## 4. Differentiable white-box optimization

For the direct primitive attack, two unconstrained optimizer leaves $u$ and $v$ are mapped into
the per-flow hard box using the logistic sigmoid $\sigma$:

$$
p=p_{hi}\,\sigma(u)\,m_p,
$$

$$
\alpha=1+(\alpha_{hi}-1)\sigma(v)m_\alpha,
$$

where $m_p,m_\alpha\in\{0,1\}$ are capability masks.

The optimization path is:

```text
u, v
  -> bounded p, alpha
  -> differentiable phi(x0,p,alpha)
  -> training-fitted RobustScaler
  -> trained victim
  -> targeted Benign loss
```

The raw transformed flow is scaled exactly as the victim expects:

$$
x_{scaled}=\frac{x_{raw}-center_{train}}{scale_{train}}.
$$

The direct optimizer minimizes targeted cross-entropy plus a small normalized control-position
penalty:

$$
\mathcal L_i=
CE(f(x_{adv,i}),y_{target}=Benign)
+\lambda_c\left(\sigma(u_i)m_{p,i}+\sigma(v_i)m_{\alpha,i}\right),
$$

with $\lambda_c=0.01$ in the reported experiment. The penalty prefers smaller controls, but it
is not the budget. The feasible set is enforced by the change of variables and final hard
projection.

### Final projection

For requested controls $p_{req},\alpha_{req}$:

$$
p_{clip}=\min(\max(p_{req},0),p_{hi}),
$$

$$
p_{proj}=\min(\operatorname{round}(p_{clip}),\lfloor p_{hi}\rfloor),
$$

$$
\alpha_{proj}=\min(\max(\alpha_{req},1),\alpha_{hi}).
$$

Capability masks are applied again. Final dependent features are regenerated from projected
controls, and the victim is evaluated again. Reported success therefore refers to the final
realized flow-feature vector, not the continuous optimization surrogate.

---

## 5. Why budgets are needed

Without explicit budgets, a primitive attack could obtain success by imposing arbitrarily large
delay or byte overhead. That would make classifier evasion difficult to interpret and could
destroy the behavior represented by the original attack flow.

The budget design follows four principles:

1. numerical values come from CICIDS2017 training data and the primitive model;
2. validation/test data do not select thresholds;
3. attack success does not select thresholds;
4. budgets are hard constraints, not merely loss penalties.

Other papers can motivate bounded-overhead evaluation, but no external paper supplies the
numerical values used here.

---

## 6. Train-only budget calibration

### 6.1 Calibration inputs

Only these files are loaded:

- `X_train_pristine.npy`;
- `y_train_cat.npy`;
- the training preprocessing manifest/class mapping.

The artifact explicitly records that these inputs are prohibited during selection:

- validation features;
- test features;
- victim predictions;
- adversarial success.

The calibration artifact stores SHA-256 hashes of the training features, labels, and
preprocessing manifest.

Training row counts used for the retained classes were:

| Class | Training rows |
|---|---:|
| DoS | 120,093 |
| DDoS | 66,568 |
| Recon / PortScan | 111,311 |
| BruteForce | 4,862 |

### 6.2 Robust descriptive statistics

For every calibrated quantity, the artifact stores:

- sample count $n$;
- median;
- first and third quartiles;
- interquartile range;
- median absolute deviation;
- P01, P05, P10, P25, P50, P75, P90, P95, and P99.

The formulas are:

$$
IQR=Q_{0.75}-Q_{0.25},
$$

$$
MAD=\operatorname{median}_i|x_i-\operatorname{median}(x)|.
$$

Robust summaries were chosen because duration, packet rates, byte rates, counts, and lengths are
strongly skewed. A mean/std-only calibration would be dominated by extreme flows.

### 6.3 Padding calibration population

For class $c$, the padding reference population contains positive `Fwd Packet Length Mean`
values from training rows satisfying:

$$
N_f\ge1,
\qquad
L_f>0,
\qquad
\bar l_f>0.
$$

The named padding budget is the rounded empirical quantile of that population:

$$
B_{p,c,q}=\operatorname{round}\left(Q_q(\bar l_f\mid c,\text{eligible})\right).
$$

Because $p$ is discrete, the selected quantile is rounded to a legal whole-byte value. If two
empirical quantiles round to the same byte budget, they remain equal; the implementation does
not invent artificial separation.

### 6.4 Timing calibration population

For class $c$, let $D_i$ be training flow duration and $\widetilde D_c$ the class median. The
relative duration-variation population is

$$
r_i=\frac{|D_i-\widetilde D_c|}{\max(\widetilde D_c,1\ \mu s)}.
$$

The class timing budget is

$$
B_{D,c,q}=Q_q(r_i\mid c).
$$

This makes timing budgets interpretable as maximum relative duration increase rather than an
arbitrary universal alpha value.

### 6.5 Named levels

The empirical levels are:

| Level | Quantile | Meaning |
|---|---:|---|
| Restricted | P25 | Lower empirical perturbation envelope |
| Intermediate | P50 | Median empirical perturbation envelope |
| Maximum-evaluated | P75 | Largest budget evaluated in the thesis experiment |

“Maximum-evaluated” is not a universal physical maximum. It is the largest evaluated point
inside the declared training-derived flow-level envelope.

### 6.6 Exact frozen budgets

Each entry is:

```text
padding bytes per forward packet / maximum relative duration increase
```

| Class | Restricted | Intermediate | Maximum-evaluated |
|---|---:|---:|---:|
| DoS | 41 / 0.0427256098 | 47 / 0.5671219512 | 54 / 1.2682256098 |
| DDoS | 2 / 0.2219977296 | 2 / 0.4352960736 | 3 / 0.6874252352 |
| Recon / PortScan | 2 / 0.0851063830 | 2 / 0.2127659574 | 10 / 0.5319148936 |
| BruteForce | 11 / 0.0617419493 | 12 / 0.1198948581 | 91 / 0.2336675743 |

The large BruteForce change from 12 to 91 bytes reflects its empirical P50-to-P75 distribution;
it was not selected because it increased attack success.

### 6.7 Additional stored semantic summaries

The calibration artifact also records class P05 `Flow Bytes/s` and class P99
`Flow Duration` for transparent distribution characterization:

| Class | P05 `Flow Packets/s` | P05 `Flow Bytes/s` | P99 duration (µs) | Packet-rate rule required? |
|---|---:|---:|---:|---|
| DoS | 0.6247806907 | 339.7497559 | 105,732,982.08 | Yes |
| DDoS | 1.0654268742 | 966.0045868 | 22,079,893.22 | Yes |
| Recon / PortScan | 22,471.9101563 | 0 | 132 | No |
| BruteForce | 2.8482956886 | 31.2864614 | 14,195,177.41 | No |

The current class-specific semantic validator consumes the packet-rate P05 only for DoS/DDoS.
The byte-rate and class-duration values remain descriptive calibration evidence; they are not
silently applied as additional required semantic checks.

---

## 7. From a named budget to a per-flow feasible bound

A class-level budget is only the first bound. Each source flow gets a stricter per-flow bound
that intersects:

1. named class budget;
2. global training P99 plausibility headroom;
3. primitive capability;
4. DoS/DDoS semantic rate headroom for timing.

Conceptually:

$$
B_{flow}=\min(B_{named},B_{train\text{-}P99},B_{semantic},B_{capability}).
$$

### 7.1 Padding per-flow cap

Let the complete-training P99 upper values be denoted
$U_{max},U_{min},U_{mean},U_{total}$. The numerical padding cap is

$$
p_{numeric}=\max\left(0,\min\left[
B_p,
U_{max}-l_{f,max},
U_{min}-l_{f,min},
U_{mean}-\bar l_f,
\frac{U_{total}-L_f}{\max(N_f,1)}
\right]\right).
$$

The global complete-training P99 is used rather than a class-local upper bound so that a
class-local structural zero is not incorrectly treated as a physical maximum.

If padding capability is unavailable, the semantic cap forces

$$
p_{hi}=0.
$$

### 7.2 Relative-duration timing cap

Let $B_D$ be the named relative-duration budget and $T_f$ the original forward-IAT total. The
relative-duration alpha cap is

$$
\alpha_{relative}=
1+B_D\frac{\max(D,1\ \mu s)}{\max(T_f,\epsilon)}.
$$

The implementation also requires every affected timing quantity to remain under its global
training P99. For a timing feature $z>0$ with P99 upper $U_z$:

$$
\alpha_z=\frac{U_z}{z}.
$$

Duration gives another cap:

$$
\alpha_{duration}=
1+\frac{U_D-D}{\max(T_f,\epsilon)}.
$$

### 7.3 DoS/DDoS semantic rate cap

For DoS/DDoS, timing dilation must not reduce `Flow Packets/s` below the class training P05.
The frozen thresholds are:

| Class | Minimum adversarial `Flow Packets/s` |
|---|---:|
| DoS | 0.6247806907 |
| DDoS | 1.0654268742 |

If $R_{min,c}$ is the class threshold, the largest permitted duration under this rule is

$$
D_{rate\ cap}=\frac{(N_f+N_b)10^6}{R_{min,c}}.
$$

This implies

$$
\alpha_{rate}=
1+\frac{D_{rate\ cap}-D}{\max(T_f,\epsilon)}.
$$

The final numerical timing cap is

$$
\alpha_{numeric}=\max\left(1,\min\left[
\alpha_{relative},
\alpha_{duration},
\alpha_{FwdIATTotal},
\alpha_{FwdIATMax},
\alpha_{FwdIATStd},
\alpha_{FwdIATMean},
\alpha_{rate}\ \text{when required}
\right]\right).
$$

If timing capability is unavailable, the final cap is $\alpha_{hi}=1$.

---

## 8. Primitive cost quantities

### 8.1 Timing cost

$$
\Delta D=D_{adv}-D_0,
$$

$$
c_D=\frac{D_{adv}-D_0}{\max(D_0,\epsilon)}.
$$

The duration ratio is

$$
\frac{D_{adv}}{\max(D_0,\epsilon)}=1+c_D.
$$

### 8.2 Byte/size cost

The represented byte quantity is

$$
B=L_f+L_b.
$$

Then

$$
\Delta B=B_{adv}-B_0,
$$

$$
c_B=\frac{B_{adv}-B_0}{\max(B_0,\epsilon)}.
$$

### 8.3 Rate retention

$$
\rho_R=\frac{R_{adv}}{\max(R_0,\epsilon)},
$$

where $R$ is `Flow Packets/s`.

Interpretation:

- $\rho_R=1$: packet rate is unchanged;
- $\rho_R<1$: timing dilation reduced packet intensity;
- $\rho_R>1$: possible only if represented rate increased; timing dilation itself does not
  produce this direction.

### 8.4 Additional normalized quantities

Padding relative to the original forward mean:

$$
100\frac{p}{\max(\bar l_f,\epsilon)}\%.
$$

Normalized padding position:

$$
\widetilde p=\frac{p}{p_{hi}}
\quad\text{when }p_{hi}>0.
$$

Normalized timing position:

$$
\widetilde\alpha=
\frac{\alpha-1}{\alpha_{hi}-1}
\quad\text{when }\alpha_{hi}>1.
$$

These are per-flow normalized positions, not packet-level reconstruction measurements.

### 8.5 Normalized feature-space cost

The runner also reports a normalized feature-space cost for plotting and optimizer diagnostics.
For $F=79$ features and the training-fitted RobustScaler scale $s_j$:

$$
c_{norm}=\frac{1}{F}\sum_{j=1}^{F}
\frac{|x_{adv,j}-x_{0,j}|}{s_j}.
$$

Padding, timing, and rate group contributions use the same expression restricted to their
declared feature groups, while retaining division by the full feature count. This value is not
the primitive budget. It summarizes the resulting model-input movement after the hard primitive
constraints have already been enforced.

---

## 9. The semantic-preservation proxy

### 9.1 Why validity and semantic preservation are different

A domain validator asks whether values and relationships look valid. It does not know whether a
DoS flow still has attack-like intensity or whether a PortScan row still corresponds to a
complete scan process.

The semantic validator asks a narrower, attack-related question:

> Do the original and adversarial flow representations preserve the measurable attack-related
> properties that CICIDS2017 actually exposes?

This is why semantic validation remains a separate component.

### 9.2 Per-check status

Every semantic check returns:

- `PASS`;
- `FAIL`;
- `NOT_TESTABLE`;
- a machine-readable reason code;
- an explanation;
- whether the check is required.

Unavailable evidence is never silently treated as success.

### 9.3 Generic required checks

| Check | What it protects | Failure reason |
|---|---|---|
| Attack label unchanged | Source attack-class metadata | `ATTACK_LABEL_CHANGED` |
| Protocol unchanged | IP protocol semantics | `PROTOCOL_CHANGED` |
| Source/destination ports unchanged | Service/endpoint semantics | `SERVICE_ENDPOINT_CHANGED` |
| Source/destination IP metadata unchanged | Endpoints and represented direction | `FLOW_ENDPOINT_OR_DIRECTION_CHANGED` |
| Packet counts unchanged | Count/injection is outside PrimAttack | `PACKET_COUNT_CHANGED` |
| TCP control flags unchanged | Connection/control semantics | `CONTROL_FLAG_CHANGED` |
| All generated features finite | Numerical safety | `NONFINITE_GENERATED_FEATURE` |
| No represented byte decrease | Padding cannot shrink volume | `TRAFFIC_VOLUME_DECREASED` |
| Only declared dependencies changed | Prevents hidden feature-space attack | `UNDECLARED_FEATURE_CHANGED` |
| Hard primitive budget obeyed | Primitive feasibility | `PRIMITIVE_BUDGET_VIOLATION` |

Packet-count fields include forward/backward counts, active-data count, and subflow packet
counts. Control checks include all feature names containing `Flag`.

### 9.4 DoS and DDoS rule

The class-specific required check is

$$
R_{adv}\ge R_{class,P05}.
$$

For DoS, $R_{class,P05}=0.6247806907$ packets/s. For DDoS,
$R_{class,P05}=1.0654268742$ packets/s.

The check asks whether the transformed flow remains inside the lower tail of observed
class-specific attack-like intensity. It does not assert that service denial would occur.

### 9.5 Recon / PortScan rule

The dataset's retained Recon category is PortScan. The following critical properties are not
available from one aggregate flow row:

- complete scanned-port set;
- scan sequence;
- number of distinct connection attempts.

Each returns `NOT_TESTABLE_FROM_FLOW_DATA`. Therefore a Recon sample becomes
`NOT_FULLY_TESTABLE` unless a testable required condition fails first.

### 9.6 BruteForce rule

The following critical properties are unavailable:

- authentication-attempt count;
- credential or payload semantics;
- server authentication outcome.

They also return `NOT_TESTABLE_FROM_FLOW_DATA`. Therefore BruteForce samples are
`NOT_FULLY_TESTABLE` unless another required check fails.

### 9.7 Sample-level status aggregation

Let $F_i$ be the number of failed required checks and $U_i$ the number of required checks that
are not testable. Then

$$
S_i=
\begin{cases}
FAIL, & F_i>0,\\
NOT\_FULLY\_TESTABLE, & F_i=0\land U_i>0,\\
PASS, & F_i=0\land U_i=0.
\end{cases}
$$

Failure takes precedence over unavailability. This ensures that a known violation is not hidden
behind an unavailable class property.

### 9.8 Semantic testability rate

$$
\text{Testability Rate}=
\frac{\#\{i:S_i\ne NOT\_FULLY\_TESTABLE\}}{N_{eligible}}.
$$

PASS, FAIL, and NOT_FULLY_TESTABLE counts are always reported separately.

---

## 10. Attack-success metrics

Let:

- $C_i$: source sample is clean-correct and therefore eligible;
- $E_i$: final adversarial prediction is Benign;
- $V_i$: domain validator passes;
- $P_i$: primitive feasibility passes;
- $S_i$: semantic status is `PASS`.

The denominator is

$$
N_{eligible}=\sum_i C_i.
$$

### Raw targeted ASR

$$
ASR_{raw}=\frac{\sum_i C_iE_i}{N_{eligible}}.
$$

### Valid targeted ASR

$$
ASR_{valid}=\frac{\sum_i C_iE_iV_i}{N_{eligible}}.
$$

### Primitive-feasible targeted ASR

$$
ASR_{feasible}=\frac{\sum_i C_iE_iV_iP_i}{N_{eligible}}.
$$

### Semantic-Preserving ASR

$$
SP\text{-}ASR=
\frac{\sum_i C_iE_iV_iP_i\mathbf 1[S_i=PASS]}{N_{eligible}}.
$$

SP-ASR means **flow-level semantic-preservation proxy ASR**. It is the strongest result supported
by this experiment, but it remains a flow-level proxy.

---

## 11. Experimental design

### 11.1 Full active-roster protocol

| Item | Value |
|---|---|
| Classes | DoS, DDoS, Recon, BruteForce |
| Victims | MLP, CNN |
| Source rows | 512 fixed rows per class |
| Optimizer steps | 40 |
| Learning rate | 0.1 |
| Cost weight | 0.01 |
| Seed | 42 |
| Budgets | Restricted, intermediate, maximum-evaluated |
| Modes | Timing-only, padding-only, joint |
| Target | Benign |

There are nine mode/budget conditions. The active roster creates eight class/victim cells per
condition. Source rows are selected before victim and condition loops, so exact sample IDs remain
paired.

### 11.2 Primitive ablations

| Mode | Hard setting | Question |
|---|---|---|
| Timing-only | $p=0$ | Can delay alone cause targeted evasion? |
| Padding-only | $\alpha=1$ | Can size augmentation alone cause targeted evasion? |
| Joint | Both enabled | Does timing add benefit beyond padding? |

Artifact validation confirmed the disabled primitive remained at identity for every row.

### 11.3 Pairing and denominators

The pairing key is:

```text
sample_id × attack_class × victim_model × seed
```

There are $4\times2\times512=4,096$ evaluated rows per condition. Samples the corresponding
victim did not originally classify correctly are excluded from ASR, leaving 4,064 eligible rows.
Across nine conditions, the active artifacts contain 36,864 saved rows.

---

## 12. Active-roster results

### 12.1 Pooled budget results

| Mode | Budget | Raw n | Raw ASR | Valid ASR | Feasible ASR | SP n | SP-ASR | Median relative duration | Median relative bytes | Median rate retention |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Timing-only | Restricted | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.000591466 | 0 | 0.999409 |
| Timing-only | Intermediate | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00339406 | 0 | 0.996617 |
| Timing-only | Maximum-evaluated | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00623234 | 0 | 0.993806 |
| Padding-only | Restricted | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0 | 0.000838082 | 1.000000 |
| Padding-only | Intermediate | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0 | 0.00120534 | 1.000000 |
| Padding-only | Maximum-evaluated | 10 | 0.25% | 0.25% | 0.25% | 0 | 0.00% | 0 | 0.00206629 | 1.000000 |
| Joint | Restricted | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.000616571 | 0.000838082 | 0.999384 |
| Joint | Intermediate | 0 | 0.00% | 0.00% | 0.00% | 0 | 0.00% | 0.00363767 | 0.00120534 | 0.996376 |
| Joint | Maximum-evaluated | 10 | 0.25% | 0.25% | 0.25% | 0 | 0.00% | 0.00700700 | 0.00206629 | 0.993042 |

### 12.2 Semantic-status coverage

Every condition produced:

| Status | Count | Rate |
|---|---:|---:|
| PASS | 1,958 | 48.18% |
| FAIL | 88 | 2.17% |
| NOT_FULLY_TESTABLE | 2,018 | 49.66% |
| Testable (`PASS` or `FAIL`) | 2,046 | 50.34% |

The 88 failures were DoS/DDoS rows below the frozen class packet-rate threshold. Recon and
BruteForce remain not fully testable because aggregate rows do not expose their critical
sequence/application properties.

### 12.3 Maximum-evaluated joint results by class

| Class | Eligible | Raw n | Raw ASR | SP n | SP-ASR | Semantic PASS | Semantic FAIL | Not fully testable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DoS | 1,024 | 0 | 0.00% | 0 | 0.00% | 96.68% | 3.32% | 0.00% |
| DDoS | 1,022 | 0 | 0.00% | 0 | 0.00% | 94.72% | 5.28% | 0.00% |
| Recon / PortScan | 1,020 | 0 | 0.00% | 0 | 0.00% | 0.00% | 0.00% | 100.00% |
| BruteForce | 998 | 10 | 1.00% | 0 | 0.00% | 0.00% | 0.00% | 100.00% |

### 12.4 Maximum-evaluated joint results by victim

| Victim | Eligible | Raw n | Raw ASR | SP n | SP-ASR |
|---|---:|---:|---:|---:|---:|
| MLP | 2,034 | 6 | 0.29% | 0 | 0.00% |
| CNN | 2,030 | 4 | 0.20% | 0 | 0.00% |

### 12.5 Gate interpretation

All ten maximum-joint successes were domain-valid and primitive-feasible:

$$
\#raw=\#valid=\#primitive\text{-}feasible=10.
$$

All ten occurred in BruteForce, whose critical application behavior is unavailable from
aggregate flow data. Therefore:

$$
SP\text{-}ASR=0.
$$

This means the available evidence cannot support the strongest semantic claim for those
successes. It does not prove application behavior was lost.

### 12.6 Figure interpretation

- Timing-only changed duration/rate but produced no targeted evasion.
- Padding-only and joint produced the same ten maximum-budget successes.
- Equality of raw, valid, and feasible ASR shows validity/budget gates removed no successes.
- Zero SP-ASR is caused by semantic testability, not domain or primitive failure.

The seven figures are under `outputs/primattack_budget_sensitivity_full/`.

---
## 13. Statistical testing

### 13.1 Why paired tests are required

The same source flows are evaluated under every budget and primitive mode. Outcomes are therefore
not independent. Paired/repeated-measures tests use this structure and avoid treating repeated
measurements as unrelated samples.

### 13.2 Cochran's Q for more than two binary conditions

For an $N\times K$ binary matrix $X$, let column totals be $C_j$ and row totals be $R_i$. The
implemented statistic is

$$
Q=(K-1)
\frac{K\sum_j C_j^2-(\sum_jC_j)^2}
{K\sum_iR_i-\sum_iR_i^2}.
$$

It is compared with $\chi^2_{K-1}$. If the denominator is zero because every condition is
identical, the implementation returns statistic 0 and $p=1$.

### 13.3 McNemar pairwise follow-up

For conditions A and B:

- $b$: A succeeds and B fails;
- $c$: A fails and B succeeds.

Only discordant pairs matter. Small discordant counts use the exact two-sided binomial test.
Larger counts use continuity-corrected McNemar:

$$
\chi^2=\frac{(|b-c|-1)^2}{b+c}.
$$

### 13.4 Continuous repeated measures

Relative duration, relative bytes, and rate retention use the Friedman omnibus test for more
than two paired conditions. Ignoring tie correction for notation, with $N$ rows, $K$ conditions,
and condition rank totals $R_j$, its statistic is

$$
Q_F=\frac{12}{NK(K+1)}\sum_{j=1}^{K}R_j^2-3N(K+1).
$$

SciPy applies the required tie correction. Significant omnibus results are followed by
Wilcoxon signed-rank comparisons. For paired differences $d_i$, Wilcoxon ranks $|d_i|$ and
compares the positive and negative rank sums; the implementation uses Pratt handling so zero
differences remain represented in rank construction. Pairwise p-values then receive Holm
correction.

### 13.5 Holm correction

Within each pairwise family, raw p-values are sorted from smallest to largest. For ordered
$p_{(1)},\ldots,p_{(m)}$, Holm compares $p_{(i)}$ with

$$
\frac{\alpha}{m-i+1}
$$

and reports monotone adjusted p-values. This controls family-wise error without assuming
independent comparisons.

### 13.6 Key statistical results

At maximum-evaluated budget:

- primitive mode affected targeted success: Cochran's Q $p=4.53999\times10^{-5}$;
- joint vs timing-only: Holm $p=0.00585938$;
- padding-only vs timing-only: Holm $p=0.00585938$;
- joint vs padding-only: Holm $p=1$.

For SP success, every condition had zero successes, so the mode omnibus and pairwise tests return
$p=1$.

For joint PrimAttack, maximum-evaluated exceeded intermediate budget for targeted success after
correction: Holm $p=0.00585938$.

Interpretation: padding and joint attacks significantly exceed timing-only at the maximum
budget, but joint and padding-only find the same ten successful rows. Timing adds no observed
evasion benefit for the active roster.

---

## 14. What the active-roster results mean

### 14.1 Padding is the only effective primitive in this run

Timing-only produced no targeted successes. Padding-only and joint each produced the same ten
maximum-budget successes. The paired comparison between joint and padding-only had Holm $p=1$.

Defensible conclusion:

> Under the calibrated CICIDS2017 primitive envelope and active MLP/CNN victims, observed
> targeted evasion came from the padding/size primitive; timing dilation added no successful
> samples.

### 14.2 Success is limited to BruteForce

All ten successful rows were BruteForce: six against MLP and four against CNN. DoS, DDoS, and
Recon produced no targeted successes under any active victim at maximum joint budget.

This is a class-conditional result, not a universal attack-success claim.

### 14.3 Why SP-ASR is zero

All ten successes were domain-valid and primitive-feasible. The semantic gate did not reject
them for a known violation; BruteForce critical application properties cannot be established
from aggregate flow rows. They are therefore `NOT_FULLY_TESTABLE`, yielding SP-ASR 0%.

Correct interpretation:

> Bounded primitive transformations evaded the active classifiers for ten BruteForce flows, but
> available CICIDS2017 evidence cannot establish preservation of authentication behavior.

### 14.4 Validity is not the limiting gate

Raw, valid, and primitive-feasible ASR are all 0.25% at maximum padding/joint budget. Thus,
validator or budget rejection did not reduce success. The difference between feasible ASR and
SP-ASR is entirely the conservative semantic testability boundary.

### 14.5 Cost behavior matches the ablation design

- timing-only: zero byte cost and reduced rate retention;
- padding-only: zero duration cost and rate retention exactly one;
- joint: both cost types present;
- median changed features: 11 timing, 10 padding, 20 joint.

These observations confirm the disabled primitive did not leak into each ablation.

---
## 15. How to write the thesis report

### 15.1 Recommended methodology structure

1. Define $p$, $\alpha$, and $\phi(x_0,p,\alpha)$.
2. State that no CICFlowMeter feature is optimized independently.
3. Give the dependency equations and identify UNKNOWN sequence-dependent fields.
4. Explain train-only robust calibration and named P25/P50/P75 levels.
5. Explain per-flow intersection with P99 headroom, capability, and DoS/DDoS rate constraints.
6. Define costs and final hard projection.
7. Separate domain validity, primitive feasibility, and semantic preservation.
8. Define tri-state semantic checks and class-specific unavailable properties.
9. Define raw, valid, feasible, and SP-ASR with one denominator.
10. Describe the paired 3-by-3 budget/primitive experiment.

### 15.2 Recommended results order

1. Present the nine-row pooled budget table.
2. State that timing-only produced zero success.
3. Compare padding-only and joint at maximum budget.
4. Report the paired significance results.
5. Break maximum-joint results down by source class.
6. Break them down by victim.
7. Explain the BruteForce raw-versus-SP gap.
8. Report semantic PASS/FAIL/NOT_FULLY_TESTABLE coverage.
9. Report costs and rate retention.

### 15.3 Example concise results paragraph

> Across 4,064 eligible clean-correct flows per condition, timing-only PrimAttack produced no
> targeted successes at any calibrated budget. At the maximum-evaluated budget, padding-only
> and joint PrimAttack each produced ten targeted successes (0.25% raw, valid, and
> primitive-feasible ASR). Both significantly exceeded timing-only, while joint and padding-only
> were identical. All ten successes were BruteForce rows whose critical application semantics
> cannot be established from aggregate flow data, so SP-ASR was 0%.

### 15.4 Example methodology paragraph

> Primitive budgets were calibrated exclusively on the pristine training split. For each attack
> class, padding levels were the rounded P25, P50, and P75 of positive forward mean packet length,
> while timing levels were the corresponding quantiles of absolute relative duration deviation
> from the class median. At attack time, each named level was intersected with global-training
> P99 feature headroom, per-flow primitive capability, and, for DoS/DDoS, a class-specific P05
> packet-rate retention constraint. The continuous change-of-variables map kept controls inside
> the feasible box at every optimizer step. Final outputs were then discretely projected, and
> all dependent features were regenerated before classifier and validator evaluation.

### 15.5 Required limitation paragraph

> This study evaluates semantic preservation at the flow-feature level. CICIDS2017 contains
> aggregated flow statistics and cannot establish complete real-world malicious behavior after
> transformation. The semantic analysis therefore uses an offline proxy based on immutable flow
> attributes, bounded primitive modifications, and class-specific training-derived behavioral
> thresholds. Complete verification would require packet-level realization, feature
> re-extraction, and isolated replay against the corresponding application or service, which are
> outside the scope of this thesis.

---

## 16. Claims that are supported and unsupported

### Supported

- Budgets were derived from training data only.
- Every reported final adversarial sample was projected into a hard primitive bound.
- No hidden arbitrary feature-space edits occur in the canonical path.
- All raw successes in the full budget experiment were domain-valid and primitive-feasible.
- Padding drove the ten targeted evasions under the evaluated active roster.
- Joint and padding-only produced identical successful rows at maximum budget.
- No maximum-joint sample passed the complete semantic proxy because all successes belonged to
  a class with critical unavailable application semantics.
- Recon and BruteForce semantic coverage is incomplete and explicitly reported.

### Unsupported

- The P75 budget is a universal physical maximum.
- Aggregate-flow consistency proves packet realizability.
- BruteForce raw successes preserve authentication behavior.
- Recon rows preserve the complete scan sequence or port set.
- DoS rate retention proves that a target would still be denied service.
- Results automatically transfer to another dataset, victim architecture, or deployment.

---

## 17. Common reporting mistakes

1. **Calling raw ASR the final result.** Always report all four gates.
2. **Treating NOT_FULLY_TESTABLE as PASS.** Keep it separate in counts and rates.
3. **Dropping unavailable classes from the denominator.** That would inflate SP results.
4. **Calling maximum-evaluated a physical maximum.** It is a P75 evaluated envelope.
5. **Claiming exact packet modifications.** The representation is flow-level.
6. **Saying timing had no effect.** Timing changed duration/rates but produced no classifier
   successes under this protocol.
7. **Saying joint is better than padding-only.** They produced the same ten successful rows.
8. **Ignoring class concentration.** Every successful row was BruteForce.
9. **Using independent-sample tests.** Conditions are paired by source ID.
10. **Tuning thresholds after seeing ASR.** Calibration is frozen before attack evaluation.

---

## 18. Per-sample audit record

Every NPZ artifact contains enough information to audit one adversarial row:

- source sample ID;
- attack class, victim, optimizer, budget, primitive mode, seed;
- original/adversarial predictions and logits;
- targeted-success flag;
- domain-valid flag and reason;
- primitive-feasible flag and reasons;
- semantic status, failures, and unavailable properties;
- original/adversarial duration and relative change;
- original/adversarial represented bytes and relative change;
- original/adversarial packet rate and retention;
- requested and projected $p,\alpha$;
- per-flow $p_{hi},\alpha_{hi}$;
- normalized primitive magnitudes;
- changed feature names and count;
- capability flags/reasons;
- code, preprocessing, scaler, checkpoint, calibration, and row provenance.

This is what permits the thesis to explain exactly which primitive changed, by how much, which
features changed, and why a successful sample did or did not enter SP-ASR.

---

## 19. Reproduction commands

### Regenerate calibration

```text
PYTHONPATH=".;src" python -m attack.primattack_budget \
  --output artifacts/primattack/budget_calibration.json
```

### Run full budget and primitive-ablation matrix

```text
PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn \
  --test-limit 512 \
  --steps 40 \
  --seeds 42 \
  --output-dir outputs/primattack_budget_sensitivity_full
```

### Run paired statistics

```text
PYTHONPATH=".;src" python scripts/analyze_primattack_experiments.py \
  --input-dir outputs/primattack_budget_sensitivity_full
```

### Rebuild the numerical results report

```text
PYTHONPATH=".;src" python scripts/build_primattack_budget_report.py \
  --input-dir outputs/primattack_budget_sensitivity_full \
  --output docs/primattack_budget_results.md
```

### Run tests

```text
PYTHONPATH=".;src" python -m pytest -q -p no:faulthandler
```

Observed final result: 217 passed, 1 skipped, with one unrelated existing PyTorch convolution
warning.

---

## 20. Output map

| Output | Meaning |
|---|---|
| `artifacts/primattack/budget_calibration.json` | Frozen train-only statistics, budgets, thresholds, and hashes |
| `outputs/primattack_budget_sensitivity_full/budget_sensitivity.csv` | Nine pooled condition rows |
| `outputs/primattack_budget_sensitivity_full/budget_sensitivity.json` | Machine-readable pooled results |
| `outputs/primattack_budget_sensitivity_full/paired_statistics.json` | Omnibus and Holm-corrected paired tests |
| `outputs/primattack_budget_sensitivity_full/source_id_consistency.json` | Pairing confirmation |
| `outputs/primattack_budget_sensitivity_full/*/*/attack_artifacts/*.npz` | Per-sample auditable records |
| `outputs/primattack_budget_sensitivity_full/01_*.png` to `07_*.png` | Requested sensitivity/ablation plots |
| `docs/primattack_budget_results.md` | Full numerical tables and statistical results |
| `primattack_sp_budget_explained.md` | This conceptual and report-writing guide |

---

## 21. Final interpretation in one paragraph

PrimAttack is a differentiable, hard-budgeted flow-primitive attack whose final outputs are
regenerated from padding and timing controls rather than independently edited CICFlowMeter
features. Its budgets are fixed from robust CICIDS2017 training statistics and intersected with
per-flow plausibility, capability, and DoS/DDoS rate constraints. Under the completed active
MLP/CNN protocol, timing dilation alone caused no targeted evasion, while maximum-budget
padding-only and joint attacks each produced ten raw, domain-valid, primitive-feasible
BruteForce successes (0.25% ASR). Their critical application behavior is unavailable from
aggregate flow data, so SP-ASR was 0%. This is the correct distinction to preserve in the
thesis: classifier evasion, domain validity, primitive feasibility, and measurable flow-level
semantic preservation are related but not interchangeable claims.
