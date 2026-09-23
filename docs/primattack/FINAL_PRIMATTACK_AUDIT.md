# Final PrimAttack audit

## Final status

PrimAttack is now a calibrated differentiable primitive-domain white-box attack with four
separate evaluation levels: classifier evasion, domain validity, primitive feasibility, and a
flow-level semantic-preservation proxy. It does not implement or claim packet-level
realization/replay or complete malicious-function preservation.

## Existing strengths retained

- Only two attacker controls reach the canonical transform; arbitrary CICFlowMeter features are
  not attack variables.
- The victim is evaluated through the original train-fitted RobustScaler.
- Known derived feature dependencies are recomputed deterministically.
- Final success is recomputed after discrete projection.
- Capability gates fail closed when the source row lacks forward payload/timing evidence.
- Validator_v2, internal primitive consistency, and VAE/IDR realism remain separate signals.
- Source IDs, code/checkpoint hashes, and preprocessing provenance remain auditable.

## Bugs/gaps found and corrected

1. `PrimitiveSpec` omitted dtype, direction, dependencies, projection, and semantic risk.
2. Budgets used CLI constants and train maxima rather than robust named train-derived levels.
3. Final projection did not independently accept/enforce a declared per-flow budget.
4. Exact identity could fail on a zero-duration boundary because generation floored duration.
5. Padding-only generation recomputed timing fields, and timing-only generation recomputed
   length fields; source residuals could therefore alter unrelated features. Writes are now
   conditional on the active primitive.
6. Tests proved non-zero gradients but did not compare victim-loss autograd to finite
   differences.
7. No separate semantic validator or conservative tri-state aggregation existed.
8. Per-sample artifacts lacked complete costs, budget compliance, semantic results, and changed
   feature provenance.
9. The direct runner's legacy `strict_valid` name conflated validity levels. Explicit staged
   metrics replace it.
10. The old golden latent-output regression pinned obsolete 1460-byte/100x defaults and exact
    optimizer internals. It and its fixture were removed rather than re-pinned to the new
    contract; behavioral tests now cover the stable public invariants.

## Files changed

Implementation:

- `src/attack/realizability/base.py`;
- `src/attack/realizability/cicids2017.py`;
- `src/attack/primattack_budget.py`;
- `src/attack/flow_semantics.py`;
- `src/attack/run_cicids2017_primitive_attack.py`;
- `src/attack/vae_latent_primitive.py`;
- `src/attack/run_cicids2017_vae_latent_attack.py`;
- `scripts/budget_sweep_primitive.py`;
- `scripts/analyze_primattack_experiments.py`;
- `scripts/latent_budget_sweep.py`;
- `scripts/latent_strength_sweep.py`.

Tests:

- `src/attack/tests/test_primitive_controls.py`;
- `src/attack/tests/test_vae_latent_primitive.py`;
- `src/attack/tests/test_primattack_transformation.py`;
- `src/attack/tests/test_primattack_budget.py`;
- `src/attack/tests/test_flow_semantics.py`;
- removed obsolete `tests/test_golden_attack_regression.py` and its pinned NPZ fixture.

Artifacts and documentation:

- `artifacts/primattack/budget_calibration.json`;
- `docs/primattack/01_PRIMITIVE_AUDIT.md` through `06_RESULTS_GUIDE.md`;
- `docs/primattack/FINAL_PRIMATTACK_AUDIT.md`;
- `docs/primattack_attack_preservation.md`.

## Final primitives

### Uniform forward packet-length augmentation $p$

- Units: bytes per forward packet.
- Type: continuous relaxation during optimization; discrete integer at final evaluation.
- Direction: increase only; identity 0.
- Capability: forward count and positive forward total/mean must be present.
- Final projection: clamp to per-flow budget, round, then cap by `floor(p_hi)`.

Equations:

$$L_f'=L_f+N_fp,$$

$$l_{f,min}'=l_{f,min}+p,\quad l_{f,max}'=l_{f,max}+p,$$

$$\bar l_f'=L_f'/\max(N_f,1).$$

Dependencies: forward total/min/max/mean, forward segment average, combined packet
min/max/mean/std/variance, average packet size, and flow byte rate. Forward length std is a
proven invariant under a uniform shift.

### Uniform forward-IAT dilation $\alpha$

- Units: dimensionless ratio.
- Type: continuous; dependent integer timing fields are quantized to microseconds.
- Direction: delay only; identity 1.
- Capability: at least two forward packets and positive forward-IAT total.
- Final projection: clamp to the per-flow alpha cap.

Equations:

$$T_f'=\alpha T_f,$$

$$D'=\max\{D+(T_f'-T_f),T_f',T_b,1\ \mu s\},$$

$$\overline{IAT}_f'=T_f'/\max(N_f-1,1),\qquad
\overline{IAT}_{flow}'=D'/\max(N_f+N_b-1,1).$$

Dependencies: forward-IAT total/mean/std/max/min, projected duration, flow-IAT mean/max, and all
four packet/byte rates.

`Flow IAT Max` and duration are conservative flow-level projections. Bulk, subflow,
active/idle, merged Flow-IAT std/min, and data-bearing-packet semantics remain UNKNOWN from the
aggregate row and are held constant under an explicit `LEVEL_C` role.

## Budget calibration

Source: pristine training split only. The artifact embeds hashes for the training features,
training labels, and preprocessing manifest. No victim prediction, adversarial success, test
feature, or validation feature participates.

- Padding population: positive class-conditional `Fwd Packet Length Mean`.
- Timing population: absolute relative deviation of `Flow Duration` from its class median.
- Restricted/intermediate/maximum-evaluated: empirical P25/P50/P75.
- Per-flow envelope: named budget intersected with complete-training P99 feature headroom,
  mathematical capability, and DoS/DDoS semantic rate headroom.

Exact values (`padding bytes / max relative duration increase`):

| Class | Restricted | Intermediate | Maximum-evaluated |
|---|---:|---:|---:|
| DoS | 41 / 0.042725609756097564 | 47 / 0.5671219512195121 | 54 / 1.2682256097560975 |
| DDoS | 2 / 0.22199772960647784 | 2 / 0.4352960736292373 | 3 / 0.6874252351844157 |
| Recon | 2 / 0.0851063829787234 | 2 / 0.2127659574468085 | 10 / 0.5319148936170213 |
| BruteForce | 11 / 0.06174194934404961 | 12 / 0.11989485807317402 | 91 / 0.23366757427033438 |

These are maximum evaluated flow-level envelopes, not universal physical maxima.

## Semantic calibration and rules

All classes require unchanged labels, protocol, service ports, endpoint/direction metadata,
packet counts, control flags, finite output, non-decreasing represented traffic volume, declared
feature writes only, and hard-budget compliance.

DoS/DDoS additionally require adversarial `Flow Packets/s` at or above the class training P05:

- DoS: 0.6247806906700134 packets/s;
- DDoS: 1.0654268741607666 packets/s.

Recon corresponds to PortScan. Complete port set, scan sequence, and distinct connection-attempt
count are not testable from a single aggregate flow row. BruteForce authentication-attempt
count, credential/payload semantics, and server authentication outcome are likewise not
testable. These critical gaps force `NOT_FULLY_TESTABLE` unless another required test fails.

## Primary metrics

For the clean-correct eligible denominator, the runner reports:

1. raw targeted ASR;
2. targeted ASR gated by validator_v2 domain validity;
3. targeted ASR gated by domain validity and primitive feasibility;
4. SP-ASR gated by domain validity, primitive feasibility, and semantic `PASS`.

SP-ASR is explicitly a flow-level semantic-preservation proxy metric.

## Verification results

### Complete unit suite

```text
PYTHONPATH=".;src" python -m pytest -q -p no:faulthandler
```

Observed: **217 passed, 1 skipped, 1 warning** in 7.92 seconds. The warning is the existing
PyTorch even-kernel `padding='same'` warning in `CNNOnly`; it is unrelated to PrimAttack. The
skip is a data-conditional legacy case with no matching sampled row.

The suite includes:

- exact zero-control identity;
- immutable-field and declared-dependency isolation;
- timing, padding, and confirmed derived equations;
- discrete projection;
- central finite-difference agreement with autograd through the scaler and trained MLP victim;
- zero/tiny-duration and non-finite-input safety;
- separate intentional $p$ and $\alpha$ budget violations;
- generic semantic-invariant violations;
- Recon/BruteForce NOT_TESTABLE behavior;
- deterministic reproduction of frozen train-only budget values.

### Smoke attack

```text
PYTHONPATH=".;src" python -m attack.run_cicids2017_primitive_attack \
  --classes DoS --victims mlp --device cpu --test-limit 16 --steps 2 --seeds 42 \
  --budget maximum-evaluated --primitive-mode joint \
  --output-dir outputs/primattack_smoke
```

Observed: 16 eligible, 100% domain-valid, 100% primitive-feasible, 15 semantic PASS, 1 semantic
FAIL, and no targeted successes. All final artifacts were finite. This tiny two-step run proves
the integrated path executes; it is not an efficacy estimate.

### Matched ablation/sensitivity pilot

```text
PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py \
  --classes DoS,DDoS,Recon,BruteForce --victims mlp \
  --test-limit 64 --steps 10 --seeds 42 \
  --output-dir outputs/primattack_budget_sensitivity
```

All nine mode/budget combinations used identical source IDs. Pooled eligible $N=254$ per
condition. This deliberately short pilot had 0 raw/valid/primitive-feasible/SP targeted ASR in
all cells; it must not be presented as the final efficacy experiment. Maximum-evaluated medians:

| Mode | Relative duration | Relative bytes | Rate retention | Changed features |
|---|---:|---:|---:|---:|
| timing-only | 0.0187785893 | 0 | 0.9815675780 | 11 |
| padding-only | 0 | 0.0007748601 | 1.0 | 10 |
| joint | 0.0188123317 | 0.0007748601 | 0.9815350339 | 20 |

The pooled semantic testability rate was 0.503937 because Recon and BruteForce critical
properties are intentionally not fully testable. Forty-one generated NPZ artifacts across the
smoke, sensitivity, and random-control runs were scanned; no required numeric field contained
NaN or Inf.

Paired statistics completed with 12 binary and 18 continuous families. Seven sensitivity plots
were generated. A same-budget random-feasible pilot also completed with 100% domain validity and
primitive feasibility and 0 targeted successes.

## Exact experiment commands

Calibration, final run, ablations, random control, paired statistics, and focused verification
commands are listed in `05_EXPERIMENT_PROTOCOL.md`. The recommended full evaluation is:

```text
PYTHONPATH=".;src" python -m attack.run_cicids2017_primitive_attack \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn,lstm,serial \
  --budget maximum-evaluated --primitive-mode joint --optimizer optimized \
  --seeds 42,43,44 --output-dir outputs/primattack_calibrated_joint_max
```

No model retraining is required.

## Remaining limitations

- The uniform-$p$ and duration/Flow-IAT-Max constructions are declared flow-level primitive
  models; they are not packet re-extractions.
- Aggregate data cannot recover merged packet order, burst boundaries, payload meaning,
  application outcomes, full scan structure, or target response.
- The semantic proxy is transparent and conservative but not a substitute for isolated
  packet-level realization/replay.
- The completed 64-row/class MLP sensitivity run is a pipeline/pairing pilot. Thesis efficacy
  tables require the predeclared full victim/sample/seed protocol and must be reported even if
  success is low.

Full verification of malicious attack functionality would require realizing the adversarial
modifications at packet level, re-extracting the resulting CICFlowMeter features, and replaying
the transformed traffic in an isolated environment against an appropriate service or target.
Such packet-level realization and replay are outside the scope of this thesis.
