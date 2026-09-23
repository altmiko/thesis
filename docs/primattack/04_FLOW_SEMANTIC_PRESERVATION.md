# Flow-level semantic preservation proxy

## Meaning and claim boundary

Domain validity asks whether a feature vector satisfies the project's network/domain rules.
Primitive feasibility asks whether the requested operation projects into the declared PrimAttack
space and calibrated hard budget. Flow-level semantic preservation asks whether measurable
attack-related properties available in CICIDS2017 are retained. These are separate gates.

The implementation is `FlowSemanticValidator` in `src/attack/flow_semantics.py`. It is not
merged into validator_v2 and is deliberately not named a functionality validator.

This study evaluates attack-semantic preservation at the flow-feature level. CICIDS2017
represents network activity through aggregated flow statistics and therefore cannot directly
establish whether a transformed adversarial flow would retain its complete real-world malicious
functionality. We consequently use an offline semantic-preservation proxy based on immutable
flow attributes, attack-specific behavioral statistics, and bounded primitive modifications.
Full functionality verification would require realizing the adversarial modifications at packet
level, re-extracting the resulting flow features, and replaying the modified traffic in an
isolated test environment against the corresponding application or service. Packet-level
realization and replay are outside the scope of this thesis.

## Check result model

Every check returns:

- `PASS`;
- `FAIL`;
- `NOT_TESTABLE`;
- a machine-readable `reason_code`;
- an explanation;
- whether the check is required.

Per sample:

- `FAIL` if any required check fails;
- `NOT_FULLY_TESTABLE` if no required check fails but at least one critical required property
  is unavailable;
- `PASS` only if every required flow-level proxy check is testable and passes.

`NOT_TESTABLE` is never converted to `PASS`.

## Generic invariants

Required checks, where represented:

1. attack-label metadata unchanged;
2. protocol unchanged;
3. source/destination service ports unchanged;
4. source/destination IP metadata unchanged, which also preserves represented direction;
5. packet counts unchanged because no count/injection primitive exists;
6. TCP control/flag aggregates unchanged;
7. all generated features finite;
8. no represented traffic-volume decrease;
9. changed features are a subset of the active primitive's declared dependencies;
10. projected controls and realized duration/byte costs comply with the named hard budget.

Unavailable metadata produces `NOT_TESTABLE`, not success.

## Class strategies

The retained final categories are `DoS`, `DDoS`, `Recon`, and `BruteForce`. The underlying
Recon training label is PortScan; BruteForce contains FTP-Patator and SSH-Patator.

### DoS and DDoS

The proxy retains protocol, endpoints/service, direction metadata, counts, traffic volume,
flags, and all generic invariants. It explicitly quantifies

$$\text{rate retention}=\frac{R_{adv}}{\max(R_0,\epsilon)}$$

and

$$\text{duration ratio}=\frac{D_{adv}}{\max(D_0,\epsilon)}.$$

Attack-like intensity is required to remain at or above the class training P05 of
`Flow Packets/s`: 0.6247806906700134 for DoS and 1.0654268741607666 for DDoS. The timing bound
also incorporates that threshold before optimization. The threshold was chosen from training
behavior, not ASR.

Passing this proxy does not prove that a DoS/DDoS flow would still deny service.

### Recon / PortScan

Protocol, service ports, endpoint/direction metadata, flags, counts, volume, declared-change,
and budget invariants remain testable. The following critical properties are not present in one
aggregate flow row:

- complete scanned-port set;
- scan sequence;
- number of distinct connection attempts.

They return `NOT_TESTABLE_FROM_FLOW_DATA`; therefore Recon samples are conservatively
`NOT_FULLY_TESTABLE`, unless another required condition fails first.

### BruteForce

Protocol, service ports, endpoint/direction metadata, flags, counts, volume, declared-change,
and budget invariants remain testable. Aggregate CICFlowMeter features cannot establish:

- authentication-attempt count;
- credentials or payload semantics;
- server authentication outcome.

These are critical and return `NOT_TESTABLE_FROM_FLOW_DATA`; therefore BruteForce samples are
conservatively `NOT_FULLY_TESTABLE`, unless another required condition fails first.

## Testability coverage

Every attack configuration reports:

$$\text{semantic testability rate}=\frac{\#\{status\ne\text{NOT\_FULLY\_TESTABLE}\}}{\#\text{eligible}}.$$

PASS, FAIL, and NOT_FULLY_TESTABLE counts and rates are separate. This prevents unavailable
checks from inflating preservation claims.

## Distinction from real-world functionality

A semantic `PASS` means only that all required, available flow-level proxy checks passed under
predefined training-derived thresholds. It provides no guarantee about complete malicious
behavior, deployment behavior, or packet-trace validity. Real-world functionality preservation
is not established by this thesis.
