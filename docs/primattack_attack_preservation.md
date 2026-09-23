# PrimAttack attack-preservation implementation record

## Purpose

This root documentation file is the running engineering record for the CICIDS2017-DistriNet
PrimAttack extension. It records decisions, evidence, implementation changes, verification,
and claim boundaries. Detailed stable methodology belongs in `docs/primattack/`; this file
tracks how the repository is being changed.

## Non-negotiable scope

- Offline, controlled academic evaluation on the public corrected DistriNet CICIDS2017 data.
- No packet crafting, PCAP rewriting, live replay, Docker replay, exploitation, scanning,
  malware execution, or vulnerable-service interaction.
- The implementation may establish domain validity, primitive feasibility, and a flow-level
  semantic-preservation proxy. It cannot establish real-world malicious functionality.
- Calibration uses training data only. Test data cannot select budgets or semantic thresholds.
- Attack success cannot select budgets or semantic thresholds.
- `phi(x_original, p)` is the only permitted path from primitive controls to model features.

## Work record

### 2026-09-24 — repository and primitive audit

Inspected:

- `src/attack/realizability/base.py`;
- `src/attack/realizability/cicids2017.py`;
- `src/attack/realizability/validator.py`;
- `src/attack/run_cicids2017_primitive_attack.py`;
- `src/attack/vae_latent_primitive.py`;
- `src/attack/tests/test_primitive_controls.py`;
- `src/datasets/cicids2017.py`;
- the preprocessing manifest, class encoder, feature reference, and existing attack-system
  methodology.

Created `docs/primattack/01_PRIMITIVE_AUDIT.md` before changing attack implementation.

Findings:

1. The active transform already restricts direct PrimAttack to two controls: uniform forward
   packet-length augmentation `p` and forward-IAT dilation `alpha`. No arbitrary CICFlowMeter
   feature is an optimizer leaf.
2. The direct gradient path is primitive controls -> deterministic feature map -> train-fitted
   scaler -> victim -> targeted loss. The latent path adds decoder -> primitive collapse before
   the same canonical feature map.
3. Known forward-length, combined-length, timing, duration, and rate dependencies are
   recomputed. Packet-sequence-dependent bulk/subflow/active/idle/merged-IAT properties remain
   unavailable and are correctly labelled `LEVEL_C`, but must be described as UNKNOWN rather
   than invariant.
4. Final evaluation rounds `p` to integer bytes and integer timing outputs to microseconds.
5. Existing feasibility caps use train maxima plus CLI ceilings (`p_max=1460`,
   `alpha_max=100`), not a robust class-conditional calibration. These values cannot remain the
   thesis budget definition.
6. Existing tests prove non-zero gradients but do not perform a finite-difference comparison
   through scaler and victim.
7. There is no separate flow-semantic preservation validator or tri-state semantic result.
8. Artifact fields do not yet satisfy the required per-sample audit schema.
9. A no-op boundary issue is possible because duration is floored inside generation; all-row
   identity and tiny-duration behavior require explicit verification and correction.
10. Existing `strict_valid` naming in the direct runner is narrower than the intended staged
    domain/primitive/semantic gates and will be replaced with explicit metrics.

### Planned implementation order

1. Expand and freeze `PrimitiveSpec`; preserve the canonical transform.
2. Add transformation contract tests, including finite-difference and boundary cases.
3. Add a train-only budget and semantic-threshold calibration stage with a machine-readable
   artifact.
4. Add hard named budget projection and explicit primitive costs/compliance.
5. Add a separate tri-state flow-semantic validator and class plugins.
6. Extend attack artifacts and metrics without retraining victims.
7. Add matched timing-only, padding-only, combined, and budget-sensitivity execution.
8. Produce the six methodology documents and final audit after behavioral verification.

## Claim vocabulary

- **Domain validity:** the feature representation satisfies project domain/network constraints.
- **Primitive feasibility:** the projected primitive values remain within the declared
  calibrated primitive space and budget.
- **Flow-level semantic preservation:** measurable attack-related properties available in
  CICIDS2017 pass frozen train-derived proxy tests.
- **Real-world functionality preservation:** not established by this thesis.

## Implementation completed

### Primitive contract and transform

- Expanded `PrimitiveSpec` with dtype, direction, absolute bounds, dependency set, projection
  identifier, and semantic risk.
- Changed final projection to require per-flow bounds; $p$ is rounded and capped by
  `floor(p_hi)`, while $alpha$ is clamped to `alpha_hi`.
- Fixed exact no-op behavior for zero/tiny-duration rows.
- Fixed primitive write isolation: padding-only no longer rewrites timing-derived fields, and
  timing-only no longer rewrites length-derived fields because of source residuals.
- Added explicit rejection of non-finite source/control tensors.

### Calibration and budgets

- Added `src/attack/primattack_budget.py`.
- Generated `artifacts/primattack/budget_calibration.json` from
  `X_train_pristine.npy` and `y_train_cat.npy` only.
- Replaced the old 1460-byte/100x CLI defaults with class-conditional empirical P25/P50/P75
  levels and per-flow train-P99/semantic intersections.
- Added fixed train-P05 flow-rate thresholds for DoS/DDoS.

### Semantic proxy

- Added separate `src/attack/flow_semantics.py`.
- Generic invariants cover labels, protocol, service ports, endpoint/direction metadata,
  packet counts, TCP flags, finite values, no volume decrease, declared dependency writes,
  and hard-budget compliance.
- DoS/DDoS use class-conditional attack-like rate retention.
- Recon/PortScan returns critical unavailable scan-set/sequence/attempt properties as
  `NOT_TESTABLE_FROM_FLOW_DATA`.
- BruteForce returns authentication/payload/outcome properties as
  `NOT_TESTABLE_FROM_FLOW_DATA`.
- Aggregation is fail-first, then `NOT_FULLY_TESTABLE`, then `PASS`.

### Evaluation and audit artifacts

- Reworked `run_cicids2017_primitive_attack.py` to emit raw, valid, primitive-feasible, and
  SP-ASR gates; semantic coverage; complete per-sample primitive costs; requested/projected
  controls; changed features; and provenance.
- Added timing-only, padding-only, joint, and random-feasible modes.
- Reworked `scripts/budget_sweep_primitive.py` for all three calibrated budgets and seven plots.
- Added `scripts/analyze_primattack_experiments.py` for Cochran's Q/Friedman omnibus tests and
  Holm-corrected McNemar/Wilcoxon follow-ups.
- Updated the latent primitive path to consume the same calibration and hard projection API.

## Verification ledger

### Focused contract suite

```text
PYTHONPATH=\".;src\" python -m pytest \
  src/attack/tests/test_primitive_controls.py \
  src/attack/tests/test_primattack_transformation.py \
  src/attack/tests/test_flow_semantics.py \
  src/attack/tests/test_primattack_budget.py \
  src/attack/tests/test_vae_latent_primitive.py \
  -q -p no:faulthandler
```

Initial focused run: **41 passed, 1 skipped**. The final repository-wide run below includes the
later separate $p$/$\alpha$ violation parametrization. Covered exact identity, immutability,
dependency completeness, confirmed equations, discrete projection, central finite-difference
agreement through the trained MLP victim, tiny-duration safety, semantic invariant violations,
and conservative NOT_TESTABLE aggregation.

### End-to-end smoke

```text
PYTHONPATH=\".;src\" python -m attack.run_cicids2017_primitive_attack \
  --classes DoS --victims mlp --device cpu --test-limit 16 --steps 2 --seeds 42 \
  --budget maximum-evaluated --primitive-mode joint \
  --output-dir outputs/primattack_smoke
```

Observed: 16/16 eligible; domain validity 100%; primitive feasibility 100%; semantic PASS
15/16 and FAIL 1/16; no NaN/Inf; zero targeted successes in this deliberately tiny two-step
smoke. This is execution proof, not an efficacy estimate.

### Matched pilot sensitivity/ablations

Executed all nine `timing-only/padding-only/joint × restricted/intermediate/maximum-evaluated`
configurations on the same 64 source rows per retained class for the MLP victim, seed 42, ten
optimizer steps. `source_id_consistency.json` confirms exact source-ID equality across all
configurations (four class/victim/seed cells).

Pooled eligible rows per condition: 254. This short pilot produced 0 raw targeted ASR at every
condition, so it is not used as a final efficacy result. It did verify increasing timing cost,
unchanged byte cost for timing-only, zero duration cost for padding-only, explicit semantic
testability (about 50.39% pooled because Recon and BruteForce remain not fully testable), and
generation of all seven requested plots.

Paired analysis completed: 12 binary and 18 continuous test families. Degenerate all-equal
outcomes are reported as statistic 0, p=1 instead of NaN.

### Random feasible control

Ran the same 64-row/class, MLP, seed-42 maximum-evaluated joint configuration with
`random-feasible`. All cells remained 100% primitive-feasible and domain-valid; no targeted
success occurred in this pilot.

### Final repository audit

```text
PYTHONPATH=\".;src\" python -m pytest -q -p no:faulthandler
```

Observed: **217 passed, 1 skipped, 1 unrelated existing PyTorch convolution warning**. The old
golden latent-output regression and its fixture were removed because they pinned obsolete
1460-byte/100x defaults and exact optimizer internals; they were not re-pinned. Stable behavior
is covered by the new contract tests.

Scanned 41 generated PrimAttack NPZ artifacts: no NaN/Inf in final features, costs, retention,
or projected controls. Verified all 31 required per-sample audit fields in the smoke artifact.
The final claim-language scan found no prohibited functionality assertions or packet-level
implementation.
