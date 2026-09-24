# 07 — Direct primitive-domain attack

**Source:** `src/attack/run_cicids2017_primitive_attack.py`

The active direct attack optimizes two flow primitives, forward packet-length augmentation `p`
and forward timing dilation `alpha`, against the trained MLP and CNN victims. It never optimizes
an arbitrary CICFlowMeter feature independently.

## Active victim loop

```text
for class in DoS, DDoS, Recon, BruteForce:
    select fixed source rows
    infer per-flow primitive capability and calibrated bounds
    for victim in mlp, cnn:
        optimize continuous controls
        project to hard budget
        regenerate dependent features
        reclassify and evaluate all gates
```

## Current command

```text
PYTHONPATH=".;src" python -m attack.run_cicids2017_primitive_attack \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn \
  --budget maximum-evaluated \
  --primitive-mode joint \
  --optimizer optimized \
  --seeds 42 \
  --output-dir outputs/primattack_active
```

## Evaluation gates

The runner reports, over the same clean-correct denominator:

1. raw targeted ASR;
2. domain-valid targeted ASR;
3. primitive-feasible targeted ASR;
4. flow-level semantic-preservation proxy ASR.

Per-sample NPZ artifacts record source IDs, requested/projected controls, bounds, predictions,
validity, semantic status, primitive costs, changed features, and provenance.

Budget calibration and current results are documented in `docs/primattack/03_BUDGET_CALIBRATION.md`
and `docs/primattack_budget_results.md`.
