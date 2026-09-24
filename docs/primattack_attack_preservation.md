# PrimAttack attack-preservation implementation record

## Active scope

- Dataset: corrected DistriNet CICIDS2017.
- Active trained neural victims: MLP and CNN.
- Future transformer victim: code present, checkpoint/evaluation not yet active.
- Primitive controls: forward length augmentation `p` and timing dilation `alpha`.
- Evaluation gates: targeted success, domain validity, primitive feasibility, and flow-level
  semantic-preservation proxy.

Historical recurrent-model code, checkpoints, outputs, and reports were moved into the dated
recurrent-model archive under `old_root_files`. They are not part of the active roster or current
reported results.

## Current full experiment

```text
PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn \
  --test-limit 512 --steps 40 --seeds 42 \
  --output-dir outputs/primattack_budget_sensitivity_full
```

The experiment contains nine paired mode/budget conditions and eight class/victim cells per
condition. Each condition has 4,064 eligible clean-correct rows.

### Maximum-evaluated results

| Mode | Raw/valid/feasible successes | ASR | SP successes | SP-ASR |
|---|---:|---:|---:|---:|
| Timing-only | 0 | 0.00% | 0 | 0.00% |
| Padding-only | 10 | 0.25% | 0 | 0.00% |
| Joint | 10 | 0.25% | 0 | 0.00% |

All ten successes were BruteForce rows: six against MLP and four against CNN. They were
network/domain-valid and primitive-feasible. Critical authentication behavior is unavailable
from aggregate flow rows, so the semantic status is conservatively `NOT_FULLY_TESTABLE` and
SP-ASR remains zero.

Joint and padding-only found the same successful rows. Both significantly exceeded timing-only;
their pairwise difference had Holm-adjusted `p=1`.

## Current artifacts

- Full results: `docs/primattack_budget_results.md`.
- Comprehensive explanation: `primattack_sp_budget_explained.md`.
- Calibration: `artifacts/primattack/budget_calibration.json`.
- Machine-readable run: `outputs/primattack_budget_sensitivity_full/`.

The active full run contains 72 NPZ artifacts and 36,864 saved rows. Source IDs are identical
across all paired conditions. All projected rows are primitive-feasible; no required numeric
field contains NaN or Inf.

## Claim boundary

SP-ASR is a flow-level semantic-preservation proxy. The experiment does not establish complete
application behavior or packet-trace behavior. Packet realization, feature re-extraction, and
isolated replay remain outside scope.
