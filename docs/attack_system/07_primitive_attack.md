# 07 — Direct primitive-domain attack

**Canonical detailed guide:** [`../full_thesis_methodology/02_primattack.md`](../full_thesis_methodology/02_primattack.md)  
**Runner:** `src/attack/run_cicids2017_primitive_attack.py`  
**Search:** `src/attack/primitive_optimizer.py`  
**Primitive map:** `src/attack/realizability/cicids2017.py`

PrimAttack is a targeted malicious-to-Benign white-box attack. It searches three controls
for two forward-flow operations:

- `p`: integer bytes added uniformly per forward packet;
- `delay`: integer microseconds added across the forward IAT sequence;
- `shape`: continuous allocation of that delay between proportional gap dilation
  (`0`) and equal additive delay per gap (`1`).

It never edits a CICFlowMeter feature independently. The canonical transform
`CICIDS2017PrimitiveModel.generate` maps the controls back to a raw 79-feature row,
recomputes declared dependent features, and copies everything else from the source.

## Attack path

```text
pristine raw flow
  -> infer padding/timing capability
  -> intersect class budget, train-p99 headroom, and semantic rate floor
  -> mask bounds for joint | timing-only | padding-only
  -> identity + exhaustive integer-padding search
  -> adaptive projected search over normalized (p, delay, shape)
  -> project p/delay to integer controls and clamp shape
  -> generate the quantized raw flow
  -> scale with the frozen train-fitted transform
  -> classify and evaluate all gates
```

Every candidate retained by the search is scored **after** projection and quantization.
The differentiable unquantized map is used only to obtain gradients.

## Search objective and selection

The objective is the targeted Benign logit margin:

```text
max(non-Benign logits) - Benign logit
```

Candidate selection is success-first. Among targeted successes it keeps the lowest
normalized `p + delay` cost; among failures it keeps the lowest margin. `shape` has no
direct cost because it only redistributes a fixed total delay.

Default search settings are 40 steps, learning rate 0.1, and two restarts. Restart zero
begins from the best identity/exhaustive-padding candidate. Later restarts are seeded
random points in the same hard box.

## Capability and hard bounds

Padding is enabled only when the source has at least one forward packet and positive
forward length/mean. Timing is enabled only when the source has at least two forward
packets and positive forward-IAT total. Unsupported operations receive a zero upper
bound before search.

Named budgets are fitted on the pristine training split:

- `restricted`: p25;
- `intermediate`: p50;
- `maximum-evaluated`: p75.

Each per-flow box also intersects the complete-training p99 feature envelope.
DoS/DDoS timing additionally stops before adversarial `Flow Packets/s` would fall below
the class-training p05.

The full paired driver also evaluates `unbounded`, which removes the class p25/p50/p75
caps but retains train-p99 headroom, capability gates, integer projection, and the
DoS/DDoS rate floor. It is envelope-only, not an unconstrained feature attack.

## Evaluation gates

All rates use the same clean-correct denominator:

1. targeted Benign success;
2. targeted success and validator_v2 `hybrid_valid`;
3. targeted success, validity, and primitive feasibility;
4. SP-ASR: all prior gates and flow-semantic status `PASS`.

The per-class VAE Mahalanobis IDR gate is reported separately as True-IDSR. It is not
part of structural validity or SP-ASR. Recon and BruteForce have critical behavior that
cannot be tested from one aggregate flow row, so their semantic status is
`NOT_FULLY_TESTABLE`; those rows remain in the denominator.

## Focused command

```powershell
$Env:PYTHONPATH = "src"
python -m attack.run_cicids2017_primitive_attack `
  --classes DoS,DDoS,Recon,BruteForce `
  --victims mlp,cnn,ft_transformer `
  --budget maximum-evaluated `
  --primitive-mode joint `
  --optimizer search `
  --steps 40 `
  --learning-rate 0.1 `
  --restarts 2 `
  --seeds 42,123,2024 `
  --output-dir outputs/primattack_calibrated
```

The standalone runner accepts `search` and `random-feasible`, the three primitive modes,
and the three named budgets. The envelope-only condition is available through
`scripts/run_full_adversarial_eval.py`.

## Outputs

The standalone runner writes:

- `run_manifest.json`;
- `attack_results.json`;
- `attack_artifacts/<class>_<victim>_seed<seed>.npz`.

Per-row artifacts include source IDs, clean/adversarial vectors, predictions and logits,
requested/projected controls, hard bounds, capability reasons, winning candidate source,
validator and feasibility masks, semantic status/reasons, IDR, primitive costs, changed
features, and provenance hashes.

For paired cross-method evaluation, use `scripts/run_full_adversarial_eval.py`. It freezes
one clean-correct row roster per victim/class, reuses it across every attack and seed, and
asserts artifact pairing by `sample_id`.

## Claim boundary

The transform is a deterministic aggregate-feature proxy. Sequence-dependent Level-C
features are held constant because packet order is unavailable. No PCAP is edited or
replayed, `NullPacketBackend` reports packet verification unavailable, and semantic PASS
does not establish complete malicious functionality.
