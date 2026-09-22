# Constrained adversarial NIDS experiments

## PAVE-style validity checking

`src/evaluation/pave_style_validator.py` provides a small, independent validity baseline inspired by the feature-constraint validation methodology in [robotumbel/PAVE_artifact](https://github.com/robotumbel/PAVE_artifact), especially `code/attacks/constraints.py` and `code/pave_audit.py`. It is an adapted evaluation baseline, not a claimed methodological novelty.

The validator operates only in original feature units. It reuses each dataset adapter's saved, training-fit `FeatureTransform` when attack artifacts are scaled; it never fits a scaler on validation, test, or adversarial data. Clear manifest domains are applied directly (for example non-negativity, `[0,1]` aggregate probabilities, and TTL/protocol bounds). Missing lower or upper bounds fall back to the raw training split's observed minimum or maximum. Finite, integer, and binary semantics are checked separately. Samples are inspected unchanged: no clipping, rounding, repair, or projection occurs before validation.

CICIoT2023 window-aggregated flags and TTL are intentionally continuous because the local schema documents fractional aggregates. CICIDS2017 packet/flag counts and exact protocol/code fields retain integer checks from the local manifest plus exact dataset overrides. Any feature lacking clear schema semantics is listed as uncertain in the generated feature audit and receives train-range validation only.

The existing constraint system remains optional and separate. When `constraints/<dataset>/mined.json` exists, the runner evaluates its Layer-2 rules after the PAVE-style checks and reports PAVE-style validity, mined-constraint validity, and their strict conjunction.

Fit on train, audit held-out genuine data, and save the fitted registry:

```bash
PYTHONPATH=".;src" python -m evaluation.run_pave_validity \
  --dataset ciciot2023 \
  --heldout-limit 10000 \
  --output-dir outputs/pave_validity
```

Audit one or more untouched scaled attack artifacts (`X_adv`/`x_adv`, `y_true`, clean predictions, and adversarial predictions):

```bash
PYTHONPATH=".;src" python -m evaluation.run_pave_validity \
  --dataset ciciot2023 \
  --load-validator outputs/pave_validity/ciciot2023_pave_validator.json \
  --attack-file path/to/attack_results.npz \
  --attack-space scaled \
  --output-dir outputs/pave_validity
```

Use `--attack-space raw` only when the stored adversarial vectors are already in original units. Use `--no-mined` to report the independent PAVE-style baseline alone. Reports include range/type/combined validity rates, per-feature and per-reason failures, mined and strict validity when enabled, raw ASR, and validity-gated ASR over the original clean-correct denominator.
