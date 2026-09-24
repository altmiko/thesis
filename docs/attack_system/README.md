# CICIDS2017-DistriNet attack subsystem

This documentation covers the active MLP/CNN victim pipeline, primitive transformation,
validators, and flow-level semantic-preservation evaluation.

## Active victim roster

- `mlp`: `SimpleMLP`;
- `cnn`: `CNNOnly`.

FT-Transformer code is reserved for future checkpoint integration and is not included in current
attack result tables.

## Attack flow

```text
pristine malicious flow
  -> primitive or latent optimizer
  -> hard primitive projection
  -> deterministic dependent-feature recomputation
  -> train-fitted scaler
  -> active victim
  -> domain-validity gate
  -> primitive-feasibility gate
  -> flow-level semantic-preservation proxy
  -> staged ASR metrics and per-sample audit record
```

## Canonical sources

| Concern | Source |
|---|---|
| Dataset contract | `src/datasets/cicids2017.py` |
| Victim loader | `src/classifiers/cicids2017d_victims.py` |
| Primitive model | `src/attack/realizability/cicids2017.py` |
| Budget calibration | `src/attack/primattack_budget.py` |
| Semantic proxy | `src/attack/flow_semantics.py` |
| Direct runner | `src/attack/run_cicids2017_primitive_attack.py` |
| Full results | `docs/primattack_budget_results.md` |

All calibration remains train-only. Packet-level realization and replay remain outside scope.
