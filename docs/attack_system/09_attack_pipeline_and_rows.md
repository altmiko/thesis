# 09 — Attack pipeline, rows, and active victims

## Pipeline

```text
TRAIN-only preprocessing/calibration
  -> fixed pristine test rows per source class
  -> active victim (mlp or cnn)
  -> primitive or latent optimization
  -> hard control projection
  -> dependent-feature regeneration
  -> train-fitted scaling
  -> victim prediction
  -> domain, primitive, semantic, and realism evaluation
  -> per-sample artifact
```

## Source rows

Rows are selected deterministically before victim and seed loops. The same source IDs are reused
across budgets and primitive ablations, which permits paired statistical testing.

## Active trained victims

- `mlp` — `SimpleMLP`;
- `cnn` — `CNNOnly`.

Both map a 79-feature RobustScaler-space vector to five category logits. Checkpoint loading fails
on preprocessing hash, class order, feature width, model type, or class-count mismatch.
Parameters are frozen while gradients to the input remain enabled.

FT-Transformer code exists for future integration but is outside the active trained/result
roster until a matching checkpoint and complete evaluation are available.

## Reproduction

```text
PYTHONPATH=".;src" python scripts/budget_sweep_primitive.py \
  --classes DoS,DDoS,Recon,BruteForce \
  --victims mlp,cnn \
  --test-limit 512 --steps 40 --seeds 42 \
  --output-dir outputs/primattack_budget_sensitivity_full
```

Outputs contain immutable row IDs and hashes for source code, preprocessing, scaler, checkpoints,
and calibration artifacts.
