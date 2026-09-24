# FT-Transformer — Final Audit

End-to-end record of adding an FT-Transformer victim to the CICIDS2017-DistriNet pipeline.

## Dataset facts (verified from repo, not documentation)
- **Feature count: 79** numeric model-space features (RobustScaler); order frozen by
  `data/processed/CICIDS_2017_Distrinet/preprocessing_manifest.json:modelling_feature_names`.
- **Classes (category victim head): 5** — `Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4`.
  No genuine categorical/embedding features exist; all columns tokenized numerically.

## Architecture implemented
Local FT-Transformer (`architecture_version = ft_transformer.v1`), paper-faithful
(Gorishniy et al., NeurIPS 2021): per-feature numerical tokenizer (`T_j=b_j+x_j·W_j`),
learned CLS token, 3 PreNorm blocks (MHSA + ReGLU FFN, first block skips pre-attn norm),
CLS-only head (LayerNorm→ReLU→Linear) → raw logits. **922,949** trainable params
(category head).

### Official package vs local — and why
**Local implementation (Option B).** `rtdl` is deprecated and neither `rtdl` nor
`rtdl_revisiting_models` is installed; `environment.yml` pins an explicit, minimal
dependency set. A compact local model adds **zero new dependencies**, is defense-auditable,
and reproduces the reference behavior (ReGLU FFN, PreNorm with first-block norm handling,
no-weight-decay parameter groups).

## Hyperparameters / training setup
AdamW (lr 1e-4, weight_decay 1e-5; no-decay for tokenizer/CLS/bias/LayerNorm),
balanced-weighted CrossEntropy (train-only), batch 2048, grad-clip 5.0, epochs 10 /
patience 3 (best val macro-F1, val-loss tie-break), seed 42, CUDA full precision.
Same split / scaler / labels / weighting / seed as MLP & CNN. **No** new split, **no**
resampling, **no** adversarial training, **no** test-set tuning.

## Scores
- Best **validation** macro-F1 (epoch 7): **0.87959**.
- **Clean test** (untouched, 312,056 rows): acc **0.984580**, balanced-acc **0.990988**,
  macro-F1 **0.977961**, weighted-F1 **0.985137**, loss 0.151714.
- Comparison (same test split): MLP macro-F1 0.977251, CNN macro-F1 0.975278,
  **FT 0.977961** — parity, FT marginally ahead. See `FT_TRANSFORMER_RESULTS.md`.
- Training time 1,523 s; test inference 7.91 s.

## Gradient / attack verification
- Unit tests: `src/classifiers/tests/test_ft_transformer.py` — **24 passed** (incl. GPU),
  covering input gradients finite/non-zero/no-NaN-Inf, autograd vs finite-difference,
  ReGLU dims, raw-logit contract, save/load determinism, invalid-schema rejection.
- Readiness harness (`scripts/ft_transformer_attack_readiness.py`, cuda, n=256):
  **ALL_READINESS_CHECKS_PASS = true** — PGD 256/256, C&W 250/256, PrimAttack `dL/dp`
  finite & non-zero, scaler applied once, Benign→class 0, feature order matches schema &
  checkpoint metadata. See `FT_TRANSFORMER_ATTACK_READINESS.md`.

## Checkpoint
`outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt`
(binary head: `ft_transformer_binary.pt`). Checkpoint keys:
`state_dict, model_type, model_kwargs, num_features, num_classes, metadata{…}` — metadata
stores architecture_version, ordered `feature_names`, label mapping, optimizer/lr/weight_decay,
seed, best_epoch, best_val_macro_f1, preprocessing_manifest_sha256, scaler/dataset/split ids.
Sibling `classifier_run_manifest.json` (preprocessing sha256) drives the victim-loader guards.
Loading fails loudly on schema/class/feature/model-type mismatch (`load_category_victim`).

## Files
**Inspected:** `datasets/cicids2017.py`, `datasets/base.py`, `datasets/feature_manifest.py`,
`classifiers/models.py`, `classifiers/cicids2017d_victims.py`, `classifiers/cicids2017d_experiments.py`,
`attack/latent_pgd.py`, `attack/input_baselines.py`, `attack/adversarial_attacks.py`,
`attack/realizability/cicids2017.py`, `attack/tests/test_primattack_transformation.py`,
`config/paths.py`, `preprocessing_manifest.json`, existing MLP/CNN checkpoints & metrics.

**Created:**
- `src/classifiers/ft_transformer.py`
- `src/classifiers/tests/test_ft_transformer.py`
- `scripts/ft_transformer_attack_readiness.py`
- `docs/models/FT_TRANSFORMER_{AUDIT,RESULTS,ATTACK_READINESS,FINAL_AUDIT}.md`

**Modified:**
- `src/classifiers/models.py` — import + register `'ft_transformer'` in `get_model`.
- `src/classifiers/cicids2017d_experiments.py` — `NN_MODELS`, `DISPLAY_NAMES`,
  `model_kwargs` (ft branch), `build_optimizer` (AdamW no-decay groups), enriched checkpoint
  `metadata`, CLI `--ft-learning-rate/--ft-weight-decay/--tasks`, `train_nn` signature + call.
- `src/attack/adversarial_attacks.py` — `_infer_model_type` recognizes `ft_transformer`.

## Exact commands
```bash
# thesis conda env (python 3.11, torch 2.5.1+cu121, CUDA)
# train FT-Transformer (both heads; category is the attack victim)
python -m src.classifiers.cicids2017d_experiments \
    --models ft_transformer --tasks all --epochs 10 --patience 3 \
    --batch-size 2048 --ft-learning-rate 1e-4 --ft-weight-decay 1e-5 \
    --class-weighting balanced --seed 42 --device cuda \
    --output-dir outputs/cicids2017distrinet_ft

# evaluate (metrics are written by the training run):
#   outputs/cicids2017distrinet_ft/metrics/ft_transformer_category_metrics.json
#   outputs/cicids2017distrinet_ft/cicids2017_classifier_results.md

# unit + gradient tests
python -m pytest src/classifiers/tests/test_ft_transformer.py -q

# attack-readiness (input-grad + PGD + C&W + targeted-Benign + PrimAttack grad)
PYTHONPATH=src python scripts/ft_transformer_attack_readiness.py \
    --victim outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt \
    --device cuda --n 256
```

## Remaining issues / deferred (by design)
- **Standalone CICIDS2017 attack suites** now use the primary victim set
  **`("mlp","cnn","ft_transformer")`**: `run_cicids2017_primitive_attack.py:VICTIMS` and
  `run_cicids2017_vae_attacks.py:VICTIMS` set it; `run_cicids2017_{input_baseline,
  latent_variants,vae_latent_attack}.py` inherit it via import. All five import cleanly and
  load all three victims (gradients verified). The FT category (and binary) checkpoints were
  copied into the runners' canonical `victim_dir` (`outputs/cicids2017distrinet/models/`),
  whose `classifier_run_manifest.json` (preprocessing sha256 `df93d07c…`, category classes,
  79 features) satisfies the `load_category_victim` guards. LSTM/CNN-LSTM are excluded from
  the primary victim set (code/checkpoints retained, not deleted — task §30).
- **Overfitting:** train accuracy saturates (~1.0) with high/oscillating val loss; validation
  macro-F1 early stopping performs selection. Not tuned further (architectural-diversity goal,
  no test-set tuning). FT-Transformer is **not** claimed superior — it is at parity with MLP/CNN.
- **Separate output dir** (`outputs/cicids2017distrinet_ft/`) used deliberately: the training
  script's `prepare_output_dir` wipes its target `models/`; this preserves the existing
  MLP/CNN/LSTM/serial checkpoints (task §30). LSTM/CNN-LSTM code and checkpoints untouched.
- **`torchattacks`** is imported transitively by the attack chain; absent in a bare runtime but
  pinned in `environment.yml` and present in the thesis env.
