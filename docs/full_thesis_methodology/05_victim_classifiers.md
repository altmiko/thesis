# 5. Victim Classifiers (MODERATE DETAIL)

Three neural victims are trained on the 79-feature CICIDS2017-DistriNet data with a
5-class category head (Benign=0, DoS=1, DDoS=2, Recon=3, BruteForce=4) and a binary
head (Benign/Attack). RF/XGBoost baselines exist (`review_baselines.py`) but are
eval-only and not attacked.

### Source files
`src/classifiers/models.py` (SimpleMLP, CNNOnly), `ft_transformer.py` (FTTransformer),
`cicids2017d_experiments.py` (training driver), `cicids2017d_victims.py`
(`load_category_victim`, sha256/identity guards). Checkpoints in
`outputs/cicids2017distrinet/models/` (FT originals also in
`outputs/cicids2017distrinet_ft/models/`).

Common input: 79 RobustScaler-scaled features; an `asinh` input transform is applied
inside every model (`input_transform="asinh"`). Output: 5 (category) or 2 (binary) logits.

---

## 5.1 SimpleMLP (`models.py:31-99`)
- Architecture: for each `hidden_dim`: `Linear → ReLU → Dropout`; then `Linear → logits`.
- Trained config: `hidden_dims=(256,128,64)`, `dropout=0.3`, `input_transform=asinh`.
- Params (category): 61,957. Test macro-F1 ≈ **0.9773**.

## 5.2 CNNOnly (`models.py:102-222`)
- Architecture: `unsqueeze(1) → Conv1d(1→32, k=3, same) → ReLU → Conv1d(32→64, k=3,
  same) → ReLU → AdaptiveMaxPool1d(pool) → flatten → Linear(64·pool→64) → ReLU →
  Dropout → Linear → logits`.
- Trained config: `pool_size=8, conv=(32,64), k=3, fc_dim=64, dropout=0.3, asinh`.
- Params (category): 39,493. Test macro-F1 ≈ **0.9753**.
- Rationale: treats the feature vector as a 1-D signal; local conv filters over the
  (ordered) feature axis, a cheap non-MLP inductive bias.

## 5.3 FTTransformer (`ft_transformer.py:197-341`)
Local, dependency-free re-implementation (no `rtdl`); `arch_version = "ft_transformer.v1"`.

- **Numerical feature tokenization** (`NumericalTokenizer`): each scalar feature `x_j`
  becomes a `d_token`-dim token `T_j = b_j + x_j · W_j` (per-feature learned weight `W_j`
  and bias `b_j`). 79 features → 79 tokens.
- **CLS token**: a learned `d_token` vector appended (at the end) → 80 tokens.
- **Transformer blocks**: `n_blocks=3` PreNorm blocks, each = multi-head self-attention
  (`heads=8`, `attn_dropout=0.2`) over the 80 feature tokens + a ReGLU feed-forward
  (`ffn_multiplier=4/3` ⇒ `d_hidden=256`, `ffn_dropout=0.1`), residual connections
  (`residual_dropout=0.0`). The first block skips the pre-attention norm (standard FT
  trick). Self-attention lets each feature token attend to every other feature token.
- **Classification head**: `LayerNorm → ReLU → Linear` applied to the final **CLS**
  token → logits.
- Defaults: `d_token=192`. Params: category 922,949 / binary 922,370. Test macro-F1 ≈
  **0.9780**.
- **Status**: implemented, trained (artifacts exist in both
  `cicids2017distrinet/models` and `cicids2017distrinet_ft/models`), covered by
  dedicated tests, and present in the attack roster. The historical 72-artifact
  budget sweep omitted FT-Transformer, but the completed
  `outputs/full_adv_eval_primattack_v2/` campaign includes it on the same
  clean-correct paired protocol as MLP/CNN. Performance is parity, not superiority;
  training accuracy is approximately 1.0.

---

## 5.4 Training (`cicids2017d_experiments.py`)
- **Loss**: train-only balanced-weighted `CrossEntropyLoss`; balanced weights `N/(C·count)`
  (the `class_weights_*.npy` from preprocessing); an effective-number `beta=0.999`
  alternative is also available.
- **Optimizer**: MLP/CNN — `Adam`, `lr=1e-3`, `wd=0`. FT — `AdamW`, `lr=1e-4`,
  `wd=1e-5` with no weight decay on tokenizer/CLS/bias/LayerNorm.
- **Scheduler**: `ReduceLROnPlateau(mode="max"` on val macro-F1, `patience=1,
  factor=0.5)`.
- **Batch size** 2048; **grad-clip norm** 5; **early stopping** on best val macro-F1
  (val-loss tie-break) with patience.
- **Actually trained with** `epochs=10, patience=3` (CLI override of the argparse
  defaults 5/2); best epochs mlp=9, cnn=8, ft=7. `SEED=42`.
- **Loader guards** (`cicids2017d_victims.py:26-84`): `load_category_victim` verifies
  79 features / 5 classes, checks a sha256, sets `eval()` + `requires_grad_(False)`
  (input gradients still flow — needed by white-box attacks).

Checkpoints (bytes): `mlp_{binary,category}.pt` ≈250k, `cnn_*` ≈160k,
`ft_transformer_*` ≈3.7M. MLP/CNN checkpoints lack a metadata key (legacy); FT has
full metadata. The v2 paired campaign records all three victims in `config.json` and
stores a separate clean-correct source roster and hash per victim/class.

---

## 5.5 Assumptions · Limitations · Claims
- **Assumptions**: scaled+asinh features are adequate; class weights address imbalance
  enough for macro-F1; val macro-F1 is the selection metric.
- **Limitations**: all three overfit (train ≈ 1.0); CICIDS2017 victims use one
  training seed; no calibration/uncertainty analysis is reported. The three v2
  attack seeds vary optimizer restarts, not victim training.
- **Can claim**: three strong closed-set victims (macro-F1 ≈ 0.975–0.978) with
  identity-guarded checkpoints, trained train-only with balanced loss and early
  stopping, all evaluated by the PrimAttack-v2 campaign.
- **Must NOT claim**: FT-Transformer superiority (parity only), calibrated
  uncertainty, victim-training seed robustness, or general robustness outside the
  evaluated attacks and frozen test roster.
