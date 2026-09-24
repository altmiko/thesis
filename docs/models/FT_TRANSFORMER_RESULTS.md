# FT-Transformer — Clean Results (CICIDS2017-DistriNet, category head)

Normally-trained victim classifier. Same split / scaler / labels / class-weighting / seed
as the existing MLP and CNN victims. Selection on **validation macro-F1**; evaluation on
the **untouched test split** (312,056 rows). No test-set tuning, no resampling, no
adversarial training.

## Architecture
Local, dependency-free FT-Transformer (`src/classifiers/ft_transformer.py`,
`architecture_version = ft_transformer.v1`) — paper-faithful
(Gorishniy et al., NeurIPS 2021):

- Per-feature `NumericalFeatureTokenizer` (`T_j = b_j + x_j·W_j`, own `W_j`,`b_j` per feature)
- Learned CLS token (appended; head reads CLS only)
- 3 PreNorm blocks: MHSA + ReGLU FFN; first block skips pre-attention norm (rtdl default)
- Head: LayerNorm → ReLU → Linear → raw logits

| Hyperparameter | Value |
|---|---|
| input features | 79 (numeric model-space, RobustScaler) |
| n_blocks | 3 |
| d_token | 192 |
| attention heads | 8 |
| attention dropout | 0.2 |
| FFN activation | ReGLU |
| FFN hidden multiplier | 4/3 (d_hidden = 256) |
| FFN dropout | 0.1 |
| residual dropout | 0.0 |
| input transform | asinh (parity with MLP/CNN) |
| optimizer | AdamW (lr 1e-4, weight_decay 1e-5; no-decay for tokenizer/CLS/bias/LayerNorm) |
| loss | balanced-weighted CrossEntropy (train-only weights) |
| batch size | 2048 |
| grad clip | 5.0 |
| epochs / patience | 10 / 3 (early-stopped; best epoch **7**) |
| seed | 42 |
| **trainable parameters** | **922,949** |

## Training
- Train rows 1,456,265 · Val 312,058 · Test 312,056 (fixed artifacts).
- Device: CUDA (RTX 4070 Ti SUPER), full precision.
- Training time: **1,523.2 s** (~25.4 min). Test inference: **7.91 s** (312,056 rows).
- Best validation macro-F1 (epoch 7): **0.87959** (val acc 0.85035, balanced-acc 0.96220).
  Train accuracy saturates ~1.0 by epoch 2 → early stopping on val macro-F1 is doing the
  model selection (no test leakage).

## Clean test metrics (category, 5-class)
| Metric | Value |
|---|---|
| accuracy | 0.984580 |
| balanced accuracy | 0.990988 |
| macro precision | 0.967661 |
| macro recall | 0.990988 |
| macro F1 | 0.977961 |
| weighted F1 | 0.985137 |
| loss | 0.151714 |

### Per-class (test)
| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Benign | 0.998773 | 0.981741 | 0.990184 | 247,164 |
| DoS | 0.999882 | 0.991800 | 0.995825 | 25,733 |
| DDoS | 1.000000 | 0.999439 | 0.999720 | 14,265 |
| Recon | 0.840625 | 0.997317 | 0.912291 | 23,852 |
| BruteForce | 0.999026 | 0.984645 | 0.991783 | 1,042 |

### Confusion matrix (test; rows = true, cols = pred; order Benign,DoS,DDoS,Recon,BruteForce)
```
[242651,     2,     0,  4510,     1]
[   211, 25522,     0,     0,     0]
[     8,     0, 14257,     0,     0]
[    63,     1,     0, 23788,     0]
[    16,     0,     0,     0,  1026]]
```
Dominant error = 4,510 Benign→Recon (drives Recon precision 0.84) — the same
Benign/Recon confusion the MLP and CNN also exhibit, i.e. a dataset property, not a
model artifact.

Artifacts:
- checkpoint `outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt`
- metrics `outputs/cicids2017distrinet_ft/metrics/ft_transformer_category_metrics.json`
- confusion plot `outputs/cicids2017distrinet_ft/plots/ft_transformer_category_confusion.png`
- history `outputs/cicids2017distrinet_ft/histories/ft_transformer_category_history.json`
- run manifest `outputs/cicids2017distrinet_ft/classifier_run_manifest.json`
- confusion plot `outputs/cicids2017distrinet_ft/plots/ft_transformer_category_test_confusion.png`
## Comparison vs MLP / CNN (category, same test split)
Source: `outputs/cicids2017distrinet/classifier_metrics_summary.csv` (MLP, CNN).

| Model | Params | Test acc | Balanced acc | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| SimpleMLP | (hidden 256-128-64) | 0.984477 | 0.990309 | 0.977251 | 0.985038 |
| CNNOnly | (pool 8) | 0.984330 | 0.989138 | 0.975278 | 0.984888 |
| **FT-Transformer** | **922,949** | **0.984580** | **0.990988** | **0.977961** | **0.985137** |

FT-Transformer is marginally ahead of MLP and CNN on every category-head aggregate here,
but the differences are small (≤0.003 macro-F1). This is an **architectural-diversity**
result, not a claim that FT-Transformer is universally superior — the three victims are
effectively at parity on clean CICIDS2017-DistriNet classification. Binary head (secondary):
FT-Transformer test acc 0.984551, macro-F1 0.977087 (`ft_transformer_binary.pt`).

## Methodological note
CICIDS2017-DistriNet rows are modeled as **independent flow-level tabular vectors**.
FT-Transformer attends across **feature tokens within one flow** — no temporal windowing,
no pseudo-sequences, no positional encoding over CICFlowMeter column order. Feature identity
is carried by the per-feature tokenizer parameters.
