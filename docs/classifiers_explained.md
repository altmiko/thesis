# NIDS Classifier Models: Implementation Guide

## 1. Purpose and scope

This guide documents the neural-network classifiers currently implemented in
`src/classifiers/models.py`:

- `SimpleMLP` (`mlp`)
- `CNNOnly` (`cnn`)
- `LSTMOnly` (`lstm`)
- `SerialCNNLSTM` (`serial`)

The models classify preprocessed CICIoT2023 and CICIDS2017-DistriNet feature
vectors. They are discriminative models: each instance produces one
unnormalized logit per configured class. They do not generate traffic, enforce
domain constraints, or perform adversarial perturbations themselves. Those
responsibilities belong to preprocessing, VAE, attack, and validation modules.

This document describes the source as it exists now, including the input and
output contracts, tensor shapes, training wiring, checkpoint requirements,
correctness-sensitive details, and extension rules.

## 2. Source of truth and module boundaries

| Concern | Source |
|---|---|
| Feature names, order, and feature roles | `src/preprocessing/schema.py` |
| CICIoT2023 processed arrays | `data/processed/X_*.npy`, `y_*.npy` |
| CICIDS2017-DistriNet processed arrays | `data/processed/CICIDS_2017_Distrinet/` |
| Neural architectures and factory | `src/classifiers/models.py` |
| CICIoT2023 baseline training | `src/classifiers/baseline_experiments.py` |
| CICIDS2017-DistriNet training | `src/classifiers/cicids2017d_experiments.py` |
| Saved-model review and probability inference | `src/classifiers/review_baselines.py` |
| Input-space attack model loading | `src/attack/adversarial_attacks.py` |

`models.py` contains the complete four-model classifier roster used by this project.
Superseded architectures such as `DualPathIDS` and attention variants are not
part of the current model roster.

The feature order is a hard contract. The current CICIoT2023 pipeline uses 39
features in the order defined by `schema.py`. Reordering columns, dropping a
column, or applying a different scaler to an attack or evaluation input makes a
checkpoint appear to work while changing the meaning of every learned weight.

## 3. Shared model contract

### 3.1 Constructor contract

Every model is created with:

```python
model = get_model(
    model_type="mlp",      # "mlp", "cnn", "lstm", or "serial"
    num_features=39,
    num_classes=8,
)
```

`num_features` is the width of the preprocessed feature vector, and
`num_classes` is the number of contiguous label IDs. The factory maps names as
follows:

```text
mlp    -> SimpleMLP
cnn    -> CNNOnly
lstm   -> LSTMOnly
serial -> SerialCNNLSTM
```

Unknown names raise `ValueError`; the factory does not silently select a
fallback architecture.

### 3.2 Input and output

The normal classifier input is a `torch.float32` tensor of shape
`(batch_size, num_features)`:

```text
X:      (B, F)
logits: (B, K)
```

where `F = num_features` and `K = num_classes`. The models return logits, not
probabilities. For ordinary classification, pass them directly to
`torch.nn.CrossEntropyLoss`:

```python
logits = model(x_batch)
loss = torch.nn.functional.cross_entropy(logits, y_batch)
```

Do **not** apply `softmax` before `CrossEntropyLoss`. For reporting or attack
selection, probabilities can be computed afterward:

```python
probabilities = torch.softmax(logits, dim=1)
predicted_class = logits.argmax(dim=1)
```

Labels must be integer class IDs in `[0, K - 1]`, represented as
`torch.int64`/`torch.long`. The baseline loader converts NumPy features to
`float32` and labels to `int64` in `prepare_tensors`.

### 3.3 Device and mode

The model and input tensor must be on the same device:

```python
model = get_model("cnn", 39, 8).to(device)
x = x.to(device)
```

Training uses `model.train()`. Evaluation and attack scoring use
`model.eval()` and normally wrap inference in `torch.no_grad()` when gradients
are not required. Every model contains dropout in its dense head; leaving a
model in training mode during evaluation makes predictions stochastic.

### 3.4 Input validation

The current implementations reject malformed input with `ValueError` rather
than allowing a later convolution or recurrent-layer error to obscure the
problem:

- `SimpleMLP`, `CNNOnly`, and `SerialCNNLSTM` require exactly `(B, F)`.
- `LSTMOnly` accepts `(B, F)` or `(B, T, F)`.
- The last feature dimension must equal the constructor's `num_features`.

The validation checks shape, not semantic validity. It does not check feature
ranges, protocol consistency, binary values, or realism. Those checks belong to
the manifest-backed validators.

## 4. Model 1: `SimpleMLP`

### 4.1 Purpose

`SimpleMLP` is the dense baseline. It treats the entire feature vector as one
input and learns global interactions through fully connected layers. It makes
no claim that neighboring feature positions form a local signal or that rows
form a temporal sequence.

### 4.2 Architecture

Constructor defaults:

```text
hidden_dims = (128, 64)
dropout     = 0.3
```

The computation is:

```text
(B, F)
  -> Linear(F, 128)       [default first hidden layer]
  -> ReLU
  -> Dropout(0.3)
  -> Linear(128, 64)      [default second hidden layer]
  -> ReLU
  -> Dropout(0.3)
  -> Linear(64, K)
  -> (B, K) logits
```

The implementation is parameterized by `hidden_dims`. If a caller supplies
`(d1, d2, ..., dn)`, the repeated block becomes:

```text
Linear(F, d1) -> ReLU -> Dropout
Linear(d1, d2) -> ReLU -> Dropout
...
Linear(dn, K)
```

An empty tuple is technically supported and produces a direct `Linear(F, K)`
classifier.

### 4.3 Baseline configuration

`baseline_experiments.py` deliberately overrides the constructor default for
MLP training:

```python
model_kwargs["hidden_dims"] = (256, 128, 64)
```

Therefore, a checkpoint trained by the baseline runner must be reconstructed
with `(256, 128, 64)`, not the constructor default `(128, 64)`. The review
pipeline applies the same override. The attack loader can infer MLP hidden
widths from `features.*.weight` keys, but explicit reconstruction with the
training configuration is preferable for new tooling.

### 4.4 Strengths and limitations

**Strengths**

- Directly matches the tabular input contract.
- No artificial spatial or temporal ordering assumptions.
- Simple to inspect, train, attack, and use as a reference baseline.
- Dense layers can model interactions between any two features.

**Limitations**

- Parameter count grows with feature width and hidden width.
- It has no explicit inductive bias for local feature neighborhoods or
  sequences.
- It does not provide `return_features`; unlike the other custom neural
  models, its public forward path returns logits only.

### 4.5 Implementation checklist

When changing the MLP:

1. Keep the final layer width equal to `num_classes`.
2. Keep the final output as logits.
3. Preserve the `(B, F)` validation contract.
4. Update both training and review reconstruction if hidden dimensions change.
5. Treat a changed hidden-width configuration as a checkpoint incompatibility;
   old state dictionaries will not load into a differently shaped MLP.

## 5. Model 2: `CNNOnly`

### 5.1 Purpose and modeling assumption

`CNNOnly` applies a one-dimensional convolution over the ordered feature
vector. The model therefore assumes that adjacent positions in the fixed
schema may contain useful local patterns. This is a spatial/feature-order
inductive bias, not temporal modeling. The rows in a batch are independent.

Because CICIoT2023 is tabular, the meaning of "sequence length" here is the
number of ordered features, not the number of packets or flow timesteps.
Changing `FEATURE_NAMES` order changes the local neighborhoods seen by the
convolutions and invalidates the interpretation of a trained checkpoint.

### 5.2 Architecture and shapes

Constructor defaults:

```text
conv_channels  = (32, 64)
kernel_size    = 3
fc_dim         = 64
dropout        = 0.3
pool_size      = 1
input_transform = None
```

For input `(B, F)`:

```text
(B, F)
  -> unsqueeze(1)
(B, 1, F)
  -> Conv1d(1, 32, kernel_size, padding="same")
(B, 32, F)
  -> ReLU
(B, 32, F)
  -> Conv1d(32, 64, kernel_size, padding="same")
(B, 64, F)
  -> ReLU
(B, 64, F)
  -> AdaptiveMaxPool1d(pool_size)
(B, 64, pool_size)
  -> flatten
(B, 64 * pool_size)
  -> Linear(64 * pool_size, fc_dim)
(B, fc_dim)
  -> ReLU -> Dropout(0.3)
(B, fc_dim)
  -> Linear(fc_dim, K)
(B, K) logits
```

`padding="same"` keeps the convolution output length equal to `F` for both
odd and even kernels. This matters because the previous `kernel_size // 2`
padding convention was one position too long for even kernels. The default
kernel size is odd, but the implementation now has correct behavior for the
full constructor range supported by PyTorch.

The adaptive pool reduces every channel to `pool_size` position bins. The legacy
default `pool_size=1` is compact but orderless after pooling. The DistriNet runner
uses eight bins so the dense head retains coarse feature-location identity.

### 5.3 Feature extraction

`CNNOnly.forward` accepts `return_features=True`:

```python
logits, features = model(x, return_features=True)
```

The returned `features` tensor has shape `(B, fc_dim)` and is the dense-head
representation after its ReLU and dropout module. In evaluation mode dropout
is disabled. In training mode, feature extraction is stochastic by design
because the returned representation includes dropout.

### 5.4 Strengths and limitations

**Strengths**

- Captures interactions among nearby schema positions with shared filters.
- Adaptive pooling gives a fixed-size representation.
- Relatively small dense head and straightforward gradient path.

**Limitations**

- The local-neighborhood assumption is artificial if adjacent schema columns
  have unrelated semantics.
- Max pooling discards most position-specific information and retains only the
  strongest activation per channel.
- A CNN does not model relationships between separate batch rows or packet
  timesteps.
- Even-kernel `padding="same"` may cause a PyTorch performance warning because
  the framework can need an explicit padded copy; this is a performance note,
  not a shape or correctness failure.

### 5.5 Implementation checklist

1. Keep the input reshape exactly `(B, F) -> (B, 1, F)`.
2. Keep convolution output channels consistent with the subsequent layer and
   dense input width.
3. Preserve `padding="same"` if feature-position alignment is intended.
4. If replacing adaptive max pooling, update the dense input dimension and
   checkpoint migration logic.
5. Do not treat the convolution axis as a temporal axis in thesis claims.

## 6. Model 3: `LSTMOnly`

### 6.1 Purpose and input modes

`LSTMOnly` has three input interpretations:

1. **Legacy/vector-timestep mode:** `(B, F)` is converted to `(B, 1, F)`.
   This has only one recurrent timestep and therefore cannot learn recurrence
   across features.
2. **True sequence mode:** `(B, T, F)` is passed directly.
3. **Feature-sequence mode:** each scalar in `(B, F)` is projected to a token,
   receives a learned feature-identity embedding, and the fixed schema positions
   become the recurrent sequence.

The CICIDS2017-DistriNet runner uses feature-sequence mode. CICIoT2023 keeps the
legacy checkpoint-compatible default. Only true sequence mode is temporal, and
only when each timestep is a genuine observation rather than a schema position.

### 6.2 Architecture and defaults

Constructor defaults:

```text
hidden_dim           = 64
num_layers           = 1
bidirectional        = True
fc_dim               = 64
dropout              = 0.3
feature_sequence     = False
feature_embedding_dim = 16
input_transform      = None
```

Legacy 2D input follows `(B, F) -> (B, 1, F)`. In feature-sequence mode:

```text
(B, F) -> (B, F, 1) -> Linear(1, embedding_dim)
       + learned feature embedding (F, embedding_dim)
       -> (B, F, embedding_dim)
```

The recurrent layer is:

```text
LSTM(
    input_size=F,                 # legacy / true-sequence mode
    # or feature_embedding_dim,   # feature-sequence mode
    hidden_size=64,
    num_layers=num_layers,
    batch_first=True,
    bidirectional=bidirectional,
    dropout=dropout if num_layers > 1 else 0,
)
```

With the default bidirectional setting:

```text
(B, T, F)
  -> BiLSTM
h_n: (num_layers * 2, B, 64)
  -> concatenate final forward and backward states
(B, 128)
  -> Linear(128, 64)
  -> ReLU -> Dropout(0.3)
(B, 64)
  -> Linear(64, K)
(B, K) logits
```

With `bidirectional=False`, the recurrent representation is `(B, hidden_dim)`
instead of `(B, 2 * hidden_dim)`.

### 6.3 Correct bidirectional state selection

For a bidirectional LSTM, `output[:, -1, :]` is **not** the complete final
state representation:

- its forward half is the forward direction's final state;
- its backward half is the backward direction's state at the last input
  timestep, which has not processed the full sequence in the backward order.

The implementation correctly uses `h_n` from the last recurrent layer:

```python
if bidirectional:
    recurrent_features = torch.cat((h_n[-2], h_n[-1]), dim=1)
else:
    recurrent_features = h_n[-1]
```

For multiple layers, `h_n[-2]` and `h_n[-1]` still refer to the final layer's
forward and backward states because PyTorch stores recurrent states by layer
and direction.

### 6.4 Dropout behavior

The LSTM's own `dropout` argument is active only when `num_layers > 1`; PyTorch
does not apply inter-layer recurrent dropout for a single-layer LSTM. The dense
head always contains `Dropout(dropout)`. Therefore, the constructor's dropout
value affects the dense head for every configuration and affects recurrent
inter-layer connections only for multi-layer configurations.

### 6.5 Feature extraction

With `return_features=True`, the model returns:

```python
logits, features = model(x, return_features=True)
```

where `features` is the post-ReLU, post-dropout dense representation of shape
`(B, fc_dim)`.

### 6.6 Limitations and sequence extensions

The current implementation does not accept sequence lengths or a padding mask.
If sequences of different lengths are padded, the final hidden state can include
padded timesteps and the result can be wrong. A correct variable-length
extension must add a lengths/mask contract and use an appropriate packed or
masked recurrent path consistently in training, evaluation, and attacks.
Do not silently infer lengths from feature values.

For true temporal windows, define explicitly what one timestep represents,
keep feature order fixed within each timestep, and ensure train/validation/test
splits remain leakage-safe. A 2D flow vector should not be relabeled as a
sequence merely because it can be reshaped to one timestep.

## 7. Model 4: `SerialCNNLSTM`

### 7.1 Purpose

`SerialCNNLSTM` applies the CNN feature extractor first and then treats each
feature position as one timestep for a recurrent layer. It is therefore a
serial feature-order model:

```text
raw ordered feature vector -> local convolutional features -> sequence model
```

The LSTM sequence length is the number of schema features after convolution,
not the number of network packets or time windows.

### 7.2 Architecture and shapes

Constructor defaults:

```text
conv_channels = (32, 64)
kernel_size  = 3
lstm_hidden  = 64
fc_dim       = 64
dropout       = 0.3
```

For input `(B, F)`:

```text
(B, F)
  -> unsqueeze(1)
(B, 1, F)
  -> Conv1d(1, 32, kernel_size, padding="same")
(B, 32, F)
  -> ReLU
  -> Conv1d(32, 64, kernel_size, padding="same")
(B, 64, F)
  -> ReLU
  -> transpose(1, 2)
(B, F, 64)
  -> bidirectional LSTM(hidden_size=64)
  -> concatenate final forward/backward h_n states
(B, 128)
  -> Linear(128, fc_dim)
(B, fc_dim)
  -> ReLU -> Dropout(0.3)
(B, fc_dim)
  -> Linear(fc_dim, K)
(B, K) logits
```

The transpose is essential. `Conv1d` uses `(B, channels, length)`, while
`batch_first=True` LSTM expects `(B, sequence, input_size)`. The second
convolution's 64 channels become the LSTM's `input_size`, and the original
feature axis becomes the LSTM sequence axis.

### 7.3 Correct bidirectional state selection

As in `LSTMOnly`, the implementation selects the final forward and backward
states from `h_n` rather than taking only `output[:, -1, :]`:

```python
_, (h_n, _) = self.lstm(sequence)
lstm_features = torch.cat((h_n[-2], h_n[-1]), dim=1)
```

This preserves full-sequence information in both directions. It also keeps the
representation width aligned with the dense layer (`2 * lstm_hidden`).

### 7.4 Feature extraction

`return_features=True` returns the post-ReLU, post-dropout dense representation
with shape `(B, fc_dim)`:

```python
logits, features = model(x, return_features=True)
```

### 7.5 Strengths and limitations

**Strengths**

- Convolutions provide local feature mixing before recurrence.
- The recurrent stage can integrate information across the ordered feature
  positions.
- It uses a fixed-size representation independent of the number of positions
  supplied to the dense head, provided the recurrent input contract remains
  valid.

**Limitations**

- It compounds two assumptions about arbitrary tabular order: local adjacency
  matters, and the ordered feature positions form a sequence.
- It has an information bottleneck: the convolutional representation is
  summarized by the final bidirectional recurrent states before classification.
- It is not a temporal flow-window model under the current `(B, F)` training
  data.
- It has no sequence-padding mask or explicit length support.

### 7.6 Implementation checklist

1. Keep convolution layout `(B, C, F)` until both convolutions finish.
2. Transpose to `(B, F, C)` before the LSTM.
3. Keep the LSTM `input_size` equal to the second convolution's output
   channels.
4. Keep both final directional states when using the bidirectional LSTM.
5. Update the dense input width if `lstm_hidden` or bidirectionality changes.
6. Preserve the feature ordering contract across preprocessing, training, and
   attacks.

## 8. Factory and model-name contract

The factory is the stable entry point used by training, review, and attack
code:

```python
from src.classifiers.models import get_model

model = get_model("serial", num_features=39, num_classes=8)
```

Supported names are lowercase and exact. The current mapping is:

| Factory name | Class | Input | Output |
|---|---|---|---|
| `mlp` | `SimpleMLP` | `(B, F)` | logits `(B, K)` |
| `cnn` | `CNNOnly` | `(B, F)` | logits `(B, K)` |
| `lstm` | `LSTMOnly` | `(B, F)` or `(B, T, F)` | logits `(B, K)` |
| `serial` | `SerialCNNLSTM` | `(B, F)` | logits `(B, K)` |

The factory forwards extra keyword arguments to the selected constructor:

```python
model = get_model(
    "cnn",
    num_features=39,
    num_classes=8,
    conv_channels=(16, 32),
    kernel_size=5,
)
```

Any architecture-changing keyword arguments must be recorded with the
checkpoint. A state dictionary alone does not reliably encode all constructor
choices, especially when a model has no parameter for a choice or when a
caller reconstructs a default architecture.

## 9. Training and evaluation integration

### 9.1 CICIoT2023 baseline tasks

`baseline_experiments.py` trains each neural model for three CICIoT2023 label framings:

| Task | Labels | Number of outputs |
|---|---:|---:|
| Binary | `y_*_bin.npy` | 2 |
| Category | `y_*_cat.npy` | 8 |
| Fine-grained | `y_*.npy` | 34 |

The runner derives `num_features` from `X_train.shape[1]` and
`num_classes` from the task configuration. It uses the same model class for all
three tasks; only the final classifier width changes.

### 9.2 CICIoT2023 optimization path

The CICIoT2023 baseline runner currently uses:

```text
loss       = CrossEntropyLoss(train-derived balanced weights)
optimizer  = Adam(lr=1e-3)
scheduler  = ReduceLROnPlateau(mode="min", patience=3, factor=0.5)
batch_size = 2048
selection  = highest validation macro F1
```

It calls `model.train()` for optimization and `model.eval()` for validation and
test evaluation. The best state is selected by validation macro F1 (validation
loss breaks exact ties) and restored before test metrics are calculated. The
test split is not used for model selection.

Loss weights come from the persisted training-label class weights. Validation
and test labels never influence them. `--no-class-weights` and
`--selection-metric val_loss` remain explicit ablation options.

The runner accepts explicit input/output directories and model/task selections. The
completed binary and eight-category CICIoT2023 run used:

```bash
python -m src.classifiers.baseline_experiments \
  --processed-dir outputs/ciciot2023_fixed \
  --output-dir outputs/ciciot2023_fixed/classifier_results \
  --models all \
  --tasks binary,8class \
  --epochs 10 \
  --batch-size 2048 \
  --early-stop-patience 11 \
  --selection-metric macro_f1 \
  --device cuda
```

Checkpoints are written under `<output-dir>/models/`; JSON and text
classification reports under `<output-dir>/metrics/`; the combined run manifest,
CSV summary, and Markdown results report remain at the output root. The completed
run intentionally contains no 34-class checkpoint.

### 9.3 Tuple-safe inference

`CNNOnly`, `LSTMOnly`, and `SerialCNNLSTM` can return either logits or
`(logits, features)`. Training and review code therefore normalize the output:

```python
output = model(x_batch)
logits = output[0] if isinstance(output, tuple) else output
```

Normal classification calls return a tensor because `return_features` defaults
to `False`. New consumers should keep this tuple-safe pattern if they accept
models with feature extraction enabled.

### 9.4 CICIoT2023 post-hoc prior correction

`prior_corrected_evaluation.py` evaluates the existing binary and eight-category
checkpoints without retraining or modifying model/scaler artifacts. It computes
the sampled-training prior directly from `y_train_bin.npy` or `y_train_cat.npy`
and the natural prior directly from the complete `y_test_bin.npy` or
`y_test_cat.npy`. For class \(k\), it applies:

\[
\ell'_k = \ell_k - \log \pi_{\mathrm{train},k}
          + \log \pi_{\mathrm{natural},k}
\]

to logits immediately before prediction. Raw and adjusted confusion matrices
are accumulated over all 8,248,312 test rows in the same inference pass.
```bash
python -m src.classifiers.prior_corrected_evaluation \
  --processed-dir outputs/ciciot2023_fixed \
  --classifier-dir outputs/ciciot2023_fixed/classifier_results \
  --output-dir outputs/ciciot2023_fixed/classifier_results/prior_correction \
  --batch-size 2048 \
  --device cuda
```

The output directory contains direct-array prior counts, a side-by-side metric
CSV, raw/corrected confusion matrices for every model/head pair, a complete JSON
result, and a Markdown report. The evaluator verifies that raw metrics reproduce
the saved reports and that checkpoint hashes remain unchanged.

### 9.5 CICIoT2023 eight-class accuracy audit

`audit_ciciot2023_8class.py` is a no-training audit of the complete
eight-category evaluation path. It verifies exact fine/category/binary label
alignment, preprocessing hashes, and full-holdout counts; measures category,
fine-subtype, and per-feature distribution shift; reevaluates all checkpoints
on full validation; and decomposes full-test errors by class pair.
```bash
python -m src.classifiers.audit_ciciot2023_8class \
  --processed-dir outputs/ciciot2023_fixed \
  --classifier-dir outputs/ciciot2023_fixed/classifier_results \
  --output-dir outputs/ciciot2023_fixed/classifier_results/audit_8class \
  --feature-sample-per-category 20000 \
  --device cuda
```

The audit’s machine-readable JSON and CSV outputs are supplemented by
`ciciot2023_8class_accuracy_audit.md`. Raw-Parquet checks in that report also
show whether train-derived clipping and rounding collapsed source features in
the saved arrays.

### 9.6 CICIDS2017-DistriNet two-head experiment

#### Dataset and output contract

`cicids2017d_experiments.py` consumes the fixed 79-feature DistriNet arrays and
trains exactly two task-specific classifiers per architecture:

| Head | Label IDs | Output width |
|---|---|---:|
| Binary | `Benign=0`, `Attack=1` | 2 |
| Category | `Benign=0`, `DoS=1`, `DDoS=2`, `Recon=3`, `BruteForce=4` | 5 |

This is eight independent checkpoints, not one network with two simultaneous
output tensors. Each architecture is instantiated once with `num_classes=2` and
once with `num_classes=5`. There is no fine-grained DistriNet head.

The production split is unchanged by classifier training:

| Split | Rows |
|---|---:|
| Train | 1,456,265 |
| Validation | 312,058 |
| Test | 312,056 |

#### Architecture configurations

The DistriNet runner records every non-default constructor option inside each
checkpoint. All four models apply `asinh` to the already train-fitted scaled
features before their first learned layer. This monotonic transform bounds the
effect of zero-IQR RobustScaler columns whose observed magnitudes reached
`7.48e8` in the training audit.

| Model | DistriNet-specific configuration |
|---|---|
| `SimpleMLP` | `hidden_dims=(256,128,64)`, `input_transform="asinh"` |
| `CNNOnly` | `pool_size=8`, `input_transform="asinh"` |
| `LSTMOnly` | `feature_sequence=True`, `feature_embedding_dim=16`, `input_transform="asinh"` |
| `SerialCNNLSTM` | `input_transform="asinh"` |

`pool_size=8` prevents the CNN from discarding all feature-location identity.
Feature-sequence mode removes the LSTM's former one-timestep degeneracy. The
serial architecture remains a feature-order baseline; it is not described as
temporal traffic modeling.

#### Train-only inverse-frequency loss weights

Both heads use weighted softmax cross-entropy. The default weight for class
\(k\), computed only from the training labels, is:

\[
w_k = \frac{N}{K n_k},
\]

where \(N\) is the number of training rows, \(K\) is the number of classes, and
\(n_k\) is the training count for class \(k\). The exact default weights are:

| Head | Class | Training rows | Weight |
|---|---|---:|---:|
| Binary | Benign | 1,153,431 | 0.631275 |
| Binary | Attack | 302,834 | 2.404395 |
| Category | Benign | 1,153,431 | 0.252510 |
| Category | DoS | 120,093 | 2.425229 |
| Category | DDoS | 66,568 | 4.375270 |
| Category | Recon | 111,311 | 2.616570 |
| Category | BruteForce | 4,862 | 59.903950 |

The former effective-number configuration used `beta=0.999`. Its saturation
scale was only \(1/(1-\beta)=1,000\), below every class count, so all weights
were approximately `0.001` and the loss was effectively unweighted. That
explains the weak MLP/CNN minority recall; it was not DoS/DDoS pair confusion.
The runner retains `effective` and `none` as explicit experiment options, but
`balanced` is the default.

#### Optimization and checkpoint selection

The production run uses:

```text
seed                    = 42
epochs                  = 10
batch_size               = 2048
optimizer                = Adam
learning_rate            = 1e-3
scheduler                = ReduceLROnPlateau(mode="max", patience=1, factor=0.5)
gradient_clip_norm       = 5.0
early_stopping_patience  = 3
selection_metric         = validation macro-F1
selection_tie_break      = validation loss
device                   = CUDA
```

The test split is evaluated only after validation-based checkpoint selection.
Each saved checkpoint is reloaded into its source architecture and exercised
on held-out rows before the run is accepted.

#### Current corrected-split run

Aggregate held-out results:

| Head | Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| Binary | `SerialCNNLSTM` | 99.666% | 99.667% | 99.494% | 99.666% |
| Binary | `CNNOnly` | 98.472% | 98.904% | 97.736% | 98.490% |
| Binary | `LSTMOnly` | 98.423% | 98.812% | 97.662% | 98.441% |
| Binary | `SimpleMLP` | 98.418% | 98.768% | 97.652% | 98.435% |
| Category | `SimpleMLP` | 98.448% | 99.031% | 97.725% | 98.504% |
| Category | `SerialCNNLSTM` | 98.426% | 98.907% | 97.622% | 98.483% |
| Category | `CNNOnly` | 98.433% | 98.914% | 97.528% | 98.489% |
| Category | `LSTMOnly` | 98.387% | 98.696% | 97.241% | 98.438% |

The corrected audit shows that direct DoS↔DDoS confusion is not the remaining
problem. Across all four category models, only one DoS row is predicted as
DDoS and no DDoS row is predicted as DoS. Test recall ranges from 98.306% to
99.417% for DoS and from 99.846% to 99.853% for DDoS.

The strongest category checkpoint, `SimpleMLP`, has:

| True \ Predicted | Benign | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|---:|
| Benign | 242,593 | 53 | 2 | 4,515 | 1 |
| DoS | 150 | 25,583 | 0 | 0 | 0 |
| DDoS | 12 | 0 | 14,243 | 10 | 0 |
| Recon | 64 | 0 | 17 | 23,771 | 0 |
| BruteForce | 18 | 0 | 0 | 2 | 1,022 |

The remaining DoS weakness is subtype-specific: `DoS Slowhttptest` recall is
32.57–57.85% depending on architecture, with most errors going to Benign.
`DoS GoldenEye`, which was absent from the former training split, now has
99.56–100% recall. Exact normalized rates are in
`outputs/cicids2017distrinet/dos_ddos_error_audit.csv`.

The complete per-model classification reports, numeric confusion matrices,
plots, predictions, histories, checkpoints, and class weights are under
`outputs/cicids2017distrinet/`. The consolidated human-readable report is
`outputs/cicids2017distrinet/cicids2017_classifier_results.md`.

## 10. Checkpoints and reconstruction

The CICIoT2023 baseline runner saves raw neural state dictionaries. The
CICIDS2017-DistriNet runner saves a package containing:

```text
state_dict, model_type, model_kwargs, num_features, num_classes
```

Both use `models/{model_type}_{task_name}.pt`. The packaged format is required
because input transforms, CNN pool bins, and LSTM feature-sequence mode cannot
be reconstructed safely from a filename. Review and attack loaders accept both
formats; the attack loader retains hidden-width inference for legacy raw MLP
state dictionaries.

Checkpoint loading requirements:

1. Use the same `num_features` as training.
2. Use the same `num_classes` and label ID mapping.
3. Use the saved architecture keyword arguments when present.
4. For legacy baseline MLP checkpoints, use `hidden_dims=(256, 128, 64)`.
5. Do not change feature order or preprocessing between training and loading.
6. Treat changed layer widths, channels, pool bins, feature-token settings,
   kernel configuration, or recurrent dimensions as a new checkpoint format.

The corrected bidirectional state selection changes the forward computation but
not parameter names or tensor shapes. Existing checkpoints with matching
architectures remain loadable; their predictions are recomputed using the
correct final-state representation.

## 11. Adversarial-attack integration

The attack code uses the same factory and checkpoint architecture. Attacks
operate on classifier logits and require gradients for PGD/CW-style methods.
The model must remain in evaluation mode so dropout does not add randomness to
the victim function.

A safe model-facing pattern is:

```python
model.eval()
logits = model(x_adv)
if isinstance(logits, tuple):
    logits = logits[0]
```

The attack implementation owns gradient management, perturbation constraints,
masking, and domain validation. The classifier must not clamp features to
`[0, 1]`, round integer features, or reimpose protocol rules internally;
putting those transformations inside a classifier would change the victim
function and interfere with the intended attack comparison.

For attack results, distinguish:

- raw classifier evasion: the predicted class changes or leaves the source
  class as defined by the attack;
- domain validity: evaluated by the validator outside the model;
- realism/in-distribution status: evaluated by the designated realism gate;
- joint attack success: the documented conjunction of evasion, validity, and
  realism gates.

## 12. Common implementation mistakes

### Wrong feature order

A checkpoint trained with the schema's 39-column order cannot consume columns
in a different order. This is especially damaging for CNN and Serial CNN-LSTM,
which explicitly use neighboring positions.

### Applying softmax before the loss

`CrossEntropyLoss` expects logits and internally applies the appropriate
log-sum-exp operation. Applying softmax first reduces numerical stability and
changes the intended loss input.

### Using training mode for evaluation

Dropout is active in training mode. Always call `eval()` before deterministic
validation, test scoring, checkpoint review, or attack victim scoring.

### Taking the last BiLSTM output as both final states

`output[:, -1, :]` is not the correct concatenation of final forward and
backward states. Use the final-layer entries in `h_n` as implemented.

### Treating a single tabular row as temporal evidence

`LSTMOnly` accepts a 2D vector by wrapping it as a one-step sequence. That
provides API compatibility, not temporal information. Temporal claims require
multiple timesteps with an explicit data-generation and split design.

### Forgetting the CNN-to-LSTM transpose

Conv1d emits `(B, channels, length)`, while `batch_first=True` LSTM consumes
`(B, length, input_size)`. Omitting or misplacing the transpose swaps semantic
axes and makes the LSTM consume the wrong dimension.

### Changing architecture without updating loaders

A changed constructor default is not enough. Update baseline training,
review reconstruction, attack loading, checkpoint metadata, and any saved
experiment configuration together.

### Moving validity logic into the model

Classifiers should map valid preprocessed tensors to logits. Protocol rules,
feature masks, integer/binary restoration, raw-space constraints, and
in-distribution gates are separate concerns and must remain outside the
classifier implementation.

## 13. Extension and maintenance rules

When adding or modifying a classifier:

1. Start from the schema and preserve its feature order.
2. State the exact accepted tensor shape in the class docstring.
3. Validate rank and feature width at the public forward boundary.
4. Return logits of shape `(B, K)` with no terminal softmax.
5. Keep training and inference reconstruction paths synchronized.
6. Preserve device and dtype behavior; do not create CPU tensors inside
   `forward` for model computation.
7. Define how `return_features` behaves, including whether dropout is active.
8. If adding sequences, define timestep meaning, sequence lengths, padding, and
   leakage-safe splitting before claiming temporal modeling.
9. Add a behavioral regression test for every changed tensor contract or state
   transition.
10. Run a forward-shape test, a loss/backward smoke test, and a checkpoint load
    test before using the model in attacks or reported experiments.

## 14. Minimal usage example

```python
import torch
from src.classifiers.models import get_model

BATCH = 32
FEATURES = 39
CLASSES = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model("serial", FEATURES, CLASSES).to(device)
model.train()

x = torch.randn(BATCH, FEATURES, device=device, dtype=torch.float32)
y = torch.randint(0, CLASSES, (BATCH,), device=device, dtype=torch.long)

logits = model(x)
assert logits.shape == (BATCH, CLASSES)
loss = torch.nn.functional.cross_entropy(logits, y)
loss.backward()

model.eval()
with torch.no_grad():
    logits = model(x)
    predictions = logits.argmax(dim=1)
    probabilities = logits.softmax(dim=1)
```

For sequence-enabled `LSTMOnly` use:

```python
sequence_model = get_model("lstm", FEATURES, CLASSES).to(device)
x_sequence = torch.randn(BATCH, 10, FEATURES, device=device)
logits = sequence_model(x_sequence)  # (BATCH, CLASSES)
```

The sequence example is shape-valid, but it is only scientifically meaningful
if each of the 10 timesteps represents a real, consistently defined temporal
observation.

## 15. Verification checklist

Before accepting a model change, verify:

- [ ] Default tabular input produces `(B, K)` logits.
- [ ] Logits are finite and work with `CrossEntropyLoss`.
- [ ] A backward pass produces gradients for trainable parameters.
- [ ] Invalid rank and feature width fail at the model boundary.
- [ ] `eval()` makes dropout behavior deterministic.
- [ ] Any bidirectional recurrent representation uses final states from both
      directions.
- [ ] CNN output length remains aligned with the documented feature axis.
- [ ] `return_features` shape and dropout semantics are documented.
- [ ] Training and review reconstruct the same architecture.
- [ ] Existing checkpoints either load unchanged or receive an explicit
      migration/version update.
- [ ] Attack code consumes logits and leaves validity/realism gates external.
- [ ] The relevant model-specific regression and smoke checks pass.
