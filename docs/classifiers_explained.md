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
conv_channels = (32, 64)
kernel_size  = 3
fc_dim        = 64
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
(B, 32, F)
  -> Conv1d(32, 64, kernel_size, padding="same")
(B, 64, F)
  -> ReLU
(B, 64, F)
  -> AdaptiveMaxPool1d(1)
(B, 64, 1)
  -> squeeze last dimension
(B, 64)
  -> Linear(64, fc_dim)
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

The adaptive pool reduces every feature-position sequence to one maximum per
channel. Consequently, the dense head always receives 64 values with default
channels, regardless of feature count.

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

`LSTMOnly` is a sequence-compatible recurrent baseline. It has two input
modes:

1. **Current tabular baseline:** `(B, F)` is converted to `(B, 1, F)`, so the
   complete feature vector is one recurrent timestep.
2. **True sequence mode:** `(B, T, F)` is passed directly, where `T` is a
   sequence length and `F` is the feature width at every timestep.

For the current CICIoT2023 baseline arrays, `T=1`. Thus the model is not
learning temporal dependencies between multiple observations in the baseline
experiment. It is a sequence-compatible architecture that can accept future
windows; it should not be described as exploiting temporal context unless
actual `T > 1` sequences are supplied.

### 6.2 Architecture and defaults

Constructor defaults:

```text
hidden_dim   = 64
num_layers   = 1
bidirectional = True
fc_dim       = 64
dropout      = 0.3
```

For 2D input, the shape path is:

```text
(B, F) -> unsqueeze(1) -> (B, 1, F)
```

The recurrent layer is:

```text
LSTM(
    input_size=F,
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
loss       = CrossEntropyLoss()
optimizer  = Adam(lr=1e-3)
scheduler  = ReduceLROnPlateau(mode="min", patience=3, factor=0.5)
batch_size = 2048
```

It calls `model.train()` for optimization and `model.eval()` for validation and
test evaluation. The best state is selected by validation loss and restored
before test metrics are calculated. The test split is not used for model
selection.

The training script intentionally does not apply the supplied class weights;
its recorded decision is that the training sample is already balanced. This is
a training-policy decision, not a behavior implemented by the model classes.

The runner accepts explicit input/output directories and model/task selections. The
completed binary and eight-category CICIoT2023 run used:

```bash
python -m src.classifiers.baseline_experiments \
  --processed-dir outputs/ciciot2023 \
  --output-dir data/classifier_results/ciciot2023 \
  --models all \
  --tasks binary,8class \
  --epochs 10 \
  --batch-size 2048 \
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

### 9.4 CICIDS2017-DistriNet two-head experiment

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
| Train | 1,456,264 |
| Validation | 312,058 |
| Test | 312,057 |

#### Architecture configurations

All four architectures use the implementations in `src/classifiers/models.py`. The
DistriNet runner changes only the input and output widths:

| Model | DistriNet-specific configuration | Binary parameters | Category parameters |
|---|---|---:|---:|
| `SimpleMLP` | `num_features=79`, `hidden_dims=(256,128,64)` | 61,762 | 61,957 |
| `CNNOnly` | `num_features=79`, constructor defaults | 10,626 | 10,821 |
| `LSTMOnly` | `num_features=79`, constructor defaults | 82,626 | 82,821 |
| `SerialCNNLSTM` | `num_features=79`, constructor defaults | 81,282 | 81,477 |

The output-layer parameter difference is caused solely by changing the final
width from 2 to 5. Convolutional, recurrent, dense-feature, and dropout
settings are otherwise identical between heads.

#### Effective-number class-balanced loss

Both heads use weighted softmax cross-entropy. The weight for class \(k\) is
computed only from its training count \(n_k\):

\[
w_k = \frac{1-\beta}{1-\beta^{n_k}}, \qquad \beta=0.999.
\]

This is the class-balanced weighting proposed by Cui et al. in
[*Class-Balanced Loss Based on Effective Number of Samples* (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html).
The runner uses the formula directly, without an extra sum-to-class-count
normalization, and passes the resulting vector to:

```python
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
```

The production training counts and float32 weights are:

| Head | Class | Training rows | Weight |
|---|---|---:|---:|
| Binary | Benign | 1,153,431 | 0.0010000000474974513 |
| Binary | Attack | 302,833 | 0.0010000000474974513 |
| Category | Benign | 1,153,431 | 0.0010000000474974513 |
| Category | DoS | 120,091 | 0.0010000000474974513 |
| Category | DDoS | 66,568 | 0.0010000000474974513 |
| Category | Recon | 111,311 | 0.0010000000474974513 |
| Category | BruteForce | 4,863 | 0.0010077683255076408 |

Why the weights are nearly equal: the effective sample count is

\[
E_n = \frac{1-\beta^n}{1-\beta},
\]

whose upper limit is \(1/(1-\beta)\). At \(\beta=0.999\), that limit is
1,000. Every DistriNet class has more than 4,800 training rows, so every class
is already near the saturation region. BruteForce therefore receives only
about \(1.0078\times\) the weight of the other categories even though its raw
count is much smaller.

This is correct for the configured beta; it is not equivalent to the former
inverse-frequency weighting. Larger beta values shift the saturation scale:

| Beta | Approximate saturation scale \(1/(1-\beta)\) |
|---:|---:|
| 0.999 | 1,000 |
| 0.9999 | 10,000 |
| 0.99999 | 100,000 |
| 0.999999 | 1,000,000 |

Beta is a loss hyperparameter. Any alternative must be selected with training
and validation data only; test performance must not choose it.

#### Optimization and checkpoint selection

The production run uses:

```text
seed                    = 42
epochs                  = 5
batch_size               = 2048
optimizer                = Adam
learning_rate            = 1e-3
scheduler                = ReduceLROnPlateau(mode="max", patience=1, factor=0.5)
gradient_clip_norm       = 5.0
early_stopping_patience  = 2
selection_metric         = validation macro-F1
selection_tie_break      = validation loss
device                   = CUDA
```

The test split is evaluated only after validation-based checkpoint selection.
Each saved checkpoint is reloaded into its source architecture and exercised
on held-out rows before the run is accepted.

#### Current effective-number run

Aggregate held-out results:

| Head | Model | Accuracy | Balanced accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|
| Binary | `SerialCNNLSTM` | 98.188% | 97.692% | 97.274% | 98.196% |
| Binary | `LSTMOnly` | 92.671% | 89.165% | 88.939% | 92.692% |
| Binary | `SimpleMLP` | 89.284% | 76.765% | 80.892% | 88.288% |
| Binary | `CNNOnly` | 89.251% | 76.610% | 80.779% | 88.233% |
| Category | `SerialCNNLSTM` | 98.773% | 97.407% | 97.832% | 98.772% |
| Category | `LSTMOnly` | 91.864% | 83.587% | 84.438% | 91.677% |
| Category | `CNNOnly` | 89.284% | 70.502% | 74.732% | 86.989% |
| Category | `SimpleMLP` | 89.260% | 70.604% | 74.624% | 86.944% |

Per-class results for the strongest model, `SerialCNNLSTM`:

| Head | Class | Support | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| Binary | Benign | 247,164 | 0.991653 | 0.985423 | 0.988528 |
| Binary | Attack | 64,893 | 0.945776 | 0.968410 | 0.956959 |
| Category | Benign | 247,164 | 0.991900 | 0.992839 | 0.992369 |
| Category | DoS | 25,734 | 0.999121 | 0.928033 | 0.962266 |
| Category | DDoS | 14,265 | 0.996154 | 0.998528 | 0.997339 |
| Category | Recon | 23,852 | 0.931343 | 0.994130 | 0.961713 |
| Category | BruteForce | 1,042 | 1.000000 | 0.956814 | 0.977930 |

`SerialCNNLSTM` test confusion matrices, with true classes in rows and predicted
classes in columns:

| Binary | Benign | Attack |
|---|---:|---:|
| Benign | 243,561 | 3,603 |
| Attack | 2,050 | 62,843 |

| Category | Benign | DoS | DDoS | Recon | BruteForce |
|---|---:|---:|---:|---:|---:|
| Benign | 245,394 | 21 | 1 | 1,748 | 0 |
| DoS | 1,852 | 23,882 | 0 | 0 | 0 |
| DDoS | 21 | 0 | 14,244 | 0 | 0 |
| Recon | 86 | 0 | 54 | 23,712 | 0 |
| BruteForce | 45 | 0 | 0 | 0 | 997 |

The complete per-model classification reports, numeric confusion matrices,
plots, predictions, histories, checkpoints, and effective-number weights are
under `outputs/cicids2017distrinet/`. The consolidated human-readable report is
`outputs/cicids2017distrinet/cicids2017_classifier_results.md`.

## 10. Checkpoints and reconstruction

The baseline runner saves neural checkpoints as state dictionaries:

```text
models/{model_type}_{task_name}.pt
```

Examples include `mlp_binary.pt`, `cnn_8class.pt`, and `serial_34class.pt`.
Review code reconstructs the architecture, loads the state dictionary, moves
the model to the selected device, and calls `eval()`.

Checkpoint loading requirements:

1. Use the same `num_features` as training.
2. Use the same `num_classes` and label ID mapping.
3. Use the same architecture keyword arguments.
4. For baseline MLP checkpoints, use `hidden_dims=(256, 128, 64)`.
5. Do not change feature order or preprocessing between training and loading.
6. Treat changed layer widths, channel counts, kernel configuration, recurrent
   hidden size, layer count, or directionality as a new checkpoint format.

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
