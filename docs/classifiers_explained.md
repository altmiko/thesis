# Active classifier models

## Roster

The active trained neural victim roster is:

- `SimpleMLP` (`mlp`);
- `CNNOnly` (`cnn`).

`FTTransformer` is present in the model factory for future integration, but it is not yet part of
the trained attack-victim roster or current reported experiments. Classical RF/XGBoost baselines
remain separate.

The active implementations are in `src/classifiers/models.py`; FT-Transformer components are in
`src/classifiers/ft_transformer.py`.

## Shared contract

```python
model = get_model(
    model_type="mlp",  # "mlp", "cnn", or future "ft_transformer"
    num_features=79,
    num_classes=5,
)
logits = model(x)  # x shape: [batch, features]
```

All active models:

- accept a two-dimensional tabular tensor;
- return raw class logits;
- preserve gradients from logits to input features;
- support the optional monotonic `asinh` input transform used by trained checkpoints.

## SimpleMLP

`SimpleMLP` applies fully connected blocks to the complete feature vector. The CICIDS2017
checkpoint uses the training configuration recorded in
`outputs/cicids2017distrinet/classifier_run_manifest.json`.

## CNNOnly

`CNNOnly` treats the ordered feature vector as a one-dimensional signal, applies two 1D
convolutions, adaptive pooling, and a dense classifier head. Feature order is therefore part of
the checkpoint contract.

## FT-Transformer status

`FTTransformer` tokenizes each numerical feature separately, appends a learned classification
token, applies self-attention blocks, and predicts from the classification token. Its operations
are differentiable with respect to input features. It must not be included in attack tables until
a matching trained checkpoint, manifest entry, and full evaluation exist.

## Loading CICIDS2017 victims

`src/classifiers/cicids2017d_victims.py::load_category_victim` validates:

- preprocessing-manifest hash;
- feature width;
- category class order;
- checkpoint model type;
- checkpoint class count.

Loaded parameters are frozen while gradients to model inputs remain available for white-box
attacks.
