# Active classifier models

The active trained neural victims are `SimpleMLP` (`mlp`) and `CNNOnly` (`cnn`). Their canonical
implementation and factory are in `src/classifiers/models.py`.

`FTTransformer` code is available in `src/classifiers/ft_transformer.py` for future integration,
but no FT-Transformer result belongs in active attack tables until a matching checkpoint and
full evaluation are produced.

## Contract

All active victim models consume a tabular tensor `[batch, features]`, return raw class logits,
and preserve gradients to input features. CICIDS2017 category victims use 79 features and five
classes in the fixed order `Benign, DoS, DDoS, Recon, BruteForce`.

## Architectures

- **SimpleMLP:** dense blocks over the complete scaled feature vector.
- **CNNOnly:** two 1D convolutions over the frozen feature order, adaptive pooling, and a dense
  classification head.

## Checkpoint loading

`src/classifiers/cicids2017d_victims.py::load_category_victim` verifies preprocessing identity,
feature width, class order, model type, and class count before loading. Victim parameters are
frozen during attack optimization; input gradients remain enabled.

See `docs/classifiers_explained.md` for the current detailed roster contract.
