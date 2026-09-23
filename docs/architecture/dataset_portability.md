# Adding a New Flow-Based NIDS Dataset

The platform is generic: the VAE, typed decoder, constraint engine, residual head,
Stage-B trainer, and validators depend only on a `FeatureManifest` + `DatasetAdapter`.
Supporting a new dataset (e.g. CICIDS2017-DistriNet) means writing **one adapter file**.

## What you write

Create `src/datasets/<name>.py` implementing `DatasetAdapter`:

```python
class MyAdapter(DatasetAdapter):
    name = "my_dataset"

    def feature_manifest(self) -> FeatureManifest:
        specs = []
        for i, col in enumerate(feature_order):           # frozen order = contract
            specs.append(FeatureSpec(
                name=col, model_index=i,
                value_type=...,      # from EXTRACTOR semantics, NOT the column name
                semantic_type=...,   # role tag for lookups
                lower=..., upper=..., # hard semantic bounds (not train min/max)
                scaling="robust",    # must match the fitted scaler family
            ))
        return FeatureManifest(specs, dataset_name=self.name)

    def class_mapping(self) -> ClassMapping:
        return ClassMapping.from_names(coarse_class_names, fine_to_coarse=...)

    def feature_transform(self) -> FeatureTransform:
        # wrap an already-train-fit scaler ...
        return FeatureTransform.from_sklearn_scaler(scaler, self.feature_manifest())
        # ... or fit fresh on TRAIN only:
        # return FeatureTransform(self.feature_manifest()).fit(X_train_raw)

    def load_split(self, split) -> Split:
        x = ...  # scaled features (N, F)
        y = ...  # coarse class ids
        return Split(name=split, x=x, y=y)
```

Then register it in `datasets/__init__.py:get_adapter` (one branch), and *optionally*
drop `constraints/<name>/mined.json` for Layer-2 rules.

## `value_type` cheat-sheet (choose by semantics, not name)

| value_type | meaning | decoder head | Layer-0 domain |
|---|---|---|---|
| `real` | unrestricted | identity | none |
| `positive_continuous` | ≥ 0, no upper | softplus | ≥ 0 |
| `bounded_continuous` | [l, u] | `l+(u−l)·sigmoid` | [l, u] |
| `probability` | [0,1] aggregate/frequency | sigmoid | [0, 1] |
| `integer_count` | true non-negative count | softplus (+STE round at eval) | ≥ 0 |
| `binary` | true {0,1} | sigmoid (+STE) | [0, 1] |
| `categorical` | finite code (cardinality) | logits (later phase) | — |
| `derived` | exact function of parents | recomputed by Layer 0 | — |

Pitfall: averaged flag/service/protocol indicators are usually `probability`
(window means in [0,1]), **not** `binary`; averaged protocol codes are
`bounded_continuous`, **not** `categorical`.

## What you must NOT touch

- `src/vae/model.py`, `src/vae/decoder.py`, `src/vae/losses.py`
- `src/constraints/*` (engine/layers/registry)
- `src/attack/residual_head.py`, `src/attack/train_attack_head.py`
- validators / metrics

If a change to any of those seems necessary to add a dataset, that is a leak of
dataset specifics into generic code — fix the manifest/adapter instead.

## Layer-2 rule file format (`constraints/<name>/mined.json`)

```json
{
  "schema_version": "1.0",
  "dataset": "my_dataset",
  "constraints": [
    {"type": "MonotoneNondecreasing", "name": "order", "params": {"features": ["Min","AVG","Max"]}},
    {"type": "ProductEquality", "name": "tot", "params": {"target": "Tot", "factors": ["N","AVG"], "rtol": 0.05}}
  ]
}
```
Types resolve through `constraints.registry.CONSTRAINT_REGISTRY`; a mining script can
overwrite this file in place with the same schema.

## Verify

```
PYTHONPATH=src python -m pytest src/datasets/tests -q -p no:faulthandler
```
Add adapter-specific tests mirroring `test_manifest.py` (manifest census, class
mapping, transform round-trip, split width/label contract).
```
