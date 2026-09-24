# FT-Transformer integration status

`src/classifiers/ft_transformer.py` contains a differentiable numerical-feature tokenizer,
classification token, self-attention blocks, ReGLU feed-forward blocks, and classification head.
`src/classifiers/models.py::get_model` recognizes `ft_transformer`.

## Current boundary

The active trained attack victims remain MLP and CNN. FT-Transformer must not appear in active
attack result tables until all of the following exist:

1. a checkpoint trained on the same frozen feature order and train-only scaler;
2. matching checkpoint metadata and classifier-run manifest entry;
3. loader identity checks;
4. clean test metrics;
5. white-box gradient verification;
6. the same attack, validity, primitive-feasibility, semantic, and provenance evaluation used by
   the active victims.

## Interface contract

The model must:

- accept `[batch, features]` tensors;
- return raw `[batch, classes]` logits;
- preserve gradients to input features;
- use the existing split and train-fitted preprocessing artifacts;
- expose constructor/checkpoint kwargs through the model factory;
- avoid test-set tuning.

Until those acceptance criteria are satisfied, FT-Transformer is future work rather than an
active experimental result.
