# FT-Transformer — Attack-Readiness Report

Confirms the trained FT-Transformer **category** victim is a valid gradient-attack target
under the EXISTING attack building blocks (no attack code was modified or weakened).

Harness: `scripts/ft_transformer_attack_readiness.py`. It loads the victim through the
canonical `load_category_victim` path (schema + preprocessing-sha256 + class-order + feature
count guards), selects clean-correct **malicious** flows, and exercises the real attack
primitives.

## Exact commands
```bash
# (run inside the thesis conda env: python 3.11, torch 2.5.1+cu121)
# train (already done -> outputs/cicids2017distrinet_ft/)
python -m src.classifiers.cicids2017d_experiments \
    --models ft_transformer --tasks all --epochs 10 --patience 3 \
    --batch-size 2048 --ft-learning-rate 1e-4 --ft-weight-decay 1e-5 \
    --class-weighting balanced --seed 42 --device cuda \
    --output-dir outputs/cicids2017distrinet_ft

# attack-readiness smoke (input-grad + PGD + C&W + targeted-Benign + PrimAttack grad)
PYTHONPATH=src python scripts/ft_transformer_attack_readiness.py \
    --victim outputs/cicids2017distrinet_ft/models/ft_transformer_category.pt \
    --device cuda --n 256
```

## Results (device cuda, n = 256 clean-correct malicious flows)
| Check | Result |
|---|---|
| model returns raw logits `[B, 5]` | ✅ true |
| logits finite | ✅ true |
| input gradient shape `[B, 79]` | ✅ true |
| input gradient finite | ✅ true |
| input gradient non-zero | ✅ true |
| feature ordering == schema (`preprocessing_manifest.modelling_feature_names`) | ✅ true |
| feature ordering == checkpoint metadata `feature_names` | ✅ true |
| scaler applied exactly once (attacks run in RobustScaler space; PrimAttack `(φ(raw)-center)/scale`) | ✅ true |
| Benign target index resolves to class **0** (`class_mapping.name_to_id["Benign"]`) | ✅ true |
| **feature-space PGD** smoke (ε=0.5, α=0.1, 20 steps, untargeted) | ✅ ran — **256/256** flipped |
| **feature-space C&W** smoke (λ=1.0, κ=0, 50 iters) | ✅ ran — **250/256** flipped |
| targeted-Benign objective differentiable (`dL/dδ` finite) | ✅ true |
| **PrimAttack** `dL/dp` finite / non-zero, `dL/dα` finite (64 DoS rows) | ✅ true |
| **ALL_READINESS_CHECKS_PASS** | ✅ **true** |

Notes:
- PGD/C&W here are **untargeted** smoke tests using the repo's `input_pgd_attack` /
  `input_cw_attack` with their default feature-space semantics; they confirm gradients flow
  and predictions move — not a validity-gated ASR. The full targeted-Benign ASR suite
  (validity + in-distribution gates) is the separate thesis evaluation and is unchanged.
- The PrimAttack check reproduces the existing gradient contract
  (`test_primattack_transformation.py::test_autograd_matches_finite_difference_through_victim`)
  with the FT-Transformer as victim: `p, α` (requires_grad) →
  `adv = CICIDS2017PrimitiveModel.generate(raw, {"p","α"})` → `(adv-center)/scale` →
  `victim` → CE(Benign) → `autograd.grad` finite and non-zero. Do **not** run the full
  expensive PrimAttack experiment before this smoke passes; it passes.
- The forward pass contains no `detach`/`argmax`/rounding/numpy/`no_grad`/CPU round-trip,
  so `d(logit)/d(input)` exists for normal numerical inputs.

## Unit-level gradient tests
`src/classifiers/tests/test_ft_transformer.py` (24 pass incl. GPU): input gradients
finite/non-zero/no-NaN-Inf, autograd-vs-finite-difference on mutable features, ReGLU dims,
raw-logit contract, save/load determinism, invalid-schema rejection, batch-size-1.
