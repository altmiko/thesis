"""Behavioral + adversarial-readiness tests for the FT-Transformer victim.

Covers task §23/§27: tokenizer identity/shapes, CLS expansion, block/output dims,
ReGLU gating, raw-logit contract, backward pass, input gradients (finite/non-zero/no
NaN-Inf), autograd vs finite-difference, CPU/GPU execution, save/load + deterministic
reload, invalid-schema rejection, batch-size-1, and multiclass loss compatibility.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from src.classifiers.ft_transformer import (
    CLSToken,
    FTTransformer,
    NumericalFeatureTokenizer,
    ReGLU,
    default_ft_transformer_kwargs,
)
from src.classifiers.models import get_model


def _small_model(num_features: int = 7, num_classes: int = 4, **overrides) -> FTTransformer:
    kwargs = dict(
        n_blocks=2, d_token=32, attention_n_heads=4,
        attention_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.0,
    )
    kwargs.update(overrides)
    model = FTTransformer(num_features, num_classes, **kwargs)
    model.eval()
    return model


# --- 1/2. tokenizer shapes + per-feature identity -----------------------------
def test_tokenizer_maps_each_scalar_to_d_token_vector():
    tok = NumericalFeatureTokenizer(num_features=7, d_token=32)
    x = torch.randn(5, 7)
    out = tok(x)
    assert out.shape == (5, 7, 32)
    # Exact formula T_j = b_j + x_j * W_j.
    expected = x.unsqueeze(-1) * tok.weight + tok.bias
    assert torch.allclose(out, expected)


def test_tokenizer_features_have_independent_parameters():
    tok = NumericalFeatureTokenizer(num_features=7, d_token=32)
    x = torch.zeros(1, 7)
    base = tok(x)
    # Bias-only at x=0 -> token j equals b_j; rows must differ (independent params).
    assert not torch.allclose(base[0, 0], base[0, 1])
    # Perturbing feature 3 changes ONLY token row 3.
    x2 = x.clone()
    x2[0, 3] = 5.0
    out2 = tok(x2)
    changed = ~torch.isclose(out2[0], base[0]).all(dim=1)
    assert changed[3] and changed.sum() == 1


# --- 3. CLS token batch expansion ---------------------------------------------
def test_cls_token_appends_one_token_per_batch_row():
    cls = CLSToken(d_token=32)
    tokens = torch.randn(6, 7, 32)
    out = cls(tokens)
    assert out.shape == (6, 8, 32)
    # CLS is the last token and identical across the batch.
    assert torch.allclose(out[:, -1], cls.weight.expand(6, 32))


# --- 4. transformer block / sequence dims -------------------------------------
def test_block_preserves_sequence_and_token_dims():
    model = _small_model()
    tokens = torch.randn(3, 8, model.d_token)  # F+1 tokens
    out, _ = model.blocks[0](tokens)
    assert out.shape == tokens.shape


def test_first_block_skips_pre_attention_norm():
    model = _small_model()
    assert isinstance(model.blocks[0].attn_norm, torch.nn.Identity)
    assert isinstance(model.blocks[1].attn_norm, torch.nn.LayerNorm)


# --- ReGLU dimensionality -----------------------------------------------------
def test_reglu_halves_last_dim_and_gates():
    reglu = ReGLU()
    a = torch.randn(4, 10)
    b = torch.randn(4, 10)
    out = reglu(torch.cat([a, b], dim=-1))
    assert out.shape == (4, 10)
    assert torch.allclose(out, a * F.relu(b))
    with pytest.raises(ValueError):
        reglu(torch.randn(4, 7))


# --- 5/14. classification output contract -------------------------------------
def test_forward_returns_raw_logits_shape_and_finite():
    model = _small_model(num_features=7, num_classes=4)
    logits = model(torch.randn(5, 7))
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (5, 4)
    assert torch.isfinite(logits).all()


def test_registry_builds_ft_transformer_with_schema_defaults():
    model = get_model(
        "ft_transformer", num_features=79, num_classes=5, input_transform="asinh"
    )
    assert isinstance(model, FTTransformer)
    assert model.d_token == 192 and model.n_blocks == 3 and model.attention_n_heads == 8
    logits = model(torch.randn(3, 79))
    assert logits.shape == (3, 5) and torch.isfinite(logits).all()


# --- 6/16. backward pass + multiclass loss ------------------------------------
def test_training_backward_populates_parameter_gradients():
    model = _small_model()
    model.train()
    logits = model(torch.randn(8, 7))
    loss = F.cross_entropy(logits, torch.randint(0, 4, (8,)))
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any((g.abs() > 0).any() for g in grads)


# --- 7. input gradients (adversarial readiness) -------------------------------
def test_input_gradients_exist_finite_nonzero():
    model = _small_model()
    x = torch.randn(6, 7, requires_grad=True)
    logits = model(x)
    loss = F.cross_entropy(logits, torch.randint(0, 4, (6,)))
    grad = torch.autograd.grad(loss, x)[0]
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all()
    assert (grad != 0).any()


def test_autograd_matches_finite_difference_on_mutable_features():
    torch.manual_seed(0)
    model = _small_model(input_transform=None)
    x = torch.randn(1, 7, dtype=torch.float64)
    model = model.double()

    def scalar(inp: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(model(inp), torch.tensor([2]))

    xg = x.clone().requires_grad_(True)
    analytic = torch.autograd.grad(scalar(xg), xg)[0][0]
    eps = 1e-5
    for j in (0, 3, 6):
        xp, xm = x.clone(), x.clone()
        xp[0, j] += eps
        xm[0, j] -= eps
        numeric = (scalar(xp) - scalar(xm)) / (2 * eps)
        assert float(analytic[j]) == pytest.approx(float(numeric), rel=1e-3, abs=1e-4)


# --- 8/9. CPU / GPU execution -------------------------------------------------
def test_cpu_execution():
    model = _small_model().to("cpu")
    assert model(torch.randn(2, 7).to("cpu")).shape == (2, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_execution():
    model = _small_model().to("cuda")
    x = torch.randn(2, 7, device="cuda", requires_grad=True)
    logits = model(x)
    grad = torch.autograd.grad(logits.sum(), x)[0]
    assert logits.is_cuda and torch.isfinite(grad).all()


# --- 10/11. save / load consistency + deterministic reload --------------------
def test_state_dict_roundtrip_is_deterministic():
    model = _small_model(num_features=79, num_classes=5)
    x = torch.randn(4, 79)
    with torch.no_grad():
        before = model(x)
    state = copy.deepcopy(model.state_dict())
    clone = _small_model(num_features=79, num_classes=5)
    clone.load_state_dict(state)
    clone.eval()
    with torch.no_grad():
        after = clone(x)
        again = clone(x)
    assert torch.allclose(before, after, atol=1e-6)
    assert torch.equal(after, again)  # eval inference is deterministic


# --- 12/13. invalid feature-count / schema mismatch rejection -----------------
def test_wrong_feature_count_is_rejected():
    model = _small_model(num_features=7)
    with pytest.raises(ValueError):
        model(torch.randn(3, 9))
    with pytest.raises(ValueError):
        model(torch.randn(3))  # not 2-D


def test_schema_mismatch_state_dict_load_fails():
    src = _small_model(num_features=79, num_classes=5)
    dst = _small_model(num_features=40, num_classes=5)
    with pytest.raises(RuntimeError):
        dst.load_state_dict(src.state_dict())


# --- 15. batch size 1 ---------------------------------------------------------
def test_batch_size_one_inference():
    model = _small_model(num_features=79, num_classes=5)
    logits = model(torch.randn(1, 79))
    assert logits.shape == (1, 5) and torch.isfinite(logits).all()


# --- optimizer parameter groups (task §13) ------------------------------------
def test_optimization_param_groups_partition_no_decay():
    model = get_model("ft_transformer", num_features=79, num_classes=5)
    groups = model.optimization_param_groups(weight_decay=1e-5)
    assert [g["weight_decay"] for g in groups] == [1e-5, 0.0]
    # Tokenizer weight/bias and CLS token must be in the no-decay group.
    no_decay_ids = {id(p) for p in groups[1]["params"]}
    assert id(model.tokenizer.weight) in no_decay_ids
    assert id(model.tokenizer.bias) in no_decay_ids
    assert id(model.cls_token.weight) in no_decay_ids
    total = sum(len(g["params"]) for g in groups)
    assert total == len([p for p in model.parameters() if p.requires_grad])


def test_default_kwargs_match_reference_defaults():
    kw = default_ft_transformer_kwargs()
    assert kw["n_blocks"] == 3 and kw["d_token"] == 192
    assert kw["attention_n_heads"] == 8 and kw["attention_dropout"] == 0.2
    assert kw["ffn_dropout"] == 0.1 and kw["residual_dropout"] == 0.0
    assert abs(kw["ffn_multiplier"] - 4 / 3) < 1e-9
