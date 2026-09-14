import copy

import pytest
import torch
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, repeat_kv

from llava.qsteer import QSTEERConfig, QSTEERContext
from llava.qsteer.core.attn_patch import _build_qsteer_attention_forward, _ensure_icr_params


def attention_pair(kv_heads=2):
    QSTEERContext.set_icr_enabled(False)
    torch.manual_seed(31)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=kv_heads,
                         attention_dropout=0.0)
    original = LlamaAttention(config, layer_idx=0).eval()
    patched = copy.deepcopy(original)
    cfg = QSTEERConfig(expert_num=4, expert_init=2, late_layer_count=1, icr_rank=8)
    _ensure_icr_params(patched, cfg.icr_rank)
    patched.forward = _build_qsteer_attention_forward(patched, 0, 0, cfg)
    return original, patched, cfg


def causal_mask(batch, length, *, width=None):
    width = length if width is None else width
    masked = torch.arange(width)[None, :] > torch.arange(length)[:, None]
    return torch.zeros(batch, 1, length, width).masked_fill(masked, torch.finfo(torch.float32).min)


@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("extra_key_column", [False, True])
def test_icr_disabled_matches_original_eager(kv_heads, extra_key_column):
    original, patched, _ = attention_pair(kv_heads)
    x = torch.randn(2, 5, 32)
    mask = causal_mask(2, 5, width=6 if extra_key_column else 5)
    mask[1, :, :, 1] = torch.finfo(mask.dtype).min  # interior key padding
    pos = torch.arange(5).unsqueeze(0)
    kwargs = dict(attention_mask=mask, position_ids=pos, output_attentions=True)
    expected = original(x, **kwargs)
    actual = patched(x, **kwargs)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)
    assert torch.count_nonzero(actual[1].triu(1)) == 0
    assert torch.count_nonzero(actual[1][1, :, 1:, 1]) == 0


@pytest.mark.parametrize("batch_size", [1, 2])
def test_icr_is_exact_pre_softmax_selected_query_update(batch_size, monkeypatch):
    _, patched, cfg = attention_pair()
    x = torch.randn(batch_size, 4, 32, requires_grad=True)
    alpha = torch.full((batch_size, 1, 8), 0.7, requires_grad=True)
    gamma = torch.full((batch_size, 1, 4), 0.6, requires_grad=True)
    lam = torch.linspace(0.2, 0.8, batch_size).view(batch_size, 1).requires_grad_()
    answer = torch.zeros(batch_size, 4, dtype=torch.bool)
    answer[:, 2] = True
    QSTEERContext.set_stage("main")
    QSTEERContext.set_icr_enabled(True)
    QSTEERContext.set_answer_mask(answer)
    QSTEERContext.set_route_payload({"alpha": alpha, "gamma": gamma, "lambda": lam})
    mask = causal_mask(batch_size, 4)
    position_ids = torch.arange(4).unsqueeze(0)
    captured = []
    softmax = torch.softmax

    def observe_softmax(logits, *args, **kwargs):
        captured.append(logits.detach().clone())
        return softmax(logits, *args, **kwargs)

    monkeypatch.setattr(torch, "softmax", observe_softmax)
    out, weights, _ = patched(x, attention_mask=mask, position_ids=position_ids,
                              output_attentions=True)
    q = patched.q_proj(x).view(batch_size, 4, 4, 8).transpose(1, 2)
    k = patched.k_proj(x).view(batch_size, 4, 2, 8).transpose(1, 2)
    cos, sin = patched.rotary_emb(k, position_ids)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    k = repeat_kv(k, 2)
    base = q @ k.transpose(-1, -2) / (8 ** 0.5)
    qi = torch.einsum("bhqd,hdr->bhqr", q, patched.qsteer_uq.reshape(4, 8, 8))
    ki = torch.einsum("bhkd,hdr->bhkr", k, patched.qsteer_uk.reshape(4, 8, 8))
    delta = torch.einsum("bhqr,br,bhkr->bhqk", qi, alpha[:, 0], ki)
    delta *= gamma[:, 0, :, None, None] * lam.reshape(batch_size, 1, 1, 1) * cfg.icr_scale
    delta *= answer[:, None, :, None]
    # Compare unmasked entries, independently of the finite masking sentinel.
    valid = mask.expand(-1, 4, -1, -1) == 0
    torch.testing.assert_close(captured[-1][valid], (base + delta)[valid])
    expected_weights = softmax(base + mask + delta, dim=-1)
    torch.testing.assert_close(weights, expected_weights)
    unselected = ~answer[:, None, :, None].expand_as(weights)
    torch.testing.assert_close(weights[unselected], softmax(base + mask, -1)[unselected])
    assert out.shape == (batch_size, 4, 32)
    out.square().sum().backward()
    for value in [alpha, gamma, lam, patched.qsteer_uq, patched.qsteer_uk]:
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize("icr", [False, True])
def test_dynamic_cache_matches_full_prefix(icr):
    original, patched, _ = attention_pair()
    x = torch.randn(2, 5, 32)
    if icr:
        QSTEERContext.set_stage("main")
        QSTEERContext.set_icr_enabled(True)
        QSTEERContext.set_answer_mask(torch.ones(2, 5, dtype=torch.bool))
        QSTEERContext.set_route_payload({
            "alpha": torch.randn(2, 1, 8), "gamma": torch.rand(2, 1, 4),
            "lambda": torch.rand(2, 1),
        })
    full = patched(x, attention_mask=causal_mask(2, 5),
                   position_ids=torch.arange(5).unsqueeze(0))[0]
    cache = DynamicCache()
    chunks = []
    for start, end in [(0, 3), (3, 4), (4, 5)]:
        QSTEERContext.set_answer_mask(torch.ones(2, end - start, dtype=torch.bool))
        out = patched(x[:, start:end], attention_mask=causal_mask(2, 5)[:, :, start:end, :end],
                      position_ids=torch.arange(start, end).unsqueeze(0),
                      cache_position=torch.arange(start, end),
                      past_key_value=cache, use_cache=True)[0]
        chunks.append(out)
    torch.testing.assert_close(torch.cat(chunks, 1), full, atol=2e-6, rtol=2e-5)
    assert cache.get_seq_length() == 5
    if not icr:
        expected = original(x, attention_mask=causal_mask(2, 5),
                            position_ids=torch.arange(5).unsqueeze(0))[0]
        torch.testing.assert_close(full, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("fault", ["kv_groups", "output_projection", "short_mask", "mask_batch",
                                   "mask_rank", "answer_length", "gamma_heads", "lambda_batch"])
def test_invalid_attention_shapes_are_errors(fault):
    _, patched, _ = attention_pair()
    x = torch.randn(2, 4, 32)
    mask = causal_mask(2, 4)
    if fault == "kv_groups":
        patched.num_key_value_groups = 3
    elif fault == "output_projection":
        patched.o_proj = torch.nn.Linear(24, 32, bias=False)
    elif fault == "short_mask":
        mask = mask[..., :3]
    elif fault == "mask_batch":
        mask = mask[:1].expand(3, -1, -1, -1)
    elif fault == "mask_rank":
        mask = torch.zeros(2, 4, 4)
    else:
        QSTEERContext.set_stage("main")
        QSTEERContext.set_icr_enabled(True)
        QSTEERContext.set_answer_mask(torch.ones(2, 3 if fault == "answer_length" else 4,
                                                dtype=torch.bool))
        QSTEERContext.set_route_payload({
            "alpha": torch.ones(2, 1, 8),
            "gamma": torch.ones(2, 1, 3 if fault == "gamma_heads" else 4),
            "lambda": torch.ones(3 if fault == "lambda_batch" else 2, 1),
        })
    with pytest.raises((ValueError, RuntimeError), match=r"layer.?0.*expected.*actual"):
        patched(x, attention_mask=mask, position_ids=torch.arange(4).unsqueeze(0))


@pytest.mark.parametrize("boolean_4d", [False, True])
def test_fully_masked_rows_rejected(boolean_4d):
    _, patched, _ = attention_pair()
    mask = torch.zeros((2, 1, 4, 4) if boolean_4d else (2, 4), dtype=torch.bool)
    with pytest.raises(ValueError, match="fully masked"):
        patched(torch.randn(2, 4, 32), attention_mask=mask,
                position_ids=torch.arange(4).unsqueeze(0))


@pytest.mark.parametrize("layout", ["bool_2d", "int_2d", "bool_4d"])
def test_valid_binary_masks_and_right_padding(layout):
    original, patched, _ = attention_pair()
    x = torch.randn(2, 5, 32)
    padding = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    allowed = torch.ones(5, 5, dtype=torch.bool).tril()[None, None] & padding[:, None, None, :]
    additive = torch.zeros(2, 1, 5, 5).masked_fill(~allowed, torch.finfo(torch.float32).min)
    mask = allowed if layout == "bool_4d" else padding.to(
        torch.int64 if layout == "int_2d" else torch.bool)
    positions = torch.arange(5).unsqueeze(0)
    expected = original(x, attention_mask=additive, position_ids=positions, output_attentions=True)
    actual = patched(x, attention_mask=mask, position_ids=positions, output_attentions=True)
    torch.testing.assert_close(actual[0], expected[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual[1], expected[1], atol=2e-6, rtol=2e-5)


def test_left_padding_with_fully_masked_queries_is_explicitly_unsupported():
    _, patched, _ = attention_pair()
    padding = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    with pytest.raises(ValueError, match="fully masked"):
        patched(torch.randn(1, 4, 32), attention_mask=padding,
                position_ids=torch.arange(4).unsqueeze(0))
