"""Synthetic CPU execution contract; this is not a VQA performance example.

No dataset, tokenizer, pretrained weights, or network access is used.
"""
from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from llava.qsteer import (
    QSTEERContext,
    QSTEERBatchMasks,
    QSTEERMOELoraConfig,
    QSTEERRuntimeSettings,
    QSTEERTwoStageRunner,
    attach_qsteer_runtime_with_settings,
    inject_qsteer_moe_lora,
)


def build_model(*, seed: int = 7, kv_heads: int = 2, num_layers: int = 2):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=num_layers, num_attention_heads=4,
        num_key_value_heads=kv_heads, max_position_embeddings=64,
        attention_dropout=0.0, pad_token_id=0, use_cache=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).cpu()
    model.requires_grad_(False)
    inject_qsteer_moe_lora(model, QSTEERMOELoraConfig(
        expert_num=4, expert_rank=8, topk_update=2, controlled_last_n=2,
    ))
    settings = QSTEERRuntimeSettings(
        expert_num=4, expert_init=2, late_layer_count=2,
        controller_hidden=16, expansion_candidate_last_n=2,
    )
    attach_qsteer_runtime_with_settings(model, settings)
    return model, QSTEERTwoStageRunner(model)


def synthetic_batch(model, *, batch_size: int = 2):
    # Visual positions are synthetic token embeddings, not encoded images.
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]]).expand(batch_size, -1).clone()
    answer = torch.tensor([[7, 8, 9]]).expand(batch_size, -1)
    full = torch.cat((prompt, answer), dim=1)
    question = torch.zeros_like(prompt, dtype=torch.bool)
    question[:, 2:5] = True
    diagnostic = torch.zeros_like(question)
    diagnostic[:, -1] = True
    visual = torch.zeros_like(question)
    visual[:, :2] = True
    answer_queries = torch.zeros_like(full, dtype=torch.bool)
    # Causal LM query t predicts label t+1, including the answer-start boundary.
    answer_queries[:, prompt.size(1) - 1 : -1] = True
    labels = full.clone()
    labels[:, :prompt.size(1)] = -100
    embeddings = model.get_input_embeddings()(prompt).detach()
    masks = QSTEERBatchMasks(question, diagnostic, visual, answer_queries)

    def prompt_forward():
        return model(input_ids=prompt, attention_mask=torch.ones_like(prompt),
                     use_cache=False)

    main_inputs = dict(input_ids=full, attention_mask=torch.ones_like(full),
                       labels=labels, use_cache=False)
    return embeddings, masks, prompt_forward, main_inputs



@pytest.fixture(autouse=True)
def isolated_context():
    torch.set_num_threads(1)
    QSTEERContext.clear()
    yield
    QSTEERContext.clear()


@pytest.fixture
def tiny():
    return build_model()


@pytest.fixture
def batch(tiny):
    return synthetic_batch(tiny[0])
