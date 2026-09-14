from dataclasses import replace

import pytest
import torch

from llava.qsteer import QSTEERContext, attach_qsteer_runtime_with_settings, QSTEERRuntimeSettings


def test_question_pool_excludes_other_tokens(tiny, batch):
    _, runner = tiny
    embeddings, masks, _, _ = batch
    actual = runner.controller.pool_context(embeddings, masks.question)
    changed = embeddings.clone()
    changed[~masks.question] = 9000
    torch.testing.assert_close(runner.controller.pool_context(changed, masks.question), actual)
    torch.testing.assert_close(actual, embeddings[:, 2:5].mean(1))


@pytest.mark.parametrize("field", ["question", "diagnostic_query", "visual_tokens"])
@pytest.mark.parametrize("kind", ["missing", "length", "batch", "empty", "nonbinary"])
def test_invalid_prompt_masks(tiny, batch, field, kind):
    _, runner = tiny
    embeddings, masks, forward, _ = batch
    old = getattr(masks, field)
    value = {"missing": None, "length": old[:, :-1], "batch": old[:1],
             "empty": torch.zeros_like(old),
             "nonbinary": old.float() * float("nan")}[kind]
    with pytest.raises(ValueError, match=field):
        runner.prepare_main_context(embeddings, replace(masks, **{field: value}),
                                    forward, training=False)


def test_required_answer_mask(tiny, batch):
    _, runner = tiny
    embeddings, masks, forward, _ = batch
    with pytest.raises(ValueError, match="answer_queries"):
        runner.prepare_main_context(embeddings, replace(masks, answer_queries=None),
                                    forward, training=False)


def test_diagnostic_no_answer_no_grad_and_float32(tiny, batch):
    model, runner = tiny
    embeddings, masks, forward, inputs = batch
    seen = []

    def inspect_input(_module, _args, kwargs):
        seen.append((kwargs["input_ids"].clone(), torch.is_grad_enabled(),
                     QSTEERContext.get_stage(), QSTEERContext.is_icr_enabled(),
                     QSTEERContext.is_old_only_routing()))

    handle = model.register_forward_pre_hook(inspect_input, with_kwargs=True)
    try:
        attention = runner.collect_initial_reference(embeddings, masks, forward)
    finally:
        handle.remove()
    assert len(seen) == 1
    ids, grad_enabled, stage, icr, old_only = seen[0]
    assert not grad_enabled and stage == "diagnostic" and not icr and old_only
    assert ids.shape == (2, 6)
    torch.testing.assert_close(ids, inputs["input_ids"][:, :6])
    assert not torch.isin(ids, torch.tensor([7, 8, 9])).any()
    assert set(attention) == {0, 1}
    for p in attention.values():
        assert p.dtype == torch.float32 and not p.requires_grad
        torch.testing.assert_close(p.sum(-1), torch.ones(2))
        assert torch.count_nonzero(p[:, 2:]) == 0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("method", ["collect_initial_reference", "prepare_main_context"])
def test_context_cleared_after_prompt_exception(tiny, batch, method):
    _, runner = tiny
    embeddings, masks, _, _ = batch

    def fail():
        raise RuntimeError("prompt failed")

    kwargs = {"training": False} if method == "prepare_main_context" else {}
    with pytest.raises(RuntimeError, match="prompt failed"):
        getattr(runner, method)(embeddings, masks, fail, **kwargs)
    assert QSTEERContext._get("z_old") is None
    assert QSTEERContext.get_diagnostic_query_mask() is None
    assert not QSTEERContext.pop_attn_stats()


def test_full_cpu_training_and_frozen_parameters(tiny, batch):
    model, runner = tiny
    embeddings, masks, forward, inputs = batch
    model.eval()
    runner.collect_initial_reference(embeddings, masks, forward)
    runner.finalize_initial_reference()
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    opt.zero_grad(set_to_none=True)
    try:
        prepared = runner.prepare_main_context(embeddings, masks, forward, training=True)
        output = model(**inputs)
        assert output.logits.shape == (2, 9, 32) and torch.isfinite(output.loss)
        output.loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in model.named_parameters() if ".experts." in n)
        assert prepared.route_payload["alpha"].requires_grad
        controller_grads = [p.grad for p in runner.controller.parameters() if p.requires_grad]
        assert any(g is not None and g.abs().sum() > 0 for g in controller_grads)
        assert model.model.layers[-1].self_attn.qsteer_uq.grad.abs().sum() > 0
        opt.step()
    finally:
        runner.clear()
    after = dict(model.named_parameters())
    assert frozen
    for name in frozen:
        assert torch.equal(before[name], after[name]), name
    assert any(not torch.equal(before[n], p) for n, p in after.items() if ".experts." in n)
    for name, p in after.items():
        if ".experts.2." in name or ".experts.3." in name:
            assert p.grad is None and torch.equal(before[name], p)


def test_reattachment_rejected_without_losing_state(tiny):
    model, runner = tiny
    old = runner.controller
    with pytest.raises(RuntimeError, match="already attached"):
        attach_qsteer_runtime_with_settings(model, QSTEERRuntimeSettings(
            expert_num=4, expert_init=2, late_layer_count=2))
    assert model._qsteer_controller is old


def test_cpu_bfloat16_diagnostic_stays_float32(tiny):
    from conftest import synthetic_batch
    model, runner = tiny
    model.bfloat16()
    embeddings, masks, forward, _ = synthetic_batch(model)
    attention = runner.collect_initial_reference(embeddings, masks, forward)
    assert all(p.dtype == torch.float32 for p in attention.values())
    runner.finalize_initial_reference()
    drift = runner.drift_buffer.compute_batch_drift(attention)
    assert drift.dtype == torch.float32 and torch.isfinite(drift).all()


def test_missing_diagnostic_layers_is_an_error(tiny, batch):
    _, runner = tiny
    embeddings, masks, _, _ = batch
    with pytest.raises(RuntimeError, match="diagnostic attention layers"):
        runner.prepare_main_context(embeddings, masks, lambda: None, training=False)
    assert QSTEERContext._get("z_old") is None
