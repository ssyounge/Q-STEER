import io

import pytest
import torch

from conftest import build_model, synthetic_batch
from llava.qsteer import (
    QSTEERContext, QSTEERExpansionProbe, QSTEERTaskFinalizeCallback,
    QSTEERTwoStageRunner, build_qsteer_expansion_probe,
)


def initialize_reference(model, runner):
    embeddings, masks, forward, inputs = synthetic_batch(model)
    runner.collect_initial_reference(embeddings, masks, forward)
    runner.finalize_initial_reference()
    return embeddings, masks, forward, inputs


def accumulate_task(runner, batch):
    embeddings, masks, forward, _ = batch
    runner.prepare_main_context(embeddings, masks, forward, training=True)
    runner.clear()


def finalize(model):
    QSTEERTaskFinalizeCallback().on_train_end(None, None, None, model=model)


def state_snapshot(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def test_task_zero_and_exact_128_sample_probe(tiny):
    model, runner = tiny
    batch = initialize_reference(model, runner)
    assert build_qsteer_expansion_probe(runner.cfg, task_index=0) is None
    with pytest.raises(ValueError, match="initial task"):
        QSTEERExpansionProbe().prepare(model, task_index=0)
    accumulate_task(runner, batch)
    finalize(model)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    identities = {n: id(p) for n, p in model.named_parameters()}
    with torch.no_grad():
        runner.controller.new_route_b[:, 2] = 2.0
        runner.controller.new_route_b[:, 3] = -2.0
    probe = build_qsteer_expansion_probe(runner.cfg, task_index=1)
    probe.prepare(model, task_index=1)
    for layer in model.model.layers:
        assert layer.qsteer_slot_state.tolist() == [3, 3, 1, 1]
    expected = {i: torch.zeros(4) for i in probe.candidate_layers}
    for batch_size, take in [(65, 65), (70, 63)]:
        emb, masks, forward, _ = synthetic_batch(model, batch_size=batch_size)
        # Distinct excess examples make truncation sensitive to which rows are counted.
        emb[-7:, 2:5] += 100
        runner.prepare_main_context(emb, masks, forward, training=False,
                                    require_answer_queries=False)
        for layer_id in expected:
            gates = QSTEERContext.get_gate(torch.zeros(batch_size, 1, 1), 4, layer_id)
            expected[layer_id] += gates[:take].detach().float().sum((0, 1))
        assert probe.observe() == take
        runner.clear()
        if take == 65:
            with pytest.raises(RuntimeError, match="65/128"):
                probe.finalize()
    assert probe.complete and probe.observed_samples == 128
    for layer_id in expected:
        torch.testing.assert_close(probe.utilization_sum[layer_id], expected[layer_id])
    result = probe.finalize()
    for layer_id, chosen in result["selected_by_layer"].items():
        values = expected[layer_id][[2, 3]] / 128
        threshold = values.mean() - 0.5 * values.std(unbiased=False)
        selected = [idx for idx, value in zip([2, 3], values) if value > threshold]
        assert chosen == selected == [2]
        assert model.model.layers[layer_id].qsteer_slot_state.tolist() == [3, 3, 2, 0]
    assert shapes == {n: tuple(p.shape) for n, p in model.named_parameters()}
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    with pytest.raises(RuntimeError, match="exactly once"):
        probe.finalize()
    before = state_snapshot(model)
    with pytest.raises(RuntimeError, match="prepare"):
        probe.observe()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_empty_task_finalization_is_atomic(tiny):
    model, runner = tiny
    initialize_reference(model, runner)
    before = state_snapshot(model)
    with pytest.raises(RuntimeError, match="No diagnostic attention"):
        finalize(model)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_repeated_prepare_preserves_pending_probe(tiny):
    model, runner = tiny
    batch = initialize_reference(model, runner)
    accumulate_task(runner, batch)
    finalize(model)
    probe = QSTEERExpansionProbe(probe_samples=3)
    probe.prepare(model, task_index=1)
    indices = dict(probe.probe_indices)
    with pytest.raises(RuntimeError, match="unfinished"):
        probe.prepare(model, task_index=1)
    assert probe.prepared and probe.probe_indices == indices


def test_new_instance_checkpoint_and_task_consolidation(tiny):
    model, runner = tiny
    batch = initialize_reference(model, runner)
    accumulate_task(runner, batch)
    finalize(model)
    probe = QSTEERExpansionProbe(probe_samples=3)
    probe.prepare(model, task_index=1)
    with torch.no_grad():
        runner.controller.new_route_b[:, 2] = 2.0
        runner.controller.new_route_b[:, 3] = -2.0
    emb, masks, forward, inputs = batch
    for _ in range(2):
        runner.prepare_main_context(emb, masks, forward, training=False,
                                    require_answer_queries=False)
        probe.observe()
        runner.clear()
    probe.finalize()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=0.01, weight_decay=0.1)
    before = state_snapshot(model)
    saved_ref = {k: v.clone() for k, v in runner.drift_buffer.ref_by_layer.items()}
    runner.prepare_main_context(emb, masks, forward, training=True)
    loss = model(**inputs).loss
    loss.backward()
    optimizer.step()
    runner.clear()
    for name, value in model.state_dict().items():
        if ".experts.0." in name or ".experts.1." in name or ".experts.3." in name:
            assert torch.equal(before[name], value), name
    assert any(not torch.equal(before[n], v) for n, v in model.state_dict().items()
               if ".experts.2." in n)
    assert all(torch.equal(saved_ref[k], v) for k, v in runner.drift_buffer.ref_by_layer.items())

    def evaluate(m, r):
        m.eval()
        e, ma, f, inp = synthetic_batch(m)
        with torch.no_grad():
            prepared = r.prepare_main_context(e, ma, f, training=False)
            outputs = m(**inp).logits.clone()
            route = {k: v.clone() for k, v in prepared.route_payload.items()
                     if torch.is_tensor(v)}
        r.clear()
        return outputs, route

    expected, expected_route = evaluate(model, runner)
    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    clone, clone_runner = build_model(seed=999)
    assert clone is not model and clone_runner.controller is not runner.controller
    checkpoint.seek(0)
    clone.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    # Attach runtime before load; context/payload are reconstructed by the runner.
    clone_runner = QSTEERTwoStageRunner(clone)
    actual, actual_route = evaluate(clone, clone_runner)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    for name in expected_route:
        torch.testing.assert_close(actual_route[name], expected_route[name])
    original_state = state_snapshot(model)
    for name, value in clone.state_dict().items():
        assert torch.equal(value, original_state[name]), name
    assert clone_runner.drift_buffer.running_count == runner.drift_buffer.running_count

    route_before = runner.controller.old_route_w.detach().clone()
    new_before = runner.controller.new_route_w.detach().clone()
    expected_reference = {k: v / runner.drift_buffer.running_count[k]
                          for k, v in runner.drift_buffer.running_sum.items()}
    finalize(model)
    for layer in model.model.layers:
        assert layer.qsteer_slot_state.tolist() == [3, 3, 3, 0]
    for slot in range(2):
        torch.testing.assert_close(runner.controller.old_route_w[slot, 2], new_before[slot, 2])
        torch.testing.assert_close(runner.controller.old_route_w[slot, :2], route_before[slot, :2])
    for k, value in expected_reference.items():
        torch.testing.assert_close(runner.drift_buffer.get_reference(k), value)
    finalized = state_snapshot(model)
    final_output, final_route = evaluate(model, runner)
    final_checkpoint = io.BytesIO()
    torch.save(model.state_dict(), final_checkpoint)
    restored, restored_runner = build_model(seed=123)
    final_checkpoint.seek(0)
    restored.load_state_dict(torch.load(final_checkpoint, weights_only=True), strict=True)
    restored_output, restored_route = evaluate(restored, restored_runner)
    torch.testing.assert_close(restored_output, final_output, atol=2e-6, rtol=2e-5)
    for name, value in restored.state_dict().items():
        assert torch.equal(finalized[name], value), name
    for name in final_route:
        torch.testing.assert_close(restored_route[name], final_route[name])
    with pytest.raises(RuntimeError, match="No diagnostic attention"):
        finalize(model)
    for name, value in model.state_dict().items():
        assert torch.equal(finalized[name], value), name


def test_finalize_initial_reference_twice_preserves_reference(tiny):
    model, runner = tiny
    initialize_reference(model, runner)
    before = state_snapshot(model)
    with pytest.raises(RuntimeError, match="already exists"):
        runner.finalize_initial_reference()
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name


def test_route_slot_mapping_uses_last_two_of_three_layers():
    model, runner = build_model(num_layers=3)
    assert runner.payload["route_layer_slots"] == {1: 0, 2: 1}
    batch = initialize_reference(model, runner)
    accumulate_task(runner, batch)
    finalize(model)
    probe = QSTEERExpansionProbe(probe_samples=1, candidate_last_n=2)
    probe.prepare(model, task_index=1)
    with torch.no_grad():
        runner.controller.new_route_b[:, 2] = 3
        runner.controller.new_route_b[:, 3] = -3
    emb, masks, forward, _ = batch
    runner.prepare_main_context(emb, masks, forward, training=False,
                                require_answer_queries=False)
    assert probe.observe() == 1
    runner.clear()
    assert probe.finalize()["selected_by_layer"] == {1: [2], 2: [2]}
    before = {name: value.detach().clone() for name, value in
              runner.controller.named_parameters() if name.startswith(("old_route", "new_route"))}
    accumulate_task(runner, batch)
    finalize(model)
    for suffix in ["w", "b"]:
        old = getattr(runner.controller, "old_route_" + suffix)
        for slot in [0, 1]:
            torch.testing.assert_close(old[slot, 2], before["new_route_" + suffix][slot, 2])
            torch.testing.assert_close(old[slot, [0, 1, 3]], before["old_route_" + suffix][slot, [0, 1, 3]])
