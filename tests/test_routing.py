import pytest
import torch

from llava.qsteer import QSTEERContext, QSTEERMOELoraConfig, QSTEERMOELoraLinear


def bank_and_logits():
    torch.manual_seed(29)
    layer = QSTEERMOELoraLinear(torch.nn.Linear(3, 2, bias=False),
                               QSTEERMOELoraConfig(expert_num=4, expert_rank=2,
                                                   lora_alpha=2, topk_update=2), layer_id=0)
    with torch.no_grad():
        for index, expert in enumerate(layer.experts):
            expert.A.weight.fill_(0.1 * (index + 1))
            expert.B.weight.fill_(0.2 * (index + 1))
    logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log().requires_grad_()
    QSTEERContext.set_route_payload({
        "z_old": logits, "z_new": logits, "lambda": torch.zeros(1, 1),
        "route_layer_slots": {0: 0}, "old_masks": {0: torch.ones(4, dtype=torch.bool)},
        "new_masks": {0: torch.zeros(4, dtype=torch.bool)},
    })
    return layer, logits


@pytest.mark.parametrize("two_dimensional", [False, True])
def test_soft_forward_sparse_parameter_gradients_and_dense_gate_gradient(two_dimensional):
    layer, logits = bank_and_logits()
    x = torch.tensor([[[0.7, -0.2, 0.8], [0.1, 0.2, 0.4]]])
    if two_dimensional:
        x = x[:, 0]
    weights = logits.softmax(-1)[0]
    expert_outputs = [expert(x) for expert in layer.experts]
    dense = layer.base_layer(x) + sum(w * output for w, output in zip(weights, expert_outputs))
    sparse = sum(weights[i] / weights[:2].sum() * expert_outputs[i] for i in (0, 1))
    expected_gate_grad = torch.autograd.grad(dense.sum(), logits, retain_graph=True)[0]
    selected_params = [p for e in layer.experts[:2] for p in e.parameters()]
    expected_expert_grad = torch.autograd.grad(sparse.sum(), selected_params, retain_graph=True)
    output = layer(x)
    torch.testing.assert_close(output, dense)
    assert not torch.allclose(output, layer.base_layer(x) + sparse)
    output.sum().backward()
    torch.testing.assert_close(logits.grad, expected_gate_grad)
    assert logits.grad.abs().sum() > 0
    for p, expected in zip(selected_params, expected_expert_grad):
        torch.testing.assert_close(p.grad, expected)
    for expert in layer.experts[2:]:
        assert all(p.grad is None for p in expert.parameters())
    assert layer.base_layer.weight.grad is None


def test_frozen_old_and_inactive_experts_survive_adamw_momentum():
    layer, _ = bank_and_logits()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.03, weight_decay=0.2)
    x = torch.ones(1, 2, 3)
    layer(x).square().sum().backward()
    optimizer.step()
    assert optimizer.state[layer.experts[0].B.weight]["exp_avg"].abs().sum() > 0
    layer.set_expert_trainable(0, False)
    for i in (2, 3):
        layer.set_expert_trainable(i, False)
    before = {n: p.clone() for n, p in layer.named_parameters()}
    optimizer.zero_grad(set_to_none=False)
    # Rebuild routing graph for the next step.
    logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log().requires_grad_()
    QSTEERContext._set("z_old", logits)
    layer(x).square().sum().backward()
    optimizer.step()
    for name, p in layer.named_parameters():
        if "experts.1." not in name:
            assert torch.equal(before[name], p), name
    assert not torch.equal(before["experts.1.B.weight"], layer.experts[1].B.weight)
    assert all(p.grad is None for p in layer.experts[0].parameters())


def test_factorized_old_new_softmax_and_no_new_bank():
    old_logits = torch.tensor([[0.1, 0.9, 5.0, -2.0]], requires_grad=True)
    new_logits = torch.tensor([[-3.0, 7.0, 0.2, 0.8]], requires_grad=True)
    lam = torch.tensor([[0.3]], requires_grad=True)
    payload = dict(z_old=old_logits, z_new=new_logits, **{"lambda": lam},
                   route_layer_slots={0: 0},
                   old_masks={0: torch.tensor([1, 1, 0, 0], dtype=torch.bool)},
                   new_masks={0: torch.tensor([0, 0, 1, 1], dtype=torch.bool)})
    QSTEERContext.set_route_payload(payload)
    actual = QSTEERContext.get_gate(torch.ones(1, 3, 2), 4, layer_id=0)
    expected = torch.cat(((1 - lam) * old_logits[:, :2].softmax(-1),
                          lam * new_logits[:, 2:].softmax(-1)), -1)
    torch.testing.assert_close(actual, expected[:, None].expand(1, 3, 4))
    (actual * torch.arange(4)).sum().backward()
    for value in [old_logits, new_logits, lam]:
        assert value.grad.abs().sum() > 0
    payload["new_masks"] = {0: torch.zeros(4, dtype=torch.bool)}
    QSTEERContext.set_route_payload(payload)
    actual = QSTEERContext.get_gate(torch.ones(1, 1, 2), 4, layer_id=0)
    torch.testing.assert_close(actual[0, 0, :2], old_logits[0, :2].softmax(-1))
    assert torch.count_nonzero(actual[..., 2:]) == 0
