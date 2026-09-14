import pytest
import torch

from llava.qsteer import QSTEERDriftBuffer


def test_smoothed_normalized_kl_current_to_reference():
    buf = QSTEERDriftBuffer(eps=1e-3)
    reference = torch.tensor([0.9, 0.1, 0.0])
    current = torch.tensor([[0.2, 0.7, 0.1], [0.0, 2.0, 3.0]], dtype=torch.float16,
                           requires_grad=True)
    buf.update_task_running(4, reference)
    buf.finalize_task_reference()
    p = current.detach().float().clamp_min(0) + buf.eps
    p /= p.sum(-1, keepdim=True)
    q = reference.float().clamp_min(0) + buf.eps
    q /= q.sum()
    expected = (p * (p.log() - q.log())).sum(-1)
    actual = buf.compute_batch_drift({4: current})
    assert actual.dtype == torch.float32 and not actual.requires_grad
    torch.testing.assert_close(actual, expected)
    reverse = (q * (q.log() - p.log())).sum(-1)
    assert not torch.allclose(actual, reverse)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_parent_cast_preserves_diagnostic_precision(dtype):
    parent = torch.nn.Module()
    parent.buffer = QSTEERDriftBuffer()
    buf = parent.buffer
    buf.update_task_running(0, torch.tensor([[1.0, 0.0]]))
    buf.finalize_task_reference()
    buf.update_task_running(0, torch.tensor([[90000.25, 0.0]]))
    buf.update_task_drift(torch.tensor([90000.25]))
    before = {k: v.clone() for k, v in buf.state_dict().items()}
    parent.to(dtype=dtype)
    for key, value in buf.state_dict().items():
        assert torch.equal(before[key], value), key
    assert buf.eps == 1e-8
    assert torch.isfinite(buf.compute_batch_drift({0: torch.tensor([[1.0, 0.0]])})).all()


def test_reference_fixed_until_boundary_and_duplicate_rejected():
    buf = QSTEERDriftBuffer()
    buf.update_task_running(0, torch.tensor([[0.7, 0.3]]))
    initial = buf.finalize_task_reference()[0].clone()
    buf.update_task_running(0, torch.tensor([[0.2, 0.8], [0.4, 0.6]]))
    buf.update_task_drift(torch.tensor([2.0, 4.0]))
    assert torch.equal(buf.get_reference(0), initial)
    buf.finalize_task_reference()
    expected = torch.tensor([0.3, 0.7])
    torch.testing.assert_close(buf.get_reference(0), expected)
    assert buf.task_drift_mean == 3.0 and buf.task_drift_count == 0
    with pytest.raises(RuntimeError, match="No diagnostic attention"):
        buf.finalize_task_reference()
    torch.testing.assert_close(buf.get_reference(0), expected)
    assert buf.task_drift_mean == 3.0
