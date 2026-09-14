from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class QSTEERDriftBuffer(nn.Module):
    """Task-wise reference buffer for late-layer attention drift.

    References are keyed by transformer layer, so their shapes cannot be
    declared as fixed buffers at construction time. The custom state-dict
    hooks below serialize those dynamic tensors alongside the registered
    scalar statistics. Attaching this object as a model submodule therefore
    preserves finalized references and in-progress accumulators.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.ref_by_layer: Dict[int, torch.Tensor] = {}
        self.running_sum: Dict[int, torch.Tensor] = {}
        self.running_count: Dict[int, int] = {}
        self.register_buffer(
            "_eps_state",
            torch.tensor(max(float(eps), 1e-12), dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "_task_drift_sum_state",
            torch.tensor(0.0, dtype=torch.float64),
            persistent=True,
        )
        self.register_buffer(
            "_task_drift_count_state",
            torch.tensor(0, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "_task_drift_mean_state",
            torch.tensor(0.0, dtype=torch.float64),
            persistent=True,
        )

    @property
    def eps(self) -> float:
        return float(self._eps_state.item())

    @eps.setter
    def eps(self, value: float) -> None:
        self._eps_state.fill_(max(float(value), 1e-12))

    @property
    def task_drift_sum(self) -> float:
        return float(self._task_drift_sum_state.item())

    @task_drift_sum.setter
    def task_drift_sum(self, value: float) -> None:
        self._task_drift_sum_state.fill_(float(value))

    @property
    def task_drift_count(self) -> int:
        return int(self._task_drift_count_state.item())

    @task_drift_count.setter
    def task_drift_count(self, value: int) -> None:
        self._task_drift_count_state.fill_(int(value))

    @property
    def task_drift_mean(self) -> float:
        return float(self._task_drift_mean_state.item())

    @task_drift_mean.setter
    def task_drift_mean(self, value: float) -> None:
        self._task_drift_mean_state.fill_(float(value))

    def _apply(self, fn, recurse=True):
        # Follow device moves without first rounding/overflowing diagnostic values
        # through the parent's target dtype (notably fp16 eps=1e-8 -> zero).
        def preserve_precision(value):
            moved = fn(value)
            return value.to(device=moved.device) if value.is_floating_point() else moved

        super()._apply(preserve_precision, recurse=recurse)
        self.ref_by_layer = {
            int(layer_id): preserve_precision(value)
            for layer_id, value in self.ref_by_layer.items()
        }
        self.running_sum = {
            int(layer_id): preserve_precision(value)
            for layer_id, value in self.running_sum.items()
        }
        return self

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for layer_id, value in sorted(self.ref_by_layer.items()):
            if torch.is_tensor(value):
                destination[f"{prefix}ref_by_layer.{int(layer_id)}"] = (
                    value if keep_vars else value.detach()
                )
        for layer_id, value in sorted(self.running_sum.items()):
            if torch.is_tensor(value):
                destination[f"{prefix}running_sum.{int(layer_id)}"] = (
                    value if keep_vars else value.detach()
                )
        count_device = self._task_drift_count_state.device
        for layer_id, value in sorted(self.running_count.items()):
            destination[f"{prefix}running_count.{int(layer_id)}"] = torch.tensor(
                int(value), dtype=torch.int64, device=count_device
            )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        dynamic = {
            "ref_by_layer": {},
            "running_sum": {},
            "running_count": {},
        }
        for field in dynamic:
            field_prefix = f"{prefix}{field}."
            for key in list(state_dict.keys()):
                if not key.startswith(field_prefix):
                    continue
                layer_text = key[len(field_prefix) :]
                try:
                    layer_id = int(layer_text)
                except ValueError:
                    continue
                dynamic[field][layer_id] = state_dict.pop(key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        target_device = self._eps_state.device
        self.ref_by_layer = {
            int(layer_id): value.detach().to(device=target_device).float().clone()
            for layer_id, value in dynamic["ref_by_layer"].items()
        }
        self.running_sum = {
            int(layer_id): value.detach().to(device=target_device).float().clone()
            for layer_id, value in dynamic["running_sum"].items()
        }
        self.running_count = {
            int(layer_id): int(value.detach().view(-1)[0].item())
            for layer_id, value in dynamic["running_count"].items()
        }

    def has_reference(self) -> bool:
        return bool(len(self.ref_by_layer) > 0)

    def get_reference(self, layer_id: int):
        return self.ref_by_layer.get(int(layer_id), None)

    @staticmethod
    def _align_1d(vec: torch.Tensor, target_dim: int) -> torch.Tensor:
        out = vec.view(-1)
        if int(out.numel()) < int(target_dim):
            pad = torch.zeros(
                int(target_dim) - int(out.numel()),
                dtype=out.dtype,
                device=out.device,
            )
            out = torch.cat([out, pad], dim=0)
        elif int(out.numel()) > int(target_dim):
            out = out[: int(target_dim)]
        return out

    def update_task_running(self, layer_id: int, p: torch.Tensor):
        lid = int(layer_id)
        if p is None or (not torch.is_tensor(p)) or p.numel() <= 0:
            return
        p_detached = p.detach().float()
        if not torch.isfinite(p_detached).all():
            p_detached = torch.nan_to_num(p_detached, nan=0.0, posinf=0.0, neginf=0.0)
        if p_detached.dim() == 1:
            mean_vec = p_detached.view(-1)
            batch_count = 1
        else:
            mean_vec = p_detached.mean(dim=0).view(-1)
            batch_count = int(max(1, p_detached.size(0)))
        weighted_sum = mean_vec * float(batch_count)
        if lid not in self.running_sum:
            self.running_sum[lid] = weighted_sum.clone()
            self.running_count[lid] = int(batch_count)
        else:
            prev = self.running_sum[lid].to(device=weighted_sum.device, dtype=weighted_sum.dtype).view(-1)
            if int(prev.numel()) != int(weighted_sum.numel()):
                target_dim = max(int(prev.numel()), int(weighted_sum.numel()))
                prev = self._align_1d(prev, target_dim)
                weighted_sum = self._align_1d(weighted_sum, target_dim)
            self.running_sum[lid] = prev + weighted_sum
            self.running_count[lid] = int(self.running_count[lid]) + int(batch_count)

    def finalize_task_reference(self):
        if not self.running_sum:
            raise RuntimeError("No diagnostic attention is pending for reference finalization.")
        ref = {}
        for lid, total in self.running_sum.items():
            count = max(1, int(self.running_count.get(lid, 0)))
            ref[int(lid)] = (total / float(count)).detach()
        self.ref_by_layer = ref
        self.running_sum = {}
        self.running_count = {}
        if int(self.task_drift_count) > 0:
            self.task_drift_mean = float(self.task_drift_sum) / float(self.task_drift_count)
        else:
            self.task_drift_mean = 0.0
        self.task_drift_sum = 0.0
        self.task_drift_count = 0
        return self.ref_by_layer

    def update_task_drift(self, drift: torch.Tensor) -> None:
        if drift is None or (not torch.is_tensor(drift)) or drift.numel() <= 0:
            return
        values = drift.detach().float().view(-1)
        if not torch.isfinite(values).all():
            values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        self.task_drift_sum += float(values.sum().item())
        self.task_drift_count += int(values.numel())

    def current_task_drift_mean(self) -> float:
        if int(self.task_drift_count) > 0:
            return float(self.task_drift_sum) / float(self.task_drift_count)
        return float(self.task_drift_mean)

    def compute_batch_drift(self, current_by_layer: dict[int, torch.Tensor]) -> torch.Tensor:
        if not isinstance(current_by_layer, dict) or len(current_by_layer) <= 0:
            # Scalar zero fallback; caller will broadcast to batch size if needed.
            return torch.tensor(0.0, dtype=torch.float32)
        valid_layers = []
        batch_size = None
        for lid, p in current_by_layer.items():
            if p is None or (not torch.is_tensor(p)) or p.numel() <= 0:
                continue
            ref = self.ref_by_layer.get(int(lid), None)
            if ref is None or (not torch.is_tensor(ref)) or ref.numel() <= 0:
                continue
            p_cur = p.detach().float()
            if not torch.isfinite(p_cur).all():
                p_cur = torch.nan_to_num(p_cur, nan=0.0, posinf=0.0, neginf=0.0)
            if p_cur.dim() == 1:
                p_cur = p_cur.unsqueeze(0)
            ref_vec = ref.detach().float().to(device=p_cur.device)
            if not torch.isfinite(ref_vec).all():
                ref_vec = torch.nan_to_num(ref_vec, nan=0.0, posinf=0.0, neginf=0.0)
            if ref_vec.dim() != 1:
                ref_vec = ref_vec.view(-1)
            if ref_vec.numel() < p_cur.size(-1):
                pad = torch.zeros(
                    p_cur.size(-1) - ref_vec.numel(),
                    dtype=ref_vec.dtype,
                    device=ref_vec.device,
                )
                ref_vec = torch.cat([ref_vec, pad], dim=0)
            ref_vec = ref_vec[: p_cur.size(-1)]
            ref_batch = ref_vec.unsqueeze(0).expand(p_cur.size(0), -1)
            p_cur = p_cur.clamp(min=0.0) + self.eps
            p_cur = p_cur / p_cur.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            ref_batch = ref_batch.clamp(min=0.0) + self.eps
            ref_batch = ref_batch / ref_batch.sum(dim=-1, keepdim=True).clamp(min=self.eps)
            # KL(current || reference), matching the paper's diagnostic input.
            # Both operands are smoothed and normalized immediately above.
            kl = (p_cur * (p_cur.log() - ref_batch.log())).sum(dim=-1)
            valid_layers.append(kl)
            batch_size = int(p_cur.size(0))
        if len(valid_layers) <= 0:
            if batch_size is None:
                return torch.tensor(0.0, dtype=torch.float32)
            return torch.zeros(batch_size, dtype=torch.float32)
        stack = torch.stack(valid_layers, dim=0)  # [L, B]
        return stack.mean(dim=0)  # [B]
