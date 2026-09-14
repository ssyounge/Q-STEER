"""Mask-only expert promotion for post-initial Q-STEER tasks.

The probe is an explicit prompt-only stage that runs before main-task
optimization. It observes exactly probe_samples examples by default and never
changes parameter shapes.
"""

from __future__ import annotations

from collections import defaultdict
import re

import torch

from .route_context import QSTEERContext
from .state_sync import assert_qsteer_layer_state, sync_qsteer_context_and_payload

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _unwrap_lm(model):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model
    if hasattr(model, "layers"):
        return model
    raise ValueError("Could not locate transformer layers on the provided model.")


def _get_layers(lm):
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    return lm.layers


def _iter_moe_modules_by_layer(model, only_moe_lora: bool = True):
    modules_by_layer = defaultdict(list)
    for module_name, module in model.named_modules():
        is_qsteer_bank = all(
            hasattr(module, name) for name in ("experts", "expert_num", "topk")
        )
        if only_moe_lora and not is_qsteer_bank:
            continue
        match = _LAYER_PATTERN.search(module_name)
        if match is None:
            continue
        layer_id = int(match.group(1))
        module.qsteer_layer_id = layer_id
        module.qsteer_module_name = str(module_name)
        modules_by_layer[layer_id].append(module)
    return modules_by_layer


def _set_expert_trainable(
    module,
    expert_idx: int,
    trainable: bool,
):
    if hasattr(module, "set_expert_trainable"):
        module.set_expert_trainable(expert_idx, trainable)
        return
    try:
        expert = module.experts[int(expert_idx)]
    except (AttributeError, IndexError, TypeError):
        return
    enabled = bool(trainable)
    for parameter in expert.parameters():
        # Keep every preallocated slot in optimizer param groups across tasks.
        parameter.requires_grad_(True)
        parameter._qsteer_effective_trainable = enabled
        if not enabled and parameter.grad is not None:
            parameter.grad = None


class QSTEERExpansionProbe:
    """Accumulate candidate-slot utilization and promote by a layer threshold.

    Call prepare once after the previous task has been finalized, run prompt-only
    forward passes, call observe after each pass, and call finalize before main
    training.
    """

    def __init__(
        self,
        *,
        probe_samples: int = 128,
        probe_slots: int = 2,
        beta_thr: float = 0.5,
        candidate_last_n: int = 6,
    ) -> None:
        if int(probe_samples) < 1:
            raise ValueError("probe_samples must be positive")
        if int(probe_slots) < 1:
            raise ValueError("probe_slots must be positive")
        if int(candidate_last_n) < 1:
            raise ValueError("candidate_last_n must be positive")
        self.probe_samples = int(probe_samples)
        self.probe_slots = int(probe_slots)
        self.beta_thr = float(beta_thr)
        self.candidate_last_n = int(candidate_last_n)
        self._reset()

    def _reset(self) -> None:
        self.model = None
        self.layers = None
        self.modules_by_layer = {}
        self.candidate_layers: list[int] = []
        self.probe_indices: dict[int, list[int]] = {}
        self.utilization_sum: dict[int, torch.Tensor] = {}
        self.observed_samples = 0
        self.prepared = False
        self.finalized = False

    @property
    def complete(self) -> bool:
        return self.observed_samples >= self.probe_samples

    def _sync(self, *, treat_probe_as_new: bool, allow_probe: bool, tag: str) -> None:
        lm = _unwrap_lm(self.model)
        payload = getattr(lm, "_qsteer", None)
        sync_qsteer_context_and_payload(
            self.layers,
            payload=payload if isinstance(payload, dict) else None,
            treat_probe_as_new=treat_probe_as_new,
        )
        assert_qsteer_layer_state(
            self.layers,
            payload=payload if isinstance(payload, dict) else None,
            treat_probe_as_new=treat_probe_as_new,
            allow_probe=allow_probe,
            source_tag=tag,
        )

    def prepare(self, model, *, task_index: int) -> None:
        """Expose candidates for a post-initial task without training them."""

        if int(task_index) <= 0:
            raise ValueError("The initial task uses E_init directly and has no expansion probe.")

        if self.prepared and not self.finalized:
            raise RuntimeError("This expansion probe is unfinished; observe/finalize it before preparing again.")
        self._reset()
        self.model = model
        lm = _unwrap_lm(model)
        self.layers = _get_layers(lm)
        self.modules_by_layer = _iter_moe_modules_by_layer(model)
        route_layers = sorted(self.modules_by_layer)
        self.candidate_layers = route_layers[-self.candidate_last_n :]
        if not self.candidate_layers:
            raise RuntimeError("No Q-STEER MoE-LoRA layers were found for expansion.")

        for layer_id in self.modules_by_layer:
            slots = getattr(self.layers[layer_id], "qsteer_slot_state", None)
            if not torch.is_tensor(slots):
                raise RuntimeError(f"Layer {layer_id} has no Q-STEER slot state.")
            if bool(((slots == 1) | (slots == 2)).any()):
                raise RuntimeError(
                    f"Layer {layer_id} has an unfinished probe or task; finalize it before preparing expansion."
                )

        for layer_id, modules in self.modules_by_layer.items():
            if layer_id >= len(self.layers):
                continue
            state = getattr(self.layers[layer_id], "qsteer_slot_state", None)
            if not torch.is_tensor(state):
                raise RuntimeError(f"Layer {layer_id} has no Q-STEER slot state.")
            if bool((state == 1).any()):
                raise RuntimeError(
                    f"Layer {layer_id} still has an unfinished probe from a prior task."
                )
            for module in modules:
                for expert_idx in range(int(state.numel())):
                    _set_expert_trainable(module, expert_idx, False)

        for layer_id in self.candidate_layers:
            layer = self.layers[layer_id]
            state = layer.qsteer_slot_state
            if bool((state == 2).any()):
                raise RuntimeError(
                    "Finalize the previous task before preparing the next expansion probe."
                )
            inactive = torch.nonzero(state == 0, as_tuple=False).flatten().tolist()
            candidates = [int(index) for index in inactive[: self.probe_slots]]
            self.probe_indices[layer_id] = candidates
            self.utilization_sum[layer_id] = torch.zeros(
                state.numel(), dtype=torch.float32, device=state.device
            )
            if candidates:
                state[candidates] = 1
                layer.qsteer_expert_mask[candidates] = True

        self._sync(
            treat_probe_as_new=True,
            allow_probe=True,
            tag="expansion_probe.prepare",
        )
        self.prepared = True

    @staticmethod
    def _route_batch_shape() -> tuple[int, torch.device, torch.dtype]:
        for name in ("z_new", "z_old", "g"):
            value = QSTEERContext._get(name)
            if torch.is_tensor(value) and value.dim() >= 2:
                dtype = value.dtype if value.dtype.is_floating_point else torch.float32
                return int(value.size(0)), value.device, dtype
        raise RuntimeError(
            "No Q-STEER route payload is available. Call observe immediately "
            "after a prompt-only forward pass."
        )

    def observe(self) -> int:
        """Accumulate utilization from the most recent prompt-only forward."""

        if not self.prepared or self.finalized:
            raise RuntimeError("prepare must be called before observe")
        if self.complete:
            return 0

        batch_size, device, dtype = self._route_batch_shape()
        take = min(batch_size, self.probe_samples - self.observed_samples)
        dummy_tokens = torch.zeros(batch_size, 1, 1, device=device, dtype=dtype)

        with torch.no_grad():
            for layer_id in self.candidate_layers:
                gate = QSTEERContext.get_gate(
                    dummy_tokens,
                    expert_num=int(self.layers[layer_id].qsteer_slot_state.numel()),
                    layer_id=layer_id,
                )
                per_example = gate[:take].float().mean(dim=1)
                self.utilization_sum[layer_id].add_(
                    per_example.sum(dim=0).to(self.utilization_sum[layer_id].device)
                )

        self.observed_samples += int(take)
        return int(take)

    def finalize(self, *, require_full_budget: bool = True) -> dict:
        """Promote candidates above mean - beta*std and remask the others."""

        if not self.prepared or self.finalized:
            raise RuntimeError("prepare must be called exactly once before finalize")
        if require_full_budget and not self.complete:
            raise RuntimeError(
                f"Probe observed {self.observed_samples}/{self.probe_samples} samples."
            )
        if self.observed_samples <= 0:
            raise RuntimeError("Cannot finalize an empty expansion probe.")

        selected_by_layer: dict[int, list[int]] = {}
        utilization_by_layer: dict[int, dict[int, float]] = {}
        threshold_by_layer: dict[int, float | None] = {}

        for layer_id in self.candidate_layers:
            layer = self.layers[layer_id]
            state = layer.qsteer_slot_state
            candidates = self.probe_indices.get(layer_id, [])
            selected: list[int] = []
            if candidates:
                values = self.utilization_sum[layer_id][candidates]
                values = values / float(self.observed_samples)
                threshold = values.mean() - self.beta_thr * values.std(unbiased=False)
                selected = [
                    int(index)
                    for index, value in zip(candidates, values)
                    if float(value.item()) > float(threshold.item())
                ]
                utilization_by_layer[layer_id] = {
                    int(index): float(value.item())
                    for index, value in zip(candidates, values)
                }
                threshold_by_layer[layer_id] = float(threshold.item())
            else:
                utilization_by_layer[layer_id] = {}
                threshold_by_layer[layer_id] = None

            selected_by_layer[layer_id] = selected
            for index in candidates:
                if index in selected:
                    state[index] = 2
                    layer.qsteer_expert_mask[index] = True
                else:
                    state[index] = 0
                    layer.qsteer_expert_mask[index] = False

            for module in self.modules_by_layer.get(layer_id, []):
                for index in candidates:
                    _set_expert_trainable(module, index, index in selected)

        self._sync(
            treat_probe_as_new=False,
            allow_probe=False,
            tag="expansion_probe.finalize",
        )
        self.finalized = True
        return {
            "observed_samples": int(self.observed_samples),
            "probe_samples": int(self.probe_samples),
            "beta_thr": float(self.beta_thr),
            "selected_by_layer": selected_by_layer,
            "utilization_by_layer": utilization_by_layer,
            "threshold_by_layer": threshold_by_layer,
        }


__all__ = [
    "QSTEERExpansionProbe",
    "_iter_moe_modules_by_layer",
    "_set_expert_trainable",
]
