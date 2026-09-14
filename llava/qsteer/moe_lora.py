"""Q-STEER's preallocated factorized MoE-LoRA adapter.

The forward pass uses the soft old/new mixture produced by QSTEERContext.
Direct expert-parameter gradients are restricted to normalized top-K routes
while all active experts retain their soft forward contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import torch
import torch.nn as nn

from .core.route_context import QSTEERContext


_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class QSTEERMOELoraConfig:
    """Configuration for Q-STEER's preallocated expert bank.

    expert_rank is the rank of each individual expert, matching r_lora in the
    paper. It is not divided by expert_num.
    """

    expert_num: int = 32
    expert_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    topk_update: int = 2
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    controlled_last_n: int = 6
    freeze_backbone: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.expert_num, bool) or int(self.expert_num) < 2:
            raise ValueError("expert_num must be an integer >= 2")
        if isinstance(self.expert_rank, bool) or int(self.expert_rank) < 1:
            raise ValueError("expert_rank must be a positive integer")
        if float(self.lora_alpha) <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= float(self.lora_dropout) < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if not 1 <= int(self.topk_update) <= int(self.expert_num):
            raise ValueError("topk_update must be in [1, expert_num]")
        if int(self.controlled_last_n) < 1:
            raise ValueError("controlled_last_n must be positive")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")


class QSTEERMOEExpert(nn.Module):
    """One rank-r LoRA expert, B(A(x))."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        nn.init.normal_(self.A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(inputs))



def _selection_mask(
    scores: torch.Tensor,
    *,
    k: int,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a per-token top-K mask restricted to active candidates."""

    if scores.numel() == 0 or int(k) <= 0 or not bool(candidate_mask.any()):
        return torch.zeros_like(scores, dtype=torch.bool)
    mask = candidate_mask.to(device=scores.device, dtype=torch.bool)
    mask = mask.view(*([1] * (scores.dim() - 1)), scores.size(-1))
    masked_scores = scores.masked_fill(~mask, float("-inf"))
    effective_k = min(int(k), int(candidate_mask.sum().item()))
    values, indices = torch.topk(masked_scores, effective_k, dim=-1)
    selected = torch.zeros_like(scores, dtype=torch.bool)
    selected.scatter_(-1, indices, torch.isfinite(values))
    return selected


class QSTEERMOELoraLinear(nn.Module):
    """Frozen linear transform plus a preallocated Q-STEER expert bank."""

    def __init__(
        self,
        base_layer: nn.Linear,
        config: QSTEERMOELoraConfig,
        *,
        layer_id: int | None = None,
        module_name: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("QSTEERMOELoraLinear requires torch.nn.Linear")
        self.base_layer = base_layer
        self.expert_num = int(config.expert_num)
        self.expert_rank = int(config.expert_rank)
        self.topk = int(config.topk_update)
        self.scaling = float(config.lora_alpha) / float(config.expert_rank)
        self.dropout = nn.Dropout(float(config.lora_dropout))
        self.experts = nn.ModuleList(
            QSTEERMOEExpert(
                base_layer.in_features,
                base_layer.out_features,
                self.expert_rank,
            )
            for _ in range(self.expert_num)
        )
        ref_weight = base_layer.weight
        expert_dtype = (
            ref_weight.dtype if ref_weight.dtype.is_floating_point else torch.float32
        )
        self.experts.to(device=ref_weight.device, dtype=expert_dtype)
        self.register_buffer(
            "qsteer_trainable_mask",
            torch.ones(self.expert_num, dtype=torch.bool, device=ref_weight.device),
        )
        self.qsteer_layer_id = None if layer_id is None else int(layer_id)
        self.qsteer_module_name = module_name
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return int(self.base_layer.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base_layer.out_features)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def set_expert_trainable(self, expert_idx: int, trainable: bool) -> None:
        expert_index = int(expert_idx)
        expert = self.experts[expert_index]
        enabled = bool(trainable)
        for parameter in expert.parameters():
            parameter.requires_grad_(True)
            # Preallocated slots stay optimizer-visible; the mask controls gradients.
            if not enabled and parameter.grad is not None:
                parameter.grad = None
        self.qsteer_trainable_mask[expert_index] = enabled


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        original_shape_2d = inputs.dim() == 2
        routed_inputs = inputs.unsqueeze(1) if original_shape_2d else inputs
        if routed_inputs.dim() != 3:
            raise ValueError("Q-STEER MoE-LoRA expects [B,T,D] or [B,D] inputs")

        result = self.base_layer(routed_inputs)
        gate = QSTEERContext.get_gate(
            routed_inputs,
            expert_num=self.expert_num,
            layer_id=self.qsteer_layer_id,
        )
        if gate.dim() == 2:
            gate = gate.unsqueeze(1)
        if gate.shape[:2] != routed_inputs.shape[:2] or gate.size(-1) != self.expert_num:
            raise RuntimeError(
                "Q-STEER gate shape mismatch: "
                f"gate={tuple(gate.shape)} inputs={tuple(routed_inputs.shape)}"
            )

        dropped = self.dropout(routed_inputs)
        trainable = self.qsteer_trainable_mask.to(device=gate.device)
        active = gate.detach().sum(dim=(0, 1)) > 0
        selected = _selection_mask(
            gate.detach().float(),
            k=self.topk,
            candidate_mask=active,
        )
        hard_gate = gate * selected.to(dtype=gate.dtype)
        hard_gate = hard_gate / hard_gate.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        for expert_idx, expert in enumerate(self.experts):
            soft_weight = gate[..., expert_idx]
            if not bool((soft_weight.detach() > 0).any()):
                continue
            branch_output = expert(dropped)
            detached_output = branch_output.detach()
            soft_weight = soft_weight.unsqueeze(-1).to(branch_output.dtype)
            if bool(trainable[expert_idx]) and bool(selected[..., expert_idx].any()):
                hard_weight = hard_gate[..., expert_idx].unsqueeze(-1).to(branch_output.dtype)
                contribution = (
                    hard_weight * branch_output
                    + (soft_weight - hard_weight) * detached_output
                )
            else:
                contribution = soft_weight * detached_output
            result = result + self.scaling * contribution

        result = result.to(dtype=inputs.dtype)
        return result.squeeze(1) if original_shape_2d else result


def _matches_target(module_name: str, targets: Iterable[str]) -> bool:
    return any(module_name == target or module_name.endswith(f".{target}") for target in targets)


def _layer_id_from_name(module_name: str) -> int | None:
    match = _LAYER_PATTERN.search(module_name)
    return None if match is None else int(match.group(1))


def inject_qsteer_moe_lora(
    model: nn.Module,
    config: QSTEERMOELoraConfig,
) -> list[str]:
    """Replace matching linear modules in-place and return their names."""
    if config.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    candidates: list[tuple[str, nn.Linear, int]] = []
    for module_name, module in model.named_modules():
        layer_id = _layer_id_from_name(module_name)
        if (
            layer_id is not None
            and isinstance(module, nn.Linear)
            and _matches_target(module_name, config.target_modules)
        ):
            candidates.append((module_name, module, layer_id))

    controlled_ids = set(
        sorted({layer_id for _, _, layer_id in candidates})[-int(config.controlled_last_n) :]
    )
    replacements = [item for item in candidates if item[2] in controlled_ids]

    replaced: list[str] = []
    for module_name, module, layer_id in replacements:
        parent_name, _, child_name = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        replacement = QSTEERMOELoraLinear(
            module,
            config,
            layer_id=layer_id,
            module_name=module_name,
        )
        setattr(parent, child_name, replacement)
        replaced.append(module_name)

    if not replaced:
        raise ValueError(
            "No target linear modules found for Q-STEER MoE-LoRA: "
            f"{config.target_modules}"
        )
    return replaced


class QSTEERMOELoraModel(nn.Module):
    """Thin wrapper that injects Q-STEER MoE-LoRA into an existing model."""

    def __init__(self, model: nn.Module, config: QSTEERMOELoraConfig) -> None:
        super().__init__()
        self.model = model
        self.qsteer_replaced_modules = inject_qsteer_moe_lora(self.model, config)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def get_base_model(self) -> nn.Module:
        return self.model


__all__ = [
    "QSTEERMOEExpert",
    "QSTEERMOELoraConfig",
    "QSTEERMOELoraLinear",
    "QSTEERMOELoraModel",
    "inject_qsteer_moe_lora",
]
