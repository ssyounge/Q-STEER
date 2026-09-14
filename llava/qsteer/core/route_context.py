"""Per-forward state for Q-STEER routing and attention steering."""

from __future__ import annotations

import threading
from typing import Any

import torch

_tls = threading.local()
_global: dict[str, Any] = {}

_ROUTE_FIELDS = ("z_old", "z_new", "alpha", "gamma", "lambda", "drift", "g")
_FORWARD_FIELDS = (
    *_ROUTE_FIELDS,
    "q_mask",
    "diagnostic_query_mask",
    "visual_token_mask",
    "answer_mask",
    "stage",
    "collect_attn_stats",
    "icr_enabled",
    "old_only_routing",
    "strict_gate_context",
    "allow_uniform_gate_fallback",
    "route_layer_slots",
    "late_layer_slots",
    "expert_masks",
    "expert_states",
    "old_masks",
    "new_masks",
    "attn_stats_buffer",
)


def _detached(value):
    if torch.is_tensor(value):
        return value.detach()
    if isinstance(value, dict):
        return {key: _detached(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_detached(item) for item in value]
    return value


class QSTEERContext:
    """Thread-local tensors used by injected Q-STEER modules."""

    @staticmethod
    def _set(name: str, value) -> None:
        setattr(_tls, name, value)
        _global[name] = _detached(value)

    @staticmethod
    def _get(name: str):
        if hasattr(_tls, name):
            return getattr(_tls, name)
        return _global.get(name)

    @staticmethod
    def clear() -> None:
        for name in _FORWARD_FIELDS:
            if hasattr(_tls, name):
                delattr(_tls, name)
            _global.pop(name, None)

    @staticmethod
    def set_stage(stage: str) -> None:
        if stage not in {"diagnostic", "main"}:
            raise ValueError("stage must be diagnostic or main")
        QSTEERContext._set("stage", stage)

    @staticmethod
    def get_stage() -> str:
        return str(QSTEERContext._get("stage") or "main")

    @staticmethod
    def set_q_mask(mask) -> None:
        QSTEERContext._set("q_mask", mask)

    @staticmethod
    def get_q_mask(device=None):
        mask = QSTEERContext._get("q_mask")
        return mask.to(device=device) if torch.is_tensor(mask) and device is not None else mask

    @staticmethod
    def set_diagnostic_query_mask(mask) -> None:
        QSTEERContext._set("diagnostic_query_mask", mask)

    @staticmethod
    def get_diagnostic_query_mask(device=None):
        mask = QSTEERContext._get("diagnostic_query_mask")
        return mask.to(device=device) if torch.is_tensor(mask) and device is not None else mask

    @staticmethod
    def set_visual_token_mask(mask) -> None:
        QSTEERContext._set("visual_token_mask", mask)

    @staticmethod
    def get_visual_token_mask(device=None):
        mask = QSTEERContext._get("visual_token_mask")
        return mask.to(device=device) if torch.is_tensor(mask) and device is not None else mask

    @staticmethod
    def set_answer_mask(mask) -> None:
        QSTEERContext._set("answer_mask", mask)

    @staticmethod
    def get_answer_mask(device=None):
        mask = QSTEERContext._get("answer_mask")
        return mask.to(device=device) if torch.is_tensor(mask) and device is not None else mask

    @staticmethod
    def set_collect_attn_stats(enabled: bool) -> None:
        QSTEERContext._set("collect_attn_stats", bool(enabled))
        if enabled:
            QSTEERContext._set("attn_stats_buffer", {})

    @staticmethod
    def collect_attn_stats_enabled() -> bool:
        return bool(QSTEERContext._get("collect_attn_stats"))

    @staticmethod
    def set_icr_enabled(enabled: bool) -> None:
        QSTEERContext._set("icr_enabled", bool(enabled))

    @staticmethod
    def is_icr_enabled() -> bool:
        value = QSTEERContext._get("icr_enabled")
        return True if value is None else bool(value)

    @staticmethod
    def set_old_only_routing(enabled: bool) -> None:
        QSTEERContext._set("old_only_routing", bool(enabled))

    @staticmethod
    def is_old_only_routing() -> bool:
        return bool(QSTEERContext._get("old_only_routing"))

    @staticmethod
    def set_gate_policy(
        *,
        strict_gate_context: bool,
        allow_uniform_gate_fallback: bool,
    ) -> None:
        QSTEERContext._set("strict_gate_context", bool(strict_gate_context))
        QSTEERContext._set(
            "allow_uniform_gate_fallback",
            bool(allow_uniform_gate_fallback),
        )

    @staticmethod
    def set_route_metadata(route_layer_slots=None, late_layer_slots=None) -> None:
        if route_layer_slots is not None:
            QSTEERContext._set(
                "route_layer_slots",
                {int(key): int(value) for key, value in dict(route_layer_slots).items()},
            )
        if late_layer_slots is not None:
            QSTEERContext._set(
                "late_layer_slots",
                {int(key): int(value) for key, value in dict(late_layer_slots).items()},
            )

    @staticmethod
    def set_expert_masks(masks) -> None:
        normalized = None
        if masks is not None:
            normalized = {
                int(layer_id): torch.as_tensor(mask).bool().view(-1)
                for layer_id, mask in dict(masks).items()
            }
        QSTEERContext._set("expert_masks", normalized)

    @staticmethod
    def set_slot_states(states) -> None:
        normalized = None
        if states is not None:
            normalized = {
                int(layer_id): torch.as_tensor(state).to(dtype=torch.int8).view(-1)
                for layer_id, state in dict(states).items()
            }
        QSTEERContext._set("expert_states", normalized)

    @staticmethod
    def set_old_new_masks(*, old_masks=None, new_masks=None) -> None:
        def normalize(values):
            if values is None:
                return None
            return {
                int(layer_id): torch.as_tensor(mask).bool().view(-1)
                for layer_id, mask in dict(values).items()
            }

        QSTEERContext._set("old_masks", normalize(old_masks))
        QSTEERContext._set("new_masks", normalize(new_masks))

    @staticmethod
    def clear_slot_state_metadata() -> None:
        for name in ("expert_masks", "expert_states", "old_masks", "new_masks"):
            QSTEERContext._set(name, None)

    @staticmethod
    def set_route_payload(payload: dict | None) -> None:
        if not isinstance(payload, dict):
            raise TypeError("route payload must be a dictionary")
        for name in _ROUTE_FIELDS:
            QSTEERContext._set(name, payload.get(name))
        QSTEERContext.set_route_metadata(
            route_layer_slots=payload.get("route_layer_slots"),
            late_layer_slots=payload.get("late_layer_slots"),
        )
        QSTEERContext.set_expert_masks(payload.get("expert_masks"))
        QSTEERContext.set_slot_states(payload.get("expert_states"))
        QSTEERContext.set_old_new_masks(
            old_masks=payload.get("old_masks"),
            new_masks=payload.get("new_masks"),
        )
        QSTEERContext.set_gate_policy(
            strict_gate_context=payload.get("strict_gate_context", True),
            allow_uniform_gate_fallback=payload.get(
                "allow_uniform_gate_fallback",
                False,
            ),
        )

    @staticmethod
    def _select_layer(value, slot: int | None):
        if not torch.is_tensor(value):
            return None
        if value.dim() == 2:
            return value
        if value.dim() == 3 and slot is not None and 0 <= int(slot) < value.size(1):
            return value[:, int(slot), :]
        return None

    @staticmethod
    def _to(value, *, device=None, dtype=None):
        if not torch.is_tensor(value):
            return None
        return value.to(device=device or value.device, dtype=dtype or value.dtype)

    @staticmethod
    def get_alpha(layer_slot: int, *, device=None, dtype=None):
        value = QSTEERContext._select_layer(
            QSTEERContext._get("alpha"),
            int(layer_slot),
        )
        return QSTEERContext._to(value, device=device, dtype=dtype)

    @staticmethod
    def get_gamma(layer_slot: int, *, device=None, dtype=None):
        value = QSTEERContext._select_layer(
            QSTEERContext._get("gamma"),
            int(layer_slot),
        )
        return QSTEERContext._to(value, device=device, dtype=dtype)

    @staticmethod
    def get_lambda(*, device=None, dtype=None):
        value = QSTEERContext._get("lambda")
        if torch.is_tensor(value) and value.dim() == 1:
            value = value.unsqueeze(-1)
        return QSTEERContext._to(value, device=device, dtype=dtype)

    @staticmethod
    def _mask_for_layer(name: str, layer_id: int, expert_num: int, device):
        masks = QSTEERContext._get(name)
        mask = masks.get(int(layer_id)) if isinstance(masks, dict) else None
        if mask is None and name in {"old_masks", "new_masks"}:
            states = QSTEERContext._get("expert_states")
            state = states.get(int(layer_id)) if isinstance(states, dict) else None
            if torch.is_tensor(state):
                mask = state == 3 if name == "old_masks" else ((state == 2) | (state == 1))
        if mask is None:
            return torch.zeros(expert_num, dtype=torch.bool, device=device)
        mask = torch.as_tensor(mask, device=device).bool().view(-1)
        if mask.numel() < expert_num:
            mask = torch.cat(
                [
                    mask,
                    torch.zeros(
                        expert_num - mask.numel(),
                        dtype=torch.bool,
                        device=device,
                    ),
                ]
            )
        return mask[:expert_num]

    @staticmethod
    def _masked_softmax(logits, mask: torch.Tensor):
        if not torch.is_tensor(logits) or not bool(mask.any()):
            return None
        masked = logits.float().masked_fill(~mask.view(1, -1), float("-inf"))
        return torch.softmax(masked, dim=-1).to(dtype=logits.dtype)

    @staticmethod
    def get_gate(x: torch.Tensor, expert_num: int, layer_id: int | None = None):
        if not torch.is_tensor(x) or x.dim() < 2:
            raise ValueError("Q-STEER gate input must include a batch and token dimension")
        if layer_id is None:
            raise ValueError("Q-STEER gate requires layer_id")

        route_slots = QSTEERContext._get("route_layer_slots")
        if not isinstance(route_slots, dict) or int(layer_id) not in route_slots:
            raise RuntimeError(f"No route slot registered for layer {layer_id}")
        slot = int(route_slots[int(layer_id)])
        z_old = QSTEERContext._select_layer(QSTEERContext._get("z_old"), slot)
        z_new = QSTEERContext._select_layer(QSTEERContext._get("z_new"), slot)
        if z_old is None and z_new is None:
            raise RuntimeError("Q-STEER route logits are missing from the current context")

        old_mask = QSTEERContext._mask_for_layer(
            "old_masks", int(layer_id), int(expert_num), x.device
        )
        new_mask = QSTEERContext._mask_for_layer(
            "new_masks", int(layer_id), int(expert_num), x.device
        )
        p_old = QSTEERContext._masked_softmax(
            QSTEERContext._to(z_old, device=x.device, dtype=x.dtype),
            old_mask,
        )
        p_new = QSTEERContext._masked_softmax(
            QSTEERContext._to(z_new, device=x.device, dtype=x.dtype),
            new_mask,
        )

        if QSTEERContext.is_old_only_routing():
            if p_old is None:
                raise RuntimeError("The diagnostic path requires an active OLD expert bank")
            mixture = p_old
        elif p_old is None and p_new is None:
            if not bool(QSTEERContext._get("allow_uniform_gate_fallback")):
                raise RuntimeError("No active Q-STEER experts are available")
            active = old_mask | new_mask
            if not bool(active.any()):
                raise RuntimeError("No active Q-STEER experts are available")
            mixture = active.float().view(1, -1).expand(x.size(0), -1)
            mixture = mixture / mixture.sum(dim=-1, keepdim=True)
            mixture = mixture.to(dtype=x.dtype)
        elif p_old is None:
            mixture = p_new
        elif p_new is None:
            mixture = p_old
        else:
            lam = QSTEERContext.get_lambda(device=x.device, dtype=x.dtype)
            if lam is None:
                raise RuntimeError("Factorized old/new routing requires lambda")
            if lam.size(0) == 1 and x.size(0) > 1:
                lam = lam.expand(x.size(0), 1)
            mixture = (1.0 - lam) * p_old + lam * p_new

        mixture = mixture[:, : int(expert_num)]
        mixture = mixture / mixture.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return mixture[:, None, :].expand(
            x.size(0),
            x.size(1),
            int(expert_num),
        )

    @staticmethod
    def push_attn_stat(layer_id: int, p_batch: torch.Tensor) -> None:
        if not QSTEERContext.collect_attn_stats_enabled():
            return
        if not torch.is_tensor(p_batch) or p_batch.numel() == 0:
            return
        stats = QSTEERContext._get("attn_stats_buffer")
        if not isinstance(stats, dict):
            stats = {}
        stats.setdefault(int(layer_id), []).append(p_batch.detach().float())
        QSTEERContext._set("attn_stats_buffer", stats)

    @staticmethod
    def pop_attn_stats() -> dict[int, torch.Tensor]:
        stats = QSTEERContext._get("attn_stats_buffer")
        output: dict[int, torch.Tensor] = {}
        if isinstance(stats, dict):
            for layer_id, values in stats.items():
                tensors = [
                    value
                    for value in values
                    if torch.is_tensor(value) and value.numel() > 0
                ]
                if not tensors:
                    continue
                max_keys = max(int(value.size(-1)) for value in tensors)
                aligned = []
                for value in tensors:
                    if value.size(-1) < max_keys:
                        value = torch.cat(
                            [
                                value,
                                torch.zeros(
                                    *value.shape[:-1],
                                    max_keys - value.size(-1),
                                    device=value.device,
                                    dtype=value.dtype,
                                ),
                            ],
                            dim=-1,
                        )
                    aligned.append(value[..., :max_keys])
                output[int(layer_id)] = torch.stack(aligned).mean(dim=0)
        QSTEERContext._set("attn_stats_buffer", {})
        return output


__all__ = ["QSTEERContext"]
