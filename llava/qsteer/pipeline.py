"""Paper-level two-stage execution for Q-STEER batches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import torch

from .core.route_context import QSTEERContext
from .core.state_sync import collect_qsteer_layer_state


def _clear_context_on_error(method):
    @wraps(method)
    def guarded(*args, **kwargs):
        QSTEERContext.clear()
        try:
            return method(*args, **kwargs)
        except BaseException:
            QSTEERContext.clear()
            raise
    return guarded


@dataclass(frozen=True)
class QSTEERBatchMasks:
    """Token masks required by the question, diagnostic, and main passes."""

    question: torch.Tensor
    diagnostic_query: torch.Tensor
    visual_tokens: torch.Tensor
    answer_queries: torch.Tensor | None = None

    def validate(
        self,
        *,
        batch_size: int,
        question_length: int,
        require_answer_queries: bool = False,
    ) -> None:
        required = {
            "question": self.question,
            "diagnostic_query": self.diagnostic_query,
            "visual_tokens": self.visual_tokens,
        }
        for name, mask in required.items():
            if not torch.is_tensor(mask) or mask.dim() != 2:
                raise ValueError(f"{name} mask must be a rank-2 tensor")
            if tuple(mask.shape) != (int(batch_size), int(question_length)):
                raise ValueError(
                    f"{name} mask expected {(int(batch_size), int(question_length))}, "
                    f"actual {tuple(mask.shape)}"
                )
            if not bool(((mask == 0) | (mask == 1)).all()):
                raise ValueError(f"{name} mask must contain only finite 0/1 values")
            if not bool((mask.bool().sum(dim=1) > 0).all()):
                raise ValueError(f"{name} mask must be non-empty for every sample")
        if int(self.question.size(1)) != int(question_length):
            raise ValueError("question mask length must match prompt_embeddings")
        diagnostic_count = self.diagnostic_query.bool().sum(dim=1)
        if not bool((diagnostic_count == 1).all()):
            raise ValueError("diagnostic_query must select exactly one answer-start position")
        if require_answer_queries and self.answer_queries is None:
            raise ValueError("answer_queries is required for a main forward")
        if self.answer_queries is not None:
            if not torch.is_tensor(self.answer_queries) or self.answer_queries.dim() != 2:
                raise ValueError("answer_queries mask must be a rank-2 tensor")
            if int(self.answer_queries.size(0)) != int(batch_size):
                raise ValueError("answer_queries mask batch size does not match embeddings")
            if not bool(((self.answer_queries == 0) | (self.answer_queries == 1)).all()):
                raise ValueError("answer_queries mask must contain only finite 0/1 values")
            if not bool((self.answer_queries.bool().sum(dim=1) > 0).all()):
                raise ValueError("answer_queries must be non-empty for every sample")


@dataclass(frozen=True)
class QSTEERPreparedBatch:
    route_payload: dict[str, Any]
    drift: torch.Tensor
    attention_by_layer: dict[int, torch.Tensor]


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


class QSTEERTwoStageRunner:
    """Build controller inputs from a prompt-only diagnostic forward."""

    def __init__(self, model) -> None:
        self.model = model
        self.lm = _unwrap_lm(model)
        self.layers = _get_layers(self.lm)
        self.payload = getattr(self.lm, "_qsteer", None)
        if not isinstance(self.payload, dict):
            raise RuntimeError("Attach the Q-STEER runtime before creating the runner.")
        self.controller = self.payload.get("controller")
        self.drift_buffer = self.payload.get("drift_buffer")
        self.cfg = self.payload.get("cfg")
        if self.controller is None or self.drift_buffer is None or self.cfg is None:
            raise RuntimeError("The attached Q-STEER runtime payload is incomplete.")

    def _layer_state(self) -> dict[str, Any]:
        return collect_qsteer_layer_state(
            self.layers,
            treat_probe_as_new=True,
        )

    def _route_metadata(self) -> dict[str, Any]:
        return {
            "route_layer_slots": self.payload.get("route_layer_slots", {}),
            "late_layer_slots": self.payload.get("late_layer_slots", {}),
            "strict_gate_context": bool(self.cfg.strict_gate_context),
            "allow_uniform_gate_fallback": bool(self.cfg.allow_uniform_gate_fallback),
        }

    def _attach_layer_state(self, route_payload: dict[str, Any]) -> None:
        state = self._layer_state()
        route_payload["expert_masks"] = state["expert_masks"]
        route_payload["expert_states"] = state["slot_states"]
        route_payload["old_masks"] = state["old_masks"]
        route_payload["new_masks"] = state["new_masks"]

    @staticmethod
    def _normalize_drift(
        drift: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(drift):
            return torch.zeros(batch_size, device=device, dtype=dtype)
        drift = drift.to(device=device, dtype=dtype)
        if drift.dim() == 0:
            return drift.view(1).expand(batch_size)
        drift = drift.reshape(drift.size(0), -1)[:, 0]
        if drift.size(0) == 1 and batch_size > 1:
            return drift.expand(batch_size)
        if drift.size(0) != batch_size:
            raise RuntimeError("Drift batch size does not match prompt embeddings")
        return drift

    @torch.no_grad()
    def _diagnostic(
        self,
        prompt_embeddings: torch.Tensor,
        masks: QSTEERBatchMasks,
        prompt_forward: Callable[[], Any],
        *,
        accumulate_reference: bool,
        record_task_drift: bool,
        require_answer_queries: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        if prompt_embeddings.dim() != 3:
            raise ValueError("prompt_embeddings must have shape [batch, tokens, hidden]")
        batch_size, prompt_length = prompt_embeddings.shape[:2]
        masks.validate(
            batch_size=batch_size,
            question_length=prompt_length,
            require_answer_queries=require_answer_queries,
        )

        pooled = self.controller.pool_context(
            prompt_embeddings.detach(),
            q_mask=masks.question,
        )
        z_old = self.controller.forward_old(pooled)
        diagnostic_payload = {
            "z_old": z_old,
            "z_new": z_old,
            "lambda": torch.zeros(
                batch_size,
                1,
                device=z_old.device,
                dtype=z_old.dtype,
            ),
            **self._route_metadata(),
        }
        self._attach_layer_state(diagnostic_payload)

        QSTEERContext.clear()
        QSTEERContext.set_stage("diagnostic")
        QSTEERContext.set_q_mask(masks.question)
        QSTEERContext.set_diagnostic_query_mask(masks.diagnostic_query)
        QSTEERContext.set_visual_token_mask(masks.visual_tokens)
        QSTEERContext.set_answer_mask(None)
        QSTEERContext.set_collect_attn_stats(True)
        QSTEERContext.set_icr_enabled(False)
        QSTEERContext.set_old_only_routing(True)
        QSTEERContext.set_gate_policy(
            strict_gate_context=bool(self.cfg.strict_gate_context),
            allow_uniform_gate_fallback=bool(self.cfg.allow_uniform_gate_fallback),
        )
        QSTEERContext.set_route_payload(diagnostic_payload)

        with torch.no_grad():
            prompt_forward()
        attention_by_layer = QSTEERContext.pop_attn_stats()
        expected_layers = set(self.payload["late_layer_ids"])
        if set(attention_by_layer) != expected_layers:
            raise RuntimeError(
                f"Q-STEER diagnostic attention layers: expected {sorted(expected_layers)}, "
                f"actual {sorted(attention_by_layer)}; prompt_forward must run the attached model."
            )
        drift = self._normalize_drift(
            self.drift_buffer.compute_batch_drift(attention_by_layer),
            batch_size=batch_size,
            device=pooled.device,
            dtype=pooled.dtype,
        )

        if accumulate_reference:
            for layer_id, distribution in attention_by_layer.items():
                self.drift_buffer.update_task_running(layer_id, distribution)
        if record_task_drift:
            self.drift_buffer.update_task_drift(drift)
        return pooled, drift, attention_by_layer

    @_clear_context_on_error
    def collect_initial_reference(
        self,
        prompt_embeddings: torch.Tensor,
        masks: QSTEERBatchMasks,
        prompt_forward: Callable[[], Any],
    ) -> dict[int, torch.Tensor]:
        """Accumulate frozen-backbone prompt summaries before task one."""

        if self.drift_buffer.has_reference():
            raise RuntimeError("The initial attention reference already exists.")
        _, _, attention = self._diagnostic(
            prompt_embeddings,
            masks,
            prompt_forward,
            accumulate_reference=True,
            record_task_drift=False,
            require_answer_queries=False,
        )
        QSTEERContext.clear()
        return attention

    def finalize_initial_reference(self) -> dict[int, torch.Tensor]:
        if self.drift_buffer.has_reference():
            raise RuntimeError("The initial attention reference already exists.")
        reference = self.drift_buffer.finalize_task_reference()
        if not reference:
            raise RuntimeError("No diagnostic attention was collected for the reference.")
        return reference

    @_clear_context_on_error
    def prepare_main_context(
        self,
        prompt_embeddings: torch.Tensor,
        masks: QSTEERBatchMasks,
        prompt_forward: Callable[[], Any],
        *,
        training: bool,
        require_answer_queries: bool = True,
    ) -> QSTEERPreparedBatch:
        """Run the diagnostic pass and install the main-pass routing context."""

        pooled, drift, attention = self._diagnostic(
            prompt_embeddings,
            masks,
            prompt_forward,
            accumulate_reference=bool(training),
            record_task_drift=bool(training),
            require_answer_queries=bool(require_answer_queries),
        )
        route_payload = self.controller(pooled, drift)
        route_payload["drift"] = drift
        route_payload["g"] = route_payload["z_new"]
        route_payload.update(self._route_metadata())
        self._attach_layer_state(route_payload)

        QSTEERContext.clear()
        QSTEERContext.set_stage("main")
        QSTEERContext.set_q_mask(masks.question)
        QSTEERContext.set_answer_mask(masks.answer_queries)
        QSTEERContext.set_collect_attn_stats(False)
        QSTEERContext.set_icr_enabled(bool(require_answer_queries))
        QSTEERContext.set_old_only_routing(False)
        QSTEERContext.set_gate_policy(
            strict_gate_context=bool(self.cfg.strict_gate_context),
            allow_uniform_gate_fallback=bool(self.cfg.allow_uniform_gate_fallback),
        )
        QSTEERContext.set_route_payload(route_payload)
        return QSTEERPreparedBatch(
            route_payload=route_payload,
            drift=drift,
            attention_by_layer=attention,
        )

    @staticmethod
    def clear() -> None:
        QSTEERContext.clear()


__all__ = [
    "QSTEERBatchMasks",
    "QSTEERPreparedBatch",
    "QSTEERTwoStageRunner",
]
