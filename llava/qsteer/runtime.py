"""Public runtime helpers for integrating Q-STEER with a frozen MLLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .core import (
    QSTEERClearCallback,
    QSTEERConfig,
    QSTEERExpansionProbe,
    QSTEERStepCallback,
    QSTEERTaskFinalizeCallback,
    enable_qsteer,
    validate_qsteer_runtime,
)


@dataclass(frozen=True)
class QSTEERRuntimeReport:
    info: dict[str, Any]
    errors: list[str]
    summary: str

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class QSTEERRuntimeSettings:
    expert_num: int = 32
    expert_init: int = 8
    topk_update: int = 2
    late_layer_count: int = 6
    icr_rank: int = 8
    icr_scale: float = 0.3
    controller_hidden: int = 256
    controller_layers: int = 2
    controller_dropout: float = 0.0
    g_temp: float = 1.0
    probe_samples: int = 128
    probe_slots: int = 2
    beta_thr: float = 0.5
    expansion_candidate_last_n: int = 6
    strict_q_only: bool = True
    drift_eps: float = 1e-8
    strict_gate_context: bool = True
    allow_uniform_gate_fallback: bool = False
    strict_runtime_check: bool = True
    require_moe_lora: bool = True

    def to_config(self) -> QSTEERConfig:
        return QSTEERConfig(
            expert_num=self.expert_num,
            expert_init=self.expert_init,
            topk_update=self.topk_update,
            late_layer_count=self.late_layer_count,
            icr_rank=self.icr_rank,
            icr_scale=self.icr_scale,
            controller_hidden=self.controller_hidden,
            controller_layers=self.controller_layers,
            controller_dropout=self.controller_dropout,
            g_temp=self.g_temp,
            probe_samples=self.probe_samples,
            probe_slots=self.probe_slots,
            beta_thr=self.beta_thr,
            expansion_candidate_last_n=self.expansion_candidate_last_n,
            strict_q_only=self.strict_q_only,
            drift_eps=self.drift_eps,
            strict_gate_context=self.strict_gate_context,
            allow_uniform_gate_fallback=self.allow_uniform_gate_fallback,
        )


def _summary(info: dict[str, Any]) -> str:
    return (
        f"late={info.get('late_layer_count', 0)} "
        f"patched={info.get('patched_layer_count', 0)} "
        f"icr={info.get('icr_param_layer_count', 0)} "
        f"moe_lora={info.get('moe_lora_layer_count', 0)}"
    )


def attach_qsteer_runtime(
    model,
    cfg: QSTEERConfig,
    *,
    strict: bool = True,
    require_moe_lora: bool = True,
) -> QSTEERRuntimeReport:
    enable_qsteer(model, cfg)
    info, errors = validate_qsteer_runtime(
        model,
        expected_late_layer_count=cfg.late_layer_count,
        require_moe_lora=require_moe_lora,
    )
    summary = _summary(info)
    if strict and errors:
        raise RuntimeError(
            f"Q-STEER runtime validation failed: {'; '.join(errors)} | {summary}"
        )
    return QSTEERRuntimeReport(info=info, errors=errors, summary=summary)


def attach_qsteer_runtime_with_settings(
    model,
    settings: QSTEERRuntimeSettings,
) -> QSTEERRuntimeReport:
    return attach_qsteer_runtime(
        model,
        settings.to_config(),
        strict=settings.strict_runtime_check,
        require_moe_lora=settings.require_moe_lora,
    )


def build_qsteer_expansion_probe(
    cfg: QSTEERConfig,
    *,
    task_index: int,
) -> QSTEERExpansionProbe | None:
    if int(task_index) <= 0:
        return None
    return QSTEERExpansionProbe(
        probe_samples=cfg.probe_samples,
        probe_slots=cfg.probe_slots,
        beta_thr=cfg.beta_thr,
        candidate_last_n=cfg.expansion_candidate_last_n,
    )


def build_qsteer_callbacks() -> list[Any]:
    """Callbacks used during main training and at the task boundary."""

    return [
        QSTEERStepCallback(),
        QSTEERTaskFinalizeCallback(),
        QSTEERClearCallback(),
    ]


__all__ = [
    "QSTEERRuntimeReport",
    "QSTEERRuntimeSettings",
    "attach_qsteer_runtime",
    "attach_qsteer_runtime_with_settings",
    "build_qsteer_callbacks",
    "build_qsteer_expansion_probe",
]
