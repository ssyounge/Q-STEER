import time

from transformers import TrainerCallback

from .debug import QSTEERDebugLogger


class QSTEERStepCallback(TrainerCallback):
    def __init__(self):
        self._substep = 0
        self._last_log_wall_time = None
        self._logged_optimizer_summary = False

    @staticmethod
    def _to_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _log_optimizer_summary_once(self, model=None, optimizer=None):
        if self._logged_optimizer_summary:
            return
        if model is None or optimizer is None:
            return

        name_by_id = {id(p): n for n, p in model.named_parameters()}
        groups = []
        trainable_numel = 0
        for idx, group in enumerate(getattr(optimizer, "param_groups", [])):
            params = [p for p in group.get("params", []) if p is not None]
            names = [name_by_id.get(id(p), "") for p in params]
            numel = int(sum(int(p.numel()) for p in params))
            trainable_numel += numel
            groups.append(
                {
                    "idx": int(idx),
                    "lr": self._to_float(group.get("lr", None)),
                    "weight_decay": self._to_float(group.get("weight_decay", None)),
                    "param_numel": numel,
                    "contains_controller": bool(any("_qsteer_controller" in n for n in names)),
                    "contains_icr_params": bool(any(".qsteer_uq" in n or ".qsteer_uk" in n for n in names)),
                    "contains_moe_lora": bool(any(".experts." in n for n in names)),
                }
            )

        QSTEERDebugLogger.log(
            "optim.param_group_summary",
            force=True,
            group_count=int(len(groups)),
            trainable_param_numel=int(trainable_numel),
            param_groups=groups,
        )
        self._logged_optimizer_summary = True

    def on_step_begin(self, args, state, control, **kwargs):
        self._substep = 0
        QSTEERDebugLogger.set_step(int(state.global_step) + 1)
        QSTEERDebugLogger.set_runtime(
            phase="train",
            epoch=self._to_float(getattr(state, "epoch", None)),
            substep=int(self._substep),
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._substep = 0
        self._last_log_wall_time = time.time()
        self._logged_optimizer_summary = False
        QSTEERDebugLogger.set_step(int(state.global_step))
        QSTEERDebugLogger.set_runtime(
            phase="train",
            epoch=self._to_float(getattr(state, "epoch", None)),
            substep=int(self._substep),
            is_probe=False,
        )
        self._log_optimizer_summary_once(model=kwargs.get("model"), optimizer=kwargs.get("optimizer"))

    def on_substep_end(self, args, state, control, **kwargs):
        self._substep += 1
        QSTEERDebugLogger.set_runtime(substep=int(self._substep))

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        now = time.time()
        step_time_ms = None
        if self._last_log_wall_time is not None:
            step_time_ms = max(0.0, (now - self._last_log_wall_time) * 1000.0)
        self._last_log_wall_time = now

        if "epoch" in logs:
            QSTEERDebugLogger.set_runtime(epoch=self._to_float(logs.get("epoch")))

        QSTEERDebugLogger.log(
            "train.step_health",
            loss=self._to_float(logs.get("loss")),
            lr=self._to_float(logs.get("learning_rate")),
            global_grad_norm=self._to_float(logs.get("grad_norm")),
            step_time_ms=self._to_float(step_time_ms),
            samples_per_sec=self._to_float(logs.get("train_samples_per_second")),
        )
