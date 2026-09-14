import json
import os
import time
from dataclasses import dataclass

import torch


def _env_str(*names: str, default: str = "") -> str:
    for name in names:
        raw = os.getenv(name, None)
        if raw is None:
            continue
        value = str(raw).strip()
        if value:
            return value
    return str(default)


def _env_bool(*names: str, default: bool = False) -> bool:
    for name in names:
        raw = os.getenv(name, None)
        if raw is None:
            continue
        value = str(raw).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
    return bool(default)


def _is_rank0():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except Exception:
        pass
    return int(os.getenv("RANK", "0")) == 0 and int(os.getenv("LOCAL_RANK", "0")) == 0


def _to_float(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach()
        if x.numel() == 0:
            return None
        return float(x.float().mean().item())
    try:
        return float(x)
    except Exception:
        return None


def _stats(x, prefix):
    if x is None:
        return {}
    if not torch.is_tensor(x):
        return {prefix: _to_float(x)}
    x = x.detach()
    if x.numel() == 0:
        return {}
    xf = x.float()
    finite = torch.isfinite(xf)
    return {
        f"{prefix}.mean": float(xf.mean().item()),
        f"{prefix}.abs_mean": float(xf.abs().mean().item()),
        f"{prefix}.std": float(xf.std(unbiased=False).item()),
        f"{prefix}.max": float(xf.max().item()),
        f"{prefix}.min": float(xf.min().item()),
        f"{prefix}.finite_ratio": float(finite.float().mean().item()),
    }


@dataclass
class QSTEERDebugConfig:
    enabled: bool = False
    interval: int = 50
    probe_interval: int = 10
    anchor_enabled: bool = False
    path: str = _env_str("QSTEER_DEBUG_PATH", default="")


class QSTEERDebugLogger:
    _cfg = QSTEERDebugConfig()
    _step = 0
    _path_resolved = None
    _t0 = time.time()
    _forced_event_last_step = {}
    _runtime = {
        "rank": 0,
        "world_size": 1,
        "phase": "train",
        "epoch": None,
        "substep": None,
        "is_probe": None,
        "is_decode_step": None,
        "job_id": os.getenv("SLURM_JOB_ID", None),
    }

    @staticmethod
    def _env_int(*names: str, default: int) -> int:
        for name in names:
            raw = os.getenv(name, "").strip()
            if not raw:
                continue
            try:
                val = int(raw)
            except Exception:
                continue
            return int(default) if val <= 0 else val
        return int(default)

    @staticmethod
    def _jsonify(value):
        if value is None:
            return None
        if isinstance(value, (bool, int, float, str)):
            return value
        if torch.is_tensor(value):
            # Tensor payloads should be passed via _stats; fallback to scalar/list here.
            if value.numel() == 1:
                return float(value.detach().float().item())
            return [float(x) for x in value.detach().float().reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            return [QSTEERDebugLogger._jsonify(v) for v in value]
        if isinstance(value, dict):
            return {str(k): QSTEERDebugLogger._jsonify(v) for k, v in value.items()}
        return str(value)

    @classmethod
    def _compact_value(cls, value, depth: int = 0):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if torch.is_tensor(value):
            tensor = value.detach().float()
            if tensor.numel() == 0:
                return {"shape": [int(x) for x in value.shape], "empty": True}
            if tensor.numel() == 1:
                return float(tensor.item())
            return {
                "shape": [int(x) for x in value.shape],
                "mean": float(tensor.mean().item()),
                "abs_mean": float(tensor.abs().mean().item()),
                "std": float(tensor.std(unbiased=False).item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
            }
        if isinstance(value, dict):
            out = {}
            items = list(value.items())
            for idx, (key, val) in enumerate(items):
                if idx >= 12:
                    out["..."] = f"+{len(items) - idx} more"
                    break
                out[str(key)] = cls._compact_value(val, depth=depth + 1)
            return out
        if isinstance(value, (list, tuple)):
            values = list(value)
            out = [cls._compact_value(v, depth=depth + 1) for v in values[:12]]
            if len(values) > 12:
                out.append(f"...(+{len(values) - 12})")
            return out
        return str(value)

    @classmethod
    def configure(cls, enabled: bool, interval: int, output_dir: str):
        anchor_enabled = _env_bool(
            "QSTEER_DEBUG_ANCHOR",
            default=False,
        )
        if anchor_enabled:
            enabled = True
            interval = 1
        interval = max(1, int(interval))
        probe_interval = cls._env_int(
            "QSTEER_DEBUG_PROBE_INTERVAL",
            default=max(1, interval // 5),
        )
        path_override = _env_str("QSTEER_DEBUG_PATH", default="")
        cls._cfg = QSTEERDebugConfig(
            enabled=bool(enabled),
            interval=interval,
            probe_interval=probe_interval,
            anchor_enabled=bool(anchor_enabled),
            path=path_override,
        )
        cls._path_resolved = cls._cfg.path or None
        cls._t0 = time.time()
        cls._forced_event_last_step = {}
        cls._runtime["rank"] = int(os.getenv("RANK", "0") or 0)
        cls._runtime["world_size"] = int(os.getenv("WORLD_SIZE", "1") or 1)
        cls._runtime["phase"] = "train"
        cls._runtime["job_id"] = os.getenv("SLURM_JOB_ID", None)

    @classmethod
    def set_step(cls, step: int):
        cls._step = int(step)

    @classmethod
    def set_runtime(cls, **kwargs):
        for k, v in kwargs.items():
            if k in cls._runtime:
                cls._runtime[k] = v

    @classmethod
    def _current_interval(cls) -> int:
        if bool(cls._runtime.get("is_probe", False)):
            return max(1, int(cls._cfg.probe_interval))
        return max(1, int(cls._cfg.interval))

    @classmethod
    def anchor_enabled(cls) -> bool:
        return bool(getattr(cls._cfg, "anchor_enabled", False))

    @classmethod
    def log(cls, event: str, force: bool = False, **fields):
        if not cls._cfg.enabled:
            return
        if not _is_rank0():
            return
        if force:
            last_step = cls._forced_event_last_step.get(event)
            if last_step == cls._step:
                return
            cls._forced_event_last_step[event] = cls._step

        rec = {
            "t": round(time.time() - cls._t0, 3),
            "step": cls._step,
            "event": event,
        }
        rec.update(
            {
                "rank": int(cls._runtime.get("rank", 0)),
                "world_size": int(cls._runtime.get("world_size", 1)),
                "phase": cls._runtime.get("phase", "train"),
                "epoch": cls._runtime.get("epoch", None),
                "substep": cls._runtime.get("substep", None),
                "is_probe": cls._runtime.get("is_probe", None),
                "is_decode_step": cls._runtime.get("is_decode_step", None),
                "job_id": cls._runtime.get("job_id", None),
            }
        )
        if not force and cls._step % cls._current_interval() != 0:
            return
        for k, v in fields.items():
            if torch.is_tensor(v):
                rec.update(_stats(v, k))
            else:
                rec[k] = cls._jsonify(v)

        if not cls._path_resolved:
            return
        try:
            dirname = os.path.dirname(cls._path_resolved)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(cls._path_resolved, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            return

    @classmethod
    def anchor(cls, event: str, **fields):
        if cls._cfg.enabled:
            cls.log(event, force=False, **fields)
        if not cls.anchor_enabled():
            return
        if not _is_rank0():
            return

        rec = {
            "step": int(cls._step),
            "event": str(event),
            "phase": cls._runtime.get("phase", "train"),
            "epoch": cls._runtime.get("epoch", None),
            "substep": cls._runtime.get("substep", None),
            "is_probe": cls._runtime.get("is_probe", None),
            "is_decode_step": cls._runtime.get("is_decode_step", None),
        }
        for key, value in fields.items():
            rec[str(key)] = cls._compact_value(value)
        print(
            f"[qsteer][anchor] {json.dumps(rec, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )
