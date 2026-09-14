from __future__ import annotations

import math
import re
import types
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

from .config import QSTEERConfig
from .route_context import QSTEERContext
from .controller import QSTEERController
from .debug import QSTEERDebugLogger as DL
from .drift_buffer import QSTEERDriftBuffer
from .state_sync import (
    assert_qsteer_layer_state,
    sync_qsteer_context_and_payload,
    validate_qsteer_layer_state,
)
from .expansion import _iter_moe_modules_by_layer, _set_expert_trainable

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def _unwrap_lm(model):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model
    if hasattr(model, "layers"):
        return model
    raise ValueError("Could not locate LLaMA layers on the provided model.")


def _get_layers(lm):
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    return lm.layers

_QSTEER_PARAMETER_MARKERS = (
    ".experts.",
    "_qsteer_controller",
    ".qsteer_uq",
    ".qsteer_uk",
)


def _is_qsteer_parameter(name: str) -> bool:
    return any(marker in str(name) for marker in _QSTEER_PARAMETER_MARKERS)



def _ensure_icr_params(attn: nn.Module, icr_rank: int):
    if hasattr(attn, "qsteer_uq") and hasattr(attn, "qsteer_uk"):
        attn.qsteer_uq.requires_grad_(True)
        attn.qsteer_uk.requires_grad_(True)
        return
    ref = attn.q_proj.weight
    ref_dtype = ref.dtype if ref.dtype.is_floating_point else torch.float32
    d_model = int(attn.q_proj.out_features)
    uq = nn.Parameter(torch.empty(d_model, icr_rank, device=ref.device, dtype=ref_dtype))
    uk = nn.Parameter(torch.empty(d_model, icr_rank, device=ref.device, dtype=ref_dtype))
    nn.init.normal_(uq, mean=0.0, std=0.02)
    nn.init.normal_(uk, mean=0.0, std=0.02)
    attn.register_parameter("qsteer_uq", uq)
    attn.register_parameter("qsteer_uk", uk)


def _slot_state_summary(layers):
    summary = {}
    for layer_id, layer in enumerate(layers):
        state = getattr(layer, "qsteer_slot_state", None)
        if state is None or (not torch.is_tensor(state)):
            continue
        view = state.detach().to(dtype=torch.int8).view(-1)
        summary[int(layer_id)] = {
            "inactive": int((view == 0).sum().item()),
            "probe": int((view == 1).sum().item()),
            "new": int((view == 2).sum().item()),
            "old": int((view == 3).sum().item()),
        }
    return summary


def _require_mask_2d(mask, target_len, *, batch_size, device, layer_id, name):
    expected = (batch_size, target_len)
    actual = tuple(mask.shape) if torch.is_tensor(mask) else type(mask).__name__
    if not torch.is_tensor(mask) or tuple(mask.shape) != expected:
        raise ValueError(
            f"Q-STEER layer {layer_id} {name}: expected {expected}, actual {actual}"
        )
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"Q-STEER layer {layer_id} {name}: expected finite 0/1, actual invalid values")
    return mask.to(device=device, dtype=torch.bool)


def _collect_attention_stats(
    *, attn_logits: torch.Tensor, layer_id: int, q_len: int, query_states: torch.Tensor,
):
    batch_size = int(attn_logits.size(0))
    query_rows = _require_mask_2d(
        QSTEERContext.get_diagnostic_query_mask(), q_len, batch_size=batch_size,
        device=query_states.device, layer_id=layer_id, name="diagnostic_query",
    )
    visual_keys = _require_mask_2d(
        QSTEERContext.get_visual_token_mask(), int(attn_logits.size(-1)),
        batch_size=batch_size, device=query_states.device, layer_id=layer_id,
        name="visual_tokens",
    )
    if not bool((query_rows.sum(1) == 1).all()) or not bool((visual_keys.sum(1) > 0).all()):
        raise ValueError(
            f"Q-STEER layer {layer_id}: expected one diagnostic query and nonempty visual keys, "
            f"actual query_counts={query_rows.sum(1).tolist()}, visual_counts={visual_keys.sum(1).tolist()}"
        )
    q_count = int(query_rows.sum().item())
    visual_count = int(visual_keys.sum().item())

    with torch.no_grad():
        prob = torch.softmax(attn_logits.float(), dim=-1)  # [B, H, Q, K]
        prob = torch.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        prob = prob.mean(dim=1)  # [B, Q, K]
        query_weights = query_rows.to(dtype=prob.dtype)
        query_denom = query_weights.sum(dim=1, keepdim=True)
        p_batch = (prob * query_weights.unsqueeze(-1)).sum(dim=1) / query_denom
        p_batch = p_batch * visual_keys.to(dtype=p_batch.dtype)
        visual_mass = p_batch.sum(dim=-1, keepdim=True)
        invalid_rows = (~torch.isfinite(visual_mass)) | (visual_mass <= 0)
        if bool(invalid_rows.any()):
            raise RuntimeError(
                f"Q-STEER layer {layer_id}: expected positive visual attention mass, "
                f"actual invalid rows={int(invalid_rows.sum().item())}"
            )
        p_batch = p_batch / visual_mass
        QSTEERContext.push_attn_stat(layer_id=layer_id, p_batch=p_batch.detach())
        DL.log(
            "qsteer.attn_collect",
            layer_id=int(layer_id),
            q_rows=int(q_count),
            visual_keys=int(visual_count),
            p_shape=[int(x) for x in p_batch.shape],
            skipped=False,
        )


def _build_qsteer_attention_forward(attn: nn.Module, layer_id: int, layer_slot: int, cfg: QSTEERConfig):
    # transformers>=4.57 LlamaAttention returns (attn_output, attn_weights)
    # and expects cache through `past_key_values` object, not tuple return.
    original_forward = getattr(attn, "_qsteer_original_forward", attn.forward)
    try:
        original_params = inspect.signature(original_forward).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Cannot inspect attention signature for Q-STEER layer {layer_id}") from exc
    uses_modern_attention_api = "past_key_values" in original_params

    def qsteer_attention_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        # Accept both legacy `past_key_value` and modern `past_key_values` call sites.
        if past_key_value is None:
            if "past_key_values" in kwargs:
                past_key_value = kwargs.pop("past_key_values")
            elif "past_key_value" in kwargs:
                past_key_value = kwargs.pop("past_key_value")

        bsz, q_len, _ = hidden_states.size()
        num_heads = getattr(self, "num_heads", self.q_proj.out_features // self.head_dim)
        num_key_value_heads = getattr(
            self, "num_key_value_heads", self.k_proj.out_features // self.head_dim
        )

        expected_projection = (num_heads * self.head_dim, num_key_value_heads * self.head_dim)
        actual_projection = (self.q_proj.out_features, self.k_proj.out_features, self.v_proj.out_features)
        if (
            num_heads <= 0 or num_key_value_heads <= 0
            or num_heads % num_key_value_heads != 0
            or self.num_key_value_groups != num_heads // num_key_value_heads
            or actual_projection != (expected_projection[0], expected_projection[1], expected_projection[1])
        ):
            raise RuntimeError(
                f"Q-STEER layer {layer_id} head/projection configuration: expected "
                f"Q={num_heads}, KV={num_key_value_heads}, head_dim={self.head_dim}, "
                f"groups={num_heads // max(num_key_value_heads, 1)}, "
                f"projection widths={expected_projection}; actual "
                f"groups={self.num_key_value_groups}, widths={actual_projection}"
            )

        query_states = self.q_proj(hidden_states).view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            bsz, q_len, num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            bsz, q_len, num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        has_cache_update = hasattr(past_key_value, "update")

        if position_embeddings is None:
            if position_ids is not None:
                try:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                except TypeError:
                    kv_seq_len = key_states.shape[-2]
                    if past_key_value is not None and not has_cache_update:
                        kv_seq_len += past_key_value[0].shape[-2]
                    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            else:
                kv_seq_len = key_states.shape[-2]
                if past_key_value is not None and not has_cache_update:
                    kv_seq_len += past_key_value[0].shape[-2]
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            if has_cache_update:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, self.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if has_cache_update:
            past_key_value_out = past_key_value
        else:
            past_key_value_out = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        expected_heads = int(query_states.size(1))
        if key_states.size(1) != expected_heads or value_states.size(1) != expected_heads:
            raise RuntimeError(
                f"Q-STEER layer {layer_id} KV heads: expected {expected_heads}, "
                f"actual K={tuple(key_states.shape)}, V={tuple(value_states.shape)}, "
                f"groups={self.num_key_value_groups}, head_dim={self.head_dim}"
            )

        runtime_num_heads = int(query_states.size(1))

        attn_logits = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)

        kv_len_now = key_states.size(-2)
        past_len = max(kv_len_now - q_len, 0)
        if cache_position is not None and torch.is_tensor(cache_position):
            flat_cache_pos = cache_position.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
            if flat_cache_pos.numel() == q_len:
                q_pos = flat_cache_pos
            else:
                raise ValueError(
                    f"Q-STEER layer {layer_id} cache_position: expected {q_len} positions, "
                    f"actual {flat_cache_pos.numel()}"
                )
        else:
            q_pos = torch.arange(q_len, device=hidden_states.device, dtype=torch.long) + past_len
        k_pos = torch.arange(kv_len_now, device=hidden_states.device, dtype=torch.long)
        causal_mask = k_pos.view(1, 1, 1, -1) > q_pos.view(1, 1, -1, 1)
        attn_logits = attn_logits.masked_fill(causal_mask, torch.finfo(attn_logits.dtype).min)

        if attention_mask is not None:
            mask_min = torch.finfo(attn_logits.dtype).min
            shape = tuple(attention_mask.shape)
            if attention_mask.dim() == 2:
                if shape != (bsz, kv_len_now):
                    raise ValueError(
                        f"Q-STEER layer {layer_id} attention_mask: expected {(bsz, kv_len_now)}, actual {shape}"
                    )
                if not bool(((attention_mask == 0) | (attention_mask == 1)).all()):
                    raise ValueError(
                        f"Q-STEER layer {layer_id} attention_mask: expected finite 0/1, actual invalid values"
                    )
                attn_logits = attn_logits.masked_fill(
                    ~attention_mask.to(device=attn_logits.device, dtype=torch.bool)[:, None, None, :], mask_min
                )
            elif attention_mask.dim() == 4:
                if (shape[0] not in (1, bsz) or shape[1] not in (1, runtime_num_heads)
                        or shape[2] not in (1, q_len) or shape[3] < kv_len_now):
                    raise ValueError(
                        f"Q-STEER layer {layer_id} attention_mask: expected "
                        f"[1 or {bsz}, 1 or {runtime_num_heads}, 1 or {q_len}, >= {kv_len_now}], actual {shape}"
                    )
                # Upstream can allocate extra key columns. Cache keys start at 0.
                mask_4d = attention_mask[..., :kv_len_now].to(device=attn_logits.device)
                if mask_4d.dtype == torch.bool:
                    attn_logits = attn_logits.masked_fill(~mask_4d, mask_min)
                else:
                    if bool(torch.isnan(mask_4d).any() or torch.isposinf(mask_4d).any()):
                        raise ValueError(f"Q-STEER layer {layer_id} attention_mask contains NaN/+inf")
                    attn_logits = attn_logits + mask_4d.to(dtype=attn_logits.dtype)
            else:
                raise ValueError(
                    f"Q-STEER layer {layer_id} attention_mask: expected rank 2 or 4, actual {shape}"
                )
        # A fully masked row has no defined distribution; finite-min softmax
        # would silently invent uniform attention (including forbidden keys).
        blocked = causal_mask | (attn_logits <= torch.finfo(attn_logits.dtype).min)
        if bool(blocked.all(dim=-1).any()):
            raise ValueError(f"Q-STEER layer {layer_id}: fully masked attention query row")

        stage = QSTEERContext.get_stage()
        if stage == "diagnostic" and QSTEERContext.collect_attn_stats_enabled():
            _collect_attention_stats(
                attn_logits=attn_logits,
                layer_id=int(layer_id),
                q_len=int(q_len),
                query_states=query_states,
            )

        if stage == "main" and QSTEERContext.is_icr_enabled():
            alpha = QSTEERContext.get_alpha(layer_slot, device=query_states.device, dtype=query_states.dtype)
            gamma = QSTEERContext.get_gamma(layer_slot, device=query_states.device, dtype=query_states.dtype)
            lam = QSTEERContext.get_lambda(device=query_states.device, dtype=query_states.dtype)
            for name, value, expected in (
                ("alpha", alpha, (bsz, cfg.icr_rank)),
                ("gamma", gamma, (bsz, runtime_num_heads)),
                ("lambda", lam, (bsz, 1)),
            ):
                actual = tuple(value.shape) if torch.is_tensor(value) else None
                # A single per-sample gamma is an intentional head broadcast.
                valid_gamma = name == "gamma" and actual == (bsz, 1)
                if actual != expected and not valid_gamma:
                    raise RuntimeError(
                        f"Q-STEER layer {layer_id} {name}: expected {expected}, actual {actual}"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise RuntimeError(f"Q-STEER layer {layer_id} {name}: expected finite values, actual nonfinite")
            answer_mask = _require_mask_2d(
                QSTEERContext.get_answer_mask(), q_len, batch_size=bsz,
                device=query_states.device, layer_id=layer_id, name="answer_queries",
            )
            uq = self.qsteer_uq.to(dtype=query_states.dtype)
            uk = self.qsteer_uk.to(dtype=key_states.dtype)
            basis_shape = (runtime_num_heads * self.head_dim, cfg.icr_rank)
            if tuple(uq.shape) != basis_shape or tuple(uk.shape) != basis_shape:
                raise RuntimeError(
                    f"Q-STEER layer {layer_id} ICR bases: expected {basis_shape}, "
                    f"actual Uq={tuple(uq.shape)}, Uk={tuple(uk.shape)}"
                )
            uq_by_head = uq.reshape(runtime_num_heads, self.head_dim, cfg.icr_rank)
            uk_by_head = uk.reshape(runtime_num_heads, self.head_dim, cfg.icr_rank)
            q_icr = torch.einsum("bhqd,hdr->bhqr", query_states, uq_by_head)
            k_icr = torch.einsum("bhkd,hdr->bhkr", key_states, uk_by_head)
            delta_base = torch.einsum("bhqr,br,bhkr->bhqk", q_icr, alpha, k_icr)
            gamma_head = gamma.expand(bsz, runtime_num_heads)
            delta_a = (
                delta_base * gamma_head[:, :, None, None]
                * lam.reshape(bsz, 1, 1, 1) * float(cfg.icr_scale)
            )
            delta_a = delta_a * answer_mask[:, None, :, None].to(delta_a.dtype)
            attn_logits = attn_logits + delta_a
            DL.anchor(
                "qsteer.anchor.icr", layer_id=int(layer_id), layer_slot=int(layer_slot),
                stage=stage, lambda_mean=float(lam.detach().mean().item()),
                answer_query_count=int(answer_mask.sum().item()),
                delta_norm_mean=float(delta_a.detach().float().norm(dim=-1).mean().item()),
                q_len=int(q_len), k_len=int(kv_len_now), icr_scale=float(cfg.icr_scale),
            )

        attn_logits = attn_logits.masked_fill(blocked, torch.finfo(attn_logits.dtype).min)
        attn_weights = torch.softmax(attn_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_drop_p = getattr(self, "attention_dropout", 0.0)
        if hasattr(attn_drop_p, "p"):
            attn_drop_p = attn_drop_p.p
        attn_drop_p = float(attn_drop_p or 0.0)
        if self.training and attn_drop_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=attn_drop_p, training=True)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        if int(self.o_proj.in_features) != int(attn_output.size(-1)):
            raise RuntimeError(
                f"Q-STEER layer {layer_id} output projection: expected "
                f"{self.o_proj.in_features} input features, actual {attn_output.size(-1)}; "
                f"heads={runtime_num_heads}, head_dim={self.head_dim}"
            )
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        if uses_modern_attention_api:
            return attn_output, attn_weights
        return attn_output, attn_weights, past_key_value_out

    return types.MethodType(qsteer_attention_forward, attn)


def _set_moe_topk(model, topk_update: int):
    for module in model.modules():
        if hasattr(module, "topk") and hasattr(module, "expert_num") and hasattr(module, "experts"):
            module.topk = min(max(int(topk_update), 1), int(module.expert_num))


def _discover_route_layer_ids(model):
    route_layer_ids = []
    for module_name, module in model.named_modules():
        if not (
            hasattr(module, "experts")
            and hasattr(module, "expert_num")
            and hasattr(module, "topk")
        ):
            continue
        match = _LAYER_PATTERN.search(module_name)
        if match is None:
            continue
        layer_id = int(match.group(1))
        module.qsteer_layer_id = layer_id
        module.qsteer_module_name = str(module_name)
        route_layer_ids.append(layer_id)
    return sorted(set(route_layer_ids))


def _init_layer_state_buffers(layers, cfg: QSTEERConfig, ref_device, *, payload: dict | None = None):
    for layer_id, layer in enumerate(layers):
        mask = getattr(layer, "qsteer_expert_mask", None)
        if mask is None or (not torch.is_tensor(mask)):
            mask = torch.zeros(cfg.expert_num, dtype=torch.bool, device=ref_device)
            mask[: cfg.expert_init] = True
            if hasattr(layer, "register_buffer"):
                layer.register_buffer("qsteer_expert_mask", mask, persistent=True)
            else:
                layer.qsteer_expert_mask = mask
        else:
            mask = mask.detach().bool().to(device=ref_device).view(-1)
            if mask.numel() < cfg.expert_num:
                pad = torch.zeros(cfg.expert_num - mask.numel(), dtype=torch.bool, device=ref_device)
                mask = torch.cat([mask, pad], dim=0)
            elif mask.numel() > cfg.expert_num:
                mask = mask[: cfg.expert_num]
            if int(mask.sum().item()) <= 0:
                mask[: cfg.expert_init] = True
            layer.qsteer_expert_mask = mask

        state = getattr(layer, "qsteer_slot_state", None)
        if state is None or (not torch.is_tensor(state)):
            state = torch.zeros(cfg.expert_num, dtype=torch.int8, device=ref_device)
            state[layer.qsteer_expert_mask.bool()] = 3  # initial active bank
            if hasattr(layer, "register_buffer"):
                layer.register_buffer("qsteer_slot_state", state, persistent=True)
            else:
                layer.qsteer_slot_state = state
        else:
            state = state.detach().to(dtype=torch.int8, device=ref_device).view(-1)
            if state.numel() < cfg.expert_num:
                pad = torch.zeros(cfg.expert_num - state.numel(), dtype=torch.int8, device=ref_device)
                state = torch.cat([state, pad], dim=0)
            elif state.numel() > cfg.expert_num:
                state = state[: cfg.expert_num]
            if int((state > 0).sum().item()) <= 0:
                state[: cfg.expert_init] = 3
            layer.qsteer_slot_state.copy_(state)

        layer.qsteer_expert_mask.copy_((layer.qsteer_slot_state > 0).to(dtype=torch.bool))
    sync_qsteer_context_and_payload(layers, payload=payload, treat_probe_as_new=False)
    assert_qsteer_layer_state(
        layers,
        payload=payload,
        treat_probe_as_new=False,
        allow_probe=False,
        source_tag="runtime_attach.init_layer_state",
    )


def _sync_expert_trainability(model, layers, *, train_initial_bank: bool) -> None:
    modules_by_layer = _iter_moe_modules_by_layer(model)
    for layer_id, modules in modules_by_layer.items():
        if layer_id >= len(layers):
            continue
        state = getattr(layers[layer_id], "qsteer_slot_state", None)
        if not torch.is_tensor(state):
            continue
        for module in modules:
            limit = min(int(state.numel()), int(module.expert_num))
            for expert_idx in range(limit):
                _set_expert_trainable(
                    module,
                    expert_idx,
                    bool(
                        int(state[expert_idx].item()) == 2
                        or (train_initial_bank and int(state[expert_idx].item()) == 3)
                    ),
                )


def enable_qsteer(model, cfg: QSTEERConfig):
    lm = _unwrap_lm(model)
    if isinstance(getattr(lm, "_qsteer", None), dict):
        raise RuntimeError("Q-STEER runtime is already attached; reuse it or build a new model.")
    layers = _get_layers(lm)
    for name, parameter in model.named_parameters():
        if not _is_qsteer_parameter(name):
            parameter.requires_grad_(False)
    hidden_size = lm.config.hidden_size if hasattr(lm, "config") else lm.model.config.hidden_size
    late_count = min(cfg.late_layer_count, len(layers))
    late_layer_ids = list(range(len(layers) - late_count, len(layers)))
    route_layer_ids = _discover_route_layer_ids(model)
    if len(route_layer_ids) <= 0:
        route_layer_ids = list(late_layer_ids) if late_layer_ids else [int(len(layers) - 1)]
    route_slot_map = {int(layer_id): int(slot) for slot, layer_id in enumerate(route_layer_ids)}
    late_slot_map = {int(layer_id): int(slot) for slot, layer_id in enumerate(late_layer_ids)}

    ref_attn = layers[late_layer_ids[-1]].self_attn if late_layer_ids else layers[-1].self_attn
    num_heads = int(
        getattr(
            ref_attn,
            "num_heads",
            getattr(getattr(lm, "config", None), "num_attention_heads", 1),
        )
    )
    route_layer_count = max(1, len(route_layer_ids))

    controller = QSTEERController(
        hidden_size=hidden_size,
        expert_num=cfg.expert_num,
        route_layer_count=route_layer_count,
        late_layer_count=max(1, late_count),
        icr_rank=cfg.icr_rank,
        controller_hidden=cfg.controller_hidden,
        controller_layers=cfg.controller_layers,
        controller_dropout=cfg.controller_dropout,
        g_temp=cfg.g_temp,
        num_heads=num_heads,
    )
    ref_param = next(lm.parameters())
    ref_dtype = ref_param.dtype if ref_param.dtype.is_floating_point else torch.float32
    controller = controller.to(device=ref_param.device, dtype=ref_dtype)
    controller.strict_q_only = bool(getattr(cfg, "strict_q_only", True))
    controller.enforce_training_policy()

    lm._qsteer_controller = controller
    lm._qsteer_drift_buffer = QSTEERDriftBuffer(
        eps=float(getattr(cfg, "drift_eps", 1e-8))
    ).to(device=ref_param.device)
    qsteer_payload = {
        "cfg": cfg,
        "controller": lm._qsteer_controller,
        "drift_buffer": lm._qsteer_drift_buffer,
    }
    lm._qsteer = qsteer_payload
    if model is not lm:
        model._qsteer = qsteer_payload

    _init_layer_state_buffers(layers, cfg, ref_param.device, payload=qsteer_payload)
    _sync_expert_trainability(model, layers, train_initial_bank=True)

    for layer_id in late_layer_ids:
        attn = layers[layer_id].self_attn
        _ensure_icr_params(attn, cfg.icr_rank)
        if not hasattr(attn, "_qsteer_original_forward"):
            attn._qsteer_original_forward = attn.forward
        layer_slot = late_slot_map[int(layer_id)]
        attn.forward = _build_qsteer_attention_forward(attn, int(layer_id), int(layer_slot), cfg)

    _set_moe_topk(model, cfg.topk_update)
    qsteer_payload["late_layer_ids"] = late_layer_ids
    qsteer_payload["late_layer_slots"] = late_slot_map
    qsteer_payload["route_layer_ids"] = route_layer_ids
    qsteer_payload["route_layer_slots"] = route_slot_map
    QSTEERContext.set_route_metadata(route_layer_slots=route_slot_map, late_layer_slots=late_slot_map)
    DL.anchor(
        "qsteer.anchor.runtime_attach",
        expert_num=int(cfg.expert_num),
        expert_init=int(cfg.expert_init),
        topk_update=int(cfg.topk_update),
        late_layer_ids=[int(x) for x in late_layer_ids],
        route_layer_ids=[int(x) for x in route_layer_ids],
        late_layer_slots={int(k): int(v) for k, v in late_slot_map.items()},
        route_layer_slots={int(k): int(v) for k, v in route_slot_map.items()},
        controller_branch_state=controller.branch_state() if hasattr(controller, "branch_state") else {},
        strict_q_only=bool(getattr(controller, "strict_q_only", True)),
        slot_state_count_by_layer=_slot_state_summary(layers),
    )
    return model


def inspect_qsteer_runtime(model):
    info = {
        "payload_present": False,
        "controller_present": False,
        "model_layer_count": 0,
        "late_layer_ids": [],
        "late_layer_count": 0,
        "route_layer_ids": [],
        "route_layer_count": 0,
        "patched_layer_ids": [],
        "patched_layer_count": 0,
        "icr_param_layer_ids": [],
        "icr_param_layer_count": 0,
        "moe_lora_layer_count": 0,
        "moe_topk_histogram": {},
        "drift_buffer_present": False,
        "trainable_backbone_param_count": 0,
        "trainable_backbone_param_names": [],
        "error": None,
    }
    try:
        lm = _unwrap_lm(model)
        layers = _get_layers(lm)
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["model_layer_count"] = int(len(layers))
    payload = getattr(lm, "_qsteer", None)
    if payload is None and model is not lm:
        payload = getattr(model, "_qsteer", None)
    info["payload_present"] = bool(isinstance(payload, dict))
    info["controller_present"] = bool(getattr(lm, "_qsteer_controller", None) is not None)

    if isinstance(payload, dict):
        raw_late_ids = payload.get("late_layer_ids", [])
        if isinstance(raw_late_ids, (list, tuple)):
            info["late_layer_ids"] = [int(v) for v in raw_late_ids if str(v).strip() != ""]
        raw_route_ids = payload.get("route_layer_ids", [])
        if isinstance(raw_route_ids, (list, tuple)):
            info["route_layer_ids"] = [int(v) for v in raw_route_ids if str(v).strip() != ""]
        info["drift_buffer_present"] = bool(payload.get("drift_buffer", None) is not None)

    patched_layer_ids = []
    icr_param_layer_ids = []
    for layer_id in info["late_layer_ids"]:
        if not (0 <= layer_id < len(layers)):
            continue
        attn = layers[layer_id].self_attn
        if hasattr(attn, "_qsteer_original_forward"):
            patched_layer_ids.append(layer_id)
        if hasattr(attn, "qsteer_uq") and hasattr(attn, "qsteer_uk"):
            icr_param_layer_ids.append(layer_id)

    info["late_layer_count"] = int(len(info["late_layer_ids"]))
    info["route_layer_count"] = int(len(info["route_layer_ids"]))
    info["patched_layer_ids"] = patched_layer_ids
    info["patched_layer_count"] = int(len(patched_layer_ids))
    info["icr_param_layer_ids"] = icr_param_layer_ids
    info["icr_param_layer_count"] = int(len(icr_param_layer_ids))
    info["slot_state_count_by_layer"] = _slot_state_summary(layers)

    trainable_backbone_names = [
        str(name)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not _is_qsteer_parameter(name)
    ]
    info["trainable_backbone_param_count"] = int(len(trainable_backbone_names))
    info["trainable_backbone_param_names"] = trainable_backbone_names[:20]
    topk_hist = {}
    moe_lora_layer_count = 0
    for module in model.modules():
        if hasattr(module, "topk") and hasattr(module, "expert_num") and hasattr(module, "experts"):
            moe_lora_layer_count += 1
            try:
                topk_key = str(int(getattr(module, "topk")))
            except Exception:
                topk_key = "unknown"
            topk_hist[topk_key] = int(topk_hist.get(topk_key, 0)) + 1
    info["moe_lora_layer_count"] = int(moe_lora_layer_count)
    info["moe_topk_histogram"] = topk_hist
    return info


def validate_qsteer_runtime(
    model,
    *,
    expected_late_layer_count=None,
    require_moe_lora=False,
):
    info = inspect_qsteer_runtime(model)
    errors = []
    if info.get("error"):
        errors.append(f"runtime inspect failed: {info['error']}")
        return info, errors

    if not info.get("payload_present", False):
        errors.append("missing model._qsteer payload")
    if not info.get("controller_present", False):
        errors.append("missing _qsteer_controller on language model")
    if not info.get("drift_buffer_present", False):
        errors.append("missing drift buffer in qsteer payload")
    if int(info.get("trainable_backbone_param_count", 0)) > 0:
        errors.append(
            "backbone parameters remain trainable: "
            f"{info.get('trainable_backbone_param_names', [])}"
        )

    expected = None
    if expected_late_layer_count is not None:
        try:
            expected = max(0, int(expected_late_layer_count))
        except Exception:
            expected = None
    if expected is None:
        expected = int(info.get("late_layer_count", 0))

    if expected > 0:
        if int(info.get("late_layer_count", 0)) != expected:
            errors.append(
                f"late-layer selection mismatch: got={info.get('late_layer_count', 0)} expected={expected}"
            )
        if int(info.get("patched_layer_count", 0)) != expected:
            errors.append(
                f"attention patch mismatch: got={info.get('patched_layer_count', 0)} expected={expected}"
            )
        if int(info.get("icr_param_layer_count", 0)) != expected:
            errors.append(
                f"ICR parameter mismatch: got={info.get('icr_param_layer_count', 0)} expected={expected}"
            )
    elif int(info.get("patched_layer_count", 0)) <= 0:
        errors.append("no attention layers patched by QSTEER")

    if int(info.get("route_layer_count", 0)) <= 0:
        errors.append("no route layers discovered for MoE-LoRA gating")

    if require_moe_lora and int(info.get("moe_lora_layer_count", 0)) <= 0:
        errors.append("no MoE-LoRA layers detected (required)")

    payload = None
    layers = None
    try:
        lm = _unwrap_lm(model)
        layers = _get_layers(lm)
        payload = getattr(lm, "_qsteer", None)
    except Exception:
        payload = None
        layers = None
    if isinstance(payload, dict):
        cfg = payload.get("cfg", None)
        if cfg is not None:
            expert_init = int(getattr(cfg, "expert_init", 0))
            expert_num = int(getattr(cfg, "expert_num", 0))
            if expert_init >= expert_num:
                errors.append("expert_init must be < expert_num for probe expansion")
        if layers is None:
            errors.append("could not locate language-model layers for qsteer state validation")
        else:
            state, state_errors = validate_qsteer_layer_state(
                layers,
                payload=payload,
                treat_probe_as_new=False,
                allow_probe=True,
            )
            if int(len(state["slot_states"])) != int(len(layers)):
                errors.append(
                    f"qsteer_slot_state buffers missing on some layers: got={len(state['slot_states'])} expected={len(layers)}"
                )
            if int(len(state["expert_masks"])) != int(len(layers)):
                errors.append(
                    f"qsteer_expert_mask buffers missing on some layers: got={len(state['expert_masks'])} expected={len(layers)}"
                )
            errors.extend(state_errors)

    return info, errors
