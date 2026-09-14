from __future__ import annotations

import torch
from transformers import TrainerCallback

from .route_context import QSTEERContext
from .debug import QSTEERDebugLogger as DL
from .expansion import _iter_moe_modules_by_layer, _set_expert_trainable
from .state_sync import assert_qsteer_layer_state, sync_qsteer_context_and_payload


def _unwrap_lm(model):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model
    if hasattr(model, "layers"):
        return model
    return None


def _get_layers(lm):
    if lm is None:
        return None
    if hasattr(lm, "model") and hasattr(lm.model, "layers"):
        return lm.model.layers
    return getattr(lm, "layers", None)


class QSTEERTaskFinalizeCallback(TrainerCallback):
    """Task-end state transition: NEW->OLD and drift-reference finalize."""

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        lm = _unwrap_lm(model)
        layers = _get_layers(lm)
        if layers is None:
            return
        # Validate before changing slot state, trainability, or controller rows.
        payload = getattr(lm, "_qsteer", None)
        drift_buffer = payload.get("drift_buffer") if isinstance(payload, dict) else None
        if drift_buffer is not None and not drift_buffer.running_sum:
            raise RuntimeError("No diagnostic attention is pending for task finalization.")
        for layer_id, layer in enumerate(layers):
            slots = getattr(layer, "qsteer_slot_state", None)
            if torch.is_tensor(slots) and bool((slots == 1).any()):
                raise RuntimeError(f"Layer {layer_id} has an unfinished expansion probe.")
            if torch.is_tensor(slots) and bool((slots == 2).any()):
                route_slots = payload.get("route_layer_slots", {}) if isinstance(payload, dict) else {}
                if not isinstance(route_slots, dict) or layer_id not in route_slots:
                    raise RuntimeError(f"Q-STEER promotions have no route slot for layer {layer_id}")
        layer_modules = _iter_moe_modules_by_layer(model, only_moe_lora=True)

        slot_states = {}
        old_masks = {}
        new_masks = {}
        old_count_by_layer = {}
        new_count_by_layer = {}
        inactive_count_by_layer = {}
        promoted_new_to_old = {}
        promoted_expert_ids_by_layer = {}
        dropped_probe_to_inactive = {}

        for layer_id, layer in enumerate(layers):
            state_buf = getattr(layer, "qsteer_slot_state", None)
            if state_buf is None or not torch.is_tensor(state_buf):
                continue
            state_view = state_buf.detach().to(dtype=torch.int8)
            retained_probe_ids = torch.nonzero(state_view == 1, as_tuple=False).view(-1).tolist()
            if retained_probe_ids:
                state_view[state_view == 1] = 0
            promoted_ids = torch.nonzero(state_view == 2, as_tuple=False).view(-1).tolist()
            promoted = int((state_view == 2).sum().item())
            state_view[state_view == 2] = 3
            if hasattr(layer, "qsteer_slot_state"):
                layer.qsteer_slot_state.copy_(state_view)

            active_mask = state_view > 0
            if hasattr(layer, "qsteer_expert_mask") and torch.is_tensor(layer.qsteer_expert_mask):
                layer.qsteer_expert_mask.copy_(active_mask.to(dtype=torch.bool))

            for module in layer_modules.get(int(layer_id), []):
                for idx in range(int(state_view.numel())):
                    _set_expert_trainable(module, idx, False)

            slot_states[int(layer_id)] = state_view.to(dtype=torch.int8)
            old_mask = state_view == 3
            new_mask = state_view == 2
            old_masks[int(layer_id)] = old_mask
            new_masks[int(layer_id)] = new_mask
            old_count_by_layer[int(layer_id)] = int(old_mask.sum().item())
            new_count_by_layer[int(layer_id)] = int(new_mask.sum().item())
            inactive_count_by_layer[int(layer_id)] = int((state_view == 0).sum().item())
            promoted_new_to_old[int(layer_id)] = int(promoted)
            dropped_probe_to_inactive[int(layer_id)] = int(len(retained_probe_ids))
            if promoted_ids:
                promoted_expert_ids_by_layer[int(layer_id)] = [int(x) for x in promoted_ids]

        payload = getattr(lm, "_qsteer", None)
        if slot_states:
            sync_qsteer_context_and_payload(
                layers,
                payload=payload if isinstance(payload, dict) else None,
                treat_probe_as_new=False,
            )
            assert_qsteer_layer_state(
                layers,
                payload=payload if isinstance(payload, dict) else None,
                treat_probe_as_new=False,
                allow_probe=False,
                source_tag="task_finalize.post_sync",
            )
        ref_layers = []
        route_branch_before = {}
        route_branch_after = {}
        task_drift_mean = None
        if isinstance(payload, dict):
            controller = payload.get("controller", None)
            if controller is not None and hasattr(controller, "branch_state"):
                route_branch_before = controller.branch_state()
            drift_buffer = payload.get("drift_buffer", None)
            if drift_buffer is not None:
                try:
                    if hasattr(drift_buffer, "current_task_drift_mean"):
                        task_drift_mean = float(drift_buffer.current_task_drift_mean())
                    else:
                        task_drift_mean = float(getattr(drift_buffer, "task_drift_mean", 0.0))
                except Exception:
                    task_drift_mean = None
            promoted_route_ids_by_slot = {}
            if promoted_expert_ids_by_layer:
                raw_route_slots = payload.get("route_layer_slots", {})
                if not isinstance(raw_route_slots, dict):
                    raise RuntimeError("Q-STEER route-layer slots are missing at task finalization")
                route_slots = {int(key): int(value) for key, value in raw_route_slots.items()}
                missing_layers = sorted(
                    int(layer_id)
                    for layer_id in promoted_expert_ids_by_layer
                    if int(layer_id) not in route_slots
                )
                if missing_layers:
                    raise RuntimeError(
                        f"Q-STEER promotions have no route slot for layers {missing_layers}"
                    )
                promoted_route_ids_by_slot = {
                    route_slots[int(layer_id)]: [int(index) for index in expert_ids]
                    for layer_id, expert_ids in promoted_expert_ids_by_layer.items()
                }
            if (
                promoted_expert_ids_by_layer
                and controller is not None
                and hasattr(controller, "promote_new_branch_to_old")
            ):
                controller.promote_new_branch_to_old(
                    promoted_expert_ids_by_layer=promoted_route_ids_by_slot,
                )
            elif controller is not None and hasattr(controller, "enforce_training_policy"):
                controller.enforce_training_policy()
            if controller is not None and hasattr(controller, "branch_state"):
                route_branch_after = controller.branch_state()
            if promoted_expert_ids_by_layer:
                payload["newly_promoted_old_expert_ids_by_layer"] = {
                    int(layer_id): [int(x) for x in expert_ids]
                    for layer_id, expert_ids in promoted_expert_ids_by_layer.items()
                }
            else:
                payload.pop("newly_promoted_old_expert_ids_by_layer", None)
            if controller is not None and hasattr(controller, "branch_state"):
                try:
                    payload["route_branch_state"] = dict(controller.branch_state())
                except Exception:
                    payload.pop("route_branch_state", None)
            if drift_buffer is not None and hasattr(drift_buffer, "finalize_task_reference"):
                ref = drift_buffer.finalize_task_reference()
                if isinstance(ref, dict):
                    ref_layers = [int(k) for k in ref.keys()]

        DL.log(
            "qsteer.task_finalize",
            force=True,
            promoted_new_to_old=promoted_new_to_old,
            dropped_probe_to_inactive=dropped_probe_to_inactive,
            old_count_by_layer=old_count_by_layer,
            new_count_by_layer=new_count_by_layer,
            inactive_count_by_layer=inactive_count_by_layer,
            drift_ref_saved_layers=ref_layers,
            promoted_expert_ids_by_layer=promoted_expert_ids_by_layer,
            task_drift_mean=float(task_drift_mean) if task_drift_mean is not None else None,
        )
        DL.anchor(
            "qsteer.anchor.task_finalize",
            promoted_new_to_old=promoted_new_to_old,
            dropped_probe_to_inactive=dropped_probe_to_inactive,
            old_count_by_layer=old_count_by_layer,
            new_count_by_layer=new_count_by_layer,
            inactive_count_by_layer=inactive_count_by_layer,
            drift_ref_saved_layers=ref_layers,
            promoted_expert_ids_by_layer=promoted_expert_ids_by_layer,
            task_drift_mean=float(task_drift_mean) if task_drift_mean is not None else None,
            route_branch_before=route_branch_before,
            route_branch_after=route_branch_after,
        )
