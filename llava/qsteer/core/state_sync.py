import torch

from .route_context import QSTEERContext
from .debug import QSTEERDebugLogger as DL


def _slot_state_summary(state: torch.Tensor) -> dict[str, int]:
    view = state.detach().to(dtype=torch.int8).view(-1)
    return {
        "inactive": int((view == 0).sum().item()),
        "probe": int((view == 1).sum().item()),
        "new": int((view == 2).sum().item()),
        "old": int((view == 3).sum().item()),
    }


def qsteer_runtime_state_is_finalized(
    slot_state_count_by_layer: dict[int, dict[str, int]] | dict[str, dict[str, int]] | None,
) -> bool:
    if not isinstance(slot_state_count_by_layer, dict) or not slot_state_count_by_layer:
        return False
    saw_layer = False
    for summary in slot_state_count_by_layer.values():
        if not isinstance(summary, dict):
            continue
        saw_layer = True
        if int(summary.get("probe", 0) or 0) != 0:
            return False
        if int(summary.get("new", 0) or 0) != 0:
            return False
    return saw_layer


def _expert_init_from_payload(payload: dict | None) -> int:
    if not isinstance(payload, dict):
        return 0
    raw_value = payload.get("expert_init", None)
    if raw_value is None:
        cfg = payload.get("cfg", None)
        raw_value = getattr(cfg, "expert_init", None)
    try:
        return max(0, int(raw_value or 0))
    except Exception:
        return 0


def _normalize_expert_id_map(mapping: dict | None) -> dict[int, list[int]]:
    if not isinstance(mapping, dict):
        return {}
    norm: dict[int, list[int]] = {}
    for raw_layer_id, raw_ids in dict(mapping).items():
        try:
            layer_id = int(raw_layer_id)
        except Exception:
            continue
        ids: list[int] = []
        for raw_expert_id in list(raw_ids or []):
            try:
                ids.append(int(raw_expert_id))
            except Exception:
                continue
        if ids:
            norm[int(layer_id)] = sorted(set(ids))
    return norm


def derive_qsteer_runtime_metadata(
    state: dict[str, dict[int, torch.Tensor] | dict[int, dict[str, int]]],
    *,
    expert_init: int = 0,
) -> dict[str, object]:
    slot_states = state.get("slot_states", {})
    slot_state_count_by_layer = state.get("slot_state_count_by_layer", {})
    active_old_expert_ids_by_layer: dict[int, list[int]] = {}
    promoted_old_expert_ids_by_layer: dict[int, list[int]] = {}

    for raw_layer_id, slot_state in dict(slot_states).items():
        try:
            layer_id = int(raw_layer_id)
        except Exception:
            continue
        if not torch.is_tensor(slot_state):
            continue
        slot_view = slot_state.detach().to(dtype=torch.int8).view(-1)
        old_ids = [
            int(idx)
            for idx in torch.nonzero(slot_view == 3, as_tuple=False).view(-1).tolist()
        ]
        if old_ids:
            active_old_expert_ids_by_layer[int(layer_id)] = old_ids
        if int(expert_init) > 0:
            promoted_ids = [int(idx) for idx in old_ids if int(idx) >= int(expert_init)]
            if promoted_ids:
                promoted_old_expert_ids_by_layer[int(layer_id)] = promoted_ids

    return {
        "task_finalized": bool(qsteer_runtime_state_is_finalized(slot_state_count_by_layer)),
        "active_old_expert_ids_by_layer": active_old_expert_ids_by_layer,
        "promoted_old_expert_ids_by_layer": promoted_old_expert_ids_by_layer,
    }


def collect_qsteer_layer_state(
    layers,
    *,
    treat_probe_as_new: bool = False,
) -> dict[str, dict[int, torch.Tensor] | dict[int, dict[str, int]]]:
    expert_masks: dict[int, torch.Tensor] = {}
    slot_states: dict[int, torch.Tensor] = {}
    old_masks: dict[int, torch.Tensor] = {}
    new_masks: dict[int, torch.Tensor] = {}
    slot_state_count_by_layer: dict[int, dict[str, int]] = {}

    if layers is None:
        return {
            "expert_masks": expert_masks,
            "slot_states": slot_states,
            "old_masks": old_masks,
            "new_masks": new_masks,
            "slot_state_count_by_layer": slot_state_count_by_layer,
        }

    for layer_id, layer in enumerate(layers):
        state = getattr(layer, "qsteer_slot_state", None)
        mask = getattr(layer, "qsteer_expert_mask", None)
        if torch.is_tensor(state):
            state_view = state.detach().to(dtype=torch.int8).view(-1)
            slot_states[int(layer_id)] = state_view
            old_masks[int(layer_id)] = state_view == 3
            new_mask = state_view == 2
            if treat_probe_as_new:
                new_mask = new_mask | (state_view == 1)
            new_masks[int(layer_id)] = new_mask
            slot_state_count_by_layer[int(layer_id)] = _slot_state_summary(state_view)
            if torch.is_tensor(mask):
                expert_masks[int(layer_id)] = mask.detach().bool().view(-1)
            else:
                expert_masks[int(layer_id)] = (state_view > 0).bool()
        elif torch.is_tensor(mask):
            expert_masks[int(layer_id)] = mask.detach().bool().view(-1)

    return {
        "expert_masks": expert_masks,
        "slot_states": slot_states,
        "old_masks": old_masks,
        "new_masks": new_masks,
        "slot_state_count_by_layer": slot_state_count_by_layer,
    }


def validate_qsteer_layer_state(
    layers,
    *,
    payload: dict | None = None,
    treat_probe_as_new: bool = False,
    allow_probe: bool = True,
) -> tuple[
    dict[str, dict[int, torch.Tensor] | dict[int, dict[str, int]]],
    list[str],
]:
    state = collect_qsteer_layer_state(layers, treat_probe_as_new=treat_probe_as_new)
    expert_masks = state["expert_masks"]
    slot_states = state["slot_states"]
    old_masks = state["old_masks"]
    new_masks = state["new_masks"]
    errors: list[str] = []
    metadata = derive_qsteer_runtime_metadata(
        state,
        expert_init=_expert_init_from_payload(payload),
    )

    layer_count = int(len(layers)) if layers is not None else 0
    for layer_id, slot_state in slot_states.items():
        slot_view = slot_state.detach().to(dtype=torch.int8).view(-1)
        invalid = torch.nonzero((slot_view < 0) | (slot_view > 3), as_tuple=False).view(-1).tolist()
        if invalid:
            errors.append(
                f"layer {int(layer_id)} has invalid slot states at indices={invalid[:8]}"
            )
        if (not allow_probe) and bool((slot_view == 1).any().item()):
            errors.append(f"layer {int(layer_id)} still has probe slots in finalized/runtime-loaded state")

        expected_expert = (slot_view > 0).bool()
        actual_expert = expert_masks.get(int(layer_id), None)
        if torch.is_tensor(actual_expert) and not torch.equal(actual_expert.bool().view(-1), expected_expert):
            errors.append(f"layer {int(layer_id)} expert mask does not match slot_state>0")

        expected_old = (slot_view == 3).bool()
        actual_old = old_masks.get(int(layer_id), None)
        if torch.is_tensor(actual_old) and not torch.equal(actual_old.bool().view(-1), expected_old):
            errors.append(f"layer {int(layer_id)} old mask does not match slot_state==3")

        expected_new = (slot_view == 2).bool()
        if treat_probe_as_new:
            expected_new = expected_new | (slot_view == 1)
        actual_new = new_masks.get(int(layer_id), None)
        if torch.is_tensor(actual_new) and not torch.equal(actual_new.bool().view(-1), expected_new):
            errors.append(
                f"layer {int(layer_id)} new mask does not match slot_state={'2|1' if treat_probe_as_new else '2'}"
            )

    if isinstance(payload, dict):
        payload_slot_summary = payload.get("slot_state_count_by_layer", None)
        if isinstance(payload_slot_summary, dict):
            norm_payload_slot_summary = {}
            for raw_layer_id, summary in payload_slot_summary.items():
                if not isinstance(summary, dict):
                    continue
                try:
                    layer_id = int(raw_layer_id)
                except Exception:
                    errors.append(
                        f"payload slot_state_count_by_layer has invalid layer key: {raw_layer_id!r}"
                    )
                    continue
                norm_summary = {}
                for name, value in dict(summary).items():
                    try:
                        norm_summary[str(name)] = int(value)
                    except Exception:
                        errors.append(
                            f"payload slot_state_count_by_layer[{layer_id}] has invalid {name!r}={value!r}"
                        )
                norm_payload_slot_summary[int(layer_id)] = norm_summary
            if norm_payload_slot_summary != state["slot_state_count_by_layer"]:
                errors.append("payload slot_state_count_by_layer does not match live layer state")
        if "task_finalized" in payload:
            if bool(payload.get("task_finalized", False)) != bool(metadata["task_finalized"]):
                errors.append(
                    f"payload task_finalized={bool(payload.get('task_finalized', False))} "
                    f"does not match live finalized={bool(metadata['task_finalized'])}"
                )
        payload_active_old = payload.get("active_old_expert_ids_by_layer", None)
        if isinstance(payload_active_old, dict):
            if _normalize_expert_id_map(payload_active_old) != metadata["active_old_expert_ids_by_layer"]:
                errors.append("payload active_old_expert_ids_by_layer does not match live layer state")
        route_layer_ids = payload.get("route_layer_ids", None)
        route_layer_slots = payload.get("route_layer_slots", None)
        late_layer_ids = payload.get("late_layer_ids", None)
        late_layer_slots = payload.get("late_layer_slots", None)
        controller = payload.get("controller", None)
        payload_route_branch_state = payload.get("route_branch_state", None)
        if (
            controller is not None
            and hasattr(controller, "branch_state")
            and isinstance(payload_route_branch_state, dict)
        ):
            try:
                current_route_branch_state = dict(controller.branch_state())
            except Exception:
                current_route_branch_state = None
            if (
                isinstance(current_route_branch_state, dict)
                and payload_route_branch_state != current_route_branch_state
            ):
                errors.append("payload route_branch_state does not match live controller branch state")

        for label, ids, slots in (
            ("route", route_layer_ids, route_layer_slots),
            ("late", late_layer_ids, late_layer_slots),
        ):
            if isinstance(ids, (list, tuple)):
                ids_norm = [int(x) for x in ids]
                if len(set(ids_norm)) != len(ids_norm):
                    errors.append(f"{label} layer ids contain duplicates: {ids_norm}")
                if any((int(x) < 0) or (int(x) >= layer_count) for x in ids_norm):
                    errors.append(f"{label} layer ids out of range: {ids_norm}")
            else:
                ids_norm = None
            if isinstance(slots, dict):
                slot_map = {int(k): int(v) for k, v in dict(slots).items()}
                expected_slots = list(range(len(slot_map)))
                if sorted(slot_map.values()) != expected_slots:
                    errors.append(f"{label} layer slots are not contiguous: {slot_map}")
                if ids_norm is not None and set(slot_map.keys()) != set(ids_norm):
                    errors.append(
                        f"{label} layer slot keys do not match {label}_layer_ids: ids={ids_norm} slots={slot_map}"
                    )

        promoted = payload.get("promoted_old_expert_ids_by_layer", None)
        if isinstance(promoted, dict):
            if _normalize_expert_id_map(promoted) != metadata["promoted_old_expert_ids_by_layer"]:
                errors.append("payload promoted_old_expert_ids_by_layer does not match live layer state")
            for raw_layer_id, expert_ids in promoted.items():
                try:
                    layer_id = int(raw_layer_id)
                except Exception:
                    errors.append(f"invalid promoted_old_expert_ids layer key: {raw_layer_id!r}")
                    continue
                slot_view = slot_states.get(layer_id, None)
                if slot_view is None:
                    errors.append(
                        f"promoted_old_expert_ids references missing layer {int(layer_id)}"
                    )
                    continue
                slot_view = slot_view.detach().to(dtype=torch.int8).view(-1)
                for raw_expert_id in list(expert_ids or []):
                    try:
                        expert_id = int(raw_expert_id)
                    except Exception:
                        errors.append(
                            f"layer {int(layer_id)} promoted expert id is not an int: {raw_expert_id!r}"
                        )
                        continue
                    if expert_id < 0 or expert_id >= int(slot_view.numel()):
                        errors.append(
                            f"layer {int(layer_id)} promoted expert id out of range: {int(expert_id)}"
                        )
                        continue
                    if int(slot_view[expert_id].item()) != 3:
                        errors.append(
                            f"layer {int(layer_id)} promoted expert id {int(expert_id)} is not OLD in slot state"
                        )

    return state, errors


def assert_qsteer_layer_state(
    layers,
    *,
    payload: dict | None = None,
    treat_probe_as_new: bool = False,
    allow_probe: bool = True,
    source_tag: str = "qsteer.state",
) -> dict[str, dict[int, torch.Tensor] | dict[int, dict[str, int]]]:
    state, errors = validate_qsteer_layer_state(
        layers,
        payload=payload,
        treat_probe_as_new=treat_probe_as_new,
        allow_probe=allow_probe,
    )
    if errors:
        DL.anchor(
            "qsteer.anchor.state_invariant_error",
            source_tag=str(source_tag),
            treat_probe_as_new=bool(treat_probe_as_new),
            allow_probe=bool(allow_probe),
            errors=errors[:20],
            slot_state_count_by_layer=state["slot_state_count_by_layer"],
        )
        raise RuntimeError(
            f"QSTEER state invariant failed at {source_tag}: {'; '.join(errors[:8])}"
        )
    return state


def _layer_buffer_device(layer, *names: str):
    for name in names:
        value = getattr(layer, name, None)
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")


def _overwrite_layer_buffer(layer, name: str, value: torch.Tensor, *, dtype) -> None:
    tensor = value.detach().to(dtype=dtype).view(-1)
    current = getattr(layer, name, None)
    if torch.is_tensor(current):
        target = torch.zeros_like(current, dtype=dtype, device=current.device).view(-1)
        n = min(int(target.numel()), int(tensor.numel()))
        if n > 0:
            target[:n].copy_(tensor[:n].to(device=target.device, dtype=dtype))
        current.copy_(target)
        return

    device = _layer_buffer_device(layer, "qsteer_slot_state", "qsteer_expert_mask")
    tensor = tensor.to(device=device)
    if hasattr(layer, "register_buffer"):
        layer.register_buffer(name, tensor, persistent=True)
    else:
        setattr(layer, name, tensor)


def _mapping_tensor(mapping: dict | None, layer_id: int):
    if not isinstance(mapping, dict):
        return None
    return mapping.get(str(int(layer_id)), mapping.get(int(layer_id), None))


def apply_qsteer_runtime_state(
    layers,
    runtime_state: dict,
    *,
    payload: dict | None = None,
    treat_probe_as_new: bool = False,
    allow_probe: bool = False,
    source_tag: str = "runtime_state",
) -> dict[str, dict[int, torch.Tensor] | dict[int, dict[str, int]]]:
    if layers is None or not isinstance(runtime_state, dict):
        state = sync_qsteer_context_and_payload(
            layers,
            payload=payload,
            treat_probe_as_new=treat_probe_as_new,
        )
        return assert_qsteer_layer_state(
            layers,
            payload=payload,
            treat_probe_as_new=treat_probe_as_new,
            allow_probe=allow_probe,
            source_tag=source_tag,
        )

    slot_states = runtime_state.get("slot_states", {})
    expert_masks = runtime_state.get("expert_masks", {})
    for layer_id, layer in enumerate(layers):
        slot = _mapping_tensor(slot_states, layer_id)
        mask = _mapping_tensor(expert_masks, layer_id)
        if torch.is_tensor(slot):
            _overwrite_layer_buffer(layer, "qsteer_slot_state", slot, dtype=torch.int8)
            derived_mask = getattr(layer, "qsteer_slot_state").detach().to(dtype=torch.int8).view(-1) > 0
            _overwrite_layer_buffer(layer, "qsteer_expert_mask", derived_mask, dtype=torch.bool)
        elif torch.is_tensor(mask):
            _overwrite_layer_buffer(layer, "qsteer_expert_mask", mask, dtype=torch.bool)

    sync_qsteer_context_and_payload(
        layers,
        payload=payload,
        treat_probe_as_new=treat_probe_as_new,
    )
    return assert_qsteer_layer_state(
        layers,
        payload=payload,
        treat_probe_as_new=treat_probe_as_new,
        allow_probe=allow_probe,
        source_tag=source_tag,
    )


def sync_qsteer_context_and_payload(
    layers,
    *,
    payload: dict | None = None,
    treat_probe_as_new: bool = False,
) -> dict[str, dict[int, torch.Tensor] | dict[int, dict[str, int]]]:
    state = collect_qsteer_layer_state(layers, treat_probe_as_new=treat_probe_as_new)
    expert_masks = state["expert_masks"]
    slot_states = state["slot_states"]
    old_masks = state["old_masks"]
    new_masks = state["new_masks"]

    if expert_masks:
        QSTEERContext.set_expert_masks(expert_masks)
    else:
        QSTEERContext.set_expert_masks(None)
    if slot_states:
        QSTEERContext.set_slot_states(slot_states)
        QSTEERContext.set_old_new_masks(old_masks=old_masks, new_masks=new_masks)
    else:
        QSTEERContext.clear_slot_state_metadata()

    if isinstance(payload, dict):
        metadata = derive_qsteer_runtime_metadata(
            state,
            expert_init=_expert_init_from_payload(payload),
        )
        if expert_masks:
            payload["expert_masks"] = {
                int(layer_id): mask.detach().bool().view(-1)
                for layer_id, mask in expert_masks.items()
            }
        else:
            payload.pop("expert_masks", None)
        if slot_states:
            payload["expert_states"] = {
                int(layer_id): slot.detach().to(dtype=torch.int8).view(-1)
                for layer_id, slot in slot_states.items()
            }
            payload["old_masks"] = {
                int(layer_id): mask.detach().bool().view(-1)
                for layer_id, mask in old_masks.items()
            }
            payload["new_masks"] = {
                int(layer_id): mask.detach().bool().view(-1)
                for layer_id, mask in new_masks.items()
            }
        else:
            payload.pop("expert_states", None)
            payload.pop("old_masks", None)
            payload.pop("new_masks", None)
        if state["slot_state_count_by_layer"]:
            payload["slot_state_count_by_layer"] = {
                int(layer_id): dict(summary)
                for layer_id, summary in state["slot_state_count_by_layer"].items()
            }
            payload["task_finalized"] = bool(metadata["task_finalized"])
            if metadata["active_old_expert_ids_by_layer"]:
                payload["active_old_expert_ids_by_layer"] = {
                    int(layer_id): [int(x) for x in expert_ids]
                    for layer_id, expert_ids in metadata["active_old_expert_ids_by_layer"].items()
                }
            else:
                payload.pop("active_old_expert_ids_by_layer", None)
            if metadata["promoted_old_expert_ids_by_layer"]:
                payload["promoted_old_expert_ids_by_layer"] = {
                    int(layer_id): [int(x) for x in expert_ids]
                    for layer_id, expert_ids in metadata["promoted_old_expert_ids_by_layer"].items()
                }
            else:
                payload.pop("promoted_old_expert_ids_by_layer", None)
        else:
            payload.pop("slot_state_count_by_layer", None)
            payload.pop("task_finalized", None)
            payload.pop("active_old_expert_ids_by_layer", None)
            payload.pop("promoted_old_expert_ids_by_layer", None)
        controller = payload.get("controller", None)
        if controller is not None and hasattr(controller, "branch_state"):
            try:
                payload["route_branch_state"] = dict(controller.branch_state())
            except Exception:
                payload.pop("route_branch_state", None)
        else:
            payload.pop("route_branch_state", None)

    return state
