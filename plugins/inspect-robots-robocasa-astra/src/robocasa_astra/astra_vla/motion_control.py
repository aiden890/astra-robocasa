"""Explicit motion profiles enforced before native execution, independently of perception."""

from __future__ import annotations

import copy

import numpy as np

from .action_format import DIM, ROW_SCHEMA, validate_chunk
from .target_transit import TARGET_SCHEMA, TRANSIT, validate_target

PROFILES = {
    "precision": {"max_steps": 4, "translation": 0.20, "rotation": 0.15, "base": 0.10},
    "transit": TRANSIT,
}


def prepare_actions(raw, mode: str | None, control: str = "legacy"):
    """Bound continuous motion without scaling the binary gripper or arm/base selector."""
    if control == "legacy":
        if mode is not None:
            raise ValueError("legacy control cannot select a motion_mode")
        return validate_chunk(raw)
    if control != "dual" or mode != "precision":
        raise ValueError(
            "dual action rows require motion_mode precision; transit requires a target"
        )
    try:
        actions = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("actions must be finite 12-D rows") from exc
    p = PROFILES[mode]
    if actions.ndim != 2 or actions.shape[1] != DIM or not 1 <= len(actions) <= p["max_steps"]:
        raise ValueError(f"{mode} requires 1..{p['max_steps']} rows of {DIM} numbers")
    if not np.isfinite(actions).all():
        raise ValueError("actions contain NaN or infinite values")
    outside = int(np.sum(np.abs(actions) > 1.0))
    actions = np.clip(actions, -1.0, 1.0)
    notes = [f"{outside} value(s) outside [-1, 1] were clipped"] if outside else []
    for section, limit in [(slice(5, 8), p["translation"]), (slice(8, 11), p["rotation"])]:
        norm = np.linalg.norm(actions[:, section], axis=1)
        changed = norm > limit
        actions[:, section] *= (limit / np.maximum(norm, limit))[:, None]
        if changed.any():
            notes.append(f"{mode}: bounded {int(changed.sum())} arm row(s) to norm {limit}")
    if np.any(np.abs(actions[:, :4]) > p["base"]):
        notes.append(f"{mode}: bounded base/torso inputs to {p['base']}")
    actions[:, :4] = np.clip(actions[:, :4], -p["base"], p["base"])
    return actions, notes


def response_schema(schema: dict, control: str) -> dict:
    """Keep the perception query schema while changing only the action-mode contract."""
    schema = copy.deepcopy(schema)
    if control == "legacy":
        return schema
    if control != "dual":
        raise ValueError("unknown motion control")
    actions = {"type": "array", "items": ROW_SCHEMA, "minItems": 1, "maxItems": 4}
    actions = {"anyOf": [actions, {"type": "null"}]}
    schema["properties"]["actions"] = actions
    schema["properties"]["motion_mode"] = {
        "anyOf": [{"type": "string", "enum": list(PROFILES)}, {"type": "null"}]
    }
    schema["properties"]["transit_target"] = {"anyOf": [TARGET_SCHEMA, {"type": "null"}]}
    schema["required"].extend(["motion_mode", "transit_target"])
    return schema


def prepare_response(parsed, control):
    """Separate target transit, bounded precision actions and query-only replies."""
    mode, target = parsed.get("motion_mode"), parsed.get("transit_target")
    if parsed.get("queries"):
        if parsed.get("actions") is not None or mode is not None or target is not None:
            raise ValueError("queries cannot include motion")
        return None, [], None
    if control == "dual" and mode == "transit":
        if parsed.get("actions") is not None:
            raise ValueError("transit uses target only, actions must be null")
        return None, [], validate_target(target)
    if target is not None:
        raise ValueError("transit_target only allowed for transit")
    actions, notes = prepare_actions(parsed.get("actions"), mode, control)
    return actions.tolist(), notes, None


def instructions() -> str:
    """Select contact manipulation versus destination motion under the experiment assumption."""
    return (
        "MOTION POLICY: use precision when contact or fine manipulation is required: "
        "final alignment/approach to contact, grasp acquisition or regrasp, turning or "
        "pushing an object, insertion, release and fine corrections. Output 1..4 native "
        "12-D delta rows; translation norm <=0.20, rotation norm <=0.15, base/torso <=0.10. "
        "Set transit_target=null. "
        "Use transit for other destination travel: to a staging position immediately "
        "before a precision operation (purpose=pre_precision), or to carry a securely "
        "grasped object (purpose=transport). Holding an object alone does not require "
        "precision during transport, but manipulating its contact with the environment "
        "does. Assume NO intermediate obstacles or external disturbances during transit; "
        "obstacle planning is outside this experiment. Do not spend precision turns on "
        "ordinary destination travel. End transit at a pre-manipulation position, then "
        "observe and switch to precision for the actual contact, adjustment or release. "
        "For transport, first confirm from current images and measured feedback that the "
        "object is actually held and follows the gripper; a close command alone is NOT "
        "proof. Explain this briefly in progress, and set grasp_confirmed=true only when "
        "supported. If uncertain or empty-handed, use precision to establish the grasp "
        "before transport. pre_precision does not require grasp confirmation. "
        "Transit outputs actions=null and "
        "transit_target={position_world_m:[x,y,z],observation_id:current_id,"
        "purpose:pre_precision or transport,grasp_confirmed:true or false}. "
        "Absolute world XYZ metres specify the fingertip centre destination, not wrist, "
        "pixel, relative offset or cm. For held objects choose the gripper destination "
        "that positions the object for the next precision operation. "
        "The native OSC controller moves to this target without further model calls, "
        "holding initial orientation, gripper command and stationary base. It performs "
        "physical motion, not teleportation. Arrival tolerance is 1 cm / 5 deg; cap 160 "
        "native steps, stop on stall, task success or episode budget. Inspect "
        "target_reached and remaining error before advancing the plan. Transport requires "
        "a closed gripper command as well as your grasp confirmation; that command does "
        "not independently verify object attachment. "
        "Orientation/gripper changes use precision. All steps are 20 Hz. "
        "Queries set actions=null, motion_mode=null, transit_target=null and do not move. "
        "On motion responses set queries=null if available. "
        "Prior demonstrations show legacy delta chunks; retain their strategy but follow "
        "this precision/transit policy and output contract for your responses."
    )


def adapt_prompt(text: str) -> str:
    """Replace fixed-16 instructions without altering task goals, memory or perception."""
    text = text.replace(
        "chunk of exactly 16 actions", "precision chunk of 1..4 actions or a transit target"
    )
    text = text.replace(
        "one chunk of 16 low-level actions", "one precision chunk or one Cartesian transit target"
    )
    text = text.replace("All 16 actions are executed", "All supplied actions are executed")
    text = text.replace("0.8 s of motion", "precision motion or a destination transit")
    text = text.replace(
        '"actions": exactly 16 rows of 12 numbers',
        '"motion_mode": precision or transit (null only for a query)\n'
        '  "actions": 1..4 rows for precision, null for transit or query; '
        "transit_target: world destination or null",
    )
    return text + "\n\n" + instructions()
