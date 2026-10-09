"""Explicit motion profiles enforced before native execution, independently of perception."""

from __future__ import annotations

import copy

import numpy as np

from .action_format import DIM, ROW_SCHEMA, validate_chunk

PROFILES = {
    "precision": {"max_steps": 4, "translation": 0.20, "rotation": 0.15, "base": 0.10},
    "transit": {"max_steps": 16, "translation": 1.0, "rotation": 1.0, "base": 1.0},
}


def prepare_actions(raw, mode: str | None, control: str = "legacy"):
    """Bound continuous motion without scaling the binary gripper or arm/base selector."""
    if control == "legacy":
        if mode is not None:
            raise ValueError("legacy control cannot select a motion_mode")
        return validate_chunk(raw)
    if control != "dual" or mode not in PROFILES:
        raise ValueError("dual control requires motion_mode precision or transit")
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
    actions = {"type": "array", "items": ROW_SCHEMA, "minItems": 1, "maxItems": 16}
    if "queries" in schema["properties"]:
        actions = {"anyOf": [actions, {"type": "null"}]}
    schema["properties"]["actions"] = actions
    schema["properties"]["motion_mode"] = {
        "anyOf": [{"type": "string", "enum": list(PROFILES)}, {"type": "null"}]
    }
    schema["required"].append("motion_mode")
    return schema


def instructions() -> str:
    """Tell the model when to switch profiles and the bounds the native worker enforces."""
    return (
        "MOTION MODES: select motion_mode for every action response. precision: "
        "1..4 native steps, translation vector norm <=0.20, rotation norm <=0.15, "
        "base/torso components <=0.10. Use for final approach, aligning, grasping, "
        "contact, insertion, near obstacles, uncertain clearance or small corrections. "
        "transit: 1..16 steps, translation/rotation norm <=1.0, base/torso <=1.0. "
        "Use only for coarse travel through visibly clear space; change to precision "
        "before reaching objects. Larger bounds permit faster movement, not guaranteed "
        "arrival or collision avoidance. All actions stay at 20 Hz. The gripper and "
        "the dataset index-4 arm/base selector retain their binary meanings. Closing "
        "the fingers may need several precision turns; observe before assuming a grasp. "
        "On a depth query response, actions=null and motion_mode=null; query without "
        "moving. On an action response motion_mode must be precision or transit. "
        "The worker bounds motion even when your requested values exceed its limits."
    )


def adapt_prompt(text: str) -> str:
    """Replace fixed-16 instructions without altering task goals, memory or perception."""
    text = text.replace("chunk of exactly 16 actions", "chunk of 1..16 actions")
    text = text.replace(
        "one chunk of 16 low-level actions", "one variable-length chunk of low-level actions"
    )
    text = text.replace("All 16 actions are executed", "All supplied actions are executed")
    text = text.replace("0.8 s of motion", "up to 0.8 s of motion")
    text = text.replace(
        '"actions": exactly 16 rows of 12 numbers',
        '"motion_mode": precision or transit (null only for a query)\n'
        '  "actions": variable-length rows of 12 numbers',
    )
    return text + "\n\n" + instructions()
