"""Cartesian destination servo with durable per-step progress and fixed world targets."""

from __future__ import annotations

import numpy as np

from ..astra_robodawn.executor import POS_TOL, ROT_TOL, rotation_vector
from .action_format import from_env
from .persistence import digest

TRANSIT = {
    "max_steps": 160,
    "position_tolerance_m": POS_TOL,
    "stall_steps": 20,
    "progress_epsilon_m": 0.002,
    "controller": "native OSC_POSE Cartesian servo",
}

TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "position_world_m": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 3,
            "maxItems": 3,
        },
        "observation_id": {"type": "string"},
        "purpose": {"type": "string", "enum": ["pre_precision", "transport"]},
        "grasp_confirmed": {"type": "boolean"},
    },
    "required": ["position_world_m", "observation_id", "purpose", "grasp_confirmed"],
    "additionalProperties": False,
}


def validate_target(target):
    """Require a finite absolute world position bound to its source observation."""
    if not isinstance(target, dict) or set(target) != set(TARGET_SCHEMA["required"]):
        raise ValueError(
            "transit requires destination, observation_id, purpose and grasp_confirmed"
        )
    xyz = np.asarray(target["position_world_m"], dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError("transit position_world_m must contain three finite numbers")
    if not isinstance(target["observation_id"], str) or not target["observation_id"]:
        raise ValueError("transit requires observation_id")
    purpose, confirmed = target["purpose"], target["grasp_confirmed"]
    if purpose not in ("pre_precision", "transport") or not isinstance(confirmed, bool):
        raise ValueError("transit requires a valid purpose and boolean grasp_confirmed")
    if purpose == "transport" and not confirmed:
        raise ValueError("transport requires a confirmed grasp; use precision to grasp first")
    return {
        "position_world_m": xyz.tolist(),
        "observation_id": target["observation_id"],
        "purpose": purpose,
        "grasp_confirmed": confirmed,
    }


def run_transit(server, target, turn, request_id):
    """Continue a target servo after exact replay, or return its durable original result."""
    target = validate_target(target)
    journal, executor = server.journal, server.executor
    if journal.data.get("faulted"):
        raise RuntimeError("native step failed; reset the frozen checkpoint before continuing")
    identity = digest({"transit_target": target, "turn": turn, "version": 2})
    chunks = journal.data["chunks"]
    if request_id in chunks:
        if chunks[request_id]["identity"] != identity:
            raise ValueError("request ID reused with different transit target")
        chunk = chunks[request_id]
        if "result" in chunk:
            return chunk["result"]
    else:
        if (
            server.observation_id != target["observation_id"]
            or server.last_observation["state"]["steps_used"] != executor.steps_used
        ):
            raise ValueError("stale transit observation_id")
        current = journal.data["current_chunk"]
        if current and "result" not in chunks[current]:
            raise ValueError("another chunk is unfinished")
        if target["purpose"] == "transport" and executor.gripper_cmd <= 0:
            raise ValueError("transport requires a closed gripper command; use precision first")
        pose = executor.pose()
        chunk = {
            "identity": identity,
            "cursor": 0,
            "start_step": len(journal.data["actions"]),
            "target": target,
            "hold_rotation_world": pose.tip_rot.tolist(),
            "gripper_cmd": executor.gripper_cmd,
            "initial_error_m": float(
                np.linalg.norm(np.asarray(target["position_world_m"]) - pose.tip)
            ),
        }
        chunks[request_id] = chunk
    journal.data["current_chunk"] = request_id
    journal.save()
    server.turn = turn
    resumed = chunk["cursor"]
    xyz = np.asarray(chunk["target"]["position_world_m"])
    rotation = np.asarray(chunk["hold_rotation_world"])
    executor.gripper_cmd = chunk["gripper_cmd"]
    caption = "transit " + target["purpose"] + " to world metres " + str(xyz.tolist())
    # Reconstruct stall tracking from committed post-step errors. An ack cannot lose progress.
    history = [
        entry["transit_error_m"]
        for entry in journal.data["actions"][chunk["start_step"] :]
        if "transit_error_m" in entry
    ]
    best, best_step = chunk["initial_error_m"], 0
    for step, error in enumerate(history, 1):
        if error < best - TRANSIT["progress_epsilon_m"]:
            best, best_step = error, step
    reason = "max_steps"
    try:
        while chunk["cursor"] < TRANSIT["max_steps"]:
            reason = executor._stop()
            if reason:
                break
            pose = executor.pose()
            error = float(np.linalg.norm(xyz - pose.tip))
            rot_error = float(np.linalg.norm(rotation_vector(rotation @ pose.tip_rot.T)))
            if error <= POS_TOL and rot_error <= ROT_TOL:
                reason = "reached"
                break
            if error < best - TRANSIT["progress_epsilon_m"]:
                best, best_step = error, chunk["cursor"]
            elif chunk["cursor"] - best_step >= TRANSIT["stall_steps"]:
                reason = "stalled"
                break
            # Existing native controller preserves dynamics and joint constraints.
            action = executor._arm_action(xyz, rotation, pose)
            executor.before_step(action)
            executor._step(action, caption)
        else:
            reason = "max_steps"
    except BaseException:
        journal.data["faulted"] = True
        journal.save()
        raise
    pose = executor.pose()
    error = float(np.linalg.norm(xyz - pose.tip))
    rot_error = float(np.linalg.norm(rotation_vector(rotation @ pose.tip_rot.T)))
    reached = bool(error <= POS_TOL and rot_error <= ROT_TOL)
    if reached and reason == "max_steps":
        reason = "reached"
    entries = journal.data["actions"][chunk["start_step"] :]
    actions = [
        from_env(np.asarray(e["action"]), executor._split, executor._mode_index).tolist()
        for e in entries
    ]
    result = {
        "command": caption,
        "kind": "transit",
        "motion_mode": "transit",
        "transit_target": target,
        "next_motion": "precision" if reached else "reassess",
        "grasp_confirmation_source": "model_observation" if target["grasp_confirmed"] else None,
        "ok": reached,
        "target_reached": reached,
        "stop_reason": reason,
        "position_error_m": error,
        "orientation_error_rad": rot_error,
        "steps": chunk["cursor"],
        "resumed_rows": resumed,
        "executed_actions": actions,
        "task_success": executor.success,
        "state_after": executor.state(),
        "note": f"{reason}; target error {error * 100:.2f} cm; {chunk['cursor']} native steps",
    }
    server.replay.mark(caption, chunk["start_step"], executor.steps_used, turn)
    journal.finish_chunk(request_id, result)
    return result


def transit_error(env, executor, journal):
    """Attach reconstructible servo progress to each native acknowledgement."""
    current = journal.data.get("current_chunk")
    chunk = journal.data["chunks"].get(current, {})
    if "target" in chunk:
        return float(
            np.linalg.norm(np.asarray(chunk["target"]["position_world_m"]) - executor.pose().tip)
        )
    return None
