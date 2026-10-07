"""The RoboCasa365 (LeRobot / GR00T) action format: 12-D actions in dataset order, 16-step chunks at 20 Hz.

Dataset ("modality") order, from ``meta/modality.json`` of every RoboCasa365 LeRobot dataset::

    0-3   base_motion             platform x, y, yaw velocity and torso (all 0 when the platform does not move)
    4     control_mode            -1 arm mode, +1 base mode (the arm goal then follows the moving platform)
    5-7   end_effector_position   fingertip position delta in the robot base frame, -1..1
    8-10  end_effector_rotation   fingertip rotation delta (axis-angle) in the base frame, -1..1
    11    gripper_close           -1 open, +1 close

The simulator (HYBRID_MOBILE_BASE composite controller) takes ``[arm 6, gripper, base 3, torso, mode]``.
:func:`to_env` follows RoboCasa's evaluation wrapper (``gym_wrapper.unmap_action``): the arm, base and torso
parts are copied, gripper and control mode become +1 when >= 0.5 and -1 otherwise. On dataset actions (whose
gripper and mode are exactly +-1) this equals RoboCasa's own playback reordering (``reorder_lerobot_action``).
Scale: the OSC controller sets its goal 5 cm / 0.5 rad ahead per unit input, but at 20 Hz the arm follows only
part of that each step. Measured in free space (scripts/robocasa-astra/vla/calibrate.py, primer kitchen): a
constant input v moves the fingertips ~1.2 v cm (forward/left) or ~1.1 v cm (up/down) and turns them ~6.3 v deg
per step, linearly for 0.1 <= |v| <= 1, plus ~0.7 cm of drift after the input stops.

GR00T N1.5 for RoboCasa predicts 16 steps per inference and executes all 16 (``action_indices = range(16)``,
``n_action_steps = 16``); the control frequency is 20 Hz, so one chunk is 0.8 s.
"""

from __future__ import annotations

import numpy as np

DIM = 12
CHUNK = 16
CONTROL_HZ = 20
CM_PER_STEP = np.array([1.2, 1.2, 1.1])  # measured travel per step per unit input, forward/left/up (docstring)
DEG_PER_STEP = 6.3  # measured fingertip rotation per step per unit end_effector_rotation input

# name -> (start, end) in dataset order
MODALITY = {
    "base_motion": (0, 4),
    "control_mode": (4, 5),
    "end_effector_position": (5, 8),
    "end_effector_rotation": (8, 11),
    "gripper_close": (11, 12),
}
# simulator order for PandaOmron (composite controller ``_action_split_indexes`` + hybrid mode index)
ENV_SPLIT = {"right": (0, 6), "right_gripper": (6, 7), "base": (7, 10), "torso": (10, 11)}
ENV_MODE_INDEX = 11

ROW_SCHEMA = {"type": "array", "items": {"type": "number"}, "minItems": DIM, "maxItems": DIM}
CHUNK_SCHEMA = {"type": "array", "items": ROW_SCHEMA, "minItems": CHUNK, "maxItems": CHUNK}


def _binary(value: float) -> float:
    return 1.0 if value >= 0.5 else -1.0


def validate_chunk(raw) -> tuple[np.ndarray, list[str]]:
    """A (CHUNK, DIM) float array clipped to [-1, 1], plus notes on what was changed. Raises ValueError if unusable."""
    try:
        chunk = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"actions must be {CHUNK} rows of {DIM} numbers") from exc
    if chunk.shape != (CHUNK, DIM):
        raise ValueError(f"actions must be {CHUNK} rows of {DIM} numbers, got shape {list(chunk.shape)}")
    if not np.isfinite(chunk).all():
        raise ValueError("actions contain NaN or infinite values")
    notes = []
    outside = int(np.sum(np.abs(chunk) > 1.0))
    if outside:
        notes.append(f"{outside} value(s) outside [-1, 1] were clipped")
    return np.clip(chunk, -1.0, 1.0), notes


def to_env(row: np.ndarray, split: dict = ENV_SPLIT, mode_index: int = ENV_MODE_INDEX, dim: int = DIM) -> np.ndarray:
    """One dataset-order action as the simulator's action vector (see the module docstring)."""
    row = np.asarray(row, dtype=float)
    out = np.zeros(dim)
    lo, hi = MODALITY["end_effector_position"]
    out[split["right"][0]:split["right"][0] + 3] = row[lo:hi]
    lo, hi = MODALITY["end_effector_rotation"]
    out[split["right"][0] + 3:split["right"][1]] = row[lo:hi]
    out[split["right_gripper"][0]] = _binary(row[MODALITY["gripper_close"][0]])
    base = row[MODALITY["base_motion"][0]:MODALITY["base_motion"][1]]
    out[split["base"][0]:split["base"][1]] = base[0:3]
    out[split["torso"][0]:split["torso"][1]] = base[3:4]
    out[mode_index] = _binary(row[MODALITY["control_mode"][0]])
    return out


def from_env(action: np.ndarray, split: dict = ENV_SPLIT, mode_index: int = ENV_MODE_INDEX) -> np.ndarray:
    """A simulator action vector in dataset order (inverse of :func:`to_env` for +-1 gripper and mode)."""
    action = np.asarray(action, dtype=float)
    row = np.zeros(DIM)
    right = action[split["right"][0]:split["right"][1]]
    row[MODALITY["end_effector_position"][0]:MODALITY["end_effector_position"][1]] = right[0:3]
    row[MODALITY["end_effector_rotation"][0]:MODALITY["end_effector_rotation"][1]] = right[3:6]
    row[MODALITY["gripper_close"][0]] = action[split["right_gripper"][0]]
    row[0:3] = action[split["base"][0]:split["base"][1]]
    row[3] = action[split["torso"][0]]
    row[MODALITY["control_mode"][0]] = action[mode_index]
    return row


def commanded_motion(chunk: np.ndarray) -> dict:
    """What a chunk should do in free space (measured scale): arm displacement (cm), rotation (deg), gripper."""
    chunk = np.asarray(chunk, dtype=float)
    arm_rows = chunk[chunk[:, 4] < 0.5]
    lo, hi = MODALITY["end_effector_position"]
    move = arm_rows[:, lo:hi].sum(axis=0) * CM_PER_STEP if len(arm_rows) else np.zeros(3)
    lo, hi = MODALITY["end_effector_rotation"]
    turn = arm_rows[:, lo:hi].sum(axis=0) * DEG_PER_STEP if len(arm_rows) else np.zeros(3)
    grip = chunk[:, 11] >= 0.5
    return {"arm_steps": len(arm_rows), "base_steps": len(chunk) - len(arm_rows),
            "move_cm": move, "turn_deg": turn,
            "gripper": "close" if grip.all() else "open" if not grip.any() else f"switches at step {int(np.argmax(grip != grip[0])) + 1}"}


def describe(chunk: np.ndarray) -> str:
    """One line summarising a chunk (used as the 'command' text in traces, memory and videos)."""
    c = commanded_motion(chunk)
    m, t = c["move_cm"], c["turn_deg"]
    text = (f"chunk {len(chunk)} steps: expected arm fwd {m[0]:+.1f} left {m[1]:+.1f} up {m[2]:+.1f} cm, "
            f"rot {t[0]:+.0f}/{t[1]:+.0f}/{t[2]:+.0f} deg, gripper {c['gripper']}")
    if c["base_steps"]:
        text += f", {c['base_steps']} base-mode steps"
    return text


def format_chunk(chunk: np.ndarray, decimals: int = 2) -> str:
    """A chunk as compact JSON rows (the form the model is asked to reply in)."""
    def num(v: float) -> str:
        v = round(float(v), decimals)
        return "0" if v == 0 else (str(int(v)) if v == int(v) else f"{v:g}")
    return "[" + ",\n ".join("[" + ",".join(num(v) for v in row) + "]" for row in np.asarray(chunk)) + "]"


INDEX_TABLE = (
    "index 0-3  base_motion: platform velocity forward, left, yaw and torso, -1..1 (0 = platform still)\n"
    "index 4    control_mode: -1 = arm mode (normal); +1 while the platform drives, so that the arm keeps its pose\n"
    "           relative to the moving platform (RoboCasa demonstrations use +1 exactly when base_motion is non-zero)\n"
    "index 5-7  end_effector_position: fingertip motion along (forward, left, up) of the robot frame, -1..1 per step\n"
    "index 8-10 end_effector_rotation: fingertip rotation about the (forward, left, up) axes of the robot frame,\n"
    "           -1..1 per step (right-hand rule: positive about 'up' turns the fingers counter-clockwise seen from above)\n"
    "index 11   gripper_close: -1 = open, +1 = close (the fingers need about 10-16 steps to finish)\n"
    "MEASURED SCALE (free space): an input v held for one step moves the fingertips about 1.2 x v cm\n"
    f"(about 1.1 x v cm up/down) and turns them about {DEG_PER_STEP} x v deg; so v = 1.0 for all 16 steps is about\n"
    "19 cm (or 100 deg), v = 0.25 for 16 steps about 5 cm, v = 0.05 for 16 steps about 1 cm. The arm keeps moving\n"
    "about 0.7 cm after the inputs return to 0. Inputs of 0 in arm mode hold the fingertips still."
)
