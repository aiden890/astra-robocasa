"""Open-loop execution of 16-step action chunks on the same executor the skill variant uses.

:class:`ChunkExecutor` inherits :class:`astra_robodawn.executor.Executor`, so stepping, the success
latch,
the video / replay hooks, the step budget and the proprioceptive state are identical in both
variants; only
the source of the per-step action differs (the model's chunk instead of a servo towards a command's
target).
"""

from __future__ import annotations

import numpy as np

from ..astra_robodawn.executor import Executor, rotation_vector
from .action_format import CHUNK, commanded_motion, describe, to_env

# RoboCasa365 observation.state, in dataset order (meta/modality.json "state")
STATE_KEYS = (
    ("base_position", "robot0_base_pos", 3),
    ("base_rotation", "robot0_base_quat", 4),
    ("end_effector_position_relative", "robot0_base_to_eef_pos", 3),
    ("end_effector_rotation_relative", "robot0_base_to_eef_quat", 4),
    ("gripper_qpos", "robot0_gripper_qpos", 2),
)


def dataset_state(obs: dict) -> list[float]:
    """The 16-D RoboCasa365 ``observation.state`` from a robosuite observation dict."""
    return [round(float(v), 4) for _, key, _ in STATE_KEYS for v in np.asarray(obs[key]).ravel()]


class ChunkExecutor(Executor):
    """Runs dataset-order action chunks step by step (no servoing, no correction: like a VLA)."""

    def __init__(self, env, step_budget: int, on_step=None):
        self.before_step = None
        self.last_obs: dict | None = None
        self._outer_on_step = on_step
        super().__init__(env, step_budget, self._remember_obs)
        controller = self.robot.composite_controller
        self._split = {
            name: tuple(controller._action_split_indexes[name])
            for name in ("right", "right_gripper", "base", "torso")
        }

    def _remember_obs(self, action, obs, caption) -> None:
        self.last_obs = obs
        if self._outer_on_step:
            self._outer_on_step(action, obs, caption)

    def dataset_state(self) -> list[float]:
        """Return the latest native state in the dataset observation order."""
        if self.last_obs is None:
            self.last_obs = self.env._get_observations(force_update=True)
        return dataset_state(self.last_obs)

    def run_chunk(self, chunk: np.ndarray, notes: list[str] | None = None) -> dict:
        """Execute every row of ``chunk`` (dataset order); stop early only on task success or
        budget."""
        chunk = np.asarray(chunk, dtype=float)
        text = describe(chunk)
        start, opening_before = self.pose(), self.gripper_opening()
        executed, stop = 0, None
        for k, row in enumerate(chunk):
            stop = self._stop()
            if stop:
                break
            action = to_env(row, self._split, self._mode_index, len(self._idle))
            self.gripper_cmd = action[self._split["right_gripper"][0]]
            if self.before_step:
                self.before_step(action)
            self._step(action, f"chunk step {k + 1}/{len(chunk)}")
            executed += 1
        end = self.pose()
        moved = (
            end.robot_frame()[0] - start.robot_frame()[0]
        ) * 100  # relative to the platform, as in the state
        turned = float(np.degrees(np.linalg.norm(rotation_vector(end.tip_rot @ start.tip_rot.T))))
        base_moved = start.base_rot.T @ (end.base - start.base) * 100
        asked = commanded_motion(chunk[:executed] if executed else np.zeros((1, chunk.shape[1])))
        opening = self.gripper_opening()
        note = (
            f"executed {executed}/{len(chunk)} steps; fingertips moved forward "
            f"{moved[0]:+.1f}, left {moved[1]:+.1f}, "
            f"up {moved[2]:+.1f} cm and turned {turned:.0f} deg (expected in free space: forward "
            f"{asked['move_cm'][0]:+.1f}, left {asked['move_cm'][1]:+.1f}, "
            f"up {asked['move_cm'][2]:+.1f} cm); gripper opening "
            f"{opening_before:.2f} -> {opening:.2f}"
        )
        if np.linalg.norm(base_moved[:2]) > 0.5:
            note += f"; platform moved forward {base_moved[0]:+.1f}, left {base_moved[1]:+.1f} cm"
        limited = self.joints_at_limit()
        if limited:
            note += f"; arm joint(s) {limited} at their limit"
        if notes:
            note += "; " + "; ".join(notes)
        if stop:
            note += f"; stopped early: {stop.replace('_', ' ')}"
        return {
            "command": text,
            "kind": "chunk",
            "ok": executed == len(chunk) or self.success,
            "note": note,
            "steps": executed,
            "moved_cm": np.round(moved, 1).tolist(),
            "turned_deg": round(turned, 1),
            "gripper_opening": round(opening, 2),
            "gripper_closed": bool(self.gripper_cmd > 0),
            "base_moved_cm": np.round(base_moved[:2], 1).tolist(),
            "joints_at_limit": limited,
            "task_success": self.success,
            "steps_used_after": self.steps_used,
            "chunk_length": CHUNK,
        }
