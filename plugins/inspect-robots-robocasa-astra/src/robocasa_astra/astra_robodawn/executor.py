"""Execute one discrete command on RoboCasa PandaOmron until the motion has finished (simulator side).

The model never emits controller values. Each command becomes a target pose of the fingertip
centre (robosuite ``grip_site``) or of the mobile base, and the executor servoes the native
HYBRID_MOBILE_BASE controller towards it step by step until the target is reached, progress
stalls, or the per-command step cap is hit. Every native step counts against the task's step
budget and is reported to ``on_step`` (video and replay log).

Robot frame: origin on the floor below the base centre site, x forward, y left, z up. Heights are
therefore heights above the floor; x/y are relative to the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .commands import DIRECTION_VECTORS, Command

# Native controller scales (default_pandaomron.json: OSC_POSE output_max).
ARM_POS_SCALE = 0.05  # metres per unit of arm input
ARM_ROT_SCALE = 0.5  # radians per unit of arm input
ARM_MAX_INPUT = 0.5  # cap per step (2.5 cm / 0.25 rad): full-scale steps destabilise OSC near singular postures
JOINT_LIMIT_MARGIN = 0.02  # fraction of a joint's range treated as "at the limit"

POS_TOL = 0.01
ROT_TOL = np.deg2rad(5.0)
MAX_ARM_STEPS = 80
MAX_SEGMENT_ROT = np.deg2rad(45.0)  # larger rotations are split; one 90 deg swing stalls where two 45 deg ones do not
STALL_STEPS = 10
STALL_POS_EPS = 0.002  # progress = position error down 2 mm ...
STALL_ROT_EPS = np.deg2rad(1.0)  # ... or orientation error down 1 deg within STALL_STEPS (tracked separately:
# while rotating, the fingertip often sags first, so a combined error falsely looked stalled)

GRIPPER_MAX_STEPS = 40
GRIPPER_STABLE_STEPS = 4
WAIT_STEPS = 10

BASE_POS_TOL = 0.02
BASE_YAW_TOL = np.deg2rad(2.0)
MAX_BASE_STEPS = 80
BASE_POS_GAIN = 8.0  # unit input per metre of remaining error
BASE_YAW_GAIN = 3.0  # unit input per radian



def _preset(approach, finger_axis) -> np.ndarray:
    """Gripper rotation (robot frame) from its approach (site z) and finger-closing axis (site x)."""
    z = np.asarray(approach, float) / np.linalg.norm(approach)
    x = np.asarray(finger_axis, float) / np.linalg.norm(finger_axis)
    return np.column_stack([x, np.cross(z, x), z])


# Full preset orientations (robot frame), all pitch rotations of the initial pose: the fingers close
# left-right. Measured: the Panda fingers separate along the grip site's x axis.
_S = np.sqrt(0.5)
POINT_PRESETS = {
    "down": _preset([0, 0, -1], [0, 1, 0]),
    "forward": _preset([1, 0, 0], [0, 1, 0]),
    "down45": _preset([_S, 0, -_S], [0, 1, 0]),
}
ROTATION_AXES = {"roll": np.array([1.0, 0, 0]), "pitch": np.array([0, 1.0, 0]), "yaw": np.array([0, 0, 1.0])}


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for ``angle`` radians about unit ``axis`` (Rodrigues)."""
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def rotation_vector(rot: np.ndarray) -> np.ndarray:
    """Axis-angle vector (axis * angle) of rotation matrix ``rot``."""
    cos = np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-6:
        return np.zeros(3)
    if np.pi - angle < 1e-4:  # near 180 deg: take the axis from the symmetric part
        w, v = np.linalg.eigh((rot + rot.T) / 2)
        return v[:, np.argmax(w)] * angle
    axis = np.array([rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]])
    return axis / (2 * np.sin(angle)) * angle


def yaw_of(rot: np.ndarray) -> float:
    return float(np.arctan2(rot[1, 0], rot[0, 0]))


def wrap(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class Pose:
    """World pose of the fingertip centre and of the base centre site."""

    tip: np.ndarray
    tip_rot: np.ndarray
    base: np.ndarray
    base_rot: np.ndarray

    def robot_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Fingertip position (x fwd, y left, z height above floor) and rotation in the robot frame."""
        rel = self.base_rot.T @ (self.tip - self.base)
        return np.array([rel[0], rel[1], self.tip[2]]), self.base_rot.T @ self.tip_rot


@dataclass
class ExecutionResult:
    """What a command achieved; ``note`` is the sentence shown to the model."""

    command: str
    kind: str
    ok: bool
    note: str = ""
    steps: int = 0
    details: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"command": self.command, "kind": self.kind, "ok": self.ok, "note": self.note,
                "steps": self.steps, **self.details}


class Executor:
    """Runs commands against one RoboCasa PandaOmron environment within a native step budget.

    ``on_step(action, obs, caption)`` is called after every native step; it is the hook for video
    recording and the replay log. ``success`` latches once the task checker has fired.
    """

    def __init__(self, env, step_budget: int, on_step: Optional[Callable] = None):
        self.env = env
        self.robot = env.robots[0]
        self.step_budget = int(step_budget)
        self.on_step = on_step
        self.steps_used = 0
        self.success = False
        self.gripper_cmd = -1.0  # -1 open, +1 close (robosuite GRIP convention)
        sim = env.sim
        self._tip_site = self.robot.eef_site_id["right"]
        self._base_site = sim.model.site_name2id(self.robot.robot_model.base.correct_naming("center"))
        joints = self.robot.gripper["right"].joints
        self._finger_addr = [sim.model.get_joint_qpos_addr(j) for j in joints]
        ranges = np.array([sim.model.jnt_range[sim.model.joint_name2id(j)] for j in joints])
        self._finger_span = float(np.sum(np.abs(ranges).max(axis=1)))
        self._finger_sign = np.array([1.0 if r[1] > 0 else -1.0 for r in ranges])
        self._arm_qpos = list(self.robot._ref_joint_pos_indexes)
        self._arm_limits = np.array([sim.model.jnt_range[sim.model.joint_name2id(n)] for n in self.robot.robot_joints])
        dim = env.action_dim
        self._mode_index = dim - 1
        self._idle = np.zeros(dim)
        tip, rot = self.pose().robot_frame()
        self.home_tip, self.home_rot = tip, rot

    # ------------------------------------------------------------------ state
    def pose(self) -> Pose:
        data = self.env.sim.data
        return Pose(
            tip=data.site_xpos[self._tip_site].copy(),
            tip_rot=data.site_xmat[self._tip_site].reshape(3, 3).copy(),
            base=data.site_xpos[self._base_site].copy(),
            base_rot=data.site_xmat[self._base_site].reshape(3, 3).copy(),
        )

    def gripper_opening(self) -> float:
        """Finger opening normalised to 0 (closed) .. 1 (fully open)."""
        q = np.array([self.env.sim.data.qpos[a] for a in self._finger_addr])
        return float(np.clip(np.sum(q * self._finger_sign) / self._finger_span, 0.0, 1.0))

    def state(self) -> dict:
        """Proprioceptive state shown to the model (robot frame, cm and unit vectors)."""
        pose = self.pose()
        tip, rot = pose.robot_frame()
        return {
            "fingertip_cm": np.round(tip * 100, 1).tolist(),
            "approach": np.round(rot[:, 2], 2).tolist(),  # direction the fingers point along
            "finger_axis": np.round(rot[:, 0], 2).tolist(),  # direction the fingers close along (site x)
            "gripper_opening": round(self.gripper_opening(), 2),
            "gripper_command": "closed" if self.gripper_cmd > 0 else "open",
            "base_world_xy_cm": np.round(pose.base[:2] * 100, 1).tolist(),
            "base_yaw_deg": round(float(np.degrees(yaw_of(pose.base_rot))), 1),
            "steps_used": self.steps_used,
            "step_budget": self.step_budget,
        }

    def joints_at_limit(self) -> list[int]:
        """1-based indices of arm joints within JOINT_LIMIT_MARGIN of either end of their range."""
        q = self.env.sim.data.qpos[self._arm_qpos]
        lo, hi = self._arm_limits[:, 0], self._arm_limits[:, 1]
        frac = (q - lo) / (hi - lo)
        return [i + 1 for i, f in enumerate(frac) if f < JOINT_LIMIT_MARGIN or f > 1 - JOINT_LIMIT_MARGIN]

    @property
    def budget_left(self) -> int:
        return self.step_budget - self.steps_used

    # ------------------------------------------------------------------ stepping
    def _step(self, action: np.ndarray, caption: str) -> None:
        obs, _, _, _ = self.env.step(action)
        self.steps_used += 1
        if not self.success and bool(self.env._check_success()):
            self.success = True
        if self.on_step:
            self.on_step(action, obs, caption)

    def _stop(self) -> Optional[str]:
        if self.success:
            return "task_success"
        if self.steps_used >= self.step_budget:
            return "budget_exhausted"
        return None

    def _arm_action(self, target_tip: np.ndarray, target_rot: np.ndarray, pose: Pose) -> np.ndarray:
        action = self._idle.copy()
        pos = pose.base_rot.T @ (target_tip - pose.tip) / ARM_POS_SCALE
        rot = pose.base_rot.T @ rotation_vector(target_rot @ pose.tip_rot.T) / ARM_ROT_SCALE
        action[0:3] = np.clip(pos, -ARM_MAX_INPUT, ARM_MAX_INPUT)
        action[3:6] = np.clip(rot, -ARM_MAX_INPUT, ARM_MAX_INPUT)
        action[6] = self.gripper_cmd
        action[self._mode_index] = -1.0
        return action

    # ------------------------------------------------------------------ commands
    def execute(self, cmd: Command) -> ExecutionResult:
        """Run ``cmd`` to completion and describe the outcome honestly."""
        text = cmd.text()
        stop = self._stop()
        if stop:
            return ExecutionResult(text, cmd.kind, False, f"not executed: {stop.replace('_', ' ')}")
        if cmd.kind in ("move", "rotate", "point", "home"):
            result = self._run_arm(cmd)
        elif cmd.kind == "gripper":
            result = self._run_gripper(cmd)
        elif cmd.kind in ("base_move", "base_turn"):
            result = self._run_base(cmd)
        elif cmd.kind == "wait":
            result = self._run_wait(cmd)
        else:
            raise ValueError(f"{cmd.kind} is handled by the agent loop, not the executor")
        if cmd.clipped:
            result.note += " (magnitude was clipped to the limit)"
        result.details["task_success"] = self.success
        result.details["steps_used_after"] = self.steps_used
        return result

    def _arm_target(self, cmd: Command, pose: Pose) -> tuple[np.ndarray, np.ndarray]:
        tip, rot = pose.tip.copy(), pose.tip_rot.copy()
        if cmd.kind == "move":
            tip = tip + pose.base_rot @ (np.array(DIRECTION_VECTORS[cmd.name]) * cmd.value / 100.0)
        elif cmd.kind == "rotate":
            axis = pose.base_rot @ ROTATION_AXES[cmd.name]
            rot = axis_angle_matrix(axis, np.radians(cmd.value)) @ rot
        elif cmd.kind == "point":
            rot = pose.base_rot @ POINT_PRESETS[cmd.name]
        elif cmd.kind == "home":
            rel = self.home_tip.copy()
            rel[2] -= pose.base[2]
            tip = pose.base + pose.base_rot @ rel
            rot = pose.base_rot @ self.home_rot
        return tip, rot

    def _waypoints(self, start: Pose, target_tip: np.ndarray, target_rot: np.ndarray) -> list[tuple]:
        """Split rotations larger than MAX_SEGMENT_ROT into equal sub-rotations (each servoed in turn)."""
        delta = rotation_vector(target_rot @ start.tip_rot.T)
        n = max(1, int(np.ceil(np.linalg.norm(delta) / MAX_SEGMENT_ROT)))
        out = []
        for k in range(1, n + 1):
            frac = k / n
            angle = np.linalg.norm(delta) * frac
            rot = axis_angle_matrix(delta, angle) @ start.tip_rot if angle > 0 else start.tip_rot
            out.append((start.tip + (target_tip - start.tip) * frac, rot, k == n))
        return out

    def _run_arm(self, cmd: Command) -> ExecutionResult:
        start = self.pose()
        target_tip, target_rot = self._arm_target(cmd, start)
        steps = 0
        stop_reason = "max_steps"
        for tip_goal, rot_goal, final in self._waypoints(start, target_tip, target_rot):
            best_pos, best_rot, best_step = np.inf, np.inf, steps
            pos_tol, rot_tol = (POS_TOL, ROT_TOL) if final else (2 * POS_TOL, 2 * ROT_TOL)
            stop_reason = "max_steps"
            while steps < MAX_ARM_STEPS:
                if self._stop():
                    stop_reason = self._stop()
                    break
                pose = self.pose()
                pos_err = np.linalg.norm(tip_goal - pose.tip)
                rot_err = np.linalg.norm(rotation_vector(rot_goal @ pose.tip_rot.T))
                if pos_err < pos_tol and rot_err < rot_tol:
                    stop_reason = "reached"
                    break
                if pos_err < best_pos - STALL_POS_EPS or rot_err < best_rot - STALL_ROT_EPS:
                    best_pos, best_rot, best_step = min(best_pos, pos_err), min(best_rot, rot_err), steps
                elif steps - best_step >= STALL_STEPS:
                    stop_reason = "stalled"
                    break
                self._step(self._arm_action(tip_goal, rot_goal, pose), cmd.text())
                steps += 1
            if stop_reason != "reached":
                break
        end = self.pose()
        pos_err = float(np.linalg.norm(target_tip - end.tip))
        rot_err = float(np.degrees(np.linalg.norm(rotation_vector(target_rot @ end.tip_rot.T))))
        moved = end.base_rot.T @ (end.tip - start.tip)
        turned = float(np.degrees(np.linalg.norm(rotation_vector(end.tip_rot @ start.tip_rot.T))))
        details = {
            "stop_reason": stop_reason,
            "moved_cm": np.round(moved * 100, 1).tolist(),
            "turned_deg": round(turned, 1),
            "position_error_cm": round(pos_err * 100, 1),
            "orientation_error_deg": round(rot_err, 1),
        }
        ok = pos_err < 1.5 * POS_TOL and np.radians(rot_err) < 1.6 * ROT_TOL
        limited = self.joints_at_limit()
        details["joints_at_limit"] = limited
        if ok:
            note = "reached"
        elif np.linalg.norm(moved) < 0.005 and turned < 2.0:
            note = (f"the arm did NOT move (blocked or out of reach); remaining error {pos_err * 100:.1f} cm / "
                    f"{rot_err:.0f} deg")
        else:
            note = (f"only part of the way: fingertips moved forward {moved[0] * 100:+.1f}, left {moved[1] * 100:+.1f}, "
                    f"up {moved[2] * 100:+.1f} cm and turned {turned:.0f} deg; remaining error {pos_err * 100:.1f} cm / "
                    f"{rot_err:.0f} deg")
        if not ok:
            note += ("; cause: an arm joint is at its limit (posture), so this pose is not reachable this way - "
                     "move the fingertips further from / closer to the robot body first, change the gripper "
                     "orientation, or move the base" if limited else
                     "; cause: probably contact with an object or fixture")
        if stop_reason in ("task_success", "budget_exhausted"):
            note += f"; stopped early: {stop_reason.replace('_', ' ')}"
        return ExecutionResult(cmd.text(), cmd.kind, bool(ok), note, steps, details)

    def _run_gripper(self, cmd: Command) -> ExecutionResult:
        self.gripper_cmd = 1.0 if cmd.name == "close" else -1.0
        hold = self.pose()
        prev, stable, steps = self.gripper_opening(), 0, 0
        while steps < GRIPPER_MAX_STEPS and not self._stop():
            self._step(self._arm_action(hold.tip, hold.tip_rot, self.pose()), cmd.text())
            steps += 1
            cur = self.gripper_opening()
            stable = stable + 1 if abs(cur - prev) < 2e-3 else 0
            prev = cur
            if stable >= GRIPPER_STABLE_STEPS:
                break
        opening = self.gripper_opening()
        settled = stable >= GRIPPER_STABLE_STEPS
        if not settled:
            note = f"fingers still moving at opening {opening:.2f} when the command ended"
        elif cmd.name == "close":
            note = (f"fingers stopped at opening {opening:.2f}: something stops them - an object between the fingers "
                    "(grasped) or a finger pressing on a surface; check the wrist camera"
                    if opening > 0.06 else "fingers closed fully: nothing between them")
        else:
            note = f"gripper opened to {opening:.2f}"
        return ExecutionResult(cmd.text(), cmd.kind, True, note, steps,
                               {"gripper_opening": round(opening, 2), "settled": settled})

    def _run_base(self, cmd: Command) -> ExecutionResult:
        start = self.pose()
        start_yaw = yaw_of(start.base_rot)
        if cmd.kind == "base_move":
            offset = np.array(DIRECTION_VECTORS[cmd.name]) * cmd.value / 100.0
            target_xy = start.base[:2] + (start.base_rot @ offset)[:2]
            target_yaw = start_yaw
        else:
            target_xy = start.base[:2].copy()
            target_yaw = wrap(start_yaw + np.radians(cmd.value))
        steps = 0
        stop_reason = "max_steps"
        while steps < MAX_BASE_STEPS:
            if self._stop():
                stop_reason = self._stop()
                break
            pose = self.pose()
            err_xy = (pose.base_rot.T @ np.append(target_xy - pose.base[:2], 0.0))[:2]
            err_yaw = wrap(target_yaw - yaw_of(pose.base_rot))
            if np.linalg.norm(err_xy) < BASE_POS_TOL and abs(err_yaw) < BASE_YAW_TOL:
                stop_reason = "reached"
                break
            action = self._idle.copy()
            action[6] = self.gripper_cmd
            action[7:9] = np.clip(BASE_POS_GAIN * err_xy, -1, 1)
            action[9] = np.clip(BASE_YAW_GAIN * err_yaw, -1, 1)
            action[self._mode_index] = 1.0
            self._step(action, cmd.text())
            steps += 1
        end = self.pose()
        moved = start.base_rot.T @ np.append(end.base[:2] - start.base[:2], 0.0)
        turned = float(np.degrees(wrap(yaw_of(end.base_rot) - start_yaw)))
        err_xy = float(np.linalg.norm(target_xy - end.base[:2]))
        err_yaw = float(np.degrees(abs(wrap(target_yaw - yaw_of(end.base_rot)))))
        ok = err_xy < 1.5 * BASE_POS_TOL and err_yaw < 1.5 * np.degrees(BASE_YAW_TOL)
        note = "reached" if ok else (
            f"base moved only forward {moved[0] * 100:+.1f}, left {moved[1] * 100:+.1f} cm and turned {turned:+.0f} deg; "
            f"remaining error {err_xy * 100:.1f} cm / {err_yaw:.0f} deg (probably blocked by a counter or wall)")
        details = {"stop_reason": stop_reason, "base_moved_cm": np.round(moved[:2] * 100, 1).tolist(),
                   "base_turned_deg": round(turned, 1)}
        return ExecutionResult(cmd.text(), cmd.kind, bool(ok), note, steps, details)

    def _run_wait(self, cmd: Command) -> ExecutionResult:
        hold = self.pose()
        steps = 0
        while steps < WAIT_STEPS and not self._stop():
            self._step(self._arm_action(hold.tip, hold.tip_rot, self.pose()), cmd.text())
            steps += 1
        return ExecutionResult(cmd.text(), cmd.kind, True, "waited", steps)
