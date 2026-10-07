"""Convert a RoboCasa expert demonstration (LeRobot v2.1) into discrete command turns (no simulator).

RoboDawn reduces each expert trajectory to end-effector waypoints and gripper events and re-expresses
it in its own command space. Here:

1. the episode is split at gripper toggles and at base-motion segments;
2. inside each arm segment the fingertip path is simplified to waypoints (Ramer-Douglas-Peucker,
   ``WAYPOINT_TOL_CM``) plus points where the gripper orientation changed by more than
   ``ORIENT_TOL_DEG``;
3. every waypoint transition becomes one turn: rotation first (``point`` preset when the target is
   close to one, else ``rotate`` about the dominant robot axes), then translations (up first when
   rising, horizontal first when descending), each at most 20 cm; gripper toggles and base motions
   become their own turns.

State conventions (checked against the live simulator in P5): ``observation.state`` holds base
position/rotation, the fingertip (grip site) position and the hand-body xyzw quaternion relative to
the base centre site (see ``SITE_FROM_EEF``), and the two finger joints; the base centre is 0.70 m
above the floor.
"""

from __future__ import annotations

import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .executor import POINT_PRESETS, axis_angle_matrix, rotation_vector

DATASET_ROOT = Path("/data/code/robocasa/datasets/v1.0")
TASK_DATASETS = {
    "OpenCabinet": "pretrain/atomic/OpenCabinet",
    "PickPlaceSinkToCounter": "pretrain/atomic/PickPlaceSinkToCounter",
    "PrepareCoffee": "pretrain/composite/PrepareCoffee",
    "PanTransfer": "target/composite/PanTransfer",
    "StirVegetables": "pretrain/composite/StirVegetables",
}
EVAL_LAYOUT_STYLE = (1, 1)  # evaluation scene; demonstrations must come from other kitchens
BASE_HEIGHT_M = 0.70
WAYPOINT_TOL_CM = 2.0
ORIENT_TOL_DEG = 15.0
PRESET_TOL_DEG = 12.0
MIN_MOVE_CM = 1.5
MIN_ROTATE_DEG = 8.0
FINGER_SPAN_M = 0.08
# The dataset quaternion is robot0_base_to_eef_quat (hand body), not the grip site the executor controls.
# Measured from one live observation: R_site = R_eef @ SITE_FROM_EEF (a -90 deg turn about the approach axis).
SITE_FROM_EEF = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def quat_xyzw_to_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class Episode:
    task: str
    index: int
    root: Path
    ep_meta: dict
    tip_cm: np.ndarray  # (T, 3) robot frame, height above floor
    rot: np.ndarray  # (T, 3, 3) gripper rotation in the robot frame
    grip_close: np.ndarray  # (T,) bool, commanded
    opening: np.ndarray  # (T,) measured 0..1
    base_xy: np.ndarray  # (T, 2) world, m
    base_yaw: np.ndarray  # (T,) rad
    base_active: np.ndarray  # (T,) bool

    @property
    def length(self) -> int:
        return len(self.tip_cm)

    def video(self, camera: str = "robot0_agentview_left") -> Path:
        return self.root / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{self.index:06d}.mp4"


def dataset_dir(task: str) -> Path:
    return Path(sorted(glob.glob(str(DATASET_ROOT / TASK_DATASETS[task] / "*" / "lerobot")))[0])


def load_episode(task: str, index: int) -> Episode:
    import pandas as pd

    root = dataset_dir(task)
    df = pd.read_parquet(root / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
    s = np.stack(df["observation.state"].values)
    a = np.stack(df["action"].values)
    meta = json.loads((root / "extras" / f"episode_{index:06d}" / "ep_meta.json").read_text())
    tip = s[:, 7:10].copy()
    tip[:, 2] += BASE_HEIGHT_M
    base_rot = [quat_xyzw_to_matrix(q) for q in s[:, 3:7]]
    return Episode(
        task=task, index=index, root=root, ep_meta=meta, tip_cm=tip * 100,
        rot=np.stack([quat_xyzw_to_matrix(q) @ SITE_FROM_EEF for q in s[:, 10:14]]),
        grip_close=a[:, 11] > 0,
        opening=np.clip((s[:, 14] - s[:, 15]) / FINGER_SPAN_M, 0, 1),
        base_xy=s[:, 0:2], base_yaw=np.array([np.arctan2(r[1, 0], r[0, 0]) for r in base_rot]),
        base_active=np.abs(a[:, 0:4]).max(axis=1) > 0.05,
    )


def candidate_episodes(task: str, instruction_hint: str | None = None) -> list[dict]:
    """Episodes outside the evaluation kitchen, ranked: same instruction kind, no base motion, median length."""
    root = dataset_dir(task)
    rows = []
    for path in sorted((root / "data" / "chunk-000").glob("episode_*.parquet")):
        index = int(path.stem.split("_")[1])
        meta = json.loads((root / "extras" / f"episode_{index:06d}" / "ep_meta.json").read_text())
        if (meta.get("layout_id"), meta.get("style_id")) == EVAL_LAYOUT_STYLE or meta.get("layout_id") == EVAL_LAYOUT_STYLE[0]:
            continue
        rows.append({"index": index, "layout": meta.get("layout_id"), "style": meta.get("style_id"),
                     "lang": meta.get("lang", ""), "path": path})
    import pandas as pd

    lengths = {}
    for row in rows:
        df = pd.read_parquet(row["path"], columns=["action"])
        act = np.stack(df["action"].values)
        lengths[row["index"]] = len(df)
        row["length"] = len(df)
        row["uses_base"] = bool((np.abs(act[:, 0:4]).max(axis=1) > 0.05).any())
        row["gripper_toggles"] = int(np.sum(np.diff((act[:, 11] > 0).astype(int)) != 0))
    median = float(np.median(list(lengths.values())))
    toggles = float(np.median([r["gripper_toggles"] for r in rows]))
    for row in rows:
        row["score"] = (
            (0 if instruction_hint is None or row["lang"].strip() == instruction_hint.strip() else 10)
            + (5 if row["uses_base"] else 0)
            + abs(row["gripper_toggles"] - toggles)
            + abs(row["length"] - median) / median
        )
        row.pop("path")
    return sorted(rows, key=lambda r: r["score"])


def _rdp(points: np.ndarray, tol: float) -> list[int]:
    """Indices kept by Ramer-Douglas-Peucker simplification of a 3-D polyline."""
    if len(points) < 3:
        return list(range(len(points)))
    start, end = points[0], points[-1]
    seg = end - start
    norm = np.linalg.norm(seg)
    if norm < 1e-9:
        dists = np.linalg.norm(points - start, axis=1)
    else:
        dists = np.linalg.norm(np.cross(points - start, seg / norm), axis=1)
    i = int(np.argmax(dists))
    if dists[i] <= tol:
        return [0, len(points) - 1]
    left = _rdp(points[: i + 1], tol)
    right = _rdp(points[i:], tol)
    return left[:-1] + [j + i for j in right]


def _segments(ep: Episode) -> list[tuple[int, int, str]]:
    """(start, end, kind) with kind 'arm' or 'base'; gripper toggles are boundaries."""
    bounds = {0, ep.length - 1}
    bounds.update(int(i) + 1 for i in np.flatnonzero(np.diff(ep.grip_close.astype(int)) != 0))
    bounds.update(int(i) + 1 for i in np.flatnonzero(np.diff(ep.base_active.astype(int)) != 0))
    bounds = sorted(bounds)
    out = []
    for a, b in zip(bounds, bounds[1:]):
        out.append((a, b, "base" if ep.base_active[a:b].mean() > 0.5 else "arm"))
    return out


def _waypoints(ep: Episode, a: int, b: int) -> list[int]:
    keep = {a + i for i in _rdp(ep.tip_cm[a : b + 1], WAYPOINT_TOL_CM)}
    last = a
    for t in range(a + 1, b + 1):
        if np.degrees(np.linalg.norm(rotation_vector(ep.rot[t] @ ep.rot[last].T))) > ORIENT_TOL_DEG:
            keep.add(t)
            last = t
    return sorted(keep)


def _nearest_preset(rot: np.ndarray) -> tuple[str, float]:
    best = min(POINT_PRESETS, key=lambda k: np.linalg.norm(rotation_vector(POINT_PRESETS[k] @ rot.T)))
    return best, float(np.degrees(np.linalg.norm(rotation_vector(POINT_PRESETS[best] @ rot.T))))


def rotation_commands(current: np.ndarray, target: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Commands turning ``current`` towards ``target``; returns them and the orientation they achieve."""
    preset, err = _nearest_preset(target)
    if err < PRESET_TOL_DEG and np.degrees(np.linalg.norm(rotation_vector(target @ current.T))) > MIN_ROTATE_DEG:
        return [f"point {preset}"], POINT_PRESETS[preset]
    delta = rotation_vector(target @ current.T)  # robot-frame rotation vector
    commands, achieved = [], current
    for axis, name in enumerate(("roll", "pitch", "yaw")):
        deg = float(np.degrees(delta[axis]))
        if abs(deg) >= MIN_ROTATE_DEG:
            deg = float(np.clip(5 * round(deg / 5), -90, 90))
            commands.append(f"rotate {name} {deg:+g}")
            axis_vec = np.eye(3)[axis]
            achieved = axis_angle_matrix(axis_vec, np.radians(deg)) @ achieved
    return commands, achieved


def move_commands(delta_cm: np.ndarray) -> list[str]:
    """Axis moves for a fingertip displacement: rise first, descend last, each chunk <= 20 cm."""
    names = {0: ("forward", "back"), 1: ("left", "right"), 2: ("up", "down")}
    order = [2, 0, 1] if delta_cm[2] > 0 else [0, 1, 2]
    out = []
    for axis in order:
        d = float(delta_cm[axis])
        if abs(d) < MIN_MOVE_CM:
            continue
        word = names[axis][0] if d > 0 else names[axis][1]
        remaining = round(abs(d))
        while remaining > 0:
            step = min(20, remaining)
            out.append(f"move {word} {step}")
            remaining -= step
    return out


@dataclass
class DemoTurn:
    commands: list[str]
    t_start: int
    t_end: int
    kind: str  # arm, gripper, base
    notes: dict = field(default_factory=dict)


def convert(ep: Episode, max_commands: int = 4) -> list[DemoTurn]:
    """The expert episode as command turns (rotation/translation per waypoint, gripper and base turns)."""
    turns: list[DemoTurn] = []
    rot_cmd = ep.rot[0]
    for a, b, kind in _segments(ep):
        if a > 0 and ep.grip_close[a] != ep.grip_close[a - 1]:
            turns.append(DemoTurn([f"gripper {'close' if ep.grip_close[a] else 'open'}"], a, a, "gripper"))
        if kind == "base":
            yaw0 = ep.base_yaw[a]
            rot0 = np.array([[np.cos(yaw0), -np.sin(yaw0)], [np.sin(yaw0), np.cos(yaw0)]])
            d = rot0.T @ (ep.base_xy[b] - ep.base_xy[a]) * 100
            cmds = []
            for value, (pos, neg) in ((d[0], ("forward", "back")), (d[1], ("left", "right"))):
                if abs(value) >= 3:
                    cmds.append(f"base {pos if value > 0 else neg} {min(50, round(abs(value)))}")
            turn = float(np.degrees(np.arctan2(np.sin(ep.base_yaw[b] - yaw0), np.cos(ep.base_yaw[b] - yaw0))))
            if abs(turn) >= 3:
                cmds.append(f"base turn {np.clip(round(turn), -45, 45):+g}")
            if cmds:
                turns.append(DemoTurn(cmds, a, b, "base"))
            continue
        points = _waypoints(ep, a, b)
        for p, q in zip(points, points[1:]):
            rcmds, rot_cmd = rotation_commands(rot_cmd, ep.rot[q])
            mcmds = move_commands(ep.tip_cm[q] - ep.tip_cm[p])
            cmds = rcmds + mcmds
            for i in range(0, len(cmds), max_commands):
                turns.append(DemoTurn(cmds[i : i + max_commands], p, q, "arm"))
    return [t for t in turns if t.commands]


def key_events(ep: Episode) -> list[dict]:
    """Start, every gripper toggle and the end: time, fingertip, approach, finger axis, opening (for authoring)."""
    idx = [0] + [int(i) + 1 for i in np.flatnonzero(np.diff(ep.grip_close.astype(int)) != 0)] + [ep.length - 1]
    out = []
    for t in idx:
        label = "start" if t == 0 else "end" if t == ep.length - 1 else ("close" if ep.grip_close[t] else "open")
        settle = min(t + 15, ep.length - 1) if label in ("close", "open") else t
        out.append({
            "t": t, "event": label, "fingertip_cm": np.round(ep.tip_cm[t], 1).tolist(),
            "approach": np.round(ep.rot[t][:, 2], 2).tolist(), "finger_axis": np.round(ep.rot[t][:, 0], 2).tolist(),
            "opening_after": round(float(ep.opening[settle]), 2),
            "base_moved_before_cm": np.round((ep.base_xy[t] - ep.base_xy[0]) * 100, 1).tolist(),
        })
    return out
