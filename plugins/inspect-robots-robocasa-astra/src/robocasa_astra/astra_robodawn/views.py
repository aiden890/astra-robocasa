"""Annotated camera views shown to the model each turn (simulator side).

RoboDawn's ablation found the metric grid and fingertip markers among the most important harness
components (position estimates from 5-10 cm down to 2-3 cm error). Here the two robot-mounted
overview cameras get, in the robot frame:

* a 10 cm grid on the work surface (the most common surface height in front of the robot), labelled
  ``F<cm>`` (distance forward) and ``L<cm>`` (distance to the left, negative = right);
* the fingertip centre as a cyan circle, with a dashed line down to its "shadow" on the surface;
* a small axis legend (forward red, left green, up blue).

The wrist camera is passed through unannotated, for final alignment before grasping.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

OVERVIEW_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right")
WRIST_CAMERA = "robot0_eye_in_hand"
OVERVIEW_SIZE = 384
WRIST_SIZE = 256
GRID_X_CM = range(20, 101, 10)  # forward
GRID_Y_CM = range(-60, 61, 10)  # left
GRID_COLOR = (150, 150, 150)
LABEL_COLOR = (255, 255, 0)
TIP_COLOR = (0, 220, 255)

CAPTIONS = {
    "robot0_agentview_left": "left overview camera (mounted on the robot, looking forward over its left side)",
    "robot0_agentview_right": "right overview camera (mounted on the robot, looking forward over its right side)",
    WRIST_CAMERA: "wrist camera (looking along the fingers; use it to check alignment before grasping)",
}


@dataclass
class View:
    name: str
    caption: str
    image: Image.Image

    def png_base64(self) -> str:
        buf = io.BytesIO()
        self.image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()


def surface_height(env, base: np.ndarray, base_rot: np.ndarray) -> float:
    """Most common surface height (m) under a patch 35-75 cm in front of the robot (vertical rays)."""
    import mujoco

    model, data = env.sim.model._model, env.sim.data._data
    hits = []
    geomid = np.zeros(1, dtype=np.int32)
    for fx in np.arange(0.35, 0.76, 0.1):
        for fy in np.arange(-0.4, 0.41, 0.1):
            start = base + base_rot @ np.array([fx, fy, 0.0])
            start[2] = 1.35  # below typical wall cabinets, above counters
            dist = mujoco.mj_ray(model, data, start, np.array([0.0, 0.0, -1.0]), None, 1, -1, geomid)
            if dist > 0:
                hits.append(round(start[2] - dist, 2))
    if not hits:
        return 0.92
    values, counts = np.unique(hits, return_counts=True)
    return float(values[np.argmax(counts)])


class Projector:
    """World point -> pixel (u = column, v = row) for one camera image of the given size."""

    def __init__(self, env, camera: str, size: int):
        from robosuite.utils.camera_utils import get_camera_transform_matrix

        self.matrix = get_camera_transform_matrix(env.sim, camera, size, size)

    def __call__(self, points: np.ndarray) -> list[tuple[float, float] | None]:
        pts = np.concatenate([np.asarray(points, float), np.ones((len(points), 1))], axis=1)
        pix = (self.matrix @ pts.T).T
        out = []
        for u, v, w, _ in pix:
            out.append((u / w, v / w) if w > 1e-6 else None)
        return out


def _text(draw: ImageDraw.ImageDraw, xy, text: str, fill) -> None:
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill)


def _robot_point(base: np.ndarray, base_rot: np.ndarray, x_cm: float, y_cm: float, z: float) -> np.ndarray:
    p = base + base_rot @ np.array([x_cm / 100.0, y_cm / 100.0, 0.0])
    p[2] = z
    return p


def annotate(env, camera: str, tip: np.ndarray, base: np.ndarray, base_rot: np.ndarray, surface_z: float,
             image: Image.Image | None = None) -> Image.Image:
    """Draw the surface grid, fingertip marker and axis legend on ``camera``'s view.

    By default the view is rendered from ``env``; pass ``image`` (e.g. a recorded frame of the same
    robot-mounted camera, square) to annotate it instead, with ``env`` only providing the projection.
    """
    if image is None:
        image = Image.fromarray(env.sim.render(width=OVERVIEW_SIZE, height=OVERVIEW_SIZE, camera_name=camera)[::-1].copy())
    img = image.copy()
    draw = ImageDraw.Draw(img)
    size = img.size[0]
    project = Projector(env, camera, size)

    def inside(p) -> bool:
        return p is not None and -size < p[0] < 2 * size and -size < p[1] < 2 * size

    for x in GRID_X_CM:
        pts = project([_robot_point(base, base_rot, x, y, surface_z) for y in (GRID_Y_CM[0], GRID_Y_CM[-1])])
        if all(inside(p) for p in pts):
            draw.line([pts[0], pts[1]], fill=GRID_COLOR, width=1)
    for y in GRID_Y_CM:
        pts = project([_robot_point(base, base_rot, x, y, surface_z) for x in (GRID_X_CM[0], GRID_X_CM[-1])])
        if all(inside(p) for p in pts):
            draw.line([pts[0], pts[1]], fill=GRID_COLOR, width=1)
    # labels: F<x> along the y = 0 line, L<y> along the nearest grid row
    for x in GRID_X_CM[1::2]:
        p = project([_robot_point(base, base_rot, x, 0, surface_z)])[0]
        if p and 0 <= p[0] < size and 0 <= p[1] < size:
            _text(draw, (p[0] + 2, p[1] - 12), f"F{x}", LABEL_COLOR)
    for y in GRID_Y_CM[::2]:
        if y == 0:
            continue
        p = project([_robot_point(base, base_rot, GRID_X_CM[1], y, surface_z)])[0]
        if p and 0 <= p[0] < size and 0 <= p[1] < size:
            _text(draw, (p[0] - 10, p[1] + 2), f"L{y:+d}", LABEL_COLOR)

    # fingertip marker and its shadow on the work surface
    shadow = tip.copy()
    shadow[2] = surface_z
    tp, sp = project([tip, shadow])
    if tp and sp:
        steps = 12
        for i in range(0, steps, 2):
            a = (tp[0] + (sp[0] - tp[0]) * i / steps, tp[1] + (sp[1] - tp[1]) * i / steps)
            b = (tp[0] + (sp[0] - tp[0]) * (i + 1) / steps, tp[1] + (sp[1] - tp[1]) * (i + 1) / steps)
            draw.line([a, b], fill=TIP_COLOR, width=1)
        draw.ellipse([sp[0] - 2, sp[1] - 2, sp[0] + 2, sp[1] + 2], fill=TIP_COLOR)
        draw.ellipse([tp[0] - 6, tp[1] - 6, tp[0] + 6, tp[1] + 6], outline=TIP_COLOR, width=2)
        _text(draw, (tp[0] + 8, tp[1] - 8), "tip", TIP_COLOR)

    # axis legend at the front-right of the grid
    origin = _robot_point(base, base_rot, 30, -50, surface_z)
    ends = {"fwd": (origin + base_rot @ np.array([0.1, 0, 0]), (255, 60, 60)),
            "left": (origin + base_rot @ np.array([0, 0.1, 0]), (60, 220, 60)),
            "up": (origin + np.array([0, 0, 0.1]), (80, 120, 255))}
    o = project([origin])[0]
    if o and 0 <= o[0] < size and 0 <= o[1] < size:
        for name, (end, color) in ends.items():
            e = project([end])[0]
            if e:
                draw.line([o, e], fill=color, width=2)
                _text(draw, (e[0] + 2, e[1] - 6), name, color)
    return img


def render_views(env, tip: np.ndarray, base: np.ndarray, base_rot: np.ndarray, surface_z: float) -> list[View]:
    """The images sent to the model this turn, in a fixed order."""
    views = [View(cam, CAPTIONS[cam], annotate(env, cam, tip, base, base_rot, surface_z)) for cam in OVERVIEW_CAMERAS]
    wrist = env.sim.render(width=WRIST_SIZE, height=WRIST_SIZE, camera_name=WRIST_CAMERA)[::-1].copy()
    views.append(View(WRIST_CAMERA, CAPTIONS[WRIST_CAMERA], Image.fromarray(wrist)))
    return views
