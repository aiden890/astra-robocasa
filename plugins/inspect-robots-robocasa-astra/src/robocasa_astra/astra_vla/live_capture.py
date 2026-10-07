"""Persist numbered native frames to the Lab mount for continuous live playback."""

from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ..astra_robodawn.recorder import VIDEO_CAMERAS
from ..depth import normalize_depth_buffer
from .persistence import atomic_json


def capture(env, obs, output: Path, step: int, condition: str):
    """Write one frame per acknowledged native step, preserving the initial frame at zero."""
    from robosuite.utils.camera_utils import get_real_depth_map

    folder = output / "live-frames"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / f"rgb-{step:06d}.jpg").exists():
        return
    rgb, depth = [], []
    for camera in VIDEO_CAMERAS:
        image = obs.get(camera + "_image")
        if image is None:
            image = env.sim.render(width=256, height=256, camera_name=camera)
        image = np.asarray(image)[::-1].astype(np.uint8)
        rgb.append(Image.fromarray(image).resize((256, 256)))
        if condition in ("color", "hybrid"):
            _, raw = env.sim.render(width=256, height=256, camera_name=camera, depth=True)
            z = get_real_depth_map(env.sim, normalize_depth_buffer(raw))[::-1]
            gray = np.nan_to_num(255 * (1 - np.clip(z, 0, 2) / 2), nan=0).astype(np.uint8)
            depth.append(Image.fromarray(gray).convert("RGB"))
    for kind, images in [("rgb", rgb), ("depth", depth)]:
        if not images:
            continue
        canvas = Image.new("RGB", (768, 278), "black")
        for index, image in enumerate(images):
            canvas.paste(image, (index * 256, 22))
        ImageDraw.Draw(canvas).text(
            (6, 5),
            f"native step {step} | 20 Hz" + (" | Z 0..2m, near white" if kind == "depth" else ""),
            fill="white",
        )
        target = folder / f"{kind}-{step:06d}.jpg"
        temp = folder / f".{kind}-{step:06d}.jpg"
        canvas.save(temp, format="JPEG", quality=85)
        os.replace(temp, target)
    atomic_json(output / "live-capture.json", {"step": step, "fps": 20, "initial_frame": True})
