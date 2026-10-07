"""Per-step video and replay records written by the simulator process.

``VideoRecorder`` streams every native step to an H.264 MP4 at the control rate (20 fps): the three
robot cameras side by side with a caption bar (step, budget, current command). Nothing is buffered
in memory and no per-frame files are kept.

``ReplayRecorder`` stores what is needed to reproduce an episode exactly: the seed and scene
metadata, the initial MuJoCo state and model XML, and every native action that was applied.
Replay = reset the model/state with ``robocasa ... reset_to`` and step the saved actions.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VIDEO_CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
VIDEO_FPS = 20
CAPTION_HEIGHT = 22


class VideoRecorder:
    """Streams captioned three-camera frames to ``path``."""

    def __init__(self, path: Path, fps: int = VIDEO_FPS):
        self.path = Path(path)
        self.fps = fps
        self._writer = None
        self.frames = 0

    def add(self, obs: dict, caption: str) -> None:
        images = [np.asarray(obs[f"{name}_image"])[::-1] for name in VIDEO_CAMERAS if f"{name}_image" in obs]
        if not images:
            return
        strip = np.concatenate(images, axis=1)
        canvas = Image.new("RGB", (strip.shape[1], strip.shape[0] + CAPTION_HEIGHT), (0, 0, 0))
        canvas.paste(Image.fromarray(strip.astype(np.uint8)), (0, CAPTION_HEIGHT))
        ImageDraw.Draw(canvas).text((6, 5), caption, fill=(255, 255, 255))
        frame = np.asarray(canvas)
        if self._writer is None:
            import imageio_ffmpeg

            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            self._writer = imageio_ffmpeg.write_frames(
                str(self.path), (width, height), fps=self.fps, codec="libx264", quality=7,
                macro_block_size=2, output_params=["-pix_fmt", "yuv420p"],
            )
            self._writer.send(None)
        self._writer.send(np.ascontiguousarray(frame))
        self.frames += 1

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class ReplayRecorder:
    """Writes ``replay/`` in the run directory: scene, initial state, model and native actions."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.actions: list[np.ndarray] = []
        self.segments: list[dict] = []  # one entry per executed command: [first_step, last_step)
        self._log = (self.folder / "actions.jsonl").open("w")

    def save_initial(self, env, scene: dict) -> None:
        """Seed, task, layout/style, asset roots, ep_meta, initial MuJoCo state and model XML."""
        (self.folder / "scene.json").write_text(json.dumps(scene, indent=2))
        (self.folder / "ep_meta.json").write_text(json.dumps(env.get_ep_meta(), indent=2, default=str))
        state = env.sim.get_state().flatten()
        np.savez_compressed(self.folder / "initial_state.npz", state=state)
        (self.folder / "model.xml.gz").write_bytes(gzip.compress(env.sim.model.get_xml().encode()))

    def add(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=float)
        self.actions.append(action)
        self._log.write(json.dumps(action.round(6).tolist()) + "\n")

    def mark(self, command: str, first_step: int, last_step: int, turn: int) -> None:
        self.segments.append({"turn": turn, "command": command, "first_step": first_step, "last_step": last_step})

    def close(self, env=None) -> None:
        """Flush the action log; with ``env``, also store the final MuJoCo state for replay checks."""
        self._log.close()
        if env is not None:
            np.savez_compressed(self.folder / "final_state.npz", state=env.sim.get_state().flatten())
        if self.actions:
            np.save(self.folder / "actions.npy", np.stack(self.actions))
        (self.folder / "command_segments.json").write_text(json.dumps(self.segments, indent=1))
