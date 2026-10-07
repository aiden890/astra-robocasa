"""Publish actual per-action clips while preserving the full native rollout recording."""

import time

import imageio_ffmpeg
import numpy as np
from robocasa_astra.checkpoint import atomic_json

from robocasa_common.depth_policy import DepthPolicy


class PairedLivePolicy(DepthPolicy):
    """Collect only the current action chunk for a low-memory playback preview."""

    def __init__(self, embodiment, output):
        self.clip_frames = []
        self.clip_start = 0
        super().__init__(embodiment, output)

    def observe(self, observation):
        """Keep every actual RGB frame, then atomically publish the acknowledged chunk."""
        super().observe(observation)
        cameras = [c for c in observation.images if not c.endswith("__depth")]
        frame = np.concatenate([observation.images[c] for c in cameras], axis=1)
        step = observation.extra["steps"]
        if not self.clip_frames:
            self.clip_start = step
        self.clip_frames.append(frame.copy())
        decision = self.progress.get("decision", {})
        if decision and (
            step >= decision["start_step"] + decision["repeat"] or observation.extra.get("success")
        ):
            target = self.output.parent / "live.mp4"
            temporary = target.with_name("live.pending.mp4")
            writer = imageio_ffmpeg.write_frames(
                str(temporary),
                (frame.shape[1], frame.shape[0]),
                fps=20,
                codec="libx264",
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
            try:
                for image in self.clip_frames:
                    writer.send(image)
            finally:
                writer.close()
            temporary.replace(target)
            atomic_json(
                self.output.parent / "live-clip.json",
                {
                    "start_step": self.clip_start,
                    "end_step": step,
                    "fps": 20,
                    "version": time.time_ns(),
                    "frames": len(self.clip_frames),
                },
            )
            self.clip_frames = [frame.copy()]
            self.clip_start = step


def create_policy(embodiment, output):
    """Enable action previews only for the separately requested paired experiment."""
    return PairedLivePolicy(embodiment, output)
