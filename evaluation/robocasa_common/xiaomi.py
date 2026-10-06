"""Official Xiaomi RoboCasa inference on the unchanged common scene protocol."""

import base64
import collections
import io
import json
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.types import Action, ActionChunk

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
CHECKPOINT = "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365"


def axis_angle(quaternion):
    """Convert native xyzw quaternions with the official evaluator's sign convention."""
    q = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float32)
    q = q / norm
    if q[3] < 0:
        q = -q
    s = np.linalg.norm(q[:3])
    if s < 1e-12:
        return np.zeros(3, dtype=np.float32)
    return (q[:3] / s * 2 * np.arctan2(s, np.clip(q[3], -1, 1))).astype(np.float32)


def robot_state(state):
    """Match PandaOmronKeyConverter and official EE-first 14D state, not pi05 state."""
    value = np.concatenate(
        [
            state["robot0_base_to_eef_pos"],
            axis_angle(state["robot0_base_to_eef_quat"]),
            state["robot0_gripper_qpos"],
            state["robot0_base_pos"],
            axis_angle(state["robot0_base_quat"]),
        ]
    ).astype(np.float32)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("Invalid official Xiaomi 14D observation")
    return value


def native_action(action, parts, mode_index=None):
    """Apply the official Gym wrapper's ordering and 0.5 discrete thresholds."""
    a = np.asarray(action, dtype=float)
    if a.shape != (12,) or not np.isfinite(a).all():
        raise ValueError("Invalid Xiaomi 12D action")
    values = {
        "right": a[:6],
        "right_gripper": [1.0 if a[6] >= 0.5 else -1.0],
        "base": a[7:10],
        "torso": a[10:11],
    }
    output = np.full(12, np.nan)
    for name, value in values.items():
        key = name if name in parts else "robot0_" + name
        start, end = parts[key]
        output[start:end] = value
    if mode_index is None:
        key = "base_mode" if "base_mode" in parts else "robot0_base_mode"
        mode_index = parts[key][0]
    if not 0 <= mode_index < 12 or np.isfinite(output[mode_index]):
        raise ValueError("Invalid or overlapping native hybrid mode index")
    output[mode_index] = 1.0 if a[11] >= 0.5 else -1.0
    if not np.isfinite(output).all():
        raise ValueError("Incomplete native controller mapping")
    return output


class XiaomiPolicy(PolicyBase):
    """Collect every real step; preprocess and infer using the official Spark client."""

    def __init__(self, embodiment, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.info = PolicyInfo(
            name="xiaomi-robotics-1-robocasa365",
            action_space=embodiment.info.action_space,
            control_hz=20,
            checkpoint=CHECKPOINT,
        )
        self.config = PolicyConfig(action_horizon=16, replan_interval=16)
        self.parts = json.loads(embodiment.info.docs)["action_parts"]
        self.mode_index = json.loads(embodiment.info.docs)["hybrid_mode_index"]
        self.history = collections.deque(maxlen=7)
        self.calls = []
        self.video = None
        self.process = None
        self.stderr = (self.output / "client.log").open("w")
        self.process = subprocess.Popen(
            json.loads(os.environ["XIAOMI_CLIENT_COMMAND"]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        if os.environ.get("XIAOMI_COMPACT_VIDEO") == "1":
            import imageio_ffmpeg

            self.video = subprocess.Popen(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    "768x256",
                    "-framerate",
                    "20",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(self.output.parent / "video.mp4"),
                ],
                stdin=subprocess.PIPE,
                stderr=self.stderr,
            )
        embodiment.observers.append(self.observe)
        (self.output / "settings.json").write_text(
            json.dumps(
                {
                    "checkpoint": CHECKPOINT,
                    "state_dim": 14,
                    "padded_state_dim": 60,
                    "cameras": CAMERAS,
                    "crop_ratio": 0.95,
                    "history_length": 4,
                    "history_interval": 2,
                    "replan_steps": 16,
                    "gripper_and_mode_threshold": 0.5,
                    "control_hz": 20,
                },
                indent=2,
            )
        )

    def reset(self, scene):
        """Clear history before the common embodiment supplies its initial observation."""
        self.history.clear()
        self.instruction = scene.instruction

    def observe(self, observation):
        """Retain initial and all intermediate observations without writing Spark media."""
        self.history.append(observation)
        if self.video is not None:
            self.video.stdin.write(
                np.concatenate([observation.images[c] for c in CAMERAS], axis=1).tobytes()
            )

    def act(self, observation):
        """Execute the official first 16 actions and log latency separately from simulation."""
        if not self.history:
            raise ValueError("Missing per-step observation history")
        items = list(self.history)
        indices = [max(0, len(items) - 1 - (3 - i) * 2) for i in range(4)]
        images = {}
        for camera in CAMERAS:
            encoded = []
            for i in indices:
                buf = io.BytesIO()
                Image.fromarray(items[i].images[camera]).save(buf, format="PNG")
                encoded.append(base64.b64encode(buf.getvalue()).decode())
            images["video." + camera] = encoded
        request = {
            "state": [robot_state(items[i].state).tolist() for i in indices],
            "images": images,
            "instruction": observation.extra.get("instruction", self.instruction),
        }
        started = time.monotonic()
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        if not select.select([self.process.stdout], [], [], 180)[0]:
            raise TimeoutError("Xiaomi inference exceeded 180 seconds")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Xiaomi client exited; see client.log")
        response = json.loads(line)
        (self.output / f"response-{len(self.calls):04d}.json").write_text(json.dumps(response))
        if "error" in response:
            raise RuntimeError(response["error"])
        actions = np.asarray(response["actions"])
        if actions.ndim != 2 or actions.shape[1] != 12 or len(actions) < 16:
            raise ValueError("Incomplete official action chunk")
        native = [native_action(a, self.parts, self.mode_index) for a in actions[:16]]
        box = self.info.action_space
        if any(np.any(a < box.low) or np.any(a > box.high) for a in native):
            raise ValueError("Xiaomi action violates common native bounds; no silent clipping")
        seconds = time.monotonic() - started
        record = {
            "call": len(self.calls),
            "env_step": observation.extra.get("env_step"),
            "round_trip_seconds": seconds,
            "model_seconds": response["seconds"],
            "history_indices": indices,
            "official_actions": actions[:16].tolist(),
            "native_actions": [a.tolist() for a in native],
        }
        self.calls.append(record)
        with (self.output / "calls.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return ActionChunk([Action(a) for a in native], control_hz=20, inference_latency_s=seconds)

    def close(self):
        """Drain the video and close this episode's private client only."""
        if self.process is not None:
            if self.process.poll() is None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=10)
            self.process = None
        if self.video is not None:
            self.video.stdin.close()
            code = self.video.wait(timeout=30)
            self.video = None
            if code != 0:
                raise RuntimeError("Video encoding failed")
        self.stderr.close()

    def on_trial_end(self, record, log_dir, run_id):
        """Finalize local video for both normal task outcomes and execution errors."""
        self.close()


def create_policy(embodiment, output):
    """Attach the official base checkpoint without modifying the fixed scenes."""
    return XiaomiPolicy(embodiment, output)
