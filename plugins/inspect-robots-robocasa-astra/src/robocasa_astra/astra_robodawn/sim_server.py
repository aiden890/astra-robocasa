"""Simulator process for astra_robodawn, speaking JSON lines on stdin/stdout.

Requests (one JSON object per line) and replies:

* ``{"op": "reset", "seed": int}`` -> instruction, scene metadata and the initial state
* ``{"op": "observe"}`` -> state, work-surface height and the annotated views (base64 PNG)
* ``{"op": "execute", "command": str, "turn": int}`` -> the executor result for one command
* ``{"op": "close"}`` -> final step count and success; flushes the video and replay files

The process owns ``<output>/video.mp4`` and ``<output>/replay/``. Library output goes to stderr so
stdout stays a clean protocol channel. Errors are returned as ``{"error", "detail"}``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import sys
import traceback
from pathlib import Path

import numpy as np

from ..worker import Simulator
from .commands import parse_command
from .executor import Executor
from .recorder import ReplayRecorder, VideoRecorder
from .views import render_views, surface_height


def _versions() -> dict:
    import mujoco
    import robocasa
    import robosuite

    return {
        "python": platform.python_version(),
        "mujoco": mujoco.__version__,
        "robosuite": getattr(robosuite, "__version__", "unknown"),
        "robocasa": getattr(robocasa, "__version__", "unknown"),
        "robocasa_path": str(Path(robocasa.__file__).parent),
        "robocasa_assets": str(Path(robocasa.__file__).parent / "models" / "assets"),
        "robosuite_path": str(Path(robosuite.__file__).parent),
        "numpy": np.__version__,
    }


class Server:
    """Owns one native environment, its executor and the per-run recorders."""

    def __init__(self, task: str, budget: int, output: Path):
        self.task, self.budget, self.output = task, budget, Path(output)
        self.sim = self.executor = self.video = self.replay = None
        self.turn = 0
        self.caption = ""

    def _on_step(self, action, obs, caption):
        self.replay.add(action)
        self.video.add(obs, f"turn {self.turn} | step {self.executor.steps_used}/{self.budget} | {caption}"
                       + (" | SUCCESS" if self.executor.success else ""))

    def reset(self, seed: int) -> dict:
        self.sim = Simulator("PandaOmron", self.task, horizon=self.budget)
        self.sim.reset(seed)
        env = self.sim.env
        self.video = VideoRecorder(self.output / "video.mp4")
        self.replay = ReplayRecorder(self.output / "replay")
        self.executor = Executor(env, self.budget, on_step=self._on_step)
        meta = env.get_ep_meta()
        scene = {
            "task": self.task, "robot": "PandaOmron", "seed": seed, "horizon": self.budget,
            "layout_id": meta.get("layout_id"), "style_id": meta.get("style_id"),
            "instruction": meta.get("lang"), "control_freq": env.control_freq,
            "replay": "scripts/robocasa-astra/replay_run.py <run>/replay (mode seed reproduces exactly)",
            "versions": _versions(),
        }
        self.replay.save_initial(env, scene)
        return {"instruction": meta.get("lang"), "scene": scene, "state": self.executor.state()}

    def observe(self) -> dict:
        pose = self.executor.pose()
        surface = surface_height(self.sim.env, pose.base, pose.base_rot)
        views = render_views(self.sim.env, pose.tip, pose.base, pose.base_rot, surface)
        state = self.executor.state()
        state["surface_z_cm"] = round(surface * 100, 1)
        state["task_success"] = self.executor.success
        return {"state": state, "views": [{"name": v.name, "caption": v.caption, "png": v.png_base64()} for v in views]}

    def execute(self, command: str, turn: int) -> dict:
        self.turn = turn
        cmd = parse_command(command)
        first = self.executor.steps_used
        result = self.executor.execute(cmd).to_json()
        self.replay.mark(cmd.text(), first, self.executor.steps_used, turn)
        result["state_after"] = self.executor.state()
        return result

    def close(self) -> dict:
        summary = {}
        if self.executor is not None:
            summary = {"steps_used": self.executor.steps_used, "task_success": self.executor.success,
                       "video_frames": self.video.frames}
            self.video.close()
            self.replay.close(self.sim.env)
            self.sim.env.close()
        return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        server = Server(args.task, args.budget, Path(args.output))
        for line in sys.stdin:
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "reset":
                    reply = server.reset(int(request["seed"]))
                elif op == "observe":
                    reply = server.observe()
                elif op == "execute":
                    reply = server.execute(request["command"], int(request.get("turn", 0)))
                elif op == "close":
                    reply = server.close()
                else:
                    raise ValueError(f"unknown op {op!r}")
            except Exception as error:  # noqa: BLE001 - reported to the host, which decides
                traceback.print_exc(file=sys.stderr)
                reply = {"error": type(error).__name__, "detail": str(error)}
            protocol.write(json.dumps(reply) + "\n")
            protocol.flush()
            if request.get("op") == "close":
                break


if __name__ == "__main__":
    main()
