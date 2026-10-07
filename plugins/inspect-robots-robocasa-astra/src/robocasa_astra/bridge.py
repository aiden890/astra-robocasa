"""Inspect Robots embodiment connected to Spark2 through SSH, without public ports."""

import base64
import io
import json
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_robots.embodiment import (
    AUTO_RESET,
    PRIVILEGED_SUCCESS,
    RENDERABLE,
    RESETTABLE,
    SEEDABLE,
    EmbodimentInfo,
)
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace
from inspect_robots.types import Observation, StepResult
from robocasa_astra.depth import decode_depth, preview_depth


class SparkEmbodiment:
    """Expose the simulator's actual robot-specific action contract to evaluation."""

    def __init__(self, command, seed, output, *, checkpoint_mode=False, checkpoint_identity=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        retained = list((self.output / "depth").glob("observation-*"))
        self.sensor_index = max((int(p.name.split("-")[-1]) for p in retained), default=-1) + 1
        self.command = command
        self.checkpoint_mode = checkpoint_mode
        self.seed = seed
        self.resume_steps = 0
        if checkpoint_mode:
            from robocasa_astra.checkpoint import ActionCheckpoint

            self.checkpoint = ActionCheckpoint(
                self.output / "checkpoint.json", checkpoint_identity or command, seed
            )
        self.stderr = (self.output / "simulator.log").open("a")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        try:
            if checkpoint_mode:
                restore = self._restore
                while True:
                    try:
                        self.latest = restore()
                        break
                    except (ConnectionError, TimeoutError, OSError):
                        time.sleep(15)
                        restore = self._reconnect
            else:
                self.latest = self.call(op="reset", seed=seed)
        except BaseException:
            self.close()
            raise
        self.seed = seed
        raw = self.latest["info"]
        dimension = raw["action_dim"]
        self.info = EmbodimentInfo(
            name="robocasa-" + raw["robot"],
            action_space=Box(
                shape=(dimension,),
                low=np.array(raw["action_low"]),
                high=np.array(raw["action_high"]),
                semantics=ActionSemantics(
                    control_mode="eef_delta_pose",
                    rotation_repr="axis_angle",
                    gripper="continuous",
                    frame="base",
                ),
            ),
            observation_space=ObservationSpace(
                cameras=tuple(
                    CameraSpec(name=k, height=256, width=256, channels=3)
                    for k in [
                        *self.latest["images"],
                        *(name + "__depth" for name in self.latest.get("depths", {})),
                    ]
                ),
                state_keys=frozenset(self.latest["state"]),
            ),
            control_hz=20,
            is_simulated=True,
            capabilities=frozenset(
                {AUTO_RESET, PRIVILEGED_SUCCESS, RENDERABLE, RESETTABLE, SEEDABLE}
            ),
            docs=json.dumps(raw),
        )

    def _exchange(self, request):
        """Exchange one request; application failures never trigger blind retries."""
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        deadline = None if request["op"] == "checkpoint_replay" else time.monotonic() + 180
        while not select.select([self.process.stdout], [], [], 1)[0]:
            if self.process.poll() is not None:
                raise ConnectionError("Spark2 worker exited during checkpoint recovery")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("Spark2 simulator response timeout")
        line = self.process.stdout.readline()
        if not line:
            raise ConnectionError("Spark2 worker exited; inspect simulator.log")
        response = json.loads(line)
        if "error" in response:
            raise ValueError(response["error"] + ": " + response["detail"])
        return response

    def _restore(self):
        """Reconstruct acknowledged controller/task state without any model calls."""
        raw = self._exchange({"op": "reset", "seed": self.seed})
        if self.checkpoint.data is None:
            self.checkpoint.reset(raw)
        else:
            actions = self.checkpoint.data["actions"]
            if actions:
                raw = self._exchange({"op": "checkpoint_replay", "actions": actions})
            self.checkpoint.verify(raw)
            self.resume_steps = self.checkpoint.data["steps"]
        return raw

    def _reconnect(self):
        """Recreate only this worker session, preserving acknowledged action identity."""
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout):
            stream.close()
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        return self._restore()

    def call(self, **request):
        """Retry transport failures only after exact checkpoint reconstruction."""
        if not self.checkpoint_mode or request["op"] != "step":
            return self._exchange(request)
        action = request["action"]
        self.checkpoint.intent(action)
        while True:
            try:
                raw = self._exchange(request)
            except (ConnectionError, BrokenPipeError, TimeoutError, OSError) as error:
                with (self.output / "recoveries.jsonl").open("a") as log:
                    log.write(
                        json.dumps(
                            {
                                "time": time.time(),
                                "steps": self.checkpoint.data["steps"],
                                "error": str(error),
                                "model_recalled": False,
                            }
                        )
                        + "\n"
                    )
                while True:
                    try:
                        self._reconnect()
                        break
                    except (ConnectionError, BrokenPipeError, TimeoutError, OSError):
                        time.sleep(15)
                continue
            self.checkpoint.commit(action, raw)
            return raw

    def observation(self, raw, instruction=None):
        """Decode actual rendered camera frames and retain numeric observations."""
        images = {
            k: np.asarray(Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB"))
            for k, v in raw["images"].items()
        }
        payloads = raw.get("depths", {})
        metadata = raw.get("depth_metadata", {})
        if payloads and (set(payloads) != set(images) or set(metadata) != set(images)):
            raise ValueError("Every RGB camera must have matching depth and calibration")
        depth_maps = {}
        depth_arrays = {}
        memory_only = os.environ.get("ASTRA_DEPTH_MEMORY_ONLY") == "1"
        if payloads:
            folder = self.output / "depth" / f"observation-{self.sensor_index:06d}"
            if not memory_only:
                folder.mkdir(parents=True, exist_ok=False)
            self.sensor_index += 1
            for camera, payload in payloads.items():
                if metadata[camera].get("unit") != "m":
                    raise ValueError("Depth calibration must specify meters")
                values = decode_depth(payload, images[camera].shape[:2])
                path = folder / (camera + ".npy")
                depth_arrays[camera] = values
                if not memory_only:
                    np.save(path, values, allow_pickle=False)
                    depth_maps[camera] = str(path.resolve())
                images[camera + "__depth"] = preview_depth(values)
            if not memory_only:
                (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))
        self.current_depth_arrays = depth_arrays
        self.current_depth_step = raw["info"].get("steps")
        return Observation(
            images=images,
            state={k: np.asarray(v, dtype=float) for k, v in raw["state"].items()},
            instruction=instruction,
            extra={
                **raw["info"],
                "depth_maps": depth_maps,
                "depth_metadata": metadata,
                "_depth_arrays": {} if memory_only else depth_arrays,
            },
        )

    def reset(self, scene, *, seed=None):
        """Reuse the initial seeded reset once; later scenes reset the native environment."""
        if self.latest is None or seed != self.seed:
            self.latest = self.call(op="reset", seed=seed if seed is not None else self.seed)
        raw = self.latest
        self.latest = None
        return self.observation(raw, scene.instruction)

    def step(self, action):
        """Execute one validated native controller step."""
        raw = self.call(op="step", action=np.asarray(action.data).tolist())
        success = raw["info"]["success"]
        return StepResult(
            self.observation(raw),
            terminated=success,
            termination_reason="success" if success else None,
            info=raw["info"],
        )

    def close(self):
        """Close only this private worker session."""
        if self.process.poll() is None:
            try:
                self.process.stdin.write('{"op":"close"}\n')
                self.process.stdin.flush()
                self.process.wait(timeout=10)
            except Exception:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        self.stderr.close()
