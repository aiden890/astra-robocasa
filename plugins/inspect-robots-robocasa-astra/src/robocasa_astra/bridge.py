"""Inspect Robots embodiment connected to Spark2 through SSH, without public ports."""

import base64
import io
import json
import os
import select
import subprocess
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

    def __init__(self, command, seed, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.sensor_index = 0
        self.stderr = (self.output / "simulator.log").open("w")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
        )
        try:
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

    def call(self, **request):
        """Fail clearly on worker errors or timeouts rather than silently changing worlds."""
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], 180)
        if not ready:
            raise TimeoutError("Spark2 simulator response timeout")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Spark2 worker exited; inspect simulator.log")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"] + ": " + response["detail"])
        return response

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
        return Observation(
            images=images,
            state={k: np.asarray(v, dtype=float) for k, v in raw["state"].items()},
            instruction=instruction,
            extra={
                **raw["info"],
                "depth_maps": depth_maps,
                "depth_metadata": metadata,
                "_depth_arrays": depth_arrays,
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
                self.process.wait(timeout=10)
        self.stderr.close()
