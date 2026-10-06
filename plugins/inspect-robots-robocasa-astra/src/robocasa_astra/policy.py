"""Codex ChatGPT subscription policy with structured, bounded action output."""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.types import Action, ActionChunk


class CodexPolicy:
    """Attach images to official codex exec; persist every prompt, output and timing."""

    def __init__(self, embodiment, output, executable, codex_home, model="gpt-6-astra", noop=False):
        self.info = PolicyInfo(
            name="codex-" + model, action_space=embodiment.info.action_space, checkpoint=model
        )
        self.config = PolicyConfig(action_horizon=8)
        self.docs = embodiment.info.docs
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.executable, self.home, self.model, self.noop = executable, codex_home, model, noop
        self.history = []
        self.index = 0
        dim = self.info.action_space.shape[0]
        self.schema = {
            "type": "object",
            "properties": {
                "action": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": dim,
                    "maxItems": dim,
                },
                "repeat": {"type": "integer", "minimum": 1, "maximum": 8},
                "reason": {"type": "string"},
            },
            "required": ["action", "repeat", "reason"],
            "additionalProperties": False,
        }
        (self.output / "schema.json").write_text(json.dumps(self.schema))

    def reset(self, scene):
        """Clear scene context; retain call files under distinct evaluation output directories."""
        self.instruction = scene.instruction
        self.history = []

    def validate(self, value):
        """Reject malformed or out-of-bounds actions before they reach the simulator."""
        action = np.asarray(value["action"], dtype=float)
        box = self.info.action_space
        if (
            action.shape != box.shape
            or not np.isfinite(action).all()
            or np.any(action < box.low)
            or np.any(action > box.high)
        ):
            raise ValueError("Invalid native action returned by model")
        repeat = value["repeat"]
        if type(repeat) is not int or not 1 <= repeat <= 8:
            raise ValueError("Invalid repeat count")
        return action, repeat

    def act(self, observation):
        """Perform one perception-to-action inference without granting shell or MCP tools."""
        folder = self.output / f"call-{self.index:04d}"
        folder.mkdir(exist_ok=False)
        self.index += 1
        images = []
        for name, image in observation.images.items():
            path = folder / (name + ".png")
            Image.fromarray(image).save(path)
            images.append(path)
        state = {k: np.asarray(v).tolist() for k, v in observation.state.items()}
        current_instruction = observation.extra.get("instruction", self.instruction)
        prompt = (
            "Control a simulated RoboCasa robot. Return action JSON only. No tools. "
            "Use the native controller. Parallel jaws: +1 closes, -1 opens. "
            "Inspect images/state each turn and use short chunks. "
            "Task goal: "
            + current_instruction
            + "\nController and action-part indices: "
            + self.docs
            + "\nCurrent state: "
            + json.dumps(state)
            + "\nRecent actions: "
            + json.dumps(self.history[-8:])
        )
        (folder / "prompt.txt").write_text(prompt)
        start = time.monotonic()
        if self.noop:
            value = {
                "action": [0.0] * self.info.action_space.shape[0],
                "repeat": 1,
                "reason": "no-model wiring test",
            }
        else:
            command = [
                self.executable,
                "exec",
                "--skip-git-repo-check",
                "--ephemeral",
                "-s",
                "read-only",
                "-m",
                self.model,
                "-c",
                'model_reasoning_effort="low"',
                "-c",
                "project_doc_max_bytes=0",
                "-c",
                "features.shell_tool=false",
                "-c",
                "features.plugins=false",
                "-c",
                "web_search=disabled",
                "--output-schema",
                str(self.output / "schema.json"),
                "-o",
                str(folder / "response.json"),
            ]
            for path in images:
                command += ["--image", str(path)]
            command += ["-"]
            env = dict(os.environ, CODEX_HOME=self.home)
            for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
                env.pop(key, None)
            cwd = Path(self.home).parent / "inference-empty"
            cwd.mkdir(exist_ok=True)
            for attempt in range(20):
                log_path = folder / ("cli.log" if attempt == 0 else f"cli-retry-{attempt:02d}.log")
                response_path = folder / "response.json"
                if response_path.exists():
                    response_path.rename(folder / f"response-incomplete-{attempt:02d}.json")
                reason = "model_at_capacity"
                try:
                    with log_path.open("w") as log:
                        result = subprocess.run(
                            command,
                            input=prompt,
                            text=True,
                            cwd=cwd,
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=180,
                            check=False,
                        )
                except subprocess.TimeoutExpired:
                    reason = "model_call_timeout"
                else:
                    if result.returncode == 0:
                        break
                    if "Selected model is at capacity" not in log_path.read_text():
                        raise subprocess.CalledProcessError(result.returncode, command)
                (folder / "retry-status.json").write_text(
                    json.dumps(
                        {
                            "attempt": attempt + 1,
                            "reason": reason,
                            "environment_held": True,
                            "retry_delay_seconds": min(120, 15 * (attempt + 1)),
                        }
                    )
                )
                if attempt == 19:
                    raise RuntimeError(
                        "Model call unavailable after 20 attempts; no action applied"
                    )
                time.sleep(min(120, 15 * (attempt + 1)))
            value = json.loads((folder / "response.json").read_text())
        action, repeat = self.validate(value)
        self.history.append(value)
        seconds = time.monotonic() - start
        (folder / "receipt.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "subscription_auth": not self.noop,
                    "seconds": seconds,
                    "response": value,
                    "robot": observation.extra.get("robot"),
                },
                indent=2,
            )
        )
        return ActionChunk(
            [Action(action.copy()) for _ in range(repeat)],
            control_hz=20,
            inference_latency_s=seconds,
            meta={"reason": value["reason"]},
        )

    def transcript(self):
        """Expose completed decisions in the official evaluation log."""
        return {"model": self.model, "decisions": self.history}
