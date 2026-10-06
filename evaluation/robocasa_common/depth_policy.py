"""Run four paired Astra input conditions with exact CLI event usage and timing."""

import hashlib
import json
import os
import select
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from robocasa_astra.depth import preview_depth, preview_scale
from robocasa_astra.depth_query import query_depth
from robocasa_astra.policy import CodexPolicy

from inspect_robots.policy import PolicyConfig, PolicyInfo
from inspect_robots.types import Action, ActionChunk

CONDITIONS = {"rgb": (), "color": (), "pixel": ("pixel",), "grid": ("grid",)}


def usage_from_events(events):
    """Use provider-reported completed-turn usage; absent counts remain unknown."""
    reports = [e["usage"] for e in events if e.get("type") == "turn.completed" and "usage" in e]
    if not reports:
        return None
    result = {}
    for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
        values = [r.get(key) for r in reports]
        result[key] = sum(values) if all(type(x) is int for x in values) else None
    if result["input_tokens"] is not None and result["output_tokens"] is not None:
        result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    else:
        result["total_tokens"] = None
    # Reasoning is part of output; the CLI may not expose a separate breakdown.
    result["reasoning_tokens"] = None
    return result


class DepthPolicy:
    """Keep camera depth private unless the selected condition explicitly exposes it."""

    def __init__(self, embodiment, output):
        self.embodiment = embodiment
        self.condition = os.environ["ASTRA_DEPTH_CONDITION"]
        if self.condition not in CONDITIONS:
            raise ValueError("Unknown ablation condition")
        self.info = PolicyInfo(
            name="astra-depth-" + self.condition,
            action_space=embodiment.info.action_space,
            checkpoint="gpt-6-astra",
            control_hz=20,
        )
        self.config = PolicyConfig(action_horizon=8, replan_interval=8)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        docs = json.loads(embodiment.info.docs)
        self.docs = {
            k: docs[k]
            for k in (
                "robot",
                "controller",
                "action_parts",
                "hybrid_mode_index",
                "hybrid_mode_semantics",
                "action_dim",
                "action_low",
                "action_high",
                "control_hz",
            )
            if k in docs
        }
        self.index, self.history = 0, []
        self.calls, self.query_count = [], 0
        self.video, self.video_log = None, None
        self.home = os.environ["ASTRA_CODEX_HOME"]
        self.executable = os.environ["ASTRA_CODEX_EXECUTABLE"]
        dim = self.info.action_space.shape[0]
        query_properties = {
            "camera": {"type": "string"},
            "observation_id": {"type": "string"},
            "kind": {"type": "string", "enum": ["pixel", "grid"]},
            "u": {"type": "integer"},
            "v": {"type": "integer"},
            "radius": {"type": "integer", "minimum": 0, "maximum": 4},
            "cell_id": {"type": "string"},
        }
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
        if CONDITIONS[self.condition]:
            self.schema["properties"]["queries"] = {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": query_properties,
                    "required": list(query_properties),
                    "additionalProperties": False,
                },
            }
            self.schema["required"].append("queries")
        (self.output / "schema.json").write_text(json.dumps(self.schema))
        (self.output / "settings.json").write_text(
            json.dumps(
                {
                    "model": "gpt-6-astra",
                    "reasoning_effort": "medium",
                    "condition": self.condition,
                    "rgb_resolution": [256, 256],
                    "rgb_encoding": "JPEG quality 85, identical across conditions",
                    "query_enabled": CONDITIONS[self.condition],
                    "grid_shape": [16, 16],
                    "action_repeat_max": 8,
                    "max_cli_attempts": 3000,
                    "max_wall_seconds": 21600,
                    "depth_quantity": "camera optical-axis Z meters",
                    "world_coordinates": False,
                    "depth_retention": "current float32 camera arrays outside per-step logs",
                    "token_source": "codex exec --json turn.completed usage; unknown is null",
                    "timing": "local monotonic CLI end-to-end; server GPU compute time not exposed",
                    "state": "robot0 proprioception only; no object poses or privileged success",
                },
                indent=2,
            )
        )
        embodiment.observers.append(self.observe)

    def reset(self, scene):
        """Pair exact snapshot IDs while resetting all episode history and budgets."""
        self.scene, self.instruction = scene.id, scene.instruction
        self.started = time.monotonic()

    def observe(self, observation):
        """Stream RGB video to Lab without per-step depth files or Spark media."""
        import imageio_ffmpeg

        cameras = [c for c in observation.images if not c.endswith("__depth")]
        if self.video is None:
            self.video_log = (self.output / "video.log").open("w")
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
                stderr=self.video_log,
            )
        self.video.stdin.write(
            np.concatenate([observation.images[c] for c in cameras], axis=1).tobytes()
        )

    def call(self, prompt, images, folder):
        """Timestamp raw structured events and measure every attempt, including retries."""
        events, timeline = [], []
        command = [
            self.executable,
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "-s",
            "read-only",
            "-m",
            "gpt-6-astra",
            "-c",
            'model_reasoning_effort="medium"',
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
        for image in images:
            command += ["--image", str(image)]
        command += ["-"]
        env = dict(os.environ, CODEX_HOME=self.home)
        for key in ("OPENAI_API_KEY", "CODEX_API_KEY"):
            env.pop(key, None)
        cwd = Path(self.home).parent / "inference-empty"
        cwd.mkdir(exist_ok=True)
        started = time.monotonic()
        with (folder / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                env=env,
                cwd=cwd,
                bufsize=1,
            )
            process.stdin.write(prompt)
            process.stdin.close()
            try:
                while True:
                    if time.monotonic() - started > 240:
                        raise TimeoutError("Model call exceeded 240 seconds")
                    if not select.select([process.stdout], [], [], 1)[0]:
                        continue
                    line = process.stdout.readline()
                    if not line:
                        break
                    event = json.loads(line)
                    events.append(event)
                    timeline.append({"elapsed_seconds": time.monotonic() - started, "event": event})
                code = process.wait(timeout=10)
            except Exception:
                process.terminate()
                process.wait(timeout=15)
                raise
            finally:
                (folder / "events.jsonl").write_text(
                    "".join(json.dumps(e) + "\n" for e in timeline)
                )
                usage = usage_from_events(events)
                completed = [
                    e["elapsed_seconds"]
                    for e in timeline
                    if e["event"].get("type") == "turn.completed"
                ]
                begun = [
                    e["elapsed_seconds"]
                    for e in timeline
                    if e["event"].get("type") == "turn.started"
                ]
                record = {
                    "attempt": self.index,
                    "condition": self.condition,
                    "effort": "medium",
                    "cli_end_to_end_seconds": time.monotonic() - started,
                    "usage": usage,
                    "turn_event_seconds": completed[-1] - begun[0] if completed and begun else None,
                    "usage_complete": usage is not None
                    and all(
                        usage.get(k) is not None
                        for k in ("input_tokens", "cached_input_tokens", "output_tokens")
                    ),
                    "image_count": len(images),
                }
                self.calls.append(record)
                (folder / "receipt.json").write_text(json.dumps(record, indent=2))
                with (self.output / "calls.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
        if code:
            raise RuntimeError("Codex model call failed; see recorded structured events and stderr")
        return json.loads((folder / "response.json").read_text())

    def act(self, observation):
        """Allow observation-bound distance requests with zero simulator actions."""
        cameras = [c for c in observation.images if not c.endswith("__depth")]
        depths = observation.extra.get("_depth_arrays", {})
        if not depths and self.condition != "rgb":
            if self.embodiment.current_depth_step != observation.extra["steps"]:
                raise ValueError("Camera depth no longer matches this observation")
            depths = self.embodiment.current_depth_arrays
        observation_id = f"{self.scene}:step-{observation.extra['steps']}"
        state = {
            k: np.asarray(v).tolist()
            for k, v in observation.state.items()
            if k.startswith("robot0_")
        }
        base = (
            "Control RoboCasa. Return native action JSON only. +1 closes gripper, -1 opens. "
            "Use short chunks and inspect each new observation. Full native task goal: "
            + self.instruction
            + "\nNative controller: "
            + json.dumps(self.docs)
            + "\nRobot proprioception: "
            + json.dumps(state)
            + "\nRecent actions: "
            + json.dumps(self.history[-8:])
            + "\nRGB cameras in order: "
            + json.dumps(cameras)
        )
        if self.condition == "color":
            base += (
                "\nAfter RGB, separate aligned camera depth images follow. Optical-axis Z meters. "
                "White is near; black is far. Per-camera/per-observation p2-p98 scaling: "
                "do not compare shades across observations. Labels give actual meter ranges."
            )
        if CONDITIONS[self.condition]:
            base += (
                "\nOptional JSON distance requests only, no shell/MCP tools. Allowed query kinds: "
                + json.dumps(CONDITIONS[self.condition])
                + ". observation_id="
                + observation_id
                + ". A nonempty queries array holds the robot still; action/repeat are ignored. "
                "For movement return queries=[]. Camera pixel (u,v) is (column,row), "
                "origin top-left. "
                "Pixel: use integer u/v and radius 0..4. Grid: cell_id r00c00..r15c15, "
                "16 rows by 16 columns, half-open image partitions. Set unused fields to 0 or ''. "
                "Query returns sampled camera Z and region distribution; no world coordinates. "
                "At most 4 query rounds before movement."
            )
        answers = []
        for query_round in range(5):
            if self.index >= 3000 or time.monotonic() - self.started > 21600:
                raise RuntimeError("Common model-call or wall-time budget exceeded")
            folder = self.output / f"call-{self.index:05d}"
            folder.mkdir(exist_ok=False)
            self.index += 1
            images = []
            for camera in cameras:
                path = folder / (camera + ".jpg")
                Image.fromarray(observation.images[camera]).save(path, quality=85)
                images.append(path)
            if self.condition == "color":
                for camera in cameras:
                    depth = depths[camera]
                    image = Image.fromarray(preview_depth(depth))
                    scale = preview_scale(depth)
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 238, 255, 255), fill="black")
                    draw.text(
                        (2, 240),
                        f"Z {scale['near_white_m']:.2f}..{scale['far_black_m']:.2f}m white=near",
                        fill="white",
                    )
                    path = folder / (camera + "__depth.png")
                    image.save(path)
                    images.append(path)
            prompt = base + ("\nDistance query answers: " + json.dumps(answers) if answers else "")
            (folder / "prompt.txt").write_text(prompt)
            try:
                value = self.call(prompt, images, folder)
            except RuntimeError:
                raw = (folder / "events.jsonl").read_text() + (folder / "stderr.log").read_text()
                if "capacity" not in raw.lower() or query_round == 4:
                    raise
                time.sleep(15)
                continue
            queries = value.get("queries", [])
            if queries:
                if query_round == 4:
                    raise RuntimeError("Maximum query rounds exceeded; no action applied")
                for request in queries:
                    try:
                        result = query_depth(
                            depths,
                            request,
                            observation_id,
                            CONDITIONS[self.condition],
                        )
                        x0, y0, x1, y1 = result["region_bounds_uv_half_open"]
                        raw = depths[request["camera"]][y0:y1, x0:x1]
                        np.savez_compressed(
                            folder / f"query-{self.query_count:05d}.npz", depth_m=raw
                        )
                        result["region_float32_sha256"] = hashlib.sha256(raw.tobytes()).hexdigest()
                    except (ValueError, KeyError) as error:
                        result = {"error": str(error), "request": request}
                    self.query_count += 1
                    answers.append(result)
                (folder / "query-results.json").write_text(json.dumps(answers, indent=2))
                continue
            action, repeat = CodexPolicy.validate(self, value)
            self.history.append(value)
            return ActionChunk(
                [Action(action.copy()) for _ in range(repeat)],
                control_hz=20,
                inference_latency_s=self.calls[-1]["cli_end_to_end_seconds"],
            )
        raise RuntimeError("No movement decision within query/capacity budget")

    def close(self):
        """Finalize only this trial's local video encoder."""
        if self.video is not None:
            self.video.stdin.close()
            code = self.video.wait(timeout=30)
            self.video = None
            self.video_log.close()
            if code:
                raise RuntimeError("Video encoding failed")

    def on_trial_end(self, record, log_dir, run_id):
        """Persist counters independently of success and execution failure."""
        self.close()
        (self.output / "summary.json").write_text(
            json.dumps(
                {
                    "model_calls": len(self.calls),
                    "query_requests": self.query_count,
                    "condition": self.condition,
                    "effort": "medium",
                }
            )
        )


def create_policy(embodiment, output):
    """Create the selected paired depth condition with subscription authentication."""
    return DepthPolicy(embodiment, output)
