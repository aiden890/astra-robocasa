"""Run four paired Astra input conditions with exact CLI event usage and timing."""

import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from robocasa_astra.checkpoint import atomic_json
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


def stream_events(process, started, events, timeline, timeout=None, event_log=None):
    """Drain until EOF; only explicitly requested deadlines stop a live call."""
    pending = b""
    while True:
        remaining = None if timeout is None else timeout - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            raise TimeoutError(f"Model call exceeded {timeout:g} seconds")
        if not select.select(
            [process.stdout], [], [], 1 if remaining is None else min(1, remaining)
        )[0]:
            continue
        chunk = os.read(process.stdout.fileno(), 65536)
        pending += chunk
        lines = pending.split(b"\n")
        pending = lines.pop()
        if not chunk and pending:
            lines.append(pending)
            pending = b""
        for line in lines:
            if line.strip():
                event = json.loads(line)
                events.append(event)
                entry = {"elapsed_seconds": time.monotonic() - started, "event": event}
                timeline.append(entry)
                if event_log is not None:
                    event_log.write(json.dumps(entry) + "\n")
                    event_log.flush()
                    os.fsync(event_log.fileno())
        if not chunk:
            return


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
        self.resumable = bool(getattr(embodiment, "checkpoint_mode", False))
        self.output.mkdir(parents=True, exist_ok=self.resumable)
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
        self.progress_path = self.output / "progress.json"
        self.progress = (
            json.loads(self.progress_path.read_text()) if self.progress_path.exists() else {}
        )
        if self.resumable:
            self.index = max(
                [int(f.name.split("-")[1]) + 1 for f in self.output.glob("call-*")] or [0]
            )
            self.history = self.progress.get("history", [])
            self.query_count = self.progress.get("query_count", 0)
            file = self.output / "calls.jsonl"
            self.calls = (
                [json.loads(line) for line in file.read_text().splitlines()]
                if file.exists()
                else []
            )
        self.video, self.video_log = None, None
        self.reuse_initial_frame = any(self.output.parent.glob("video-part-*.mp4"))
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
                    "max_wall_seconds": None,
                    "model_call_timeout_seconds": None,
                    "capacity_retry_wait_seconds": 15,
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

        if (
            self.resumable
            and self.reuse_initial_frame
            and self.video is None
            and observation.extra.get("steps") == self.embodiment.resume_steps
        ):
            return

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
                    str(
                        self.output.parent
                        / (f"video-part-{time.time_ns()}.mp4" if self.resumable else "video.mp4")
                    ),
                ],
                stdin=subprocess.PIPE,
                stderr=self.video_log,
            )
        self.video.stdin.write(
            np.concatenate([observation.images[c] for c in cameras], axis=1).tobytes()
        )

    def save_progress(self, **updates):
        """Save action-chunk/query state before any physical motion."""
        self.progress.update(updates, history=self.history[-8:], query_count=self.query_count)
        if self.resumable:
            atomic_json(self.progress_path, self.progress)

    def call(self, prompt, images, folder):
        """Adopt a detached live model runner instead of recalling the same observation."""
        if not self.resumable:
            return self.blocking_call(prompt, images, folder)
        request = folder / "runner-request.json"
        if not request.exists():
            atomic_json(
                request,
                {
                    "prompt": prompt,
                    "images": [str(x) for x in images],
                    "home": self.home,
                    "executable": self.executable,
                    "condition": self.condition,
                    "index": self.index,
                    "output": str(self.output),
                    "started_at": time.time(),
                },
            )
        outcome = folder / "runner-result.json"
        if not outcome.exists():
            # A wrapper may wait on an existing runner's lock. The lock, rather than
            # a PID check/spawn race, determines who can start the actual CLI call.
            with (folder / "runner.log").open("a") as log:
                runner = subprocess.Popen(
                    [sys.executable, "-m", "robocasa_common.depth_call_runner", str(folder)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
                while not outcome.exists():
                    if runner.poll() is not None:
                        time.sleep(15)
                        runner = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "robocasa_common.depth_call_runner",
                                str(folder),
                            ],
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                        )
                    time.sleep(0.2)
        record = json.loads((folder / "receipt.json").read_text())
        if not any(c["attempt"] == record["attempt"] for c in self.calls):
            self.calls.append(record)
        result = json.loads(outcome.read_text())
        if not result["ok"]:
            raise RuntimeError(result["error"])
        return result["value"]

    def blocking_call(self, prompt, images, folder):
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
        started_wall = time.time()
        with (folder / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=False,
                env=env,
                cwd=cwd,
                bufsize=0,
            )
            process.stdin.write(prompt.encode("utf-8"))
            process.stdin.close()
            try:
                atomic_json(
                    folder / "cli-pid.json", {"pid": process.pid, "started_at": time.time()}
                )
                with (folder / "events-live.jsonl").open("a") as live:
                    stream_events(process, started, events, timeline, event_log=live)
                code = process.wait()
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
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
                    "started_at": started_wall,
                    "ended_at": time.time(),
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
                atomic_json(folder / "receipt.json", record)
                with (self.output / "calls.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
        if code:
            raise RuntimeError("Codex model call failed; see recorded structured events and stderr")
        return json.loads((folder / "response.json").read_text())

    def act(self, observation):
        """Allow observation-bound distance requests with zero simulator actions."""
        cameras = [c for c in observation.images if not c.endswith("__depth")]
        step = observation.extra["steps"]
        decision = self.progress.get("decision") if self.resumable else None
        if decision is not None:
            remaining = decision["repeat"] - (step - decision["start_step"])
            if remaining > 0:
                action = np.asarray(decision["action"], dtype=float)
                return ActionChunk(
                    [Action(action.copy()) for _ in range(remaining)],
                    control_hz=20,
                    inference_latency_s=0,
                )
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
        same_observation = self.progress.get("observation_id") == observation_id
        answers = self.progress.get("answers", []) if same_observation else []
        query_round = self.progress.get("query_round", 0) if same_observation else 0
        inflight = self.progress.get("inflight") if same_observation else None
        while query_round <= 4:
            if self.index >= 3000 and not inflight:
                raise RuntimeError("Common model-call attempt budget exceeded")
            folder = self.output / inflight if inflight else self.output / f"call-{self.index:05d}"
            if not inflight:
                folder.mkdir(exist_ok=False)
                self.index += 1
            self.save_progress(
                observation_id=observation_id,
                answers=answers,
                query_round=query_round,
                inflight=folder.name,
            )
            inflight = None
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
            except RuntimeError as error:
                self.save_progress(inflight=None)
                raw = "".join(
                    file.read_text()
                    for file in (
                        folder / "events.jsonl",
                        folder / "events-live.jsonl",
                        folder / "stderr.log",
                    )
                    if file.exists()
                )
                if "capacity" not in raw.lower() and "Interrupted model runner retained" not in str(
                    error
                ):
                    raise
                self.save_progress(inflight=None)
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
                query_round += 1
                self.save_progress(answers=answers, query_round=query_round, inflight=None)
                continue
            try:
                action, repeat = CodexPolicy.validate(self, value)
            except ValueError:
                self.save_progress(inflight=None)
                raise
            self.history.append(value)
            self.save_progress(
                inflight=None,
                decision={"action": action.tolist(), "repeat": repeat, "start_step": step},
            )
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
        if self.resumable and any(self.output.parent.glob("video-part-*.mp4")):
            import imageio_ffmpeg

            parts = sorted(self.output.parent.glob("video-part-*.mp4"))
            listing = self.output / "video-parts.txt"
            listing.write_text("".join("file '" + str(p) + "'\n" for p in parts))
            subprocess.run(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listing),
                    "-vf",
                    "setpts=N/(20*TB)",
                    "-r",
                    "20",
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
                check=True,
            )

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
