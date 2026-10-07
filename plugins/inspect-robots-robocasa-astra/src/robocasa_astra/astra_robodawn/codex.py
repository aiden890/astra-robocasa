"""One model decision via the official ``codex exec`` (runs P4-P9; new runs use ``appserver.py``) with the saved ChatGPT subscription login.

Every call writes, under its own folder: the exact command line, the user prompt, the attached image
list, the raw JSONL event stream, stderr, the final message, and a ``call.json`` with usage, reasoning
summaries, latency and every retry. Nothing is summarised away, so a run can be reviewed call by call.

Isolation: the system prompt replaces Codex's built-in instructions (``model_instructions_file``);
shell tool, plugins, web search and project docs are disabled; API-key variables are removed so the
subscription login is used. Reasoning effort is fixed to ``low`` by the experiment protocol.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

REASONING_EFFORT = "low"  # fixed by the experiment protocol; deliberately not configurable
MODEL = "gpt-6-astra"
CALL_TIMEOUT_S = 300
MAX_ATTEMPTS = 6
CAPACITY_MARKERS = ("Selected model is at capacity", "at capacity", "rate limit", "Rate limit")


class ModelCallError(RuntimeError):
    """The model did not produce a final answer after the allowed attempts."""


@dataclass
class CallResult:
    text: str
    usage: dict
    reasoning: list[str]
    seconds: float
    attempts: list[dict] = field(default_factory=list)
    weekly: dict | None = None  # subscription weekly-limit snapshot after the call (app-server only)


def parse_events(lines: list[str]) -> dict:
    """Pull usage, reasoning summaries, agent messages and errors out of a Codex JSONL event stream."""
    out = {"usage": {}, "reasoning": [], "messages": [], "errors": [], "failed": False, "completed": False}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "turn.completed":
            out["completed"] = True
            out["usage"] = event.get("usage") or {}
        elif kind == "turn.failed":
            out["failed"] = True
            out["errors"].append(json.dumps(event.get("error")))
        elif kind == "error":
            out["errors"].append(event.get("message", ""))
        elif kind == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "reasoning":
                out["reasoning"].append(item.get("text", ""))
            elif item.get("type") == "agent_message":
                out["messages"].append(item.get("text", ""))
            elif item.get("type") == "error":
                out["errors"].append(item.get("message", ""))
    return out


def write_input(folder: Path, parts: list[dict]) -> None:
    """Review copies of one request's ordered input: ``input.json`` (parts, images as paths) and ``prompt.txt``."""
    from .prompts import parts_text

    (folder / "input.json").write_text(json.dumps(parts, indent=1))
    (folder / "prompt.txt").write_text(parts_text(parts))


class CodexCaller:
    """Calls ``codex exec`` for one turn; see the module docstring for what is recorded."""

    def __init__(self, executable: str, codex_home: str, system_prompt_path: Path, schema_path: Path,
                 workdir: Path, model: str = MODEL, reasoning_summary: bool = True):
        self.executable, self.codex_home = executable, codex_home
        self.system_prompt_path, self.schema_path = Path(system_prompt_path), Path(schema_path)
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.reasoning_summary = reasoning_summary

    def command(self, images: list[Path], last_message: Path) -> list[str]:
        command = [
            self.executable, "exec", "--strict-config", "--ephemeral", "--skip-git-repo-check",
            "-s", "read-only", "-m", self.model,
            "-c", f'model_reasoning_effort="{REASONING_EFFORT}"',
            "-c", f'model_instructions_file="{self.system_prompt_path}"',
            "-c", "project_doc_max_bytes=0",
            "-c", "features.shell_tool=false",
            "-c", "features.plugins=false",
            "-c", "web_search=disabled",
            "--json", "--output-schema", str(self.schema_path), "-o", str(last_message),
        ]
        if self.reasoning_summary:
            command[command.index("--json"):command.index("--json")] = ["-c", 'model_reasoning_summary="detailed"']
        for image in images:
            command += ["--image", str(image)]
        return command + ["-"]

    def call(self, parts: list[dict], folder: Path) -> CallResult:
        """``codex exec`` cannot interleave: all images go first (in order), then all texts joined."""
        folder.mkdir(parents=True, exist_ok=True)
        write_input(folder, parts)
        images = [Path(p["path"]) for p in parts if p["type"] == "image"]
        prompt = "\n\n".join(p["text"] for p in parts if p["type"] == "text")
        env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")}
        env["CODEX_HOME"] = self.codex_home
        attempts = []
        start = time.monotonic()
        for attempt in range(MAX_ATTEMPTS):
            suffix = "" if attempt == 0 else f".retry{attempt}"
            last_message = folder / f"response{suffix}.json"
            command = self.command(images, last_message)
            (folder / f"command{suffix}.json").write_text(json.dumps(command, indent=1))
            t0 = time.monotonic()
            timed_out = False
            with (folder / f"events{suffix}.jsonl").open("w") as stdout, (folder / f"stderr{suffix}.log").open("w") as stderr:
                try:
                    proc = subprocess.run(command, input=prompt, text=True, cwd=self.workdir, env=env,
                                          stdout=stdout, stderr=stderr, timeout=CALL_TIMEOUT_S, check=False)
                    returncode = proc.returncode
                except subprocess.TimeoutExpired:
                    timed_out, returncode = True, None
            events = parse_events((folder / f"events{suffix}.jsonl").read_text().splitlines())
            stderr_text = (folder / f"stderr{suffix}.log").read_text()
            record = {"attempt": attempt + 1, "returncode": returncode, "timed_out": timed_out,
                      "seconds": round(time.monotonic() - t0, 2), "usage": events["usage"], "errors": events["errors"]}
            attempts.append(record)
            if returncode == 0 and events["completed"] and last_message.exists():
                result = CallResult(last_message.read_text(), events["usage"], events["reasoning"],
                                    round(time.monotonic() - start, 2), attempts)
                self._write_record(folder, result)
                return result
            error_text = " ".join(events["errors"]) + stderr_text
            if self.reasoning_summary and "summary" in error_text.lower() and "reason" in error_text.lower():
                record["action"] = "retry without reasoning summary (rejected by the model/endpoint)"
                self.reasoning_summary = False
                continue
            if timed_out or any(marker in error_text for marker in CAPACITY_MARKERS):
                wait = min(120, 15 * (attempt + 1))
                record["action"] = f"retry after {wait} s (simulator is paused, no action applied)"
                (folder / "attempts.json").write_text(json.dumps(attempts, indent=1))
                time.sleep(wait)
                continue
            break
        (folder / "attempts.json").write_text(json.dumps(attempts, indent=1))
        raise ModelCallError(f"no final answer after {len(attempts)} attempt(s); see {folder}")

    def _write_record(self, folder: Path, result: CallResult) -> None:
        (folder / "call.json").write_text(json.dumps({
            "model": self.model, "reasoning_effort": REASONING_EFFORT, "reasoning_summary_requested": self.reasoning_summary,
            "seconds": result.seconds, "usage": result.usage, "reasoning": result.reasoning,
            "final_text": result.text, "attempts": result.attempts,
        }, indent=1))
