"""Adopt detached model requests; persist usage and elapsed time without duplicate billing."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from ..astra_robodawn.codex import CallResult, ModelCallError
from ..astra_robodawn.cost import add_usage
from .persistence import atomic_json, digest, exact_process, locked, read_json


class DurableCaller:
    """The host may disappear while the runner finishes; resume reuses the same response."""

    def __init__(
        self,
        executable,
        codex_home,
        system,
        schema,
        workdir,
        run_dir,
        model="gpt-6-astra",
        effort="low",
        attempt_budget=3000,
    ):
        self.settings = {
            "executable": executable,
            "codex_home": codex_home,
            "system": str(system),
            "schema": str(schema),
            "workdir": str(workdir),
            "run_dir": str(run_dir),
            "model": model,
            "effort": effort,
            "attempt_budget": attempt_budget,
        }
        self.weekly = None

    def call(self, parts: list[dict], folder: Path) -> CallResult:
        """Wait without a model deadline and verify immutable input before adopting an existing
        request."""
        folder.mkdir(parents=True, exist_ok=True)
        request = {**self.settings, "parts": parts}
        identity = digest(
            {
                **request,
                "system_text": Path(request["system"]).read_text(),
                "schema_text": Path(request["schema"]).read_text(),
                "images": {
                    p["path"]: hashlib.sha256(Path(p["path"]).read_bytes()).hexdigest()
                    for p in parts
                    if p["type"] == "image"
                },
            }
        )
        with locked(folder / "startup.lock"):
            old = read_json(folder / "request.json")
            if old and old["identity"] != identity:
                raise ModelCallError("immutable model request changed during resume")
            if not old:
                atomic_json(folder / "request.json", {**request, "identity": identity})
            if not (folder / "result.json").exists() and not exact_process(
                read_json(folder / "lease.json", {})
            ):
                try:
                    with locked(folder / "owner.lock", blocking=False):
                        pass
                except BlockingIOError:
                    pass  # a runner is initializing and has not written its lease yet
                else:
                    # A living orphan app-server still owns the old request. Never submit a
                    # duplicate.
                    while exact_process(read_json(folder / "appserver-lease.json", {})):
                        time.sleep(1)
                    # A crashed runner cannot produce a result. Preserve its uncompleted attempt as
                    # unknown.
                    inflight = read_json(folder / "inflight.json")
                    if inflight and (folder / "lease.json").exists():
                        from .persistence import append_json

                        receipts = folder / "receipts.jsonl"
                        prior = (
                            [json.loads(line) for line in receipts.read_text().splitlines()]
                            if receipts.exists()
                            else []
                        )
                        if inflight["attempt"] not in [r["attempt"] for r in prior]:
                            from .call_runner import reported_usage

                            events = folder / f"events.attempt{inflight['attempt']}.jsonl"
                            reported = reported_usage(events) if events.exists() else None
                            append_json(
                                receipts,
                                {
                                    **inflight,
                                    "completed_at": time.time(),
                                    "seconds": time.time() - inflight["started_at"],
                                    "status": "runner_interrupted",
                                    "usage": reported,
                                    "errors": ["runner exited before final receipt"],
                                },
                            )
                    with (folder / "runner.log").open("a") as log:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "robocasa_astra.astra_vla.call_runner",
                                "--folder",
                                str(folder),
                            ],
                            stdout=log,
                            stderr=log,
                            stdin=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                    while not (folder / "result.json").exists() and not exact_process(
                        read_json(folder / "lease.json", {})
                    ):
                        if process.poll() is not None:
                            raise ModelCallError(
                                f"model runner startup failed; see {folder / 'runner.log'}"
                            )
                        time.sleep(0.1)
        while not (folder / "result.json").exists():
            if not exact_process(read_json(folder / "lease.json", {})):
                raise ModelCallError(
                    "model runner exited; resume will preserve the interrupted attempt and retry"
                )
            time.sleep(0.5)
        result = read_json(folder / "result.json")
        usage = {}
        for receipt in result["receipts"]:
            if receipt["usage"] is not None:
                add_usage(usage, receipt["usage"])
        if result.get("error"):
            error = ModelCallError(result["error"])
            error.usage = usage
            raise error
        self.weekly = result.get("weekly")
        seconds = result.get("seconds", sum(r["seconds"] for r in result["receipts"]))
        reply = CallResult(
            result["text"], usage, result["reasoning"], seconds, result["receipts"], self.weekly
        )
        atomic_json(
            folder / "call.json",
            {
                "model": self.settings["model"],
                "effort": self.settings["effort"],
                "usage": usage or None,
                "unknown_usage_attempts": sum(r["usage"] is None for r in result["receipts"]),
                "seconds": seconds,
                "attempt_seconds": result.get("attempt_seconds", seconds),
                "attempts": result["receipts"],
                "reasoning": reply.reasoning,
                "final_text": reply.text,
            },
        )
        return reply

    def close(self) -> None:
        """Detach from the runner without terminating a model response in progress."""
