"""Keep a model call alive and its receipt durable when the policy client exits."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

from robocasa_astra.checkpoint import atomic_json

from robocasa_common.depth_policy import DepthPolicy, usage_from_events


def retained_call(folder, request):
    """Wait for an existing exact CLI PID; never duplicate an interrupted request."""
    pid_file = folder / "cli-pid.json"
    if not pid_file.exists():
        return None
    pid = json.loads(pid_file.read_text())["pid"]
    while True:
        try:
            args = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
            alive = "-o" in args and args[args.index("-o") + 1] == str(folder / "response.json")
        except (OSError, ValueError, IndexError):
            alive = False
        if not alive:
            break
        time.sleep(0.2)
    timeline = (
        [json.loads(line) for line in (folder / "events-live.jsonl").read_text().splitlines()]
        if (folder / "events-live.jsonl").exists()
        else []
    )
    receipt = folder / "receipt.json"
    if not receipt.exists():
        usage = usage_from_events([e["event"] for e in timeline])
        record = {
            "attempt": request["index"],
            "condition": request["condition"],
            "effort": "medium",
            "cli_end_to_end_seconds": time.time() - request["started_at"],
            "usage": usage,
            "usage_complete": usage is not None,
            "turn_event_seconds": None,
            "image_count": len(request["images"]),
            "runner_interrupted": True,
            "timing_includes_recovery": True,
        }
        atomic_json(receipt, record)
        with (Path(request["output"]) / "calls.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
    response = folder / "response.json"
    if response.exists() and any(e["event"].get("type") == "turn.completed" for e in timeline):
        return {"ok": True, "value": json.loads(response.read_text()), "adopted_cli": True}
    return {
        "ok": False,
        "error": "Interrupted model runner retained; CLI ended without a verified response",
    }


def main():
    """Use a per-call lock to prevent duplicate subscription requests on resume."""
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    folder = Path(parser.parse_args().folder)
    with (folder / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / "runner-result.json").exists():
            return
        request = json.loads((folder / "runner-request.json").read_text())
        atomic_json(folder / "runner-pid.json", {"pid": os.getpid(), "started_at": time.time()})
        policy = DepthPolicy.__new__(DepthPolicy)
        for name in ("home", "executable", "condition", "index"):
            setattr(policy, name, request[name])
        policy.output = Path(request["output"])
        policy.calls = []
        try:
            value = policy.blocking_call(
                request["prompt"], [Path(x) for x in request["images"]], folder
            )
            atomic_json(folder / "runner-result.json", {"ok": True, "value": value})
        except Exception as error:
            if not (folder / "receipt.json").exists():
                record = {
                    "attempt": request["index"],
                    "condition": request["condition"],
                    "effort": "medium",
                    "usage": None,
                    "usage_complete": False,
                    "cli_end_to_end_seconds": time.time() - request["started_at"],
                    "turn_event_seconds": None,
                    "image_count": len(request["images"]),
                    "launch_error": str(error),
                }
                atomic_json(folder / "receipt.json", record)
                with (policy.output / "calls.jsonl").open("a") as log:
                    log.write(json.dumps(record) + "\n")
            atomic_json(folder / "runner-result.json", {"ok": False, "error": str(error)})


if __name__ == "__main__":
    main()
