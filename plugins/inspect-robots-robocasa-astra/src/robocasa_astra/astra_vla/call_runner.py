"""One detached model request, with indefinite wait and a durable receipt for every actual
attempt."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..astra_robodawn.appserver import AppServerCaller
from ..astra_robodawn.codex import CAPACITY_MARKERS, write_input
from .persistence import append_json, atomic_json, locked, process_lease, read_json


class RecordedCaller(AppServerCaller):
    """Publish the app-server identity so a replacement runner cannot overlap an orphan."""

    def process_started(self) -> None:
        """Persist the provider process identity before its first RPC can start a turn."""
        argv = (
            __import__("subprocess")
            .check_output(["ps", "-p", str(self.proc.pid), "-o", "command="], text=True)
            .strip()
        )
        atomic_json(
            self.receipt_folder / "appserver-lease.json", {"pid": self.proc.pid, "argv": argv}
        )


def reported_usage(path: Path):
    """Recover the last reported usage from original events, including interrupted calls."""
    from ..astra_robodawn.appserver import usage_from

    usage = None
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # preserve partial event bytes without counting them as reported usage
        if event.get("method") == "thread/tokenUsage/updated":
            usage = usage_from(event["params"]["tokenUsage"]["last"])
    return usage


def run_request(folder: Path) -> None:
    """Hold the per-call lock until a validated final response or an actual process/provider
    failure."""
    with locked(folder / "owner.lock", blocking=False):
        if (folder / "result.json").exists():
            return
        request = read_json(folder / "request.json")
        atomic_json(folder / "lease.json", process_lease())
        caller = RecordedCaller(
            request["executable"],
            request["codex_home"],
            Path(request["system"]),
            Path(request["schema"]),
            Path(request["workdir"]),
            model=request["model"],
            call_timeout=None,
            effort=request["effort"],
        )
        caller.receipt_folder = folder
        write_input(folder, request["parts"])
        receipts = folder / "receipts.jsonl"
        existing = (
            [json.loads(line) for line in receipts.read_text().splitlines()]
            if receipts.exists()
            else []
        )

        def finish(result):
            result["seconds"] = time.time() - existing[0]["started_at"] if existing else 0
            result["attempt_seconds"] = sum(r["seconds"] for r in existing)
            atomic_json(folder / "result.json", result)

        try:
            while True:
                ledger = Path(request["run_dir"]) / "policy" / "attempts.jsonl"
                with locked(ledger.with_suffix(".lock")):
                    count = len(ledger.read_text().splitlines()) if ledger.exists() else 0
                    if count >= request["attempt_budget"]:
                        finish(
                            {"error": "model attempt budget exhausted", "receipts": existing},
                        )
                        return
                    attempt = count + 1
                    started = time.time()
                    append_json(
                        ledger, {"attempt": attempt, "started_at": started, "call_dir": str(folder)}
                    )
                suffix = f".attempt{attempt}"
                atomic_json(folder / "inflight.json", {"attempt": attempt, "started_at": started})
                with (folder / f"events{suffix}.jsonl").open("a", buffering=1) as log:
                    caller.log = log
                    try:
                        out = caller._turn(request["parts"], folder, suffix)
                    except (ConnectionError, RuntimeError, TimeoutError, OSError) as exc:
                        out = {
                            "status": "client_error",
                            "errors": [f"{type(exc).__name__}: {exc}"],
                            "usage": None,
                        }
                    finally:
                        caller.log = None
                receipt = {
                    "attempt": attempt,
                    "started_at": started,
                    "completed_at": time.time(),
                    "seconds": time.time() - started,
                    "status": out["status"],
                    "usage": out.get("usage") or reported_usage(folder / f"events{suffix}.jsonl"),
                    "errors": out.get("errors", []),
                    "reasoning": out.get("reasoning", []),
                }
                append_json(receipts, receipt)
                existing.append(receipt)
                if out["status"] == "completed" and out.get("text") and not out["errors"]:
                    # Action/query validation is performed by the host. This must still be a JSON
                    # object.
                    try:
                        parsed = json.loads(out["text"])
                        if not isinstance(parsed, dict):
                            raise ValueError("response is not an object")
                    except (ValueError, TypeError) as exc:
                        finish({"error": str(exc), "receipts": existing})
                        return
                    atomic_json(folder / "response.json", parsed)
                    finish(
                        {
                            "text": out["text"],
                            "reasoning": out.get("reasoning", []),
                            "receipts": existing,
                            "weekly": caller.weekly,
                        },
                    )
                    return
                error = " ".join(out.get("errors", []))
                caller.close()
                if out["status"] == "client_error" or any(
                    m.lower() in error.lower() for m in CAPACITY_MARKERS
                ):
                    time.sleep(
                        15
                    )  # same immutable observation; no simulator action or query round consumed
                    continue
                finish({"error": error or out["status"], "receipts": existing})
                return
        finally:
            caller.close()


def main() -> None:
    """Entry point used exclusively by DurableCaller, not an experiment launcher."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    run_request(parser.parse_args().folder)


if __name__ == "__main__":
    main()
