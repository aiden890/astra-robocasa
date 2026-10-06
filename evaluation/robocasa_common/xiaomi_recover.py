"""Finish unscored adapter-error scenes once after the initial queue releases its lock."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from robocasa_common.evaluate import evaluate_scene
from robocasa_common.xiaomi_queue import atomic_json


def main():
    """Preserve all first-pass results and record repaired attempts separately."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--site-root", required=True)
    args = parser.parse_args()
    root, site = Path(args.runtime), Path(args.site_root)
    own = (root / "recovery.lock").open("w")
    fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "recovery.pid").write_text(str(os.getpid()))
    queue = (root / "queue.lock").open("a")
    while True:
        try:
            fcntl.flock(queue, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(20)
    state = json.loads((root / "status.json").read_text())
    if not state.get("complete"):
        raise RuntimeError("Initial queue exited early; preserve state and inspect before recovery")
    first = root / "first-pass-status.json"
    if not first.exists():
        shutil.copyfile(root / "status.json", first)
    original = json.loads(first.read_text())
    rows = original["results"]
    os.environ["XIAOMI_COMPACT_VIDEO"] = "1"
    os.environ["XIAOMI_CLIENT_COMMAND"] = json.dumps(
        [
            "ssh",
            "spark2",
            "docker",
            "exec",
            "-i",
            "-e",
            "PYTHONPATH=/eval-code/evaluation",
            "xiaomi-common-scenes-model-20261007",
            "python3",
            "-m",
            "robocasa_common.xiaomi_client_worker",
        ]
    )
    run_args = SimpleNamespace(
        output=str(root / "recovered-results"),
        host="spark2",
        container="xiaomi-common-scenes-sim-20261007",
        mounted_root="/scene-bundles",
        policy="robocasa_common.xiaomi:create_policy",
        compact_video=True,
        remote_verification=True,
    )
    public = site / "media/xiaomi-common-scenes-20261007"
    for index, row in enumerate(rows):
        if row["execution_status"] != "error":
            continue
        identity = row["scene"]
        folder = root / "recovered-results" / identity
        result_path = folder / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
        elif folder.exists():
            raise RuntimeError("Partial recovery preserved; do not reroll it")
        else:
            atomic_json(root / "recovery-status.json", {"active": identity, "complete": False})
            result = evaluate_scene(root / "scene-metadata/scenes" / identity, run_args)
        result.update(task=row["task"], horizon=row["horizon"], original_attempt=row)
        logs = list((folder / "eval").glob("*.json"))
        if logs:
            stats = json.loads(logs[0].read_text())["stats"]
            result.update(steps=stats["total_steps"], seconds=stats["duration_s"])
        calls = folder / "policy/calls.jsonl"
        result["model_calls"] = len(calls.read_text().splitlines()) if calls.exists() else 0
        if result["execution_status"] == "success":
            shutil.copyfile(folder / "video.mp4", public / (identity + ".mp4"))
            result.update(video=identity + ".mp4", fps=20)
        rows[index] = result
        atomic_json(
            root / "recovery-status.json", {"active": None, "results": rows, "complete": False}
        )
    state.update(results=rows, active=None, adapter_recovery_complete=True)
    for task, summary in state["tasks"].items():
        selected = [r for r in rows if r["task"] == task]
        normal = [r for r in selected if r["execution_status"] == "success"]
        summary.update(
            evaluated=len(normal),
            execution_errors=len(selected) - len(normal),
            task_successes=sum(r["task_success"] is True for r in normal),
            success_rate=sum(r["task_success"] is True for r in normal) / 10
            if len(normal) == 10
            else None,
        )
    atomic_json(root / "status.json", state)
    atomic_json(public / "status.json", state)
    atomic_json(root / "recovery-status.json", {"active": None, "complete": True})
    # The publisher verifies actual FPS/dimensions/HTTP before adding repaired videos.
    subprocess.run(
        [
            os.sys.executable,
            "-m",
            "robocasa_common.xiaomi_publish",
            "--runtime",
            str(root),
            "--site-root",
            str(site),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
