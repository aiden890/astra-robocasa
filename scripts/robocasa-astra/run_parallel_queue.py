"""Adopt the existing rollout and compare two then four independent simulator sessions."""

import json
import statistics
import subprocess
import time
from pathlib import Path

from publish_videos import publish
from runtime_io import alive, write_json_atomic


def snapshot(repo, runs, since):
    """Measure only completed inference receipts inside the selected wall-clock window."""
    calls, steps, latencies = 0, 0, []
    per_run = {}
    for run in runs:
        rows = []
        for path in (repo / "runs" / run / "inference").glob("call-*/receipt.json"):
            if path.stat().st_mtime >= since:
                try:
                    rows.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    continue
        total = sum(row["response"]["repeat"] for row in rows)
        per_run[run] = {"calls": len(rows), "steps": total}
        calls += len(rows)
        steps += total
        latencies += [row["seconds"] for row in rows]
    wall = time.time() - since
    return {
        "wall_seconds": wall,
        "calls": calls,
        "steps": steps,
        "aggregate_steps_per_second": steps / wall if wall else 0,
        "median_inference_seconds": statistics.median(latencies) if latencies else None,
        "per_run": per_run,
    }


def main():
    """Keep active runs intact; every added run has isolated simulator and output paths."""
    repo = Path(__file__).resolve().parents[2]
    runtime = repo / ".runtime"
    plan = json.loads((runtime / "parallel-plan.json").read_text())
    runs = [plan["adopt_run"]]
    pids = [plan["adopt_pid"]]
    children = []
    report = {"baseline": plan["baseline"], "stages": [], "max_configured": 4}
    for stage, additions in [(2, plan["new_runs"][:1]), (4, plan["new_runs"][1:])]:
        for entry in additions:
            output = repo / "runs" / entry["id"]
            if output.exists():
                raise FileExistsError(f"Refusing to repeat existing run: {output}")
            with (runtime / (entry["id"] + ".log")).open("w") as log:
                child = subprocess.Popen(
                    [
                        "bash",
                        "scripts/robocasa-astra/run.sh",
                        "--robot",
                        entry["robot"],
                        "--container",
                        entry["container"],
                        "--seed",
                        str(entry["seed"]),
                        "--steps",
                        "1800",
                        "--output",
                        str(output),
                    ],
                    cwd=repo,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            children.append(child)
            runs.append(entry["id"])
            pids.append(child.pid)
        start = time.time()
        while time.time() - start < 150:
            publish(repo)
            measure = snapshot(repo, runs, start)
            report["current"] = {"concurrency": stage, "pids": pids, **measure}
            write_json_atomic(runtime / "parallel-report.json", report)
            if any(not alive(pid) for pid in pids):
                report["status"] = "worker_ended_during_scaling"
                write_json_atomic(runtime / "parallel-report.json", report)
                return
            time.sleep(15)
        report["stages"].append({"concurrency": stage, **snapshot(repo, runs, start)})
    report["status"] = "four_parallel_running"
    write_json_atomic(runtime / "parallel-report.json", report)
    while any(alive(pid) for pid in pids):
        for child in children:
            child.poll()
        publish(repo)
        time.sleep(15)
    publish(repo)
    report["status"] = "complete"
    write_json_atomic(runtime / "parallel-report.json", report)


if __name__ == "__main__":
    main()
