"""Supervise 200 immutable paired depth trials with resource-aware concurrency."""

import argparse
import fcntl
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

from robocasa_common.xiaomi_queue import atomic_json


def resource_sample():
    """Read available RAM and CPU load without exposing process environments."""
    script = (
        "import json,pathlib,os; m={l.split(':')[0]:int(l.split()[1]) "
        "for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if ':' in l "
        "and l.split()[1].isdigit()}; print(json.dumps({'available_gib':"
        "m['MemAvailable']/1024**2,'load1':os.getloadavg()[0],'cpus':os.cpu_count()}))"
    )
    remote = json.loads(
        subprocess.check_output(
            ["ssh", "spark2", "python3 -c " + shlex.quote(script)], text=True, timeout=30
        )
    )
    local_available = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            local_available = int(line.split()[1]) / 1024**2
    remote.update(lab_available_gib=local_available)
    return remote


def recent_call_seconds(root, now, window=300):
    """Measure recently finished calls across trials, independent of directory order."""
    records = []
    for file in (root / "results").glob("*/*/policy/call-*/receipt.json"):
        modified = file.stat().st_mtime
        if 0 <= now - modified <= window:
            try:
                record = json.loads(file.read_text())
            except json.JSONDecodeError:
                continue  # A receipt may still be being written by an active call.
            records.append((modified, record["cli_end_to_end_seconds"]))
    return [seconds for _, seconds in sorted(records)[-200:]]


class AdoptedProcess:
    """Observe an orphaned trial by exact argv identity without restarting it."""

    def __init__(self, pid, root, job, proc_root=Path("/proc")):
        self.pid = pid
        self.root = str(root)
        self.job = job
        self.proc_root = proc_root
        self.returncode = None  # An orphan's original exit status is not available.

    def poll(self):
        """Treat PID reuse or a zombie as ended; never signal the process."""
        try:
            args = (self.proc_root / str(self.pid) / "cmdline").read_bytes().decode().split("\0")
            if "robocasa_common.depth_trial" not in args:
                return 1
            for flag, value in (
                ("--runtime", self.root),
                ("--scene", self.job["scene"]),
                ("--condition", self.job["condition"]),
            ):
                if flag not in args or args[args.index(flag) + 1] != value:
                    return 1
            stat = (self.proc_root / str(self.pid) / "stat").read_text()
            return 1 if stat.rsplit(") ", 1)[1].startswith("Z") else None
        except (OSError, ValueError, IndexError):
            return 1


def main():
    """Adopt only terminal receipts; preserve partial output rather than rerolling it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--site-root", required=True)
    args = parser.parse_args()
    root = Path(args.runtime)
    lock = (root / "study.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "study.pid").write_text(str(os.getpid()))
    plan = json.loads((root / "plan.json").read_text())
    proof = json.loads((root / "preflight.json").read_text())
    if not proof.get("passed"):
        raise RuntimeError("Native depth and actual model-call preflight has not passed")
    previous = (
        json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
    )
    prior_active = {item["id"]: item for item in previous.get("active", [])}
    jobs = plan["jobs"]
    active, finished, logs = {}, {}, {}
    # Do not overlap the Xiaomi learner GPU server; the first pass plus repair is the gate.
    baseline = Path(plan["xiaomi_runtime"])
    while not (
        (baseline / "recovery-status.json").exists()
        and json.loads((baseline / "recovery-status.json").read_text()).get("complete")
    ):
        atomic_json(
            root / "status.json",
            {
                "phase": "waiting_for_xiaomi",
                "expected": len(jobs),
                "finished": 0,
                "active": [],
                "parallel_limit": 0,
            },
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "robocasa_common.depth_site",
                "--runtime",
                str(root),
                "--site-root",
                args.site_root,
            ],
            check=True,
        )
        time.sleep(20)
    # Completed, project-owned baseline containers only; results and model files remain.
    if not prior_active:
        subprocess.run(
            [
                "ssh",
                "spark2",
                "docker",
                "stop",
                "xiaomi-common-scenes-model-20261007",
                "xiaomi-common-scenes-sim-20261007",
            ],
            check=True,
        )
    slots = [f"astra-depth-medium-{i:02d}-20261007" for i in range(1, 33)]
    free = slots.copy()
    for job in jobs:
        folder = root / "results" / job["condition"] / job["scene"]
        result = folder / "result.json"
        old = prior_active.get(job["id"])
        adopted = AdoptedProcess(old["pid"], root, job) if old else None
        if adopted is not None and adopted.poll() is None:
            slot = old["container"]
            if slot not in free:
                raise ValueError("Previously active trials share an invalid slot")
            free.remove(slot)
            active[job["id"]] = (adopted, slot, job)
            logs[job["id"]] = (root / "logs" / (job["id"] + ".log")).open("a")
        elif result.exists():
            finished[job["id"]] = json.loads(result.read_text())
        elif folder.exists():
            finished[job["id"]] = {
                "execution_status": "error",
                "task_success": None,
                "error": "Partial trial retained; no automatic reroll",
            }
    pending = [j for j in jobs if j["id"] not in finished and j["id"] not in active]
    limit = max(1, previous.get("parallel_limit", 8)) if prior_active else 8
    last_ramp, started = time.monotonic(), previous.get("started_at", time.time())
    atomic_json(
        root / "adoption-receipt.json",
        {
            "time": time.time(),
            "adopted": [
                {"id": identity, "pid": item[0].pid, "container": item[1]}
                for identity, item in active.items()
            ],
            "restarted_trials": 0,
        },
    )
    while pending or active:
        for identity, (process, slot, job) in list(active.items()):
            if process.poll() is None:
                continue
            folder = root / "results" / job["condition"] / job["scene"]
            path = folder / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
            else:
                result = {
                    "execution_status": "error",
                    "task_success": None,
                    "exit_code": process.returncode,
                    "error": "Trial subprocess ended without result",
                }
                folder.mkdir(parents=True, exist_ok=True)
                atomic_json(path, result)
            finished[identity] = result
            logs.pop(identity).close()
            active.pop(identity)
            free.append(slot)
        try:
            sample = resource_sample()
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            limit = 1
            atomic_json(
                root / "telemetry-error.json",
                {"time": time.time(), "error": str(error), "dispatch_blocked": True},
            )
            atomic_json(
                root / "status.json",
                {
                    "phase": "telemetry_guard",
                    "expected": len(jobs),
                    "finished": len(finished),
                    "results": finished,
                    "active": [
                        {"id": i, "pid": v[0].pid, "container": v[1]} for i, v in active.items()
                    ],
                    "parallel_limit": limit,
                    "pending": len(pending),
                    "started_at": started,
                    "model": "gpt-6-astra",
                    "effort": "medium",
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "robocasa_common.depth_site",
                    "--runtime",
                    str(root),
                    "--site-root",
                    args.site_root,
                ],
                check=False,
            )
            time.sleep(20)
            continue

        sample.update(
            time=time.time(),
            active=len(active),
            target=limit,
            lab_disk_free_gib=shutil.disk_usage(root).free / 1024**3,
        )
        with (root / "resources.jsonl").open("a") as f:
            f.write(json.dumps(sample) + "\n")
        receipts = []
        for folder in (root / "results").glob("*/*/policy"):
            file = folder / "calls.jsonl"
            if file.exists():
                for line in file.read_text().splitlines():
                    try:
                        receipts.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        recent_errors = 0
        for folder in (root / "results").glob("*/*/policy/call-*"):
            file = folder / "events.jsonl"
            if file.exists() and time.time() - file.stat().st_mtime < 300:
                recent_errors += "capacity" in file.read_text().lower()
        seconds = recent_call_seconds(root, time.time())
        # RAM guard is conservative until per-slot peak usage is measured.
        memory_slots = max(1, len(active) + int((sample["available_gib"] - 16) / 2.5))
        if sample["available_gib"] < 16 or sample["lab_available_gib"] < 2 or recent_errors:
            limit = max(1, min(limit - 4, memory_slots))
            last_ramp = time.monotonic()
        elif (
            time.monotonic() - last_ramp > 120
            and len(receipts) >= limit * 3
            and sample["load1"] < sample["cpus"] * 0.85
            and seconds
            and sorted(seconds)[int(0.95 * (len(seconds) - 1))] < 120
        ):
            limit = min(32, limit + 4, memory_slots)
            last_ramp = time.monotonic()
        while pending and free and len(active) < min(limit, memory_slots):
            if (
                sample["lab_disk_free_gib"] < 2
                or sample["available_gib"] < 16
                or sample["lab_available_gib"] < 2
            ):
                break
            job = pending.pop(0)
            slot = free.pop(0)
            log = (root / "logs" / (job["id"] + ".log")).open("a")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "robocasa_common.depth_trial",
                    "--runtime",
                    str(root),
                    "--scene",
                    job["scene"],
                    "--condition",
                    job["condition"],
                    "--container",
                    slot,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            active[job["id"]] = (process, slot, job)
            logs[job["id"]] = log
            with (root / "dispatch.jsonl").open("a") as f:
                f.write(
                    json.dumps({**job, "pid": process.pid, "container": slot, "time": time.time()})
                    + "\n"
                )
        status = {
            "phase": "running" if active else "storage_gate",
            "expected": len(jobs),
            "finished": len(finished),
            "results": finished,
            "active": [{"id": i, "pid": v[0].pid, "container": v[1]} for i, v in active.items()],
            "parallel_limit": limit,
            "pending": len(pending),
            "resources": sample,
            "started_at": started,
            "model": "gpt-6-astra",
            "effort": "medium",
        }
        atomic_json(root / "status.json", status)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "robocasa_common.depth_site",
                "--runtime",
                str(root),
                "--site-root",
                args.site_root,
            ],
            check=True,
        )
        time.sleep(20)
    status.update(phase="complete", active=[], finished=len(finished))
    atomic_json(root / "status.json", status)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "robocasa_common.depth_site",
            "--runtime",
            str(root),
            "--site-root",
            args.site_root,
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
