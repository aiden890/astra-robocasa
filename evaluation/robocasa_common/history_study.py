"""Supervise separate native-time history trials with resource guards and live playback."""

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
from PIL import Image
from robocasa_astra.checkpoint import atomic_json

from robocasa_common.depth_study import resource_sample, retain_failed_attempt, valid_evaluation
from robocasa_common.evaluate import evaluate_scene


def argv(root, job):
    """Identify one client by complete runtime and immutable job ID."""
    return [
        sys.executable,
        "-m",
        "robocasa_common.history_study",
        "--runtime",
        str(root),
        "--job",
        job["id"],
    ]


def alive(pid, expected):
    """Reject PID reuse, unrelated clients, and zombies before any adoption."""
    try:
        actual = Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
        stat = Path(f"/proc/{pid}/stat").read_text()
        return actual == expected and not stat.rsplit(") ", 1)[1].startswith("Z")
    except OSError:
        return False


def run_job(root, job):
    """Hold one private job lock and keep the original checkpoint/output namespace."""
    lock = (root / "locks" / (job["id"] + ".lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(
        ASTRA_DEPTH_CONDITION=job["condition"],
        ASTRA_DEPTH_MEMORY_ONLY="1",
        ASTRA_HISTORY_SECONDS=str(job["history_seconds"]),
        ASTRA_CODEX_HOME=str(root.parent / "auth"),
        ASTRA_CODEX_EXECUTABLE=str(
            root.parent
            / "codex/node_modules/@openai/codex-linux-x64/vendor"
            / "x86_64-unknown-linux-musl/bin/codex"
        ),
    )
    settings = SimpleNamespace(
        output=str(root / "results" / job["namespace"]),
        host="spark2",
        container=job["container"],
        mounted_root="/scene-bundles",
        policy="robocasa_common.history_policy:create_policy",
        compact_video=True,
        remote_verification=True,
        depth=True,
    )
    try:
        result = evaluate_scene(root / "scene-metadata/scenes" / job["scene"], settings)
    except Exception as error:
        result = {"execution_status": "error", "task_success": None, "error": str(error)}
        folder = root / "results" / job["namespace"] / job["scene"]
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "result.json", result)
    print(json.dumps(result), flush=True)


def publish(root, job, result, active):
    """Publish complete native frame chronology plus original per-call inputs and usage."""
    site = Path("/home/aiden/Desktop/lab/robot/astra-robocasa/status-page")
    folder = root / "results" / job["namespace"] / job["scene"]
    dest = site / "media" / root.name / job["id"]
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("visualization.json", "latest.jpg"):
        if (folder / name).exists():
            temporary = dest / (name + ".pending")
            shutil.copy2(folder / name, temporary)
            temporary.replace(dest / name)
    inputs = {}
    for call in sorted((folder / "policy").glob("call-*")):
        if not (call / "response.json").exists():
            continue
        manifest = call / "history-inputs.json"
        timeline = json.loads(manifest.read_text())["observations"] if manifest.exists() else []
        rows = []
        for offset, entry in enumerate(timeline):
            for name in entry["files"]:
                original = name if offset == 0 else f"history-{entry['step']:07d}-{name}"
                source = call / original
                target = dest / "inputs" / call.name / original
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(source, target)
                rows.append(
                    {
                        "camera": name.split(".")[0].removesuffix("__depth")
                        + f" ({entry['relative_seconds']:g}s)",
                        "kind": "depth" if "__depth" in name else "rgb",
                        "url": "/" + str(target.relative_to(site)),
                        "observation_id": entry["observation_id"],
                        "step": entry["step"],
                    }
                )
        inputs[call.name] = rows
    atomic_json(dest / "inputs.json", inputs)
    checkpoint = folder / "worker/checkpoint.json"
    step = json.loads(checkpoint.read_text())["steps"] if checkpoint.exists() else 0
    clip = folder / "live-clip.json"
    version = json.loads(clip.read_text())["version"] if clip.exists() else None
    previous = (
        json.loads((dest / "playback.json").read_text())
        if (dest / "playback.json").exists()
        else {}
    )
    if version and previous.get("source_version") != version:
        segments = sorted((folder / "live-segments").glob("*.mp4"))
        initial = np.array(Image.open(folder / "initial.png").convert("RGB"))
        pending = dest / "cumulative.pending.mp4"
        writer = imageio_ffmpeg.write_frames(
            str(pending),
            (768, 256),
            fps=20,
            codec="libx264",
            output_params=["-threads", "2", "-movflags", "+faststart"],
        )
        writer.send(None)
        last = 0
        try:
            writer.send(initial)
            for segment in segments:
                if int(segment.stem) != last:
                    raise ValueError("Cumulative segments have a gap or duplicate action")
                reader = imageio_ffmpeg.read_frames(str(segment))
                metadata = next(reader)
                if metadata["fps"] != 20:
                    raise ValueError("Segment frame rate changed")
                for index, frame in enumerate(reader):
                    if index:
                        writer.send(np.frombuffer(frame, np.uint8).reshape(256, 768, 3))
                        last += 1
        finally:
            writer.close()
        pending.replace(dest / "cumulative.mp4")
        atomic_json(
            dest / "playback.json",
            {
                "video": "/" + str((dest / "cumulative.mp4").relative_to(site)),
                "version": time.time_ns(),
                "source_version": version,
                "start_step": 0,
                "end_step": last,
                "frames": last + 1,
                "fps": 20,
            },
        )
    normal = valid_evaluation(result)
    if normal and (folder / "video.mp4").exists() and not (dest / "video.mp4").exists():
        reader = imageio_ffmpeg.read_frames(str(folder / "video.mp4"))
        metadata = next(reader)
        reader.close()
        if metadata["fps"] != 20:
            raise ValueError("Completed video is not 20fps")
        shutil.copy2(folder / "video.mp4", dest / "video.mp4")
        url = "http://100.86.183.64:8906/" + str((dest / "video.mp4").relative_to(site))
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=10
        ) as response:
            if response.status != 200:
                raise ValueError("Completed video HEAD failed")
        atomic_json(dest / "verification.json", {"fps": 20, "head": 200, "result": result})
    video = dest / ("video.mp4" if normal else "cumulative.mp4")
    if not video.exists():
        return None
    return {
        "id": root.name + "-" + job["id"],
        "collection": root.name,
        "task": job["task"],
        "scene_id": job["scene"],
        "condition": job["condition"],
        "model": "gpt-6-astra",
        "robot": "PandaOmron",
        "status": "user-stopped"
        if result.get("user_assessment") == "failed"
        else "complete"
        if normal
        else "running"
        if active
        else "error",
        "user_assessment": result.get("user_assessment"),
        "task_success": result.get("task_success"),
        "steps": step,
        "fps": 20,
        "video": str(video.relative_to(site)),
        "poster": str((dest / "latest.jpg").relative_to(site)),
        "visualization_data": str((dest / "visualization.json").relative_to(site)),
        "live_clip": not normal,
        "video_start_step": 0,
        "live_version": version,
        "description": f"과거 {job['history_seconds']}초 관측 · {job['condition']}",
    }


def supervise(root, plan):
    """Adopt exact live clients and dispatch bounded work only after fresh telemetry."""
    lock = (root / "study.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "study.pid").write_text(str(os.getpid()))
    jobs = plan["jobs"]
    active, finished, failed, attempts, children, assessed = {}, {}, {}, {}, {}, {}
    for job in jobs:
        pid_file = root / "pids" / (job["id"] + ".json")
        result_file = root / "results" / job["namespace"] / job["scene"] / "result.json"
        result = json.loads(result_file.read_text()) if result_file.exists() else {}
        if valid_evaluation(result):
            finished[job["id"]] = result
        elif result.get("user_assessment") == "failed":
            assessed[job["id"]] = result
        if pid_file.exists():
            old = json.loads(pid_file.read_text())
            attempts[job["id"]] = old["attempt"]
            if alive(old["pid"], argv(root, job)):
                job["container"] = old["container"]
                active[job["id"]] = (old["pid"], job)
    limit, last_ramp = plan.get("initial_parallel", 8), time.monotonic()
    while True:
        if (root / "STOP_REQUESTED").exists():
            return  # An exact stop operation must separately stop existing clients.
        for identity, (pid, job) in list(active.items()):
            if alive(pid, argv(root, job)):
                continue
            if identity in children:
                children.pop(identity).wait()
            folder = root / "results" / job["namespace"] / job["scene"]
            result = (
                json.loads((folder / "result.json").read_text())
                if (folder / "result.json").exists()
                else {
                    "execution_status": "error",
                    "task_success": None,
                    "error": "Client ended without result",
                }
            )
            if valid_evaluation(result):
                finished[identity] = result
            elif attempts.get(identity, 0) >= 3:
                failed[identity] = result
            else:
                retain_failed_attempt(root, {**job, "condition": job["namespace"]}, folder, result)
            del active[identity]
        pending = [
            j
            for j in jobs
            if j["id"] not in active
            and j["id"] not in finished
            and j["id"] not in failed
            and j["id"] not in assessed
        ]
        try:
            sample = resource_sample()
            sample.update(
                time=time.time(), lab_disk_free_gib=shutil.disk_usage(root).free / 1024**3
            )
            with (root / "resources.jsonl").open("a") as stream:
                stream.write(json.dumps(sample) + "\n")
            slots = max(
                0,
                len(active)
                + min(
                    int((sample["available_gib"] - 20) / 2.5),
                    int((sample["lab_available_gib"] - 4) / 1.5),
                ),
            )
            seconds = []
            capacity = False
            for receipt in (root / "results").glob("*/*/policy/call-*/receipt.json"):
                if time.time() - receipt.stat().st_mtime < 300:
                    row = json.loads(receipt.read_text())
                    seconds.append(row["cli_end_to_end_seconds"])
                    events = receipt.parent / "events.jsonl"
                    capacity |= events.exists() and '"capacity' in events.read_text().lower()
            p95 = sorted(seconds)[int(0.95 * (len(seconds) - 1))] if seconds else None
            if capacity or (p95 is not None and p95 > 120):
                limit = max(1, limit - 2)
                last_ramp = time.monotonic()
            elif time.monotonic() - last_ramp > 120 and len(seconds) >= limit * 3 and p95 < 120:
                limit = min(12, limit + 2)
                last_ramp = time.monotonic()
            used = {j["container"] for _, j in active.values()}
            free = [c for c in plan["containers"] if c not in used]
            if (
                sample["lab_available_gib"] >= 4
                and sample["available_gib"] >= 20
                and sample["lab_disk_free_gib"] >= 8
                and sample["load1"] < sample["cpus"] * 0.85
            ):
                for job in pending:
                    if not free or len(active) >= min(limit, slots):
                        break
                    saved = (
                        root / "results" / job["namespace"] / job["scene"] / "resume-container.json"
                    )
                    container = (
                        json.loads(saved.read_text())["container"] if saved.exists() else free[0]
                    )
                    if container not in free:
                        continue
                    free.remove(container)
                    job["container"] = container
                    folder = saved.parent
                    folder.mkdir(parents=True, exist_ok=True)
                    atomic_json(saved, {"container": container})
                    attempts[job["id"]] = attempts.get(job["id"], 0) + 1
                    atomic_json(root / "active-jobs" / (job["id"] + ".json"), job)
                    with (root / "logs" / (job["id"] + ".log")).open("a") as log:
                        proc = subprocess.Popen(
                            argv(root, job), stdout=log, stderr=log, start_new_session=True
                        )
                    active[job["id"]] = (proc.pid, job)
                    children[job["id"]] = proc
                    atomic_json(
                        root / "pids" / (job["id"] + ".json"),
                        {
                            "pid": proc.pid,
                            "argv": argv(root, job),
                            "container": container,
                            "attempt": attempts[job["id"]],
                        },
                    )
        except Exception as error:
            sample = {"telemetry_error": str(error), "time": time.time()}
        state = {
            "phase": "running"
            if active or pending
            else "complete"
            if len(finished) == len(jobs)
            else "recovery-needed",
            "expected": len(jobs),
            "normal_results": len(finished),
            "user_assessed_failures": assessed,
            "assessed_results": len(finished) + len(assessed),
            "results": finished,
            "failed": failed,
            "active": [
                {"id": i, "pid": p, "container": j["container"]} for i, (p, j) in active.items()
            ],
            "pending": len(
                [
                    j
                    for j in jobs
                    if j["id"] not in active
                    and j["id"] not in finished
                    and j["id"] not in failed
                    and j["id"] not in assessed
                ]
            ),
            "parallel_limit": limit,
            "resources": sample,
            "time": time.time(),
        }
        atomic_json(root / "status.json", state)
        records = []
        for job in jobs:
            if (
                job["id"] not in active
                and job["id"] not in finished
                and job["id"] not in failed
                and job["id"] not in assessed
            ):
                continue
            try:
                row = publish(
                    root,
                    job,
                    finished.get(job["id"], failed.get(job["id"], assessed.get(job["id"], {}))),
                    job["id"] in active,
                )
                if row:
                    records.append(row)
            except Exception as error:
                atomic_json(
                    root / "publication-errors" / (job["id"] + ".json"),
                    {"error": str(error), "time": time.time()},
                )
        site = Path("/home/aiden/Desktop/lab/robot/astra-robocasa/status-page")
        catalog = site / "media/supplemental-catalog.json"
        with (site / "media/history-catalog.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            before = json.loads(catalog.read_text())
            atomic_json(catalog, [r for r in before if r.get("collection") != root.name] + records)
        if not active and not pending:
            return
        time.sleep(10)


def main():
    """Run either the single locked supervisor or one immutable history job."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--job")
    args = parser.parse_args()
    root = Path(args.runtime)
    plan = json.loads((root / "plan.json").read_text())
    if args.job:
        # The supervisor saves slot ownership before spawning the client.
        path = root / "active-jobs" / (args.job + ".json")
        while not path.exists():
            time.sleep(0.05)
        run_job(root, json.loads(path.read_text()))
    else:
        supervise(root, plan)


if __name__ == "__main__":
    main()
