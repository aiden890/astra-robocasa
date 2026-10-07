"""Publish the complete experimental protocol and verified medium-depth trial metrics."""

import argparse
import hashlib
import html
import json
import math
import shutil
import urllib.request
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from robocasa_common.xiaomi_queue import atomic_json

LABELS = {
    "rgb": "RGB only",
    "color": "RGB + depth image",
    "pixel": "RGB + pixel Z query",
    "grid": "RGB + 16x16 grid Z query",
}


def summarize_calls(records):
    """Report observed token sums alongside unknown usage; cache is not added twice."""
    known = [r["usage"] for r in records if r.get("usage") is not None]
    times = [r["cli_end_to_end_seconds"] for r in records]
    return {
        "attempts": len(records),
        "usage_missing_attempts": len(records) - len(known),
        "reported_tokens": {
            k: sum(u[k] for u in known if u.get(k) is not None)
            if any(u.get(k) is not None for u in known)
            else None
            for k in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
        },
        "token_accounting_complete": len(known) == len(records)
        and all(
            all(
                u.get(k) is not None
                for k in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
            )
            for u in known
        ),
        "cli_seconds": {
            "sum": sum(times),
            "mean": float(np.mean(times)) if times else None,
            "median": float(np.median(times)) if times else None,
            "p95": float(np.percentile(times, 95)) if times else None,
        },
        "reasoning_tokens": None,
        "server_gpu_compute_seconds": None,
    }


def main():
    """Keep raw credentials and CLI transcripts private; publish measured receipt fields."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--site-root", required=True)
    args = parser.parse_args()
    root, site = Path(args.runtime), Path(args.site_root)
    public = site / "media/astra-depth-medium-20261007"
    public.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    status = (
        json.loads((root / "status.json").read_text())
        if (root / "status.json").exists()
        else {"phase": "prepared", "finished": 0, "active": []}
    )
    proof_path = root / "publication-verification.json"
    proofs = json.loads(proof_path.read_text()) if proof_path.exists() else {}
    rows, board = [], []
    for job in plan["jobs"]:
        folder = root / "results" / job["condition"] / job["scene"]
        calls = []
        path = folder / "policy/calls.jsonl"
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    calls.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        result = (
            json.loads((folder / "result.json").read_text())
            if (folder / "result.json").exists()
            else {}
        )
        row = {
            **job,
            "execution_status": result.get("execution_status", "pending"),
            "task_success": result.get("task_success"),
            "calls": summarize_calls(calls),
        }
        if result.get("native_steps") is not None:
            row["steps"] = result["native_steps"]
        log_files = sorted(
            (folder / "eval").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if log_files:
            stats = json.loads(log_files[0].read_text())["stats"]
            row.update(
                steps=result.get("native_steps") or stats["total_steps"],
                wall_seconds=stats["duration_s"],
            )
        counters = folder / "policy/summary.json"
        if counters.exists():
            row["query_requests"] = json.loads(counters.read_text())["query_requests"]
        if result.get("execution_status") == "success":
            video = folder / "video.mp4"
            name = job["id"] + ".mp4"
            target = public / name
            if job["id"] not in proofs:
                reader = imageio_ffmpeg.read_frames(str(video))
                try:
                    metadata, first = next(reader), next(reader)
                finally:
                    reader.close()
                if metadata["fps"] != 20 or tuple(metadata["size"]) != (768, 256):
                    raise ValueError("Completed video has wrong FPS or camera layout")
                shutil.copyfile(video, target)
                Image.fromarray(np.frombuffer(first, np.uint8).reshape(256, 768, 3)).save(
                    public / (job["id"] + ".jpg")
                )
                request = urllib.request.Request(
                    "http://100.86.183.64:8906/media/astra-depth-medium-20261007/" + name,
                    method="HEAD",
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    if (
                        response.status != 200
                        or int(response.headers["Content-Length"]) != target.stat().st_size
                    ):
                        raise ValueError("Video publication failed")
                proofs[job["id"]] = {
                    "fps": 20,
                    "size": [768, 256],
                    "head": 200,
                    "bytes": target.stat().st_size,
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
                atomic_json(proof_path, proofs)
            row["video"] = name
            board.append(
                {
                    "id": job["id"],
                    "kind": "rollout",
                    "task": job["task"],
                    "robot": "PandaOmron",
                    "model": "gpt-6-astra",
                    "collection": "astra-depth-medium-20261007",
                    "status": "complete",
                    "task_success": row["task_success"],
                    "success": row["task_success"],
                    "steps": row.get("steps", 0),
                    "max_steps": job["horizon"],
                    "fps": 20,
                    "duration": (row.get("steps", 0) + 1) / 20,
                    "video": "media/astra-depth-medium-20261007/" + name,
                    "poster": "media/astra-depth-medium-20261007/" + job["id"] + ".jpg",
                    "type": LABELS[job["condition"]],
                    "description": "medium · " + job["scene"] + " · " + LABELS[job["condition"]],
                }
            )
        rows.append(row)
    conditions = {}
    for condition in LABELS:
        conditions[condition] = {}
        for task in plan["tasks"]:
            group = [r for r in rows if r["condition"] == condition and r["task"] == task]
            normal = [r for r in group if r["execution_status"] == "success"]
            conditions[condition][task] = {
                "expected": 10,
                "evaluated": len(normal),
                "execution_errors": sum(r["execution_status"] == "error" for r in group),
                "native_successes": sum(r["task_success"] is True for r in normal),
                "success_rate": sum(r["task_success"] is True for r in normal) / 10
                if len(normal) == 10
                else None,
            }
    comparisons = []
    for condition in ["color", "pixel", "grid"]:
        for task in plan["tasks"]:
            control = {
                r["scene"]: r
                for r in rows
                if r["condition"] == "rgb"
                and r["task"] == task
                and r["execution_status"] == "success"
            }
            treatment = {
                r["scene"]: r
                for r in rows
                if r["condition"] == condition
                and r["task"] == task
                and r["execution_status"] == "success"
            }
            if len(control) == 10 and len(treatment) == 10:
                gained = sum(
                    not control[s]["task_success"] and treatment[s]["task_success"] for s in control
                )
                lost = sum(
                    control[s]["task_success"] and not treatment[s]["task_success"] for s in control
                )
                n = gained + lost
                p = (
                    min(1.0, 2 * sum(math.comb(n, k) for k in range(min(gained, lost) + 1)) / 2**n)
                    if n
                    else 1.0
                )
                comparisons.append(
                    {
                        "condition": condition,
                        "task": task,
                        "paired_n": 10,
                        "gained": gained,
                        "lost": lost,
                        "exact_mcnemar_p": p,
                    }
                )
    report = {
        "protocol": plan["protocol"],
        "phase": status["phase"],
        "finished": status["finished"],
        "active": status.get("active", []),
        "parallel_limit": status.get("parallel_limit", 0),
        "resources": status.get("resources"),
        "conditions": conditions,
        "rows": rows,
        "paired_comparisons": comparisons,
    }
    atomic_json(public / "experiment.json", report)
    atomic_json(public / "plan.json", {k: v for k, v in plan.items() if k != "xiaomi_runtime"})
    table = "".join(
        f"<tr><td>{html.escape(r['task'])}</td><td>{html.escape(r['scene'])}</td>"
        f"<td>{LABELS[r['condition']]}</td><td>{html.escape(r['execution_status'])}</td>"
        f"<td>{r['task_success'] if r['task_success'] is not None else '—'}</td>"
        f"<td>{r.get('steps', '—')}</td>"
        f"<td>{r['calls']['attempts']}</td><td>{r['calls']['reported_tokens']['input_tokens']}</td>"
        f"<td>{r['calls']['reported_tokens']['cached_input_tokens']}</td>"
        f"<td>{r['calls']['reported_tokens']['output_tokens']}</td>"
        f"<td>{r['calls']['usage_missing_attempts']}</td><td>{r['calls']['cli_seconds']['median']}</td>"
        f"<td>{('<a href=' + r['video'] + '>영상</a>') if r.get('video') else '—'}</td></tr>"
        for r in rows
    )
    page = Path(__file__).with_name("depth_protocol.html").read_text()
    page += (
        f"<p>단계: {html.escape(status['phase'])} · 종료 {status['finished']}/200 · "
        f"병렬 상한 {status.get('parallel_limit', 0)}</p>"
    )
    page += (
        '<div class="scroll"><table><thead><tr><th>Task</th><th>Scene</th>'
        "<th>Condition</th><th>Status</th><th>Native success</th><th>Steps</th><th>Calls</th>"
        "<th>Input</th><th>Cached</th><th>Output</th><th>Usage unknown</th>"
        "<th>Call median(s)</th><th>Video</th></tr></thead><tbody>"
        + table
        + "</tbody></table></div>"
    )
    (public / "index.html").write_text(page)
    if board:
        path = site / "media/supplemental-catalog.json"
        before = json.loads(path.read_text()) if path.exists() else []
        before = [r for r in before if r.get("collection") != "astra-depth-medium-20261007"]
        atomic_json(path, before + board)
    path = site / "topics.json"
    topics = json.loads(path.read_text())
    topics = [t for t in topics if t["id"] != "astra-depth-medium"]
    topic = {
        "id": "astra-depth-medium",
        "title": "Astra RGB·depth 비교 실험 · medium · 200회",
        "category": "공통 씬 평가",
        "task": "5개 태스크",
        "model": "gpt-6-astra · medium",
        "environment": "Spark2 · PandaOmron",
        "date": "2026-10-07",
        "tags": ["RGB", "depth", "medium", "200회"],
        "description": "실험 조건·입출력·토큰·호출 시간 상세: http://100.86.183.64:8906/media/astra-depth-medium-20261007/index.html",
        "run_ids": [r["id"] for r in board],
    }
    atomic_json(path, [topic, *topics])


if __name__ == "__main__":
    main()
