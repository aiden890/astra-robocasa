"""Publish VLA native frames and original model receipts to the existing Lab viewer."""

from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from .persistence import atomic_json, locked, read_json


def publish(run, site):
    """Update cumulative 20fps playback and model output without changing evaluation results."""
    config = read_json(run / "config.json")
    if not config:
        return None
    ident = "vla-first-three-20261008-" + config["task"] + "-hybrid-scene0"
    relative = Path("media/vla-first-three-20261008") / config["task"]
    target = site / relative
    target.mkdir(parents=True, exist_ok=True)
    if not (target / "turns").exists():
        (target / "turns").symlink_to(run / "turns", target_is_directory=True)
    frames = run / "live-frames"
    end = -1
    while (frames / f"rgb-{end + 1:06d}.jpg").exists():
        end += 1
    previous = read_json(target / "playback.json", {})
    if end >= 0 and (previous.get("end_step") != end):
        for kind, filename in [("rgb", "cumulative.mp4"), ("depth", "depth.mp4")]:
            if not (frames / f"{kind}-{end:06d}.jpg").exists():
                continue
            temp = target / ("." + filename)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-framerate",
                    "20",
                    "-start_number",
                    "0",
                    "-i",
                    str(frames / (kind + "-%06d.jpg")),
                    "-frames:v",
                    str(end + 1),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "22",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temp),
                ],
                check=True,
            )
            temp.replace(target / filename)
        shutil.copyfile(frames / f"rgb-{end:06d}.jpg", target / "latest.jpg")
        atomic_json(
            target / "playback.json",
            {
                "video": str(relative / "cumulative.mp4"),
                "version": time.time_ns(),
                "start_step": 0,
                "end_step": end,
                "frames": end + 1,
                "fps": 20,
            },
        )
    previous = read_json(target / "playback.json", {})
    records = {
        r["turn"]: r
        for r in (read_json(f) for f in sorted((run / "policy" / "trace-records").glob("*.json")))
    }
    progress = read_json(run / "policy" / "progress.json", {})
    if progress.get("record"):
        records[progress["turn"]] = {
            **progress["record"],
            "query_history": progress.get("query_history", []),
        }
    calls, inputs = [], {}
    for folder in sorted((run / "calls").glob("turn*-q*")):
        response = read_json(folder / "response.json")
        if not response:
            continue
        turn = int(folder.name.split("-")[0][4:])
        round_number = int(folder.name.split("-q")[1])
        record = records.get(turn, {})
        call = read_json(folder / "call.json", {})
        result = read_json(folder / "result.json", {})
        receipt = read_json(folder / "request.json", {})
        usage = call.get("usage")
        if not usage:
            receipts = result.get("receipts", [])
            known = [r["usage"] for r in receipts if r.get("usage")]
            if known:
                usage = {k: sum(r.get(k, 0) for r in known) for k in known[0]}
        step = record.get("state", {}).get("steps_used", 0)
        answers = next(
            (
                q["answers"]
                for q in record.get("query_history", [])
                if q["round"] == round_number + 1
            ),
            [],
        )
        text = dict(response)
        text["reason"] = "\n\n".join(
            str(response.get(k, ""))
            for k in ["scene", "progress", "plan", "memory"]
            if response.get(k)
        )
        calls.append(
            {
                "call": folder.name,
                "step": step,
                "response": text,
                "usage": usage,
                "cli_seconds": call.get("seconds", result.get("seconds")),
                "query_answers": answers,
                "action_order": "dataset-vla12",
                "reasoning": call.get("reasoning", result.get("reasoning")),
            }
        )
        pictures = []
        for image in sorted((run / "turns").glob(f"turn{turn:03d}_*.png")):
            pictures.append(
                {
                    "camera": image.stem.split("_", 1)[1],
                    "kind": "rgb",
                    "url": str(relative / "turns" / image.name),
                    "observation_id": record.get("observation_id"),
                    "step": step,
                }
            )
        for image in sorted((run / "turns" / f"turn{turn:03d}").glob("*-depth.png")):
            pictures.append(
                {
                    "camera": image.stem[:-6],
                    "kind": "depth",
                    "url": str(relative / "turns" / f"turn{turn:03d}" / image.name),
                    "observation_id": record.get("observation_id"),
                    "step": step,
                }
            )
        inputs[folder.name] = pictures
    atomic_json(
        target / "visualization.json",
        {
            "control_hz": 20,
            "calls": calls,
            "timeline_note": "VLA 16-step chunk, dataset action order. Original responses and per-attempt receipts are preserved.",
        },
    )
    atomic_json(target / "inputs.json", inputs)
    summary = read_json(run / "summary.json", {})
    checkpoint = read_json(run / "worker" / "checkpoint.json", {})
    row = {
        "id": ident,
        "collection": "vla-first-three-20261008",
        "task": config["task"],
        "scene_id": config["scene_info"].get("scene_id", config["scene_info"].get("folder")),
        "condition": "hybrid",
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "robot": "PandaOmron",
        "status": "complete"
        if summary.get("task_success") is not None
        else "error"
        if summary
        else "running",
        "task_success": summary.get("task_success"),
        "steps": summary.get("steps_used", max(end, 0)),
        "fps": 20,
        "control_hz": 20,
        "video": str(relative / "cumulative.mp4"),
        "poster": str(relative / "latest.jpg"),
        "visualization_data": str(relative / "visualization.json"),
        "live_clip": True,
        "cumulative_live": True,
        "live_version": previous.get("version", 0),
        "video_start_step": 0,
        "action_order": "dataset-vla12",
        "depth_video": str(relative / "depth.mp4"),
        "depth_start_step": 0,
        "depth_end_step": max(end, 0),
        "depth_version": previous.get("version", 0),
        "description": "C: RGB + depth images + optional pixel Z; VLA context, shots=0, chunk=16",
    }
    with locked(site / "media" / "supplemental-catalog.lock"):
        catalog = read_json(site / "media" / "supplemental-catalog.json", [])
        catalog = [r for r in catalog if r["id"] != ident]
        catalog.insert(0, row)
        atomic_json(site / "media" / "supplemental-catalog.json", catalog)
    return {
        "id": ident,
        "steps": row["steps"],
        "responses": len(calls),
        "status": row["status"],
        "updated_at": time.time(),
    }


def main():
    """Publish only this three-trial run group, preserving all existing viewer records."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--site", type=Path, required=True)
    args = parser.parse_args()
    with locked(args.root / "publisher.lock", blocking=False):
        while True:
            statuses = []
            for run in sorted(args.root.glob("*-scene0")):
                try:
                    status = publish(run, args.site)
                    if status:
                        statuses.append(status)
                except Exception as error:
                    statuses.append({"run": str(run), "publisher_error": str(error)})
            atomic_json(args.root / "viewer-status.json", statuses)
            time.sleep(5)


if __name__ == "__main__":
    main()
