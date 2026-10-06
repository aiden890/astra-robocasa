"""Publish completed Xiaomi evaluations after actual video and HTTP verification."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image

from robocasa_common.xiaomi_queue import atomic_json


def main():
    """Update the existing board, retaining unrelated topics and user tombstones."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--site-root", required=True)
    args = parser.parse_args()
    runtime, site = Path(args.runtime), Path(args.site_root)
    public = site / "media/xiaomi-common-scenes-20261007"
    checked = {}
    proof = runtime / "publication-verification.json"
    if proof.exists():
        checked = json.loads(proof.read_text())
    while True:
        status_path = runtime / "status.json"
        if not status_path.exists():
            time.sleep(10)
            continue
        state = json.loads(status_path.read_text())
        for row in state["results"]:
            identity = row["scene"]
            if row["execution_status"] != "success" or identity in checked:
                continue
            video = public / (identity + ".mp4")
            reader = imageio_ffmpeg.read_frames(str(video))
            try:
                meta = next(reader)
                first = next(reader)
            finally:
                reader.close()
            if meta["fps"] != 20 or tuple(meta["size"]) != (768, 256):
                raise ValueError("Completed video has incorrect FPS or camera layout")
            request = urllib.request.Request(
                "http://100.86.183.64:8906/media/xiaomi-common-scenes-20261007/" + video.name,
                method="HEAD",
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                if (
                    response.status != 200
                    or int(response.headers["Content-Length"]) != video.stat().st_size
                ):
                    raise ValueError("Published video HEAD verification failed")
            Image.fromarray(np.frombuffer(first, dtype=np.uint8).reshape(256, 768, 3)).save(
                public / (identity + ".jpg")
            )
            checked[identity] = {
                "fps": meta["fps"],
                "size": meta["size"],
                "head": 200,
                "bytes": video.stat().st_size,
                "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            }
            atomic_json(proof, checked)
            print(identity, "20fps HEAD200 verified", flush=True)
        records = []
        for row in state["results"]:
            if row["scene"] not in checked:
                continue
            identity = row["scene"]
            records.append(
                {
                    "id": "xiaomi-common-" + identity,
                    "kind": "rollout",
                    "task": row["task"],
                    "robot": "PandaOmron",
                    "model": "Xiaomi-Robotics-1-RoboCasa365",
                    "collection": "xiaomi-common-scenes-20261007",
                    "status": "complete",
                    "success": row["task_success"],
                    "task_success": row["task_success"],
                    "steps": row.get("steps", 0),
                    "max_steps": row["horizon"],
                    "fps": 20,
                    "duration": (row.get("steps", 0) + 1) / 20,
                    "video": "media/xiaomi-common-scenes-20261007/" + identity + ".mp4",
                    "poster": "media/xiaomi-common-scenes-20261007/" + identity + ".jpg",
                    "type": "샤오미 · 공통 고정 씬 평가",
                    "description": (
                        f"{row['task']} · {identity} · native 전체 성공 조건 · 단일 평가"
                    ),
                    "scene_id": identity,
                    "seed": int(identity.rsplit("-", 1)[1]),
                }
            )
        if records:
            path = site / "media/supplemental-catalog.json"
            before = json.loads(path.read_text()) if path.exists() else []
            before = [r for r in before if r.get("collection") != "xiaomi-common-scenes-20261007"]
            atomic_json(path, before + records)
            path = site / "topics.json"
            topics = json.loads(path.read_text())
            topics = [t for t in topics if t["id"] != "xiaomi-common-evaluation"]
            topic = {
                "id": "xiaomi-common-evaluation",
                "title": "샤오미 · 공통 고정 씬 50개 평가",
                "category": "공통 씬 평가",
                "task": "5개 태스크",
                "model": "Xiaomi-Robotics-1",
                "environment": "Spark2 · RoboCasa · PandaOmron",
                "date": "2026-10-07",
                "tags": ["샤오미", "고정 씬", "50개 평가", "20Hz"],
                "description": (
                    "PrepareCoffee, PanTransfer, OpenCabinet, "
                    "PickPlaceSinkToCounter, StirVegetables. "
                    "각 10개 동일 씬에서 전체 native 태스크를 한 번씩 평가합니다."
                ),
                "run_ids": [r["id"] for r in records],
            }
            atomic_json(path, [topic, *topics])
        if state["complete"]:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
