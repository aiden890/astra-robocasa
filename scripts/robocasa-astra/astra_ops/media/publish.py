"""Publish complete per-step recordings and compact live rollout metadata."""

import hashlib
import json
import re
import subprocess

import imageio_ffmpeg
import numpy as np
from PIL import Image

from astra_ops.common.paths import REPO_ROOT


def publish(repo):
    """Encode saved native frames once and atomically refresh the public catalog."""
    media = repo / "status-page/media"
    media.mkdir(exist_ok=True)
    records = []
    plan_path = repo / ".runtime/multitask-20261006-v1/plan.json"
    planned = json.loads(plan_path.read_text())["jobs"] if plan_path.exists() else []
    requests = {row["id"]: row for row in planned}
    for run in sorted((repo / "runs").iterdir()):
        if not run.is_dir():
            continue
        logs = list((run / "eval").glob("*.json"))
        frame_files = sorted((run / "eval/frames").glob("*/*.npy"))
        frames = {}
        for file in frame_files:
            match = re.search(r"~(robot0_.+)_(\d{6})\.npy$", file.name)
            if match:
                frames.setdefault(int(match[2]), {})[match[1]] = file
        robot = "PandaOmron" if run.name.startswith("panda") else "GR1FloatingBody"
        status = "진행 중"
        steps = max(frames, default=0)
        max_steps = 1800 if "1800" in run.name else 64
        request = requests.get(run.name, {})
        task = request.get("task", "PrepareCoffee")
        max_steps = request.get("max_steps", max_steps)
        environment = run / "environment.json"
        alignment = None
        if environment.exists():
            docs = json.loads(json.loads(environment.read_text())["docs"])
            task = docs.get("task", task)
            max_steps = docs.get("horizon", max_steps)
            alignment = docs.get("initial_alignment")
        if logs:
            log = json.loads(logs[0].read_text())
            max_steps = log["eval"]["max_steps"]
            task = log["eval"]["task"]
            status = (
                "오류"
                if log["status"] != "success"
                else ("성공" if log["results"]["metrics"].get("success_at_end") == 1 else "미성공")
            )
        elif (run / "failed.json").exists():
            status = "오류"
        cameras = ["robot0_agentview_left", "robot0_agentview_right"] + (
            ["robot0_eye_in_hand"]
            if robot == "PandaOmron"
            else ["robot0_eye_in_right_hand", "robot0_eye_in_left_hand"]
        )
        url = None
        if status != "진행 중" and steps:
            target = media / (run.name + "-20fps.mp4")
            if not target.exists():
                ordered = [frames[i] for i in range(1, steps + 1)]
                if any(set(cameras) - set(row) for row in ordered):
                    raise ValueError(f"Missing synchronized camera frame: {run.name}")
                first = np.concatenate(
                    [np.load(ordered[0][c], allow_pickle=False) for c in cameras], axis=1
                )
                height, width, _ = first.shape
                temporary = target.with_suffix(".partial.mp4")
                command = [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    f"{width}x{height}",
                    "-r",
                    "20",
                    "-i",
                    "-",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ]
                process = subprocess.Popen(command, stdin=subprocess.PIPE)
                try:
                    for row in ordered:
                        image = np.concatenate(
                            [np.load(row[c], allow_pickle=False) for c in cameras], axis=1
                        )
                        process.stdin.write(image.astype(np.uint8).tobytes())
                finally:
                    process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError("Video encoding failed")
                temporary.replace(target)
                Image.fromarray(first).save(media / (run.name + "-20fps.jpg"))
                target.with_suffix(".sha256").write_text(
                    hashlib.sha256(target.read_bytes()).hexdigest()
                )
            url = "media/" + target.name
        records.append(
            {
                "id": run.name,
                "robot": robot,
                "task": task,
                "status": status,
                "steps": steps,
                "max_steps": max_steps,
                "fps": 20,
                "duration": steps / 20,
                "video": url,
                "poster": "media/" + run.name + "-20fps.jpg" if url else None,
                "type": "매 스텝 연속 기록",
                "initial_alignment": alignment,
                "cameras": cameras,
            }
        )
    recorded = {row["id"] for row in records}
    for row in planned:
        if row["id"] not in recorded:
            records.append(
                {
                    "id": row["id"],
                    "robot": row["robot"],
                    "task": row["task"],
                    "status": "대기",
                    "steps": 0,
                    "max_steps": row["max_steps"],
                    "fps": 20,
                    "duration": 0.0,
                    "video": None,
                    "poster": None,
                    "type": "매 스텝 연속 기록",
                    "cameras": [],
                }
            )
    temporary = media / "catalog.tmp.json"
    temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    temporary.replace(media / "catalog.json")
    return records


if __name__ == "__main__":
    publish(REPO_ROOT)
