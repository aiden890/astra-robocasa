"""Resume the fixed fifty-scene Xiaomi evaluation without repeating completed episodes."""

import argparse
import fcntl
import html
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from robocasa_common.evaluate import evaluate_scene


def atomic_json(path, value):
    """Replace progress only after serializing the complete next state."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    """Adopt existing results; distinguish task failure from execution errors."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--site-root", required=True)
    parser.add_argument("--host", default="spark2")
    args = parser.parse_args()
    runtime = Path(args.runtime).resolve()
    lock = (runtime / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (runtime / "queue.pid").write_text(str(os.getpid()))
    scenes = json.loads((runtime / "scene-metadata/catalog.json").read_text())["scenes"]
    if len(scenes) != 50 or len({r["id"] for r in scenes}) != 50:
        raise ValueError("Expected fifty distinct fixed scenes")
    tasks = [
        "PrepareCoffee",
        "PanTransfer",
        "OpenCabinet",
        "PickPlaceSinkToCounter",
        "StirVegetables",
    ]
    groups = {task: [r for r in scenes if r["task"] == task] for task in tasks}
    if any(len(g) != 10 for g in groups.values()):
        raise ValueError("Expected ten fixed scenes per task")
    ordered = [groups[task][index] for index in range(10) for task in tasks]
    output = runtime / "results"
    public = Path(args.site_root) / "media/xiaomi-common-scenes-20261007"
    public.mkdir(parents=True, exist_ok=True)
    os.environ["XIAOMI_COMPACT_VIDEO"] = "1"
    os.environ["XIAOMI_CLIENT_COMMAND"] = json.dumps(
        [
            "ssh",
            args.host,
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
        output=str(output),
        host=args.host,
        container="xiaomi-common-scenes-sim-20261007",
        mounted_root="/scene-bundles",
        policy="robocasa_common.xiaomi:create_policy",
        compact_video=True,
        remote_verification=True,
    )
    started = time.time()

    def refresh(active=None):
        rows = []
        for scene in ordered:
            folder = output / scene["id"]
            path = folder / "result.json"
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            result.update(task=scene["task"], horizon=scene["horizon"])
            calls_path = folder / "policy/calls.jsonl"
            calls = (
                [json.loads(line) for line in calls_path.read_text().splitlines()]
                if calls_path.exists()
                else []
            )
            logs = list((folder / "eval").glob("*.json"))
            if logs:
                log = json.loads(logs[0].read_text())
                result["steps"] = log["stats"]["total_steps"]
                result["seconds"] = log["stats"]["duration_s"]
            result["model_calls"] = len(calls)
            if calls:
                result["mean_inference_round_trip_seconds"] = sum(
                    c["round_trip_seconds"] for c in calls
                ) / len(calls)
            video = folder / "video.mp4"
            # Publish full normal evaluations only. Setup-error previews remain private.
            if result["execution_status"] == "success" and video.exists():
                import imageio_ffmpeg

                reader = imageio_ffmpeg.read_frames(str(video))
                try:
                    metadata = next(reader)
                finally:
                    reader.close()
                if metadata["fps"] != 20 or tuple(metadata["size"]) != (768, 256):
                    raise ValueError("Completed video violates the 20fps 768x256 contract")
                destination = public / (scene["id"] + ".mp4")
                if not destination.exists():
                    shutil.copyfile(video, destination)
                result["video"] = destination.name
                result["fps"] = 20
                request = urllib.request.Request(
                    "http://100.86.183.64:8906/media/xiaomi-common-scenes-20261007/"
                    + destination.name,
                    method="HEAD",
                )
                with urllib.request.urlopen(request, timeout=15) as response:
                    if (
                        response.status != 200
                        or int(response.headers["Content-Length"]) != video.stat().st_size
                    ):
                        raise ValueError("Published video HEAD/size verification failed")
                atomic_json(public / (scene["id"] + ".json"), result)
            rows.append(result)
        summary = {}
        for task in tasks:
            selected = [r for r in rows if r["task"] == task]
            normal = [r for r in selected if r["execution_status"] == "success"]
            summary[task] = {
                "expected": 10,
                "finished": len(selected),
                "evaluated": len(normal),
                "execution_errors": len(selected) - len(normal),
                "task_successes": sum(r["task_success"] is True for r in normal),
                "success_rate": sum(r["task_success"] is True for r in normal) / 10
                if len(normal) == 10
                else None,
            }
        state = {
            "checkpoint": "XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa365",
            "expected": 50,
            "finished": len(rows),
            "active": active,
            "complete": len(rows) == 50,
            "elapsed_queue_seconds": time.time() - started,
            "tasks": summary,
            "results": rows,
            "protocol": "PandaOmron, fixed scene v2, native success, 20Hz, one rollout per scene",
        }
        atomic_json(runtime / "status.json", state)
        atomic_json(public / "status.json", state)
        blocks = []
        for task in tasks:
            group = [r for r in rows if r["task"] == task]
            clips = "".join(
                f"<article><h3>{html.escape(r['scene'])}</h3>"
                f"<p>{'성공' if r['task_success'] else '태스크 미성공'} · "
                f"{r.get('steps', 0)}스텝 · 20fps</p>"
                '<video controls preload="metadata" '
                f'src="{html.escape(r["video"])}"></video></article>'
                for r in group
                if r.get("video")
            )
            blocks.append(f"<section><h2>{task} ({len(group)}/10)</h2>{clips}</section>")
        page = (
            '<!doctype html><html lang="ko"><meta charset="utf-8">'
            "<title>샤오미 공통 씬 평가</title><style>"
            "body{max-width:1100px;margin:40px auto;font-family:sans-serif;background:#fafafa}"
            "video{width:100%;max-width:768px}"
            "article{padding:16px;background:white;margin:20px 0;border:1px solid #ddd}"
            "a{color:#2455aa}</style><h1>샤오미 공통 씬 평가</h1>"
        )
        page += (
            f"<p>완료 {len(rows)}/50 · 실행 중: {html.escape(active or '없음')}.</p>"
            "<p>각 씬의 전체 실행 결과이며 초기화 오류는 성공률에 넣지 않습니다.</p>"
            '<a href="/videos.html">영상 게시판</a>'
            '<p><a href="status.json">전체 결과와 진행 정보</a></p>' + "".join(blocks)
        )
        temporary = public / "index.html.tmp"
        temporary.write_text(page)
        temporary.replace(public / "index.html")

    for scene in ordered:
        folder = output / scene["id"]
        if (folder / "result.json").exists():
            continue
        refresh(scene["id"])
        try:
            if folder.exists():
                raise RuntimeError("Interrupted directory preserved; episode will not be repeated")
            result = evaluate_scene(runtime / "scene-metadata/scenes" / scene["id"], run_args)
        except Exception as error:
            folder.mkdir(parents=True, exist_ok=True)
            result = {
                "scene": scene["id"],
                "execution_status": "error",
                "task_success": None,
                "error": repr(error),
            }
            atomic_json(folder / "result.json", result)
        print(json.dumps(result), flush=True)
        refresh()
    refresh()


if __name__ == "__main__":
    main()
