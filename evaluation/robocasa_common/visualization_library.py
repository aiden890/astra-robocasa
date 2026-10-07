"""Publish searchable playback and evidence-aligned model output without changing runs."""

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
from robocasa_astra.checkpoint import atomic_json


def backfill_timeline(folder):
    """Recover legacy observation IDs and verify movement against native action logs."""
    folder = Path(folder)
    actions = {}
    for file in folder.glob("eval/actions/*/*.jsonl"):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if "t" in row and "action" in row:
                if row["t"] in actions:
                    raise ValueError("Ambiguous legacy action logs")
                actions[row["t"]] = row["action"]
    if not actions:
        raise ValueError("Native action evidence is required for legacy synchronization")
    calls, verified = [], 0
    for call in sorted((folder / "policy").glob("call-*")):
        response, receipt, prompt = (
            call / "response.json",
            call / "receipt.json",
            call / "prompt.txt",
        )
        if not all(p.exists() for p in (response, receipt, prompt)):
            continue
        text = prompt.read_text()
        match = re.search(r"observation_id=([^\s]+):step-(\d+)\.", text)
        if not match:
            raise ValueError("Legacy observation step is not recorded explicitly")
        step = int(match[2])
        value, record = json.loads(response.read_text()), json.loads(receipt.read_text())
        if not value.get("queries"):
            repeat = value.get("repeat")
            if type(repeat) is not int or not 1 <= repeat <= 8:
                raise ValueError("Invalid legacy action chunk")
            for t in range(step, min(step + repeat, max(actions) + 1)):
                if t not in actions or not np.allclose(actions[t], value["action"], atol=1e-6):
                    raise ValueError(f"Legacy output diverges from native action at step {t}")
                verified += 1
        answers = []
        if "\nDistance query answers: " in text:
            answers = json.loads(text.split("\nDistance query answers: ", 1)[1])
        if value.get("queries") and (call / "query-results.json").exists():
            answers = json.loads((call / "query-results.json").read_text())
        calls.append(
            {
                "call": call.name,
                "step": step,
                "condition": folder.parent.name,
                "response": value,
                "usage": record.get("usage"),
                "cli_seconds": record.get("cli_end_to_end_seconds"),
                "query_answers": answers,
            }
        )
    if verified != len(actions):
        raise ValueError("Legacy output does not cover every recorded native action")
    return {
        "control_hz": 20,
        "calls": calls,
        "native_actions_verified": verified,
        "timeline_note": "원본 관측 ID와 native 행동 로그를 대조한 기록 · 20Hz 재생 위치 기준",
    }


def publish_library(runtime, site_root, frontend_root=None):
    """Expose only public explanations and usage; retain the original video and results."""
    runtime, site = Path(runtime), Path(site_root)
    media = site / "media"
    index_path = media / "model-output-index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    preserved = []
    folders = list(runtime.glob("results/*/*")) + list(runtime.glob("exclusions/*/results/*/*"))
    for folder in folders:
        condition, scene = folder.parent.name, folder.name
        identity = "astra-depth-" + condition + "-" + scene
        public_video = media / "astra-depth-medium-20261007" / (identity + ".mp4")
        if not public_video.exists():
            continue
        target = media / "model-output" / identity
        target.mkdir(parents=True, exist_ok=True)
        if (folder / "visualization.json").exists():
            shutil.copyfile(folder / "visualization.json", target / "visualization.json")
        elif not (target / "visualization.json").exists():
            atomic_json(target / "visualization.json", backfill_timeline(folder))
        index[identity] = str((target / "visualization.json").relative_to(site))
        if "exclusions" in folder.parts:
            result = json.loads((folder / "result.json").read_text())
            if result.get("execution_status") != "success":
                continue
            preserved.append(
                {
                    "id": identity,
                    "task": scene.rsplit("-", 1)[0],
                    "scene_id": scene,
                    "condition": condition,
                    "model": "gpt-6-astra",
                    "robot": "PandaOmron",
                    "status": "complete",
                    "task_success": result.get("task_success"),
                    "steps": result.get("native_steps")
                    or json.loads((target / "visualization.json").read_text()).get(
                        "native_actions_verified"
                    ),
                    "fps": 20,
                    "video": str(public_video.relative_to(site)),
                    "poster": str(public_video.with_suffix(".jpg").relative_to(site)),
                    "visualization_data": index[identity],
                    "excluded_from_current_target": True,
                }
            )
    atomic_json(index_path, index)
    atomic_json(media / "model-output-catalog.json", preserved)
    if frontend_root:
        frontend = Path(frontend_root)
        for name in ("visualizer.html", "assets/visualizer.css", "assets/visualizer.js"):
            shutil.copyfile(frontend / name, site / name)
    return {"model_output_videos": len(index), "preserved_excluded_videos": len(preserved)}


def main():
    """Publish the read-only visualization panel without launching any evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--site-root", required=True)
    parser.add_argument("--frontend-root")
    args = parser.parse_args()
    print(json.dumps(publish_library(args.runtime, args.site_root, args.frontend_root)))


if __name__ == "__main__":
    main()
