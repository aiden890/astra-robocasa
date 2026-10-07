"""Run one immutable Astra medium depth condition in its own process."""

import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

from robocasa_astra.checkpoint import atomic_json

from robocasa_common.evaluate import evaluate_scene


def main():
    """Separate process environments prevent cross-condition input leakage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--condition", choices=["rgb", "color", "pixel", "grid"], required=True)
    parser.add_argument("--container", required=True)
    args = parser.parse_args()
    root = Path(args.runtime)
    lock_path = root / "trial-locks" / (args.condition + "-" + args.scene + ".lock")
    lock_path.parent.mkdir(exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = root / "results" / args.condition
    os.environ["ASTRA_DEPTH_CONDITION"] = args.condition
    os.environ["ASTRA_DEPTH_MEMORY_ONLY"] = "1"
    os.environ["ASTRA_CODEX_HOME"] = "/home/aiden/Desktop/lab/robot/astra-robocasa/.runtime/auth"
    os.environ["ASTRA_CODEX_EXECUTABLE"] = (
        "/home/aiden/Desktop/lab/robot/astra-robocasa/.runtime/codex/node_modules/"
        "@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex"
    )
    settings = SimpleNamespace(
        output=str(output),
        host="spark2",
        container=args.container,
        mounted_root="/scene-bundles",
        policy="robocasa_common.depth_policy:create_policy",
        compact_video=True,
        remote_verification=True,
        depth=True,
    )
    try:
        result = evaluate_scene(root / "scene-metadata/scenes" / args.scene, settings)
    except Exception as error:
        result_path = output / args.scene / "result.json"
        existing = json.loads(result_path.read_text()) if result_path.exists() else {}
        if existing.get("execution_status") == "success" and isinstance(
            existing.get("task_success"), bool
        ):
            atomic_json(output / args.scene / "postprocessing-error.json", {"error": str(error)})
            print(json.dumps(existing), flush=True)
            return
        result = {"execution_status": "error", "task_success": None, "error": str(error)}
        atomic_json(output / args.scene / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
