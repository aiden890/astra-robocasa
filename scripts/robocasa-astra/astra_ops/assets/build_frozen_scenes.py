"""Build immutable Panda scene bundles and check exact restored state and camera frames."""

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

from robocasa_astra.frozen_scene import export_scene, image_hashes, restore_scene, sha256
from robocasa_astra.worker import Simulator

from inspect_robots.rollout import derive_seed


def build_one(arguments):
    """Export and independently initialize a second environment before admitting a scene."""
    task, seed, root, layout_id, style_id = arguments
    folder = Path(root) / "scenes" / f"{task}-{seed}"
    if (folder / "verified.json").exists():
        return json.loads((folder / "verified.json").read_text())
    if folder.exists() and not (folder / "manifest.json").exists():
        raise ValueError("Incomplete scene preserved; use a new attempt directory")
    started = time.time()
    actual_seed = derive_seed(0, seed, 0)
    horizon = 2400 if task == "StirVegetables" else 1800
    if not folder.exists():
        source = Simulator(
            "PandaOmron", task, horizon=horizon, layout_id=layout_id, style_id=style_id
        )
        try:
            source.reset(actual_seed)
            folder = export_scene(source, root, actual_seed, seed)
        finally:
            if source.env:
                source.env.close()
    target = Simulator("PandaOmron", task, horizon=horizon, layout_id=layout_id, style_id=style_id)
    try:
        target.reset(actual_seed)
        manifest = restore_scene(target, folder, actual_seed)
        observation = target.observe()
        matches = image_hashes(observation) == manifest["initial_images"]
        meta = json.loads((folder / "episode-meta.json").read_text())
        if layout_id is not None and (meta["layout_id"], meta["style_id"]) != (layout_id, style_id):
            raise ValueError("Exported scene differs from requested layout/style")
        asset_hashes = sorted(
            {value for name, value in manifest["files"].items() if name.startswith("../../assets/")}
        )
        asset_fingerprint = hashlib.sha256(json.dumps(asset_hashes).encode()).hexdigest()
        result = {
            "layout_id": meta["layout_id"],
            "style_id": meta["style_id"],
            "asset_fingerprint": asset_fingerprint,
            "task": task,
            "rollout_seed": seed,
            "simulator_seed": actual_seed,
            "robot": "PandaOmron",
            "state_exact": True,
            "initial_images_exact": matches,
            "world_geometry_exact": True,
            "manifest_sha256": sha256(folder / "manifest.json"),
            "seconds": time.time() - started,
        }
        (folder / "verified.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        if target.env:
            target.env.close()


def main():
    """Resume verified episodes without rerolling or overwriting incomplete attempts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--representative", action="store_true")
    parser.add_argument("--diverse", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(args.seeds).read_text())
    jobs = [
        (
            task,
            seed,
            str(root),
            index + 1 if args.diverse else None,
            index + 1 if args.diverse else None,
        )
        for task, seeds in data["tasks"].items()
        for index, seed in enumerate(seeds[:1] if args.representative else seeds)
    ]
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(build_one, jobs):
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.diverse:
        if len({row["asset_fingerprint"] for row in results}) != len(results):
            raise ValueError("Duplicate asset composition across collection")
        for task in data["tasks"]:
            rows = [row for row in results if row["task"] == task]
            if len({(row["layout_id"], row["style_id"]) for row in rows}) != len(rows):
                raise ValueError("Duplicate layout/style within task")
            if len({row["asset_fingerprint"] for row in rows}) != len(rows):
                raise ValueError("Duplicate asset composition within task")
    summary = {
        "diverse": args.diverse,
        "scenes": results,
        "episodes": len(jobs),
        "robot": "PandaOmron",
        "verified": True,
        "representative": args.representative,
    }
    (root / ("representative.json" if args.representative else "complete.json")).write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
