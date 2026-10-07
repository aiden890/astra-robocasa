"""Run one astra_robodawn episode and write a self-contained, reviewable run directory.

    python -m robocasa_astra.astra_robodawn.run --task OpenCabinet --shots 0 --scripted responses.json
    python -m robocasa_astra.astra_robodawn.run --task OpenCabinet --shots 0 --allow-astra

Real model runs need ``--allow-astra`` and are counted in ``runs/astra_ledger.jsonl``; the protocol
allows ``ASTRA_RUN_LIMIT`` real runs in total (P4: 1, P6: 5) and further runs are refused.

Run directory: config.json, system_prompt.md, demo_block.json/.txt (the demonstration block that starts
every request), response_schema.json, episode_start.json, trace.jsonl,
calls/turnNNN/ (every model call, raw), turns/ (images the model saw), memory.json, summary.json,
video.mp4, replay/ (seed, initial state, model XML, native actions; exact replay), sim.log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from .appserver import AppServerCaller
from .codex import MODEL, REASONING_EFFORT, CodexCaller
from .loop import EpisodeConfig, run_episode
from .prompts import PROFILE_PATH, RESPONSE_SCHEMA, demo_parts, demos_for, load_profile, parts_text, system_prompt
from .scripted import ScriptedCaller
from .sim_client import SimClient

REPO = Path(__file__).resolve().parents[5]
LEDGER = REPO / "runs" / "astra_ledger.jsonl"
# P4: 1 + P6: 5 (scene-less, layout 1 / style 1) + P7: 5 (common scene 0 with success conditions)
# + P8: 1 (OpenCabinet scene 0 with a 3600-step budget; stopped at turn 17, budget option since removed)
# + P9: 1 (OpenCabinet scene 0 with the "open to the stop, do not touch an opened door" success wording)
# + P10: 1 (PrepareCoffee scene 0, RoboDawn-style interleaved few-shot, "a coffee stream shows the button worked").
ASTRA_RUN_LIMIT = 14
SCENE_INDEX = Path(__file__).resolve().parent / "assets" / "scenes.json"
STEPS_PER_TURN_CAP = 40  # max turns = horizon // 40 for common scenes (45 for 1800 steps, 60 for 2400)

# Native step budgets (RoboCasa dataset_registry horizons) and turn caps per task.
TASKS = {
    "OpenCabinet": {"budget": 1050, "max_turns": 30},
    "PickPlaceSinkToCounter": {"budget": 900, "max_turns": 30},
    "PrepareCoffee": {"budget": 1800, "max_turns": 45},
    "PanTransfer": {"budget": 1800, "max_turns": 45},
    "StirVegetables": {"budget": 2400, "max_turns": 60},
}


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _ledger_count() -> int:
    return len(LEDGER.read_text().splitlines()) if LEDGER.exists() else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--shots", type=int, choices=[0, 1], required=True)
    parser.add_argument("--seed", type=int, default=771001, help="legacy native-scene seed (ignored with --scene)")
    parser.add_argument("--scene", type=int, help="common frozen scene number (0-9) from assets/scenes.json")
    parser.add_argument("--scene-root", default=os.environ.get("ASTRA_SCENE_ROOT"),
                        help="extracted common scene collection (default $ASTRA_SCENE_ROOT)")
    parser.add_argument("--output", help="new run directory (default runs/robodawn/<task>-<shots>shot-s<seed>-<time>)")
    parser.add_argument("--scripted", help="JSON list of replies to use instead of the model (no model call)")
    parser.add_argument("--allow-astra", action="store_true", help="really call gpt-6-astra (counted in the ledger)")
    parser.add_argument("--max-turns", type=int, help="override the per-task turn cap (dry runs only)")
    parser.add_argument("--budget", type=int, help="native step budget instead of the protocol horizon (recorded in "
                        "config.json as a deviation; max turns become budget // 40)")
    parser.add_argument("--caller", choices=["appserver", "exec"], default="appserver",
                        help="appserver: demonstration image+text interleaved (RoboDawn format, default); "
                             "exec: codex exec, all images before the text (runs P4-P9)")
    parser.add_argument("--python", default=os.environ.get("ASTRA_PYTHON", sys.executable))
    parser.add_argument("--codex", default=os.environ.get("ASTRA_CODEX", "codex"))
    parser.add_argument("--codex-home", default=os.environ.get("ASTRA_CODEX_HOME", str(REPO / ".runtime" / "auth")))
    args = parser.parse_args()

    if bool(args.scripted) == bool(args.allow_astra):
        parser.error("choose exactly one of --scripted (dry run) or --allow-astra (real model)")
    if args.allow_astra and args.max_turns:
        parser.error("--max-turns is for dry runs; real runs use the protocol's turn caps")
    if args.allow_astra and _ledger_count() >= ASTRA_RUN_LIMIT:
        parser.error(f"{LEDGER} already lists {ASTRA_RUN_LIMIT} real runs; the protocol allows no more")

    spec = dict(TASKS[args.task])
    scene, scene_dir, seed = None, None, args.seed
    if args.scene is not None:
        if not args.scene_root:
            parser.error("--scene needs --scene-root or ASTRA_SCENE_ROOT")
        scene = json.loads(SCENE_INDEX.read_text())["tasks"][args.task][args.scene]
        scene_dir = str(Path(args.scene_root) / scene["folder"])
        seed = scene["simulator_seed"]
        spec = {"budget": scene["horizon"], "max_turns": scene["horizon"] // STEPS_PER_TURN_CAP}
    if args.budget:
        spec = {"budget": args.budget, "max_turns": args.budget // STEPS_PER_TURN_CAP}
    stamp = time.strftime("%Y%m%d-%H%M%S")
    label = f"scene{args.scene}" if scene else f"s{args.seed}"
    # Absolute: codex runs in an empty working directory, so attached image paths must not be relative.
    run_dir = Path(args.output or REPO / "runs" / "robodawn" / f"{args.task}-{label}-{args.shots}shot-{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = EpisodeConfig(task=args.task, seed=seed, shots=args.shots, budget=spec["budget"],
                        max_turns=args.max_turns or spec["max_turns"])

    demos = demos_for(args.task, args.shots)
    demos_block = demo_parts(demos)
    prompt = system_prompt(load_profile(), args.task, shown_demos=bool(demos_block))
    (run_dir / "system_prompt.md").write_text(prompt)
    (run_dir / "demo_block.json").write_text(json.dumps(demos_block, indent=1))
    (run_dir / "demo_block.txt").write_text(parts_text(demos_block))
    (run_dir / "response_schema.json").write_text(json.dumps(RESPONSE_SCHEMA, indent=1))
    codex_version = subprocess.run([args.codex, "--version"], capture_output=True, text=True).stdout.strip() \
        if not args.scripted else "not used (scripted)"
    config = {
        "task": args.task, "seed": seed, "scene": args.scene, "scene_info": scene, "scene_dir": scene_dir,
        "shots": args.shots,
        "protocol_horizon": scene["horizon"] if scene else None,
        "budget_deviates_from_protocol": bool(args.budget and scene and args.budget != scene["horizon"]),
        "budget": cfg.budget, "max_turns": cfg.max_turns,
        "mode": "astra" if args.allow_astra else "scripted", "model": MODEL if args.allow_astra else "scripted",
        "reasoning_effort": REASONING_EFFORT, "codex_version": codex_version, "git_commit": _git_commit(),
        "profile": {"path": str(PROFILE_PATH), "md5": _md5(PROFILE_PATH)},
        "demos": [{"name": d.name, "md5": _md5(d.folder / "demo.json")} for d in demos],
        "demo_block": {"parts": len(demos_block), "images": sum(p["type"] == "image" for p in demos_block),
                       "files": ["demo_block.json", "demo_block.txt"]},
        "caller": args.caller if args.allow_astra else "scripted", "argv": sys.argv, "started": stamp,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    if args.allow_astra:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as ledger:
            ledger.write(json.dumps({"started": stamp, "task": args.task, "shots": args.shots, "seed": seed,
                                     "scene": args.scene, "budget": cfg.budget, "run_dir": str(run_dir)}) + "\n")
        kind = AppServerCaller if args.caller == "appserver" else CodexCaller
        caller = kind(args.codex, args.codex_home, run_dir / "system_prompt.md", run_dir / "response_schema.json",
                      REPO / ".runtime" / "inference-empty")
    else:
        caller = ScriptedCaller(json.loads(Path(args.scripted).read_text()))

    sim = SimClient(args.task, cfg.budget, run_dir, python=args.python, scene_dir=scene_dir)
    summary = {}
    try:
        summary = run_episode(sim, caller, cfg, run_dir, demos_block)
    finally:
        if hasattr(caller, "close"):
            caller.close()
        closing = sim.close()
        summary["simulator"] = closing
        summary["run_dir"] = str(run_dir)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("task", "success", "finished_reason", "turns", "steps_used",
                                                   "usage", "cost", "run_dir")}, indent=1))


if __name__ == "__main__":
    main()
