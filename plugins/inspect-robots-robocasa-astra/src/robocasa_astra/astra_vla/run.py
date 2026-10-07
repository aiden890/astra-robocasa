"""Run one VLA-output (12-D action chunk) episode and write a reviewable run directory under ``runs/vla12/``.

    python -m robocasa_astra.astra_vla.run --task OpenCabinet --shots 1 --scene 0 --scripted replies.json
    python -m robocasa_astra.astra_vla.run --task OpenCabinet --shots 1 --scene 0 --allow-astra

Same scenes, step budgets, caller and recording as ``astra_robodawn.run`` (the skill variant); the turn cap is
budget // 16 because every turn executes exactly one 16-step chunk. Real model runs need ``--allow-astra`` and
are counted in ``runs/astra_ledger_vla.jsonl`` (separate from the skill ledger) up to ``VLA_RUN_LIMIT``.

Run directory: as in the skill variant (config.json, system_prompt.md, demo_block.json/.txt, trace.jsonl,
calls/, turns/, replay/, video.mp4, summary.json ...); trace records also carry the 16-D dataset state and the
full chunk in ``response.actions``.
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

from ..astra_robodawn.appserver import AppServerCaller
from ..astra_robodawn.codex import MODEL, REASONING_EFFORT
from ..astra_robodawn.loop import EpisodeConfig, partial_summary
from ..astra_robodawn.scripted import ScriptedCaller
from .action_format import CHUNK
from .loop import run_episode
from .prompts import PROFILE_PATH, RESPONSE_SCHEMA, demo_parts, load_profile, parts_text, system_prompt, vla_demos
from .sim_client import ChunkSimClient

REPO = Path(__file__).resolve().parents[5]
LEDGER = REPO / "runs" / "astra_ledger_vla.jsonl"
# V1: OpenCabinet scene 0 (stopped at turn 11: shared weekly meter at 91%); after the meter reset (agreed
# 2026-10-07): V2: OpenCabinet scene 0 again, V3: PrepareCoffee scene 0, one run each.
VLA_RUN_LIMIT = 3
SCENE_INDEX = Path(__file__).resolve().parents[1] / "astra_robodawn" / "assets" / "scenes.json"
TASKS = ("OpenCabinet", "PickPlaceSinkToCounter", "PrepareCoffee", "PanTransfer", "StirVegetables")


def _md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _ledger_count() -> int:
    return len(LEDGER.read_text().splitlines()) if LEDGER.exists() else 0


def _weekly(caller, run_dir: Path) -> dict | None:
    """Subscription weekly-usage snapshot (app-server caller only; no model call); None if unavailable."""
    if not hasattr(caller, "weekly_snapshot"):
        return None
    try:
        return caller.weekly_snapshot(run_dir / "appserver.stderr.log")
    except Exception as exc:  # noqa: BLE001 - informational only, never stops a run
        print(f"weekly usage snapshot failed: {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--shots", type=int, choices=[0, 1], required=True)
    parser.add_argument("--scene", type=int, required=True, help="common frozen scene number (0-9)")
    parser.add_argument("--scene-root", default=os.environ.get("ASTRA_SCENE_ROOT"))
    parser.add_argument("--budget", type=int, help="native step budget instead of the protocol horizon")
    parser.add_argument("--output", help="new run directory (default runs/vla12/<task>-scene<N>-<shots>shot-<time>)")
    parser.add_argument("--scripted", help="JSON list of replies to use instead of the model (no model call)")
    parser.add_argument("--allow-astra", action="store_true", help="really call gpt-6-astra (counted in the VLA ledger)")
    parser.add_argument("--max-turns", type=int, help="override the turn cap (dry runs only)")
    parser.add_argument("--python", default=os.environ.get("ASTRA_PYTHON", sys.executable))
    parser.add_argument("--codex", default=os.environ.get("ASTRA_CODEX", "codex"))
    parser.add_argument("--codex-home", default=os.environ.get("ASTRA_CODEX_HOME", str(REPO / ".runtime" / "auth")))
    args = parser.parse_args()

    if bool(args.scripted) == bool(args.allow_astra):
        parser.error("choose exactly one of --scripted (dry run) or --allow-astra (real model)")
    if args.allow_astra and args.max_turns:
        parser.error("--max-turns is for dry runs; real runs use budget // 16")
    if args.allow_astra and _ledger_count() >= VLA_RUN_LIMIT:
        parser.error(f"{LEDGER} already lists {VLA_RUN_LIMIT} real runs; the agreed batch is used up")
    if not args.scene_root:
        parser.error("--scene needs --scene-root or ASTRA_SCENE_ROOT")

    scene = json.loads(SCENE_INDEX.read_text())["tasks"][args.task][args.scene]
    scene_dir = str(Path(args.scene_root) / scene["folder"])
    budget = args.budget or scene["horizon"]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output or REPO / "runs" / "vla12" / f"{args.task}-scene{args.scene}-{args.shots}shot-{stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    cfg = EpisodeConfig(task=args.task, seed=scene["simulator_seed"], shots=args.shots, budget=budget,
                        max_turns=args.max_turns or budget // CHUNK)

    demos = vla_demos(args.task, args.shots)
    block = demo_parts(demos)
    (run_dir / "system_prompt.md").write_text(system_prompt(load_profile(), args.task, shown_demos=bool(block)))
    (run_dir / "response_schema.json").write_text(json.dumps(RESPONSE_SCHEMA, indent=1))
    (run_dir / "demo_block.json").write_text(json.dumps(block, indent=1))
    (run_dir / "demo_block.txt").write_text(parts_text(block))
    codex_version = subprocess.run([args.codex, "--version"], capture_output=True, text=True).stdout.strip() \
        if args.allow_astra else "not used (scripted)"
    config = {
        "variant": "vla12", "output_format": f"one chunk of {CHUNK} x 12-D RoboCasa365 actions per call (open loop)",
        "task": args.task, "seed": cfg.seed, "scene": args.scene, "scene_info": scene, "scene_dir": scene_dir,
        "shots": args.shots, "protocol_horizon": scene["horizon"], "budget": budget, "max_turns": cfg.max_turns,
        "budget_deviates_from_protocol": budget != scene["horizon"],
        "mode": "astra" if args.allow_astra else "scripted", "model": MODEL if args.allow_astra else "scripted",
        "reasoning_effort": REASONING_EFFORT, "codex_version": codex_version, "git_commit": _git_commit(),
        "caller": "appserver" if args.allow_astra else "scripted",
        "profile": {"path": str(PROFILE_PATH), "md5": _md5(PROFILE_PATH)},
        "demos": [{"name": d.name, "md5": _md5(d.folder / "demo.json")} for d in demos],
        "demo_block": {"parts": len(block), "images": sum(p["type"] == "image" for p in block),
                       "files": ["demo_block.json", "demo_block.txt"]},
        "argv": sys.argv, "started": stamp,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=1))

    if args.allow_astra:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a") as ledger:
            ledger.write(json.dumps({"started": stamp, "variant": "vla12", "task": args.task, "shots": args.shots,
                                     "seed": cfg.seed, "scene": args.scene, "budget": budget,
                                     "run_dir": str(run_dir)}) + "\n")
        caller = AppServerCaller(args.codex, args.codex_home, run_dir / "system_prompt.md",
                                 run_dir / "response_schema.json", REPO / ".runtime" / "inference-empty")
    else:
        caller = ScriptedCaller(json.loads(Path(args.scripted).read_text()))

    sim = ChunkSimClient(args.task, budget, run_dir, python=args.python, scene_dir=scene_dir)
    summary: dict = {}
    weekly = {"start": _weekly(caller, run_dir)}
    try:
        summary = run_episode(sim, caller, cfg, run_dir, block)
    finally:
        weekly["end"] = _weekly(caller, run_dir)
        if hasattr(caller, "close"):
            caller.close()
        closing = sim.close()
        if "task" not in summary:  # stopped early (e.g. Ctrl-C): rebuild what the trace recorded
            summary = partial_summary(run_dir, cfg, "interrupted", closing.get("steps_used", 0),
                                      bool(closing.get("task_success")))
        summary["simulator"] = closing
        if weekly["start"] and weekly["end"]:
            weekly["delta_percent"] = weekly["end"]["used_percent"] - weekly["start"]["used_percent"]
            weekly["note"] = "the meter has 1% resolution; other use of the same account in this window also counts"
        summary["weekly_usage"] = weekly
        summary["variant"] = "vla12"
        summary["run_dir"] = str(run_dir)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary.get(k) for k in ("task", "success", "finished_reason", "turns", "steps_used",
                                                   "usage", "cost", "run_dir")}, indent=1))


if __name__ == "__main__":
    main()
