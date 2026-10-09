"""Experiment options, immutable provenance and configuration adoption on resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..astra_robodawn.codex import MODEL
from ..astra_robodawn.loop import EpisodeConfig
from .action_format import CHUNK
from .depth_input import CONDITIONS, instructions, response_schema
from .motion_control import PROFILES, adapt_prompt
from .motion_control import response_schema as motion_schema
from .persistence import atomic_json, read_json
from .prompts import PROFILE_PATH, demo_parts, load_profile, parts_text, system_prompt, vla_demos

REPO = Path(__file__).resolve().parents[5]
LEDGER = REPO / "runs" / "astra_ledger_vla.jsonl"
SCENE_INDEX = Path(__file__).resolve().parents[1] / "astra_robodawn" / "assets" / "scenes.json"
TASKS = ("OpenCabinet", "PickPlaceSinkToCounter", "PrepareCoffee", "PanTransfer", "StirVegetables")


def source_manifest() -> dict:
    """Record source hashes and reject a mixed-version continuation."""
    root = Path(__file__).resolve().parents[1]
    paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".json", ".yaml", ".png")
    ]
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)
    }


def prepare(argv: list[str] | None = None):
    """Create a fresh manifest or adopt an unfinished run with matching source and policy mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--scene", type=int, choices=range(10))
    parser.add_argument("--scene-root", default=os.environ.get("ASTRA_SCENE_ROOT"))
    parser.add_argument("--shots", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--primer",
        action="store_true",
        help="include the action primer independently of successful shots",
    )
    parser.add_argument("--condition", choices=CONDITIONS, default="rgb")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument(
        "--context",
        choices=("full", "none"),
        default="full",
        help="none omits prior feedback, recent turns and accumulated notes from model inputs",
    )
    parser.add_argument("--motion-control", choices=("dual", "legacy"), default="dual")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="adopt/recover this existing unfinished run")
    parser.add_argument("--scripted", type=Path)
    parser.add_argument("--allow-astra", action="store_true")
    parser.add_argument("--max-turns", type=int, help="dry runs only")
    parser.add_argument("--attempt-budget", type=int, default=3000)
    parser.add_argument("--python", default=os.environ.get("ASTRA_PYTHON", sys.executable))
    parser.add_argument("--codex", default=os.environ.get("ASTRA_CODEX", "codex"))
    parser.add_argument(
        "--codex-home", default=os.environ.get("ASTRA_CODEX_HOME", str(REPO / ".runtime" / "auth"))
    )
    args = parser.parse_args(argv)
    if bool(args.scripted) == bool(args.allow_astra):
        parser.error("choose exactly one of --scripted or --allow-astra")
    if args.allow_astra and args.max_turns:
        parser.error("--max-turns is for dry runs")
    if args.attempt_budget < 1 or (args.budget is not None and args.budget < 1):
        parser.error("budgets must be positive")
    if args.resume and args.scripted:
        parser.error("scripted continuation is unsupported; use a new dry run")
    if args.resume and args.output:
        parser.error("--resume and --output are mutually exclusive")
    if args.resume:
        supplied = argv if argv is not None else sys.argv[1:]
        immutable_flags = (
            "--task",
            "--scene",
            "--scene-root",
            "--shots",
            "--primer",
            "--condition",
            "--effort",
            "--context",
            "--motion-control",
            "--budget",
            "--max-turns",
            "--attempt-budget",
        )
        if any(item.split("=", 1)[0] in immutable_flags for item in supplied):
            parser.error("resume uses stored settings; omit configuration overrides")
        run_dir = args.resume.resolve()
        config = read_json(run_dir / "config.json")
        if not config or read_json(run_dir / "worker" / "closed.json") is not None:
            parser.error("resume requires an unfinished configured run")
        if config["source_manifest"] != source_manifest():
            parser.error("source changed; use the recorded commit for exact continuation")
        if config["mode"] != ("astra" if args.allow_astra else "scripted"):
            parser.error("resume cannot change the policy mode")
    else:
        if args.task is None or args.scene is None or not args.scene_root:
            parser.error("new runs require --task, --scene and --scene-root")
        scene = json.loads(SCENE_INDEX.read_text())["tasks"][args.task][args.scene]
        budget = args.budget or scene["horizon"]
        run_dir = (
            args.output
            or REPO
            / "runs"
            / "vla12"
            / f"{args.task}-scene{args.scene}-{args.condition}-{time.time_ns()}"
        ).resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
        cfg = EpisodeConfig(
            task=args.task,
            seed=scene["simulator_seed"],
            shots=args.shots,
            budget=budget,
            max_turns=args.max_turns or max(args.attempt_budget, (budget + CHUNK - 1) // CHUNK),
        )
        demos = vla_demos(args.task, args.shots, primer=args.primer)
        block = demo_parts(demos)
        prompt = system_prompt(load_profile(), args.task, bool(block), context=args.context)
        if args.motion_control == "dual":
            prompt = adapt_prompt(prompt)
        depth_note = instructions(args.condition)
        if args.motion_control == "dual":
            depth_note = depth_note.replace(
                "Exactly one field must be non-null.",
                "For precision set actions non-null and transit_target=null; "
                "for transit set actions=null, queries=null and transit_target non-null; "
                "for queries set actions=null, motion_mode=null and transit_target=null.",
            ).replace("then return actions.", "then return a precision action or transit target.")
        (run_dir / "system_prompt.md").write_text(prompt + "\n\n" + depth_note)
        atomic_json(
            run_dir / "response_schema.json",
            motion_schema(response_schema(args.condition), args.motion_control),
        )
        atomic_json(run_dir / "demo_block.json", block)
        (run_dir / "demo_block.txt").write_text(parts_text(block))
        config = {
            "variant": "precision-vla12-transit-world-target-recovery",
            "motion_control": args.motion_control,
            "motion_profiles": PROFILES if args.motion_control == "dual" else None,
            "transport": json.loads(os.environ["ASTRA_SPARK_TRANSPORT"])
            if os.environ.get("ASTRA_SPARK_TRANSPORT")
            else None,
            "task": args.task,
            "seed": cfg.seed,
            "scene": args.scene,
            "scene_info": scene,
            "scene_dir": str(Path(args.scene_root) / scene["folder"]),
            "shots": args.shots,
            "primer": args.primer,
            "condition": args.condition,
            "context": args.context,
            "budget": budget,
            "protocol_horizon": scene["horizon"],
            "max_turns": cfg.max_turns,
            "mode": "astra" if args.allow_astra else "scripted",
            "model": MODEL,
            "reasoning_effort": args.effort,
            "attempt_budget": args.attempt_budget,
            "model_response_timeout_s": None,
            "wall_timeout_s": None,
            "git_commit": subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_manifest": source_manifest(),
            "started_at": time.time(),
            "argv": sys.argv,
            "profile_sha256": hashlib.sha256(PROFILE_PATH.read_bytes()).hexdigest(),
            "demos": [
                {
                    "name": d.name,
                    "sha256": hashlib.sha256((d.folder / "demo.json").read_bytes()).hexdigest(),
                }
                for d in demos
            ],
        }
        atomic_json(run_dir / "config.json", config)
    return args, run_dir, config
