"""Launch one configured VLA episode, retaining unfinished work for --resume."""

from __future__ import annotations

import json

from ..astra_robodawn.loop import EpisodeConfig
from ..astra_robodawn.scripted import ScriptedCaller
from .caller import DurableCaller
from .experiment import LEDGER, REPO, prepare
from .loop import run_episode
from .persistence import append_json, atomic_json, locked, read_json
from .sim_client import ChunkSimClient


def main() -> None:
    """An interruption detaches; only a completed native evaluation closes the simulator."""
    args, run_dir, config = prepare()
    with locked(run_dir / "policy" / "host.lock", blocking=False):
        cfg = EpisodeConfig(
            task=config["task"],
            seed=config["seed"],
            shots=config["shots"],
            budget=config["budget"],
            max_turns=config["max_turns"],
        )
        block = read_json(run_dir / "demo_block.json")
        if args.allow_astra:
            if not args.resume:
                append_json(
                    LEDGER,
                    {
                        "started_at": config["started_at"],
                        "run_dir": str(run_dir),
                        "task": cfg.task,
                        "condition": config["condition"],
                    },
                )
            caller = DurableCaller(
                args.codex,
                args.codex_home,
                run_dir / "system_prompt.md",
                run_dir / "response_schema.json",
                REPO / ".runtime" / "inference-empty",
                run_dir,
                model=config["model"],
                effort=config["reasoning_effort"],
                attempt_budget=config["attempt_budget"],
            )
        else:
            caller = ScriptedCaller(json.loads(args.scripted.read_text()))
        if config.get("transport"):
            from .spark_transport import SparkChunkSimClient

            sim = SparkChunkSimClient(
                cfg.task,
                cfg.budget,
                run_dir,
                config["scene_dir"],
                config["condition"],
                config["transport"],
            )
        else:
            sim = ChunkSimClient(
                cfg.task,
                cfg.budget,
                run_dir,
                python=args.python,
                scene_dir=config["scene_dir"],
                condition=config["condition"],
            )
        try:
            summary = run_episode(
                sim,
                caller,
                cfg,
                run_dir,
                block,
                config["condition"],
                bool(args.resume),
                config.get("context", "full"),
                config.get("motion_control", "legacy"),
            )
        except BaseException:
            sim.detach()  # keyboard interrupt, transport error and host exit preserve the native
            raise
        else:
            # Actual execution errors remain resumable. Normal native evaluation finalizes
            # video/replay.
            if summary["task_success"] is not None:
                summary["simulator"] = sim.close()
            else:
                sim.detach()
            summary["run_dir"] = str(run_dir)
            atomic_json(run_dir / "summary.json", summary)
            print(json.dumps(summary, indent=1))
        finally:
            caller.close() if hasattr(caller, "close") else None


if __name__ == "__main__":
    main()
