"""Build the VLA demonstrations in ``astra_vla/assets/demos/`` (offline, one simulator, no model calls).

* ``primer``: what action chunks do. Each step executes one 16-step chunk with :class:`ChunkExecutor` in the
  same non-evaluation kitchen as the skill primer and records the image before it, the chunk and the measured
  effect (how far the fingertips really moved compared with what the chunk asked for).
* ``task <Task>``: the SAME demonstration as the skill variant (``astra_robodawn/assets/demos/<Task>``: same
  episode, same turns, images, states, scene / plan texts and net effects), with the commands replaced by the
  native 12-D actions that produced that trajectory, cut into 16-step chunks:

  - executor-built demos: the skill plan is executed again with the skill executor in the expert episode's
    scene and every native action is recorded (the run must reproduce the skill demo's step count);
  - expert-recording demos: the RoboCasa365 dataset actions of that episode, which are already in this format.

  Chunks are cut over the whole episode (as a VLA would emit them) and listed under the turn they start in.

    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src nice -n 10 $ASTRA_PYTHON \
        -m robocasa_astra.astra_vla.demo_builder primer | task OpenCabinet
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ..astra_robodawn.demo_builder import (DEMO_IMAGE_SIZE, PLAN_ROOT, PRIMER_SCENE, _state_line, make_env,
                                           make_env_from_ep_meta)
from ..astra_robodawn.prompts import DEMO_ROOT as SKILL_DEMO_ROOT
from ..astra_robodawn.views import OVERVIEW_CAMERAS, annotate, surface_height
from .action_format import CHUNK, DIM, from_env
from .chunk_runner import ChunkExecutor

DEMO_ROOT = Path(__file__).resolve().parent / "assets" / "demos"


def _rows(**parts) -> np.ndarray:
    """A chunk of identical rows: arm mode, gripper open unless given, other entries from ``parts``."""
    row = np.zeros(DIM)
    row[4], row[11] = -1.0, -1.0
    for index, value in parts.items():
        row[int(index[1:])] = value
    return np.tile(row, (CHUNK, 1))


def _primer_steps() -> list[tuple[str, np.ndarray]]:
    steps = [
        ("end_effector_position forward (index 5) = 0.2 for all 16 steps: about 0.2 x 1.2 cm x 16 = 4 cm forward",
         _rows(i5=0.2)),
        ("index 6 (left) = 0.2: about 4 cm to the robot's left", _rows(i6=0.2)),
        ("index 7 (up) = -0.15: about 3 cm down", _rows(i7=-0.15)),
        ("a large input: index 5 = 1.0 for all 16 steps: about 19 cm forward (the largest move one chunk can make)",
         _rows(i5=1.0)),
        ("mixing motion and holding: index 5 = -1.0 for 8 steps, then all-zero arm inputs (hold still) for 8 steps",
         np.vstack([_rows(i5=-1.0)[:8], _rows()[:8]])),
        ("index 10 (rotation about the up axis) = 0.3: about 0.3 x 6.3 deg x 16 = 30 deg of yaw; the fingertip "
         "position stays", _rows(i10=0.3)),
        ("index 11 = +1 (close) for all steps: the fingers close on nothing (opening goes to about 0.00)",
         _rows(i11=1.0)),
        ("index 11 = -1 again: the fingers open", _rows()),
        ("base mode: index 4 = +1 and base_motion index 1 (left) = -0.5: the platform drives to the right; the arm "
         "keeps its pose relative to the platform", _rows(i4=1.0, i1=-0.5)),
    ]
    return steps


def build_primer(out: Path) -> dict:
    scene = PRIMER_SCENE
    env = make_env(scene["task"], scene["seed"], scene["layout"], scene["style"])
    runner = ChunkExecutor(env, step_budget=100000)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (explanation, chunk) in enumerate(_primer_steps(), start=1):
        pose = runner.pose()
        surface = surface_height(env, pose.base, pose.base_rot)
        annotate(env, OVERVIEW_CAMERAS[0], pose.tip, pose.base, pose.base_rot, surface) \
            .resize((DEMO_IMAGE_SIZE, DEMO_IMAGE_SIZE)).save(out / "frames" / f"step{i}.png")
        before = runner.state()
        result = runner.run_chunk(chunk)
        frames.append({"label": f"primer step {i} (image: left overview camera before the chunk)",
                       "image": f"frames/step{i}.png", "state": _state_line(before, surface * 100),
                       "plan": explanation, "chunks": [{"start_step": runner.steps_used - result["steps"],
                                                        "actions": chunk.tolist(), "full": True}],
                       "effect": result["note"]})
        print(f"step {i}: {result['note'][:200]}")
    env.close()
    demo = {"kind": "primer", "task": "primer", "instruction": "(none: this primer only shows what action chunks do)",
            "source": {**scene, "built_by": "robocasa_astra.astra_vla.demo_builder primer",
                       "note": "executed with ChunkExecutor; not an evaluation scene"},
            "note": "Each step shows the image BEFORE one 16-step chunk, the chunk, and its measured effect.",
            "frames": frames}
    (out / "demo.json").write_text(json.dumps(demo, indent=1))
    return demo


def dataset_actions(task: str, index: int) -> np.ndarray:
    """The RoboCasa365 actions of an expert episode (dataset order, i.e. already the VLA format)."""
    import pandas as pd

    from ..astra_robodawn.expert import dataset_dir

    frame = pd.read_parquet(Path(dataset_dir(task)) / "data" / "chunk-000" / f"episode_{index:06d}.parquet")
    return np.stack(frame["action"].to_list())


def executor_actions(task: str, plan: dict) -> tuple[np.ndarray, list[int], bool]:
    """Re-run a skill plan in its expert scene; native actions in dataset order, turn start steps, success."""
    from ..astra_robodawn.commands import parse_command
    from ..astra_robodawn.executor import Executor
    from ..astra_robodawn.expert import load_episode

    episode = load_episode(task, plan["episode"])
    env = make_env_from_ep_meta(task, episode.ep_meta)
    recorded = []
    executor = Executor(env, step_budget=plan.get("budget", 100000), on_step=lambda a, o, c: recorded.append(a))
    starts = []
    for turn in plan["turns"]:
        if executor.success:
            break
        starts.append(executor.steps_used)
        for command in turn["commands"]:
            executor.execute(parse_command(command))
    env.close()
    return np.stack([from_env(a) for a in recorded]), starts, executor.success


def build_task(task: str, out: Path) -> dict:
    """The skill demonstration of ``task`` with its commands replaced by 16-step chunks of native actions."""
    from ..astra_robodawn.demo_builder import expand_turns
    from ..astra_robodawn.expert import load_episode

    skill_folder = SKILL_DEMO_ROOT / task
    skill = json.loads((skill_folder / "demo.json").read_text())
    plan = json.loads((PLAN_ROOT / f"{task}.json").read_text())
    if plan.get("mode") == "expert":
        actions = dataset_actions(task, plan["episode"])
        turns = expand_turns(load_episode(task, plan["episode"]), plan["turns"])
        starts, success, source = [t["t_start"] for t in turns], True, "RoboCasa365 dataset actions of the episode"
    else:
        actions, starts, success = executor_actions(task, plan)
        expected = skill["source"]["verified"]["steps"]
        if not success or len(actions) != expected:
            raise RuntimeError(f"re-run of the skill plan gave {len(actions)} steps (success={success}), "
                               f"the skill demo recorded {expected}: the trajectories would differ")
        source = "native actions recorded while re-running the skill plan (same trajectory as the skill demo)"
    turn_frames = [f for f in skill["frames"] if f["label"].startswith("turn ")]
    if len(turn_frames) != len(starts):
        raise RuntimeError(f"{len(turn_frames)} skill turns but {len(starts)} turn starts")
    owner = lambda step: max(i for i, s in enumerate(starts) if s <= step)  # noqa: E731
    per_turn: list[list[dict]] = [[] for _ in starts]
    for first in range(0, len(actions), CHUNK):
        per_turn[owner(first)].append({"start_step": first, "actions": np.round(actions[first:first + CHUNK], 3).tolist()})
    (out / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in skill["frames"]:
        frame = {k: v for k, v in frame.items() if k not in ("commands", "failed", "command_results")}
        if frame.get("image"):
            shutil.copy(skill_folder / frame["image"], out / frame["image"])
        if frame["label"].startswith("turn "):
            i = turn_frames.index(next(f for f in turn_frames if f["label"] == frame["label"]))
            chunks = per_turn[i]
            for k, chunk in enumerate(chunks):
                chunk["full"] = bool(frame.get("image")) and k == 0
            frame["turn_start_step"] = starts[i]
            frame["chunks"] = chunks
        frames.append(frame)
    demo = {**{k: v for k, v in skill.items() if k != "frames"}, "frames": frames,
            "source": {**skill["source"], "skill_demo": f"astra_robodawn/assets/demos/{task}",
                       "actions": source, "total_steps": len(actions), "chunks": -(-len(actions) // CHUNK),
                       "built_by": "robocasa_astra.astra_vla.demo_builder task"}}
    (out / "demo.json").write_text(json.dumps(demo, indent=1))
    print(json.dumps({"task": task, "steps": len(actions), "chunks": demo["source"]["chunks"],
                      "full_chunks": sum(c["full"] for f in frames for c in f.get("chunks", [])), "success": success}))
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["primer", "task"])
    parser.add_argument("task", nargs="?")
    parser.add_argument("--root", default=str(DEMO_ROOT))
    args = parser.parse_args()
    out = Path(args.root) / ("primer" if args.what == "primer" else args.task)
    if (out / "demo.json").exists():
        raise SystemExit(f"{out} already exists; remove it explicitly to rebuild")
    if args.what == "primer":
        build_primer(out)
    else:
        build_task(args.task, out)


if __name__ == "__main__":
    main()
