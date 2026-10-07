"""Build the in-context demonstrations in ``assets/demos/`` (offline, no model calls).

* ``primer``: the task-independent command primer. Each step is executed for real with the
  executor in a kitchen that is NOT an evaluation scene (layout/style 11), and records the image
  before the step, the state, the commands and their measured effect.
* ``task <Task>`` with ``"mode": "expert"`` in the plan: when a task's contact-rich steps could not be
  reproduced reliably with the executor, the demonstration is the successful human expert recording
  itself: key frames from its own camera video, annotated like the live views (same robot-mounted
  camera, so the same projection), the expert's state at each turn start, the commands written as an
  approximate translation of the expert's motion, and the measured effect from the recording.
* ``task <Task>`` (default mode): one task demonstration. The command plan in ``demo_plans/<Task>.json`` was written
  from the key events of a RoboCasa expert episode (``expert.key_events``); it is executed with the
  executor in that expert episode's own scene (rebuilt from its ep_meta, never the evaluation
  kitchen), so images and effects are real and the plan is verified to succeed.

Run from the repository root with one simulator:

    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src nice -n 10 $ASTRA_PYTHON \
        -m robocasa_astra.astra_robodawn.demo_builder primer
    ... -m robocasa_astra.astra_robodawn.demo_builder task OpenCabinet [--dry]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .commands import parse_command
from .executor import Executor
from .prompts import DEMO_ROOT
from .views import OVERVIEW_CAMERAS, annotate, surface_height

DEMO_IMAGE_SIZE = 256
CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

PRIMER_SCENE = {"task": "PickPlaceCounterToSink", "layout": 11, "style": 11, "seed": 771101}
PRIMER_STEPS = [
    ("move: translate the fingertips; the cyan tip circle and its shadow dot move over the grid",
     ["move up 10", "move forward 10"]),
    ("move left: positive left is the robot's left; the L labels of the grid increase to the left", ["move left 15"]),
    ("rotate yaw: turns the gripper about the vertical axis through the fingertips (the finger axis turns, the "
     "fingertip position stays)", ["rotate yaw 30"]),
    ("point forward: the fingers turn to point straight forward (approach becomes (1, 0, 0)); the fingertip "
     "position stays", ["point forward"]),
    ("point down then move down: approach becomes (0, 0, -1); descend towards the counter", ["point down", "move down 15"]),
    ("gripper close on nothing: the opening drops to 0.00, meaning nothing is held; then open again",
     ["gripper close", "gripper open"]),
    ("base: the whole platform drives; fingertip coordinates are robot-relative so they barely change, "
     "but the scene shifts in the images", ["base right 20"]),
    ("a blocked command: the fingertips cannot go through the counter, so the second move stops part of the way "
     "and the result says why", ["move down 20", "move down 20"]),
]


def make_env(task: str, seed: int, layout: int, style: int, horizon: int = 100000):
    """A native PandaOmron environment like the evaluation one, but in the given layout/style."""
    import robocasa  # noqa: F401
    from robocasa.utils.env_utils import create_env

    from ..worker import protect_assets

    protect_assets()
    env = create_env(task, robots="PandaOmron", seed=seed, layout_ids=[layout], style_ids=[style],
                     camera_names=CAMERAS, camera_widths=256, camera_heights=256, generative_textures=None,
                     control_freq=20, horizon=horizon)
    env.reset()
    return env


def _state_line(state: dict, surface_cm: float) -> str:
    f, l, h = state["fingertip_cm"]
    a = state["approach"]
    return (f"fingertips (forward {f:.0f}, left {l:.0f}, height {h:.0f}) cm, approach ({a[0]:.2f}, {a[1]:.2f}, "
            f"{a[2]:.2f}), opening {state['gripper_opening']:.2f}, work surface {surface_cm:.0f} cm")


def _effect(results: list[dict]) -> str:
    parts = []
    for r in results:
        text = f"{r['command']}: {'ok' if r['ok'] else 'FAILED'} ({r['note']})"
        if r.get("moved_cm") and r["ok"]:
            m = r["moved_cm"]
            text += f"; fingertips moved forward {m[0]:+.0f}, left {m[1]:+.0f}, up {m[2]:+.0f} cm"
        if r.get("base_moved_cm"):
            m = r["base_moved_cm"]
            text += f"; platform moved forward {m[0]:+.0f}, left {m[1]:+.0f} cm"
        parts.append(text)
    return " | ".join(parts)


def build_primer(out: Path) -> dict:
    scene = PRIMER_SCENE
    env = make_env(scene["task"], scene["seed"], scene["layout"], scene["style"])
    executor = Executor(env, step_budget=100000)
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (explanation, lines) in enumerate(PRIMER_STEPS, start=1):
        pose = executor.pose()
        surface = surface_height(env, pose.base, pose.base_rot)
        image = annotate(env, OVERVIEW_CAMERAS[0], pose.tip, pose.base, pose.base_rot, surface)
        image = image.resize((DEMO_IMAGE_SIZE, DEMO_IMAGE_SIZE))
        image.save(frames_dir / f"step{i}.png")
        before = executor.state()
        results = [executor.execute(parse_command(line)).to_json() for line in lines]
        frames.append({"label": f"primer step {i} (image: left overview camera before the step)",
                       "image": f"frames/step{i}.png", "state": _state_line(before, surface * 100),
                       "plan": explanation, "commands": lines, "effect": _effect(results)})
        print(f"step {i}: {lines} -> {_effect(results)[:150]}")
    meta = env.get_ep_meta()
    env.close()
    demo = {"kind": "primer", "task": "primer", "instruction": "(none: this primer only shows what each command does)",
            "source": {**scene, "instruction_of_scene": meta.get("lang"),
                       "built_by": "robocasa_astra.astra_robodawn.demo_builder primer",
                       "note": "executed with the real executor; not an evaluation scene (evaluation uses layout/style 1)"},
            "note": "Each step shows the image BEFORE the commands, then their measured effect.",
            "frames": frames}
    (out / "demo.json").write_text(json.dumps(demo, indent=1))
    return demo


PLAN_ROOT = Path(__file__).resolve().parent / "assets" / "demo_plans"
KEY_KINDS = ("gripper", "point")
MAX_DEMO_IMAGES = 6


def read_video_frames(path: Path, indices: list[int]) -> dict:
    """Decode the requested frame indices of a recorded MP4 into PIL images."""
    import imageio_ffmpeg
    from PIL import Image

    wanted, out = set(indices), {}
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    width, height = meta["size"]
    for i, raw in enumerate(reader):
        if i in wanted:
            out[i] = Image.frombytes("RGB", (width, height), raw)
        if len(out) == len(wanted):
            break
    reader.close()
    return out


def make_env_from_ep_meta(task: str, ep_meta: dict, seed: int = 0):
    """Rebuild a recorded episode's scene (layout, style, fixtures and objects) from its ep_meta."""
    env = make_env(task, seed, ep_meta["layout_id"], ep_meta["style_id"])
    env.set_ep_meta(ep_meta)
    env.reset()
    return env


def _expert_effect(ep, a: int, b: int) -> str:
    d = ep.tip_cm[b] - ep.tip_cm[a]
    appr = ep.rot[b][:, 2]
    settle = min(b + 10, ep.length - 1)
    return (f"in the recording the fingertips moved forward {d[0]:+.0f}, left {d[1]:+.0f}, up {d[2]:+.0f} cm; approach became "
            f"({appr[0]:.2f}, {appr[1]:.2f}, {appr[2]:.2f}); gripper opening {ep.opening[settle]:.2f}")


def expand_turns(ep, turns: list[dict], max_commands: int = 4) -> list[dict]:
    """Fill ``"pre" + auto moves`` turns from the recording and split any turn longer than ``max_commands``.

    A turn without explicit ``commands`` gets its intent commands ``pre`` (rotations, presets) followed by
    the axis moves of the recorded fingertip displacement over [t_start, t_end], so that the commands
    always agree with the recorded effect. Turns needing more than ``max_commands`` commands are split
    at the time the fingertip has covered half of its path, recursively.
    """
    from .expert import move_commands

    out = []
    for turn in turns:
        if "commands" in turn:
            out.append(turn)
            continue
        a, b = turn["t_start"], turn["t_end"]
        cmds = list(turn.get("pre", [])) + move_commands(ep.tip_cm[b] - ep.tip_cm[a])
        if len(cmds) <= max_commands or b - a < 4:
            out.append({**turn, "commands": cmds[:max_commands] if b - a < 4 else cmds})
            continue
        path = np.cumsum(np.r_[0, np.linalg.norm(np.diff(ep.tip_cm[a : b + 1], axis=0), axis=1)])
        mid = a + int(np.searchsorted(path, path[-1] / 2))
        mid = min(max(mid, a + 1), b - 1)
        first = {**turn, "t_end": mid}
        second = {**turn, "t_start": mid, "pre": [], "image": False,
                  "scene": "(continuing the previous turn)", "plan": "Continue: " + turn["plan"][0].lower() + turn["plan"][1:]}
        out += expand_turns(ep, [first, second], max_commands)
    return out


def build_expert_demo(task: str, plan: dict, out: Path, dry: bool = False) -> dict:
    """Demonstration from the expert recording itself (see the module docstring)."""
    from .expert import dataset_dir, key_events, load_episode

    episode = load_episode(task, plan["episode"])
    env = make_env_from_ep_meta(task, episode.ep_meta)
    pose = Executor(env, step_budget=1).pose()
    surface = surface_height(env, pose.base, pose.base_rot)
    turns = expand_turns(episode, plan["turns"])
    image_times = [t["t_start"] for t in turns if t.get("image")][: MAX_DEMO_IMAGES - 1] + [episode.length - 1]
    frames_raw = read_video_frames(episode.video(OVERVIEW_CAMERAS[0]), image_times)

    def picture(t: int, name: str) -> str:
        tip_r = episode.tip_cm[t] / 100.0
        tip_w = pose.base + pose.base_rot @ np.array([tip_r[0], tip_r[1], 0.0])
        tip_w[2] = tip_r[2]
        img = annotate(env, OVERVIEW_CAMERAS[0], tip_w, pose.base, pose.base_rot, surface, image=frames_raw[t])
        if not dry:
            img.resize((DEMO_IMAGE_SIZE, DEMO_IMAGE_SIZE)).save(out / "frames" / name)
        return f"frames/{name}"

    if not dry:
        (out / "frames").mkdir(parents=True, exist_ok=True)
    frames = []
    for i, turn in enumerate(turns, start=1):
        a, b = turn["t_start"], turn["t_end"]
        image = picture(a, f"turn{i:02d}.png") if a in image_times[:-1] and turn.get("image") else None
        state = {"fingertip_cm": episode.tip_cm[a].tolist(), "approach": episode.rot[a][:, 2].tolist(),
                 "gripper_opening": float(episode.opening[a])}
        frames.append({"label": f"turn {i}" + (" (image: left overview camera at the start of the turn)" if image else ""),
                       "image": image, "state": _state_line(state, surface * 100), "scene": turn.get("scene", ""),
                       "plan": turn["plan"], "commands": turn["commands"], "effect": _expert_effect(episode, a, b)})
    final_t = episode.length - 1
    frames.append({"label": "final state of the recording (image)", "image": picture(final_t, "final.png"),
                   "state": _state_line({"fingertip_cm": episode.tip_cm[final_t].tolist(),
                                         "approach": episode.rot[final_t][:, 2].tolist(),
                                         "gripper_opening": float(episode.opening[final_t])}, surface * 100),
                   "effect": "the recorded episode ended with the task checker reporting success"})
    env.close()
    demo = {
        "kind": "task", "task": task, "instruction": episode.ep_meta.get("lang"),
        "source": {"dataset": str(dataset_dir(task)), "episode": plan["episode"],
                   "layout_id": episode.ep_meta.get("layout_id"), "style_id": episode.ep_meta.get("style_id"),
                   "expert_key_events": key_events(episode), "plan_file": f"demo_plans/{task}.json",
                   "verified": {"mode": "expert recording", "executor_replay": False}},
        "note": plan.get("note", ""), "frames": frames,
    }
    if not dry:
        (out / "demo.json").write_text(json.dumps(demo, indent=1))
    print(json.dumps({"mode": "expert", "turns": len(frames), "images": len(image_times)}))
    return demo


def build_task_demo(task: str, out: Path, dry: bool = False) -> dict:
    """Execute ``demo_plans/<task>.json`` in its expert episode's scene and record the demonstration."""
    from .expert import dataset_dir, key_events, load_episode

    plan = json.loads((PLAN_ROOT / f"{task}.json").read_text())
    if plan.get("mode") == "expert":
        return build_expert_demo(task, plan, out, dry)
    episode = load_episode(task, plan["episode"])
    env = make_env_from_ep_meta(task, episode.ep_meta)
    executor = Executor(env, step_budget=plan.get("budget", 100000))
    frames, images = [], 0
    if not dry:
        (out / "frames").mkdir(parents=True, exist_ok=True)
    for i, turn in enumerate(plan["turns"], start=1):
        pose = executor.pose()
        surface = surface_height(env, pose.base, pose.base_rot)
        key = i == 1 or any(c.split()[0] in KEY_KINDS for c in turn["commands"])
        image = None
        if key and images < MAX_DEMO_IMAGES - 1 and not dry:
            images += 1
            img = annotate(env, OVERVIEW_CAMERAS[0], pose.tip, pose.base, pose.base_rot, surface)
            image = f"frames/turn{i:02d}.png"
            img.resize((DEMO_IMAGE_SIZE, DEMO_IMAGE_SIZE)).save(out / image)
        before = executor.state()
        results = [executor.execute(parse_command(c)).to_json() for c in turn["commands"]]
        frames.append({"label": f"turn {i}" + (" (image: left overview camera before the turn)" if image else ""),
                       "image": image, "state": _state_line(before, surface * 100), "scene": turn.get("scene", ""),
                       "plan": turn["plan"], "commands": turn["commands"], "effect": _effect(results)})
        print(f"turn {i}: {turn['commands']} -> {_effect(results)[:200]} | success={executor.success}")
        if executor.success:
            break
    pose = executor.pose()
    surface = surface_height(env, pose.base, pose.base_rot)
    if not dry:
        img = annotate(env, OVERVIEW_CAMERAS[0], pose.tip, pose.base, pose.base_rot, surface)
        img.resize((DEMO_IMAGE_SIZE, DEMO_IMAGE_SIZE)).save(out / "frames" / "final.png")
        frames.append({"label": "final state (image)", "image": "frames/final.png",
                       "state": _state_line(executor.state(), surface * 100),
                       "effect": "task checker: SUCCESS" if executor.success else "task checker: not successful"})
    result = {"success": executor.success, "steps": executor.steps_used, "turns": len(frames)}
    env.close()
    demo = {
        "kind": "task", "task": task, "instruction": episode.ep_meta.get("lang"),
        "source": {"dataset": str(dataset_dir(task)), "episode": plan["episode"],
                   "layout_id": episode.ep_meta.get("layout_id"), "style_id": episode.ep_meta.get("style_id"),
                   "expert_key_events": key_events(episode), "plan_file": f"demo_plans/{task}.json",
                   "verified": result},
        "note": plan.get("note", ""), "frames": frames,
    }
    if not dry:
        (out / "demo.json").write_text(json.dumps(demo, indent=1))
    print(json.dumps(result))
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["primer", "task"])
    parser.add_argument("task", nargs="?")
    parser.add_argument("--root", default=str(DEMO_ROOT))
    parser.add_argument("--dry", action="store_true", help="execute and report only; write nothing")
    args = parser.parse_args()
    out = Path(args.root) / ("primer" if args.what == "primer" else args.task)
    if (out / "demo.json").exists() and not args.dry:
        raise SystemExit(f"{out} already exists; remove it explicitly to rebuild")
    np.set_printoptions(precision=2)
    if args.what == "primer":
        build_primer(out)
    else:
        build_task_demo(args.task, out, dry=args.dry)


if __name__ == "__main__":
    main()
