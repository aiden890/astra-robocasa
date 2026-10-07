"""Prompt construction (adapted from RoboDawn ``harness/agent/prompts.py`` and ``demos.py``).

Everything that is identical across the turns of an episode goes into the system prompt file
(``model_instructions_file``: role, command grammar, robot profile, primer and task demonstration
text, reply rules) so that it forms a stable, cacheable prefix. Demonstration images are attached
first, then the current views, then the per-turn text.

Demonstration format (``assets/demos/<name>/demo.json``)::

    {"kind": "primer" | "task", "task": str, "instruction": str, "source": {...}, "note": str,
     "frames": [{"label": str, "image": "frames/x.png" | null, "state": str, "scene": str,
                 "plan": str, "commands": [str], "effect": str}]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from .commands import GRAMMAR_HELP

ASSETS = Path(__file__).resolve().parent / "assets"
PROFILE_PATH = ASSETS / "profile_panda_omron.yaml"
DEMO_ROOT = ASSETS / "demos"
MAX_COMMANDS_PER_TURN = 4

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "progress": {"type": "string"},
        "memory": {"type": "string"},
        "plan": {"type": "string"},
        "commands": {"type": "array", "items": {"type": "string"}, "minItems": 1,
                     "maxItems": MAX_COMMANDS_PER_TURN},
    },
    "required": ["scene", "progress", "memory", "plan", "commands"],
    "additionalProperties": False,
}

DEMO_NOTE = (
    "DEMONSTRATIONS are shown below: first a PRIMER showing what each command does, then (if present) one "
    "successful episode of the same kind of task recorded by an expert in a DIFFERENT kitchen (other layout, "
    "object positions and heights). Copy the strategy (order of sub-goals, gripper orientation, how it aligned, "
    "approach and retreat, when it checked the wrist camera), NOT its numbers: read positions for YOUR scene from "
    "your own images and state. Demonstration images are attached first, labelled D1, D2, ..."
)


@dataclass
class Demo:
    name: str
    data: dict
    folder: Path


def load_profile(path: Path = PROFILE_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_demo(name: str, root: Path = DEMO_ROOT) -> Demo:
    folder = Path(root) / name
    return Demo(name, json.loads((folder / "demo.json").read_text()), folder)


def demos_for(task: str, shots: int, root: Path = DEMO_ROOT) -> list[Demo]:
    """The primer always; the task demonstration only for ``shots >= 1`` (one demonstration per task)."""
    demos = [load_demo("primer", root)] if (Path(root) / "primer" / "demo.json").exists() else []
    if shots >= 1:
        demos.append(load_demo(task, root))
    return demos


def _profile_text(profile: dict) -> str:
    surface = "the 'work surface height' given in CURRENT STATE"
    p = {k: (v.replace("{surface_z}", surface) if isinstance(v, str) else [t.replace("{surface_z}", surface) for t in v])
         for k, v in profile.items()}
    tips = "\n".join(f"- {t}" for t in p["tips"])
    return (f"ROBOT: {p['robot']}\n\nCOORDINATE FRAME: {p['frame']}\n\nWORKSPACE: {p['workspace']}\n\n"
            f"CAMERAS: {p['cameras']}\n\nGRIPPER: {p['gripper']}\n\nTIPS:\n{tips}")


def render_demos(demos: list[Demo]) -> tuple[str, list[Path]]:
    """Demonstration text for the system prompt and the image files it refers to (D1, D2, ...)."""
    blocks, images = [], []
    for demo in demos:
        d = demo.data
        title = "PRIMER (what each command does)" if d["kind"] == "primer" else f"DEMONSTRATION of task {d['task']}"
        lines = [f"=== {title} ===", f"Instruction: {d['instruction']}"]
        if d.get("note"):
            lines.append(f"Note: {d['note']}")
        for frame in d["frames"]:
            tag = ""
            if frame.get("image"):
                images.append(demo.folder / frame["image"])
                tag = f" [image D{len(images)}]"
            lines.append(f"-- {frame['label']}{tag}")
            for key in ("state", "scene", "plan"):
                if frame.get(key):
                    lines.append(f"   {key}: {frame[key]}")
            if frame.get("commands"):
                lines.append(f"   commands: {json.dumps(frame['commands'])}")
            if frame.get("effect"):
                lines.append(f"   effect: {frame['effect']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), images


def system_prompt(profile: dict, demo_text: str) -> str:
    """Static instructions for the whole episode (written to the model instructions file)."""
    parts = [
        "You are the controller of a robot in a physics simulator. Each turn you receive camera images and the "
        "robot state, and you reply with a few discrete commands that are executed in order. Then you get new "
        "images and the outcome of every command. You cannot use tools; reply only with the JSON object.",
        _profile_text(profile),
        GRAMMAR_HELP,
    ]
    if demo_text:
        parts += [DEMO_NOTE, demo_text]
    parts.append(
        "RESPONSE FORMAT: reply with ONE JSON object with these fields, in this order:\n"
        '  "scene": one or two sentences: where the relevant objects, handles and the fingertips are (in cm, robot frame)\n'
        '  "progress": which sub-goal you are on and whether your last commands had the intended effect\n'
        '  "memory": rewrite your running notes: what you achieved, what you learned (heights that worked, commands that '
        "failed and why), what remains; under 120 words\n"
        '  "plan": the next few steps in words\n'
        f'  "commands": 1 to {MAX_COMMANDS_PER_TURN} commands, executed in order\n'
        "Think in the scene/progress/plan fields BEFORE choosing commands. Keep each text field short (about 40 words, "
        "memory up to 120). The episode ends automatically as soon as the task checker registers success, so as long "
        "as you keep receiving turns the task is NOT complete yet."
    )
    return "\n\n".join(parts)


def state_text(state: dict) -> str:
    f, l, h = state["fingertip_cm"]
    a, x = state["approach"], state["finger_axis"]
    return (
        f"fingertips at (forward {f:.1f}, left {l:.1f}, height {h:.1f}) cm\n"
        f"approach (fingers point along) = ({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}), finger axis = ({x[0]:.2f}, {x[1]:.2f}, {x[2]:.2f})\n"
        f"gripper opening = {state['gripper_opening']:.2f} (last command: {state['gripper_command']})\n"
        f"work surface height = {state['surface_z_cm']:.0f} cm\n"
        f"native steps used: {state['steps_used']} / {state['step_budget']}"
    )


def turn_text(turn: int, max_turns: int, instruction: str, state: dict, last_results: list[dict], memory_text: str,
              captions: list[str], n_demo_images: int) -> str:
    parts = [f"TASK: {instruction}", f"TURN {turn} of at most {max_turns}."]
    if last_results:
        lines = [f"- {r['command']}: {'ok' if r.get('ok') else 'FAILED'}" + (f" ({r['note']})" if r.get("note") else "")
                 for r in last_results]
        parts.append("RESULT OF YOUR LAST COMMANDS:\n" + "\n".join(lines))
    parts.append("CURRENT STATE:\n" + state_text(state))
    parts.append(memory_text)
    images = []
    if n_demo_images:
        images.append(f"D1..D{n_demo_images} = demonstration images (see the system instructions)")
    images += [f"C{i + 1} = {c}" for i, c in enumerate(captions)]
    parts.append("IMAGES ATTACHED (in order): " + "; ".join(images))
    parts.append("Reply with the JSON object.")
    return "\n\n".join(parts)
