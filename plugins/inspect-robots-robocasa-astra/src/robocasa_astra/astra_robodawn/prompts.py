"""Prompt construction (adapted from RoboDawn ``harness/agent/prompts.py`` and ``demos.py``).

The system prompt holds what is identical for every model (role, robot profile, command grammar,
success condition, reply rules). Each request's user input is, in order (as in RoboDawn's
``demo_messages``): the demonstration block, where every demonstration turn's image is immediately
followed by its own text (state, scene, plan, commands, net effect, failed commands), closed by an
END sentence; then the current camera views; then the per-turn text. The demonstration block is the
same in every request of an episode, so it forms a stable, cacheable prefix.

Input parts are ``{"type": "text", "text": str}`` or ``{"type": "image", "path": str}``.

Demonstration format (``assets/demos/<name>/demo.json``)::

    {"kind": "primer" | "task", "task": str, "instruction": str, "source": {...}, "note": str,
     "frames": [{"label": str, "image": "frames/x.png" | null, "state": str, "scene": str,
                 "plan": str, "commands": [str], "failed": [str], "effect": str}]}

``commands`` are the commands that worked; ``failed`` the ones that did not (with the reason), shown
apart as in RoboDawn. ``effect`` is the net effect of the turn, relative to the task objects where the
builder could measure them.
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

# The native RoboCasa success checks (_check_success) in plain words, with the same thresholds. The checker runs
# after every simulation step; the episode ends the moment every listed condition holds at the same time.
SUCCESS_CONDITIONS = {
    "OpenCabinet": (
        "Open EACH door of the target cabinet ALL THE WAY: keep swinging it until it stops at its hinge limit "
        "(about 90 deg, the door panel perpendicular to the cabinet front). The checker needs every door at >= 90% of "
        "that limit at the same time (for a double-door cabinet: BOTH doors); a door that only LOOKS open (about 80 "
        "deg) still fails, so always finish the swing to the stop. Once a door is fully open, do NOT touch it again: "
        "moving the gripper or arm near an open door can push it partly closed.",
        "every door swung open to its stop (>= 90%); do not touch an already opened door"),
    "PickPlaceSinkToCounter": (
        "(1) the object rests ON / IN the container (plate) on the counter: touching it and with its centre "
        "horizontally within 70% of the container's radius from the container centre; (2) the container still "
        "touches the counter; (3) the gripper (fingertip centre) is MORE THAN 25 cm away from the object. "
        "So: place the object on the middle of the container, release, then move the gripper away.",
        "object on the container centre, container on the counter, gripper > 25 cm from the object"),
    "PrepareCoffee": (
        "(1) the mug stands under the coffee machine's dispenser: its centre within 4 cm horizontally of the "
        "machine's mug spot and within 10 cm vertically; (2) the gripper is MORE THAN 25 cm away from the mug; "
        "(3) the machine has been started: the gripper must have TOUCHED the start button at least once (any "
        "contact turns it on for the rest of the episode); (4) the gripper is MORE THAN 15 cm away from the start "
        "button. So: put the mug under the dispenser, release and back off, touch the start button, then move away. "
        "HOW TO SEE THAT THE BUTTON WORKED: the moment the start button is really pressed, a thin brown stream of "
        "coffee appears, falling from the machine's dispenser down to the mug spot, and it stays visible for the rest "
        "of the episode (it is thin: a few pixels wide in the overview images). If you do not see that brown stream, "
        "the button was NOT pressed: go back and press it again (move the fingertips onto the button until they touch "
        "it). Only after you see the stream, move the gripper away.",
        "mug within 4 cm of the dispenser spot, gripper > 25 cm from the mug, start button pressed (a thin brown coffee "
        "stream is visible under the dispenser), gripper > 15 cm from the button"),
    "PanTransfer": (
        "(1) the vegetable is ON the plate (touching it, centre within 70% of the plate radius); (2) the pan is back "
        "on the stove, its centre within 8 cm of a burner centre; (3) the gripper is MORE THAN 25 cm away from the "
        "pan; (4) the robot NEVER touched the food during the whole episode (one touch of the gripper on the "
        "vegetable fails the task for good). So: hold only the pan handle, tip the vegetable onto the plate, put "
        "the pan back on a burner, release and move away.",
        "vegetable on the plate, pan on a burner (<= 8 cm), gripper > 25 cm from the pan, food never touched"),
    "StirVegetables": (
        "(1) BOTH named vegetables are inside the pot (touching it, centre within 70% of its radius); (2) the pot "
        "stays on the burner that is on (within 15 cm of its centre); (3) the spatula is grasped (fingers closed on "
        "it, touching it); (4) while all of that holds, the spatula moves BOTH vegetables (each moves at least "
        "0.5 mm horizontally in a simulation step while the spatula touches at least one of them) for at least 5 "
        "simulation steps in total. So: put both vegetables in the pot, grasp the spatula, put its blade among the "
        "vegetables and move it back and forth in small strokes.",
        "both vegetables in the pot on the lit burner, spatula grasped, stir so both vegetables move for >= 5 steps"),
}

END_OF_DEMOS = ("--- END OF THE DEMONSTRATIONS. Your own episode starts with the next images; its kitchen, object "
                "positions and heights differ, so measure everything again from your own images and state.")


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


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(path: Path | str) -> dict:
    return {"type": "image", "path": str(path)}


def _primer_parts(demo: Demo) -> list[dict]:
    d = demo.data
    n_img = sum(1 for f in d["frames"] if f.get("image"))
    parts = [text_part(f"COMMAND PRIMER ({len(d['frames'])} steps, {n_img} images; not a task): what each command does, "
                       "shown in a kitchen that is not yours. For every step you see the image BEFORE the commands, then "
                       "the state, an explanation, the commands and their measured effect; the next image shows the "
                       "result. Read it once: the task demonstration follows.")]
    for frame in d["frames"]:
        if frame.get("image"):
            parts.append(image_part(demo.folder / frame["image"]))
        lines = [f"--- PRIMER {frame['label']}"]
        if frame.get("state"):
            lines.append(f"state: {frame['state']}")
        if frame.get("plan"):
            lines.append(f"explanation: {frame['plan']}")
        if frame.get("commands"):
            lines.append("commands: " + json.dumps(frame["commands"]))
        if frame.get("effect"):
            lines.append("effect: " + frame["effect"])
        parts.append(text_part("\n".join(lines)))
    return parts


def _task_parts(demo: Demo) -> list[dict]:
    d = demo.data
    n_img = sum(1 for f in d["frames"] if f.get("image"))
    header = (f"DEMONSTRATION (a successful episode of the same kind of task in a DIFFERENT kitchen; {len(d['frames'])} "
              f"turns, {n_img} images). Its instruction was: \"{d['instruction']}\".\n"
              "For each turn you see the image the controller received (left overview camera, when shown), then its "
              "state, the scene as read off that image, its plan and the commands it sent; the next turn starts after "
              "they were executed. Copy the strategy (order of sub-goals, gripper orientation, how it aligned, approach "
              "and retreat), NOT its numbers: positions and heights in your kitchen differ.")
    if any(f.get("effect") for f in d["frames"]):
        header += ("\nEach turn also states its NET EFFECT, relative to the task objects where it was measured (how far "
                   "the fingertips ended up from the handle/object, how far a door opened). Reproduce these "
                   "object-relative effects in your scene; the command numbers belong to the demonstration's positions.")
    if d.get("note"):
        header += "\nNote: " + d["note"]
    parts = [text_part(header)]
    for frame in d["frames"]:
        if frame.get("image"):
            parts.append(image_part(demo.folder / frame["image"]))
        lines = [f"--- DEMO {frame['label']}"]
        for key in ("state", "scene", "plan"):
            if frame.get(key):
                lines.append(f"{key}: {frame[key]}")
        if frame.get("commands"):
            lines.append("commands: " + json.dumps(frame["commands"]))
        if frame.get("effect"):
            lines.append("net effect: " + frame["effect"])
        if frame.get("failed"):
            lines.append("(FAILED: " + "; ".join(frame["failed"]) + ")")
        parts.append(text_part("\n".join(lines)))
    return parts


def demo_parts(demos: list[Demo]) -> list[dict]:
    """The demonstration block placed at the start of every request: image, then its text, per turn; END."""
    if not demos:
        return []
    parts = []
    for demo in demos:
        parts += _primer_parts(demo) if demo.data["kind"] == "primer" else _task_parts(demo)
    return parts + [text_part(END_OF_DEMOS)]


def parts_text(parts: list[dict]) -> str:
    """Readable rendering of input parts for review files: texts verbatim, images as ``[image: path]``."""
    return "\n\n".join(p["text"] if p["type"] == "text" else f"[image: {p['path']}]" for p in parts)


def system_prompt(profile: dict, task: str | None = None, shown_demos: bool = False) -> str:
    """Static instructions for the whole episode (written to the model instructions file)."""
    success = SUCCESS_CONDITIONS.get(task)
    parts = [
        "You are the controller of a robot in a physics simulator. Each turn you receive camera images and the "
        "robot state, and you reply with a few discrete commands that are executed in order. Then you get new "
        "images and the outcome of every command. You cannot use tools; reply only with the JSON object.",
        _profile_text(profile),
        GRAMMAR_HELP,
    ]
    if success:
        parts.append("TASK SUCCESS CONDITION (checked after every simulation step; the episode ends successfully the "
                     "moment ALL of these hold at the same time, and only then):\n" + success[0])
    if shown_demos:
        parts.append("DEMONSTRATIONS: every request starts with a demonstration block (a command primer, then possibly one "
                     "successful episode of the same kind of task in another kitchen), each image followed by its text, "
                     "and closed by an END line. After it come your own current images and the turn text.")
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
              captions: list[str], task: str | None = None) -> str:
    parts = [f"TASK: {instruction}"]
    if task in SUCCESS_CONDITIONS:
        parts.append("SUCCESS WHEN (all at once): " + SUCCESS_CONDITIONS[task][1])
    parts.append(f"TURN {turn} of at most {max_turns}.")
    if last_results:
        lines = [f"- {r['command']}: {'ok' if r.get('ok') else 'FAILED'}" + (f" ({r['note']})" if r.get("note") else "")
                 for r in last_results]
        parts.append("RESULT OF YOUR LAST COMMANDS:\n" + "\n".join(lines))
    parts.append("CURRENT STATE:\n" + state_text(state))
    parts.append(memory_text)
    images = [f"C{i + 1} = {c}" for i, c in enumerate(captions)]
    parts.append("YOUR CURRENT IMAGES (the last images before this text, in order): " + "; ".join(images))
    parts.append("Reply with the JSON object.")
    return "\n\n".join(parts)
