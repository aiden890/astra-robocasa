"""Prompts for the VLA-output variant: the skill variant's context with a 12-D action-chunk reply.

The system prompt has the same sections as ``astra_robodawn.prompts.system_prompt`` (role, robot
profile,
output format, success condition, demonstration note, response format); the command grammar is
replaced by
the action-chunk format. Each request is laid out the same way as in the skill variant: the
demonstration
block (image, then that turn's text, ...; END line), the current views, then the turn text, which
adds the
16-D RoboCasa365 state vector to the readable state.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..astra_robodawn.prompts import (
    END_OF_DEMOS,
    SUCCESS_CONDITIONS,
    Demo,
    _profile_text,
    demos_for,
    image_part,
    parts_text,
    state_text,
    text_part,
)
from ..astra_robodawn.prompts import (
    load_profile as _load_profile,
)
from .action_format import CHUNK, CHUNK_SCHEMA, DIM, INDEX_TABLE, describe, format_chunk

ASSETS = Path(__file__).resolve().parent / "assets"
PROFILE_PATH = ASSETS / "profile_vla.yaml"
DEMO_ROOT = ASSETS / "demos"

__all__ = [
    "RESPONSE_SCHEMA",
    "demo_parts",
    "image_part",
    "load_profile",
    "parts_text",
    "system_prompt",
    "text_part",
    "turn_text",
    "vla_demos",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {"type": "string"},
        "progress": {"type": "string"},
        "memory": {"type": "string"},
        "plan": {"type": "string"},
        "actions": CHUNK_SCHEMA,
    },
    "required": ["scene", "progress", "memory", "plan", "actions"],
    "additionalProperties": False,
}

ACTION_FORMAT = (
    "ACTION FORMAT (RoboCasa365 / GR00T action space): each reply contains ONE "
    f"chunk of exactly {CHUNK} actions; "
    f"each action is {DIM} numbers and drives the robot for one control step "
    "(20 steps per second), so a chunk is "
    "0.8 s of motion. All 16 actions are executed in order, open loop "
    "(nothing is corrected while they run); then "
    "you get new images, the state and what the chunk did. Every number "
    "must be within -1..1 (values outside are "
    "clipped).\n" + INDEX_TABLE + "\n"
    "Example action row (arm mode, fingertips forward about 1.2 cm and "
    "slightly up, gripper open):\n"
    "  [0, 0, 0, 0, -1, 1.0, 0, 0.2, 0, 0, 0, -1]"
)
STATE_VECTOR_NOTE = (
    "STATE VECTOR (RoboCasa365 observation.state, metres and xyzw quaternions): base position (3), "
    "base rotation (4), fingertip position relative to the base (3), "
    "fingertip rotation relative to "
    "the base (4), finger joint positions (2)"
)


def load_profile(path: Path = PROFILE_PATH) -> dict:
    """Read the VLA robot profile used in every request."""
    return _load_profile(path)


def vla_demos(task: str, shots: int = 0, *, primer: bool = False) -> list[Demo]:
    """Select successful examples and the action primer independently; zero means no examples."""
    demos = demos_for(task, shots, root=DEMO_ROOT)
    return [d for d in demos if primer or d.data["kind"] != "primer"]


def system_prompt(profile: dict, task: str | None = None, shown_demos: bool = False, *, context: str = "full") -> str:
    """Static instructions for the whole episode (the app-server thread's base instructions)."""
    success = SUCCESS_CONDITIONS.get(task)
    parts = [
        "You are the controller of a robot in a physics simulator. Each turn "
        "you receive camera images and the "
        f"robot state, and you reply with one chunk of {CHUNK} low-level actions "
        "that is executed step by step. Then "
        "you get new images and what the chunk did. You cannot use tools; "
        "reply only with the JSON object.",
        _profile_text(profile),
        ACTION_FORMAT,
    ]
    if success:
        parts.append(
            "TASK SUCCESS CONDITION (checked after every simulation step; the "
            "episode ends successfully the "
            "moment ALL of these hold at the same time, and only then):\n" + success[0]
        )
    if shown_demos:
        parts.append(
            "DEMONSTRATIONS: every request starts with a demonstration block (an "
            "action primer, then possibly "
            "one successful episode of the same kind of task in another kitchen, "
            "with the actions that "
            "produced it), each image followed by its text, and closed by an END "
            "line. After it come your own "
            "current images and the turn text."
        )
    parts.append(
        "RESPONSE FORMAT: reply with ONE JSON object with these fields, in this order:\n"
        '  "scene": one or two sentences: where the relevant objects, '
        "handles and the fingertips are (in cm, robot frame)\n"
        '  "progress": which sub-goal you are on and whether your last '
        "chunk had the intended effect\n"
        '"memory": rewrite your running notes: what you achieved, what you '
        "learned (heights that worked, motions that "
        "failed and why), what remains; under 120 words\n"
        '  "plan": the next few steps in words, and what this chunk does\n'
        f'  "actions": exactly {CHUNK} rows of {DIM} numbers (the chunk), in the order above\n'
        "Think in the scene/progress/plan fields BEFORE writing the actions. "
        "Keep each text field short (about 40 "
        "words, memory up to 120). The episode ends automatically as soon as "
        "the task checker registers success, so "
        "as long as you keep receiving turns the task is NOT complete yet."
    )
    text = "\n\n".join(parts)
    if context == "none":
        text = text.replace("you get new images and what the chunk did.", "you get new images and the current robot state.")
        text = text.replace("you get new images, the state and what the chunk did.", "you get new images and the current robot state.")
        text = text.replace("which sub-goal you are on and whether your last ", "the sub-goal indicated by the current observation; describe this ")
        text = text.replace("chunk had the intended effect", "observation only")
        text = text.replace(
            '"memory": rewrite your running notes: what you achieved, what you learned (heights that worked, motions that failed and why), what remains; under 120 words',
            '"memory": a brief note about the CURRENT observation only; these notes are recorded for review but never supplied in later requests'
        )
        text += "\n\nCONTEXT ABLATION: each request is independent. Previous actions, their execution feedback, recent turn history and accumulated notes are NOT provided. Use only the current RGB observations, current robot state, task goal and control rules. Do not assume knowledge of earlier attempts."
    return text


def turn_text(
    turn: int,
    max_turns: int,
    instruction: str,
    state: dict,
    dataset_state: list[float],
    last_results: list[dict],
    memory_text: str,
    captions: list[str],
    task: str | None = None,
) -> str:
    """Describe the current observation and measured feedback to the policy."""
    parts = [f"TASK: {instruction}"]
    if task in SUCCESS_CONDITIONS:
        parts.append("SUCCESS WHEN (all at once): " + SUCCESS_CONDITIONS[task][1])
    parts.append(f"TURN {turn} of at most {max_turns}.")
    if last_results:
        lines = [
            f"- {r['command']}: {'ok' if r.get('ok') else 'FAILED'}"
            + (f" ({r['note']})" if r.get("note") else "")
            for r in last_results
        ]
        parts.append("RESULT OF YOUR LAST CHUNK:\n" + "\n".join(lines))
    parts.append(
        "CURRENT STATE:\n"
        + state_text(state)
        + "\n"
        + STATE_VECTOR_NOTE
        + ":\n"
        + json.dumps(dataset_state)
    )
    parts.append(memory_text)
    images = [f"C{i + 1} = {c}" for i, c in enumerate(captions)]
    parts.append(
        "YOUR CURRENT IMAGES (the last images before this text, in order): " + "; ".join(images)
    )
    parts.append("Reply with the JSON object.")
    return "\n\n".join(parts)


def _chunk_lines(chunks: list[dict]) -> list[str]:
    lines = []
    for chunk in chunks:
        actions = np.asarray(chunk["actions"])
        head = (
            f"chunk at steps "
            f"{chunk['start_step']}-{chunk['start_step'] + len(actions) - 1}: {describe(actions)}"
        )
        lines.append(head + (":\n" + format_chunk(actions) if chunk.get("full") else ""))
    return lines


def _primer_parts(demo: Demo) -> list[dict]:
    d = demo.data
    parts = [
        text_part(
            f"ACTION PRIMER ({len(d['frames'])} chunks; not a task): what "
            f"action chunks do, shown in a "
            "kitchen that is not yours. For every step you see the image BEFORE "
            "the chunk, then the state, "
            "an explanation, the chunk itself and its measured effect; the next "
            "image shows the result. "
            "Read it once; any selected task demonstration follows."
        )
    ]
    for frame in d["frames"]:
        parts.append(image_part(demo.folder / frame["image"]))
        lines = [
            f"--- PRIMER {frame['label']}",
            f"state: {frame['state']}",
            f"explanation: {frame['plan']}",
        ]
        lines += _chunk_lines(frame["chunks"])
        lines.append("effect: " + frame["effect"])
        parts.append(text_part("\n".join(lines)))
    return parts


def _task_parts(demo: Demo) -> list[dict]:
    d = demo.data
    n_img = sum(1 for f in d["frames"] if f.get("image"))
    header = (
        f"DEMONSTRATION (a successful episode of the same kind of task in a DIFFERENT kitchen; "
        f"{d['source']['total_steps']} steps = {d['source']['chunks']} "
        f"chunks of {CHUNK}, grouped into "
        f'{len(d["frames"])} turns, {n_img} images). Its instruction was: "{d["instruction"]}".\n'
        "For each turn you see the image at the turn's start (an overview "
        "camera named in the label, when shown), "
        "then its state, the scene as read off that image, its plan, and the "
        "chunks of actions executed during "
        "the turn: the first chunk of every turn with an image is written out "
        "in full, the others are "
        "summarised. Copy the strategy (order of sub-goals, gripper "
        "orientation, how it aligned, approach and "
        "retreat, input sizes), NOT its positions: positions and heights in your kitchen differ."
    )
    if any(f.get("effect") for f in d["frames"]):
        header += (
            "\nEach turn also states its NET EFFECT, relative to the task objects "
            "where it was measured "
            "(how far the fingertips ended up from the handle/object, how far a door opened)."
        )
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
        lines += _chunk_lines(frame.get("chunks", []))
        if frame.get("effect"):
            lines.append("net effect: " + frame["effect"])
        parts.append(text_part("\n".join(lines)))
    return parts


def demo_parts(demos: list[Demo]) -> list[dict]:
    """The demonstration block at the start of every request (same layout as the skill variant)."""
    if not demos:
        return []
    parts = []
    for demo in demos:
        parts += _primer_parts(demo) if demo.data["kind"] == "primer" else _task_parts(demo)
    return [*parts, text_part(END_OF_DEMOS)]
