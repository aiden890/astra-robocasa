"""Discrete command vocabulary for a single-arm mobile manipulator (RoboCasa PandaOmron).

Adapted from RoboDawn ``harness/core/commands.py``: one arm instead of two, directions named in the
robot frame (forward / left / up) instead of world axes, and base commands for the mobile platform.

    move down 5          translate the fingertips 5 cm down, keeping the orientation
    rotate yaw 30        turn the gripper 30 deg about the robot's up axis, through the fingertips
    point forward        snap the gripper to a preset orientation
    gripper close
    base left 20         drive the platform 20 cm to the robot's left
    base turn -15        turn the platform 15 deg clockwise (seen from above)
    home / wait / done

Magnitudes are clipped to :data:`LIMITS` so that one command can never fling the arm or the base.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MOVE_DIRECTIONS = ("forward", "back", "left", "right", "up", "down")
ROTATION_AXES = ("roll", "pitch", "yaw")
POINT_PRESETS = ("down", "forward", "down45")
BASE_DIRECTIONS = ("forward", "back", "left", "right")

LIMITS = {
    "move_cm": 20.0,
    "rotate_deg": 90.0,
    "base_cm": 50.0,
    "base_turn_deg": 45.0,
}

# Unit vector of every direction word in the robot frame (x forward, y left, z up).
DIRECTION_VECTORS = {
    "forward": (1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
}


@dataclass(frozen=True)
class Command:
    """One parsed command.

    ``kind`` is one of move, rotate, point, gripper, base_move, base_turn, home, wait, done.
    ``name`` holds the direction, rotation axis, preset or gripper state; ``value`` the magnitude
    (cm or deg) after clipping. ``clipped`` tells whether the requested magnitude exceeded the limit.
    """

    kind: str
    name: str = ""
    value: float = 0.0
    raw: str = ""
    clipped: bool = False

    def text(self) -> str:
        """Canonical form, as echoed back to the model and written to the trace."""
        if self.kind == "move":
            return f"move {self.name} {self.value:g}"
        if self.kind == "rotate":
            return f"rotate {self.name} {self.value:+g}"
        if self.kind == "point":
            return f"point {self.name}"
        if self.kind == "gripper":
            return f"gripper {self.name}"
        if self.kind == "base_move":
            return f"base {self.name} {self.value:g}"
        if self.kind == "base_turn":
            return f"base turn {self.value:+g}"
        return self.kind


class CommandError(ValueError):
    """Raised for a line that matches no command of the grammar."""


_NUM = r"([-+]?\d+(?:\.\d+)?)"
_PATTERNS = [
    ("move", re.compile(rf"^move\s+({'|'.join(MOVE_DIRECTIONS)})\s+{_NUM}\s*(?:cm)?$")),
    ("rotate", re.compile(rf"^rotate\s+({'|'.join(ROTATION_AXES)})\s+{_NUM}\s*(?:deg)?$")),
    ("point", re.compile(rf"^point\s+({'|'.join(POINT_PRESETS)})$")),
    ("gripper", re.compile(r"^gripper\s+(open|close)$")),
    ("base_turn", re.compile(rf"^base\s+turn\s+{_NUM}\s*(?:deg)?$")),
    ("base_move", re.compile(rf"^base\s+({'|'.join(BASE_DIRECTIONS)})\s+{_NUM}\s*(?:cm)?$")),
    ("home", re.compile(r"^home$")),
    ("wait", re.compile(r"^wait$")),
    ("done", re.compile(r"^done$")),
]

_OPPOSITE = {"forward": "back", "back": "forward", "left": "right", "right": "left", "up": "down", "down": "up"}


def _clip(value: float, limit: float) -> tuple[float, bool]:
    return max(-limit, min(limit, value)), abs(value) > limit


def parse_command(line: str) -> Command:
    """Parse one command line (case-insensitive, tolerant of code ticks and trailing comments)."""
    text = line.strip().strip("`").split("#", 1)[0]
    text = re.sub(r"\s+", " ", text).strip().rstrip(".").lower()
    if not text:
        raise CommandError("empty command")
    for kind, pattern in _PATTERNS:
        match = pattern.match(text)
        if not match:
            continue
        if kind in ("move", "base_move"):
            direction, value = match.group(1), float(match.group(2))
            if value < 0:  # "move left -5" means "move right 5"
                direction, value = _OPPOSITE[direction], -value
            limit = LIMITS["move_cm"] if kind == "move" else LIMITS["base_cm"]
            value, clipped = _clip(value, limit)
            return Command(kind, direction, value, raw=line, clipped=clipped)
        if kind == "rotate":
            value, clipped = _clip(float(match.group(2)), LIMITS["rotate_deg"])
            return Command(kind, match.group(1), value, raw=line, clipped=clipped)
        if kind == "base_turn":
            value, clipped = _clip(float(match.group(1)), LIMITS["base_turn_deg"])
            return Command(kind, "turn", value, raw=line, clipped=clipped)
        if kind in ("point", "gripper"):
            return Command(kind, match.group(1), raw=line)
        return Command(kind, raw=line)
    raise CommandError(f"unrecognised command: {line!r}")


def parse_command_list(lines: list) -> tuple[list[Command], list[str]]:
    """Parse every line; returns the valid commands and one error message per invalid line."""
    commands, errors = [], []
    for line in lines:
        if not isinstance(line, str):
            errors.append(f"command is not a string: {line!r}")
            continue
        try:
            commands.append(parse_command(line))
        except CommandError as exc:
            errors.append(str(exc))
    return commands, errors


GRAMMAR_HELP = """\
COMMAND GRAMMAR (one command per string, case-insensitive). Directions are in the ROBOT frame:
forward = the way the robot faces, left = the robot's left, up = up.
  move forward|back|left|right|up|down <cm>
                        translate the fingertips, keeping the gripper orientation. 0 < cm <= 20.
  rotate roll|pitch|yaw <deg>
                        rotate the gripper about a robot axis through the fingertips. |deg| <= 90.
                        roll = about the forward axis, pitch = about the left axis, yaw = about the up axis;
                        signs follow the right-hand rule (e.g. yaw +30 turns the fingers 30 deg
                        counter-clockwise seen from above).
  point down|forward|down45
                        snap the gripper to a fixed preset orientation (this also undoes earlier rotations):
                        fingers pointing straight down, straight forward (horizontal), or 45 deg between the
                        two. In every preset the fingers close left-right. Use "rotate ..." to fine-tune.
  gripper open|close    open or close the fingers.
  base forward|back|left|right <cm>
                        drive the mobile platform (the arm moves with it). 0 < cm <= 50.
  base turn <deg>       turn the platform in place, positive = to the left (counter-clockwise). |deg| <= 45.
  home                  return the arm to its initial pose.
  wait                  let physics settle for a moment without moving.
  done                  declare the task complete (the episode only ends if the checker agrees).
"""
