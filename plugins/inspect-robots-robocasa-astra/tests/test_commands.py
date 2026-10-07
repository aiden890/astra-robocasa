"""Grammar of the discrete RoboCasa commands: parsing, sign handling and magnitude limits."""

import pytest

from robocasa_astra.astra_robodawn.commands import (
    LIMITS,
    CommandError,
    parse_command,
    parse_command_list,
)


@pytest.mark.parametrize(
    "line, kind, name, value",
    [
        ("move down 5", "move", "down", 5.0),
        ("Move Forward 12.5 cm", "move", "forward", 12.5),
        ("move left -4", "move", "right", 4.0),
        ("rotate yaw -30", "rotate", "yaw", -30.0),
        ("rotate pitch 15 deg", "rotate", "pitch", 15.0),
        ("point down45", "point", "down45", 0.0),
        ("gripper close", "gripper", "close", 0.0),
        ("base left 20", "base_move", "left", 20.0),
        ("base back -10", "base_move", "forward", 10.0),
        ("base turn -15", "base_turn", "turn", -15.0),
        ("`home`", "home", "", 0.0),
        ("wait  # settle", "wait", "", 0.0),
        ("done.", "done", "", 0.0),
    ],
)
def test_parse(line, kind, name, value):
    cmd = parse_command(line)
    assert (cmd.kind, cmd.name, cmd.value) == (kind, name, value)
    assert not cmd.clipped


def test_limits_clip_and_flag():
    move = parse_command("move up 35")
    assert move.value == LIMITS["move_cm"] and move.clipped
    rot = parse_command("rotate roll -120")
    assert rot.value == -LIMITS["rotate_deg"] and rot.clipped
    base = parse_command("base forward 80")
    assert base.value == LIMITS["base_cm"] and base.clipped
    turn = parse_command("base turn 90")
    assert turn.value == LIMITS["base_turn_deg"] and turn.clipped


@pytest.mark.parametrize("line", ["", "move sideways 3", "left move x 5", "gripper 0.5", "fly up"])
def test_rejects(line):
    with pytest.raises(CommandError):
        parse_command(line)


def test_parse_list_collects_errors():
    commands, errors = parse_command_list(["move up 5", "jump", 7, "gripper open"])
    assert [c.text() for c in commands] == ["move up 5", "gripper open"]
    assert len(errors) == 2


def test_canonical_text_roundtrip():
    for line in ["move back 3", "rotate yaw +45", "base right 12", "base turn -10", "point forward"]:
        cmd = parse_command(line)
        assert parse_command(cmd.text()) == parse_command(cmd.text())
        assert parse_command(cmd.text()).value == cmd.value
