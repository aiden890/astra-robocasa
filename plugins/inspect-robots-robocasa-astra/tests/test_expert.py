"""Expert-trajectory helpers: frame conventions, path simplification and command generation."""

import numpy as np

from robocasa_astra.astra_robodawn.executor import POINT_PRESETS, axis_angle_matrix
from robocasa_astra.astra_robodawn.expert import (
    SITE_FROM_EEF,
    _rdp,
    move_commands,
    quat_xyzw_to_matrix,
    rotation_commands,
)


def test_quaternion_and_site_offset():
    assert np.allclose(quat_xyzw_to_matrix([0, 0, 0, 1]), np.eye(3))
    assert np.allclose(quat_xyzw_to_matrix([0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]), axis_angle_matrix([0, 0, 1], np.pi / 2))
    # hand body x (forward at the start pose) becomes the site's finger axis (left)
    hand = np.column_stack([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(float)
    site = hand @ SITE_FROM_EEF
    assert np.allclose(site[:, 2], [0, 0, -1]) and np.allclose(site[:, 0], [0, 1, 0])


def test_rdp_keeps_corners_only():
    line = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0], [10, 5, 0], [10, 10, 0]], float)
    assert _rdp(line, 1.0) == [0, 2, 4]


def test_move_commands_order_and_chunks():
    assert move_commands(np.array([25.0, -3.0, 12.0])) == ["move up 12", "move forward 20", "move forward 5", "move right 3"]
    assert move_commands(np.array([0.5, 4.0, -30.0])) == ["move left 4", "move down 20", "move down 10"]


def test_rotation_commands_prefers_presets():
    cmds, achieved = rotation_commands(POINT_PRESETS["down"], POINT_PRESETS["forward"])
    assert cmds == ["point forward"] and np.allclose(achieved, POINT_PRESETS["forward"])
    target = axis_angle_matrix([0, 0, 1], np.radians(30)) @ POINT_PRESETS["down"]
    cmds, _ = rotation_commands(POINT_PRESETS["down"], target)
    assert cmds == ["rotate yaw +30"]
