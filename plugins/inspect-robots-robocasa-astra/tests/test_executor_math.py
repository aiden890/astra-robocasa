"""Rotation helpers of the executor: Rodrigues, axis-angle extraction, point presets, yaw wrap."""

import numpy as np
import pytest

from robocasa_astra.astra_robodawn.executor import (
    POINT_PRESETS,
    axis_angle_matrix,
    rotation_vector,
    wrap,
    yaw_of,
)


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 2, -0.5]])
@pytest.mark.parametrize("angle", [0.0, 0.3, -1.2, 2.9, np.pi])
def test_axis_angle_roundtrip(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    rot = axis_angle_matrix(axis, angle)
    assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-9)
    vec = rotation_vector(rot)
    assert np.isclose(np.linalg.norm(vec), abs(angle), atol=1e-6)
    assert np.allclose(axis_angle_matrix(vec, np.linalg.norm(vec)) if np.linalg.norm(vec) else np.eye(3), rot, atol=1e-6)


@pytest.mark.parametrize("name, approach", [("down", [0, 0, -1]), ("forward", [1, 0, 0]),
                                            ("down45", [np.sqrt(0.5), 0, -np.sqrt(0.5)])])
def test_point_presets_are_rotations_with_left_right_fingers(name, approach):
    rot = POINT_PRESETS[name]
    assert np.allclose(rot.T @ rot, np.eye(3)) and np.isclose(np.linalg.det(rot), 1.0)
    assert np.allclose(rot[:, 2], approach) and np.allclose(rot[:, 0], [0, 1, 0])


def test_yaw_and_wrap():
    assert np.isclose(yaw_of(axis_angle_matrix([0, 0, 1], 0.7)), 0.7)
    assert np.isclose(wrap(np.pi + 0.1), -np.pi + 0.1)
    assert np.isclose(wrap(-0.2), -0.2)
