"""Dataset action order must match the native robot controller."""

import numpy as np
from robocasa_astra.astra_vla import action_format as af


def test_dataset_action_maps_to_native_controller():
    """A known action maps translation, rotation, gripper, base and torso correctly."""
    row = np.array([0.1, 0.2, 0.3, 0.4, -1, 0.5, 0.6, 0.7, 0.8, 0.9, 1, -1])
    expected = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1, -1, 0.1, 0.2, 0.3, 0.4, -1])
    native = af.to_env(row)
    assert np.array_equal(native, expected)
    assert np.array_equal(af.from_env(native), row)
