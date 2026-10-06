"""Check Xiaomi state, discrete controls and native ordering against the official contract."""

import numpy as np
import pytest
from robocasa_astra.worker import native_diagnostics
from robocasa_common.evaluate import scene_identifiers
from robocasa_common.xiaomi import axis_angle, native_action, robot_state


def test_native_renderer_diagnostics_do_not_corrupt_json(capfd):
    """Native OpenGL writes bypass Python stdout redirection, so redirect the FD too."""
    import os

    with native_diagnostics():
        os.write(1, b"native renderer diagnostic\n")
        print("Python diagnostic")
    print('{"ready":true}')
    captured = capfd.readouterr()
    assert captured.out == '{"ready":true}\n'
    assert "native renderer diagnostic" in captured.err
    assert "Python diagnostic" in captured.err


def test_seed_metadata_does_not_become_a_scene_filename():
    """Both published catalog revisions identify scenes by integer seed."""
    assert scene_identifiers({"PrepareCoffee": [{"seed": 123, "layout_id": 1}]}) == [
        "PrepareCoffee-123"
    ]
    assert scene_identifiers({"OpenCabinet": [456]}) == ["OpenCabinet-456"]


def test_state_matches_official_ee_first_order():
    """Avoid the unrelated pi05 quaternion state and preserve relative EE semantics."""
    state = {
        "robot0_base_to_eef_pos": [1, 2, 3],
        "robot0_base_to_eef_quat": [0, 0, 0, 1],
        "robot0_gripper_qpos": [4, 5],
        "robot0_base_pos": [6, 7, 8],
        "robot0_base_quat": [0, 0, 0, -1],
    }
    np.testing.assert_array_equal(robot_state(state), [1, 2, 3, 0, 0, 0, 4, 5, 6, 7, 8, 0, 0, 0])
    np.testing.assert_allclose(axis_angle([0, 0, 1, 0]), [0, 0, np.pi])
    np.testing.assert_array_equal(axis_angle([0, 0, 0, 0]), [0, 0, 0])


def test_official_thresholds_and_actual_native_part_order():
    """Quantization noise below 0.5 must not engage the native base controller."""
    parts = {
        "base": [0, 3],
        "torso": [3, 4],
        "right": [4, 10],
        "right_gripper": [10, 11],
        "base_mode": [11, 12],
    }
    official = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.49, 0.7, 0.8, 0.9, 0.01, 0.0078125]
    np.testing.assert_array_equal(
        native_action(official, parts), [0.7, 0.8, 0.9, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -1, -1]
    )
    official[6] = official[11] = 0.5
    assert native_action(official, parts)[-2:].tolist() == [1, 1]
    with pytest.raises(ValueError, match="12D"):
        native_action([0] * 14, parts)


def test_real_panda_hybrid_mode_is_separate_from_action_parts():
    """The live Panda controller appends hybrid mode outside _action_split_indexes."""
    parts = {"right": [0, 6], "right_gripper": [6, 7], "base": [7, 10], "torso": [10, 11]}
    a = [0.1] * 12
    expected = a.copy()
    expected[6] = expected[11] = -1
    np.testing.assert_array_equal(native_action(a, parts, 11), expected)


def test_controller_saturation_preserves_discrete_values_and_source():
    """Native scaling clips continuous axes but binary conversion still uses 0.5."""
    from robocasa_common.xiaomi_client_worker import controller_compatible_actions

    original = np.array([[2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0, -3.0, 0.0, 4.0, 2.0]])
    before = original.copy()
    expected = np.array([[1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, -1.0, 0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(controller_compatible_actions(original), expected)
    np.testing.assert_array_equal(original, before)
    with pytest.raises(ValueError, match="finite"):
        controller_compatible_actions(np.full((1, 12), np.nan))


def test_recovery_preserves_original_errors_and_never_repeats_normal_trials(tmp_path, monkeypatch):
    """Only unscored adapter failures get one separate attempt after queue completion."""
    import json
    import sys

    from robocasa_common import xiaomi_recover

    root = tmp_path / "runtime"
    site = tmp_path / "site"
    root.mkdir()
    public = site / "media/xiaomi-common-scenes-20261007"
    public.mkdir(parents=True)
    normal = {
        "scene": "PrepareCoffee-1",
        "task": "PrepareCoffee",
        "horizon": 1800,
        "execution_status": "success",
        "task_success": False,
    }
    error = {
        "scene": "PrepareCoffee-2",
        "task": "PrepareCoffee",
        "horizon": 1800,
        "execution_status": "error",
        "task_success": None,
    }
    state = {"complete": True, "results": [normal, error], "tasks": {"PrepareCoffee": {}}}
    (root / "status.json").write_text(json.dumps(state))
    executed = []

    def evaluate(folder, args):
        executed.append(folder.name)
        result = {"scene": folder.name, "execution_status": "success", "task_success": True}
        output = Path(args.output) / folder.name
        output.mkdir(parents=True)
        (output / "result.json").write_text(json.dumps(result))
        (output / "video.mp4").write_bytes(b"mock-video")
        return result

    from pathlib import Path

    monkeypatch.setattr(xiaomi_recover, "evaluate_scene", evaluate)
    monkeypatch.setattr(xiaomi_recover.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(sys, "argv", ["recover", "--runtime", str(root), "--site-root", str(site)])
    xiaomi_recover.main()
    final = json.loads((root / "status.json").read_text())
    assert executed == ["PrepareCoffee-2"]
    assert final["results"][0] == normal
    assert final["results"][1]["original_attempt"] == error
    assert json.loads((root / "first-pass-status.json").read_text()) == state
    assert final["tasks"]["PrepareCoffee"]["evaluated"] == 2
    assert final["tasks"]["PrepareCoffee"]["success_rate"] is None
