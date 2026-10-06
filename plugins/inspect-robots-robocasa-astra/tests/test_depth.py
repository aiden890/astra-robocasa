"""Verify metric depth alignment, transport, and actual model attachments."""

import base64
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from robocasa_astra.bridge import SparkEmbodiment
from robocasa_astra.depth import decode_depth, depth_summary, encode_depth, preview_depth
from robocasa_astra.policy import CodexPolicy

from inspect_robots.spaces import Box


def raw_observation(depth):
    """Build a small trusted worker observation with RGB and metric depth."""
    png = io.BytesIO()
    Image.fromarray(np.zeros((*depth.shape, 3), np.uint8)).save(png, format="PNG")
    return {
        "images": {"camera": base64.b64encode(png.getvalue()).decode()},
        "depths": {"camera": encode_depth(depth)},
        "depth_metadata": {"camera": {"unit": "m", "simulation_step": 1}},
        "state": {},
        "info": {},
    }


def test_lossless_float32_and_preview_scale():
    """Relative contrast exposes close depth differences without altering metric transport."""
    depth = np.array([[0.4, 0.5], [0.6, 0.7]], np.float32)
    original = depth.copy()
    np.testing.assert_array_equal(decode_depth(encode_depth(depth), (2, 2)), depth)
    image = preview_depth(depth)
    assert image.shape == (2, 2, 3)
    assert image[0, 0, 0] == 255 and image[1, 1, 0] == 0
    assert image[0, 0, 0] > image[0, 1, 0] > image[1, 0, 0] > image[1, 1, 0]
    np.testing.assert_array_equal(depth, original)
    np.testing.assert_allclose(image, preview_depth(depth + 1), atol=1)
    summary = depth_summary(depth)
    near, far = np.percentile(depth, [2, 98])
    assert summary["preview"]["near_white_m"] == near
    assert summary["preview"]["far_black_m"] == far
    assert summary["preview"]["colormap"] == "gray_r"
    assert summary["preview"]["renderer"] == "Pillow.ImageOps.colorize"


def test_constant_preview_and_dense_numeric_grid():
    """Flat maps stay finite, while dense samples preserve exact source pixel alignment."""
    assert np.all(preview_depth(np.full((4, 4), 1.25, np.float32)) == 128)
    assert depth_summary(np.full((4, 4), 1.25))["preview"]["constant_depth"]
    depth = np.arange(256 * 256, dtype=np.float32).reshape(256, 256) / 1000 + 0.1
    summary = depth_summary(depth)
    xs, ys = summary["sample_x_pixels"], summary["sample_y_pixels"]
    assert len(xs) == len(ys) == 16
    assert xs[0] == ys[0] == 0 and xs[-1] == ys[-1] == 255
    np.testing.assert_allclose(summary["sample_depth_m"], depth[np.ix_(ys, xs)], atol=0.00005)
    assert depth_summary(np.ones((2, 3)))["sample_x_pixels"] == [0, 1, 2]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_metric_depth_rejected(value):
    """Corrupt metric data never appears as a valid sensor measurement."""
    with pytest.raises(ValueError):
        encode_depth(np.full((2, 2), value))


def test_shape_mismatch_rejected():
    """Depth cannot silently shift or rescale relative to the paired RGB image."""
    with pytest.raises(ValueError):
        decode_depth(encode_depth(np.ones((2, 2))), (3, 3))


def test_bridge_and_model_receive_depth(tmp_path, monkeypatch):
    """Keep numeric depth separate and attach RGB plus its scaled depth preview to Codex."""
    env = SparkEmbodiment.__new__(SparkEmbodiment)
    env.output, env.sensor_index = tmp_path / "worker", 0
    observation = env.observation(raw_observation(np.full((4, 4), 1.25, np.float32)))
    assert list(observation.images) == ["camera", "camera__depth"]
    source = observation.extra["depth_maps"]["camera"]
    assert np.load(source).dtype == np.float32
    env.info = SimpleNamespace(
        action_space=Box(shape=(3,), low=-np.ones(3), high=np.ones(3)), docs="fixture"
    )
    policy = CodexPolicy(env, tmp_path / "calls", "codex", str(tmp_path / "auth"))
    policy.reset(SimpleNamespace(instruction="Place the mug"))
    captured = {}

    def execute(command, **kwargs):
        captured.update(command=command, prompt=kwargs["input"])
        output = command[command.index("-o") + 1]
        from pathlib import Path

        Path(output).write_text(json.dumps({"action": [0, 0, 0], "repeat": 1, "reason": "test"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("robocasa_astra.policy.subprocess.run", execute)
    policy.act(observation)
    assert captured["command"].count("--image") == 2
    assert "camera__depth" in captured["prompt"] and "1.25" in captured["prompt"]
    assert "optical-axis depth" in captured["prompt"]
    assert "relative_per_camera_per_observation" in captured["prompt"]
    assert "near_white_m" in captured["prompt"] and "far_black_m" in captured["prompt"]
    assert (tmp_path / "calls/call-0000/camera__depth_m.npy").is_file()
    receipt = json.loads((tmp_path / "calls/call-0000/receipt.json").read_text())
    assert receipt["depth_cameras"] == ["camera"]


def test_missing_camera_depth_rejected(tmp_path):
    """Require one depth map and calibration entry for each RGB camera."""
    env = SparkEmbodiment.__new__(SparkEmbodiment)
    env.output, env.sensor_index = tmp_path, 0
    raw = raw_observation(np.ones((4, 4), np.float32))
    raw["images"]["other_camera"] = raw["images"]["camera"]
    with pytest.raises(ValueError, match="Every RGB camera"):
        env.observation(raw)


def test_renderer_depth_shape_and_corruption():
    """Robosuite variants use HxW or HxWx1; NaNs cannot become invented distances."""
    from robocasa_astra.depth import normalize_depth_buffer

    value = np.full((4, 4), 0.5)
    np.testing.assert_array_equal(normalize_depth_buffer(value[..., None]), value)
    with pytest.raises(ValueError):
        normalize_depth_buffer(np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        normalize_depth_buffer(np.full((4, 4), 1.1))


def test_worker_rgb_depth_share_render_and_orientation(monkeypatch):
    """Metric depth and RGB use one render per camera and the same vertical flip."""
    import sys
    from types import ModuleType

    from robocasa_astra.worker import Simulator

    utilities = ModuleType("robosuite.utils.camera_utils")
    utilities.get_real_depth_map = lambda sim, depth: depth * 10
    utilities.get_camera_intrinsic_matrix = lambda *args: np.eye(3)
    utilities.get_camera_extrinsic_matrix = lambda *args: np.eye(4)
    monkeypatch.setitem(sys.modules, "robosuite.utils.camera_utils", utilities)
    calls = []
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    rgb = np.zeros((256, 256, 3), np.uint8)
    rgb[0, :, 0] = 200
    buffer = np.full((256, 256, 1), 0.5)
    buffer[0] = 0.2

    def render(**kwargs):
        calls.append(kwargs)
        return rgb, buffer

    sim = SimpleNamespace(
        render=render,
        model=SimpleNamespace(
            camera_names=cameras,
            stat=SimpleNamespace(extent=1),
            vis=SimpleNamespace(map=SimpleNamespace(znear=0.01, zfar=10)),
        ),
        data=SimpleNamespace(time=0.15),
    )
    worker = Simulator("PandaOmron", "test", depth=True)
    worker.parts = {}
    worker.steps = 3
    worker.env = SimpleNamespace(
        sim=sim,
        _get_observations=lambda **kwargs: {},
        _check_success=lambda: False,
        get_ep_meta=lambda: {"lang": "test"},
        action_dim=3,
        action_spec=(np.full(3, -1), np.ones(3)),
        control_freq=20,
        robots=[SimpleNamespace(composite_controller_config={})],
    )
    raw = worker.observe()
    assert len(calls) == 3 and all(call["depth"] for call in calls)
    assert set(raw["images"]) == set(raw["depths"]) == set(cameras)
    for camera in cameras:
        image = np.asarray(Image.open(io.BytesIO(base64.b64decode(raw["images"][camera]))))
        depth = decode_depth(raw["depths"][camera], (256, 256))
        assert image[-1, 0, 0] == 200 and depth[-1, 0] == 2
        assert image[0, 0, 0] == 0 and depth[0, 0] == 5
        assert raw["depth_metadata"][camera]["simulation_step"] == 3
