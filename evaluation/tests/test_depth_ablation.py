"""Protect exact usage accounting, camera conventions and the RGB-only control."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_astra.depth_query import query_depth
from robocasa_common.depth_policy import DepthPolicy, usage_from_events

from inspect_robots.spaces import Box


def test_usage_does_not_double_count_cache_or_invent_missing_counts():
    """Cached input belongs to input; unknown failed-call usage is not zero."""
    result = usage_from_events(
        [
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 20},
            }
        ]
    )
    assert result["total_tokens"] == 120
    assert result["cached_input_tokens"] == 60
    assert result["reasoning_tokens"] is None
    assert usage_from_events([{"type": "turn.failed"}]) is None


def test_query_xy_region_and_stale_identity():
    """Pixel is column,row; a cell distribution differs from its center sample."""
    a = np.arange(1, 65537, dtype=np.float32).reshape(256, 256)
    request = {
        "camera": "left",
        "observation_id": "s:0",
        "kind": "pixel",
        "u": 3,
        "v": 8,
        "radius": 1,
        "cell_id": "",
    }
    result = query_depth({"left": a}, request, "s:0", ("pixel",))
    assert result["pixel_depth_m"] == a[8, 3]
    assert result["region_statistics"]["count"] == 9
    request.update(kind="grid", cell_id="r01c02")
    result = query_depth({"left": a}, request, "s:0", ("grid",))
    assert result["region_bounds_uv_half_open"] == [32, 16, 48, 32]
    assert result["region_statistics"]["count"] == 256
    with pytest.raises(ValueError, match="Stale"):
        query_depth({"left": a}, request, "s:1", ("grid",))
    with pytest.raises(ValueError, match="unavailable"):
        query_depth({"left": a}, request, "s:0", ())
    request.update(kind="pixel", u=-1)
    with pytest.raises(ValueError, match="bounds"):
        query_depth({"left": a}, request, "s:0", ("pixel",))


def test_rgb_control_excludes_all_depth_and_privileged_metadata(tmp_path, monkeypatch):
    """Private renderer arrays must never reach RGB-only prompts, images or schema."""
    monkeypatch.setenv("ASTRA_DEPTH_CONDITION", "rgb")
    monkeypatch.setenv("ASTRA_CODEX_HOME", str(tmp_path))
    monkeypatch.setenv("ASTRA_CODEX_EXECUTABLE", "unused")
    env = SimpleNamespace(
        info=SimpleNamespace(
            action_space=Box(shape=(12,), low=-np.ones(12), high=np.ones(12)),
            docs=json.dumps({"robot": "PandaOmron", "action_dim": 12, "depth_metadata": "secret"}),
        ),
        observers=[],
    )
    policy = DepthPolicy(env, tmp_path / "policy")
    policy.reset(SimpleNamespace(id="scene", instruction="Goal"))
    captured = {}

    def call(prompt, images, folder):
        captured.update(prompt=prompt, images=images)
        policy.calls.append({"cli_end_to_end_seconds": 1})
        return {"action": [0] * 12, "repeat": 1, "reason": "test"}

    monkeypatch.setattr(policy, "call", call)
    camera = np.zeros((256, 256, 3), np.uint8)
    obs = SimpleNamespace(
        images={"left": camera, "right": camera, "hand": camera, "left__depth": camera},
        state={"robot0_joint_pos": np.zeros(3), "object_position": np.ones(3)},
        extra={
            "steps": 0,
            "_depth_arrays": {"left": np.ones((256, 256), np.float32)},
            "depth_metadata": {"left": "secret"},
        },
    )
    policy.act(obs)
    assert len(captured["images"]) == 3
    assert all("__depth" not in str(p) for p in captured["images"])
    assert "secret" not in captured["prompt"]
    assert "object_position" not in captured["prompt"]
    assert "depth" not in captured["prompt"].lower()
    assert "queries" not in policy.schema["properties"]


def test_summary_keeps_missing_token_fields_unknown():
    """A partial usage object cannot silently become a complete zero-token report."""
    from robocasa_common.depth_site import summarize_calls

    summary = summarize_calls([{"cli_end_to_end_seconds": 2, "usage": {"input_tokens": 100}}])
    assert summary["reported_tokens"]["input_tokens"] == 100
    assert summary["reported_tokens"]["output_tokens"] is None
    assert summary["token_accounting_complete"] is False


def test_adoption_checks_exact_runtime_scene_and_condition(tmp_path):
    """A matching orphan is observed; unrelated or reused PIDs are never adopted."""
    from robocasa_common.depth_study import AdoptedProcess

    proc = tmp_path / "proc" / "123"
    proc.mkdir(parents=True)
    job = {"scene": "scene-one", "condition": "pixel"}
    args = [
        "python",
        "-m",
        "robocasa_common.depth_trial",
        "--runtime",
        str(tmp_path),
        "--scene",
        "scene-one",
        "--condition",
        "pixel",
    ]
    (proc / "cmdline").write_bytes("\0".join(args).encode())
    (proc / "stat").write_text("123 (python) S 1")
    adopted = AdoptedProcess(123, tmp_path, job, tmp_path / "proc")
    assert adopted.poll() is None
    assert adopted.returncode is None
    args[-1] = "rgb"
    (proc / "cmdline").write_bytes("\0".join(args).encode())
    assert adopted.poll() is not None
    args[-1] = "pixel"
    (proc / "cmdline").write_bytes("\0".join(args).encode())
    (proc / "stat").write_text("123 (python) Z 1")
    assert adopted.poll() is not None
