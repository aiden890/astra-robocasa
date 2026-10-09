"""Coordinate signs, frozen robot frames, condition isolation and query round trips."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
from robocasa_astra.astra_vla.depth_input import answer_queries
from robocasa_astra.astra_vla.spatial_query import query_spatial


def geometry():
    """Use a non-origin gripper and a rotated base so incorrect frames cannot pass."""
    rot = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    return {
        "steps_used": 7,
        "cameras": {
            "cam": {
                "intrinsics": [[2, 0, 2], [0, 2, 2], [0, 0, 1]],
                "optical_to_world": np.diag([1, -1, -1]).tolist(),
                "position_world_m": [1, 2, 3],
                "far_m": 20,
            }
        },
        "robot_parts": {
            "gripper": {"position_world_m": [1, 2, 2], "rotation_world": rot, "description": "tip"},
            "base": {"position_world_m": [0, 0, 0], "rotation_world": rot, "description": "base"},
        },
    }


def request(**overrides):
    """Build one aligned pixel-to-gripper query."""
    return {
        "kind": "spatial",
        "observation_id": "obs",
        "camera": "cam",
        "u": 3,
        "v": 1,
        "radius": 0,
        "robot_part": "gripper",
        **overrides,
    }


def test_projection_and_relative_axes():
    """Image-right/up signs, world translation and local rotations are distinct."""
    answer = query_spatial({"cam": np.full((4, 4), 2.0)}, geometry(), request(), "obs")
    assert answer["point_world_m"] == [2, 3, 1]
    assert answer["delta_world_m"] == [1, 1, -1]
    assert answer["delta_part_local_m"] == [1, -1, -1]
    assert answer["delta_base_local_m"] == [1, -1, -1]
    assert answer["distance_m"] == pytest.approx(np.sqrt(3))
    assert answer["steps_used"] == 7


def test_server_frozen_query_guard(tmp_path):
    """Server persists calibration and rejects querying after a native action."""
    import json

    from robocasa_astra.astra_vla.sim_server import ChunkServer

    server = ChunkServer.__new__(ChunkServer)
    server.output = tmp_path
    server.condition = "spatial"
    server.geometry = geometry()
    server.depths = {"cam": np.full((4, 4), 2.0)}
    server.observation_id = "obs"
    server.last_observation = {"state": {"steps_used": 7}}
    server.executor = NS(steps_used=7)
    answers = server.query([request()], "obs")["answers"]
    assert answers[0]["distance_m"] == pytest.approx(np.sqrt(3))
    assert server.executor.steps_used == 7
    json.dumps(answers, allow_nan=False)
    server.executor.steps_used = 8
    with pytest.raises(ValueError, match="physics advanced"):
        server.query([request()], "obs")
    with pytest.raises(ValueError, match="stale"):
        server.query([request()], "old")


def test_query_answer_reaches_next_model_call_and_trace(tmp_path):
    """One spatial query reaches the model before any action, with usage counted once."""
    import base64
    import io
    import json

    from PIL import Image
    from robocasa_astra.astra_robodawn.codex import CallResult
    from robocasa_astra.astra_robodawn.loop import EpisodeConfig
    from robocasa_astra.astra_vla.loop import run_episode

    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    png = base64.b64encode(buf.getvalue()).decode()

    class Sim:
        steps = 0

        def state(self):
            return {
                "fingertip_cm": [25.0, 0, 129.0],
                "approach": [0, 0, -1],
                "finger_axis": [1, 0, 0],
                "gripper_opening": 1.0,
                "gripper_command": "open",
                "surface_z_cm": 92.0,
                "steps_used": self.steps,
                "step_budget": 16,
                "task_success": self.steps == 16,
            }

        def request(self, op, **fields):
            if op == "reset":
                return {"instruction": "open", "state": self.state()}
            if op == "observe":
                return {
                    "state": self.state(),
                    "dataset_state": [0] * 16,
                    "observation_id": "obs",
                    "views": [{"name": "cam", "caption": "camera", "png": png}],
                    "depth_views": [{"name": "cam", "png": png, "scale_m": [1, 2]}],
                    "robot_parts": {"gripper": "fingertip center"},
                }
            if op == "query":
                assert self.steps == 0
                return {
                    "answers": answer_queries(
                        {"cam": np.full((4, 4), 2.0)},
                        fields["queries"],
                        fields["observation_id"],
                        "spatial",
                        geometry(),
                    )
                }
            if op == "act_chunk":
                self.steps = 16
                return {
                    "command": "chunk",
                    "task_success": True,
                    "gripper_closed": False,
                    "gripper_opening": 1.0,
                    "state_after": self.state(),
                }
            raise AssertionError(op)

    class Caller:
        calls = 0

        def call(self, parts, folder):
            self.calls += 1
            if self.calls == 1:
                assert any("AVAILABLE ROBOT PARTS" in p.get("text", "") for p in parts)
                response = {"queries": [request()], "actions": None}
            else:
                assert sim.steps == 0
                text = " ".join(p.get("text", "") for p in parts)
                assert "distance_m" in text and "delta_base_local_m" in text
                response = {"queries": None, "actions": [[0] * 12] * 16}
            return CallResult(
                json.dumps(response), {"input_tokens": 100, "output_tokens": 10}, [], 15
            )

    sim, caller = Sim(), Caller()
    cfg = EpisodeConfig(task="OpenCabinet", seed=1, shots=0, max_turns=2, budget=16)
    summary = run_episode(sim, caller, cfg, tmp_path, [], condition="spatial")
    assert summary["task_success"] is True
    assert summary["usage"]["input_tokens"] == 200
    assert summary["usage"]["output_tokens"] == 20
    trace = json.loads((tmp_path / "trace.jsonl").read_text().splitlines()[0])
    assert trace["query_history"][0]["answers"][0]["distance_m"] == pytest.approx(np.sqrt(3))
    assert trace["latency_s"] == 30
    assert caller.calls == 2
