"""Coordinate signs, frozen robot frames, condition isolation and query round trips."""

from types import SimpleNamespace as NS

import jsonschema
import numpy as np
import pytest
from robocasa_astra.astra_vla.depth_input import answer_queries, instructions, response_schema
from robocasa_astra.astra_vla.spatial_query import query_spatial, snapshot_geometry


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


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"observation_id": "old"}, "Stale"),
        ({"u": 4}, "bounds"),
        ({"v": -1}, "bounds"),
        ({"u": 1.5}, "integers"),
        ({"camera": "missing"}, "camera"),
        ({"robot_part": "object:fish"}, "robot_part"),
        ({"radius": 1}, "radius=0"),
    ],
)
def test_invalid_query_rejected(overrides, message):
    """Reject mismatched observations and coordinates rather than inventing measurements."""
    with pytest.raises(ValueError, match=message):
        query_spatial({"cam": np.ones((4, 4))}, geometry(), request(**overrides), "obs")


def test_far_plane_rejected():
    """An empty render cannot become a target at a fabricated range."""
    with pytest.raises(ValueError, match="surface"):
        query_spatial({"cam": np.full((4, 4), 20.0)}, geometry(), request(), "obs")


def test_schema_and_mixed_queries():
    """Spatial is opt-in and shares existing pixel query and action accounting."""
    parsed = {
        "scene": "s",
        "progress": "p",
        "plan": "p",
        "memory": "m",
        "actions": None,
        "queries": [request()],
    }
    jsonschema.validate(parsed, response_schema("spatial"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(parsed, response_schema("hybrid"))
    depths = {"cam": np.full((4, 4), 2.0)}
    answers = answer_queries(
        depths, [request(), request(kind="pixel", robot_part=None)], "obs", "spatial", geometry()
    )
    assert answers[0]["distance_m"] == pytest.approx(np.sqrt(3))
    assert answers[1]["pixel_depth_m"] == 2
    assert np.all(depths["cam"] == 2)
    assert (
        "robot_part"
        not in response_schema("hybrid")["properties"]["queries"]["anyOf"][0]["items"]["properties"]
    )
    assert instructions("rgb") == ""
    assert "FROM the robot part TO" in instructions("spatial")


def test_snapshot_copies_robot_frames_only():
    """Moving live arrays after capture cannot change a query's frozen geometry."""
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    model = NS(
        camera_name2id=lambda _: 0,
        cam_fovy=[90],
        vis=NS(map=NS(zfar=10)),
        stat=NS(extent=2),
        body_names=["robot0_link", "fish"],
        site_names=["gripper0_tip", "plate"],
    )
    data = NS(
        cam_xpos=np.array([[1.0, 2.0, 3.0]]),
        cam_xmat=np.eye(3)[None],
        body_xpos=positions,
        body_xmat=np.tile(np.eye(3), (2, 1, 1)),
        site_xpos=positions,
        site_xmat=np.tile(np.eye(3), (2, 1, 1)),
    )
    robot = NS(
        robot_model=NS(naming_prefix="robot0_", base=NS(naming_prefix="mobilebase0_")),
        gripper={"right": NS(naming_prefix="gripper0_")},
    )
    pose = NS(
        tip=np.array([1.0, 2.0, 3.0]), tip_rot=np.eye(3), base=np.zeros(3), base_rot=np.eye(3)
    )
    env = NS(sim=NS(model=model, data=data), robots=[robot])
    frozen = snapshot_geometry(env, NS(pose=lambda: pose, steps_used=7), {"cam": np.ones((4, 4))})
    positions[:] = 99
    pose.tip[:] = 99
    assert frozen["robot_parts"]["gripper"]["position_world_m"] == [1, 2, 3]
    assert frozen["robot_parts"]["body:robot0_link"]["position_world_m"] == [1, 2, 3]
    assert set(frozen["robot_parts"]) == {
        "gripper",
        "base",
        "body:robot0_link",
        "site:gripper0_tip",
    }


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
