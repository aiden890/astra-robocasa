"""Motion bounds, independent schemas and idempotent native continuation contracts."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_astra.astra_vla.recovery import ActionJournal
from robocasa_astra.astra_vla.sim_server import ChunkServer


def test_worker_guard_and_mode_bound_idempotence(tmp_path):
    """Worker applies bounded rows once, with same-ID mode changes rejected."""
    server = ChunkServer.__new__(ChunkServer)
    server.motion_control = "dual"
    server.journal = ActionJournal(tmp_path, {})
    server.executor = SimpleNamespace(steps_used=0)
    calls = []

    def run_chunk(actions, notes):
        calls.append(actions.copy())
        server.executor.steps_used += len(actions)
        return {"command": "chunk", "steps": len(actions), "note": "", "task_success": False}

    server.executor.run_chunk = run_chunk
    server.executor.state = lambda: {"steps_used": server.executor.steps_used}
    server.replay = SimpleNamespace(mark=lambda *args: None)
    raw = [[1] * 12] * 2
    result = server.act_chunk(raw, [], 1, "t1", "precision")
    assert np.allclose(np.linalg.norm(calls[0][:, 5:8], axis=1), 0.2)
    assert result["motion_mode"] == "precision" and result["chunk_length"] == 2
    assert server.act_chunk(raw, [], 1, "t1", "precision") == result
    assert len(calls) == 1
    with pytest.raises(ValueError, match="target only"):
        server.act_chunk(raw, [], 1, "t1", "transit")
    with pytest.raises(ValueError, match="rows"):
        server.act_chunk([[1] * 12] * 5, [], 2, "t2", "precision")
    assert len(calls) == 1


def test_dual_host_resume_keeps_mode_and_original_reply(tmp_path):
    """Reconnect during transit reuses the saved destination without another model call."""
    import base64
    import io

    from PIL import Image
    from robocasa_astra.astra_robodawn.codex import CallResult
    from robocasa_astra.astra_robodawn.loop import EpisodeConfig
    from robocasa_astra.astra_vla.loop import run_episode

    buf = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buf, format="PNG")
    png = base64.b64encode(buf.getvalue()).decode()

    class Sim:
        steps = 0
        interrupt = True

        def __init__(self):
            self.seen = []

        def state(self):
            return {
                "fingertip_cm": [25.0, 0, 129.0],
                "approach": [0, 0, -1],
                "finger_axis": [1, 0, 0],
                "gripper_opening": 1.0,
                "gripper_command": "open",
                "surface_z_cm": 92.0,
                "steps_used": self.steps,
                "step_budget": 20,
                "task_success": self.steps >= 20,
            }

        def request(self, op, **fields):
            if op == "reset":
                return {"instruction": "open", "state": self.state()}
            if op == "observe":
                return {
                    "state": self.state(),
                    "dataset_state": [0] * 16,
                    "observation_id": str(self.steps),
                    "views": [{"name": "cam", "caption": "camera", "png": png}],
                }
            if op == "act_chunk":
                self.seen.append(fields)
                if self.interrupt and fields["motion_mode"] == "transit":
                    self.interrupt = False
                    self.steps = 6
                    raise KeyboardInterrupt
                self.steps = 4 if fields["motion_mode"] == "precision" else 20
                return {
                    "command": fields["motion_mode"],
                    "task_success": self.steps == 20,
                    "gripper_closed": False,
                    "gripper_opening": 1.0,
                    "state_after": self.state(),
                }
            raise AssertionError(op)

    class Caller:
        count = 0

        def call(self, parts, folder):
            self.count += 1
            mode = "precision" if self.count == 1 else "transit"
            raw = [[0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, -1]] * (4 if self.count == 1 else 16)
            return CallResult(
                json.dumps(
                    {
                        "actions": raw if mode == "precision" else None,
                        "motion_mode": mode,
                        "transit_target": None
                        if mode == "precision"
                        else {
                            "position_world_m": [0.5, 0, 0.8],
                            "observation_id": "4",
                            "purpose": "pre_precision",
                            "grasp_confirmed": False,
                        },
                    }
                ),
                {"input_tokens": 100, "output_tokens": 20},
                [],
                15.0,
            )

    sim, caller = Sim(), Caller()
    cfg = EpisodeConfig(task="OpenCabinet", seed=1, shots=0, max_turns=3, budget=20)
    with pytest.raises(KeyboardInterrupt):
        run_episode(sim, caller, cfg, tmp_path, [], motion_control="dual")
    summary = run_episode(sim, caller, cfg, tmp_path, [], resume=True, motion_control="dual")
    assert sim.seen[1] == sim.seen[2]
    assert sim.seen[0]["motion_mode"] == "precision"
    assert sim.seen[0]["actions"][0][5] == 0.2
    assert sim.seen[2]["motion_mode"] == "transit"
    assert sim.seen[2]["actions"] is None
    assert sim.seen[2]["transit_target"] == {
        "position_world_m": [0.5, 0, 0.8],
        "observation_id": "4",
        "purpose": "pre_precision",
        "grasp_confirmed": False,
    }
    assert caller.count == 2 and summary["usage"]["input_tokens"] == 200
    assert summary["output_format"] == "precision-vla12-transit-world-target"
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert records[0]["response"]["actions"][0][5] == 1
    assert records[0]["latency_s"] == 15.0
