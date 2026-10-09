"""Motion bounds, independent schemas and idempotent native continuation contracts."""

import json
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
from robocasa_astra.astra_vla.depth_input import response_schema as depth_schema
from robocasa_astra.astra_vla.motion_control import adapt_prompt, prepare_actions, response_schema
from robocasa_astra.astra_vla.prompts import load_profile, system_prompt
from robocasa_astra.astra_vla.recovery import ActionJournal
from robocasa_astra.astra_vla.sim_server import ChunkServer


def test_profiles_bound_vectors_without_changing_binary_fields():
    """Precision has lower physical inputs; gripper and arm/base signs remain intact."""
    raw = np.ones((4, 12))
    raw[:, 4] = -1
    p, notes = prepare_actions(raw, "precision", "dual")
    t, _ = prepare_actions(raw, "transit", "dual")
    assert np.allclose(np.linalg.norm(p[:, 5:8], axis=1), 0.2)
    assert np.allclose(np.linalg.norm(p[:, 8:11], axis=1), 0.15)
    assert np.allclose(np.linalg.norm(t[:, 5:8], axis=1), 1.0)
    assert np.all(p[:, :4] == 0.1)
    assert np.array_equal(p[:, [4, 11]], raw[:, [4, 11]])
    assert len(notes) == 3
    assert np.array_equal(raw[:, 5:8], np.ones((4, 3)))
    bounded, _ = prepare_actions(p, "precision", "dual")
    assert np.allclose(bounded, p)


@pytest.mark.parametrize("mode,rows", [("precision", 0), ("precision", 5), ("transit", 17)])
def test_length_limits(mode, rows):
    """Worker rejects too-long chunks instead of silently losing model commands."""
    with pytest.raises(ValueError, match="rows"):
        prepare_actions(np.zeros((rows, 12)), mode, "dual")


def test_invalid_mode_and_nonfinite_and_legacy():
    """Unspecified dual modes fail closed; legacy retains the old 16-row format."""
    for mode in (None, "fast", ""):
        with pytest.raises(ValueError, match="requires motion_mode"):
            prepare_actions([[0] * 12], mode, "dual")
    with pytest.raises(ValueError, match="NaN"):
        prepare_actions([[float("nan")] * 12], "precision", "dual")
    a, _ = prepare_actions([[0] * 12] * 16, None)
    assert a.shape == (16, 12)
    with pytest.raises(ValueError, match="legacy"):
        prepare_actions(a, "precision")


def test_schema_and_prompt_conditions_are_independent():
    """Modes compose with spatial queries; the source prompt/schema remains unchanged."""
    base = depth_schema("spatial")
    s = response_schema(base, "dual")
    obj = {
        "scene": "s",
        "progress": "p",
        "memory": "m",
        "plan": "p",
        "actions": [[0] * 12],
        "queries": None,
        "motion_mode": "precision",
    }
    jsonschema.validate(obj, s)
    obj["motion_mode"] = "unknown"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(obj, s)
    assert "motion_mode" not in base["properties"]
    assert response_schema(base, "legacy") == base
    text = adapt_prompt(system_prompt(load_profile(), "OpenCabinet"))
    assert "exactly 16" not in text
    assert "precision" in text and "transit" in text
    assert "TASK SUCCESS CONDITION" in text and "index 4" in text


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
    with pytest.raises(ValueError, match="reused"):
        server.act_chunk(raw, [], 1, "t1", "transit")
    with pytest.raises(ValueError, match="rows"):
        server.act_chunk([[1] * 12] * 5, [], 2, "t2", "precision")
    assert len(calls) == 1


def test_precision_partial_checkpoint_uses_only_remaining_rows(tmp_path):
    """An acknowledged row is never reapplied after host reconnection."""
    server = ChunkServer.__new__(ChunkServer)
    server.motion_control = "dual"
    server.journal = ActionJournal(tmp_path, {})
    raw = [[0] * 12] * 3
    bounded, _ = prepare_actions(raw, "precision", "dual")
    chunk = server.journal.begin_chunk("t1", bounded.tolist(), 1, "precision")
    chunk["cursor"] = 2
    server.journal.save()
    server.executor = SimpleNamespace(steps_used=2)
    calls = []

    def run_chunk(actions, notes):
        calls.append(actions.copy())
        server.executor.steps_used += len(actions)
        return {"command": "chunk", "note": "", "task_success": False}

    server.executor.run_chunk = run_chunk
    server.executor.state = lambda: {"steps_used": server.executor.steps_used}
    server.replay = SimpleNamespace(mark=lambda *args: None)
    result = server.act_chunk(raw, [], 1, "t1", "precision")
    assert len(calls[0]) == 1 and result["resumed_rows"] == 2
    assert result["steps"] == 3 and result["chunk_length"] == 3
    persisted = json.loads((tmp_path / "worker" / "checkpoint.json").read_text())
    assert persisted["chunks"]["t1"]["result"]["motion_mode"] == "precision"


def test_experiment_config_records_dual_default_and_legacy_override(tmp_path):
    """CLI saves profile provenance and emits matching prompt/schema without running physics."""
    from robocasa_astra.astra_vla.experiment import prepare

    flags = [
        "--task",
        "OpenCabinet",
        "--scene",
        "0",
        "--scene-root",
        str(tmp_path),
        "--scripted",
        str(tmp_path / "script.json"),
    ]
    _, run, config = prepare([*flags, "--output", str(tmp_path / "dual"), "--condition", "spatial"])
    assert config["motion_control"] == "dual"
    assert config["motion_profiles"]["precision"]["max_steps"] == 4
    schema = json.loads((run / "response_schema.json").read_text())
    assert "motion_mode" in schema["required"] and "queries" in schema["required"]
    assert "precision" in (run / "system_prompt.md").read_text()
    _, run, config = prepare(
        [*flags, "--output", str(tmp_path / "legacy"), "--motion-control", "legacy"]
    )
    assert config["motion_control"] == "legacy" and config["motion_profiles"] is None
    assert "motion_mode" not in json.loads((run / "response_schema.json").read_text())["required"]


def test_dual_host_resume_keeps_mode_and_original_reply(tmp_path):
    """Host loss during precision execution reuses its saved decision, then switches to transit."""
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
                if self.interrupt:
                    self.interrupt = False
                    self.steps = 2
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
                json.dumps({"actions": raw, "motion_mode": mode}),
                {"input_tokens": 100, "output_tokens": 20},
                [],
                15.0,
            )

    sim, caller = Sim(), Caller()
    cfg = EpisodeConfig(task="OpenCabinet", seed=1, shots=0, max_turns=3, budget=20)
    with pytest.raises(KeyboardInterrupt):
        run_episode(sim, caller, cfg, tmp_path, [], motion_control="dual")
    summary = run_episode(sim, caller, cfg, tmp_path, [], resume=True, motion_control="dual")
    assert sim.seen[0] == sim.seen[1]
    assert sim.seen[0]["motion_mode"] == "precision"
    assert sim.seen[0]["actions"][0][5] == 0.2
    assert sim.seen[2]["motion_mode"] == "transit"
    assert caller.count == 2 and summary["usage"]["input_tokens"] == 200
    assert summary["output_format"] == "vla12-dual-variable-chunk"
    records = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert records[0]["response"]["actions"][0][5] == 1
    assert records[0]["latency_s"] == 15.0
