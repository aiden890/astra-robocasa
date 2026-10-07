"""astra_vla without a simulator or model: the 12-D action format, chunk validation and
description."""

import numpy as np
import pytest
from robocasa_astra.astra_vla import action_format as af


def _random_dataset_rows(n=50, seed=0):
    rng = np.random.default_rng(seed)
    rows = rng.uniform(-1, 1, size=(n, af.DIM))
    rows[:, 4] = rng.choice([-1.0, 1.0], n)  # control_mode and gripper are +-1 in RoboCasa365 data
    rows[:, 11] = rng.choice([-1.0, 1.0], n)
    return rows


def test_to_env_equals_robocasa_playback_reordering():
    """Dataset vectors map to native indices exactly like official playback."""
    lerobot_utils = pytest.importorskip("robocasa.utils.lerobot_utils")
    rows = _random_dataset_rows()
    official = np.zeros_like(rows)
    for key, (lo, hi) in af.MODALITY.items():
        h_lo, h_hi = lerobot_utils.ACTION_KEY_ORDERING_HDF5[key]
        official[:, h_lo:h_hi] = rows[:, lo:hi]
    ours = np.stack([af.to_env(r) for r in rows])
    assert np.array_equal(ours, official)


def test_to_env_binarises_gripper_and_mode_like_the_eval_wrapper():
    """Binary control fields retain the evaluation wrapper threshold."""
    row = np.zeros(af.DIM)
    row[11], row[4] = 0.6, 0.4
    env = af.to_env(row)
    assert env[6] == 1.0 and env[11] == -1.0
    row[11], row[4] = 0.49, 0.5
    env = af.to_env(row)
    assert env[6] == -1.0 and env[11] == 1.0


def test_from_env_inverts_to_env():
    """Dataset and native orders round-trip without changing values."""
    rows = _random_dataset_rows(seed=1)
    for row in rows:
        assert np.allclose(af.from_env(af.to_env(row)), row)


def test_validate_chunk_shape_clipping_and_nan():
    """Reject malformed chunks and record clipping instead of hiding it."""
    chunk, notes = af.validate_chunk(np.full((af.CHUNK, af.DIM), 0.2).tolist())
    assert chunk.shape == (16, 12) and notes == []
    raw = np.zeros((af.CHUNK, af.DIM))
    raw[0, 5] = 3.0
    chunk, notes = af.validate_chunk(raw)
    assert chunk[0, 5] == 1.0 and "1 value(s)" in notes[0]
    with pytest.raises(ValueError, match="16 rows of 12"):
        af.validate_chunk(np.zeros((15, 12)))
    raw[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        af.validate_chunk(raw)


def test_commanded_motion_and_describe():
    """Describe the measured action scale and gripper switch accurately."""
    chunk = np.zeros((af.CHUNK, af.DIM))
    chunk[:, 4] = -1
    chunk[:, 11] = -1
    chunk[:, 5] = 0.25  # forward 0.25 * 1.2 cm * 16 = 4.8 cm (measured scale)
    chunk[8:, 11] = 1
    motion = af.commanded_motion(chunk)
    assert np.isclose(motion["move_cm"][0], 4.8) and motion["gripper"] == "switches at step 9"
    text = af.describe(chunk)
    assert "fwd +4.8" in text and "gripper switches at step 9" in text
    assert af.CHUNK_SCHEMA["minItems"] == af.CHUNK_SCHEMA["maxItems"] == 16
    assert af.format_chunk(chunk[:1]).startswith("[[0,0,0,0,-1,0.25,0,0,0,0,0,-1]")


def test_vla_prompts_and_demo_block():
    """The selected demonstrations preserve image-text association and task rules."""
    from robocasa_astra.astra_vla import prompts

    text = prompts.system_prompt(prompts.load_profile(), "OpenCabinet", shown_demos=True)
    assert (
        "ACTION FORMAT" in text
        and "index 11   gripper_close" in text
        and "TASK SUCCESS CONDITION" in text
    )
    assert "COMMAND GRAMMAR" not in text and "'point forward'" not in text
    assert list(prompts.RESPONSE_SCHEMA["properties"]) == [
        "scene",
        "progress",
        "memory",
        "plan",
        "actions",
    ]
    block = prompts.demo_parts(prompts.vla_demos("OpenCabinet", 1, primer=True))
    assert block[-1]["text"].startswith("--- END OF THE DEMONSTRATIONS")
    texts = [p["text"] for p in block if p["type"] == "text"]
    assert any(t.startswith("ACTION PRIMER") for t in texts) and any(
        t.startswith("DEMONSTRATION") for t in texts
    )
    assert any("[[0,0,0,0,-1," in t for t in texts)  # at least one chunk written out in full
    state = {
        "fingertip_cm": [1, 2, 3],
        "approach": [0, 0, -1],
        "finger_axis": [0, 1, 0],
        "gripper_opening": 1.0,
        "gripper_command": "open",
        "surface_z_cm": 92,
        "steps_used": 0,
        "step_budget": 10,
    }
    turn = prompts.turn_text(1, 5, "do it", state, [0.0] * 16, [], "MEM", ["c"], "OpenCabinet")
    assert "STATE VECTOR" in turn and "SUCCESS WHEN" in turn


class FakeChunkSim:
    """Answers the astra_vla sim_server protocol; success after the second chunk."""

    def __init__(self):
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        self.png = base64.b64encode(buf.getvalue()).decode()
        self.chunks = []

    def state(self):
        """Return fake measured state after each acknowledged chunk."""
        return {
            "fingertip_cm": [25.0, 0.0, 129.0],
            "approach": [0, 0, -1],
            "finger_axis": [1, 0, 0],
            "gripper_opening": 1.0,
            "gripper_command": "open",
            "surface_z_cm": 92.0,
            "steps_used": 16 * len(self.chunks),
            "step_budget": 100,
            "task_success": len(self.chunks) >= 2,
        }

    def request(self, op, **fields):
        """Exercise the VLA protocol without a native simulator or real model."""
        if op == "reset":
            return {"instruction": "Open the cabinet doors.", "scene": {}, "state": self.state()}
        if op == "observe":
            return {
                "state": self.state(),
                "dataset_state": [0.0] * 16,
                "views": [{"name": "cam", "caption": "a camera", "png": self.png}],
            }
        if op == "act_chunk":
            self.chunks.append(fields["actions"])
            return {
                "command": "chunk 16 steps: ...",
                "kind": "chunk",
                "ok": True,
                "note": "executed 16/16 steps",
                "steps": 16,
                "task_success": len(self.chunks) >= 2,
                "gripper_closed": False,
                "gripper_opening": 1.0,
                "state_after": self.state(),
            }
        raise AssertionError(op)


def test_vla_loop_validates_and_executes_chunks(tmp_path):
    """Unusable responses do not execute actions and feedback reaches the next call."""
    import json

    from robocasa_astra.astra_robodawn.loop import EpisodeConfig
    from robocasa_astra.astra_robodawn.scripted import ScriptedCaller
    from robocasa_astra.astra_vla.loop import run_episode

    good = np.zeros((16, 12))
    good[:, 4], good[:, 11], good[:, 5] = -1, -1, 0.5
    replies = [
        {"actions": good.tolist()},
        {"actions": [[0.0] * 12] * 15},
        {"actions": good.tolist()},
    ]
    sim = FakeChunkSim()
    cfg = EpisodeConfig(task="OpenCabinet", seed=1, shots=0, max_turns=5, budget=100)
    summary = run_episode(sim, ScriptedCaller(replies), cfg, tmp_path, [])
    assert summary["success"] and summary["turns"] == 3 and len(sim.chunks) == 2
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert "16 rows of 12" in trace[1]["error"] and trace[0]["commands"][0].startswith(
        "chunk 16 steps"
    )
    assert (
        "RESULT OF YOUR LAST CHUNK"
        in (tmp_path / "calls" / "turn003-q0" / "prompt.txt").read_text()
    )
