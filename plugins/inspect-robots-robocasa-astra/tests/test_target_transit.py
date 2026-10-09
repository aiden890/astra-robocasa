"""Destination semantics, physical feedback and interrupted native servo recovery."""

from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_astra.astra_robodawn.executor import Pose
from robocasa_astra.astra_vla.recovery import ActionJournal
from robocasa_astra.astra_vla.sim_server import ChunkServer


def target(x=0.05):
    """World destination in metres, not a relative offset."""
    return {"position_world_m": [x, 0, 0], "observation_id": "obs0"}


def server(tmp_path, monkeypatch, *, frozen=False, budget=1000, interrupt=None, success_at=None):
    """Deterministic physical stepping with the real per-action journal."""
    import robocasa_astra.astra_vla.recovery as recovery

    obj = ChunkServer.__new__(ChunkServer)
    obj.motion_control = "dual"
    obj.observation_id = "obs0"
    obj.last_observation = {"state": {"steps_used": 0}}
    obj.journal = ActionJournal(tmp_path, {})
    obj.replay = SimpleNamespace(mark=lambda *args: None)
    e = obj.executor = SimpleNamespace(
        steps_used=0,
        success=False,
        gripper_cmd=-1,
        before_step=obj.journal.intent,
        _mode_index=11,
        _split={"right": (0, 6), "right_gripper": (6, 7), "base": (7, 10), "torso": (10, 11)},
    )
    xyz = np.zeros(3)
    e.pose = lambda: Pose(xyz.copy(), np.eye(3), np.zeros(3), np.eye(3))
    e.state = lambda: {"steps_used": e.steps_used}
    e._stop = lambda: (
        "task_success" if e.success else "budget_exhausted" if e.steps_used >= budget else None
    )

    def arm(goal, rotation, pose):
        action = np.zeros(12)
        action[:3] = np.clip((goal - pose.tip) * 20, -0.5, 0.5)
        action[6] = e.gripper_cmd
        action[11] = -1
        return action

    def step(action, caption):
        if not frozen:
            xyz[:] += np.asarray(action[:3]) * 0.02
        e.steps_used += 1
        e.success = success_at is not None and e.steps_used >= success_at
        obj.journal.commit(None, e, action)
        if interrupt is not None and e.steps_used == interrupt and not obj.journal.replaying:
            raise KeyboardInterrupt

    e._arm_action, e._step = arm, step
    monkeypatch.setattr(
        recovery, "native_digest", lambda env, ex: [*np.round(ex.pose().tip, 10), ex.steps_used]
    )
    return obj


def test_destination_reached_and_lost_reply_is_idempotent(tmp_path, monkeypatch):
    """Feedback determines arrival; replaying a request cannot move the arm twice."""
    s = server(tmp_path, monkeypatch)
    result = s.act_chunk(None, [], 1, "t1", "transit", target())
    assert result["target_reached"] and result["position_error_m"] <= 0.01
    assert 0 < result["steps"] < 160
    assert len(result["executed_actions"]) == result["steps"]
    steps = s.executor.steps_used
    assert s.act_chunk(None, [], 1, "t1", "transit", target()) == result
    assert s.executor.steps_used == steps
    with pytest.raises(ValueError, match="reused"):
        s.act_chunk(None, [], 1, "t1", "transit", target(0.1))


def test_stale_observation_rejected_before_motion(tmp_path, monkeypatch):
    """A new target cannot be attached to an older physics state."""
    s = server(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="stale"):
        s.act_chunk(None, [], 1, "t1", "transit", {**target(), "observation_id": "old"})
    assert s.executor.steps_used == 0


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"frozen": True}, "stalled"),
        ({"budget": 2}, "budget_exhausted"),
        ({"success_at": 2}, "task_success"),
    ],
)
def test_unreached_target_is_not_reported_as_arrival(tmp_path, monkeypatch, kwargs, reason):
    """Contact-like stalls, episode budget and native success retain truthful arrival flags."""
    s = server(tmp_path, monkeypatch, **kwargs)
    r = s.act_chunk(None, [], 1, "t1", "transit", target(1.0))
    assert r["stop_reason"] == reason and not r["target_reached"]
    assert r["task_success"] == (reason == "task_success")
    assert r["steps"] == len(r["executed_actions"])


def test_pending_native_action_replayed_once_then_target_continues(tmp_path, monkeypatch):
    """A crash between physics and ACK preserves and applies the pending step exactly once."""
    s = server(tmp_path, monkeypatch)
    step = s.executor._step

    def die_before_ack(action, caption):
        if s.executor.steps_used == 2:
            raise RuntimeError("simulator disconnected before acknowledgement")
        step(action, caption)

    s.executor._step = die_before_ack
    with pytest.raises(RuntimeError, match="disconnected"):
        s.act_chunk(None, [], 1, "t1", "transit", target(0.1))
    assert s.journal.data["pending"] is not None
    s.executor._step = step
    resumed = server(tmp_path, monkeypatch)
    resumed.journal.restore(None, resumed.executor)
    assert resumed.executor.steps_used == 3
    resumed.journal.data["faulted"] = False
    result = resumed.act_chunk(None, [], 1, "t1", "transit", target(0.1))
    uninterrupted = server(tmp_path / "reference", monkeypatch)
    reference = uninterrupted.act_chunk(None, [], 1, "t1", "transit", target(0.1))
    assert result["resumed_rows"] == 3
    assert result["executed_actions"] == reference["executed_actions"]


def test_interrupted_stall_preserves_original_no_progress_budget(tmp_path, monkeypatch):
    """Repeated reconnects cannot extend a stalled target's step allowance."""
    s = server(tmp_path, monkeypatch, frozen=True, interrupt=3)
    with pytest.raises(KeyboardInterrupt):
        s.act_chunk(None, [], 1, "t1", "transit", target(1.0))
    resumed = server(tmp_path, monkeypatch, frozen=True)
    resumed.journal.restore(None, resumed.executor)
    resumed.journal.data["faulted"] = False
    r = resumed.act_chunk(None, [], 1, "t1", "transit", target(1.0))
    assert r["stop_reason"] == "stalled" and r["steps"] == 20
