"""Reject guessed playback alignment and preserve exact original model usage."""

import json

import pytest
from robocasa_common.visualization_library import backfill_timeline


def make_legacy(tmp_path, action):
    """Build a legacy observation with one query and a verified two-step action."""
    folder = tmp_path / "grid" / "OpenCabinet-123"
    log = folder / "eval/actions/run/scene.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(json.dumps({"t": t, "action": [0.2]}) for t in (0, 1)))
    for i, queries in enumerate(([{"kind": "grid"}], [])):
        call = folder / "policy" / f"call-{i:05d}"
        call.mkdir(parents=True)
        (call / "prompt.txt").write_text("observation_id=OpenCabinet-123:step-0.")
        (call / "response.json").write_text(
            json.dumps({"action": [action], "repeat": 2, "queries": queries, "reason": "test"})
        )
        (call / "receipt.json").write_text(json.dumps({"usage": None}))
    return folder


def test_legacy_query_stays_at_same_step_and_unknown_usage_is_preserved(tmp_path):
    """Query waiting consumes no native steps and no token count is invented."""
    data = backfill_timeline(make_legacy(tmp_path, 0.2))
    assert [r["step"] for r in data["calls"]] == [0, 0]
    assert data["native_actions_verified"] == 2
    assert all(r["usage"] is None for r in data["calls"])


def test_legacy_alignment_refuses_different_applied_action(tmp_path):
    """A response that disagrees with native actions must not be linked to playback."""
    with pytest.raises(ValueError, match="diverges"):
        backfill_timeline(make_legacy(tmp_path, 0.9))


def test_legacy_alignment_requires_explicit_observation_step(tmp_path):
    """Never infer observation IDs from call counts or response durations."""
    folder = make_legacy(tmp_path, 0.2)
    (folder / "policy/call-00000/prompt.txt").write_text("no recorded observation")
    with pytest.raises(ValueError, match="not recorded explicitly"):
        backfill_timeline(folder)
