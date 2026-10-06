"""Verify throughput accounting uses independent completed receipts and wall time."""

import json

import run_parallel_queue as queue


def test_snapshot_excludes_old_and_partial_receipts(tmp_path, monkeypatch):
    """Count action repeats once per completed run without counting pending model calls."""
    first = tmp_path / "runs" / "first" / "inference" / "call-0000"
    second = tmp_path / "runs" / "second" / "inference" / "call-0000"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "receipt.json").write_text(json.dumps({"seconds": 10, "response": {"repeat": 4}}))
    (second / "receipt.json").write_text("{")
    monkeypatch.setattr(queue.time, "time", lambda: 20)
    result = queue.snapshot(tmp_path, ["first", "second"], 0)
    assert result["calls"] == 1
    assert result["steps"] == 4
    assert result["aggregate_steps_per_second"] == 0.2
    assert result["per_run"]["second"]["calls"] == 0
    assert queue.snapshot(tmp_path, ["first"], 99999999999)["calls"] == 0


def test_missing_process_is_finished():
    """A missing PID cannot be mistaken for a running rollout."""
    assert not queue.alive(999999999)
