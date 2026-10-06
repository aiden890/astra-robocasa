"""Test supervisor publication failure handling and native rollout invariants."""

import json
import os
from pathlib import Path

import pytest
from astra_ops.common import runtime_io
from astra_ops.queue.multitask import build_run_command


def test_atomic_failure_preserves_previous_snapshot(tmp_path, monkeypatch):
    """Readers retain the last complete state when replacement fails."""
    target = tmp_path / "status.json"
    target.write_text('{"previous": true}')

    def fail_replace(source, destination):
        raise OSError("disk write failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk write failure"):
        runtime_io.write_json_atomic(target, {"next": True})
    assert json.loads(target.read_text()) == {"previous": True}
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_snapshot_retains_unicode(tmp_path):
    """Status labels survive round-trip publication without partial temporary files."""
    target = tmp_path / "status.json"
    runtime_io.write_json_atomic(target, {"status": "진행 중"})
    assert json.loads(target.read_text()) == {"status": "진행 중"}
    assert list(tmp_path.iterdir()) == [target]


def test_process_exit_race_is_not_active(monkeypatch):
    """A process exiting during inspection cannot stop queue adoption."""

    def disappeared(self):
        raise ProcessLookupError("exited")

    monkeypatch.setattr(Path, "read_text", disappeared)
    assert not runtime_io.alive(123)
    assert not runtime_io.alive(-1)


@pytest.mark.parametrize("robot", ["PandaOmron", "GR1FloatingBody"])
def test_native_command_preserves_task_settings(robot):
    """Both robots use the original horizon and scene; GR1 retains heading alignment."""
    command = build_run_command(
        {"task": "PanTransfer", "robot": robot, "max_steps": 1800, "seed": 781101},
        "test-container",
        Path("/runs/test"),
    )
    assert command[command.index("--steps") + 1] == "1800"
    assert command[command.index("--seed") + 1] == "781101"
    assert command[command.index("--task") + 1] == "PanTransfer"
    assert "--native-scene" in command
    assert ("--face-workstation" in command) == (robot == "GR1FloatingBody")
