"""Require branch-local depth code rather than a stale external worker file."""

import io
import tarfile
from types import SimpleNamespace

from astra_ops.common import worker_transport


def test_complete_package_is_staged(monkeypatch):
    """A new idle slot receives depth utilities together with its worker and exact path."""
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(worker_transport.subprocess, "run", execute)
    assert worker_transport.stage_worker("test-container", 781101)
    assert calls[0][0][-1] == "/tmp/astra-depth-781101"
    with tarfile.open(fileobj=io.BytesIO(calls[1][1]["input"])) as bundle:
        names = bundle.getnames()
    assert "robocasa_astra/worker.py" in names
    assert "robocasa_astra/depth.py" in names
    assert all(name.endswith(".py") for name in names)


def test_branch_worker_process_blocks_container_lease(monkeypatch):
    """A worker launched by file path must keep its container unavailable."""
    from astra_ops.queue import multitask

    monkeypatch.setattr(
        multitask.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="python3 /tmp/astra-depth-786601/robocasa_astra/worker.py --depth",
        ),
    )
    assert not multitask.container_idle("test-container")


def test_frozen_scene_keeps_worker_and_pythonpath_overrides(tmp_path):
    """A frozen scene can use the mounted worker while retaining branch-local transport."""
    from astra_ops.queue.multitask import build_run_command

    job = {
        "task": "PrepareCoffee",
        "robot": "PandaOmron",
        "seed": 8806552,
        "max_steps": 1800,
        "frozen_scene": "/scene-bundles/scenes/PrepareCoffee-8806552",
        "frozen_worker_path": "/frozen-code/robocasa_astra/worker.py",
        "frozen_worker_pythonpath": "/frozen-code:/astra/src",
    }
    command = build_run_command(job, "test-container", tmp_path)
    assert command[command.index("--worker-script") + 1] == job["frozen_worker_path"]
    assert command[command.index("--worker-pythonpath") + 1] == job["frozen_worker_pythonpath"]
    assert command[command.index("--frozen-scene") + 1] == job["frozen_scene"]
