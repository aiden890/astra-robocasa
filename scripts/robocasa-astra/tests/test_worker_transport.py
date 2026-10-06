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
