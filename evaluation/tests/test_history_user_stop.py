"""Keep user-assessed stops terminal without claiming native evaluation success."""

import json
from pathlib import Path

import pytest
from robocasa_common import history_study


def test_user_failed_is_terminal_and_not_native(tmp_path, monkeypatch):
    """Restarting the manager preserves stopped evidence and never redispatches it."""
    root = tmp_path / "study"
    root.mkdir()
    result = {"execution_status": "user_stopped", "task_success": None, "user_assessment": "failed"}
    folder = root / "results/rgb-h0/scene"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(result))
    site = tmp_path / "site"
    (site / "media").mkdir(parents=True)
    (site / "media/supplemental-catalog.json").write_text("[]")
    real_path = Path
    monkeypatch.setattr(
        history_study,
        "Path",
        lambda p: site if str(p).endswith("astra-robocasa/status-page") else real_path(p),
    )
    monkeypatch.setattr(history_study, "publish", lambda *args: None)
    monkeypatch.setattr(
        history_study,
        "resource_sample",
        lambda: {"available_gib": 100, "lab_available_gib": 30, "load1": 0, "cpus": 20},
    )
    monkeypatch.setattr(
        history_study.subprocess, "Popen", lambda *a, **k: pytest.fail("redispatched")
    )
    monkeypatch.setattr(history_study.time, "sleep", lambda *a: pytest.fail("not terminal"))
    history_study.supervise(
        root,
        {"jobs": [{"id": "rgb-h0", "namespace": "rgb-h0", "scene": "scene"}], "containers": []},
    )
    status = json.loads((root / "status.json").read_text())
    assert status["normal_results"] == 0
    assert status["assessed_results"] == 1
    assert status["user_assessed_failures"] == {"rgb-h0": result}
    assert status["pending"] == 0 and status["active"] == []
    assert json.loads((folder / "result.json").read_text()) == result
