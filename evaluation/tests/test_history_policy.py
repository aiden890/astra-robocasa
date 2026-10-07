"""Check exact native-time histories, condition isolation, and resumed snapshots."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_common.depth_policy import DepthPolicy
from robocasa_common.history_policy import HistoryPolicy, history_steps
from robocasa_common.paired_live_policy import PairedLivePolicy


@pytest.mark.parametrize("seconds", range(6))
def test_exact_offsets(seconds):
    """Each interval is twenty native steps; absent early observations stay absent."""
    assert history_steps(111, seconds) == [111 - 20 * i for i in range(seconds + 1)]
    assert history_steps(19, seconds) == [19]


@pytest.mark.parametrize("condition,maximum", [("rgb", 18), ("pixel", 18), ("color", 36)])
def test_inputs_and_restart(tmp_path, monkeypatch, condition, maximum):
    """Saved history survives policy replacement and never exposes unrequested depth."""
    monkeypatch.setattr(PairedLivePolicy, "observe", lambda *args: None)
    policy = HistoryPolicy.__new__(HistoryPolicy)
    policy.history_seconds, policy.condition, policy.scene = 5, condition, "scene"
    policy.snapshots = tmp_path / "history"
    policy.snapshots.mkdir()
    policy.output = tmp_path / "policy"
    policy.output.mkdir()
    for step in range(112):
        policy.embodiment = SimpleNamespace(
            current_depth_step=step,
            current_depth_arrays={c: np.full((256, 256), 0.2, np.float32) for c in "abc"},
        )
        policy.observe(
            SimpleNamespace(
                images={c: np.full((256, 256, 3), step, np.uint8) for c in "abc"},
                extra={"steps": step},
            )
        )
    assert len(list(policy.snapshots.glob("step-*"))) == 101
    source = policy.snapshots / "step-0000091"
    (source / "query-answers.json").write_text(
        json.dumps([{"observation_id": "scene:step-91", "pixel_depth_m": 0.2}])
    )
    replacement = HistoryPolicy.__new__(HistoryPolicy)
    replacement.__dict__.update(policy.__dict__)
    folder = tmp_path / "call"
    folder.mkdir()
    current = json.loads((policy.snapshots / "step-0000111/observation.json").read_text())
    images = [policy.snapshots / "step-0000111" / n for n in current["files"]]
    captured = {}

    def call(self, prompt, images, folder):
        captured.update(prompt=prompt, images=list(images))
        return {"action": []}

    monkeypatch.setattr(DepthPolicy, "call", call)
    replacement.call("current prompt", images, folder)
    manifest = json.loads((folder / "history-inputs.json").read_text())
    assert manifest["image_count"] == len(captured["images"]) == maximum
    assert [x["step"] for x in manifest["observations"]] == [111, 91, 71, 51, 31, 11]
    assert [x["relative_seconds"] for x in manifest["observations"]] == [0, -1, -2, -3, -4, -5]
    assert ("actually_observed_query_answers" in captured["prompt"]) == (condition == "pixel")
    if condition != "color":
        assert not list(folder.glob("*__depth.png"))
        assert not list(policy.snapshots.glob("*/*__depth.png"))


def test_missing_history_fails_closed(tmp_path, monkeypatch):
    """A damaged recovery window cannot silently substitute nearby observations."""
    monkeypatch.setattr(DepthPolicy, "call", lambda *args: pytest.fail("model called"))
    policy = HistoryPolicy.__new__(HistoryPolicy)
    policy.snapshot_step, policy.history_seconds = 20, 1
    policy.snapshots = tmp_path
    with pytest.raises(FileNotFoundError):
        policy.call("prompt", [], tmp_path)
