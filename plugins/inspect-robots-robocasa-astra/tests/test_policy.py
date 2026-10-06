"""Verify malformed actions never reach a native simulator."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_astra.policy import CodexPolicy

from inspect_robots.spaces import Box
from inspect_robots.types import Observation


def policy(tmp_path):
    """Build an isolated three-dimensional fixture without model access."""
    embodiment = SimpleNamespace(
        info=SimpleNamespace(
            action_space=Box(shape=(3,), low=-np.ones(3), high=np.ones(3)), docs="test fixture"
        )
    )
    return CodexPolicy(embodiment, tmp_path, "unused", str(tmp_path), noop=True)


@pytest.mark.parametrize("action", [[0, 0], [0, 0, 2], [0, float("nan"), 0]])
def test_invalid_action_rejected(tmp_path, action):
    """Enforce finite native dimensions and actual bounds."""
    with pytest.raises(ValueError):
        policy(tmp_path).validate({"action": action, "repeat": 1})


@pytest.mark.parametrize("repeat", [0, 9, True, 1.5])
def test_invalid_repeat_rejected(tmp_path, repeat):
    """Keep open-loop movement to at most eight actual physics steps."""
    with pytest.raises(ValueError):
        policy(tmp_path).validate({"action": [0, 0, 0], "repeat": repeat})


def test_noop_records_real_frames(tmp_path):
    """The no-model mode records observation files and identifies itself accurately."""
    p = policy(tmp_path)
    p.reset(SimpleNamespace(instruction="test"))
    result = p.act(
        Observation(images={"test": np.zeros((8, 8, 3), np.uint8)}, state={"qpos": np.zeros(3)})
    )
    assert len(result.actions) == 1
    assert (tmp_path / "call-0000/test.png").is_file()
    assert "false" in (tmp_path / "call-0000/receipt.json").read_text()


def test_codex_uses_subscription_and_tool_isolation(tmp_path, monkeypatch):
    """Strip API keys and constrain model calls to isolated structured output."""
    import json

    p = policy(tmp_path / "calls")
    p.noop = False
    p.home = str(tmp_path / "auth")
    Path(p.home).mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("CODEX_API_KEY", "test-placeholder")
    captured = {}

    def execute(command, **kwargs):
        captured.update(command=command, **kwargs)
        output = Path(command[command.index("-o") + 1])
        output.write_text(json.dumps({"action": [0, 0, 0], "repeat": 2, "reason": "fixture"}))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("robocasa_astra.policy.subprocess.run", execute)
    p.reset(SimpleNamespace(instruction="test"))
    chunk = p.act(Observation())
    assert len(chunk) == 2
    assert "OPENAI_API_KEY" not in captured["env"]
    assert "CODEX_API_KEY" not in captured["env"]
    assert captured["command"][captured["command"].index("-m") + 1] == "gpt-6-astra"
    assert "model_reasoning_effort=\"low\"" in captured["command"]
    assert "features.shell_tool=false" in captured["command"]
    assert "features.plugins=false" in captured["command"]
    assert "project_doc_max_bytes=0" in captured["command"]
    assert "--output-schema" in captured["command"]
    assert captured["cwd"].name == "inference-empty"


@pytest.mark.parametrize("failure", ["capacity", "timeout"])
def test_capacity_retry_keeps_observation(tmp_path, monkeypatch, failure):
    """Retry provider capacity without producing an action or changing the prompt."""
    import json

    p = policy(tmp_path / "calls")
    p.noop = False
    p.home = str(tmp_path / "auth")
    prompts = []
    sleeps = []

    def execute(command, **kwargs):
        prompts.append(kwargs["input"])
        if len(prompts) == 1:
            if failure == "timeout":
                import subprocess

                raise subprocess.TimeoutExpired(command, 180)
            kwargs["stdout"].write("ERROR: Selected model is at capacity")
            return SimpleNamespace(returncode=1)
        Path(command[command.index("-o") + 1]).write_text(
            json.dumps({"action": [0, 0, 0], "repeat": 2, "reason": "recovered"})
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("robocasa_astra.policy.subprocess.run", execute)
    monkeypatch.setattr("robocasa_astra.policy.time.sleep", sleeps.append)
    p.reset(SimpleNamespace(instruction="test"))
    chunk = p.act(Observation())
    assert len(chunk) == 2
    assert len(prompts) == 2 and prompts[0] == prompts[1]
    assert sleeps == [15]
    assert len(p.history) == 1
