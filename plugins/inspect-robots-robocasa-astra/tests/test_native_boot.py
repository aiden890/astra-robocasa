"""Verify native tasks start with the exact seed later used by official evaluation."""

import json
from types import SimpleNamespace

from robocasa_astra import run

from inspect_robots.rollout import derive_seed


def test_native_boot_matches_eval_seed(tmp_path, monkeypatch):
    """Avoid rebuilding a different randomized world before the first model call."""
    captured = {}

    class Environment:
        """Capture construction without launching a simulator."""

        def __init__(self, command, seed, output):
            captured.update(command=command, seed=seed)
            self.info = SimpleNamespace(
                name="robocasa-GR1FloatingBody",
                docs=json.dumps({"instruction": "Turn off the rear burner"}),
            )

        def close(self):
            """Record normal resource release."""
            captured["closed"] = True

    def evaluate(task, policy, env, **kwargs):
        assert captured["seed"] == derive_seed(0, task.scenes[0].init_seed, 0)
        assert task.scenes[0].instruction == "Turn off the rear burner"
        assert task.max_steps == 750
        return [
            SimpleNamespace(
                status="success", results=SimpleNamespace(metrics={"success_at_end": 0})
            )
        ]

    monkeypatch.setattr(run, "SparkEmbodiment", Environment)
    monkeypatch.setattr(run, "CodexPolicy", lambda *args: object())
    monkeypatch.setattr(run, "eval", evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--native-scene",
            "--face-workstation",
            "--robot",
            "GR1FloatingBody",
            "--task",
            "TurnOffStove",
            "--steps",
            "750",
            "--seed",
            "781107",
            "--output",
            str(tmp_path / "run"),
            "--codex",
            "unused",
            "--codex-home",
            "unused",
        ],
    )
    run.main()
    assert captured["closed"]
    assert "--face-workstation" in captured["command"]
    assert captured["command"][captured["command"].index("--horizon") + 1] == "750"
