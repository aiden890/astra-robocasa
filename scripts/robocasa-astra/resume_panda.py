"""Recover an interrupted trial only after matching the saved observation."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from robocasa_astra.bridge import SparkEmbodiment
from robocasa_astra.policy import CodexPolicy

from inspect_robots import eval
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "runs/panda-coffee-1800-002"
OUT = ROOT / "runs/panda-coffee-1800-002-resume-001"


def main():
    """Replay original actions without model calls, then evaluate the remaining budget."""
    OUT.mkdir(exist_ok=False)
    receipts = [
        json.loads(f.read_text()) for f in sorted((OLD / "inference").glob("call-*/receipt.json"))
    ]
    actions = [r["response"]["action"] for r in receipts for _ in range(r["response"]["repeat"])]
    assert len(actions) == 1791
    command = [
        "ssh",
        "spark2",
        "docker",
        "exec",
        "-i",
        "-e",
        "PYTHONPATH=/astra/src:/astra/plugins/inspect-robots-robocasa-astra/src",
        "astra-robocasa-parallel-03-20261006",
        "python3",
        "/tmp/resume-worker.py",
        "--robot",
        "PandaOmron",
        "--task",
        "PrepareCoffee",
        "--fixture",
        "/fixtures/episode-005",
    ]
    env = None
    started = time.time()
    try:
        env = SparkEmbodiment(command, 771003, OUT / "worker")
        for offset in range(0, len(actions), 64):
            env.latest = env.call(op="replay", actions=actions[offset : offset + 64])
            (OUT / "recovery-progress.json").write_text(
                json.dumps(
                    {
                        "replayed_steps": env.latest["info"]["steps"],
                        "target": 1791,
                        "seconds": time.time() - started,
                    }
                )
            )
        saved = OLD / "inference/call-0487"
        prompt = (saved / "prompt.txt").read_text()
        expected = json.loads(
            prompt.split("\nCurrent state: ", 1)[1].split("\nRecent actions: ", 1)[0]
        )
        actual = env.latest["state"]
        errors = {
            k: float(np.max(np.abs(np.asarray(v) - np.asarray(actual[k]))))
            for k, v in expected.items()
        }
        image_equal = {}
        obs = env.observation(env.latest)
        for k, img in obs.images.items():
            Image.fromarray(img).save(OUT / (k + "-restored.png"))
            image_equal[k] = bool(np.array_equal(img, np.asarray(Image.open(saved / (k + ".png")))))
        passed = max(errors.values()) <= 1e-9 and all(image_equal.values())
        (OUT / "recovery-verification.json").write_text(
            json.dumps(
                {
                    "passed": passed,
                    "state_max_error": max(errors.values()),
                    "state_errors": errors,
                    "images_equal": image_equal,
                    "original_run": OLD.name,
                    "original_steps": 1791,
                    "remaining_steps": 9,
                    "seed": 771003,
                },
                indent=2,
            )
        )
        if not passed:
            raise RuntimeError(
                "Restored observation did not match; no new model call/action permitted"
            )
        (OUT / "environment.json").write_text((OLD / "environment.json").read_text())

        class ContinuedPolicy(CodexPolicy):
            """Retain the original last actions when the evaluation resets policy context."""

            def reset(self, scene):
                super().reset(scene)
                self.history = [r["response"] for r in receipts][-8:]

        policy = ContinuedPolicy(
            env,
            OUT / "inference",
            str(
                ROOT
                / ".runtime/codex/node_modules/@openai/codex-linux-x64/vendor"
                / "x86_64-unknown-linux-musl/bin/codex"
            ),
            str(ROOT / ".runtime/auth"),
        )
        instruction = (
            "Pick the mug from the cabinet, place it under the dispenser, "
            "release it, and press the coffee start button."
        )
        task = Task(
            name="PrepareCoffee-resumed",
            scenes=[
                Scene(id="PandaOmron-771003-resumed", instruction=instruction, init_seed=771003)
            ],
            scorer=success_at_end(),
            max_steps=9,
        )
        logs = eval(task, policy, env, log_dir=str(OUT / "eval"), store_frames=True)
        (OUT / "continuation-result.json").write_text(
            json.dumps(
                {
                    "original_steps": 1791,
                    "remaining_budget": 9,
                    "status": logs[0].status,
                    "metrics": logs[0].results.metrics,
                },
                indent=2,
            )
        )
        print("continuation", logs[0].status, logs[0].results.metrics, flush=True)
    finally:
        if env:
            env.close()


if __name__ == "__main__":
    main()
