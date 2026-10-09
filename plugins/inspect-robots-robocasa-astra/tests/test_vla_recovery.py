"""Recovery contracts, condition isolation and original per-attempt accounting."""

import subprocess
import sys
import time

import numpy as np
import pytest
from robocasa_astra.astra_robodawn.codex import ModelCallError
from robocasa_astra.astra_vla.caller import DurableCaller
from robocasa_astra.astra_vla.persistence import (
    atomic_json,
    exact_process,
    read_json,
)
from robocasa_astra.astra_vla.recovery import ActionJournal


class FakeNative:
    """Deterministic physics with the same write-ahead and ack hooks as the real executor."""

    def __init__(self, journal):
        self.env = self
        self.sim = self
        self.robots = []
        self.steps_used = 0
        self.success = False
        self.gripper_cmd = -1.0
        self._split = {"right_gripper": (0, 1)}
        self.position = 0.0
        self.journal = journal

    def get_state(self):
        """Expose deterministic mock physics for exact checkpoint verification."""
        return np.array([self.position])

    def _get_observations(self, **kwargs):
        return {"camera_image": np.array([self.position])}

    def _check_success(self):
        return self.success

    def _step(self, action, caption):
        self.position += action[1]
        self.steps_used += 1
        self.journal.commit(self, self, action)


def test_native_ack_replay_and_pending_once(tmp_path):
    """Replay acknowledged actions and apply a pending action exactly once."""
    identity = {"scene": "frozen", "seed": 10}
    journal = ActionJournal(tmp_path, identity)
    original = FakeNative(journal)
    journal.restore(original, original)
    journal.begin_chunk("turn1", [[-1, 1]] * 8, 1)
    for _ in range(4):
        journal.intent([-1, 1])
        original._step(np.array([-1, 1]), "original")
    journal.intent([-1, 1])  # worker died before durable ack of the fifth action
    restored_journal = ActionJournal(tmp_path, identity)
    resumed = FakeNative(restored_journal)
    restored_journal.restore(resumed, resumed)
    assert resumed.position == resumed.steps_used == 5
    assert restored_journal.data["chunks"]["turn1"]["cursor"] == 5
    assert restored_journal.data["pending"] is None
    again_journal = ActionJournal(tmp_path, identity)
    again = FakeNative(again_journal)
    again_journal.restore(again, again)
    assert again.position == 5  # repeated recovery cannot execute pending a second time
    with pytest.raises(ValueError, match="identity"):
        ActionJournal(tmp_path, {"scene": "different"})
    resumed.position += 1
    changed = FakeNative(restored_journal)
    changed.position = 99
    with pytest.raises(ValueError, match="drift"):
        restored_journal.restore(changed, changed)
    with pytest.raises(ValueError, match="different actions"):
        restored_journal.begin_chunk("turn1", [[-1, 2]], 1)


def test_durable_completed_response_counts_retry_usage_once(tmp_path):
    """Include billed retries and preserve unknown usage in the final receipt."""
    system, schema = tmp_path / "system.md", tmp_path / "schema.json"
    system.write_text("instructions")
    schema.write_text("{}")
    caller = DurableCaller("unused", "unused", system, schema, tmp_path, tmp_path)
    folder = tmp_path / "calls" / "turn001-q0"
    folder.mkdir(parents=True)
    atomic_json(
        folder / "result.json",
        {
            "text": "{}",
            "reasoning": [],
            "receipts": [
                {"usage": {"input_tokens": 10}, "seconds": 15},
                {"usage": None, "seconds": 30},
                {"usage": {"input_tokens": 20, "output_tokens": 5}, "seconds": 10},
            ],
        },
    )
    for _ in range(2):
        reply = caller.call([{"type": "text", "text": "same observation"}], folder)
        assert reply.usage["input_tokens"] == 30
        assert reply.seconds == 55
    assert read_json(folder / "call.json")["unknown_usage_attempts"] == 1
    with pytest.raises(ModelCallError, match="immutable"):
        caller.call([{"type": "text", "text": "changed"}], folder)


def test_detached_runner_survives_host_exit(tmp_path):
    """The model runner finishes after host termination and is adopted without duplication."""
    fake = tmp_path / "fake-codex"
    fake.write_text("""#!/usr/bin/env python3
import json,sys,time
for line in sys.stdin:
 r=json.loads(line);m=r.get("method");p=r.get("params",{})
 if "id" not in r: continue
 if m=="account/read": result={"account":{"type":"chatgpt"}}
 elif m=="thread/start": result={"thread":{"id":"thread1"},"model":"gpt-6-astra"}
 elif m=="turn/start": result={"turn":{"id":"turn1"}}
 else: result={}
 print(json.dumps({"id":r["id"],"result":result}),flush=True)
 if m=="turn/start":
  time.sleep(2)
  print(json.dumps({"method":"thread/tokenUsage/updated","params":{"threadId":"thread1","turnId":"turn1","tokenUsage":{"last":{"inputTokens":123,"outputTokens":4}}}}),flush=True)
  print(json.dumps({"method":"item/completed","params":{"threadId":"thread1","item":{"type":"agentMessage","text":"{}","phase":"final_answer"}}}),flush=True)
  print(json.dumps({"method":"turn/completed","params":{"threadId":"thread1","turn":{"id":"turn1","status":"completed"}}}),flush=True)
""")
    fake.chmod(0o700)
    system, schema = tmp_path / "system.md", tmp_path / "schema.json"
    system.write_text("instructions")
    schema.write_text("{}")
    folder = tmp_path / "calls" / "turn001-q0"
    script = (
        "from pathlib import Path; from robocasa_astra.astra_vla.caller import DurableCaller; "
        f"DurableCaller({str(fake)!r}, 'unused', Path({str(system)!r}), Path({str(schema)!r}), "
        f"Path({str(tmp_path)!r}), Path({str(tmp_path)!r})).call([], Path({str(folder)!r}))"
    )
    host = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 10
        while not (folder / "appserver-lease.json").exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        host.terminate()
        host.wait(timeout=5)
        runner = read_json(folder / "lease.json")
        assert exact_process(runner)
        reply = DurableCaller(str(fake), "unused", system, schema, tmp_path, tmp_path).call(
            [], folder
        )
        assert reply.usage["input_tokens"] == 123
        assert len(reply.attempts) == 1
        ledger = (tmp_path / "policy" / "attempts.jsonl").read_text().splitlines()
        assert len(ledger) == 1
    finally:
        if host.poll() is None:
            host.terminate()
            host.wait(timeout=5)


def test_attempt_budget_and_capacity_receipts(tmp_path, monkeypatch):
    """Actual failures consume the global attempt budget and preserve unknown usage."""
    from robocasa_astra.astra_vla import call_runner

    folder = tmp_path / "calls" / "turn001-q0"
    folder.mkdir(parents=True)
    system, schema = tmp_path / "system.md", tmp_path / "schema.json"
    system.write_text("instructions")
    schema.write_text("{}")
    atomic_json(
        folder / "request.json",
        {
            "executable": "unused",
            "codex_home": "unused",
            "system": str(system),
            "schema": str(schema),
            "workdir": str(tmp_path),
            "run_dir": str(tmp_path),
            "model": "gpt-6-astra",
            "effort": "medium",
            "parts": [],
            "attempt_budget": 2,
        },
    )
    waits = []
    monkeypatch.setattr(call_runner.time, "sleep", waits.append)

    class Caller:
        """A provider capacity failure fixture with no reported usage."""

        def __init__(self, *args, **kwargs):
            assert kwargs["call_timeout"] is None
            assert kwargs["effort"] == "medium"

        def _turn(self, *args):
            return {"status": "failed", "errors": ["Selected model is at capacity"], "usage": {}}

        def close(self):
            """No subprocess exists in this fixture."""

    monkeypatch.setattr(call_runner, "RecordedCaller", Caller)
    call_runner.run_request(folder)
    result = read_json(folder / "result.json")
    assert "budget exhausted" in result["error"]
    assert len(result["receipts"]) == 2
    assert all(r["usage"] is None for r in result["receipts"])
    assert waits == [15, 15]
