"""Recovery contracts, condition isolation and original per-attempt accounting."""

import json
import os
import queue
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_astra.astra_robodawn.appserver import AppServerCaller
from robocasa_astra.astra_robodawn.codex import CallResult, ModelCallError
from robocasa_astra.astra_robodawn.loop import EpisodeConfig
from robocasa_astra.astra_vla.caller import DurableCaller
from robocasa_astra.astra_vla.depth_input import answer_queries, extra_parts, response_schema
from robocasa_astra.astra_vla.loop import run_episode
from robocasa_astra.astra_vla.persistence import (
    atomic_json,
    exact_process,
    process_lease,
    read_json,
)
from robocasa_astra.astra_vla.prompts import vla_demos
from robocasa_astra.astra_vla.recovery import ActionJournal


def test_examples_are_independent():
    """Zero-shot contains neither primer nor successful demonstrations."""
    assert vla_demos("OpenCabinet", 0) == []
    assert [d.data["kind"] for d in vla_demos("OpenCabinet", 1)] == ["task"]
    assert [d.data["kind"] for d in vla_demos("OpenCabinet", 0, primer=True)] == ["primer"]


def test_depth_conditions_are_isolated(tmp_path):
    """Enabled depth queries require the exact RGB observation and coordinates."""
    import jsonschema

    assert extra_parts({}, tmp_path, "rgb") == []
    action = {"scene": "s", "progress": "p", "memory": "m", "plan": "p", "actions": [[0] * 12] * 16}
    query = {
        "observation_id": "obs1",
        "camera": "wrist",
        "kind": "pixel",
        "u": 1,
        "v": 2,
        "radius": 0,
    }
    depth = np.arange(1, 17, dtype=float).reshape(4, 4)
    answers = answer_queries({"wrist": depth}, [query], "obs1", "pixel")
    assert answers[0]["pixel_depth_m"] == 10
    assert answers[0]["pixel_uv"] == [1, 2]
    query_reply = {k: v for k, v in action.items() if k != "actions"} | {"queries": [query]}
    jsonschema.validate(query_reply, response_schema("pixel"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(query_reply, response_schema("rgb"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(action | {"queries": [query]}, response_schema("pixel"))
    with pytest.raises(ValueError, match="Stale"):
        answer_queries({"wrist": depth}, [query], "obs2", "pixel")
    with pytest.raises(ValueError, match="unavailable"):
        answer_queries({"wrist": depth}, [query], "obs1", "rgb")
    grid = {"observation_id": "obs1", "camera": "wrist", "kind": "grid", "cell_id": "r15c15"}
    grid_answer = answer_queries({"wrist": np.ones((256, 256))}, [grid], "obs1", "grid")[0]
    assert grid_answer["region_bounds_uv_half_open"] == [240, 240, 256, 256]
    assert grid_answer["pixel_uv"] == [247, 247]


def test_wait_has_no_model_deadline():
    """Indefinite model waits still detect an actual app-server exit."""
    caller = AppServerCaller.__new__(AppServerCaller)
    caller.incoming = queue.Queue()
    caller.incoming.put({"method": "test"})
    caller.log = None
    assert caller._receive(None) == {"method": "test"}
    caller.incoming.put(None)
    with pytest.raises(ConnectionError):
        caller._receive(None)


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


def test_chunk_reply_survives_lost_ack(tmp_path):
    """Retrieving a saved chunk reply must not repeat its motion."""
    from robocasa_astra.astra_vla.sim_server import ChunkServer

    server = ChunkServer("OpenCabinet", 100, tmp_path)
    journal = ActionJournal(tmp_path, {"scene": "fake"})
    native = FakeNative(journal)
    journal.restore(native, native)
    server.journal = journal
    server.executor = native
    server.replay = SimpleNamespace(mark=lambda *args: None)
    native.state = lambda: {"steps_used": native.steps_used}

    def chunk_run(actions, notes):
        for row in actions:
            journal.intent(row)
            native._step(row, "chunk")
        return {"command": "chunk", "steps": len(actions), "task_success": False}

    native.run_chunk = chunk_run
    actions = [[-1, 1]] * 8
    first = server.act_chunk(actions, [], 1, "turn1")
    second = server.act_chunk(actions, [], 1, "turn1")
    assert first == second
    assert native.steps_used == 8


class HostSim:
    """A lost host connection after eight rows leaves the scene alive."""

    def __init__(self):
        import base64
        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buffer, format="PNG")
        self.png = base64.b64encode(buffer.getvalue()).decode()
        self.steps = 0
        self.crash = True
        self.finished = {}

    def state(self):
        """Provide measured mock proprioception at the current native step."""
        return {
            "fingertip_cm": [25.0, 0, 129.0],
            "approach": [0, 0, -1],
            "finger_axis": [1, 0, 0],
            "gripper_opening": 1.0,
            "gripper_command": "open",
            "surface_z_cm": 92.0,
            "steps_used": self.steps,
            "step_budget": 32,
            "task_success": self.steps >= 32,
        }

    def request(self, op, **fields):
        """Simulate host loss after a partial chunk and idempotent continuation."""
        if op == "reset":
            return {"instruction": "open", "state": self.state()}
        if op == "observe":
            return {
                "state": self.state(),
                "dataset_state": [0] * 16,
                "observation_id": str(self.steps),
                "views": [{"name": "cam", "caption": "camera", "png": self.png}],
            }
        if op == "act_chunk":
            key = fields["request_id"]
            if key in self.finished:
                return self.finished[key]
            if self.crash:
                self.steps = 8
                self.crash = False
                raise KeyboardInterrupt
            self.steps = 16 if key == "turn1" else 32
            reply = {
                "command": "chunk",
                "kind": "chunk",
                "ok": True,
                "gripper_closed": False,
                "gripper_opening": 1.0,
                "task_success": self.steps >= 32,
                "state_after": self.state(),
            }
            self.finished[key] = reply
            return reply
        raise AssertionError(op)


def test_host_resume_reuses_model_response_and_remaining_rows(tmp_path):
    """Resume a partial chunk without making another model request."""
    sim = HostSim()
    calls = []

    class Caller:
        def call(self, parts, folder):
            calls.append(folder)
            return CallResult(
                json.dumps({"actions": [[0] * 12] * 16}), {"input_tokens": 100}, [], 15
            )

    cfg = EpisodeConfig(task="OpenCabinet", seed=1, shots=0, max_turns=5, budget=32)
    with pytest.raises(KeyboardInterrupt):
        run_episode(sim, Caller(), cfg, tmp_path, [])
    assert read_json(tmp_path / "policy" / "progress.json")["phase"] == "action"
    summary = run_episode(sim, Caller(), cfg, tmp_path, [], resume=True)
    assert summary["task_success"] is True
    assert sim.steps == 32
    assert len(calls) == 2
    assert summary["usage"]["input_tokens"] == 200


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


def test_exact_process_rejects_reused_pid():
    """An identical PID alone cannot authorize adopting a process."""
    assert exact_process(process_lease())
    assert not exact_process({"pid": os.getpid(), "argv": "another process"})


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


def test_four_queries_return_all_answers_before_motion(tmp_path):
    """Queries keep the observation fixed and all four answers reach the action request."""
    sim = HostSim()
    sim.crash = False
    original = sim.request
    query_steps = []

    def request(op, **fields):
        """Simulate host loss after a partial chunk and idempotent continuation."""
        if op == "query":
            query_steps.append(sim.steps)
            return {"answers": [{"observation_id": "0", "pixel_depth_m": 0.1 * len(query_steps)}]}
        return original(op, **fields)

    sim.request = request
    parts_seen = []

    class Caller:
        """Four valid depth rounds followed by one chunk and a second successful chunk."""

        def call(self, parts, folder):
            """Save each exact prompt to assert query answers are retained."""
            parts_seen.append(parts)
            reply = (
                {
                    "queries": [
                        {
                            "observation_id": "0",
                            "camera": "cam",
                            "kind": "pixel",
                            "u": 1,
                            "v": 1,
                            "radius": 0,
                        }
                    ]
                }
                if len(parts_seen) <= 4
                else {"actions": [[0] * 12] * 16}
            )
            return CallResult(json.dumps(reply), {"input_tokens": 10}, [], 12)

    cfg = EpisodeConfig(task="OpenCabinet", seed=1, shots=0, max_turns=5, budget=32)
    summary = run_episode(sim, Caller(), cfg, tmp_path, [], condition="pixel")
    assert summary["task_success"] is True
    assert query_steps == [0, 0, 0, 0]
    assert len(parts_seen) == 6
    assert sum("DEPTH QUERY AND ANSWERS" in p.get("text", "") for p in parts_seen[4]) == 4
    assert summary["usage"]["input_tokens"] == 60


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


def test_native_exception_blocks_uncertain_live_state(tmp_path):
    """An action exception cannot repeat a row against physics lacking a durable acknowledgement."""
    from robocasa_astra.astra_vla.sim_server import ChunkServer

    server = ChunkServer("OpenCabinet", 100, tmp_path)
    server.journal = ActionJournal(tmp_path, {"scene": "fake"})

    def failed(actions, notes):
        server.journal.intent([-1, 1])
        raise RuntimeError("physics advanced but acknowledgement failed")

    server.executor = SimpleNamespace(run_chunk=failed)
    with pytest.raises(RuntimeError, match="acknowledgement"):
        server.act_chunk([[-1, 1]], [], 1, "turn1")
    assert server.journal.data["faulted"] is True
    with pytest.raises(RuntimeError, match="reset"):
        server.act_chunk([[-1, 1]], [], 1, "turn1")
    assert server.journal.data["chunks"]["turn1"]["cursor"] == 0
