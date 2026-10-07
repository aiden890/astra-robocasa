"""Host-side pieces of astra_robodawn without a simulator or model: cost, memory, prompts, events, loop."""

import base64
import io
import json
import sys

from PIL import Image

from robocasa_astra.astra_robodawn import prompts
from robocasa_astra.astra_robodawn.appserver import AppServerCaller
from robocasa_astra.astra_robodawn.codex import CodexCaller, parse_events
from robocasa_astra.astra_robodawn.cost import add_usage, estimate
from robocasa_astra.astra_robodawn.loop import EpisodeConfig, run_episode
from robocasa_astra.astra_robodawn.memory import AgentMemory
from robocasa_astra.astra_robodawn.scripted import ScriptedCaller


def test_cost_matches_rate_card():
    usage = {"input_tokens": 1_000_000, "cached_input_tokens": 600_000, "output_tokens": 10_000,
             "reasoning_output_tokens": 4_000}
    est = estimate(usage)
    assert est["credits"] == 400_000 * 250 / 1e6 + 600_000 * 25 / 1e6 + 10_000 * 1250 / 1e6
    assert est["usd_api_equivalent"] == round(0.4 * 10 + 0.6 * 1 + 0.01 * 50, 4)
    assert est["credits_if_reasoning_separate"] > est["credits"]
    assert est["cache_hit_ratio"] == 0.6
    total = add_usage({}, usage)
    add_usage(total, {"input_tokens": 5})
    assert total["input_tokens"] == 1_000_005 and total["output_tokens"] == 10_000


def test_parse_events_extracts_usage_reasoning_and_errors():
    lines = [
        json.dumps({"type": "thread.started"}),
        json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "think"}}),
        json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}}),
        json.dumps({"type": "item.completed", "item": {"type": "error", "message": "code mode"}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}}),
        "not json",
    ]
    out = parse_events(lines)
    assert out["completed"] and out["reasoning"] == ["think"] and out["messages"] == ["{}"]
    assert out["usage"]["input_tokens"] == 10 and out["errors"] == ["code mode"]


def test_codex_command_is_isolated_and_low_effort(tmp_path):
    caller = CodexCaller("codex", "/home", tmp_path / "sys.md", tmp_path / "schema.json", tmp_path / "w")
    cmd = caller.command([tmp_path / "a.png"], tmp_path / "out.json")
    joined = " ".join(cmd)
    assert 'model_reasoning_effort="low"' in joined and "--strict-config" in joined
    assert "model_instructions_file" in joined and "features.shell_tool=false" in joined
    assert 'model_reasoning_summary="detailed"' in joined
    assert cmd[-3:] == ["--image", str(tmp_path / "a.png"), "-"]


def test_memory_history_and_grasp_fact():
    mem = AgentMemory(history_turns=2)
    state = {"fingertip_cm": [40.0, 5.0, 100.0], "gripper_opening": 0.3, "steps_used": 50}
    close = {"command": "gripper close", "kind": "gripper", "ok": True, "note": "fingers stopped",
             "gripper_opening": 0.3, "state_after": state}
    mem.record_turn(1, [{"command": "move down 5", "ok": True, "note": "reached"}], state, 92.0)
    mem.record_turn(2, [close], state, 92.0)
    mem.record_turn(3, [{"command": "move up 10", "ok": False, "note": "x" * 200}], state, 92.0)
    text = mem.render()
    assert "1 earlier turns omitted" in text and "..." in text
    assert "GRASP FACT" in text and "100 + (H - 92)" in text
    mem.record_turn(4, [{"command": "gripper open", "kind": "gripper", "ok": True}], state, 92.0)
    assert "GRASP FACT" not in mem.render()


def test_system_prompt_is_static_and_complete():
    text = prompts.system_prompt(prompts.load_profile())
    assert "{surface_z}" not in text and "move forward|back|left|right|up|down" in text
    assert "RESPONSE FORMAT" in text
    assert list(prompts.RESPONSE_SCHEMA["properties"]) == ["scene", "progress", "memory", "plan", "commands"]


def _write_demo(root, name, kind):
    folder = root / name
    (folder / "frames").mkdir(parents=True)
    demo = {"kind": kind, "task": "x", "instruction": "do", "frames": [
        {"label": "turn 1", "image": "frames/a.png", "state": "s", "commands": ["move up 5"], "effect": "up",
         "failed": ["move down 40: blocked by the counter"]},
        {"label": "turn 2", "image": None, "commands": ["gripper close"]}]}
    (folder / "demo.json").write_text(json.dumps(demo))


def test_demo_parts_interleave_image_then_its_text(tmp_path):
    _write_demo(tmp_path, "primer", "primer")
    _write_demo(tmp_path, "x", "task")
    parts = prompts.demo_parts(prompts.demos_for("x", 1, root=tmp_path))
    kinds = [p["type"] for p in parts]
    # primer: header, image, its text, text; task: header, image, its text, text; END
    assert kinds == ["text", "image", "text", "text", "text", "image", "text", "text", "text"]
    assert parts[-1]["text"] == prompts.END_OF_DEMOS
    task_turn = parts[6]["text"]
    assert task_turn.startswith("--- DEMO turn 1") and "net effect: up" in task_turn
    assert "(FAILED: move down 40: blocked by the counter)" in task_turn
    assert parts[5]["path"].endswith("x/frames/a.png")
    assert "[image: " in prompts.parts_text(parts)
    assert prompts.demo_parts([]) == []


class FakeSim:
    """Answers the sim_server protocol with a fixed state; success after the third executed command."""

    def __init__(self):
        buf = io.BytesIO()
        Image.new("RGB", (4, 4)).save(buf, format="PNG")
        self.png = base64.b64encode(buf.getvalue()).decode()
        self.steps = 0
        self.executed = []

    def state(self):
        return {"fingertip_cm": [25.0, 0.0, 129.0], "approach": [0, 0, -1], "finger_axis": [1, 0, 0],
                "gripper_opening": 1.0, "gripper_command": "open", "surface_z_cm": 92.0,
                "steps_used": self.steps, "step_budget": 100, "task_success": len(self.executed) >= 3}

    def request(self, op, **fields):
        if op == "reset":
            return {"instruction": "Open the cabinet doors.", "scene": {}, "state": self.state()}
        if op == "observe":
            return {"state": self.state(), "views": [{"name": "cam", "caption": "a camera", "png": self.png}]}
        if op == "execute":
            self.executed.append(fields["command"])
            self.steps += 10
            return {"command": fields["command"], "kind": "move", "ok": True, "note": "reached", "steps": 10,
                    "task_success": len(self.executed) >= 3, "state_after": self.state()}
        raise AssertionError(op)


def test_loop_runs_to_success_and_writes_trace(tmp_path):
    replies = [
        {"commands": ["move up 5", "jump"]},
        {"commands": ["done"]},
        {"commands": ["move down 5", "gripper close", "move up 3"]},
    ]
    sim = FakeSim()
    cfg = EpisodeConfig(task="OpenCabinet", seed=1, shots=0, max_turns=5, budget=100)
    demo = [prompts.text_part("DEMO"), prompts.text_part(prompts.END_OF_DEMOS)]
    summary = run_episode(sim, ScriptedCaller(replies), cfg, tmp_path, demo)
    assert summary["success"] and summary["finished_reason"] == "success" and summary["turns"] == 3
    assert sim.executed == ["move up 5", "move down 5", "gripper close"]
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert trace[0]["command_errors"] and trace[1]["results"][0]["command"] == "done"
    sent = json.loads((tmp_path / "calls" / "turn001" / "input.json").read_text())
    assert sent[0]["text"] == "DEMO" and sent[2]["type"] == "image" and sent[-1]["text"].startswith("TASK:")
    assert (tmp_path / "turns" / "turn001_cam.png").exists() and (tmp_path / "memory.json").exists()
    assert "RESULT OF YOUR LAST COMMANDS" in (tmp_path / "calls" / "turn002" / "prompt.txt").read_text()


def test_success_conditions_in_prompts():
    for task in ("OpenCabinet", "PickPlaceSinkToCounter", "PrepareCoffee", "PanTransfer", "StirVegetables"):
        text = prompts.system_prompt(prompts.load_profile(), task)
        assert "TASK SUCCESS CONDITION" in text and prompts.SUCCESS_CONDITIONS[task][0] in text
        state = {"fingertip_cm": [1, 2, 3], "approach": [0, 0, -1], "finger_axis": [0, 1, 0], "gripper_opening": 1.0,
                 "gripper_command": "open", "surface_z_cm": 92, "steps_used": 0, "step_budget": 10}
        turn = prompts.turn_text(1, 5, "do it", state, [], "", ["c"], task)
        assert "SUCCESS WHEN (all at once): " + prompts.SUCCESS_CONDITIONS[task][1] in turn
    assert "TASK SUCCESS CONDITION" not in prompts.system_prompt(prompts.load_profile())


FAKE_APP_SERVER = """
import json, sys
log = open(sys.argv[1], "a")
def send(m):
    print(json.dumps(m), flush=True)
for line in sys.stdin:
    req = json.loads(line)
    if "id" not in req:
        continue
    m = req["method"]
    if m == "initialize":
        send({"id": req["id"], "result": {}})
    elif m == "account/read":
        send({"id": req["id"], "result": {"account": {"type": "chatgpt"}}})
    elif m == "thread/start":
        log.write(json.dumps({"thread": req["params"]}) + "\\n"); log.flush()
        send({"method": "thread/started", "params": {"thread": {"id": "t1"}}})
        send({"id": req["id"], "result": {"thread": {"id": "t1"}, "model": req["params"]["model"]}})
    elif m == "turn/start":
        log.write(json.dumps({"turn": req["params"]}) + "\\n"); log.flush()
        # notifications before the RPC result, as the real server may send them
        send({"method": "error", "params": {"threadId": "t1", "turnId": "u1", "willRetry": True,
              "error": {"message": "Reconnecting... 2/5"}}})
        send({"method": "item/completed", "params": {"threadId": "t1", "turnId": "u1",
              "item": {"type": "reasoning", "id": "r", "summary": ["look at the handle"]}}})
        send({"method": "item/completed", "params": {"threadId": "t1", "turnId": "u1",
              "item": {"type": "agentMessage", "id": "a", "phase": "final_answer", "text": "{\\"commands\\": []}"}}})
        send({"method": "thread/tokenUsage/updated", "params": {"threadId": "t1", "turnId": "u1", "tokenUsage": {
              "last": {"inputTokens": 100, "cachedInputTokens": 60, "outputTokens": 7, "reasoningOutputTokens": 3,
                       "totalTokens": 107}, "total": {}}}})
        send({"method": "turn/completed", "params": {"threadId": "t1", "turn": {"id": "u1", "status": "completed"}}})
        send({"id": req["id"], "result": {"turn": {"id": "u1"}}})
"""


def test_appserver_caller_sends_ordered_parts_and_records(tmp_path):
    server = tmp_path / "fake_server.py"
    server.write_text(FAKE_APP_SERVER)
    log = tmp_path / "server.log"
    png = tmp_path / "a.png"
    Image.new("RGB", (2, 2)).save(png)
    (tmp_path / "sys.md").write_text("SYSTEM")
    (tmp_path / "schema.json").write_text(json.dumps(prompts.RESPONSE_SCHEMA))
    caller = AppServerCaller(sys.executable, str(tmp_path), tmp_path / "sys.md", tmp_path / "schema.json", tmp_path / "w")
    caller.argv = lambda: [sys.executable, str(server), str(log)]
    parts = [prompts.text_part("demo text"), prompts.image_part(png), prompts.text_part("its text"),
             prompts.image_part(png), prompts.text_part("TURN")]
    folder = tmp_path / "call"
    try:
        result = caller.call(parts, folder)
    finally:
        caller.close()
    assert result.text == '{"commands": []}' and result.reasoning == ["look at the handle"]
    assert result.usage == {"input_tokens": 100, "cached_input_tokens": 60, "output_tokens": 7,
                            "reasoning_output_tokens": 3}
    sent = [json.loads(line) for line in log.read_text().splitlines()]
    assert sent[0]["thread"]["baseInstructions"] == "SYSTEM" and sent[0]["thread"]["ephemeral"] is True
    turn = sent[1]["turn"]
    assert turn["effort"] == "low" and turn["summary"] == "detailed"
    assert [p["type"] for p in turn["input"]] == ["text", "image", "text", "image", "text"]
    assert turn["input"][1]["url"].startswith("data:image/png;base64,")
    assert json.loads((folder / "input.json").read_text())[1]["path"] == str(png)
    assert "data:image" not in (folder / "events.jsonl").read_text()
    call = json.loads((folder / "call.json").read_text())
    assert call["caller"] == "codex app-server" and call["attempts"][0]["warnings"] == ["Reconnecting... 2/5"]
