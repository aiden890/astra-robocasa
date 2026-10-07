"""Host-side pieces of astra_robodawn without a simulator or model: cost, memory, prompts, events, loop."""

import base64
import io
import json

from PIL import Image

from robocasa_astra.astra_robodawn import prompts
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
    text = prompts.system_prompt(prompts.load_profile(), "")
    assert "{surface_z}" not in text and "move forward|back|left|right|up|down" in text
    assert "RESPONSE FORMAT" in text
    assert list(prompts.RESPONSE_SCHEMA["properties"]) == ["scene", "progress", "memory", "plan", "commands"]


def test_render_demos_numbers_images(tmp_path):
    folder = tmp_path / "primer"
    (folder / "frames").mkdir(parents=True)
    demo = {"kind": "primer", "task": "x", "instruction": "do", "frames": [
        {"label": "step 1", "image": "frames/a.png", "state": "s", "commands": ["move up 5"], "effect": "up"},
        {"label": "step 2", "image": None, "commands": ["gripper close"]}]}
    (folder / "demo.json").write_text(json.dumps(demo))
    text, images = prompts.render_demos(prompts.demos_for("x", 0, root=tmp_path))
    assert "[image D1]" in text and len(images) == 1 and "step 2" in text


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
    summary = run_episode(sim, ScriptedCaller(replies), cfg, tmp_path, [])
    assert summary["success"] and summary["finished_reason"] == "success" and summary["turns"] == 3
    assert sim.executed == ["move up 5", "move down 5", "gripper close"]
    trace = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert trace[0]["command_errors"] and trace[1]["results"][0]["command"] == "done"
    assert (tmp_path / "calls" / "turn001" / "prompt.txt").exists()
    assert (tmp_path / "turns" / "turn001_cam.png").exists() and (tmp_path / "memory.json").exists()
    assert "RESULT OF YOUR LAST COMMANDS" in (tmp_path / "calls" / "turn002" / "prompt.txt").read_text()


def test_success_conditions_in_prompts():
    for task in ("OpenCabinet", "PickPlaceSinkToCounter", "PrepareCoffee", "PanTransfer", "StirVegetables"):
        text = prompts.system_prompt(prompts.load_profile(), "", task)
        assert "TASK SUCCESS CONDITION" in text and prompts.SUCCESS_CONDITIONS[task][0] in text
        state = {"fingertip_cm": [1, 2, 3], "approach": [0, 0, -1], "finger_axis": [0, 1, 0], "gripper_opening": 1.0,
                 "gripper_command": "open", "surface_z_cm": 92, "steps_used": 0, "step_budget": 10}
        turn = prompts.turn_text(1, 5, "do it", state, [], "", ["c"], 0, task)
        assert "SUCCESS WHEN (all at once): " + prompts.SUCCESS_CONDITIONS[task][1] in turn
    assert "TASK SUCCESS CONDITION" not in prompts.system_prompt(prompts.load_profile(), "")
