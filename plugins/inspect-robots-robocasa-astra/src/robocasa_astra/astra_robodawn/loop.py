"""The closed loop: observe -> one model decision -> execute its commands -> feedback -> next turn.

Adapted from RoboDawn ``harness/agent/mllm_agent.py``. Per turn the run directory receives the
images the model saw (``turns/``), the full call record (``calls/turnNNN/``) and one line of
``trace.jsonl`` that ties together state, prompt, reasoning, reply, commands, outcomes, usage and cost.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .codex import ModelCallError
from .commands import parse_command_list
from .cost import add_usage, estimate
from .memory import AgentMemory
from .prompts import image_part, text_part, turn_text

DONE_NOTE = ("NOT finished: the task checker has not registered success, so the episode continues. Re-read the "
             "instruction and compare it with the images (e.g. all doors fully open, object inside the container, "
             "machine started, gripper moved away); then keep acting.")


@dataclass
class EpisodeConfig:
    task: str
    seed: int
    shots: int
    max_turns: int
    budget: int
    done_limit: int = 3
    failure_limit: int = 3


def _save_views(views: list[dict], folder: Path, stem: str) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for view in views:
        path = folder / f"{stem}_{view['name']}.png"
        path.write_bytes(base64.b64decode(view["png"]))
        paths.append(path)
    return paths


def run_episode(sim, caller, cfg: EpisodeConfig, run_dir: Path, demo_parts: list[dict]) -> dict:
    """Run one episode against an already started simulator client; returns the summary dict.

    Every request is ``demo_parts`` (the fixed demonstration block) + the current views + the turn text.
    """
    run_dir = Path(run_dir)
    start = sim.request("reset", seed=cfg.seed)
    instruction = start["instruction"]
    (run_dir / "episode_start.json").write_text(json.dumps(start, indent=1))
    memory = AgentMemory()
    usage_total: dict = {}
    trace_file = (run_dir / "trace.jsonl").open("w")
    last_results: list[dict] = []
    failures = done_count = 0
    finished = "max_turns"
    success = False
    t_start = time.time()
    turn = 0
    for turn in range(1, cfg.max_turns + 1):
        obs = sim.request("observe")
        state = obs["state"]
        view_paths = _save_views(obs["views"], run_dir / "turns", f"turn{turn:03d}")
        text = turn_text(turn, cfg.max_turns, instruction, state, last_results, memory.render(),
                         [v["caption"] for v in obs["views"]], cfg.task)
        record = {"turn": turn, "time": round(time.time() - t_start, 1), "state": state,
                  "images": [str(p.relative_to(run_dir)) for p in view_paths],
                  "call_dir": f"calls/turn{turn:03d}", "prompt": text}
        steps_before = state["steps_used"]
        t_call = time.time()
        try:
            parts = list(demo_parts) + [image_part(p) for p in view_paths] + [text_part(text)]
            reply = caller.call(parts, run_dir / "calls" / f"turn{turn:03d}")
        except ModelCallError as exc:
            failures += 1
            add_usage(usage_total, getattr(exc, "usage", {}))
            record.update(error=str(exc), usage=getattr(exc, "usage", {}), cost=estimate(getattr(exc, "usage", {})))
            trace_file.write(json.dumps(record) + "\n")
            trace_file.flush()
            if failures >= cfg.failure_limit:
                finished = "model_error"
                break
            last_results = [{"command": "(no command)", "ok": False,
                             "note": "your previous reply could not be obtained; reply with the JSON object"}]
            continue
        t_reply = time.time()
        add_usage(usage_total, reply.usage)
        record.update(usage=reply.usage, cost=estimate(reply.usage), latency_s=reply.seconds,
                      reasoning=reply.reasoning, reply_text=reply.text)
        try:
            parsed = json.loads(reply.text)
        except json.JSONDecodeError as exc:
            failures += 1
            record.update(error=f"unparseable reply: {exc}")
            trace_file.write(json.dumps(record) + "\n")
            trace_file.flush()
            if failures >= cfg.failure_limit:
                finished = "parse_failure"
                break
            last_results = [{"command": "(unparseable reply)", "ok": False,
                             "note": "your reply was not a valid JSON object; reply with the JSON object only"}]
            continue
        failures = 0
        memory.update_scratchpad(parsed.get("memory"))
        commands, errors = parse_command_list(parsed.get("commands") or [])
        results, said_done = [], False
        for cmd in commands:
            if cmd.kind == "done":
                said_done = True
                results.append({"command": "done", "kind": "done", "ok": False, "note": DONE_NOTE})
                break
            res = sim.request("execute", command=cmd.text(), turn=turn)
            results.append(res)
            if res.get("task_success") or res["state_after"]["steps_used"] >= cfg.budget:
                break
        results += [{"command": "(invalid)", "kind": "invalid", "ok": False, "note": err} for err in errors]
        after = next((r["state_after"] for r in reversed(results) if "state_after" in r), state)
        success = any(r.get("task_success") for r in results)
        t_done = time.time()
        memory.record_turn(turn, results, after, state.get("surface_z_cm"))
        record.update(response=parsed, commands=[c.text() for c in commands], command_errors=errors,
                      results=[{k: v for k, v in r.items() if k != "state_after"} for r in results],
                      state_after=after, task_success=success,
                      timing={"call_start_s": round(t_call - t_start, 2), "model_s": round(t_reply - t_call, 2),
                              "exec_wall_s": round(t_done - t_reply, 2),
                              "motion_sim_s": round((after["steps_used"] - steps_before) / 20.0, 2)})
        trace_file.write(json.dumps(record) + "\n")
        trace_file.flush()
        last_results = results
        if success:
            finished = "success"
            break
        if after["steps_used"] >= cfg.budget:
            finished = "step_budget"
            break
        done_count = done_count + 1 if said_done else 0
        if done_count >= cfg.done_limit:
            finished = "agent_done"
            break
    trace_file.close()
    final = sim.request("observe")
    _save_views(final["views"], run_dir / "turns", "final")
    (run_dir / "memory.json").write_text(json.dumps(memory.to_json(), indent=1))
    summary = {
        "task": cfg.task, "seed": cfg.seed, "shots": cfg.shots, "instruction": instruction,
        "success": bool(success or final["state"].get("task_success")), "finished_reason": finished,
        "turns": turn, "steps_used": final["state"]["steps_used"], "step_budget": cfg.budget,
        "max_turns": cfg.max_turns, "wall_seconds": round(time.time() - t_start, 1),
        "usage": usage_total, "cost": estimate(usage_total), "config": asdict(cfg),
    }
    return summary


def partial_summary(run_dir: Path, cfg: EpisodeConfig, reason: str, steps_used: int, success: bool) -> dict:
    """Summary of an episode that ended early (interrupted), rebuilt from ``trace.jsonl``."""
    usage: dict = {}
    trace = Path(run_dir) / "trace.jsonl"
    records = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
    for record in records:
        add_usage(usage, record.get("usage") or {})
    return {"task": cfg.task, "seed": cfg.seed, "shots": cfg.shots, "success": success, "finished_reason": reason,
            "turns": len(records), "steps_used": steps_used, "step_budget": cfg.budget, "max_turns": cfg.max_turns,
            "wall_seconds": records[-1]["time"] if records else 0.0, "usage": usage, "cost": estimate(usage),
            "config": asdict(cfg)}
