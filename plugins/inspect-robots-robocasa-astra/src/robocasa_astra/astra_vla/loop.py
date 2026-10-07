"""The VLA variant's closed loop: observe -> one model call -> execute its 16-step chunk -> feedback -> next turn.

Mirrors ``astra_robodawn.loop.run_episode`` (same request layout, memory, failure handling, timing and
``trace.jsonl`` keys, so ``debug_video`` and the review scripts work on both); the model returns ``actions``
(one chunk) instead of ``commands``, and the chunk's description stands in for the command text.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from ..astra_robodawn.codex import ModelCallError
from ..astra_robodawn.cost import add_usage, estimate
from ..astra_robodawn.loop import EpisodeConfig, _save_views
from ..astra_robodawn.memory import AgentMemory
from .action_format import validate_chunk
from .prompts import image_part, text_part, turn_text


class ChunkMemory(AgentMemory):
    """The skill variant's memory; the grasp fact is taken from chunks that close the gripper on something."""

    def record_turn(self, turn: int, results: list[dict], state: dict, surface_cm: float | None = None) -> None:
        super().record_turn(turn, results, state, surface_cm)
        for r in results:
            if r.get("kind") != "chunk":
                continue
            if r.get("gripper_closed") and r.get("gripper_opening", 0) > 0.06:
                if self.grasp is None:
                    self.grasp = {"height_cm": state["fingertip_cm"][2], "opening": r["gripper_opening"],
                                  "surface_cm": surface_cm}
            elif not r.get("gripper_closed"):
                self.grasp = None


def run_episode(sim, caller, cfg: EpisodeConfig, run_dir: Path, demo_parts: list[dict]) -> dict:
    """Run one episode against an already started ``ChunkSimClient``; returns the summary dict."""
    run_dir = Path(run_dir)
    start = sim.request("reset", seed=cfg.seed)
    instruction = start["instruction"]
    (run_dir / "episode_start.json").write_text(json.dumps(start, indent=1))
    memory = ChunkMemory()
    usage_total: dict = {}
    trace_file = (run_dir / "trace.jsonl").open("w")
    last_results: list[dict] = []
    failures = 0
    finished, success = "max_turns", False
    t_start = time.time()
    turn = 0
    for turn in range(1, cfg.max_turns + 1):
        obs = sim.request("observe")
        state = obs["state"]
        view_paths = _save_views(obs["views"], run_dir / "turns", f"turn{turn:03d}")
        text = turn_text(turn, cfg.max_turns, instruction, state, obs["dataset_state"], last_results, memory.render(),
                         [v["caption"] for v in obs["views"]], cfg.task)
        record = {"turn": turn, "time": round(time.time() - t_start, 1), "state": state,
                  "dataset_state": obs["dataset_state"], "images": [str(p.relative_to(run_dir)) for p in view_paths],
                  "call_dir": f"calls/turn{turn:03d}", "prompt": text}
        steps_before = state["steps_used"]
        t_call = time.time()
        try:
            parts = list(demo_parts) + [image_part(p) for p in view_paths] + [text_part(text)]
            reply = caller.call(parts, run_dir / "calls" / f"turn{turn:03d}")
        except ModelCallError as exc:
            failures += 1
            add_usage(usage_total, getattr(exc, "usage", {}))
            record.update(error=str(exc), usage=getattr(exc, "usage", {}), cost=estimate(getattr(exc, "usage", {})),
                          weekly=getattr(caller, "weekly", None))
            trace_file.write(json.dumps(record) + "\n")
            trace_file.flush()
            if failures >= cfg.failure_limit:
                finished = "model_error"
                break
            last_results = [{"command": "(no chunk)", "ok": False,
                             "note": "your previous reply could not be obtained; reply with the JSON object"}]
            continue
        t_reply = time.time()
        add_usage(usage_total, reply.usage)
        record.update(usage=reply.usage, cost=estimate(reply.usage), latency_s=reply.seconds, weekly=reply.weekly,
                      reasoning=reply.reasoning, reply_text=reply.text)
        try:
            parsed = json.loads(reply.text)
            chunk, notes = validate_chunk(parsed.get("actions"))
        except (json.JSONDecodeError, ValueError) as exc:
            failures += 1
            record.update(error=f"unusable reply: {exc}")
            trace_file.write(json.dumps(record) + "\n")
            trace_file.flush()
            if failures >= cfg.failure_limit:
                finished = "parse_failure"
                break
            last_results = [{"command": "(unusable reply)", "ok": False,
                             "note": f"{exc}; reply with the JSON object and exactly 16 rows of 12 numbers"}]
            continue
        failures = 0
        memory.update_scratchpad(parsed.get("memory"))
        result = sim.request("act_chunk", actions=chunk.tolist(), notes=notes, turn=turn)
        after = result["state_after"]
        success = bool(result.get("task_success"))
        t_done = time.time()
        memory.record_turn(turn, [result], after, state.get("surface_z_cm"))
        record.update(response=parsed, commands=[result["command"]], command_errors=notes,
                      results=[{k: v for k, v in result.items() if k != "state_after"}],
                      state_after=after, task_success=success,
                      timing={"call_start_s": round(t_call - t_start, 2), "model_s": round(t_reply - t_call, 2),
                              "exec_wall_s": round(t_done - t_reply, 2),
                              "motion_sim_s": round((after["steps_used"] - steps_before) / 20.0, 2)})
        trace_file.write(json.dumps(record) + "\n")
        trace_file.flush()
        last_results = [result]
        if success:
            finished = "success"
            break
        if after["steps_used"] >= cfg.budget:
            finished = "step_budget"
            break
    trace_file.close()
    final = sim.request("observe")
    _save_views(final["views"], run_dir / "turns", "final")
    (run_dir / "memory.json").write_text(json.dumps(memory.to_json(), indent=1))
    return {
        "task": cfg.task, "seed": cfg.seed, "shots": cfg.shots, "instruction": instruction,
        "success": bool(success or final["state"].get("task_success")), "finished_reason": finished,
        "turns": turn, "steps_used": final["state"]["steps_used"], "step_budget": cfg.budget,
        "max_turns": cfg.max_turns, "wall_seconds": round(time.time() - t_start, 1),
        "usage": usage_total, "cost": estimate(usage_total), "config": asdict(cfg), "output_format": "vla12-chunk16",
    }
