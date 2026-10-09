"""Resumable VLA decisions, depth-query rounds, native chunks and immutable call accounting."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from ..astra_robodawn.codex import ModelCallError
from ..astra_robodawn.cost import add_usage, estimate
from ..astra_robodawn.loop import EpisodeConfig, _save_views
from .depth_input import QUERY_ROUNDS, extra_parts
from .memory import ChunkMemory
from .motion_control import prepare_response
from .persistence import append_json, atomic_json, read_json
from .prompts import image_part, text_part, turn_text


def run_episode(
    sim,
    caller,
    cfg: EpisodeConfig,
    run_dir: Path,
    demo_parts: list[dict],
    condition: str = "rgb",
    resume: bool = False,
    context: str = "full",
    motion_control: str = "legacy",
) -> dict:
    """Resume the same decision and native chunk without applying acknowledged rows again."""
    run_dir = Path(run_dir)
    progress_path = run_dir / "policy" / "progress.json"
    trace_path = run_dir / "trace.jsonl"
    if trace_path.exists() and not resume:
        raise ValueError("existing episode requires --resume")
    start = sim.request("reset", seed=cfg.seed)
    instruction = start["instruction"]
    if not (run_dir / "episode_start.json").exists():
        atomic_json(run_dir / "episode_start.json", start)
    trace_records = run_dir / "policy" / "trace-records"
    if trace_records.exists():
        records = [read_json(path) for path in sorted(trace_records.glob("*.json"))]
    else:
        records = (
            [json.loads(line) for line in trace_path.read_text().splitlines()]
            if trace_path.exists()
            else []
        )

    def commit_record(record):
        # Atomic records are the resume source; retain the original append-only review trace.
        atomic_json(trace_records / f"{record['turn']:06d}.json", record)
        append_json(trace_path, record)

    memory = ChunkMemory()
    last_results = []
    for record in records:
        if record.get("response") and record.get("state_after"):
            memory.update_scratchpad(record["response"].get("memory"))
            last_results = record["results"]
            memory.record_turn(
                record["turn"],
                last_results,
                record["state_after"],
                record["state"].get("surface_z_cm"),
            )
    progress = read_json(progress_path, {})
    t_start = progress.get("started_at", time.time())
    finished, success = "max_turns", False
    first_turn = max((r["turn"] for r in records), default=0) + 1
    failures = 0
    turn = first_turn - 1
    for turn in range(first_turn, cfg.max_turns + 1):
        obs_now = sim.request("observe")
        pending_chunk = progress.get("turn") == turn and progress.get("phase") == "action"
        if not pending_chunk and obs_now["state"].get("task_success"):
            finished, success = "success", True
            break
        if not pending_chunk and obs_now["state"]["steps_used"] >= cfg.budget:
            finished = "step_budget"
            break
        if progress.get("turn") == turn and progress.get("phase") in ("calling", "action"):
            obs, parts, record = progress["obs"], progress["parts"], progress["record"]
            # A model response must remain bound to the observation saved before that call.
            if progress["phase"] == "calling" and obs.get("observation_id") != obs_now.get(
                "observation_id"
            ):
                raise ValueError("observation drift while waiting for model response")
        else:
            obs = obs_now
            state = obs["state"]
            paths = _save_views(obs["views"], run_dir / "turns", f"turn{turn:03d}")
            text = turn_text(
                turn,
                cfg.max_turns,
                instruction,
                state,
                obs["dataset_state"],
                last_results if context == "full" else [],
                memory.render() if context == "full" else "",
                [v["caption"] for v in obs["views"]],
                cfg.task,
            )
            if motion_control == "dual":
                text += "\n\nTRANSIT COORDINATES: " + json.dumps(
                    {
                        "observation_id": obs.get("observation_id"),
                        "fingertip_world_m": obs.get("fingertip_world_m"),
                    }
                )
            parts = list(demo_parts) + [image_part(p) for p in paths] + [text_part(text)]
            parts += extra_parts(obs, run_dir / "turns" / f"turn{turn:03d}", condition)
            record = {
                "turn": turn,
                "state": state,
                "dataset_state": obs["dataset_state"],
                "prompt": text,
                "observation_id": obs.get("observation_id"),
                "images": [str(p.relative_to(run_dir)) for p in paths],
            }
            progress = {
                "started_at": t_start,
                "turn": turn,
                "phase": "calling",
                "obs": obs,
                "parts": parts,
                "record": record,
                "query_round": 0,
                "query_history": [],
                "calls": [],
            }
            atomic_json(progress_path, progress)
        try:
            if progress["phase"] == "calling":
                while True:
                    round_number = progress["query_round"]
                    folder = run_dir / "calls" / f"turn{turn:03d}-q{round_number}"
                    reply = caller.call(parts, folder)
                    parsed = json.loads(reply.text)
                    # Reusing this response after a crash does not add another accounting entry.
                    if str(folder) not in [c["folder"] for c in progress["calls"]]:
                        progress["calls"].append(
                            {
                                "folder": str(folder),
                                "usage": reply.usage,
                                "latency_s": reply.seconds,
                                "reasoning": reply.reasoning,
                                "reply_text": reply.text,
                            }
                        )
                    if not parsed.get("queries"):
                        actions, notes, target = prepare_response(parsed, motion_control)
                        progress.update(
                            phase="action",
                            response=parsed,
                            actions=actions,
                            transit_target=target,
                            notes=notes,
                            motion_mode=parsed.get("motion_mode"),
                        )
                        atomic_json(progress_path, progress)
                        break
                    if (
                        parsed.get("actions") is not None
                        or parsed.get("motion_mode") is not None
                        or parsed.get("transit_target") is not None
                        or condition not in ("pixel", "grid", "hybrid", "spatial")
                        or round_number >= QUERY_ROUNDS
                    ):
                        raise ValueError("invalid query round or simultaneous actions/queries")
                    answers = sim.request(
                        "query", queries=parsed["queries"], observation_id=obs["observation_id"]
                    )["answers"]
                    entry = {
                        "round": round_number + 1,
                        "queries": parsed["queries"],
                        "answers": answers,
                    }
                    progress["query_history"].append(entry)
                    parts = [
                        *parts,
                        text_part(
                            "DEPTH QUERY AND ANSWERS (same observation): " + json.dumps(entry)
                        ),
                    ]
                    progress.update(parts=parts, query_round=round_number + 1)
                    atomic_json(progress_path, progress)
            parsed = progress["response"]
            mode_fields = (
                {
                    "motion_mode": progress["motion_mode"],
                    "transit_target": progress.get("transit_target"),
                }
                if motion_control == "dual"
                else {}
            )
            result = sim.request(
                "act_chunk",
                actions=progress["actions"],
                notes=progress["notes"],
                turn=turn,
                request_id=f"turn{turn}",
                **mode_fields,
            )
        except (ModelCallError, ValueError, json.JSONDecodeError) as exc:
            # An actual unusable response is not a successful native evaluation. Preserve all
            # evidence.
            progress.update(error=str(exc), phase="failed")
            atomic_json(progress_path, progress)
            record.update(
                error=str(exc), calls=progress["calls"], query_history=progress["query_history"]
            )
            commit_record(record)
            failures += 1
            if failures >= cfg.failure_limit:
                finished = "execution_error"
                break
            progress = {}
            continue
        failures = 0
        after = result["state_after"]
        success = bool(result.get("task_success"))
        memory.update_scratchpad(parsed.get("memory"))
        memory.record_turn(turn, [result], after, obs["state"].get("surface_z_cm"))
        usage = {}
        for call in progress["calls"]:
            add_usage(usage, call["usage"])
        record.update(
            response=parsed,
            calls=progress["calls"],
            query_history=progress["query_history"],
            usage=usage,
            cost=estimate(usage),
            latency_s=sum(c["latency_s"] for c in progress["calls"]),
            results=[result],
            commands=[result["command"]],
            state_after=after,
            task_success=success,
        )
        commit_record(record)
        records.append(record)
        progress.update(phase="committed")
        atomic_json(progress_path, progress)
        last_results = [result]
        if success:
            finished = "success"
            break
        if after["steps_used"] >= cfg.budget:
            finished = "step_budget"
            break
    final = sim.request("observe")
    _save_views(final["views"], run_dir / "turns", "final")
    atomic_json(run_dir / "memory.json", memory.to_json())
    # Unique original per-call receipts include failed attempts, even if the host died before
    # tracing them.
    usage_total, unknown = {}, 0
    for path in (run_dir / "calls").glob("*/receipts.jsonl"):
        for line in path.read_text().splitlines():
            receipt = json.loads(line)
            if receipt["usage"] is None:
                unknown += 1
            else:
                add_usage(usage_total, receipt["usage"])
    if not usage_total and not unknown:
        for record in records:
            add_usage(usage_total, record.get("usage", {}))
    success = bool(success or final["state"].get("task_success"))
    valid = success or final["state"]["steps_used"] >= cfg.budget
    return {
        "task": cfg.task,
        "seed": cfg.seed,
        "shots": cfg.shots,
        "instruction": instruction,
        "success": success,
        "task_success": success if valid else None,
        "finished_reason": finished,
        "turns": turn,
        "steps_used": final["state"]["steps_used"],
        "step_budget": cfg.budget,
        "max_turns": cfg.max_turns,
        "wall_seconds": time.time() - t_start,
        "usage": usage_total or None,
        "unknown_usage_attempts": unknown,
        "cost": estimate(usage_total),
        "cost_is_lower_bound": unknown > 0,
        "config": asdict(cfg),
        "condition": condition,
        "context": context,
        "motion_control": motion_control,
        "output_format": "precision-vla12-transit-world-target"
        if motion_control == "dual"
        else "vla12-chunk16",
    }
