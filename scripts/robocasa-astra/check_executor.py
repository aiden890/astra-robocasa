"""Model-free check of the astra_robodawn executor on one native RoboCasa scene.

Runs a fixed command script (every command kind, both signs), then probes the reachable
fingertip region along each robot axis, and writes ``executor_check.json`` plus the video and
replay files to ``--output``. No model is called. Run with one simulator only, e.g.:

    source .runtime/amp2.env
    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src nice -n 10 $ASTRA_PYTHON \
        scripts/robocasa-astra/check_executor.py --task OpenCabinet --output runs/p1-executor-check
"""

import argparse
import json
from pathlib import Path

import numpy as np

from robocasa_astra.astra_robodawn.commands import parse_command
from robocasa_astra.astra_robodawn.executor import Executor
from robocasa_astra.astra_robodawn.recorder import ReplayRecorder, VideoRecorder
from robocasa_astra.worker import Simulator

SCRIPT = [
    "move up 10", "move down 10", "move forward 10", "move back 10", "move left 10", "move right 10",
    "move forward 20", "move back 20",
    "rotate yaw 30", "rotate yaw -30", "rotate pitch 20", "rotate pitch -20", "rotate roll 20", "rotate roll -20",
    "point forward", "point down45", "point down",
    "gripper open", "gripper close", "gripper open",
    "base forward 20", "base back 20", "base left 20", "base right 20", "base turn 20", "base turn -20",
    "home", "wait",
]
PROBE_DIRECTIONS = ("forward", "back", "left", "right", "up", "down")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="OpenCabinet")
    parser.add_argument("--seed", type=int, default=771001)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    sim = Simulator("PandaOmron", args.task, horizon=args.budget)
    sim.reset(args.seed)
    env = sim.env
    video = VideoRecorder(out / "video.mp4")
    replay = ReplayRecorder(out / "replay")
    replay.save_initial(env, {"task": args.task, "seed": args.seed, "horizon": args.budget, "purpose": "executor check"})
    executor = Executor(env, args.budget, on_step=lambda a, o, c: (video.add(o, c), replay.add(a)))

    report = {"task": args.task, "seed": args.seed, "initial_state": executor.state(), "script": [], "reach": {}}
    for line in SCRIPT:
        before = executor.state()
        first = executor.steps_used
        result = executor.execute(parse_command(line)).to_json()
        replay.mark(line, first, executor.steps_used, turn=0)
        result["before_fingertip_cm"] = before["fingertip_cm"]
        result["after"] = executor.state()
        report["script"].append(result)
        print(f"{line:22s} ok={result['ok']!s:5s} steps={result['steps']:3d} {result['note'][:90]}")

    # Reach probe: from home, for each gripper orientation, step 5 cm along each axis until a command fails.
    for preset in ("down", "forward"):
        for direction in PROBE_DIRECTIONS:
            for _ in range(2):
                if executor.execute(parse_command("home")).ok:
                    break
            executor.execute(parse_command(f"point {preset}"))
            start = np.array(executor.state()["fingertip_cm"])
            travelled, last = 0, None
            for _ in range(16):
                last = executor.execute(parse_command(f"move {direction} 5"))
                if not last.ok:
                    break
                travelled += 5
            end = np.array(executor.state()["fingertip_cm"])
            report["reach"][f"{preset}/{direction}"] = {
                "start_cm": start.tolist(), "end_cm": end.tolist(), "commanded_ok_cm": travelled,
                "stop_note": last.note if last is not None and not last.ok else "",
            }
            print(f"reach {preset:7s} {direction:7s}: {travelled:2d} cm ok, end {end.tolist()} {report['reach'][f'{preset}/{direction}']['stop_note'][-60:]}")

    report["steps_used"] = executor.steps_used
    video.close()
    replay.close(env)
    (out / "executor_check.json").write_text(json.dumps(report, indent=1))
    env.close()


if __name__ == "__main__":
    main()
