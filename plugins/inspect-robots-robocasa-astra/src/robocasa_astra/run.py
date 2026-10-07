"""Run the official Inspect Robots evaluation loop against Spark2 RoboCasa."""

import argparse
import json
import sys
from pathlib import Path

from inspect_robots import eval
from inspect_robots.rollout import derive_seed
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task
from robocasa_astra.bridge import SparkEmbodiment
from robocasa_astra.policy import CodexPolicy


def main():
    """Select default PandaOmron or native bimanual GR1 and record a complete trial."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=["PandaOmron", "GR1FloatingBody"], default="PandaOmron")
    parser.add_argument("--task", default="PrepareCoffee")
    parser.add_argument("--placement", action="store_true")
    parser.add_argument("--native-scene", action="store_true")
    parser.add_argument("--face-workstation", action="store_true")
    parser.add_argument("--worker-script")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--seed", type=int, default=771001)
    parser.add_argument("--output", required=True)
    parser.add_argument("--container", default="astra-robocasa-20261006")
    parser.add_argument(
        "--local", action="store_true", help="run the worker on this host instead of Spark2"
    )
    parser.add_argument("--python", default=sys.executable, help="worker interpreter for --local")
    parser.add_argument("--codex", required=True)
    parser.add_argument("--codex-home", required=True)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--noop", action="store_true")
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.placement and (args.robot != "PandaOmron" or args.task != "PrepareCoffee"):
        parser.error("Held-mug fixtures are verified only for PandaOmron; use full GR1 task")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.local:
        # Inherits PYTHONPATH and MUJOCO_GL from run.sh; lower priority than co-located training.
        launcher = ["nice", "-n", "10", args.python]
    else:
        launcher = [
            "ssh",
            "spark2",
            "docker",
            "exec",
            "-i",
            "-e",
            "PYTHONPATH=/astra/src:/astra/plugins/inspect-robots-robocasa-astra/src",
            args.container,
            "python3",
        ]
    command = launcher + ["-m", "robocasa_astra.worker", "--robot", args.robot, "--task", args.task]
    if args.worker_script:
        marker = command.index("-m")
        command[marker : marker + 2] = [args.worker_script]
    command += ["--horizon", str(args.steps)]
    if args.face_workstation:
        command += ["--face-workstation"]
    if args.robot == "PandaOmron" and args.task == "PrepareCoffee" and not args.native_scene:
        command += ["--fixture", "/fixtures/episode-005"]
    if args.placement:
        command += ["--placement"]
    env = None
    try:
        initial_seed = (
            derive_seed(0, args.seed, 0) if args.native_scene and not args.probe else args.seed
        )
        env = SparkEmbodiment(command, initial_seed, output / "worker")
        (output / "environment.json").write_text(
            json.dumps({"name": env.info.name, "docs": env.info.docs}, indent=2)
        )
        if args.probe:
            import numpy as np

            from inspect_robots.types import Action

            scene = Scene(id="probe", instruction="Do not move", init_seed=args.seed)
            obs = env.reset(scene, seed=args.seed)
            from PIL import Image

            for name, image in obs.images.items():
                Image.fromarray(image).save(output / (name + ".png"))
            step = env.step(Action(np.zeros(env.info.action_space.shape)))
            (output / "probe.json").write_text(
                json.dumps(
                    {
                        "camera_names": list(obs.images),
                        "action_dim": env.info.action_space.shape[0],
                        "info": step.info,
                    },
                    indent=2,
                )
            )
            print("native reset/render/step passed", env.info.name)
            return
        policy = CodexPolicy(
            env,
            output / "inference",
            args.codex,
            args.codex_home,
            args.model,
            args.noop,
        )
        instruction = (
            "Place the held mug under the dispenser and release. Do not press the button."
            if args.placement
            else json.loads(env.info.docs)["instruction"]
        )
        task = Task(
            name=args.task + ("-placement" if args.placement else ""),
            scenes=[
                Scene(id=f"{args.robot}-{args.seed}", instruction=instruction, init_seed=args.seed)
            ],
            scorer=success_at_end(),
            max_steps=args.steps,
        )
        logs = eval(task, policy, env, log_dir=str(output / "eval"), store_frames=True)
        print("evaluation status", logs[0].status, "task metrics", logs[0].results.metrics)
        if logs[0].status != "success":
            raise RuntimeError("Inspect evaluation did not complete; inspect the saved error log")
    finally:
        if env:
            env.close()


if __name__ == "__main__":
    main()
