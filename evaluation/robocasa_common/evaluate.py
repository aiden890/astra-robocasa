"""Evaluate interchangeable policies on verified, immutable single-arm RoboCasa scenes."""

import argparse
import importlib
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from robocasa_astra.bridge import SparkEmbodiment
from robocasa_astra.frozen_scene import sha256

from inspect_robots import eval
from inspect_robots.rollout import derive_seed
from inspect_robots.scene import Scene
from inspect_robots.scorer import success_at_end
from inspect_robots.task import Task


class ObservedSparkEmbodiment(SparkEmbodiment):
    """Notify policy history collectors after every actual environment observation."""

    def __init__(self, *args, **kwargs):
        self.observers = []
        super().__init__(*args, **kwargs)

    def reset(self, scene, *, seed=None):
        """Include the initial observation in registered histories."""
        observation = super().reset(scene, seed=seed)
        for observer in self.observers:
            observer(observation)
        return observation

    def step(self, action):
        """Record intermediate steps even when a policy executes an action chunk."""
        result = super().step(action)
        for observer in self.observers:
            observer(result.observation)
        return result


def scene_identifiers(tasks):
    """Accept recorded seed metadata as well as the original integer seed lists."""
    return [
        f"{task}-{item['seed'] if isinstance(item, dict) else item}"
        for task, values in tasks.items()
        for item in values
    ]


def load_scene(folder, remote_proof=None):
    """Require a verified Panda snapshot before allocating a simulator or calling a model."""
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    verified = json.loads((folder / "verified.json").read_text())
    if manifest["robot"] != "PandaOmron":
        raise ValueError("Shared evaluation protocol uses PandaOmron only")
    if not verified["state_exact"] or not verified["world_geometry_exact"]:
        raise ValueError("Scene has not passed restoration verification")
    if verified["manifest_sha256"] != sha256(folder / "manifest.json"):
        raise ValueError("Scene manifest changed after verification")
    if remote_proof is not None and remote_proof["manifest_sha256"] != verified["manifest_sha256"]:
        raise ValueError("Remote scene manifest differs from local metadata")
    for name, digest in manifest["files"].items():
        target = (folder / name).resolve()
        actual = sha256(target) if remote_proof is None else remote_proof["files"].get(name)
        if folder.parents[1] not in target.parents or actual != digest:
            raise ValueError("Scene content checksum or path mismatch")
    return manifest


def resolve_factory(reference):
    """Import the participant's policy factory without changing the common environment."""
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError("Policy factory must be module:function")
    return getattr(importlib.import_module(module), name)


def evaluate_scene(folder, args):
    """Run each scene separately so official seed derivation always uses scene index zero."""
    remote_proof = None
    if getattr(args, "remote_verification", False):
        verify = [
            "docker",
            "exec",
            "-e",
            "PYTHONPATH=/eval-code/evaluation:/eval-code/src:/frozen-code",
            args.container,
            "python3",
            "-m",
            "robocasa_common.verify",
            args.mounted_root.rstrip("/") + "/scenes/" + Path(folder).name,
        ]
        command = verify if args.host == "local" else ["ssh", args.host, shlex.join(verify)]
        remote_proof = json.loads(subprocess.check_output(command, text=True, timeout=180))
    manifest = load_scene(folder, remote_proof)
    if derive_seed(0, manifest["rollout_seed"], 0) != manifest["simulator_seed"]:
        raise ValueError("Scene seed derivation does not match the fixed protocol")
    output = Path(args.output).resolve() / Path(folder).name
    resumable = getattr(args, "depth", False)
    checkpoint = output / "worker/checkpoint.json"
    continuing = resumable and checkpoint.exists()
    empty_dispatch = output.exists() and {p.name for p in output.iterdir()} <= {
        "resume-container.json"
    }
    if output.exists() and not continuing and not empty_dispatch:
        raise FileExistsError(
            "Existing episode lacks a verified checkpoint; preserve it before retry"
        )
    if continuing and (output / "result.json").exists():
        previous = json.loads((output / "result.json").read_text())
        if previous.get("execution_status") == "success":
            return previous
        archive = (
            output.parents[2]
            / "attempts"
            / f"resume-error-{time.time_ns()}"
            / "results"
            / output.parent.name
            / output.name
        )
        shutil.copytree(output, archive)
        (archive.parents[2] / "snapshot-only.json").write_text(
            json.dumps(
                {
                    "note": "Checkpoint snapshot; call IDs remain in live continuation",
                    "do_not_double_count_calls": True,
                }
            )
        )
        (output / "result.json").unlink()
    output.mkdir(parents=True, exist_ok=continuing or empty_dispatch)
    remote = [
        "docker",
        "exec",
        "-i",
        "-e",
        "PYTHONPATH=/frozen-code:/astra/src",
        args.container,
        "python3",
        "-m",
        "robocasa_astra.worker",
        "--task",
        manifest["task"],
        "--robot",
        "PandaOmron",
        "--horizon",
        str(manifest["horizon"]),
        "--frozen-scene",
        args.mounted_root.rstrip("/") + "/scenes/" + Path(folder).name,
    ]
    if getattr(args, "depth", False):
        remote.extend(["--depth", "--session-id", output.parent.name + "-" + output.name])
    command = remote if args.host == "local" else ["ssh", args.host, shlex.join(remote)]
    env = ObservedSparkEmbodiment(
        command,
        manifest["simulator_seed"],
        output / "worker",
        checkpoint_mode=resumable,
        checkpoint_identity=[
            sha256(Path(folder) / "manifest.json"),
            args.policy,
            output.parent.name,
            "PandaOmron",
            manifest["horizon"],
        ]
        if resumable
        else None,
    )
    policy = None
    try:
        receipt = json.loads(env.info.docs)["frozen_scene"]
        if receipt["manifest_sha256"] != sha256(Path(folder) / "manifest.json"):
            raise ValueError("Worker restored a different scene")
        if not receipt["state_exact"] or not receipt["world_geometry_exact"]:
            raise ValueError("Worker did not verify exact initial state and world/camera geometry")
        (output / "scene-receipt.json").write_text(json.dumps(receipt, indent=2))
        if remote_proof is not None:
            (output / "asset-proof.json").write_text(json.dumps(remote_proof, indent=2))
        policy = resolve_factory(args.policy)(env, output / "policy")
        if continuing and (
            env.resume_steps >= manifest["horizon"] or env.latest["info"]["success"]
        ):
            result = {
                "scene": Path(folder).name,
                "policy": args.policy,
                "manifest_sha256": receipt["manifest_sha256"],
                "execution_status": "success",
                "task_success": bool(env.latest["info"]["success"]),
                "checkpoint_resumed_steps": env.resume_steps,
                "native_steps": env.resume_steps,
            }
            (output / "result.json").write_text(json.dumps(result, indent=2))
            return result
        task = Task(
            name=manifest["task"],
            scenes=[
                Scene(
                    id=Path(folder).name,
                    init_seed=manifest["rollout_seed"],
                    instruction=json.loads(env.info.docs)["instruction"],
                )
            ],
            scorer=success_at_end(),
            max_steps=max(1, manifest["horizon"] - env.resume_steps),
        )
        log = eval(
            task,
            policy,
            env,
            log_dir=str(output / "eval"),
            store_frames=not getattr(args, "compact_video", False),
        )[0]
        result = {
            "scene": Path(folder).name,
            "policy": args.policy,
            "manifest_sha256": receipt["manifest_sha256"],
            "checkpoint_resumed_steps": env.resume_steps,
            "native_steps": env.checkpoint.data["steps"] if resumable else None,
            "execution_status": log.status,
            "task_success": log.results.metrics.get("success_at_end") == 1
            if log.status == "success"
            else None,
        }
        (output / "result.json").write_text(json.dumps(result, indent=2))
        return result
    finally:
        try:
            if policy is not None and hasattr(policy, "close"):
                policy.close()
        finally:
            env.close()


def main():
    """Select one scene or the complete catalog without rewriting scene data or old results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--scene-id")
    choice.add_argument("--all", action="store_true")
    parser.add_argument("--seeds-file", default=str(Path(__file__).parents[1] / "seeds.json"))
    parser.add_argument("--policy", required=True, help="Participant module:function")
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default="spark2")
    parser.add_argument("--container", required=True)
    parser.add_argument("--mounted-root", default="/scene-bundles")
    parser.add_argument(
        "--compact-video",
        action="store_true",
        help="Policy streams video instead of retaining frame sidecars",
    )
    parser.add_argument(
        "--remote-verification",
        action="store_true",
        help="Hash scene assets on the simulation host; retain metadata locally",
    )
    args = parser.parse_args()
    if args.all:
        seeds = json.loads(Path(args.seeds_file).read_text())["tasks"]
        identifiers = scene_identifiers(seeds)
    else:
        identifiers = [args.scene_id]
    folders = [Path(args.scene_root) / "scenes" / name for name in identifiers]
    if not args.remote_verification:
        for folder in folders:
            load_scene(folder)
    for folder in folders:
        print(json.dumps(evaluate_scene(folder, args)), flush=True)


if __name__ == "__main__":
    main()
