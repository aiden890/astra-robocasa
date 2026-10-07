"""Replay a recorded astra_robodawn run from its ``replay/`` folder and check it is exact.

Default (``--mode seed``): rebuild the native scene from task + seed exactly as the run did and step
every saved native action; this reproduces the final MuJoCo state exactly (verified in P1).
``--mode state`` restores the saved model XML / initial state instead; physics starts identical but
the controllers are rebuilt, so long replays drift. Both compare against ``final_state.npz``.
Optionally re-encodes a video of the replay. One simulator only; no model is called.

    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src nice -n 10 $ASTRA_PYTHON \\
        scripts/robocasa-astra/replay_run.py runs/<run>/replay [--video replay.mp4]
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from robocasa_astra.astra_robodawn.recorder import VideoRecorder
from robocasa_astra.astra_robodawn.replay import rebuild_matching
from robocasa_astra.worker import Simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("replay_dir")
    parser.add_argument("--video", help="optional output MP4 of the replay")
    parser.add_argument("--mode", choices=["seed", "state"], default="seed",
                        help="seed: rebuild the scene from task+seed exactly as the run did (default); "
                             "state: additionally restore the saved model XML and initial state")
    args = parser.parse_args()
    folder = Path(args.replay_dir)
    scene = json.loads((folder / "scene.json").read_text())
    actions = np.load(folder / "actions.npy")

    from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import reset_to

    tries = 1
    if args.mode == "seed":
        # RoboCasa scene builds vary slightly between attempts; rebuild until the saved initial state matches.
        sim, scene, tries = rebuild_matching(folder)
    else:
        sim = Simulator("PandaOmron", scene["task"], horizon=scene.get("horizon", len(actions) + 1))
        sim.reset(scene["seed"])
    env = sim.env
    if args.mode == "state":
        # Approximate: restores physics exactly but rebuilds controllers, so long replays drift.
        reset_to(env, {
            "model": gzip.decompress((folder / "model.xml.gz").read_bytes()).decode(),
            "states": np.load(folder / "initial_state.npz")["state"],
            "ep_meta": (folder / "ep_meta.json").read_text(),
        })
    initial_diff = float(np.abs(env.sim.get_state().flatten() - np.load(folder / "initial_state.npz")["state"]).max())
    video = VideoRecorder(Path(args.video)) if args.video else None
    success_step = None
    for i, action in enumerate(actions):
        obs, _, _, _ = env.step(action)
        if success_step is None and env._check_success():
            success_step = i + 1
        if video:
            video.add(obs, f"replay step {i + 1}/{len(actions)}")
    if video:
        video.close()
    report = {"mode": args.mode, "scene_builds": tries, "steps": int(len(actions)), "initial_state_max_abs_diff": initial_diff, "first_success_step": success_step}
    final_path = folder / "final_state.npz"
    if final_path.exists():
        final = np.load(final_path)["state"]
        report["final_state_max_abs_diff"] = float(np.abs(env.sim.get_state().flatten() - final).max())
    print(json.dumps(report, indent=1))
    env.close()


if __name__ == "__main__":
    main()
