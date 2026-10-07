"""Check the VLA action path: replay RoboCasa365 expert actions through astra_vla in 16-step chunks.

The episode is restored exactly as RoboCasa's own playback does it (recorded model XML, ep_meta and initial
MuJoCo state, ``playback_dataset.reset_to``); its dataset-order actions are then executed chunk by chunk with
``ChunkExecutor.run_chunk`` (the code the VLA variant uses). Reports, per chunk, the divergence from the
recorded states and, at the end, whether the task checker succeeded. One simulator, no model calls.

    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src nice -n 10 $ASTRA_PYTHON \
        scripts/robocasa-astra/vla/check_chunks.py OpenCabinet 18 [--out runs/vla12/checks]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from robocasa_astra.astra_robodawn.expert import dataset_dir
from robocasa_astra.astra_vla.action_format import CHUNK
from robocasa_astra.astra_vla.chunk_runner import ChunkExecutor


def make_playback_env(root: Path):
    """The environment exactly as RoboCasa's playback script builds it (no cameras needed)."""
    import robocasa  # noqa: F401
    import robosuite
    from robocasa.utils import lerobot_utils as LU

    meta = LU.get_env_metadata(root)
    kwargs = dict(meta["env_kwargs"], env_name=meta["env_name"], has_renderer=False,
                  has_offscreen_renderer=False, use_camera_obs=False)
    return robosuite.make(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task")
    parser.add_argument("episode", type=int)
    parser.add_argument("--out", default="runs/vla12/checks")
    parser.add_argument("--official", action="store_true",
                        help="reference: step RoboCasa's own reordered actions (reorder_lerobot_action) instead")
    args = parser.parse_args()

    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
    from robocasa.utils import lerobot_utils as LU

    root = Path(dataset_dir(args.task))
    states = LU.get_episode_states(root, args.episode)
    initial = {"states": states[0], "model": LU.get_episode_model_xml(root, args.episode),
               "ep_meta": json.dumps(LU.get_episode_meta(root, args.episode))}
    import pandas as pd

    frame = pd.read_parquet(next(root.glob(f"data/*/episode_{args.episode:06d}.parquet")))
    actions = np.stack(frame["action"].to_list())  # dataset (modality) order, as a VLA outputs them

    env = make_playback_env(root)
    reset_to(env, initial)
    runner = ChunkExecutor(env, step_budget=len(actions))
    rows, max_err = [], 0.0
    for start in range(0, len(actions), CHUNK):
        chunk = actions[start:start + CHUNK]
        if args.official:
            for action in LU.reorder_lerobot_action(chunk, root):
                runner._step(action, "official playback")
            result = {"steps": len(chunk), "task_success": runner.success, "command": "official reorder"}
        else:
            result = runner.run_chunk(chunk)
        t = runner.steps_used
        err = float(np.linalg.norm(states[t] - env.sim.get_state().flatten())) if t < len(states) else float("nan")
        max_err = max(max_err, err) if np.isfinite(err) else max_err
        rows.append({"chunk": start // CHUNK + 1, "steps": result["steps"], "state_error": round(err, 6),
                     "success": result["task_success"], "summary": result["command"]})
        if result["task_success"]:
            break
    report = {"task": args.task, "episode": args.episode, "path": "official" if args.official else "astra_vla", "dataset_steps": len(actions), "chunk": CHUNK,
              "steps_executed": runner.steps_used, "chunks": len(rows), "success": runner.success,
              "max_state_error": round(max_err, 6), "per_chunk": rows}
    env.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.task}-ep{args.episode}{'-official' if args.official else ''}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_chunk"}))


if __name__ == "__main__":
    main()
