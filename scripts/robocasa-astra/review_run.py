"""Replay a finished run exactly (seed mode) and report task sub-goals at the end (no model calls).

The binary task checker hides partial progress, so this reports the task-specific conditions that
make up each RoboCasa success check, plus the replay's final-state difference. One simulator only.

    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src nice -n 10 $ASTRA_PYTHON \
        scripts/robocasa-astra/review_run.py runs/robodawn/<run>
"""

import argparse
import json
from pathlib import Path

import numpy as np

from robocasa_astra.astra_robodawn.replay import rebuild_matching


def subgoals(env, task: str) -> dict:
    """Task-specific components of RoboCasa's ``_check_success`` (same helpers and thresholds)."""
    from robocasa.utils import object_utils as OU

    if task == "OpenCabinet":
        return {k.split("_")[-1]: round(float(v), 2) for k, v in
                env.fxtr.get_joint_state(env, env.fxtr.door_joint_names).items()} | {"needed": ">= 0.90 each"}
    if task == "PickPlaceSinkToCounter":
        return {"obj_in_container": bool(OU.check_obj_in_receptacle(env, "obj", "container")),
                "container_on_counter": bool(env.check_contact(env.objects["container"], env.counter)),
                "gripper_far_from_obj": bool(OU.gripper_obj_far(env)),
                "obj_grasped_at_end": bool(OU.check_obj_grasped(env, "obj"))}
    if task == "PrepareCoffee":
        return {"mug_under_dispenser": bool(env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")),
                "gripper_far_from_mug": bool(OU.gripper_obj_far(env)),
                "machine_turned_on": bool(env.coffee_machine._turned_on),
                "gripper_far_from_button": bool(env.coffee_machine.gripper_button_far(env))}
    if task == "PanTransfer":
        return {"vegetable_on_plate": bool(OU.check_obj_in_receptacle(env, "vegetable", "plate")),
                "pan_on_burner": env._check_obj_location_on_stove("vegetable_container") is not None,
                "gripper_far_from_pan": bool(OU.gripper_obj_far(env, "vegetable_container")),
                "robot_touched_food": bool(env._robot_touched_food)}
    if task == "StirVegetables":
        return {"veg1_in_pot": bool(OU.check_obj_in_receptacle(env, "veg1", "pot")),
                "veg2_in_pot": bool(OU.check_obj_in_receptacle(env, "veg2", "pot")),
                "spatula_grasped": bool(OU.check_obj_grasped(env, "spatula")),
                "stir_steps": int(env.success_time), "needed_stir_steps": 5}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    args = parser.parse_args()
    folder = Path(args.run_dir) / "replay"
    actions = np.load(folder / "actions.npy")
    sim, scene, tries = rebuild_matching(folder)
    env = sim.env
    first_success = None
    for i, action in enumerate(actions):
        env.step(action)
        if first_success is None and env._check_success():
            first_success = i + 1
    final = np.load(folder / "final_state.npz")["state"]
    report = {
        "task": scene["task"], "steps": int(len(actions)), "first_success_step": first_success,
        "scene_builds_until_initial_state_matched": tries,
        "replay_final_state_max_abs_diff": float(np.abs(env.sim.get_state().flatten() - final).max()),
        "subgoals_at_end": subgoals(env, scene["task"]),
    }
    print(json.dumps(report, indent=1))
    (Path(args.run_dir) / "review.json").write_text(json.dumps(report, indent=1))
    env.close()


if __name__ == "__main__":
    main()
