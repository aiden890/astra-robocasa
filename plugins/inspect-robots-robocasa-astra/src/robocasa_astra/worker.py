"""Spark2 simulator worker, speaking JSON lines over a private SSH session."""

import argparse
import base64
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import traceback

import numpy as np


def protect_assets():
    """Redirect generated object XML into private temporary directories."""
    import robocasa.models.objects.objects as module

    class PathProxy:
        def __getattr__(self, name):
            return getattr(os.path, name)

        def join(self, *parts):
            if (
                len(parts) == 2
                and str(parts[0]).startswith("/opt/robocasa/")
                and re.fullmatch(r"[0-9_]+\.xml", str(parts[1]))
            ):
                private = tempfile.mkdtemp(prefix="astra-xml-")
                for entry in os.scandir(parts[0]):
                    os.symlink(entry.path, os.path.join(private, entry.name))
                return os.path.join(private, parts[1])
            return os.path.join(*parts)

    class OSProxy:
        path = PathProxy()

        def __getattr__(self, name):
            return getattr(os, name)

    module.os = OSProxy()


class Simulator:
    """Own one native environment; retain actual controller and success semantics."""

    def __init__(self, robot, task, fixture=None, placement=False):
        self.robot, self.task = robot, task
        self.fixture, self.placement = fixture, placement
        self.env = None
        self.steps = 0
        self.streak = 0

    def reset(self, seed):
        """Restore the selected robot scene with a reproducible seed."""
        import robocasa  # noqa: F401
        import robosuite
        from robocasa.utils.env_utils import create_env

        protect_assets()
        if self.env:
            self.env.close()
        if self.fixture and self.robot == "PandaOmron":
            import gzip
            from pathlib import Path
            from xml.etree import ElementTree as ET

            from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import reset_to

            folder = Path(self.fixture)
            kwargs = json.loads((folder / "dataset_meta.json").read_text())["env_args"][
                "env_kwargs"
            ]
            kwargs.update(
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=False,
                seed=seed,
                control_freq=20,
                horizon=1800,
            )
            self.env = robosuite.make(**kwargs)
            xml = ET.fromstring(gzip.decompress((folder / "model.xml.gz").read_bytes()).decode())
            for e in xml.iter():
                value = e.get("file")
                if value:
                    for package in ("robocasa", "robosuite"):
                        marker = package + "/models/assets/"
                        if marker in value:
                            e.set(
                                "file", "/opt/" + package + "/" + marker + value.split(marker, 1)[1]
                            )
            rows = json.loads((folder.parent / "episodes.json").read_text())
            row = next(r for r in rows if r["folder"] == folder.name)
            frame = row["start"] if self.placement else 0
            reset_to(
                self.env,
                {
                    "model": ET.tostring(xml, encoding="unicode"),
                    "states": np.load(folder / "states.npz")["states"][frame],
                    "ep_meta": (folder / "ep_meta.json").read_text(),
                },
            )
        else:
            self.env = create_env(
                self.task,
                robots=self.robot,
                seed=seed,
                layout_ids=[1],
                style_ids=[1],
                camera_names=[
                    "robot0_agentview_left",
                    "robot0_agentview_right",
                    "robot0_eye_in_right_hand",
                    "robot0_eye_in_left_hand",
                ],
                camera_widths=256,
                camera_heights=256,
                generative_textures=None,
                control_freq=20,
                horizon=1800,
            )
            self.env.reset()
        if self.env.control_freq != 20:
            raise ValueError("Native control frequency must be 20 Hz")
        self.steps, self.streak = 0, 0
        robot = self.env.robots[0]
        self.parts = {key: list(value) for key, value in robot._action_split_indexes.items()}
        return self.observe()

    def observe(self):
        """Render real cameras and report separate native and placement outcomes."""
        from PIL import Image

        env = self.env
        raw = env._get_observations(force_update=True)
        state = {
            k: np.asarray(v).tolist()
            for k, v in raw.items()
            if not k.endswith(("image", "depth")) and np.asarray(v).size < 100
        }
        images = {}
        names = set(env.sim.model.camera_names)
        for suffix in (
            "agentview_left",
            "agentview_right",
            "eye_in_hand",
            "right_eye_in_hand",
            "left_eye_in_hand",
            "eye_in_right_hand",
            "eye_in_left_hand",
        ):
            camera = "robot0_" + suffix
            if camera not in names:
                continue
            image = env.sim.render(width=256, height=256, camera_name=camera)[::-1].copy()
            buf = io.BytesIO()
            Image.fromarray(image).save(buf, format="PNG")
            images[camera] = base64.b64encode(buf.getvalue()).decode()
        success = bool(env._check_success())
        placement_success = False
        if hasattr(env, "coffee_machine"):
            from robocasa.utils import object_utils as OU

            placed = bool(env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj"))
            released = bool(OU.gripper_obj_far(env))
            self.streak = self.streak + 1 if placed and released else 0
            placement_success = self.streak >= 5
        info = {
            "native_task_success": success,
            "placement_success": placement_success,
            "success": placement_success if self.placement else success,
            "steps": self.steps,
            "control_hz": self.env.control_freq,
            "robot": self.robot,
            "task": self.task,
            "action_parts": self.parts,
            "hybrid_mode_index": env.action_dim - 1,
            "hybrid_mode_semantics": "last index: <=0 arm mode, >0 base mode",
            "action_dim": env.action_dim,
            "action_low": env.action_spec[0].tolist(),
            "action_high": env.action_spec[1].tolist(),
            "controller": env.robots[0].composite_controller_config,
        }
        return {"images": images, "state": state, "info": info}

    def step(self, action):
        """Apply a finite bounded action using the native controller."""
        a = np.asarray(action, dtype=float)
        low, high = self.env.action_spec
        if a.shape != low.shape or not np.isfinite(a).all() or np.any(a < low) or np.any(a > high):
            raise ValueError("Action violates actual native shape/bounds")
        self.env.step(a)
        self.steps += 1
        return self.observe()


def main():
    """Serve only stdin requests; no public network listener or credentials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", default="PandaOmron")
    parser.add_argument("--task", default="PrepareCoffee")
    parser.add_argument("--fixture")
    parser.add_argument("--placement", action="store_true")
    args = parser.parse_args()
    with contextlib.redirect_stdout(sys.stderr):
        sim = Simulator(args.robot, args.task, args.fixture, args.placement)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                if request["op"] == "reset":
                    result = sim.reset(request["seed"])
                elif request["op"] == "step":
                    result = sim.step(request["action"])
                elif request["op"] == "replay":
                    for action in request["actions"]:
                        a = np.asarray(action, dtype=float)
                        low, high = sim.env.action_spec
                        if (
                            a.shape != low.shape
                            or not np.isfinite(a).all()
                            or np.any(a < low)
                            or np.any(a > high)
                        ):
                            raise ValueError("Invalid replay action")
                        sim.env.step(a)
                        sim.steps += 1
                    result = sim.observe()
                elif request["op"] == "close":
                    if sim.env:
                        sim.env.close()
                    break
                else:
                    raise ValueError("Unknown operation")
            print(json.dumps(result), flush=True)
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({"error": type(error).__name__, "detail": str(error)}), flush=True)


if __name__ == "__main__":
    main()
