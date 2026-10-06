"""Spark2 simulator worker, speaking JSON lines over a private SSH session."""

import argparse
import base64
import contextlib
import ctypes
import io
import json
import os
import re
import sys
import tempfile
import traceback

import numpy as np

from robocasa_astra.depth import encode_depth, normalize_depth_buffer, validate_depth


@contextlib.contextmanager
def native_diagnostics():
    """Keep Python and native renderer diagnostics off the JSON response stream."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved, 1)
        os.close(saved)


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

    def __init__(
        self,
        robot,
        task,
        fixture=None,
        placement=False,
        horizon=1800,
        face_workstation=False,
        depth=False,
        frozen_scene=None,
        layout_id=None,
        style_id=None,
    ):
        self.robot, self.task = robot, task
        self.fixture, self.placement = fixture, placement
        self.horizon, self.face_workstation = horizon, face_workstation
        self.depth = depth
        self.frozen_scene = frozen_scene
        self.layout_id, self.style_id = layout_id, style_id
        self.frozen_scene_receipt = None
        self.initial_alignment = None
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
                horizon=self.horizon,
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
            from robosuite.environments.base import REGISTERED_ENVS

            task_class = REGISTERED_ENVS[self.task]
            layouts = [i for i in range(1, 11) if i not in task_class.EXCLUDE_LAYOUTS]
            styles = [i for i in range(1, 11) if i not in task_class.EXCLUDE_STYLES]
            if not layouts or not styles:
                raise ValueError("Task has no compatible target layout/style")
            layout_id, style_id = self.layout_id, self.style_id
            if self.frozen_scene:
                from pathlib import Path

                meta = json.loads((Path(self.frozen_scene) / "episode-meta.json").read_text())
                layout_id, style_id = meta["layout_id"], meta["style_id"]
            if layout_id is not None and layout_id not in layouts:
                raise ValueError("Requested layout is incompatible with task")
            if style_id is not None and style_id not in styles:
                raise ValueError("Requested style is incompatible with task")
            self.env = create_env(
                self.task,
                robots=self.robot,
                seed=seed,
                layout_ids=[layout_id] if layout_id is not None else layouts[:1],
                style_ids=[style_id] if style_id is not None else styles[:1],
                camera_names=["robot0_agentview_left", "robot0_agentview_right"]
                + (
                    ["robot0_eye_in_hand"]
                    if self.robot == "PandaOmron"
                    else ["robot0_eye_in_right_hand", "robot0_eye_in_left_hand"]
                ),
                camera_widths=256,
                camera_heights=256,
                generative_textures=None,
                control_freq=20,
                horizon=self.horizon,
            )
            self.env.reset()
        frozen_manifest = None
        if self.frozen_scene:
            from robocasa_astra.frozen_scene import restore_scene

            frozen_manifest = restore_scene(self, self.frozen_scene, seed)
        if self.face_workstation and self.robot == "GR1FloatingBody":
            self.align_workstation()
        if self.env.control_freq != 20:
            raise ValueError("Native control frequency must be 20 Hz")
        self.steps, self.streak = 0, 0
        robot = self.env.robots[0]
        self.parts = {key: list(value) for key, value in robot._action_split_indexes.items()}
        observation = self.observe()
        if frozen_manifest:
            from robocasa_astra.frozen_scene import image_hashes

            self.frozen_scene_receipt["initial_images_exact"] = (
                image_hashes(observation) == frozen_manifest["initial_images"]
            )
        return observation

    def align_workstation(self):
        """Face the native task fixture and preserve collision-free starting clearance."""
        from robocasa.utils import env_utils as EU
        from robosuite.utils import transform_utils as T

        env = self.env
        if env.init_robot_base_ref is None:
            raise ValueError("Task has no explicit starting workstation reference")
        fixture = env.get_fixture(env.init_robot_base_ref)
        raw = env._get_observations(force_update=True)
        position = np.asarray(raw["robot0_base_pos"])
        target = np.asarray(fixture.pos)
        direction = target[:2] - position[:2]
        if np.linalg.norm(direction) < 0.1:
            raise ValueError("Starting fixture direction is ambiguous")
        unit = direction / np.linalg.norm(direction)
        address = env.sim.model.get_joint_qpos_addr("mobilebase0_joint_mobile_yaw")
        total_delta = 0.0
        collision = True
        for attempt in range(7):
            backoff = attempt * 0.1
            if attempt:
                safe_position = position.copy()
                safe_position[:2] -= unit * backoff
                EU.set_robot_to_position(env, safe_position)
            for _ in range(5):
                raw = env._get_observations(force_update=True)
                delta_xy = target[:2] - np.asarray(raw["robot0_base_pos"])[:2]
                desired = np.arctan2(delta_xy[1], delta_xy[0])
                forward = T.quat2mat(np.asarray(raw["robot0_base_quat"]))[:2, 0]
                heading = np.arctan2(forward[1], forward[0])
                delta = float((desired - heading + np.pi) % (2 * np.pi) - np.pi)
                env.sim.data.qpos[address] += delta
                total_delta += delta
                env.sim.forward()
            collision = bool(EU.detect_robot_collision(env))
            if not collision:
                break
        for controller in env.robots[0].part_controllers.values():
            controller.update(force=True)
            controller.reset_goal()
        final_raw = env._get_observations(force_update=True)
        final_direction = target[:2] - np.asarray(final_raw["robot0_base_pos"])[:2]
        desired = np.arctan2(final_direction[1], final_direction[0])
        forward = T.quat2mat(np.asarray(final_raw["robot0_base_quat"]))[:2, 0]
        heading = np.arctan2(forward[1], forward[0])
        error = float(abs((desired - heading + np.pi) % (2 * np.pi) - np.pi))
        self.initial_alignment = {
            "fixture": fixture.name,
            "base_position": position.tolist(),
            "fixture_position": target.tolist(),
            "yaw_delta_rad": total_delta,
            "heading_error_rad": error,
            "robot_collision": collision,
            "method": "native_task_fixture_direction",
            "collision_clearance_backoff_m": backoff,
            "final_base_position": final_raw["robot0_base_pos"].tolist(),
        }
        if error > np.deg2rad(5) or collision:
            raise ValueError(
                "Workstation-facing initialization failed: " + json.dumps(self.initial_alignment)
            )

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
        images, depths, depth_metadata = {}, {}, {}
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
            if self.depth:
                from robosuite.utils.camera_utils import (
                    get_camera_extrinsic_matrix,
                    get_camera_intrinsic_matrix,
                    get_real_depth_map,
                )

                rgb, buffer = env.sim.render(width=256, height=256, camera_name=camera, depth=True)
                image = rgb[::-1].copy()
                metric = validate_depth(
                    get_real_depth_map(env.sim, normalize_depth_buffer(buffer))[::-1], (256, 256)
                )
                depths[camera] = encode_depth(metric)
                depth_metadata[camera] = {
                    "unit": "m",
                    "quantity": "camera optical-axis depth",
                    "encoding": "npy-float32-zlib-base64",
                    "shape": [256, 256],
                    "pixel_origin": "top-left",
                    "simulation_step": self.steps,
                    "simulation_time_s": float(env.sim.data.time),
                    "depth_near_m": float(env.sim.model.vis.map.znear * env.sim.model.stat.extent),
                    "depth_far_m": float(env.sim.model.vis.map.zfar * env.sim.model.stat.extent),
                    "intrinsics": get_camera_intrinsic_matrix(env.sim, camera, 256, 256).tolist(),
                    "camera_to_world": get_camera_extrinsic_matrix(env.sim, camera).tolist(),
                }
            else:
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
            "instruction": env.get_ep_meta().get("lang", self.task),
            "horizon": self.horizon,
            "initial_alignment": self.initial_alignment,
            "frozen_scene": self.frozen_scene_receipt,
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
        return {
            "images": images,
            "depths": depths,
            "depth_metadata": depth_metadata,
            "state": state,
            "info": info,
        }

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
    parser.add_argument("--frozen-scene")
    parser.add_argument("--horizon", type=int, default=1800)
    parser.add_argument("--face-workstation", action="store_true")
    parser.add_argument("--placement", action="store_true")
    parser.add_argument("--depth", action="store_true")
    args = parser.parse_args()
    with native_diagnostics():
        sim = Simulator(
            args.robot,
            args.task,
            args.fixture,
            args.placement,
            args.horizon,
            args.face_workstation,
            args.depth,
            args.frozen_scene,
        )
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with native_diagnostics():
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
