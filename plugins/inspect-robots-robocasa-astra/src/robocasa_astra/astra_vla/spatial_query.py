"""Frozen pixel-to-robot geometry; queries never move physics."""

from __future__ import annotations

import numpy as np

from ..depth_query import query_depth


def snapshot_geometry(env, executor, depths: dict) -> dict:
    """Copy calibration and robot frames at the RGB/depth observation step."""
    model, data = env.sim.model, env.sim.data
    cameras = {}
    for name, depth in depths.items():
        height, width = depth.shape
        cid = model.camera_name2id(name)
        f = height / (2 * np.tan(np.deg2rad(model.cam_fovy[cid]) / 2))
        cameras[name] = {
            "intrinsics": [[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]],
            "optical_to_world": (data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])).tolist(),
            "position_world_m": data.cam_xpos[cid].copy().tolist(),
            "far_m": float(model.vis.map.zfar * model.stat.extent),
        }
    pose = executor.pose()
    parts = {
        "gripper": {
            "position_world_m": pose.tip.tolist(),
            "rotation_world": pose.tip_rot.tolist(),
            "description": "fingertip-center grip site; local x closing, z approach",
        },
        "base": {
            "position_world_m": pose.base.tolist(),
            "rotation_world": pose.base_rot.tolist(),
            "description": "base center site; local x forward, y left, z up",
        },
    }
    robot = env.robots[0]
    prefixes = [robot.robot_model.naming_prefix, robot.robot_model.base.naming_prefix]
    prefixes += [g.naming_prefix for g in robot.gripper.values()]
    for kind, names, positions, rotations in (
        ("body", model.body_names, data.body_xpos, data.body_xmat),
        ("site", model.site_names, data.site_xpos, data.site_xmat),
    ):
        for index, name in enumerate(names):
            if name and any(name.startswith(prefix) for prefix in prefixes):
                parts[f"{kind}:{name}"] = {
                    "position_world_m": positions[index].copy().tolist(),
                    "rotation_world": rotations[index].reshape(3, 3).copy().tolist(),
                    "description": f"robot {kind} origin and local axes, not nearest surface",
                }
    return {"cameras": cameras, "robot_parts": parts, "steps_used": executor.steps_used}


def query_spatial(depths: dict, geometry: dict, request: dict, observation_id: str) -> dict:
    """Unproject a visible surface pixel and measure to a frozen robot part origin."""
    if request.get("kind") != "spatial" or request.get("radius") != 0:
        raise ValueError("spatial query requires kind=spatial and radius=0")
    name = request.get("robot_part")
    if name not in geometry["robot_parts"]:
        raise ValueError("unknown robot_part; use a listed part")
    answer = query_depth(depths, {**request, "kind": "pixel"}, observation_id, {"pixel"})
    camera = geometry["cameras"][request["camera"]]
    z = answer["pixel_depth_m"]
    if z >= camera["far_m"] * 0.9999:
        raise ValueError("no measurable surface before camera far plane")
    optical = np.linalg.solve(camera["intrinsics"], [request["u"], request["v"], 1]) * z
    point = (
        np.asarray(camera["position_world_m"]) + np.asarray(camera["optical_to_world"]) @ optical
    )
    part = geometry["robot_parts"][name]
    delta = point - np.asarray(part["position_world_m"])
    return {
        "kind": "spatial",
        "observation_id": observation_id,
        "camera": request["camera"],
        "pixel_uv": [request["u"], request["v"]],
        "robot_part": name,
        "unit": "m",
        "camera_z_m": z,
        "point_world_m": point.tolist(),
        "robot_part_world_m": part["position_world_m"],
        "delta_world_m": delta.tolist(),
        "delta_part_local_m": (np.asarray(part["rotation_world"]).T @ delta).tolist(),
        "delta_base_local_m": (
            np.asarray(geometry["robot_parts"]["base"]["rotation_world"]).T @ delta
        ).tolist(),
        "distance_m": float(np.linalg.norm(delta)),
        "steps_used": geometry["steps_used"],
        "meaning": (
            "vector FROM robot origin TO visible pixel surface; "
            "straight-line range, not a collision-free path"
        ),
        "part_axes": part["description"],
        "base_axes": "x forward, y left, z up",
    }
