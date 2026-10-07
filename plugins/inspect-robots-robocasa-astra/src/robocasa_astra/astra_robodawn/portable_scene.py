"""Restore a common frozen scene on a different CPU platform (x86_64 here; the scenes were built on aarch64).

Same procedure as ``robocasa_astra.frozen_scene.restore_scene`` (package versions, file checksums, native
XML, saved model and data arrays, renderer settings, exact initial state, controller reset) with two
documented relaxations, measured on amp2 (P7):

1. The nine ``mesh_poly*`` arrays are kept as compiled locally. MuJoCo 3.3.1 on x86_64 builds
   ``mesh_polymap``/``mesh_polyvert`` two elements shorter than on aarch64, and these arrays index
   each other, so they cannot be overwritten piecewise. They describe mesh polygons, not inertia,
   geometry poses or the physics state.
2. World and camera geometry after the restore is compared numerically instead of byte for byte,
   and the maximum differences are recorded in the receipt.
3. Arrays whose element type differs only by the platform's ``char`` signedness (e.g. ``plugin_attr``:
   uint8 on aarch64, int8 on x86_64) are copied byte for byte.

The physics state itself is set from the saved bytes and checked to be byte-identical. Results from
this restore are a separate environment from the reference aarch64 one (see evaluation/README.md).
"""

from __future__ import annotations

import json
import platform
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from ..frozen_scene import sha256

PLATFORM_DEPENDENT_ARRAYS = ("mesh_poly",)
GEOMETRY_FIELDS = ("geom_xpos", "geom_xmat", "cam_xpos", "cam_xmat", "xpos", "xquat", "site_xpos", "site_xmat")


def _array(folder: Path, description: dict) -> np.ndarray:
    raw = b"".join(zlib.decompress((folder / piece).read_bytes()) for piece in description["pieces"])
    return np.frombuffer(raw, dtype=np.dtype(description["dtype"])).reshape(description["shape"])


def restore_portable(simulator, folder, seed) -> dict:
    """Restore ``folder`` into ``simulator`` (already reset in the scene's layout/style); return the receipt."""
    from importlib.metadata import version

    from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import reset_to

    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest["robot"], manifest["task"], manifest["simulator_seed"]) != (simulator.robot, simulator.task, seed):
        raise ValueError("Frozen scene robot, task or simulator seed mismatch")
    collection = folder.parents[1]
    lock = json.loads((collection / "environment.lock.json").read_text())
    for package in ("numpy", "mujoco", "robosuite", "robocasa"):
        if version(package) != lock["packages"][package]:
            raise ValueError("Frozen scene runtime package mismatch: " + package)
    for relative, expected in manifest["files"].items():
        path = (folder / relative).resolve()
        if collection not in path.parents or sha256(path) != expected:
            raise ValueError("Frozen scene path or checksum mismatch: " + relative)
    xml = ET.fromstring((folder / "model.xml").read_text())
    for element in xml.iter():
        if element.get("file"):
            element.set("file", str((folder / element.get("file")).resolve(strict=True)))
    state = np.load(folder / "initial-state.npy", allow_pickle=False)
    env = simulator.env
    reset_to(env, {"model": ET.tostring(xml, encoding="unicode"), "states": state,
                   "ep_meta": (folder / "episode-meta.json").read_text()})
    model = env.sim.model._model
    kept_local, reinterpreted, overwritten = [], [], 0
    for name, description in json.loads((folder / "model-arrays.json").read_text()).items():
        destination = getattr(model, name)
        if name.startswith(PLATFORM_DEPENDENT_ARRAYS):
            kept_local.append(f"{name} local {tuple(destination.shape)} saved {tuple(description['shape'])}")
            continue
        saved = _array(folder, description)
        if destination.shape != saved.shape:
            raise ValueError("Frozen model structure mismatch: " + name)
        if destination.dtype != saved.dtype:
            if destination.dtype.itemsize != saved.dtype.itemsize:
                raise ValueError("Frozen model element type mismatch: " + name)
            saved = saved.view(destination.dtype)
            reinterpreted.append(f"{name} {description['dtype']} -> {destination.dtype}")
        destination[...] = saved
        overwritten += 1
    for section, fields in json.loads((folder / "model-statistics.json").read_text()).items():
        obj = model
        for part in section.split("."):
            obj = getattr(obj, part)
        for name, saved in fields.items():
            destination = getattr(obj, name)
            if isinstance(destination, np.ndarray):
                destination[...] = saved
            else:
                setattr(obj, name, saved)
    context = env.sim._render_context_offscreen
    context.gl_ctx.make_current()
    context.con.free()
    context._set_mujoco_context_and_buffers()
    render_options = json.loads((folder / "renderer-settings.json").read_text())
    context.scn.flags[...] = render_options.pop("scene_flags")
    for name, saved in render_options.items():
        destination = getattr(context.vopt, name)
        if isinstance(destination, np.ndarray):
            destination[...] = saved
        else:
            setattr(context.vopt, name, saved)
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    data_state = json.loads((folder / "data-arrays.json").read_text())
    data = env.sim.data._data
    for name, description in data_state.items():
        destination = getattr(data, name)
        saved = _array(folder, description)
        if destination.shape != saved.shape or destination.dtype.itemsize != saved.dtype.itemsize:
            raise ValueError("Frozen simulation data structure mismatch: " + name)
        destination[...] = saved.view(destination.dtype) if destination.dtype != saved.dtype else saved
    for controller in env.robots[0].part_controllers.values():
        controller.update(force=True)
        controller.reset_goal()
    for name in ("ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied", "userdata"):
        getattr(data, name)[...] = _array(folder, data_state[name])
    geometry = {}
    for name in GEOMETRY_FIELDS:
        actual = np.asarray(getattr(data, name), dtype=float)
        saved = _array(folder, data_state[name]).astype(float)
        geometry[name] = float(np.abs(actual - saved).max()) if actual.size else 0.0
    actual_state = env.sim.get_state().flatten()
    if actual_state.dtype != state.dtype or actual_state.tobytes() != state.tobytes():
        raise ValueError("Restored simulator state is not byte-equivalent")
    simulator.frozen_scene_receipt = {
        "restore": "portable (astra_robodawn.portable_scene)",
        "platform": f"{platform.machine()} / python {platform.python_version()}",
        "reference_platform": f"{lock.get('platform')} / python {lock.get('python')}",
        "manifest_sha256": sha256(folder / "manifest.json"),
        "state_sha256": manifest["files"]["initial-state.npy"],
        "state_exact": True,
        "model_arrays_overwritten": overwritten,
        "model_arrays_kept_local": kept_local,
        "model_arrays_byte_reinterpreted": reinterpreted,
        "world_geometry_max_abs_diff": geometry,
        "robot": manifest["robot"],
    }
    return manifest


def open_scene(task: str, horizon: int, scene_dir, seed: int):
    """A ``worker.Simulator`` in the scene's kitchen with the frozen scene restored portably.

    The receipt (``simulator.frozen_scene_receipt``) also records whether the first rendered camera
    images equal the ones stored with the scene (a diagnostic; physics equality does not need it).
    """
    from ..frozen_scene import image_hashes
    from ..worker import Simulator

    meta = json.loads((Path(scene_dir) / "episode-meta.json").read_text())
    simulator = Simulator("PandaOmron", task, horizon=horizon, layout_id=meta["layout_id"], style_id=meta["style_id"])
    simulator.reset(seed)
    manifest = restore_portable(simulator, scene_dir, seed)
    simulator.steps, simulator.streak = 0, 0
    simulator.frozen_scene_receipt["initial_images_exact"] = image_hashes(simulator.observe()) == manifest["initial_images"]
    return simulator
