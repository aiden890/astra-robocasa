"""Portable, checksummed RoboCasa initial scenes with exact simulator-state restoration."""

import hashlib
import json
import shutil
import uuid
import zlib
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np


def sha256(path):
    """Hash complete file bytes without loading large meshes into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hashes(observation):
    """Identify the exact encoded initial camera frames returned by the worker."""
    return {
        name: hashlib.sha256(value.encode()).hexdigest()
        for name, value in observation["images"].items()
    }


def save_model_arrays(arrays, folder, checksums, filename="model-arrays.json"):
    """Deduplicate model data in compressed content-addressed blocks across all scenes."""
    pool = folder.parents[1] / "model-blobs"
    pool.mkdir(exist_ok=True)
    descriptions = {}
    for name, array in arrays.items():
        raw = array.tobytes(order="C")
        pieces = []
        for offset in range(0, len(raw), 1024 * 1024):
            chunk = raw[offset : offset + 1024 * 1024]
            digest = hashlib.sha256(chunk).hexdigest()
            target = pool / (digest + ".zlib")
            compressed = zlib.compress(chunk, level=1)
            expected = hashlib.sha256(compressed).hexdigest()
            if not target.exists():
                temporary = pool / (digest + "." + uuid.uuid4().hex + ".tmp")
                temporary.write_bytes(compressed)
                temporary.replace(target)
            elif sha256(target) != expected:
                raise ValueError("Shared model block checksum mismatch")
            relative = "../../model-blobs/" + target.name
            checksums[relative] = expected
            pieces.append(relative)
        descriptions[name] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "pieces": pieces,
        }
    (folder / filename).write_text(json.dumps(descriptions))


def export_scene(simulator, root, seed, rollout_seed):
    """Save one immutable initial state and deduplicate all referenced XML assets."""
    root = Path(root)
    folder = root / "scenes" / f"{simulator.task}-{rollout_seed}"
    folder.mkdir(parents=True, exist_ok=False)
    assets = root / "assets"
    assets.mkdir(exist_ok=True)
    xml = ET.fromstring(simulator.env.sim.model.get_xml())
    checksums = {}
    for element in xml.iter():
        source = element.get("file")
        if not source:
            continue
        source = Path(source).resolve(strict=True)
        digest = sha256(source)
        target = assets / (digest + source.suffix)
        if not target.exists():
            shutil.copyfile(source, target)
        elif sha256(target) != digest:
            raise ValueError("Existing shared asset checksum mismatch")
        relative = "../../assets/" + target.name
        element.set("file", relative)
        checksums[relative] = digest
    (folder / "model.xml").write_text(ET.tostring(xml, encoding="unicode"))
    np.save(
        folder / "initial-state.npy", simulator.env.sim.get_state().flatten(), allow_pickle=False
    )
    native_model = simulator.env.sim.model._model
    arrays = {}
    for name in dir(native_model):
        value = getattr(native_model, name)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            arrays[name] = value.copy()
    save_model_arrays(arrays, folder, checksums)
    data_arrays = {}
    for name in (
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "mocap_pos",
        "mocap_quat",
        "qacc_warmstart",
        "xfrc_applied",
        "qfrc_applied",
        "userdata",
        "geom_xpos",
        "geom_xmat",
        "cam_xpos",
        "cam_xmat",
        "xpos",
        "xquat",
        "site_xpos",
        "site_xmat",
    ):
        data_arrays[name] = np.asarray(getattr(simulator.env.sim.data._data, name)).copy()
    save_model_arrays(data_arrays, folder, checksums, "data-arrays.json")
    settings = {}
    for section in (
        "stat",
        "opt",
        "vis.global_",
        "vis.headlight",
        "vis.map",
        "vis.quality",
        "vis.rgba",
        "vis.scale",
    ):
        obj = native_model
        for part in section.split("."):
            obj = getattr(obj, part)
        fields = {}
        for name in dir(obj):
            if name.startswith("_"):
                continue
            value = getattr(obj, name)
            if isinstance(value, (int, float, bool, np.ndarray)):
                fields[name] = np.asarray(value).tolist()
        settings[section] = fields
    (folder / "model-statistics.json").write_text(json.dumps(settings))
    context = simulator.env.sim._render_context_offscreen
    render_options = {}
    for name in dir(context.vopt):
        if name.startswith("_"):
            continue
        value = getattr(context.vopt, name)
        if isinstance(value, (int, float, bool, np.ndarray)):
            render_options[name] = np.asarray(value).tolist()
    render_options["scene_flags"] = np.asarray(context.scn.flags).tolist()
    (folder / "renderer-settings.json").write_text(json.dumps(render_options))
    (folder / "episode-meta.json").write_text(json.dumps(simulator.env.get_ep_meta()))
    for name in (
        "model.xml",
        "initial-state.npy",
        "model-arrays.json",
        "data-arrays.json",
        "model-statistics.json",
        "renderer-settings.json",
        "episode-meta.json",
    ):
        checksums[name] = sha256(folder / name)
    observation = simulator.observe()
    manifest = {
        "schema": 1,
        "task": simulator.task,
        "robot": simulator.robot,
        "rollout_seed": rollout_seed,
        "simulator_seed": seed,
        "control_hz": 20,
        "horizon": simulator.horizon,
        "files": checksums,
        "initial_images": image_hashes(observation),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return folder


def restore_scene(simulator, folder, seed):
    """Reject incompatible or corrupted scenes before restoring the native XML and state."""
    from robocasa.scripts.dataset_scripts.playback_dataset_hdf5 import reset_to

    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if (manifest["robot"], manifest["task"], manifest["simulator_seed"]) != (
        simulator.robot,
        simulator.task,
        seed,
    ):
        raise ValueError("Frozen scene robot, task or simulator seed mismatch")
    collection = folder.parents[1]
    lock_path = collection / "environment.lock.json"
    if lock_path.exists():
        from importlib.metadata import version

        lock = json.loads(lock_path.read_text())
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
    reset_to(
        simulator.env,
        {
            "model": ET.tostring(xml, encoding="unicode"),
            "states": state,
            "ep_meta": (folder / "episode-meta.json").read_text(),
        },
    )
    native_model = simulator.env.sim.model._model
    model_state = json.loads((folder / "model-arrays.json").read_text())
    for name, description in model_state.items():
        destination = getattr(native_model, name)
        pieces = [zlib.decompress((folder / piece).read_bytes()) for piece in description["pieces"]]
        saved = np.frombuffer(b"".join(pieces), dtype=np.dtype(description["dtype"])).reshape(
            description["shape"]
        )
        if destination.shape != saved.shape or destination.dtype != saved.dtype:
            raise ValueError("Frozen model structure mismatch: " + name)
        destination[...] = saved
    settings = json.loads((folder / "model-statistics.json").read_text())
    for section, fields in settings.items():
        obj = native_model
        for part in section.split("."):
            obj = getattr(obj, part)
        for name, saved in fields.items():
            destination = getattr(obj, name)
            if isinstance(destination, np.ndarray):
                destination[...] = saved
            else:
                setattr(obj, name, saved)
    context = simulator.env.sim._render_context_offscreen
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
    simulator.env.sim.set_state_from_flattened(state)
    simulator.env.sim.forward()
    data_state = json.loads((folder / "data-arrays.json").read_text())
    for name, description in data_state.items():
        destination = getattr(simulator.env.sim.data._data, name)
        raw = b"".join(
            zlib.decompress((folder / piece).read_bytes()) for piece in description["pieces"]
        )
        saved = np.frombuffer(raw, dtype=np.dtype(description["dtype"])).reshape(
            description["shape"]
        )
        if destination.shape != saved.shape or destination.dtype != saved.dtype:
            raise ValueError("Frozen simulation data structure mismatch: " + name)
        destination[...] = saved
    for controller in simulator.env.robots[0].part_controllers.values():
        controller.update(force=True)
        controller.reset_goal()
    for name in ("ctrl", "qacc_warmstart", "xfrc_applied", "qfrc_applied", "userdata"):
        description = data_state[name]
        raw = b"".join(
            zlib.decompress((folder / piece).read_bytes()) for piece in description["pieces"]
        )
        saved = np.frombuffer(raw, dtype=np.dtype(description["dtype"])).reshape(
            description["shape"]
        )
        getattr(simulator.env.sim.data._data, name)[...] = saved
    for name in (
        "geom_xpos",
        "geom_xmat",
        "cam_xpos",
        "cam_xmat",
        "xpos",
        "xquat",
        "site_xpos",
        "site_xmat",
    ):
        description = data_state[name]
        raw = b"".join(
            zlib.decompress((folder / piece).read_bytes()) for piece in description["pieces"]
        )
        actual_geometry = np.asarray(getattr(simulator.env.sim.data._data, name))
        if actual_geometry.tobytes() != raw:
            raise ValueError("Frozen world or camera geometry differs: " + name)
    actual = simulator.env.sim.get_state().flatten()
    if actual.dtype != state.dtype or actual.tobytes() != state.tobytes():
        raise ValueError("Restored simulator state is not byte-equivalent")
    simulator.frozen_scene_receipt = {
        "manifest_sha256": sha256(folder / "manifest.json"),
        "state_sha256": manifest["files"]["initial-state.npy"],
        "state_exact": True,
        "model_arrays_exact": True,
        "world_geometry_exact": True,
        "robot": manifest["robot"],
    }
    return manifest
