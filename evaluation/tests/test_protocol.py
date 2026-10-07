"""Guard the common scene protocol without invoking simulators or model accounts."""

import json

import pytest
from robocasa_astra.frozen_scene import sha256
from robocasa_common.evaluate import load_scene, resolve_factory


def test_load_scene_rejects_wrong_robot_and_changed_manifest(tmp_path):
    """Panda identity and the previously verified manifest digest are mandatory."""
    folder = tmp_path / "collection/scenes/task-123"
    folder.mkdir(parents=True)
    manifest = {"robot": "PandaOmron", "files": {}}
    path = folder / "manifest.json"
    path.write_text(json.dumps(manifest))
    (folder / "verified.json").write_text(
        json.dumps(
            {
                "state_exact": True,
                "world_geometry_exact": True,
                "initial_images_exact": False,
                "manifest_sha256": sha256(path),
            }
        )
    )
    assert load_scene(folder)["robot"] == "PandaOmron"
    path.write_text(json.dumps({**manifest, "task": "changed"}))
    with pytest.raises(ValueError, match="changed"):
        load_scene(folder)
    path.write_text(json.dumps({**manifest, "robot": "GR1FloatingBody"}))
    with pytest.raises(ValueError, match="PandaOmron"):
        load_scene(folder)


def test_factory_requires_explicit_module_and_function():
    """Participant policy selection must be unambiguous."""
    with pytest.raises(ValueError, match="module:function"):
        resolve_factory("missing_colon")
    assert resolve_factory("json:loads") is json.loads
