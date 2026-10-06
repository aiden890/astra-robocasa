"""Guard frozen scenes against incompatible identities, corruption and escaping assets."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_astra.frozen_scene import export_scene, sha256


def test_export_deduplicates_assets_and_keeps_state(tmp_path):
    """A saved scene retains exact numeric state and content-addressed mesh bytes."""
    mesh = tmp_path / "mesh.obj"
    mesh.write_text("mesh bytes")
    state = np.array([0.0, 1.25, -2.5])
    simulator = SimpleNamespace(
        task="PrepareCoffee",
        robot="PandaOmron",
        horizon=1800,
        env=SimpleNamespace(
            sim=SimpleNamespace(
                model=SimpleNamespace(
                    _model=SimpleNamespace(
                        body_pos=np.zeros((2, 3)),
                        stat=SimpleNamespace(extent=1.0),
                        opt=SimpleNamespace(timestep=0.05),
                        vis=SimpleNamespace(
                            **{
                                name: SimpleNamespace()
                                for name in (
                                    "global_",
                                    "headlight",
                                    "map",
                                    "quality",
                                    "rgba",
                                    "scale",
                                )
                            }
                        ),
                    ),
                    get_xml=lambda: f'<mujoco><asset><mesh file="{mesh}"/></asset></mujoco>',
                ),
                get_state=lambda: SimpleNamespace(flatten=lambda: state),
                data=SimpleNamespace(_data=SimpleNamespace(qpos=state.copy())),
            ),
            get_ep_meta=lambda: {"lang": "Prepare coffee"},
        ),
        observe=lambda: {"images": {"left": "camera bytes"}},
    )
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
        setattr(simulator.env.sim.data._data, name, np.zeros(0))
    simulator.env.sim._render_context_offscreen = SimpleNamespace(
        vopt=SimpleNamespace(geomgroup=np.array([0, 1])),
        scn=SimpleNamespace(flags=np.array([1, 0])),
    )
    folder = export_scene(simulator, tmp_path / "collection", 123, 456)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert np.array_equal(np.load(folder / "initial-state.npy"), state)
    assert manifest["robot"] == "PandaOmron"
    asset = tmp_path / "collection/assets" / (sha256(mesh) + ".obj")
    assert asset.read_bytes() == mesh.read_bytes()
    assert "../../assets/" in (folder / "model.xml").read_text()
    with pytest.raises(FileExistsError):
        export_scene(simulator, tmp_path / "collection", 123, 456)


def test_restore_rejects_corrupt_bundle_before_reset(tmp_path, monkeypatch):
    """Corrupt state files cannot reach the native reset API."""
    import sys
    from types import ModuleType

    from robocasa_astra.frozen_scene import restore_scene

    native = ModuleType("robocasa.scripts.dataset_scripts.playback_dataset_hdf5")
    native.reset_to = lambda *args: pytest.fail("Corrupted bundle reached native reset")
    monkeypatch.setitem(sys.modules, native.__name__, native)
    folder = tmp_path / "collection/scenes/PrepareCoffee-123"
    folder.mkdir(parents=True)
    (folder / "initial-state.npy").write_bytes(b"changed")
    (folder / "manifest.json").write_text(
        json.dumps(
            {
                "robot": "PandaOmron",
                "task": "PrepareCoffee",
                "simulator_seed": 123,
                "files": {"initial-state.npy": "wrong"},
            }
        )
    )
    sim = SimpleNamespace(robot="PandaOmron", task="PrepareCoffee")
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_scene(sim, folder, 123)
    with pytest.raises(ValueError, match="seed mismatch"):
        restore_scene(sim, folder, 124)
