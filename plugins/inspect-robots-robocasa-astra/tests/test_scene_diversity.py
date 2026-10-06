"""Keep explicit kitchen selection and snapshot restoration independent of random seed."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from robocasa_astra.worker import Simulator


@pytest.fixture
def constructor(monkeypatch):
    """Capture native environment settings without loading GPU or simulator dependencies."""
    captured = {}

    def create_env(task, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("captured constructor")

    modules = {
        name: ModuleType(name)
        for name in (
            "robocasa",
            "robocasa.utils",
            "robocasa.utils.env_utils",
            "robosuite",
            "robosuite.environments",
            "robosuite.environments.base",
        )
    }
    modules["robocasa.utils.env_utils"].create_env = create_env
    modules["robosuite.environments.base"].REGISTERED_ENVS = {
        "PrepareCoffee": SimpleNamespace(EXCLUDE_LAYOUTS=[3], EXCLUDE_STYLES=[4])
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("robocasa_astra.worker.protect_assets", lambda: None)
    return captured


def test_explicit_kitchen_selection(constructor):
    """A recorded layout/style pair reaches native construction without being truncated."""
    with pytest.raises(RuntimeError, match="captured constructor"):
        Simulator("PandaOmron", "PrepareCoffee", layout_id=7, style_id=8).reset(123)
    assert constructor["layout_ids"] == [7]
    assert constructor["style_ids"] == [8]


def test_frozen_scene_uses_recorded_kitchen(constructor, tmp_path):
    """Restore compatible kitchen references before applying saved model arrays."""
    (tmp_path / "episode-meta.json").write_text(json.dumps({"layout_id": 9, "style_id": 10}))
    with pytest.raises(RuntimeError, match="captured constructor"):
        Simulator("PandaOmron", "PrepareCoffee", frozen_scene=tmp_path).reset(123)
    assert constructor["layout_ids"] == [9]
    assert constructor["style_ids"] == [10]


def test_incompatible_kitchen_rejected(constructor):
    """Task-specific excluded layouts never silently fall back to another kitchen."""
    with pytest.raises(ValueError, match="incompatible"):
        Simulator("PandaOmron", "PrepareCoffee", layout_id=3, style_id=2).reset(123)
    assert not constructor
