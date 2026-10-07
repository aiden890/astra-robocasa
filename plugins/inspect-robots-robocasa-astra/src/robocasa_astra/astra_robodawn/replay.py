"""Rebuild a recorded run's scene so that replaying its native actions is exact (simulator side).

RoboCasa scene construction is not fully deterministic for a given seed: the robot's spawn pose and
initial arm joints (and sometimes object poses) can come out in one of a few variants (observed in
P6 on PickPlaceSinkToCounter: the base differed by ~2 cm in some builds, also with global RNGs
seeded and PYTHONHASHSEED fixed). The run's exact initial MuJoCo state is saved, so the scene is
rebuilt until it matches that state; the controllers are then the ones of a normal reset and the
saved actions reproduce the run exactly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..worker import Simulator

MATCH_TOL = 1e-9


def rebuild_matching(replay_dir: Path, max_tries: int = 12):
    """Return (simulator, scene, tries) whose initial state equals ``initial_state.npz``.

    Raises RuntimeError if no build matches within ``max_tries``.
    """
    folder = Path(replay_dir)
    scene = json.loads((folder / "scene.json").read_text())
    reference = np.load(folder / "initial_state.npz")["state"]
    best = np.inf
    for attempt in range(1, max_tries + 1):
        sim = Simulator("PandaOmron", scene["task"], horizon=scene["horizon"])
        sim.reset(scene["seed"])
        diff = float(np.abs(sim.env.sim.get_state().flatten() - reference).max())
        if diff < MATCH_TOL:
            return sim, scene, attempt
        best = min(best, diff)
        sim.env.close()
    raise RuntimeError(f"no scene build matched the recorded initial state in {max_tries} tries (best diff {best:.3g})")
