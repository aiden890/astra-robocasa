"""Measure how far a constant 16-step chunk moves / turns the fingertips (the scale in action_format).

Runs in the skill primer's non-evaluation kitchen with ChunkExecutor; after every chunk an all-zero chunk lets
the arm settle. One simulator, no model calls.

    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src nice -n 10 $ASTRA_PYTHON \
        scripts/robocasa-astra/vla/calibrate.py        (writes runs/vla12/checks/calibration.json)
"""

import json
from pathlib import Path

import numpy as np

from robocasa_astra.astra_robodawn.demo_builder import PRIMER_SCENE, make_env
from robocasa_astra.astra_vla.action_format import CHUNK, DIM
from robocasa_astra.astra_vla.chunk_runner import ChunkExecutor


def constant(index: int, value: float) -> np.ndarray:
    chunk = np.zeros((CHUNK, DIM))
    chunk[:, 4], chunk[:, 11] = -1.0, -1.0
    chunk[:, index] = value
    return chunk


def main() -> None:
    scene = PRIMER_SCENE
    env = make_env(scene["task"], scene["seed"], scene["layout"], scene["style"])
    runner = ChunkExecutor(env, step_budget=100000)
    rows = []
    for value in (0.1, 0.2, 0.5, 1.0):
        for index, axis in ((5, 0), (6, 1), (7, 2)):
            for sign in (1.0, -1.0):
                moved = runner.run_chunk(constant(index, sign * value))["moved_cm"][axis]
                runner.run_chunk(constant(5, 0.0))
                rows.append({"index": index, "input": sign * value, "moved_cm": moved,
                             "cm_per_step_per_unit": round(moved / (sign * value) / CHUNK, 3)})
    for value in (0.1, 0.3):
        turned = runner.run_chunk(constant(10, value))["turned_deg"]
        runner.run_chunk(constant(10, -value))
        rows.append({"index": 10, "input": value, "turned_deg": turned,
                     "deg_per_step_per_unit": round(turned / value / CHUNK, 2)})
    env.close()
    out = Path("runs/vla12/checks/calibration.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
