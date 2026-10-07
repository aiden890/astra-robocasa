"""Host-side handle on the VLA simulator process (``astra_vla.sim_server``); requests and closing are inherited."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..astra_robodawn.sim_client import SimClient

SERVER_MODULE = "robocasa_astra.astra_vla.sim_server"


class ChunkSimClient(SimClient):
    """Starts ``python -m robocasa_astra.astra_vla.sim_server`` at low CPU priority (same protocol + act_chunk)."""

    def __init__(self, task: str, budget: int, output: Path, python: str | None = None, nice: int = 10,
                 scene_dir: str | None = None):
        self.output = Path(output)
        self.log_path = self.output / "sim.log"
        self._log = self.log_path.open("w")
        command = ["nice", "-n", str(nice), python or sys.executable, "-m", SERVER_MODULE,
                   "--task", task, "--budget", str(budget), "--output", str(self.output)]
        if scene_dir:
            command += ["--scene-dir", str(scene_dir)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                        text=True, bufsize=1, env=dict(os.environ))
