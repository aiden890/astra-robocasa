"""Host-side handle on the simulator process (``sim_server``), at low CPU priority."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from pathlib import Path

REPLY_TIMEOUT_S = 600  # a single command is at most a few hundred native steps


class SimulatorError(RuntimeError):
    """The simulator process failed or replied with an error; details are in ``sim.log``."""


class SimClient:
    """Starts ``python -m robocasa_astra.astra_robodawn.sim_server`` and exchanges JSON lines with it."""

    def __init__(self, task: str, budget: int, output: Path, python: str | None = None, nice: int = 10):
        self.output = Path(output)
        self.log_path = self.output / "sim.log"
        self._log = self.log_path.open("w")
        command = ["nice", "-n", str(nice), python or sys.executable, "-m", "robocasa_astra.astra_robodawn.sim_server",
                   "--task", task, "--budget", str(budget), "--output", str(self.output)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log,
                                        text=True, bufsize=1, env=dict(os.environ))

    def request(self, op: str, **fields) -> dict:
        self.process.stdin.write(json.dumps({"op": op, **fields}) + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select([self.process.stdout], [], [], REPLY_TIMEOUT_S)
        if not ready:
            raise SimulatorError(f"no reply to {op!r} within {REPLY_TIMEOUT_S} s; see {self.log_path}")
        line = self.process.stdout.readline()
        if not line:
            raise SimulatorError(f"simulator exited during {op!r}; see {self.log_path}")
        reply = json.loads(line)
        if "error" in reply:
            raise SimulatorError(f"{op}: {reply['error']}: {reply['detail']} (see {self.log_path})")
        return reply

    def close(self) -> dict:
        summary = {}
        if self.process.poll() is None:
            try:
                summary = self.request("close")
                self.process.wait(timeout=60)
            except Exception:  # noqa: BLE001 - make sure the process does not outlive the run
                self.process.terminate()
                self.process.wait(timeout=30)
        self._log.close()
        return summary
