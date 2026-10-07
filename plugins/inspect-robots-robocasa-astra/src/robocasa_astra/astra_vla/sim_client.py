"""Detached simulator transport with exact process adoption and frozen-scene recovery."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..astra_robodawn.sim_client import SimulatorError
from .persistence import locked, read_json

SERVER_MODULE = "robocasa_astra.astra_vla.sim_server"


class ChunkSimClient:
    """Disconnecting the host leaves native physics alive; dead workers restore the action
    journal."""

    def __init__(
        self,
        task: str,
        budget: int,
        output: Path,
        python: str | None = None,
        nice: int = 10,
        scene_dir: str | None = None,
        condition: str = "rgb",
    ):
        self.output = Path(output)
        self.worker = self.output / "worker"
        self.worker.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:24]
        self.socket_path = Path(tempfile.gettempdir()) / f"astra-vla-{name}.sock"
        self.command = [
            "nice",
            "-n",
            str(nice),
            python or sys.executable,
            "-m",
            SERVER_MODULE,
            "--task",
            task,
            "--budget",
            str(budget),
            "--output",
            str(self.output),
            "--condition",
            condition,
            "--socket",
            str(self.socket_path),
        ]
        if scene_dir:
            self.command += ["--scene-dir", scene_dir]
        self.seed = None
        self._ensure_worker()

    def _alive(self):
        from .persistence import exact_process

        return exact_process(read_json(self.worker / "lease.json", {}))

    def _ensure_worker(self):
        with locked(self.worker / "startup.lock"):
            if self._alive():
                return
            # A lock held by a worker whose lease is not ready must never spawn a second owner.
            try:
                with locked(self.worker / "owner.lock", blocking=False):
                    pass
            except BlockingIOError:
                while not self._alive():
                    try:
                        with locked(self.worker / "owner.lock", blocking=False):
                            raise SimulatorError("worker exited before publishing its identity")
                    except BlockingIOError:
                        time.sleep(0.1)
                return
            if (self.worker / "closed.json").exists():
                raise SimulatorError(
                    "native worker was closed; this completed run cannot be resumed"
                )
            with (self.output / "sim.log").open("a") as log:
                process = subprocess.Popen(
                    self.command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    env=dict(os.environ),
                )
            while not self._alive():
                if process.poll() is not None:
                    raise SimulatorError(f"simulator startup failed; see {self.output / 'sim.log'}")
                time.sleep(0.1)

    def request(self, op: str, **kwargs) -> dict:
        """Wait while the exact worker lives; replay only after confirmed exit, never on delay."""
        if op == "reset":
            self.seed = kwargs["seed"]
        recovery = 0
        while recovery < 3:
            try:
                with socket.socket(socket.AF_UNIX) as connection:
                    connection.connect(str(self.socket_path))
                    connection.settimeout(2)
                    connection.sendall((json.dumps({"op": op, **kwargs}) + "\n").encode())
                    chunks = bytearray()
                    while b"\n" not in chunks:
                        try:
                            data = connection.recv(65536)
                        except TimeoutError:
                            if self._alive():
                                continue
                            raise ConnectionError("native worker exited") from None
                        if not data:
                            raise ConnectionError("worker disconnected")
                        chunks.extend(data)
                reply = json.loads(chunks)
                if "error" in reply:
                    raise SimulatorError(f"{reply['error']}: {reply.get('detail')}")
                return reply
            except (ConnectionError, FileNotFoundError, ConnectionRefusedError):
                if self._alive():
                    time.sleep(1)
                    continue  # a lost chunk reply is retrievable using the same request_id
                if recovery == 2:
                    raise SimulatorError(
                        "native recovery failed after confirmed worker exits"
                    ) from None
                recovery += 1
                self._ensure_worker()
                if op != "reset":
                    if self.seed is None:
                        raise SimulatorError("cannot restore without the original seed") from None
                    self.request("reset", seed=self.seed)
        raise SimulatorError("simulator transport unavailable; checkpoints preserved")

    def close(self) -> dict:
        """Explicit close finalizes recorders; interruption should only detach."""
        return self.request("close")

    def detach(self) -> None:
        """Leave the private native worker untouched for a future --resume host."""
