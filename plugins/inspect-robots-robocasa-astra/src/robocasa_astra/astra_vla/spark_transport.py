"""Spark native transport; all episode files live on a Lab-backed shared mount."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from ..astra_robodawn.sim_client import SimulatorError
from .persistence import exact_process, locked, read_json


class SparkChunkSimClient:
    """Keep remote native workers alive and recover only after confirmed process exit."""

    def __init__(self, task, budget, output, scene_dir, condition, transport):
        self.output = Path(output)
        self.output.joinpath("worker").mkdir(parents=True, exist_ok=True)
        self.transport = transport
        self.seed = None
        name = hashlib.sha256(str(self.output.resolve()).encode()).hexdigest()[:24]
        self.socket_path = "/tmp/astra-vla-" + name + ".sock"
        self.command = [
            "nice",
            "-n",
            "10",
            "python",
            "-m",
            "robocasa_astra.astra_vla.sim_server",
            "--task",
            task,
            "--budget",
            str(budget),
            "--output",
            str(output),
            "--scene-dir",
            scene_dir,
            "--condition",
            condition,
            "--socket",
            self.socket_path,
        ]
        self._ensure_worker()

    def _remote(self, mode, request=None):
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            self.transport["host"],
            "docker",
            "exec",
            "-i",
            self.transport["container"],
            "python",
            "-m",
            "robocasa_astra.astra_vla.spark_transport",
            "--mode",
            mode,
        ]
        payload = {
            "output": str(self.output),
            "socket": self.socket_path,
            "command": self.command,
            "request": request,
        }
        result = subprocess.run(command, input=json.dumps(payload), text=True, capture_output=True)
        if result.returncode:
            raise ConnectionError("Spark transport unavailable: " + result.stderr[-1000:])
        reply = json.loads(result.stdout)
        if reply.get("transport_error"):
            raise ConnectionError(reply["transport_error"])
        return reply

    def _ensure_worker(self):
        with locked(self.output / "worker" / "startup.lock"):
            self._remote("ensure")

    def request(self, op, **kwargs):
        """Retry disconnected RPCs with the same chunk ID, without terminating physics."""
        if op == "reset":
            self.seed = kwargs["seed"]
        recoveries = 0
        while True:
            try:
                reply = self._remote("request", {"op": op, **kwargs})
                if "error" in reply:
                    raise SimulatorError(f"{reply['error']}: {reply.get('detail')}")
                return reply
            except (ConnectionError, json.JSONDecodeError):
                try:
                    alive = self._remote("status")["alive"]
                except (ConnectionError, json.JSONDecodeError):
                    time.sleep(15)
                    continue  # unknown remote state cannot authorize another worker
                if alive:
                    time.sleep(1)
                    continue
                if recoveries >= 3:
                    raise SimulatorError("Spark native recovery failed; checkpoint preserved")
                recoveries += 1
                self._ensure_worker()
                if op != "reset":
                    if self.seed is None:
                        raise SimulatorError("Original seed unavailable")
                    self.request("reset", seed=self.seed)

    def close(self):
        """Finalize only a completed native evaluation."""
        return self.request("close")

    def detach(self):
        """Leave the exact remote worker available for continuation."""


def main():
    """Run inside the private Spark container; return only transport JSON on stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["status", "ensure", "request"], required=True)
    args = parser.parse_args()
    data = json.load(sys.stdin)
    output = Path(data["output"])
    lease = output / "worker" / "lease.json"
    if args.mode == "status":
        print(json.dumps({"alive": exact_process(read_json(lease, {}))}))
    elif args.mode == "ensure":
        if not exact_process(read_json(lease, {})):
            if (output / "worker" / "closed.json").exists():
                raise RuntimeError("Completed worker cannot reopen")
            with (output / "sim.log").open("a") as log:
                process = subprocess.Popen(
                    data["command"],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            while not exact_process(read_json(lease, {})):
                if process.poll() is not None:
                    raise RuntimeError("Native startup failed; see sim.log")
                time.sleep(0.1)
        print(json.dumps({"alive": True, "lease": read_json(lease)}))
    else:
        try:
            with socket.socket(socket.AF_UNIX) as connection:
                connection.connect(data["socket"])
                connection.sendall((json.dumps(data["request"]) + "\n").encode())
                with connection.makefile("rb") as stream:
                    reply = stream.readline()
            if not reply:
                raise ConnectionError("Native socket disconnected")
            sys.stdout.write(reply.decode())
        except (OSError, ConnectionError) as error:
            print(json.dumps({"transport_error": str(error)}))


if __name__ == "__main__":
    main()
