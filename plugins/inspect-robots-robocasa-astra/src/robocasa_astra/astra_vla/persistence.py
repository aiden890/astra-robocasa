"""Atomic progress, process identity, locks and exact simulator state fingerprints."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path

from ..checkpoint import atomic_json


def digest(value: dict) -> str:
    """Stable identity for immutable requests and checkpoint manifests."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def exact_process(lease: dict) -> bool:
    """Reject stale/reused PIDs unless the full process argv still matches."""
    try:
        argv = subprocess.check_output(
            ["ps", "-p", str(lease["pid"]), "-o", "command="], text=True
        ).strip()
        return bool(argv) and argv == lease["argv"]
    except (subprocess.CalledProcessError, KeyError):
        return False


def process_lease() -> dict:
    """Capture the current process's exact argv for future adoption."""
    return {
        "pid": os.getpid(),
        "argv": subprocess.check_output(
            ["ps", "-p", str(os.getpid()), "-o", "command="], text=True
        ).strip(),
    }


@contextlib.contextmanager
def locked(path: Path, *, blocking: bool = True):
    """One owner for a host, worker or request; kernel releases the lock after a crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def read_json(path: Path, default=None):
    """Read durable data without manufacturing an empty checkpoint."""
    return json.loads(path.read_text()) if path.exists() else default


def append_json(path: Path, value: dict) -> None:
    """Append original evidence and flush it to stable storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(value) + "\n")
        file.flush()
        os.fsync(file.fileno())


def native_digest(env, executor) -> str:
    """Fingerprint exact physics, native flags and controller continuation state."""
    import numpy as np

    physics = np.asarray(env.sim.get_state().flatten()).tobytes()
    # Controller history affects the next action even when physics agrees.
    values = {
        "steps": executor.steps_used,
        "success": executor.success,
        "gripper_cmd": float(executor.gripper_cmd),
    }
    h = hashlib.sha256(physics + json.dumps(values, sort_keys=True).encode())
    observations = env._get_observations(force_update=True)
    for name in sorted(observations):
        if name.endswith(("_image", "_depth")):
            h.update(name.encode() + np.asarray(observations[name]).tobytes())
    h.update(json.dumps(bool(env._check_success()), sort_keys=True).encode())
    for robot in env.robots:
        for controller in robot.composite_controller.part_controllers.values():
            for name in ("goal_pos", "goal_ori", "goal_qpos"):
                value = getattr(controller, name, None)
                if value is not None:
                    h.update(name.encode() + np.asarray(value).tobytes())
    return h.hexdigest()


__all__ = [
    "append_json",
    "atomic_json",
    "digest",
    "exact_process",
    "locked",
    "native_digest",
    "process_lease",
    "read_json",
]
