"""Local durable acknowledged-action checkpoints, bound to an immutable scene."""

import copy
import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, value):
    """Commit one complete checkpoint before acknowledging a native step."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as file:
        json.dump(value, file, sort_keys=True)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def observation_digest(raw):
    """Require identical physical state, sensor payloads and native task outcome."""
    value = {key: raw.get(key) for key in ("images", "depths", "depth_metadata", "state")}
    value["native"] = {
        key: raw["info"].get(key)
        for key in (
            "steps",
            "native_task_success",
            "placement_success",
            "success",
            "physics_sha256",
        )
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ActionCheckpoint:
    """Retain acknowledged steps; an unacknowledged step never advances the journal."""

    def __init__(self, path, command, seed):
        self.path = Path(path)
        self.identity = hashlib.sha256(json.dumps([command, seed]).encode()).hexdigest()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else None
        if self.data is not None and self.data.get("identity") != self.identity:
            raise ValueError("Checkpoint belongs to another scene/worker configuration")

    def reset(self, raw):
        """Commit the exact initial observation after frozen-scene verification."""
        self.data = {
            "version": 1,
            "identity": self.identity,
            "actions": [],
            "digest": observation_digest(raw),
            "steps": raw["info"]["steps"],
        }
        atomic_json(self.path, self.data)

    def intent(self, action):
        """Save the next action before sending it, without marking it committed."""
        self.data["pending_action"] = action
        atomic_json(self.path, self.data)

    def commit(self, action, raw):
        """Persist both the action and its resulting observation atomically."""
        if raw["info"]["steps"] != self.data["steps"] + 1:
            raise ValueError("Checkpoint step did not advance by exactly one")
        committed = copy.deepcopy(self.data)
        committed["actions"].append(action)
        committed.update(digest=observation_digest(raw), steps=raw["info"]["steps"])
        committed.pop("pending_action", None)
        atomic_json(self.path, committed)
        self.data = committed

    def verify(self, raw):
        """Reject replay drift rather than continuing from a different state."""
        if observation_digest(raw) != self.data["digest"]:
            raise ValueError("Checkpoint replay diverged; continuation refused")
