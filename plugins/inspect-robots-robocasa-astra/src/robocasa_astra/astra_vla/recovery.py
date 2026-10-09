"""Per-native-action write-ahead checkpoints and idempotent chunk continuation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .persistence import atomic_json, digest, native_digest, read_json


class ActionJournal:
    """Commit every native action; a lost acknowledgement cannot apply a chunk twice."""

    def __init__(self, output: Path, identity: dict):
        self.path = output / "worker" / "checkpoint.json"
        self.data = read_json(
            self.path,
            {
                "identity": digest(identity),
                "actions": [],
                "pending": None,
                "chunks": {},
                "current_chunk": None,
            },
        )
        if self.data["identity"] != digest(identity):
            raise ValueError("checkpoint identity mismatch; refusing another scene/config")
        self.replaying = False

    def save(self) -> None:
        """Flush a complete checkpoint before returning any native acknowledgement."""
        atomic_json(self.path, self.data)

    def initialize(self, env, executor) -> None:
        """Verify exact initial physics before replaying committed actions."""
        current = native_digest(env, executor)
        if self.data.get("initial_digest", current) != current:
            raise ValueError("initial checkpoint drift")
        self.data["initial_digest"] = current
        self.save()

    def intent(self, action) -> None:
        """Write the unacknowledged native action before stepping physics."""
        if not self.replaying:
            self.data["pending"] = np.asarray(action).tolist()
            self.save()

    def commit(self, env, executor, action) -> None:
        """Acknowledge exactly one action and its resulting state."""
        if self.replaying:
            return
        self.data["actions"].append(
            {"action": np.asarray(action).tolist(), "digest": native_digest(env, executor)}
        )
        current = self.data["current_chunk"]
        if current:
            self.data["chunks"][current]["cursor"] += 1
        self.data["pending"] = None
        self.save()

    def restore(self, env, executor) -> None:
        """Replay from the frozen scene, rejecting any acknowledged-state mismatch."""
        self.initialize(env, executor)
        self.replaying = True
        try:
            for entry in self.data["actions"]:
                action = np.asarray(entry["action"], dtype=float)
                executor.gripper_cmd = float(action[executor._split["right_gripper"][0]])
                executor._step(action, "checkpoint replay")
                if native_digest(env, executor) != entry["digest"]:
                    raise ValueError("checkpoint drift: refusing pending action")
        finally:
            self.replaying = False
        # A pending action had no durable ack. Apply once after exact replay, then commit.
        pending = self.data.get("pending")
        if pending is not None:
            action = np.asarray(pending, dtype=float)
            executor.gripper_cmd = float(action[executor._split["right_gripper"][0]])
            executor._step(action, "checkpoint pending action")
        self.data["restored_at"] = time.time()
        self.save()

    def begin_chunk(
        self, request_id: str, actions: list, turn: int, motion_mode: str | None = None
    ) -> dict:
        """Bind a request ID to immutable actions, retaining a completed reply forever."""
        identity = digest(
            {
                "actions": actions,
                "turn": turn,
                **({"motion_mode": motion_mode} if motion_mode is not None else {}),
            }
        )
        chunks = self.data["chunks"]
        if request_id in chunks and chunks[request_id]["identity"] != identity:
            raise ValueError("request ID reused with different actions")
        if request_id not in chunks:
            current = self.data["current_chunk"]
            if current and "result" not in chunks[current]:
                raise ValueError("another chunk is unfinished")
            chunks[request_id] = {
                "identity": identity,
                "cursor": 0,
                "start_step": len(self.data["actions"]),
            }
        self.data["current_chunk"] = request_id
        self.save()
        return chunks[request_id]

    def finish_chunk(self, request_id: str, result: dict) -> None:
        """Persist the reply before sending it, making a lost reply recoverable."""
        self.data["chunks"][request_id]["result"] = result
        self.data["current_chunk"] = None
        self.save()
