"""The VLA variant's simulator process: the skill variant's server with a chunk executor and an
``act_chunk`` op.

Reset (frozen common scenes, recorders, replay files), observation rendering and closing are
inherited from
:class:`astra_robodawn.sim_server.Server`; observations additionally carry the 16-D RoboCasa365
state.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from ..astra_robodawn.sim_server import Server
from .chunk_runner import ChunkExecutor
from .depth_input import answer_queries, attach_previews, render_depths
from .persistence import atomic_json, digest, locked, process_lease
from .recovery import ActionJournal


class ChunkServer(Server):
    """Own a frozen scene, depth observations and durable native-action acknowledgements."""

    def __init__(self, task, budget, output, scene_dir=None, condition="rgb"):
        super().__init__(task, budget, output, scene_dir)
        self.condition = condition
        self.journal = None
        self.depths = {}
        self.observation_id = None
        self.start_reply = None
        self.closed = False

    def _on_step(self, action, obs, caption):
        if self.journal:
            self.journal.commit(self.sim.env, self.executor, action)
        super()._on_step(action, obs, caption)

    def reset(self, seed: int) -> dict:
        """Adopt a live environment or restore acknowledged actions from a frozen scene."""
        if self.start_reply and not self.journal.data.get("faulted"):
            return self.start_reply
        if self.start_reply:
            # Restore after a native exception; repeating a row in uncertain live state is unsafe.
            super().close()
            self.start_reply = None
        checkpoint = self.output / "worker" / "checkpoint.json"
        if checkpoint.exists():
            if not self.scene_dir:
                raise ValueError("native recovery requires a frozen scene")
            archive = self.output / "segments" / str(time.time_ns())
            archive.mkdir(parents=True)
            for name in ("video.mp4", "replay"):
                if (self.output / name).exists():
                    shutil.move(str(self.output / name), str(archive / name))
        reply = super().reset(seed)
        self.executor = ChunkExecutor(self.sim.env, self.budget, on_step=self._on_step)
        scene_hash = (
            digest(
                {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in Path(self.scene_dir).iterdir()
                    if p.is_file()
                }
            )
            if self.scene_dir
            else None
        )
        self.journal = ActionJournal(
            self.output,
            {
                "task": self.task,
                "budget": self.budget,
                "seed": seed,
                "scene_hash": scene_hash,
                "condition": self.condition,
            },
        )
        self.journal.restore(self.sim.env, self.executor)
        self.journal.data["faulted"] = False
        self.journal.save()
        self.executor.before_step = self.journal.intent
        reply["state"] = self.executor.state()
        self.start_reply = reply
        return reply

    def observe(self) -> dict:
        """Return aligned RGB and only the depth input enabled for this condition."""
        reply = super().observe()
        reply["dataset_state"] = self.executor.dataset_state()
        if self.condition != "rgb":
            self.depths = render_depths(self.sim.env, reply["views"])
        self.observation_id = digest(
            {
                "steps": self.executor.steps_used,
                "views": reply["views"],
                "depth_sha256": {
                    camera: hashlib.sha256(depth.tobytes()).hexdigest()
                    for camera, depth in self.depths.items()
                },
            }
        )
        reply["observation_id"] = self.observation_id
        if self.condition == "color":
            attach_previews(reply, self.depths)
        return reply

    def query(self, requests, observation_id):
        """Answer against the exact latest RGB observation without stepping physics."""
        if observation_id != self.observation_id:
            raise ValueError("stale observation_id")
        return {
            "answers": answer_queries(self.depths, requests, self.observation_id, self.condition)
        }

    def act_chunk(
        self, actions: list, notes: list[str], turn: int, request_id: str | None = None
    ) -> dict:
        """Continue only remaining rows, returning a persisted result after lost
        acknowledgements."""
        if self.journal.data.get("faulted"):
            raise RuntimeError("native step failed; reset the frozen checkpoint before continuing")
        self.turn = turn
        request_id = request_id or f"turn{turn}"
        chunk = self.journal.begin_chunk(request_id, actions, turn)
        if "result" in chunk:
            return chunk["result"]
        first = chunk["start_step"]
        resumed_rows = chunk["cursor"]
        try:
            result = self.executor.run_chunk(
                np.asarray(actions[chunk["cursor"] :], dtype=float), notes
            )
        except BaseException:
            self.journal.data["faulted"] = True
            self.journal.save()
            raise
        self.replay.mark(result["command"], first, self.executor.steps_used, turn)
        result["steps"] = self.executor.steps_used - first
        result["resumed_rows"] = resumed_rows
        if resumed_rows:
            result["measurement_scope"] = "remaining rows after checkpoint"
            result["note"] = (
                result.get("note", "") + f"; resumed after {resumed_rows} acknowledged rows"
            )
        result["state_after"] = self.executor.state()
        self.journal.finish_chunk(request_id, result)
        return result

    def dispatch(self, request: dict) -> dict:
        """Serve one idempotent protocol operation."""
        op = request["op"]
        if op == "reset":
            return self.reset(int(request["seed"]))
        if op == "observe":
            return self.observe()
        if op == "query":
            return self.query(request["queries"], request["observation_id"])
        if op == "act_chunk":
            return self.act_chunk(
                request["actions"],
                request.get("notes") or [],
                int(request.get("turn", 0)),
                request.get("request_id"),
            )
        if op == "close":
            result = self.close()
            self.closed = True
            atomic_json(self.output / "worker" / "closed.json", result)
            return result
        raise ValueError(f"unknown op {op!r}")


def serve_socket(server, path: Path) -> None:
    """A detached, private worker survives host disconnection; a lock prevents duplicate owners."""
    with locked(server.output / "worker" / "owner.lock", blocking=False):
        path.unlink(missing_ok=True)
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(str(path))
            os.chmod(path, 0o600)
            listener.listen(4)
            atomic_json(server.output / "worker" / "lease.json", process_lease())
            while not server.closed:
                connection, _ = listener.accept()
                with connection, connection.makefile("rwb") as stream:
                    line = stream.readline()
                    try:
                        reply = server.dispatch(json.loads(line))
                    except Exception as error:
                        traceback.print_exc(file=sys.stderr)
                        reply = {"error": type(error).__name__, "detail": str(error)}
                    try:
                        stream.write((json.dumps(reply) + "\n").encode())
                        stream.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass  # native ack is durable even when the host disappeared
        path.unlink(missing_ok=True)


def main() -> None:
    """Run the private socket worker or the legacy JSON-line protocol."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--scene-dir", help="common frozen scene folder to restore (exact initial state)"
    )
    parser.add_argument("--condition", choices=("rgb", "color", "pixel", "grid"), default="rgb")
    parser.add_argument("--socket")
    args = parser.parse_args()
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        server = ChunkServer(
            args.task, args.budget, Path(args.output), args.scene_dir, args.condition
        )
        if args.socket:
            serve_socket(server, Path(args.socket))
            return
        for line in sys.stdin:
            try:
                request = json.loads(line)
                reply = server.dispatch(request)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                reply = {"error": type(error).__name__, "detail": str(error)}
            protocol.write(json.dumps(reply) + "\n")
            protocol.flush()
            if server.closed:
                break


if __name__ == "__main__":
    main()
