"""The VLA variant's simulator process: the skill variant's server with a chunk executor and an ``act_chunk`` op.

Reset (frozen common scenes, recorders, replay files), observation rendering and closing are inherited from
:class:`astra_robodawn.sim_server.Server`; observations additionally carry the 16-D RoboCasa365 state.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

import numpy as np

from ..astra_robodawn.sim_server import Server
from .chunk_runner import ChunkExecutor


class ChunkServer(Server):
    def reset(self, seed: int) -> dict:
        reply = super().reset(seed)
        self.executor = ChunkExecutor(self.sim.env, self.budget, on_step=self._on_step)
        reply["state"] = self.executor.state()
        return reply

    def observe(self) -> dict:
        reply = super().observe()
        reply["dataset_state"] = self.executor.dataset_state()
        return reply

    def act_chunk(self, actions: list, notes: list[str], turn: int) -> dict:
        self.turn = turn
        first = self.executor.steps_used
        result = self.executor.run_chunk(np.asarray(actions, dtype=float), notes)
        self.replay.mark(result["command"], first, self.executor.steps_used, turn)
        result["state_after"] = self.executor.state()
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scene-dir", help="common frozen scene folder to restore (exact initial state)")
    args = parser.parse_args()
    protocol = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        server = ChunkServer(args.task, args.budget, Path(args.output), args.scene_dir)
        for line in sys.stdin:
            request = {}
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "reset":
                    reply = server.reset(int(request["seed"]))
                elif op == "observe":
                    reply = server.observe()
                elif op == "act_chunk":
                    reply = server.act_chunk(request["actions"], request.get("notes") or [], int(request.get("turn", 0)))
                elif op == "close":
                    reply = server.close()
                else:
                    raise ValueError(f"unknown op {op!r}")
            except Exception as error:  # noqa: BLE001 - reported to the host, which decides
                traceback.print_exc(file=sys.stderr)
                reply = {"error": type(error).__name__, "detail": str(error)}
            protocol.write(json.dumps(reply) + "\n")
            protocol.flush()
            if request.get("op") == "close":
                break


if __name__ == "__main__":
    main()
