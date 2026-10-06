"""Serve public page assets and reversible board visibility changes on the private host."""

import argparse
import json
import re
import threading
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from astra_ops.common.runtime_io import write_json_atomic


class BoardHandler(SimpleHTTPRequestHandler):
    """Keep raw recordings intact; store only recoverable visibility tombstones."""

    state_file = None
    state_lock = threading.Lock()

    def read_state(self):
        """Return all board tombstones without exposing any runtime files."""
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {"deleted": []}

    def respond(self, status, data):
        """Send JSON with cache disabled for shared board state."""
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        """Expose the board state; serve only the configured public directory."""
        if self.path == "/api/board":
            with self.state_lock:
                self.respond(200, self.read_state())
            return
        super().do_GET()

    def do_POST(self):
        """Apply same-origin reversible delete or restore without touching recordings."""
        if self.path not in {"/api/board/delete", "/api/board/restore"}:
            self.respond(404, {"error": "Unknown operation"})
            return
        expected = "http://" + self.headers.get("Host", "")
        if self.headers.get("Origin") != expected:
            self.respond(403, {"error": "Same-origin request required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length))
            kind, identity = data.get("kind"), data.get("id", "")
            if kind not in {"topic", "video"} or not isinstance(identity, str):
                raise ValueError("Invalid board item")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identity):
                raise ValueError("Invalid item ID")
            title = data.get("title", identity)
            if not isinstance(title, str) or len(title) > 300:
                raise ValueError("Invalid title")
        except (ValueError, TypeError, AttributeError):
            self.respond(400, {"error": "Invalid request"})
            return
        with self.state_lock:
            state = self.read_state()
            state["deleted"] = [
                row for row in state["deleted"] if (row["kind"], row["id"]) != (kind, identity)
            ]
            if self.path.endswith("/delete"):
                state["deleted"].append(
                    {
                        "kind": kind,
                        "id": identity,
                        "title": title,
                        "deleted_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(self.state_file, state)
            self.respond(200, state)


def main():
    """Use the existing private bind address; never serve the repository root."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="100.86.183.64")
    parser.add_argument("--port", type=int, default=8906)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--state", required=True)
    args = parser.parse_args()
    BoardHandler.state_file = Path(args.state).resolve()
    handler = partial(BoardHandler, directory=str(Path(args.directory).resolve()))
    ThreadingHTTPServer((args.bind, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
