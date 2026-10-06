"""Exercise board persistence against an isolated HTTP server and temporary state."""

import json
import threading
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

from astra_ops.media.board import BoardHandler


def test_delete_restore_preserves_recordings(tmp_path, monkeypatch):
    """Module relocation and atomic state writes preserve the board's reversible API."""
    public = tmp_path / "public"
    public.mkdir()
    recording = public / "video.mp4"
    recording.write_bytes(b"retained recording")
    monkeypatch.setattr(BoardHandler, "state_file", tmp_path / "private/board.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(BoardHandler, directory=str(public)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        for operation, expected in [("delete", 1), ("restore", 0)]:
            request = urllib.request.Request(
                base + "/api/board/" + operation,
                data=json.dumps({"kind": "video", "id": "test-video"}).encode(),
                headers={"Origin": base, "Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                assert response.status == 200
                assert len(json.load(response)["deleted"]) == expected
        with urllib.request.urlopen(base + "/api/board", timeout=5) as response:
            assert json.load(response) == {"deleted": []}
        assert recording.read_bytes() == b"retained recording"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
