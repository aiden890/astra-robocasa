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


def test_video_ranges_enable_seek_and_head_retains_full_size(tmp_path, monkeypatch):
    """Seek and suffix ranges return exact bytes, including boundary and invalid requests."""
    import urllib.error

    import pytest

    (tmp_path / "clip.mp4").write_bytes(b"0123456789")
    monkeypatch.setattr(BoardHandler, "state_file", tmp_path / "board.json")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(BoardHandler, directory=str(tmp_path)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/clip.mp4"
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Length"] == "10"
            assert response.read() == b""
        for value, expected, content_range in (
            ("bytes=2-5", b"2345", "bytes 2-5/10"),
            ("bytes=7-", b"789", "bytes 7-9/10"),
            ("bytes=-3", b"789", "bytes 7-9/10"),
            ("bytes=8-99", b"89", "bytes 8-9/10"),
        ):
            request = urllib.request.Request(url, headers={"Range": value})
            with urllib.request.urlopen(request) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == content_range
                assert response.read() == expected
        for value in ("bytes=20-", "bytes=5-2", "bytes=-0", "bytes=0-1,5-6"):
            request = urllib.request.Request(url, headers={"Range": value})
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(request)
            assert caught.value.code == 416
        assert (tmp_path / "clip.mp4").read_bytes() == b"0123456789"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
