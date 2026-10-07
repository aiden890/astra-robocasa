"""Keep action clip timestamps aligned to real observations without repeating model calls."""

import json
from types import SimpleNamespace

import numpy as np
from robocasa_common.depth_policy import DepthPolicy
from robocasa_common.paired_live_policy import PairedLivePolicy


def test_clip_contains_only_acknowledged_frames_and_native_start_step(tmp_path, monkeypatch):
    """A two-step action clip includes its initial state and two observed movements."""
    monkeypatch.setattr(DepthPolicy, "observe", lambda self, obs: None)
    encoded = []

    def writer(path, size, **kwargs):
        assert size == (768, 256)
        assert kwargs["fps"] == 20
        try:
            while True:
                encoded.append((yield).copy())
        finally:
            from pathlib import Path

            Path(path).write_bytes(b"verified-test-clip")

    monkeypatch.setattr("robocasa_common.paired_live_policy.imageio_ffmpeg.write_frames", writer)
    policy = PairedLivePolicy.__new__(PairedLivePolicy)
    policy.output = tmp_path / "policy"
    policy.clip_frames = []
    policy.clip_start = 0
    policy.progress = {"decision": {"start_step": 10, "repeat": 2}}
    for step in (10, 11, 12):
        observation = SimpleNamespace(
            images={name: np.full((256, 256, 3), step, dtype=np.uint8) for name in ("a", "b", "c")},
            extra={"steps": step},
        )
        policy.observe(observation)
    metadata = json.loads((tmp_path / "live-clip.json").read_text())
    assert metadata["start_step"] == 10 and metadata["end_step"] == 12
    assert len(encoded) == metadata["frames"] == 3
    assert [int(frame[0, 0, 0]) for frame in encoded] == [10, 11, 12]
    assert len(policy.clip_frames) == 1
