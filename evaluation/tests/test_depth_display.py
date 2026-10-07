"""Verify display-only depth movie frame rate and native timeline alignment."""

import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest
from PIL import Image
from robocasa_common import history_study


def test_depth_movie_preserves_steps_and_rejects_gaps(tmp_path, monkeypatch):
    """A partial captured interval is declared accurately and never padded as full history."""
    site = tmp_path / "site"
    root = tmp_path / "study"
    folder = root / "results/pixel-h3/scene"
    frames = folder / "depth-frames"
    frames.mkdir(parents=True)
    for step in (10, 11, 12):
        Image.fromarray(np.full((256, 768, 3), step, np.uint8)).save(frames / f"{step:07d}.jpg")
    worker = folder / "worker"
    worker.mkdir()
    (worker / "checkpoint.json").write_text(json.dumps({"steps": 12}))
    dest = site / "media/study/job"
    dest.mkdir(parents=True)
    (dest / "cumulative.mp4").touch()
    real_path = Path
    monkeypatch.setattr(
        history_study,
        "Path",
        lambda p: site if str(p).endswith("astra-robocasa/status-page") else real_path(p),
    )
    job = {
        "id": "job",
        "namespace": "pixel-h3",
        "scene": "scene",
        "task": "task",
        "condition": "pixel",
        "history_seconds": 3,
    }
    row = history_study.publish(root, job, {}, True)
    assert (row["depth_start_step"], row["depth_end_step"]) == (10, 12)
    reader = imageio_ffmpeg.read_frames(str(site / row["depth_video"]))
    assert next(reader)["fps"] == 20
    assert len(list(reader)) == 3
    assert json.loads((dest / "inputs.json").read_text()) == {}
    (frames / "0000011.jpg").unlink()
    with pytest.raises(ValueError, match="missing or duplicate"):
        history_study.publish(root, job, {}, True)
