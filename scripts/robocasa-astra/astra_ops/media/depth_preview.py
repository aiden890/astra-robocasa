"""Render an isolated moving geometry scene to inspect synchronized metric depth."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from robocasa_astra.depth import decode_depth, preview_depth, preview_scale
from robocasa_astra.worker import Simulator


def main():
    """Create a new CPU diagnostic; never treat the scene as a scored robot rollout."""
    from robosuite.utils.binding_utils import MjRenderContextOffscreen, MjSim

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    camera_xml = "".join(
        f'<camera name="{name}" pos="{x} 0 {z}" fovy="45"/>'
        for name, x, z in zip(cameras, [-0.25, 0.25, 0], [2, 2, 1.5], strict=True)
    )
    xml = (
        '<mujoco><visual><map znear="0.01" zfar="10"/></visual><worldbody>'
        '<light pos="0 0 3"/><geom type="plane" size="5 5 .1" rgba=".4 .4 .4 1"/>'
        '<body name="target" mocap="true" pos="0 0 .1">'
        '<geom type="box" size=".12 .12 .1" rgba="1 .15 .1 1"/></body>'
        + camera_xml
        + "</worldbody></mujoco>"
    )
    sim = MjSim.from_xml_string(xml)
    context = MjRenderContextOffscreen(sim, device_id=-1)
    worker = Simulator("PandaOmron", "DepthDiagnostic", depth=True)
    worker.parts = {}
    worker.env = SimpleNamespace(
        sim=sim,
        _get_observations=lambda **kwargs: {},
        _check_success=lambda: False,
        get_ep_meta=lambda: {"lang": "depth diagnostic"},
        action_dim=3,
        action_spec=(np.full(3, -1), np.ones(3)),
        control_freq=20,
        robots=[SimpleNamespace(composite_controller_config={})],
    )
    video = output / "camera-depth-preview-20fps.mp4"
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "768x584",
            "-r",
            "20",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        stdin=subprocess.PIPE,
    )
    samples = []
    try:
        for frame in range(160):
            phase = frame / 159 * 2 * np.pi
            target_z = 0.1 + 0.3 * (1 - np.cos(phase))
            sim.data.set_mocap_pos("target", np.array([0.15 * np.sin(phase), 0, target_z]))
            sim.data.time = frame / 20
            sim.forward()
            worker.steps = frame
            raw = worker.observe()
            canvas = Image.new("RGB", (768, 584), "#17202b")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (10, 5),
                "SENSOR DIAGNOSTIC | RGB above / optical-axis depth below | not a robot task",
                fill="white",
            )
            draw.text(
                (10, 24),
                "Depth: per-camera p2 white -> p98 black | range below in meters",
                fill="white",
            )
            row = {"frame": frame, "time_s": frame / 20, "cameras": {}}
            import base64
            import io

            for index, name in enumerate(cameras):
                rgb = Image.open(io.BytesIO(base64.b64decode(raw["images"][name]))).convert("RGB")
                metric = decode_depth(raw["depths"][name], (256, 256))
                canvas.paste(rgb, (index * 256, 48))
                canvas.paste(Image.fromarray(preview_depth(metric)), (index * 256, 320))
                scale = preview_scale(metric)
                draw.text(
                    (index * 256 + 4, 306),
                    f"{name.replace('robot0_', '')} "
                    f"{scale['near_white_m']:.2f}-{scale['far_black_m']:.2f}m",
                    fill="white",
                )
                row["cameras"][name] = {
                    "min_m": float(metric.min()),
                    "max_m": float(metric.max()),
                    "preview": scale,
                }
                if frame in (0, 40, 80, 120, 159):
                    folder = output / "samples" / f"frame-{frame:04d}"
                    folder.mkdir(parents=True, exist_ok=True)
                    np.save(folder / (name + ".npy"), metric, allow_pickle=False)
                    rgb.save(folder / (name + ".png"))
                    (folder / (name + ".json")).write_text(
                        json.dumps(raw["depth_metadata"][name], indent=2)
                    )
            if frame == 40:
                canvas.save(output / "camera-depth-preview-20fps.jpg")
            process.stdin.write(np.asarray(canvas).tobytes())
            samples.append(row)
    finally:
        process.stdin.close()
        sim.free()
        del context
    if process.wait() != 0:
        raise RuntimeError("Depth preview encoding failed")
    report = {
        "kind": "sensor_diagnostic",
        "scene": "isolated moving MuJoCo geometry; not RoboCasa",
        "model_calls": 0,
        "native_rollouts": 0,
        "fps": 20,
        "frames": 160,
        "duration_s": 8,
        "resolution": [768, 584],
        "unit": "m",
        "quantity": "camera optical-axis depth",
        "preview_range_m": [0, 3],
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "samples": samples,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}))


if __name__ == "__main__":
    main()
