"""Record native RoboCasa camera depth without model calls or a task score."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from robocasa_astra.bridge import SparkEmbodiment

from astra_ops.common.worker_transport import stage_worker
from astra_ops.queue.multitask import container_idle


def main():
    """Use a reserved idle container and save every sensor frame only on Lab."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--host", default="spark2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--motion", action="store_true")
    args = parser.parse_args()
    if not container_idle(args.container, args.host):
        raise RuntimeError("Container has an active simulator")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    seed = 786602 if args.motion else 786601
    frame_count = 160 if args.motion else 60
    if not stage_worker(args.container, seed, args.host):
        raise RuntimeError("Worker staging failed")
    command = [
        "ssh",
        args.host,
        "docker",
        "exec",
        "-i",
        "-e",
        f"PYTHONPATH=/tmp/astra-depth-{seed}",
        args.container,
        "python3",
        f"/tmp/astra-depth-{seed}/robocasa_astra/worker.py",
        "--robot",
        "PandaOmron",
        "--task",
        "PrepareCoffee",
        "--depth",
        "--horizon",
        str(frame_count + 60),
    ]
    env = None
    encoder = None
    try:
        env = SparkEmbodiment(command, seed, output / "worker")
        (output / "environment.json").write_text(env.info.docs)
        cameras = list(env.latest["images"])
        width = len(cameras) * 256
        video = output / "robocasa-camera-depth-20fps.mp4"
        encoder = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x584",
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
        raw = env.latest
        eef_positions = []
        eef_field = next(
            (key for key in ("robot0_right_eef_pos", "robot0_eef_pos") if key in raw["state"]),
            None,
        )
        for frame in range(frame_count):
            obs = env.observation(raw)
            canvas = Image.new("RGB", (width, 584), "#17202b")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (8, 5),
                "RoboCasa PrepareCoffee | PandaOmron | RGB above / camera depth below",
                fill="white",
            )
            draw.text(
                (8, 24),
                "0m white -> 3m black | scripted arm motion, no model, unscored"
                if args.motion
                else "0m white -> 3m black | zero actions, no model, unscored",
                fill="white",
            )
            summary = {}
            for i, name in enumerate(cameras):
                canvas.paste(Image.fromarray(obs.images[name]), (i * 256, 48))
                canvas.paste(Image.fromarray(obs.images[name + "__depth"]), (i * 256, 320))
                values = np.load(obs.extra["depth_maps"][name], allow_pickle=False)
                summary[name] = {"min_m": float(values.min()), "max_m": float(values.max())}
                draw.text((i * 256 + 4, 306), name.replace("robot0_", ""), fill="white")
            if eef_field is not None:
                eef_positions.append(np.asarray(raw["state"][eef_field], dtype=float))
            if frame == (40 if args.motion else 0):
                canvas.save(output / "robocasa-camera-depth-20fps.jpg")
            encoder.stdin.write(np.asarray(canvas).tobytes())
            samples.append(
                {
                    "frame": frame,
                    "simulation_step": raw["depth_metadata"][cameras[0]]["simulation_step"],
                    "cameras": summary,
                }
            )
            if frame < frame_count - 1:
                action = np.zeros(env.info.action_space.shape)
                if args.motion:
                    parts = json.loads(env.info.docs)["action_parts"]
                    start, end = parts["right"]
                    if end - start != 6:
                        raise ValueError(
                            "Motion preview requires native six-dimensional arm control"
                        )
                    phase = frame // 40
                    axis = 2 if phase in (0, 3) else 1
                    action[start + axis] = 0.04 if phase in (0, 1) else -0.04
                raw = env.call(op="step", action=action.tolist())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoding failed")
        report = {
            "kind": "sensor_diagnostic",
            "environment": "RoboCasa",
            "task": "PrepareCoffee",
            "robot": "PandaOmron",
            "seed": seed,
            "fps": 20,
            "frames": frame_count,
            "steps": frame_count - 1,
            "duration_s": frame_count / 20,
            "cameras": cameras,
            "unit": "m",
            "preview_range_m": [0, 3],
            "model_calls": 0,
            "task_score": None,
            "action": "scripted lift, lateral shift, return"
            if args.motion
            else "zero native controller action",
            "motion": args.motion,
            "eef_position_field": eef_field,
            "max_eef_displacement_m": float(
                max(np.linalg.norm(p - eef_positions[0]) for p in eef_positions)
            )
            if eef_positions
            else None,
            "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "samples": samples,
        }
        if args.motion and (
            report["max_eef_displacement_m"] is None or report["max_eef_displacement_m"] < 0.02
        ):
            raise RuntimeError("Robot motion was not verified")
        (output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "samples"}), flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.stdin.close()
            encoder.wait(timeout=30)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
