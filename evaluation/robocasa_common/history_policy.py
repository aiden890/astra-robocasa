"""Compare exact native-time image histories without privileged depth leakage."""

import hashlib
import json
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw
from robocasa_astra.checkpoint import atomic_json
from robocasa_astra.depth import preview_depth, preview_scale

from robocasa_common.paired_live_policy import PairedLivePolicy


def history_steps(step, seconds):
    """Return current then existing one-second native offsets at exactly 20 Hz."""
    return [step - 20 * offset for offset in range(seconds + 1) if step >= 20 * offset]


class HistoryPolicy(PairedLivePolicy):
    """Persist a bounded observation window and retain each actual call's inputs."""

    def __init__(self, embodiment, output):
        self.history_seconds = int(os.environ["ASTRA_HISTORY_SECONDS"])
        if self.history_seconds not in range(6):
            raise ValueError("History must be 0..5 simulation seconds")
        self.snapshot_step = None
        super().__init__(embodiment, output)
        if self.condition == "grid":
            raise ValueError("Grid is excluded from the history comparison")
        self.snapshots = self.output / "history-observations"
        self.snapshots.mkdir(exist_ok=True)
        settings = self.output / "history-settings.json"
        value = {
            "seconds": self.history_seconds,
            "hz": 20,
            "offset_steps": 20,
            "missing_early_history": "omit",
            "order": "current, -1s, -2s, ...",
            "condition": self.condition,
        }
        if settings.exists() and json.loads(settings.read_text()) != value:
            raise ValueError("Cannot resume under a different history protocol")
        atomic_json(settings, value)

    def observe(self, observation):
        """Capture every acknowledged step, including those inside action chunks."""
        super().observe(observation)
        step = observation.extra["steps"]
        self.snapshot_step = step
        cameras = [c for c in observation.images if not c.endswith("__depth")]
        if self.condition in ("color", "pixel"):
            depths = observation.extra.get("_depth_arrays", {})
            if not depths and self.embodiment.current_depth_step == step:
                depths = self.embodiment.current_depth_arrays
            if depths:
                # Display-only files never enter the model's history/image manifest.
                display = self.output.parent / "depth-frames"
                display.mkdir(exist_ok=True)
                frames = []
                for camera in cameras:
                    image = Image.fromarray(preview_depth(depths[camera])).convert("RGB")
                    scale = preview_scale(depths[camera])
                    draw = ImageDraw.Draw(image)
                    draw.rectangle((0, 236, 255, 255), fill="black")
                    draw.text(
                        (2, 238),
                        f"Z {scale['near_white_m']:.2f}..{scale['far_black_m']:.2f}m",
                        fill="white",
                    )
                    frames.append(np.asarray(image))
                temporary = display / "frame.pending.jpg"
                Image.fromarray(np.concatenate(frames, axis=1)).save(temporary, quality=85)
                temporary.replace(display / f"{step:07d}.jpg")
        joined = Image.fromarray(np.concatenate([observation.images[c] for c in cameras], axis=1))
        joined.save(self.output.parent / "latest.jpg")
        if step == 0:
            joined.save(self.output.parent / "initial.png")
        clip = self.output.parent / "live-clip.json"
        if clip.exists():
            metadata = json.loads(clip.read_text())
            segments = self.output.parent / "live-segments"
            segments.mkdir(exist_ok=True)
            segment = segments / f"{metadata['start_step']:07d}.mp4"
            marker = segment.with_suffix(".json")
            previous = json.loads(marker.read_text()) if marker.exists() else {}
            if previous.get("version") != metadata["version"]:
                shutil.copy2(self.output.parent / "live.mp4", segment)
                atomic_json(marker, metadata)
        folder = self.snapshots / f"step-{step:07d}"
        folder.mkdir(exist_ok=True)
        files, scales = [], {}
        for camera in cameras:
            path = folder / (camera + ".jpg")
            Image.fromarray(observation.images[camera]).save(path, quality=85)
            files.append(path.name)
        if self.condition == "color":
            depths = observation.extra.get("_depth_arrays", {})
            if not depths:
                if self.embodiment.current_depth_step != step:
                    raise ValueError("Historical depth does not match RGB observation")
                depths = self.embodiment.current_depth_arrays
            for camera in cameras:
                scale = preview_scale(depths[camera])
                image = Image.fromarray(preview_depth(depths[camera]))
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 238, 255, 255), fill="black")
                draw.text(
                    (2, 240),
                    f"Z {scale['near_white_m']:.2f}..{scale['far_black_m']:.2f}m white=near",
                    fill="white",
                )
                path = folder / (camera + "__depth.png")
                image.save(path)
                files.append(path.name)
                scales[camera] = scale
        atomic_json(
            folder / "observation.json",
            {
                "step": step,
                "observation_id": f"{self.scene}:step-{step}",
                "cameras": cameras,
                "files": files,
                "depth_scales": scales,
                "sha256": {
                    name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files
                },
            },
        )
        for older in self.snapshots.glob("step-*"):
            if int(older.name[5:]) < step - self.history_seconds * 20:
                shutil.rmtree(older)

    def call(self, prompt, images, folder):
        """Send only real saved observations and previously answered pixel queries."""
        if (folder / "runner-request.json").exists():
            # Retained calls used their saved prompt; preserve its input evidence on resume.
            return super().call(prompt, images, folder)
        step = self.snapshot_step
        entries = []
        current = json.loads((self.snapshots / f"step-{step:07d}" / "observation.json").read_text())
        cursor = 0
        for past in history_steps(step, self.history_seconds):
            source = self.snapshots / f"step-{past:07d}"
            metadata = json.loads((source / "observation.json").read_text())
            entry = {
                **metadata,
                "relative_seconds": (past - step) / 20,
                "image_indices_1_based": list(
                    range(cursor + 1, cursor + len(metadata["files"]) + 1)
                ),
            }
            if past == step:
                if len(images) != len(metadata["files"]):
                    raise ValueError("Current camera input count mismatch")
            else:
                if metadata["cameras"] != current["cameras"]:
                    raise ValueError("Historical camera order changed")
                for name in metadata["files"]:
                    path = folder / f"history-{past:07d}-{name}"
                    shutil.copy2(source / name, path)
                    images.append(path)
            cursor += len(metadata["files"])
            entries.append(entry)
        query_history = self.query_history(step)
        manifest = {
            "history_seconds": self.history_seconds,
            "current_step": step,
            "image_count": len(images),
            "observations": entries,
            "query_history_protocol": "all-native-steps-v1",
            "actually_observed_query_answers": query_history,
        }
        atomic_json(folder / "history-inputs.json", manifest)
        labels = [
            {k: v for k, v in entry.items() if k not in ("sha256", "files")} for entry in entries
        ]
        prompt += (
            "\nImage timeline (simulation time, not model wall time): "
            + json.dumps(labels)
            + "\nEach entry lists exact image order. "
            "Compare motion across these observations. Missing early offsets are omitted. "
            "New distance queries must target the current observation_id only."
        )
        if query_history:
            prompt += (
                "\nactually_observed_query_answers (every native step in the history window): "
                + json.dumps(query_history)
                + "\nThese are historical measurements, not current distances. "
                "No additional images are attached for these query-only observations."
            )
        (folder / "prompt.txt").write_text(prompt)
        return super().call(prompt, images, folder)

    def query_history(self, step):
        """Retain every answered observation in the window, independently of image sampling."""
        if self.condition != "pixel" or not self.history_seconds:
            return []
        records = []
        for source in sorted(self.snapshots.glob("step-*")):
            past = int(source.name[5:])
            if not max(0, step - self.history_seconds * 20) <= past < step:
                continue
            path = source / "query-answers.json"
            if not path.exists():
                continue
            metadata = json.loads((source / "observation.json").read_text())
            answers, seen = [], set()
            for answer in json.loads(path.read_text()):
                if (
                    answer.get("observation_id", metadata["observation_id"])
                    != metadata["observation_id"]
                ):
                    raise ValueError("Historical query observation ID mismatch")
                key = json.dumps(answer, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    answers.append(answer)
            if answers:
                records.append(
                    {
                        "step": past,
                        "observation_id": metadata["observation_id"],
                        "relative_seconds": (past - step) / 20,
                        "answers": answers,
                    }
                )
        return records

    def act(self, observation):
        """Keep only actual pixel answers for this observation's future history."""
        try:
            return super().act(observation)
        finally:
            answers = self.progress.get("answers", [])
            if self.condition == "pixel" and answers:
                folder = self.snapshots / f"step-{observation.extra['steps']:07d}"
                atomic_json(folder / "query-answers.json", answers)


def create_policy(embodiment, output):
    """Enable history only for the separately authorized follow-up trials."""
    return HistoryPolicy(embodiment, output)
