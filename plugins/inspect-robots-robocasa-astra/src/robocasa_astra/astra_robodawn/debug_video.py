"""Debug videos of a finished run: simulation on top, prompt / reply / outcomes / tokens / timing below.

Two versions are written into the run directory:

* ``debug_sim.mp4``: simulated time only (20 fps, one frame per native step), like ``video.mp4``.
* ``debug_realtime.mp4``: before each turn's motion the picture freezes for as long as the model took
  to answer, while the panel shows the prompt it was given; motion is then played at simulated speed.

Everything comes from the run directory (``video.mp4``, ``trace.jsonl``, ``replay/command_segments.json``,
``summary.json``, ``config.json``); nothing is re-simulated and no model is called.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FPS = 20
WIDTH = 1280
TOP_HEIGHT = 426  # three 256 px cameras scaled to the full width
PANEL_HEIGHT = 478  # 426 + 478 = 904, divisible by the encoder macro block (8)
CAPTION_HEIGHT = 22  # caption bar of video.mp4 (recorder.CAPTION_HEIGHT)
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")

BG = (24, 26, 30)
FG = (225, 228, 232)
DIM = (140, 146, 156)
OK = (110, 200, 120)
BAD = (235, 95, 85)
ACCENT = (250, 200, 80)
HEAD = (120, 180, 250)


def _font(name: str, size: int):
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


FONT = _font("DejaVuSans.ttf", 13)
BOLD = _font("DejaVuSans-Bold.ttf", 14)
MONO = _font("DejaVuSansMono.ttf", 12)
BIG = _font("DejaVuSans-Bold.ttf", 16)


@dataclass
class Turn:
    index: int
    first_step: int
    last_step: int
    prompt: str
    reply: dict
    results: list[dict]
    usage: dict
    model_s: float
    exec_wall_s: float | None
    error: str = ""
    failed: bool = False
    cum_usage: dict = field(default_factory=dict)

    @property
    def steps(self) -> int:
        return self.last_step - self.first_step


def load_turns(run: Path) -> list[Turn]:
    records = [json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()]
    turns, prev_end = [], 0
    cum = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for k, rec in enumerate(records):
        end = rec.get("state_after", rec.get("state", {})).get("steps_used", prev_end)
        timing = rec.get("timing", {})
        model_s = timing.get("model_s", rec.get("latency_s", 0.0)) or 0.0
        exec_wall = timing.get("exec_wall_s")
        if exec_wall is None and k + 1 < len(records) and "time" in records[k + 1]:
            exec_wall = max(records[k + 1]["time"] - rec.get("time", 0.0) - model_s, 0.0)
        usage = rec.get("usage", {}) or {}
        for key in cum:
            cum[key] += int(usage.get(key) or 0)
        results = rec.get("results", [])
        turns.append(Turn(
            index=rec["turn"], first_step=prev_end, last_step=end, prompt=rec.get("prompt", ""),
            reply=rec.get("response", {}) or {}, results=results, usage=usage, model_s=float(model_s),
            exec_wall_s=exec_wall, error=rec.get("error", ""),
            failed=bool(rec.get("error")) or any(not r.get("ok") for r in results), cum_usage=dict(cum),
        ))
        prev_end = end
    return turns


def _wrap(text: str, width: int, max_lines: int) -> list[str]:
    lines = []
    for paragraph in str(text).splitlines() or [""]:
        lines += textwrap.wrap(paragraph, width) or [""]
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + [lines[max_lines - 1][: width - 3] + "..."]
    return lines


def _prompt_digest(prompt: str) -> str:
    """The parts of a turn prompt that change per turn: last results and current state."""
    keep, take = [], False
    for line in prompt.splitlines():
        if line.startswith(("RESULT OF YOUR LAST COMMANDS", "CURRENT STATE")):
            take = True
        elif line.startswith(("YOUR NOTES", "IMAGES ATTACHED", "HISTORY", "Reply with")):
            take = False
        if take and line.strip():
            keep.append(line)
    return "\n".join(keep) or prompt[:600]


class Renderer:
    def __init__(self, run: Path):
        self.run = Path(run)
        self.turns = load_turns(self.run)
        self.summary = json.loads((self.run / "summary.json").read_text())
        self.config = json.loads((self.run / "config.json").read_text())
        self.segments = json.loads((self.run / "replay" / "command_segments.json").read_text())
        self.budget = self.summary.get("step_budget") or self.config.get("budget")
        scene = self.config.get("scene")
        self.title = (f"{self.summary['task']}" + (f" | scene {scene}" if scene is not None else "")
                      + f" | {self.summary.get('shots', self.config.get('shots'))}-shot | {self.config.get('model')} "
                      f"(effort {self.config.get('reasoning_effort')}) | result: "
                      f"{'SUCCESS' if self.summary['success'] else 'fail'} ({self.summary['finished_reason']})")

    # ------------------------------------------------------------------ panel
    def _base_panel(self, turn: Turn, phase: str) -> Image.Image:
        img = Image.new("RGB", (WIDTH, PANEL_HEIGHT), BG)
        d = ImageDraw.Draw(img)
        d.text((10, 6), self.title, font=BOLD, fill=HEAD)
        # prompt column
        x2, x3 = 350, 820
        d.text((x2, 32), "PROMPT TO MODEL (this turn)", font=BOLD, fill=ACCENT if phase == "think" else FG)
        y = 52
        for line in _wrap(_prompt_digest(turn.prompt), 66, 22):
            d.text((x2, y), line, font=MONO, fill=FG if phase == "think" else DIM)
            y += 16
        # reply column
        d.text((x3, 32), "MODEL REPLY", font=BOLD, fill=FG)
        y = 52
        if phase == "think":
            d.text((x3, y), "(model is reasoning ...)", font=FONT, fill=ACCENT)
        elif turn.error:
            for line in _wrap("call failed: " + turn.error, 62, 4):
                d.text((x3, y), line, font=FONT, fill=BAD)
                y += 17
        else:
            for key in ("scene", "progress", "plan"):
                for k, line in enumerate(_wrap(f"{key}: {turn.reply.get(key, '')}", 62, 4)):
                    d.text((x3, y), line, font=FONT, fill=FG if k == 0 else DIM)
                    y += 17
                y += 3
        return img

    def _dynamic(self, panel: Image.Image, turn: Turn, phase: str, step: int, clock: dict, current: int | None) -> Image.Image:
        img = panel.copy()
        d = ImageDraw.Draw(img)
        x3 = 820
        # commands with outcomes
        y = 282
        d.text((x3, y - 20), "COMMANDS -> OUTCOME", font=BOLD, fill=FG)
        if phase != "think":
            for k, r in enumerate(turn.results[:6]):
                color = ACCENT if k == current else (OK if r.get("ok") else BAD)
                mark = ">" if k == current else ("ok" if r.get("ok") else "X ")
                d.text((x3, y), f"{mark} {r['command']}  [{r.get('steps', 0)} st]", font=MONO, fill=color)
                y += 15
                note = r.get("note", "")
                if note and not r.get("ok"):
                    for line in _wrap(note, 58, 2):
                        d.text((x3 + 18, y), line, font=MONO, fill=DIM)
                        y += 14
        # time and tokens column
        x1, y = 10, 32
        d.text((x1, y), f"TURN {turn.index}/{len(self.turns)}   STEP {step}/{self.budget}", font=BIG, fill=FG)
        y += 28
        motion_sim = turn.steps / FPS
        lines = [
            ("THIS TURN", HEAD),
            (f"model reasoning     {turn.model_s:6.1f} s", FG),
            (f"motion (simulated)  {motion_sim:6.1f} s  ({turn.steps} steps)", FG),
            (f"motion (sim wall)   {turn.exec_wall_s:6.1f} s" if turn.exec_wall_s is not None else "motion (sim wall)      n/a", DIM),
            (f"reasoning + motion  {turn.model_s + motion_sim:6.1f} s", FG),
            ("", FG),
            ("CUMULATIVE (shown so far)", HEAD),
            (f"reasoning           {clock['reason']:6.1f} s", FG),
            (f"motion (simulated)  {clock['motion']:6.1f} s", FG),
            (f"total               {clock['reason'] + clock['motion']:6.1f} s", FG),
            ("", FG),
            ("TOKENS this turn / total", HEAD),
            (f"input   {int(turn.usage.get('input_tokens') or 0):7,} / {turn.cum_usage['input_tokens']:9,}", FG),
            (f"cached  {int(turn.usage.get('cached_input_tokens') or 0):7,} / {turn.cum_usage['cached_input_tokens']:9,}", FG),
            (f"output  {int(turn.usage.get('output_tokens') or 0):7,} / {turn.cum_usage['output_tokens']:9,}", FG),
        ]
        credits = (max(turn.cum_usage["input_tokens"] - turn.cum_usage["cached_input_tokens"], 0) * 250
                   + turn.cum_usage["cached_input_tokens"] * 25 + turn.cum_usage["output_tokens"] * 1250) / 1e6
        lines.append((f"credits so far  {credits:6.1f}  (~${credits * 0.04:.2f} API)", FG))
        if phase == "think":
            lines.insert(1, (f">> MODEL THINKING {clock['think_elapsed']:5.1f} / {turn.model_s:.1f} s", ACCENT))
        for text, color in lines:
            d.text((x1, y), text, font=MONO, fill=color)
            y += 16
        # timeline
        y0, x0, w = PANEL_HEIGHT - 26, 10, WIDTH - 20
        total = sum(max(t.steps, 1) for t in self.turns)
        x = x0
        for t in self.turns:
            tw = w * max(t.steps, 1) / total
            d.rectangle([x, y0, x + tw - 1, y0 + 14], fill=BAD if t.failed else (70, 90, 110),
                        outline=ACCENT if t.index == turn.index else BG)
            x += tw
        cursor = x0 + w * (sum(max(t.steps, 1) for t in self.turns if t.index < turn.index)
                           + max(step - turn.first_step, 0)) / total
        d.line([cursor, y0 - 4, cursor, y0 + 18], fill=ACCENT, width=2)
        d.text((x0, y0 - 18), "turns (width = steps, red = a command failed)", font=MONO, fill=DIM)
        return img

    # ------------------------------------------------------------------ frames
    def _model_inputs(self, turn: Turn) -> Image.Image | None:
        """The three images the model received this turn (annotated), side by side at the top height."""
        names = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
        paths = [self.run / "turns" / f"turn{turn.index:03d}_{n}.png" for n in names]
        if not all(p.exists() for p in paths):
            return None
        images = [Image.open(p).convert("RGB") for p in paths]
        scale = TOP_HEIGHT / images[0].height
        images = [im.resize((int(im.width * scale * (images[0].height / im.height)), TOP_HEIGHT)) for im in images]
        canvas = Image.new("RGB", (WIDTH, TOP_HEIGHT), BG)
        x = (WIDTH - sum(im.width for im in images)) // 2
        for im in images:
            canvas.paste(im, (x, 0))
            x += im.width
        ImageDraw.Draw(canvas).text((8, 6), f"images sent to the model (turn {turn.index})", font=BOLD, fill=ACCENT)
        return canvas

    def _camera(self, frame: np.ndarray) -> Image.Image:
        strip = Image.fromarray(frame[CAPTION_HEIGHT:])
        return strip.resize((WIDTH, TOP_HEIGHT))

    def _command_at(self, turn: Turn, step: int) -> int | None:
        k = 0
        for seg in self.segments:
            if seg["turn"] != turn.index:
                continue
            if seg["first_step"] < step <= seg["last_step"]:
                return k
            k += 1
        return None

    def render(self, realtime: bool) -> Path:
        import imageio_ffmpeg

        reader = imageio_ffmpeg.read_frames(str(self.run / "video.mp4"))
        meta = next(reader)
        width, height = meta["size"]

        def frames():
            for raw in reader:
                yield np.frombuffer(raw, np.uint8).reshape(height, width, 3)

        source = frames()
        out = self.run / ("debug_realtime.mp4" if realtime else "debug_sim.mp4")
        writer = imageio_ffmpeg.write_frames(str(out), (WIDTH, TOP_HEIGHT + PANEL_HEIGHT), fps=FPS, codec="libx264",
                                             quality=7, macro_block_size=8, output_params=["-pix_fmt", "yuv420p"])
        writer.send(None)
        clock = {"reason": 0.0, "motion": 0.0, "think_elapsed": 0.0}
        frame_buf = np.zeros((TOP_HEIGHT + PANEL_HEIGHT, WIDTH, 3), np.uint8)

        def send(top: Image.Image | None, panel_img: Image.Image | None) -> None:
            if top is not None:
                frame_buf[:TOP_HEIGHT] = np.asarray(top)
            if panel_img is not None:
                frame_buf[TOP_HEIGHT:] = np.asarray(panel_img)
            writer.send(frame_buf)

        last = None
        step = 0
        for turn in self.turns:
            if realtime and turn.model_s > 0:
                # The picture is frozen while the model reasons: redraw the panel once per second only.
                panel = self._base_panel(turn, "think")
                still = self._model_inputs(turn) or (self._camera(last) if last is not None else Image.new("RGB", (WIDTH, TOP_HEIGHT), BG))
                n = max(int(round(turn.model_s * FPS)), 1)
                frame_buf[:TOP_HEIGHT] = np.asarray(still)
                for i in range(n):
                    if i % FPS == 0 or i == n - 1:
                        clock["think_elapsed"] = (i + 1) / FPS
                        dyn = self._dynamic(panel, turn, "think", step,
                                            {**clock, "reason": clock["reason"] + clock["think_elapsed"]}, None)
                        frame_buf[TOP_HEIGHT:] = np.asarray(dyn)
                    writer.send(frame_buf)
            clock["reason"] += turn.model_s
            panel = self._base_panel(turn, "act")
            if turn.steps == 0:
                top = self._camera(last) if last is not None else Image.new("RGB", (WIDTH, TOP_HEIGHT), BG)
                dyn = self._dynamic(panel, turn, "act", step, clock, None)
                for _ in range(FPS // 2):  # hold half a second so zero-step turns stay visible
                    send(top, dyn)
                continue
            dyn, dyn_key = None, None
            for _ in range(turn.steps):
                frame = next(source, None)
                if frame is None:
                    break
                last = frame
                step += 1
                clock["motion"] += 1.0 / FPS
                current = self._command_at(turn, step)
                key = (current, (step - turn.first_step) // 5)  # refresh the panel every 0.25 s or on a new command
                if key != dyn_key:
                    dyn, dyn_key = self._dynamic(panel, turn, "act", step, clock, current), key
                    frame_buf[TOP_HEIGHT:] = np.asarray(dyn)
                frame_buf[:TOP_HEIGHT] = np.asarray(self._camera(frame))
                writer.send(frame_buf)
        writer.close()
        reader.close()
        return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--only", choices=["sim", "realtime"])
    args = parser.parse_args()
    versions = [v for v in ("sim", "realtime") if not args.only or args.only == v]
    if len(versions) == 1:
        print(Renderer(Path(args.run_dir)).render(versions[0] == "realtime"))
        return
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=2) as pool:  # the two versions render in parallel
        for path in pool.map(_render_one, [(args.run_dir, v == "realtime") for v in versions]):
            print(path)


def _render_one(job: tuple) -> Path:
    run_dir, realtime = job
    return Renderer(Path(run_dir)).render(realtime)


if __name__ == "__main__":
    main()
