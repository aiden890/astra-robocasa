"""Debug video of a VLA-variant run: the skill variant's renderer, with the model's 16 x 12 action chunk shown.

The camera row, timeline, timing / token / weekly-usage column and the two versions (``--only sim`` /
``realtime``) come from :class:`astra_robodawn.debug_video.Renderer`. The reply column shows scene / progress /
plan and the chunk as a 16 x 12 grid (rows = steps, columns = action indices; blue negative, red positive, the
step being executed highlighted) with the per-index mean underneath; the outcome column shows the chunk's
measured effect.

    PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src $ASTRA_PYTHON \
        -m robocasa_astra.astra_vla.debug_video runs/vla12/<run> --only sim
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..astra_robodawn.debug_video import (ACCENT, BAD, BG, BOLD, DIM, FG, FONT, MONO, OK, PANEL_HEIGHT, WIDTH, Renderer,
                                          Turn, _wrap)

COLUMNS = ("bx", "by", "byaw", "trs", "mode", "fwd", "left", "up", "rfwd", "rlft", "rup", "grip")
CELL_W, CELL_H = 36, 9
GRID_X, GRID_Y = 820, 142
OUTCOME_Y = 345


def _prompt_digest(prompt: str) -> str:
    """The parts of a VLA turn prompt that change per turn: last chunk's result and the current state."""
    keep, take = [], False
    for line in prompt.splitlines():
        if line.startswith(("RESULT OF YOUR LAST CHUNK", "CURRENT STATE")):
            take = True
        elif line.startswith(("YOUR NOTES", "YOUR CURRENT IMAGES", "HISTORY", "Reply with", "STATE VECTOR")):
            take = False
        if take and line.strip():
            keep.append(line)
    return "\n".join(keep) or prompt[:600]


def _cell_color(value: float) -> tuple[int, int, int]:
    v = float(np.clip(value, -1.0, 1.0))
    if v >= 0:
        return (int(BG[0] + (230 - BG[0]) * v), int(BG[1] + (70 - BG[1]) * v), int(BG[2] + (60 - BG[2]) * v))
    v = -v
    return (int(BG[0] + (60 - BG[0]) * v), int(BG[1] + (120 - BG[1]) * v), int(BG[2] + (235 - BG[2]) * v))


class ChunkRenderer(Renderer):
    def __init__(self, run: Path):
        super().__init__(run)
        self.title = self.title.replace(" | result:", " | VLA 12-D x 16 chunk | result:")

    def _base_panel(self, turn: Turn, phase: str) -> Image.Image:
        img = Image.new("RGB", (WIDTH, PANEL_HEIGHT), BG)
        d = ImageDraw.Draw(img)
        d.text((10, 6), self.title, font=BOLD, fill=(120, 180, 250))
        x2, x3 = 350, 820
        d.text((x2, 32), "PROMPT TO MODEL (this turn)", font=BOLD, fill=ACCENT if phase == "think" else FG)
        y = 52
        for line in _wrap(_prompt_digest(turn.prompt), 66, 22):
            d.text((x2, y), line, font=MONO, fill=FG if phase == "think" else DIM)
            y += 16
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
                for k, line in enumerate(_wrap(f"{key}: {turn.reply.get(key, '')}", 62, 2)):
                    d.text((x3, y), line, font=FONT, fill=FG if k == 0 else DIM)
                    y += 14
        return img

    def _draw_chunk(self, d: ImageDraw.ImageDraw, turn: Turn, row: int | None) -> None:
        actions = np.asarray(turn.reply.get("actions") or [], dtype=float)
        if actions.ndim != 2 or actions.shape[1] != len(COLUMNS):
            return
        d.text((GRID_X, GRID_Y - 14), "ACTIONS (rows = 16 steps; bottom: mean per index)", font=MONO, fill=FG)
        for c, name in enumerate(COLUMNS):
            d.text((GRID_X + c * CELL_W, GRID_Y), name[:5], font=MONO, fill=DIM)
        top = GRID_Y + 14
        for r, values in enumerate(actions):
            for c, value in enumerate(values):
                x, y = GRID_X + c * CELL_W, top + r * CELL_H
                d.rectangle([x, y, x + CELL_W - 2, y + CELL_H - 2], fill=_cell_color(value))
            if r == row:
                d.rectangle([GRID_X - 2, top + r * CELL_H - 1, GRID_X + len(COLUMNS) * CELL_W, top + (r + 1) * CELL_H - 1],
                            outline=ACCENT)
        means = actions.mean(axis=0)
        y = top + len(actions) * CELL_H + 2
        for c, value in enumerate(means):
            text = f"{value:+.1f}" if abs(value) >= 0.05 else "0"
            d.text((GRID_X + c * CELL_W, y), text, font=MONO, fill=FG if text != "0" else DIM)

    def _dynamic(self, panel: Image.Image, turn: Turn, phase: str, step: int, clock: dict, current: int | None) -> Image.Image:
        img = super()._dynamic(panel, turn, phase, step, clock, None)
        d = ImageDraw.Draw(img)
        x3, y = 820, OUTCOME_Y
        d.rectangle([x3, GRID_Y - 16, WIDTH - 1, 448], fill=BG)  # replaces the skill-style command list
        if phase == "think" or turn.error:
            return img
        row = step - turn.first_step
        self._draw_chunk(d, turn, row if 0 <= row < 16 else None)
        d.text((x3, y - 20), "CHUNK -> OUTCOME", font=BOLD, fill=FG)
        for r in turn.results[:1]:
            for k, line in enumerate(_wrap(r["command"], 62, 2) + _wrap(r.get("note", ""), 62, 4)):
                d.text((x3, y + 15 * k), line, font=MONO, fill=(OK if r.get("ok") else BAD) if k < 2 else DIM)
        return img


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--only", choices=["sim", "realtime"], default="sim")
    args = parser.parse_args()
    print(ChunkRenderer(Path(args.run_dir)).render(args.only == "realtime"))


if __name__ == "__main__":
    main()
