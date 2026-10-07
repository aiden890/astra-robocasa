"""A stand-in for the model that replays fixed replies, for dry runs without any model call.

It writes the same per-call files as the real callers (input, image list, response, call.json) so
the whole pipeline, including the run directory layout, can be checked before a real run.
"""

from __future__ import annotations

import json
from pathlib import Path

from .codex import CallResult, write_input

FALLBACK = {"scene": "(scripted)", "progress": "(scripted)", "memory": "(scripted)", "plan": "(scripted)",
            "commands": ["done"]}


class ScriptedCaller:
    def __init__(self, replies: list[dict]):
        self.replies = list(replies)
        self.index = 0

    def call(self, parts: list[dict], folder: Path) -> CallResult:
        folder.mkdir(parents=True, exist_ok=True)
        reply = self.replies[self.index] if self.index < len(self.replies) else FALLBACK
        self.index += 1
        text = json.dumps({**FALLBACK, **reply})
        write_input(folder, parts)
        (folder / "response.json").write_text(text)
        usage = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0}
        (folder / "call.json").write_text(json.dumps({"model": "scripted", "usage": usage, "final_text": text}, indent=1))
        return CallResult(text, usage, ["(scripted: no reasoning)"], 0.0)
