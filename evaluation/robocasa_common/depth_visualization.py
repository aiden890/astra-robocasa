"""Pair simulator frames with public model explanations and reported call usage."""

import json
from pathlib import Path

from robocasa_astra.checkpoint import atomic_json


def record_response(policy_root, call_folder, step, condition, response, query_answers=None):
    """Upsert one original call without duplicating reused responses or guessing usage."""
    root = Path(policy_root).parent
    path = root / "visualization.json"
    data = json.loads(path.read_text()) if path.exists() else {"control_hz": 20, "calls": []}
    receipt = Path(call_folder) / "receipt.json"
    usage = json.loads(receipt.read_text()).get("usage") if receipt.exists() else None
    row = {
        "call": Path(call_folder).name,
        "step": step,
        "condition": condition,
        "response": response,
        "usage": usage,
        "query_answers": query_answers or [],
    }
    data["calls"] = [x for x in data["calls"] if x["call"] != row["call"]] + [row]
    atomic_json(path, data)
    (root / "visualization.html").write_text(Path(__file__).with_suffix(".html").read_text())
