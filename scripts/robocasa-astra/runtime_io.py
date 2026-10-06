"""Shared process inspection and atomic JSON snapshots for rollout supervisors."""

import json
import os
import tempfile
from pathlib import Path


def alive(pid: int) -> bool:
    """Return false for missing or zombie Linux processes, including exit races."""
    if pid <= 0:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1]
    except (FileNotFoundError, ProcessLookupError):
        return False
    return state[0] != "Z"


def write_json_atomic(target: Path, value: object) -> None:
    """Replace a snapshot only after complete serialization; clean up failed writes."""
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
