"""Print (and optionally save) the subscription's weekly usage now, via codex app-server; no model call.

    python scripts/robocasa-astra/weekly_usage.py [--save runs/robodawn/<run>/weekly_after.json]
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from robocasa_astra.astra_robodawn.appserver import AppServerCaller

parser = argparse.ArgumentParser()
parser.add_argument("--save")
args = parser.parse_args()
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    (tmp / "sys.md").write_text("")
    (tmp / "schema.json").write_text("{}")
    caller = AppServerCaller(os.environ["ASTRA_CODEX"], os.environ["ASTRA_CODEX_HOME"], tmp / "sys.md",
                             tmp / "schema.json", tmp / "w")
    try:
        snapshot = caller.weekly_snapshot(tmp / "stderr.log")
    finally:
        caller.close()
print(json.dumps(snapshot))
if args.save:
    Path(args.save).write_text(json.dumps(snapshot, indent=1))
