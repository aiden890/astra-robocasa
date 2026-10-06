"""Locate repository resources independently of command entry-point depth."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
RUNTIME_ROOT = REPO_ROOT / ".runtime/multitask-20261006-v1"
TASK_MANIFEST = REPO_ROOT / "scripts/robocasa-astra/config/multitask_tasks.json"
