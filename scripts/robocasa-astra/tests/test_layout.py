"""Keep moved commands anchored to the repository and preserve manifest coverage."""

import json

from astra_ops.common.paths import REPO_ROOT, TASK_MANIFEST


def test_repository_layout_and_manifest():
    """The package depth cannot redirect queue outputs outside the repository."""
    assert (REPO_ROOT / "pyproject.toml").is_file()
    data = json.loads(TASK_MANIFEST.read_text())
    rows = data if isinstance(data, list) else data["tasks"]
    assert len(rows) == 31
    assert (REPO_ROOT / "scripts/robocasa-astra/run.sh").is_file()
