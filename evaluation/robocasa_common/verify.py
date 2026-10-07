"""Verify all scene bytes on their owning host without duplicating large asset files."""

import json
import sys
from pathlib import Path

from robocasa_astra.frozen_scene import sha256

from robocasa_common.evaluate import load_scene


def main():
    """Return a manifest-bound proof only after normal full file verification passes."""
    folder = Path(sys.argv[1])
    manifest = load_scene(folder)
    print(
        json.dumps(
            {"manifest_sha256": sha256(folder / "manifest.json"), "files": manifest["files"]}
        )
    )


if __name__ == "__main__":
    main()
