"""Run the two authorized full-horizon trials sequentially and publish their progress."""

import json
import subprocess
import time
from pathlib import Path

from publish_videos import publish


def main():
    """Preserve existing trials; keep other simulator jobs outside this queue."""
    repo = Path(__file__).resolve().parents[2]
    for robot, prefix in [("PandaOmron", "panda"), ("GR1FloatingBody", "gr1")]:
        run = repo / "runs" / f"{prefix}-coffee-1800-001"
        if run.exists():
            raise FileExistsError(f"Refusing to repeat existing trial: {run}")
        log = repo / ".runtime" / f"{prefix}-coffee-1800-001.log"
        with log.open("w") as output:
            process = subprocess.Popen(
                [
                    "bash",
                    "scripts/robocasa-astra/run.sh",
                    "--robot",
                    robot,
                    "--steps",
                    "1800",
                    "--output",
                    str(run),
                ],
                cwd=repo,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            while process.poll() is None:
                publish(repo)
                time.sleep(15)
            if process.returncode != 0:
                run.mkdir(exist_ok=True)
                (run / "failed.json").write_text(json.dumps({"returncode": process.returncode}))
            publish(repo)


if __name__ == "__main__":
    main()
