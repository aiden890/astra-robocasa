"""Transfer the complete branch-local simulator worker package into an idle container."""

import io
import subprocess
import tarfile

from astra_ops.common.paths import REPO_ROOT


def stage_worker(container: str, seed: int, host: str = "spark2") -> bool:
    """Stage this branch's complete Python worker package only in a verified idle slot."""
    destination = f"/tmp/astra-depth-{seed}"
    create = subprocess.run(
        ["ssh", host, "docker", "exec", container, "mkdir", "-p", destination],
        capture_output=True,
        timeout=30,
    )
    if create.returncode:
        return False
    source = REPO_ROOT / "plugins/inspect-robots-robocasa-astra/src/robocasa_astra"
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        for path in sorted(source.glob("*.py")):
            bundle.add(path, arcname="robocasa_astra/" + path.name)
    copied = subprocess.run(
        ["ssh", host, "docker", "exec", "-i", container, "tar", "-xf", "-", "-C", destination],
        input=archive.getvalue(),
        capture_output=True,
        timeout=30,
    )
    return copied.returncode == 0
