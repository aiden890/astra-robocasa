"""Activate isolated official assets only after native GR1 boot and action checks."""

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / ".runtime/multitask-20261006-v1"
ASSETS = "/home/csi-agent-dgx_spark2/workspace/astra-assets-complete-20261006"
BASE = "/home/csi-agent-dgx_spark2/workspace/astra-robocasa-20261006"
NAMES = [f"astra-multitask-complete-assets-{i:02d}-20261006" for i in range(1, 5)]
VALIDATOR = "astra-multitask-assets-validation-20261006"


def remote(args):
    """Run a bounded Spark2 setup operation and preserve errors."""
    return subprocess.run(
        ["ssh", "spark2", *args], capture_output=True, text=True, timeout=90, check=True
    ).stdout


def create(name, cpu, memory):
    """Create a new isolated container without changing any active worker."""
    remote(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--cpus",
            str(cpu),
            "--memory",
            memory,
            "--gpus",
            "all",
            "--network",
            "none",
            "-e",
            "MUJOCO_GL=egl",
            "-e",
            "OMP_NUM_THREADS=1",
            "-e",
            "OPENBLAS_NUM_THREADS=1",
            "-v",
            ASSETS + "/assets:/opt/robocasa/robocasa/models/assets:ro",
            "-v",
            BASE + "/parallel/code:/astra:ro",
            "-v",
            BASE + "/parallel/fixtures:/fixtures:ro",
            "rlinf-mibot:spark-a4ee4562",
            "sleep",
            "infinity",
        ]
    )
    remote(
        ["docker", "cp", BASE + "/multitask-worker-v1.py", name + ":/tmp/multitask-worker-v1.py"]
    )


def main():
    """Wait for complete downloads and enable queued runs after native checks."""
    while True:
        result = subprocess.run(
            ["ssh", "spark2", "cat", ASSETS + "/ready.json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            receipt = json.loads(result.stdout)
            assert receipt["download_complete"]
            break
        err = subprocess.run(
            ["ssh", "spark2", "cat", ASSETS + "/error.json"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if err.returncode == 0:
            raise RuntimeError(err.stdout)
        time.sleep(30)
    create(VALIDATOR, 1, "8g")
    proofs = []
    try:
        for task, seed in [
            ("PanTransfer", 781102),
            ("TurnOnMicrowave", 781104),
            ("CoffeeSetupMug", 781106),
        ]:
            output = R / ("assets-validation-" + task)
            cmd = [
                "bash",
                "scripts/robocasa-astra/run.sh",
                "--task",
                task,
                "--robot",
                "GR1FloatingBody",
                "--steps",
                "2",
                "--seed",
                str(seed),
                "--container",
                VALIDATOR,
                "--worker-script",
                "/tmp/multitask-worker-v1.py",
                "--native-scene",
                "--face-workstation",
                "--noop",
                "--output",
                str(output),
            ]
            with (R / ("assets-validation-" + task + ".log")).open("w") as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=log, timeout=900, check=True)
            logs = list((output / "eval").glob("*.json"))
            assert len(logs) == 1
            data = json.loads(logs[0].read_text())
            assert data["status"] == "success"
            proofs.append(
                {
                    "task": task,
                    "output": str(output),
                    "execution_status": data["status"],
                    "model_calls": 0,
                    "steps": 2,
                }
            )
    finally:
        remote(["docker", "stop", VALIDATOR])
    for name in NAMES:
        create(name, 2, "12g")
    final = {
        "ready": True,
        "reason": "Official assets prepared; GR1 native boot/reset/render/step passed",
        "containers": NAMES,
        "packs": receipt["packs"],
        "validations": proofs,
        "time": time.time(),
    }
    tmp = R / "asset-setup.tmp"
    tmp.write_text(json.dumps(final, indent=2))
    tmp.replace(R / "asset-setup.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        (R / "asset-activation-error.json").write_text(
            json.dumps({"error": str(error), "time": time.time()})
        )
        raise
