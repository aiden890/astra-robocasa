"""Run a persistent four-slot native RoboCasa task queue without disturbing existing workers."""

import fcntl
import json
import subprocess
import time
from pathlib import Path

from publish_videos import publish

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / ".runtime/multitask-20261006-v1"
CONTAINERS = [
    "astra-robocasa-20261006",
    "astra-robocasa-parallel-03-20261006",
    "astra-robocasa-parallel-04-20261006",
    "astra-robocasa-parallel-02-20261006",
]


def alive(pid):
    """Recognize live and non-zombie processes without sending signals."""
    p = Path(f"/proc/{pid}/stat")
    return p.exists() and p.read_text().split(") ", 1)[1][0] != "Z"


def write_status(state):
    """Publish atomic progress without authentication or model prompts."""
    state["updated_at"] = time.time()
    for target in [RUNTIME / "status.json", ROOT / "status-page/media/task-queue.json"]:
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        tmp.replace(target)


def container_idle(name):
    """Only lease an own container with no existing simulator process."""
    r = subprocess.run(
        ["ssh", "spark2", "docker", "top", name, "-eo", "pid,args"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return (
        r.returncode == 0
        and "robocasa_astra.worker" not in r.stdout
        and "multitask-worker" not in r.stdout
        and "resume-worker" not in r.stdout
    )


def main():
    """Start new immutable runs in idle slots and retain every terminal result."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lock = (RUNTIME / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((RUNTIME / "plan.json").read_text())
    state_path = RUNTIME / "status.json"
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {
            "parallel_limit": 4,
            "model": "gpt-6-astra",
            "server": "Spark2",
            "jobs": [{**row, "status": "pending"} for row in plan["jobs"]],
        }
    )
    children = {}
    while True:
        for pid, process in list(children.items()):
            if process.poll() is not None:
                children.pop(pid)
        for job in state["jobs"]:
            if job["status"] != "running" or alive(job["pid"]):
                continue
            logs = list((ROOT / "runs" / job["id"] / "eval").glob("*.json"))
            job["ended_at"] = time.time()
            if logs:
                data = json.loads(logs[0].read_text())
                job["status"] = "complete" if data["status"] == "success" else "error"
                job["task_success"] = (
                    data["results"]["metrics"].get("success_at_end") == 1
                    if data["status"] == "success"
                    else None
                )
            else:
                job["status"] = "error"
                output = ROOT / "runs" / job["id"]
                output.mkdir(exist_ok=True)
                (output / "failed.json").write_text(
                    json.dumps(
                        {
                            "stage": "setup_or_execution",
                            "run_log": str(RUNTIME / (job["id"] + ".log")),
                            "time": time.time(),
                        }
                    )
                )
        active = {job["container"] for job in state["jobs"] if job["status"] == "running"}
        setup_path = RUNTIME / "asset-setup.json"
        setup = json.loads(setup_path.read_text()) if setup_path.exists() else {"ready": False}
        state["setup"] = {
            "ready": setup.get("ready", False),
            "reason": setup.get("reason", "Preparing official assets"),
        }
        containers = setup.get("containers", CONTAINERS) if setup.get("ready") else []
        for container in containers:
            if len(active) >= state["parallel_limit"]:
                break
            if container in active or not container_idle(container):
                continue
            job = next((j for j in state["jobs"] if j["status"] == "pending"), None)
            if job is None:
                break
            output = ROOT / "runs" / job["id"]
            if output.exists():
                job["status"] = "attention_existing_output"
                continue
            copy = subprocess.run(
                [
                    "ssh",
                    "spark2",
                    "docker",
                    "cp",
                    "/home/csi-agent-dgx_spark2/workspace/astra-robocasa-20261006/multitask-worker-v1.py",
                    container + ":/tmp/multitask-worker-v1.py",
                ],
                capture_output=True,
                timeout=30,
            )
            if copy.returncode:
                continue
            command = [
                "bash",
                "scripts/robocasa-astra/run.sh",
                "--task",
                job["task"],
                "--robot",
                job["robot"],
                "--steps",
                str(job["max_steps"]),
                "--seed",
                str(job["seed"]),
                "--container",
                container,
                "--worker-script",
                "/tmp/multitask-worker-v1.py",
                "--native-scene",
                "--output",
                str(output),
            ]
            if job["robot"] == "GR1FloatingBody":
                command.append("--face-workstation")
            with (RUNTIME / (job["id"] + ".log")).open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                )
            job.update(
                status="running", container=container, pid=process.pid, started_at=time.time()
            )
            children[process.pid] = process
            active.add(container)
            write_status(state)
        write_status(state)
        try:
            publish(ROOT)
        except Exception as error:
            (RUNTIME / "publication-error.json").write_text(
                json.dumps({"error": str(error), "time": time.time()})
            )
        if all(job["status"] not in ("pending", "running") for job in state["jobs"]):
            state["status"] = "complete"
            write_status(state)
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
