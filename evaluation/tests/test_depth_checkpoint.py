"""Exercise acknowledgement loss, client restart, replay drift and model-call adoption."""

import json
import shlex
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from robocasa_astra.bridge import SparkEmbodiment
from robocasa_astra.checkpoint import ActionCheckpoint, atomic_json
from robocasa_common.depth_policy import DepthPolicy
from robocasa_common.depth_study import retain_failed_attempt, valid_evaluation

from inspect_robots.types import Action

WORKER = """import json,sys,pathlib
marker=pathlib.Path(sys.argv[1]); steps=0; position=0
for line in sys.stdin:
 r=json.loads(line)
 if r['op']=='close': break
 if r['op']=='reset': steps=0; position=0
 if r['op']=='checkpoint_replay':
  for a in r['actions']: steps+=1; position+=a[0]
 if r['op']=='step':
  steps+=1; position+=r['action'][0]
  if steps==2 and not marker.exists(): marker.touch(); sys.exit(0)
 raw={'images':{},'depths':{},'state':{'robot0_position':[position]},'info':{
 'steps':steps,'success':False,'native_task_success':False,'physics_sha256':str(position),
 'action_dim':1,'action_low':[-1],'action_high':[1],'robot':'PandaOmron'}}
 print(json.dumps(raw),flush=True)
"""


def test_lost_step_ack_replays_then_applies_pending_once(tmp_path):
    """An applied but unacknowledged action is discarded with the dead world, never doubled."""
    script = tmp_path / "worker.py"
    script.write_text(WORKER)
    command = [sys.executable, str(script), str(tmp_path / "crashed")]
    env = SparkEmbodiment(command, 1, tmp_path / "checkpoint", checkpoint_mode=True)
    try:
        env.step(Action(np.array([1.0])))
        result = env.step(Action(np.array([1.0])))
        assert result.observation.extra["steps"] == 2
        assert result.observation.state["robot0_position"][0] == 2
        assert env.checkpoint.data["actions"] == [[1.0], [1.0]]
        assert json.loads((env.output / "recoveries.jsonl").read_text())["model_recalled"] is False
    finally:
        env.close()
    resumed = SparkEmbodiment(command, 1, tmp_path / "checkpoint", checkpoint_mode=True)
    try:
        assert resumed.resume_steps == 2
        assert resumed.latest["state"]["robot0_position"] == [2]
        assert resumed.step(Action(np.array([1.0]))).observation.extra["steps"] == 3
    finally:
        resumed.close()


def test_checkpoint_refuses_drift_and_other_scene(tmp_path):
    """A different scene or replay state cannot silently count as continuation."""
    checkpoint = ActionCheckpoint(tmp_path / "checkpoint.json", ["scene-A"], 1)
    raw = {"images": {}, "state": {}, "info": {"steps": 0, "success": False}}
    checkpoint.reset(raw)
    with pytest.raises(ValueError, match="another scene"):
        ActionCheckpoint(checkpoint.path, ["scene-B"], 1)
    with pytest.raises(ValueError, match="diverged"):
        checkpoint.verify({**raw, "state": {"different": 1}})


def test_partial_chunk_resumes_only_remaining_actions():
    """Resume a saved eight-step decision after three acknowledgements with no new call."""
    policy = DepthPolicy.__new__(DepthPolicy)
    policy.resumable = True
    policy.progress = {"decision": {"start_step": 16, "repeat": 8, "action": [0.2]}}
    chunk = policy.act(SimpleNamespace(images={}, extra={"steps": 19}))
    assert len(chunk.actions) == 5
    assert all(a.data.tolist() == [0.2] for a in chunk.actions)
    assert chunk.inference_latency_s == 0


def test_interrupted_model_wait_reuses_one_detached_call(tmp_path):
    """A killed client can wait for the original runner and its exact response."""
    policy_root = tmp_path / "policy"
    policy_root.mkdir()
    (policy_root / "schema.json").write_text("{}")
    call = policy_root / "call-00000"
    call.mkdir()
    executable = tmp_path / "fake-cli"
    script = tmp_path / "fake_cli.py"
    script.write_text(
        """import json,pathlib,sys,time
root=pathlib.Path(sys.argv[sys.argv.index('-o')+1]).parent
with (root/'launches').open('a') as log: log.write('1\\n')
print(json.dumps({'type':'turn.started'}),flush=True)
time.sleep(1.5)
(root/'response.json').write_text(json.dumps({'action':[0],'repeat':1}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'cached_input_tokens':0,'output_tokens':2}}),flush=True)
"""
    )
    executable.write_text(
        "#!/bin/sh\nexec "
        + shlex.quote(sys.executable)
        + " "
        + shlex.quote(str(script))
        + ' "$@"\n'
    )
    executable.chmod(0o755)
    request = {
        "prompt": "test",
        "images": [],
        "home": str(tmp_path / "auth"),
        "executable": str(executable),
        "condition": "rgb",
        "index": 1,
        "output": str(policy_root),
        "started_at": time.time(),
    }
    (tmp_path / "auth").mkdir()
    atomic_json(call / "runner-request.json", request)
    client_code = """import sys,json
from pathlib import Path
from robocasa_common.depth_policy import DepthPolicy
p=DepthPolicy.__new__(DepthPolicy); p.resumable=True; p.calls=[]
p.home=sys.argv[2]; p.executable=sys.argv[3]; p.condition='rgb'; p.index=1
p.output=Path(sys.argv[1]).parent
p.call('test',[],Path(sys.argv[1]))
"""
    command = [sys.executable, "-c", client_code, str(call), request["home"], str(executable)]
    first = subprocess.Popen(command)
    second = None
    try:
        deadline = time.time() + 5
        while not (call / "launches").exists() and time.time() < deadline:
            time.sleep(0.02)
        first.kill()
        first.wait(timeout=5)
        second = subprocess.Popen(command)
        second.wait(timeout=8)
        assert (call / "launches").read_text().splitlines() == ["1"]
        assert json.loads((call / "runner-result.json").read_text())["ok"] is True
        assert len((policy_root / "calls.jsonl").read_text().splitlines()) == 1
        assert json.loads((call / "receipt.json").read_text())["usage"]["total_tokens"] == 12
    finally:
        for process in (first, second):
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()


def test_errors_preserved_and_do_not_count_as_native_completion(tmp_path):
    """Failure evidence is copied for checkpoint continuation without double-counting calls."""
    job = {"condition": "rgb", "scene": "scene"}
    folder = tmp_path / "results/rgb/scene"
    (folder / "worker").mkdir(parents=True)
    (folder / "worker/checkpoint.json").write_text('{"steps": 3}')
    (folder / "result.json").write_text('{"task_success": null}')
    (folder / "original-events.jsonl").write_text("exact evidence")
    failed = {"execution_status": "error", "task_success": None}
    retain_failed_attempt(tmp_path, job, folder, failed)
    assert not valid_evaluation(failed)
    assert valid_evaluation({"execution_status": "success", "task_success": False})
    assert (folder / "worker/checkpoint.json").exists()
    assert not (folder / "result.json").exists()
    copies = list((tmp_path / "attempts").glob("*/results/rgb/scene/original-events.jsonl"))
    assert len(copies) == 1 and copies[0].read_text() == "exact evidence"
    assert json.loads((copies[0].parents[3] / "snapshot-only.json").read_text())[
        "do_not_double_count_calls"
    ]


def test_client_restart_preserves_call_chunk_and_local_video(tmp_path):
    """Kill a waiting client, interrupt its resumed chunk, then finish without a second call."""
    import base64
    import io

    import imageio_ffmpeg
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)).save(buffer, format="PNG")
    cameras = {
        name: base64.b64encode(buffer.getvalue()).decode() for name in ("left", "right", "hand")
    }
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER.replace("'images':{}", "'images':" + repr(cameras)))
    (tmp_path / "crashed").touch()
    cli = tmp_path / "cli.py"
    cli.write_text("""import json,pathlib,sys,time
folder=pathlib.Path(sys.argv[sys.argv.index('-o')+1]).parent
with (folder.parent/'launches').open('a') as log: log.write('1\\n')
print(json.dumps({'type':'turn.started'}),flush=True)
time.sleep(1)
(folder/'response.json').write_text(json.dumps({'action':[0.1],'repeat':8,'reason':'test'}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':10,'cached_input_tokens':0,'output_tokens':2}}),flush=True)
""")
    executable = tmp_path / "fake-cli"
    executable.write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + " " + shlex.quote(str(cli)) + ' "$@"\n'
    )
    executable.chmod(0o755)
    (tmp_path / "auth").mkdir()
    client = tmp_path / "client.py"
    client.write_text("""import os,sys
from pathlib import Path
from types import SimpleNamespace
from robocasa_common.evaluate import ObservedSparkEmbodiment
from robocasa_common.depth_policy import DepthPolicy
r=Path(sys.argv[1]); stage=int(sys.argv[2])
os.environ.update(ASTRA_DEPTH_CONDITION='rgb',ASTRA_CODEX_HOME=str(r/'auth'),ASTRA_CODEX_EXECUTABLE=str(r/'fake-cli'))
e=ObservedSparkEmbodiment([sys.executable,str(r/'worker.py'),str(r/'crashed')],1,r/'episode/worker',checkpoint_mode=True)
p=DepthPolicy(e,r/'episode/policy'); scene=SimpleNamespace(id='scene',instruction='test')
p.reset(scene); obs=e.reset(scene,seed=1); chunk=p.act(obs)
if stage==2:
 for action in chunk.actions[:3]: e.step(action)
 os._exit(99)
for action in chunk.actions: e.step(action)
p.close(); e.close()
""")
    command = [sys.executable, str(client), str(tmp_path)]
    first = subprocess.Popen([*command, "1"])
    try:
        deadline = time.time() + 8
        launches = tmp_path / "episode/policy/launches"
        while not launches.exists() and time.time() < deadline:
            time.sleep(0.02)
        assert launches.exists()
        first.kill()
        first.wait(timeout=5)
        assert subprocess.run([*command, "2"], timeout=12).returncode == 99
        subprocess.run([*command, "3"], timeout=12, check=True)
        assert launches.read_text().splitlines() == ["1"]
        checkpoint = json.loads((tmp_path / "episode/worker/checkpoint.json").read_text())
        assert checkpoint["steps"] == 8 and len(checkpoint["actions"]) == 8
        calls = (tmp_path / "episode/policy/calls.jsonl").read_text().splitlines()
        assert len(calls) == 1
        reader = imageio_ffmpeg.read_frames(str(tmp_path / "episode/video.mp4"))
        try:
            metadata = next(reader)
            frames = sum(1 for _ in reader)
        finally:
            reader.close()
        assert metadata["fps"] == 20 and frames >= 9
    finally:
        if first.poll() is None:
            first.kill()
            first.wait()
