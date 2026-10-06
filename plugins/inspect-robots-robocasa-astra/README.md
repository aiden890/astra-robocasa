# RoboCasa Astra subscription experiment

This plugin uses the unchanged Inspect Robots evaluation loop, the official
Codex CLI with an existing ChatGPT login, and Spark2's native RoboCasa simulation.
No learning, optimizer, GPU model weights or API key billing is involved.

## Machines and repository

The working repository is `/home/aiden/Desktop/lab/robot/astra-robocasa` on
lab-desktop. The parent project is https://github.com/robocurve/inspect-robots.
The experiment branch is `codex/robocasa-astra`. The upstream framework source is
unchanged; this plugin contains policy and embodiment adapters.

- lab-desktop runs Codex `gpt-6-astra`, structured action validation and evaluation.
- Spark2 runs the native simulator in `astra-robocasa-20261006`.
- A private SSH stdin/stdout session transports observations and actions. No
  public simulator port is exposed. Other containers are left running.
- The simulator uses the existing `rlinf-mibot:spark-a4ee4562` image with CPU2,
  memory12GiB, GPU rendering, NumPy2.2.5, MuJoCo3.3.1 and robosuite1.5.2.
  The RoboCasa source reports0.5.1. Image-installed package metadata may differ.

## Supported experiments

`PrepareCoffee` is the full native task: pick the mug from the cabinet, place it
under the dispenser, release it and press the start button. `--placement` is a
separate held-mug evaluation and forbids button pressing.
The held-mug fixture is verified for PandaOmron only. GR1 currently runs the
full native task; it cannot load the Panda robot's recorded held-mug state.

1. `PandaOmron`: native default single-arm mobile robot,12D controller.
2. `GR1FloatingBody`: native bimanual robot,34D controller including both6D
   arm commands, torso/head/base, two6D hands and the hybrid control-mode flag.
   This tests a two-arm embodiment; it does not yet establish successful
   coordinated bimanual manipulation.

Do not use `GR1FixedLowerBody` in this installed RoboCasa build. Its reset expects
mobile-base joints which the fixed variant lacks. The original failure logs are
retained. FloatingBody uses the native supported mobile-base reset and controller.

The Panda fixture uses a complete recorded public PrepareCoffee scene. Full task
runs reset to original frame0; placement runs reset to the recorded held-mug
boundary. Fixture provenance is in `/fixtures/episodes.json` inside the container.
Source: RoboCasa Team, CC BY4.0. GR1 uses generated native target layout1/style1;
its scene differs from the Panda fixture, so scores are not a paired comparison.

## Run on lab-desktop

```bash
cd /home/aiden/Desktop/lab/robot/astra-robocasa
# No-model environment checks:
bash scripts/robocasa-astra/run.sh --probe --placement --output runs/panda-probe-new
bash scripts/robocasa-astra/run.sh --probe --robot GR1FloatingBody --output runs/gr1-probe-new
# Real Astra subscription calls:
bash scripts/robocasa-astra/run.sh --placement --steps 160 --output runs/panda-placement-new
bash scripts/robocasa-astra/run.sh --steps 300 --output runs/panda-coffee-new
bash scripts/robocasa-astra/run.sh --robot GR1FloatingBody --steps 300 --output runs/gr1-coffee-new
```

Every output path must be new. Calls are synchronous, with at most8 physical
steps per decision and a180s call timeout. The simulator is paused during model
inference. `--steps` counts actual environment steps, not model calls. More calls
consume the signed-in account's subscription quota. No alternative model is
silently substituted if Astra is unavailable.

The policy sees real images and environment-provided robot/object observations.
This is an environment-state-assisted experiment, not an image-only benchmark.
Panda has left/right/hand cameras; GR1 has left/right and both hand cameras.
Native controller configurations, input bounds and part indices accompany every
initial prompt. Returned action dimensions, finite values, bounds and repetition
are checked before execution. Parallel-jaw conventions do not apply to GR1's
individual dexterous hand joints; those retain their actual native contract.

## Authentication and tool isolation

The experiment has its own private `.runtime/auth` directory containing a local
copy of the existing lab-desktop Codex login. No credential is copied to Spark2,
committed to Git, returned by the simulator or sent in prompts. The runtime uses
the official Linux Codex0.154.0 binary directly because the system Snap could not
access its configuration, and the system Node12 cannot launch the JS wrapper.

Calls run in an empty working directory with project documentation disabled,
shell tool disabled, web search disabled and plugins disabled. No MCP servers,
user skills or hooks are configured in this isolated Codex home. API key variables
are removed from the invocation environment so this path uses the saved login.
Authentication refresh uses the official CLI and may require human login later.

## Evidence and interpretation

`runs/<name>/worker/simulator.log` preserves native startup and failure details.
`environment.json` records controller contract. `inference/call-*/` contains
observed PNGs, prompt, model response, receipt and CLI usage events.
`eval/` contains the official framework's trial metrics, action sequence and
observation frame sidecars.

An EvalLog `status=success` means the evaluation finished without an error.
Task success is only `results.metrics.success_at_end`. Native full-task
success and placement-only success are recorded separately. Short smoke tests
prove wiring, not task competence or a success rate.

## Tests

```bash
PYTHONPATH=src:plugins/inspect-robots-robocasa-astra/src \
.runtime/venv/bin/python -m pytest plugins/inspect-robots-robocasa-astra/tests -q -o addopts=
```

The core framework is not changed. Plugin tests reject malformed dimensions,
NaN, out-of-range actions and invalid repeat counts; the no-model test checks
camera-sidecar recording. Native resets/renders/steps and actual subscription
calls are checked separately on the two machines.

Spark code and fixture snapshots can be refreshed with `bash scripts/robocasa-astra/prepare_spark.sh`. This starts only the experiment container and does not restart it if already running.

## Full-horizon runs and video board

Default trial horizon is now 1800 steps. Both native robot environments use
20Hz control and ignore their internal time limit, allowing Inspect Robots to
stop at native success or the explicit step budget. This extends task time;
it does not establish task success or 20Hz model inference.

Saved per-step NPY sidecars can be encoded by
`astra_ops.media.publish`. Use the private runtime's
`imageio-ffmpeg` package. The full queue supervisor publishes live step counts
and final 20fps videos on the standalone video board.

### Interrupted trials

Capacity responses are retried up to 20 times with increasing waits capped at 120 seconds. The simulator receives no action while waiting, and each attempt keeps its own log. Other errors still fail explicitly. Running processes retain their imported code.

A historical one-off recovery (retained in Git history) recovered the specific interrupted Panda trial in a separate output directory. It replays the 1,791 recorded native actions with seed 771003, requires all saved numeric observations to match within 1e-9 and all three camera images to match exactly, and only then evaluates the remaining nine steps. Original failure logs remain intact. This is verified action replay, not a saved full simulator checkpoint.
