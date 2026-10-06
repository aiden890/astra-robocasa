# Common RoboCasa evaluation

The common environment uses PandaOmron, 20 Hz native control and frozen scene snapshots.
Each task has ten recorded seed IDs. A snapshot, rather than a seed alone, defines an episode.
The participant supplies only a policy factory. Do not change assets, initial state, instruction,
controller, action bounds, horizon or native success check when comparing policies.

## Participant policy

Install the common framework and adapter in your client environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e . -e plugins/inspect-robots-robocasa-astra
```

Create a module in your own project with `create_policy(embodiment, output)`.
Return an Inspect Robots Policy compatible with the embodiment's action and observation spaces.
Model credentials belong in your own runtime configuration, outside this repository.
The common evaluator neither supplies nor requires a Codex account.

## Run one scene

Mount the shared scene archive read-only at `/scene-bundles`, this plugin at `/frozen-code`,
and the compatible Inspect Robots source at `/astra/src` inside the simulator container.
Use the exact RoboCasa, robosuite, MuJoCo and renderer versions recorded in the bundle.

```bash
PYTHONPATH=evaluation:src:plugins/inspect-robots-robocasa-astra/src \
python -m robocasa_common.evaluate \
  --scene-root /path/to/shared-scenes \
  --scene-id PrepareCoffee-8806552 \
  --policy my_project.policy:create_policy \
  --host spark2 --container my-evaluation-container \
  --output /path/to/new-model-results
```

Use `--all` instead of `--scene-id` to evaluate the fixed fifty-scene protocol.

Each episode is evaluated separately so seed derivation uses index zero.
Scene checksums, the exact initial simulator state and world/camera geometry are checked before
model calls. Existing result directories are never overwritten. Interrupted runs are unscored;
normal completion can still be a task failure. Initial camera pixel agreement is recorded separately. Matching initial scenes does not imply identical
future trajectories for different policies or bitwise rendering across different GPU stacks.

## Merge into your project

```bash
git remote add common-eval https://github.com/aiden890/astra-robocasa.git
git fetch common-eval codex/evaluation-common-scenes
git merge --no-ff common-eval/codex/evaluation-common-scenes
```

For a different repository without shared Git history, copy `evaluation/` and install the
`inspect-robots-robocasa-astra` plugin, or cherry-pick the published feature commit. The fork
retains the official Inspect Robots license. Generated scene archives remain separate from code.

## Environment lock

`environment.lock.json` records the tested Spark2 image ID, package versions and simulator source hashes.
The scene archive includes model data, asset blobs, seed metadata and per-scene verification receipts.
The scene identity is its manifest SHA256. Keep the archive separate from Git and mount it read-only.

## Download the fixed scene set

[Scene archive (2.09 GB)](http://100.86.183.64:8906/media/common-eval-panda-scenes-20261006.tar.gz)
contains the fifty initial snapshots and their required asset/model blobs.
[Archive checksum](http://100.86.183.64:8906/media/common-eval-panda-scenes-20261006.json)
records its size and SHA256. These addresses require access to the lab network.

```bash
curl -fLO http://100.86.183.64:8906/media/common-eval-panda-scenes-20261006.tar.gz
printf '%s  %s\n' 17c228bd779d448c2090dcff56a424899e08a2c6c8e5b5a636ed349dc147a220 common-eval-panda-scenes-20261006.tar.gz | sha256sum -c -
tar -xzf common-eval-panda-scenes-20261006.tar.gz
```

The archive is scene data, not a complete simulator installation. Install official RoboCasa assets
and the locked simulator dependencies before running the worker. The recorded Docker image is
currently local to Spark2; its tag is not a public image download. The worker checks simulator
package versions before restoration. Results from another software or renderer configuration
must be reported as a separate environment.

## Verified scope

All 50 scenes passed independent restoration of simulator state, compiled model arrays, object
geometry and camera poses. Initial images were byte-identical in 29 scenes. The other 21 retain
pixel disagreement as a diagnostic. The cause has not been established. This protocol fixes
the initial physical scene but does not promise byte-identical rendered observations.

Scoped evaluator/plugin/operations tests: 24 passed. Existing core tests: 2,089 passed,
6 skipped, with 100% core coverage. Whole-repository Ruff and mypy still report pre-existing
issues in unrelated core/plugin files. They are not changed by this branch.
