# Common RoboCasa evaluation

The common environment uses PandaOmron, 20 Hz native control and frozen scene snapshots.
Each task has ten recorded seed IDs with ten distinct kitchen layouts and ten distinct styles.
The realized asset composition hashes must also be distinct within each task. A snapshot, rather than a seed alone, defines an episode.
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

[Scene archive (2.09 GB)](http://100.86.183.64:8906/media/common-eval-panda-diverse-scenes-20261006.tar.gz)
contains the fifty initial snapshots and their required asset/model blobs.
[Archive checksum](http://100.86.183.64:8906/media/common-eval-panda-diverse-scenes-20261006.json)
records its size and SHA256. These addresses require access to the lab network.

```bash
curl -fLO http://100.86.183.64:8906/media/common-eval-panda-diverse-scenes-20261006.tar.gz
# Verify the SHA256 against the downloaded archive checksum JSON before extraction.
tar -xzf common-eval-panda-diverse-scenes-20261006.tar.gz
```

The archive is scene data, not a complete simulator installation. Install official RoboCasa assets
and the locked simulator dependencies before running the worker. The recorded Docker image is
currently local to Spark2; its tag is not a public image download. The worker checks simulator
package versions before restoration. Results from another software or renderer configuration
must be reported as a separate environment.

## Diversity and restoration checks

The original layout-1/style-1 scene set has been deleted at the owner's request.
Seed IDs remain fixed; the new archive is a different scene version identified by new manifest
checksums. Never mix its results with the retired set.

Each task uses layouts 1 through 10 and styles 1 through 10, one pair per seed in seed-list
order. Generation verifies the actual episode metadata rather than relying on the constructor
arguments. SHA256 fingerprints of the complete referenced mesh/texture asset sets must be
unique within each task. Shared robot assets may be reused across scenes. Different kitchen
asset composition does not mean every individual object model is unique.

The builder independently restores every snapshot and checks the exact simulator state,
world geometry and camera poses before accepting it. Initial camera pixel agreement remains
a separate diagnostic; physical scene equality does not imply bitwise rendering equality.

Generate a new collection with explicit kitchen diversity:

```bash
PYTHONPATH=evaluation:src:plugins/inspect-robots-robocasa-astra/src \
python scripts/robocasa-astra/astra_ops/assets/build_frozen_scenes.py \
  --seeds evaluation/seeds.json --root /path/to/new-scene-collection --workers 2 --diverse
```

Completed snapshots are immutable. Failed attempts remain separate from the shared archive.
