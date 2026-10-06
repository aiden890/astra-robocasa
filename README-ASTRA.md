# Astra RoboCasa experiments

This repository extends the official Inspect Robots framework with a RoboCasa
adapter and a Codex subscription policy. See LICENSE for the upstream license.

- Task runner: `scripts/robocasa-astra/run.sh`
- Robot and simulator adapter: `plugins/inspect-robots-robocasa-astra/`
- Task queue: `astra_ops.queue.multitask`
- Task manifest: `scripts/robocasa-astra/config/multitask_tasks.json`
- Setup and queue details: `docs/robocasa-astra-multitask.md`
- Video board: `status-page/`

The policy uses `gpt-6-astra` with explicit reasoning effort `low`, the lowest
supported level. It uses a locally authenticated Codex subscription. Log in
separately on the execution host; authentication files are not included.
Shell tools, plugins, project instructions and web search are disabled for
policy calls. Responses must satisfy the action JSON schema and native bounds.
Model capacity errors and call timeouts retry the same observation without
advancing the simulator. A complete native evaluation records task success;
an interrupted execution has no scored result.

The current deployment runs up to four simulators on Spark2. Videos are
recorded at 20 fps and stored on Lab-desktop. Queue jobs use the native task
instruction and the manifest step budget. GR1 starts facing the native
workstation with collision and heading checks.

Runtime paths and deployment hosts in scripts reflect the current experiment.
Install the compatible RoboCasa/robosuite assets and adapt host paths before
running on another machine. Runtime logs, recordings, model login credentials,
private configuration and datasets are intentionally excluded from Git.

Upstream: https://github.com/robocurve/inspect-robots

## Code layout

- `plugins/inspect-robots-robocasa-astra/src/robocasa_astra/`: policy, simulator bridge and runner.
- `scripts/robocasa-astra/astra_ops/queue/`: persistent scheduling.
- `scripts/robocasa-astra/astra_ops/assets/`: isolated asset preparation and activation.
- `scripts/robocasa-astra/astra_ops/media/`: recording publication and board server.
- `scripts/robocasa-astra/astra_ops/common/`: repository paths and atomic status writes.
- `scripts/robocasa-astra/config/`: task manifest.
- `scripts/robocasa-astra/tests/`: operational regression tests.
- `status-page/`: static user interface.
- `docs/robocasa-astra-storage.md`: recording storage ownership.

Operational modules run with `PYTHONPATH=scripts/robocasa-astra python -m
astra_ops.<group>.<module>`. Completed one-off recovery and scaling scripts were
removed; their source remains in Git history. Generated recordings and evaluation receipts
remain outside version control and are not removed during source cleanup.

Run operational tests with `PYTHONPATH=scripts/robocasa-astra pytest scripts/robocasa-astra/tests`.

## RGB and depth branch

The `codex/rgb-depth` branch enables synchronized metric depth by default for
new runner invocations. Use `--no-depth` for an RGB-only control. See
[depth implementation and validation](docs/robocasa-astra-depth.md).
The existing production RGB queue remains on its original branch; switching
source in this checkout does not upgrade already running episodes.
