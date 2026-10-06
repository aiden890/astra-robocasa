# Astra RoboCasa experiments

This repository extends the official Inspect Robots framework with a RoboCasa
adapter and a Codex subscription policy. See LICENSE for the upstream license.

- Task runner: `scripts/robocasa-astra/run.sh`
- Robot and simulator adapter: `plugins/inspect-robots-robocasa-astra/`
- Task queue: `scripts/robocasa-astra/run_multitask_queue.py`
- Task manifest: `scripts/robocasa-astra/multitask_tasks.json`
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
