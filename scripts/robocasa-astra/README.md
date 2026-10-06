# Astra operations

The Python package `astra_ops` contains deployment operations. The simulator
and model policy remain in the RoboCasa Astra plugin.

| Module | Responsibility |
| --- | --- |
| `queue.multitask` | Adopt active runs and schedule the native task queue |
| `assets.prepare` | Download official assets to the isolated Spark2 directory |
| `assets.activate` | Validate assets and enable future simulator containers |
| `media.publish` | Encode and publish terminal recordings |
| `media.board` | Serve the static video board and its deletion API |
| `common.paths` | Locate repository, runtime and task configuration |
| `common.runtime_io` | Inspect PIDs and write atomic status snapshots |

Run from the repository root:

```bash
export PYTHONPATH="$PWD/scripts/robocasa-astra"
.runtime/venv/bin/python -m astra_ops.queue.multitask
.runtime/venv/bin/python -m astra_ops.media.publish
.runtime/venv/bin/python -m astra_ops.media.board --help
.runtime/venv/bin/python -m pytest scripts/robocasa-astra/tests
```

The queue requires the private runtime plan and verified asset receipt. Do not
run asset preparation or activation again for a completed deployment. Runtime
plans, receipts and videos are not package configuration or source files.

`run.sh` launches the plugin's native rollout runner. `prepare_spark.sh` prepares
its remote code and fixture snapshot. Task definitions live in `config/`; tests
live in `tests/`; storage rules live in `docs/robocasa-astra-storage.md`.
