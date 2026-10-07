#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_dir/src:$repo_dir/plugins/inspect-robots-robocasa-astra/src"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Overridable for hosts other than lab-desktop (e.g. a conda env and a standalone Codex binary).
python_bin="${ASTRA_PYTHON:-$repo_dir/.runtime/venv/bin/python}"
codex_bin="${ASTRA_CODEX:-$repo_dir/.runtime/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex}"
exec "$python_bin" -m robocasa_astra.run \
  --codex "$codex_bin" \
  --codex-home "${ASTRA_CODEX_HOME:-$repo_dir/.runtime/auth}" "$@"
