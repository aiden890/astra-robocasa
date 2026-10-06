#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$repo_dir/src:$repo_dir/plugins/inspect-robots-robocasa-astra/src"
exec "$repo_dir/.runtime/venv/bin/python" -m robocasa_astra.run \
  --codex "$repo_dir/.runtime/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex" \
  --codex-home "$repo_dir/.runtime/auth" "$@"
