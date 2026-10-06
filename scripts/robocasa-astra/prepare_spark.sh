#!/usr/bin/env bash
# Publish only this plugin's simulator code into its private Spark2 container.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
archive_file="$(mktemp /tmp/astra-robocasa-code.XXXXXX.tar.gz)"
trap 'rm -f "$archive_file"' EXIT
cd "$repo_dir"
tar -czf "$archive_file" src plugins/inspect-robots-robocasa-astra
scp "$archive_file" spark2:/tmp/astra-robocasa-code.tar.gz
ssh spark2 'set -eu
runtime_root=/home/csi-agent-dgx_spark2/workspace/astra-robocasa-20261006
mkdir -p "$runtime_root/code" "$runtime_root/fixtures"
tar xzf /tmp/astra-robocasa-code.tar.gz -C "$runtime_root/code"
if ! test -f "$runtime_root/fixtures/episodes.json"; then
  docker cp coffee-state-dppo-probe-20261006:/input-ready/. "$runtime_root/fixtures/"
fi
if docker inspect astra-robocasa-20261006 >/dev/null 2>&1; then
  docker start astra-robocasa-20261006 >/dev/null
  docker cp /tmp/astra-robocasa-code.tar.gz astra-robocasa-20261006:/tmp/code.tar.gz
  docker exec astra-robocasa-20261006 tar xzf /tmp/code.tar.gz -C /astra
else
  docker run -d --name astra-robocasa-20261006 --cpus 2 --memory 12g --gpus all \
    -e MUJOCO_GL=egl \
    -v "$runtime_root/code:/astra:ro" \
    -v "$runtime_root/fixtures:/fixtures:ro" \
    -v /home/csi-agent-dgx_spark2/workspace/rlinf-mibot-first-attempt-440a72b/assets:/opt/robocasa/robocasa/models/assets:ro \
    rlinf-mibot:spark-a4ee4562 sleep infinity
fi'
