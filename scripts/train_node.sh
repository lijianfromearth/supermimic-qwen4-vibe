#!/usr/bin/env bash
# Launch a detached training run on one GPU.
#
#   GPU=5 RUN=sm scripts/train_node.sh --iterations 320000 --compile --no-recompute
#
# Extra arguments go straight to supermimic.scripts.train.  Logs, checkpoints and
# the exit code land in runs/$RUN, which is gitignored.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-5}"
RUN="${RUN:-sm}"
dir="runs/$RUN"
mkdir -p "$dir"

if [ -f "$dir/train.pid" ] && kill -0 "$(cat "$dir/train.pid")" 2>/dev/null; then
  echo "refusing: $dir already has a live training pid $(cat "$dir/train.pid")" >&2
  exit 1
fi

(
  CUDA_VISIBLE_DEVICES="$GPU" .venv/bin/python -m supermimic.scripts.train \
    --device cuda:0 --out-dir "$dir" "$@"
  rc=$?
  echo "$rc" > "$dir/train.rc"
  touch "$dir/train.done"
) > "$dir/train.log" 2>&1 < /dev/null &

echo $! > "$dir/train.pid"
echo "started $dir on GPU $GPU, pid $(cat "$dir/train.pid")"