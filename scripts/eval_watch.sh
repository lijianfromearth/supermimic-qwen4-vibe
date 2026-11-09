#!/usr/bin/env bash
# Score every checkpoint with the Appendix F.1 protocol as it appears.
#
#   EVAL_GPU=7 scripts/eval_watch.sh runs/sm
#
# Runs alongside training and exits once train_node.sh writes train.done.
set -u
cd "$(dirname "$0")/.."
RUN="${1:?usage: eval_watch.sh RUN_DIR}"
EVAL_GPU="${EVAL_GPU:-5}"

while true; do
  # -mmin +1 avoids reading a checkpoint torch.save is still writing.
  while IFS= read -r c; do
    j="${c%.pt}.eval.json"
    if [ ! -f "$j" ]; then
      echo "[$(date -Is)] scoring $c"
      if CUDA_VISIBLE_DEVICES="$EVAL_GPU" .venv/bin/python -m supermimic.scripts.eval \
          --checkpoint "$c" --device cuda:0 --out "$j" >> "$RUN/eval.log" 2>&1; then
        echo "[$(date -Is)] done $c"
      else
        echo "[$(date -Is)] FAILED $c (will retry)"
        rm -f "$j"
      fi
    fi
  done < <(find "$RUN" -maxdepth 1 -name 'ckpt_*.pt' -mmin +1 | sort)
  if [ -f "$RUN/train.done" ]; then
    echo "[$(date -Is)] training finished; watcher exiting"
    break
  fi
  sleep 60
done