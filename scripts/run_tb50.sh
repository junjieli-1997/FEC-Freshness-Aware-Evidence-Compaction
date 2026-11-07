#!/usr/bin/env bash
# Run several methods on Terminal-Bench 2.1 (50 tasks).  Usage: scripts/run_tb50.sh [model] [method ...]
#   scripts/run_tb50.sh claude-haiku-5-5 baseline cliff fec
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:-claude-haiku-5-5}; shift || true
methods=${*:-baseline cliff fec}
tag=${model//[-.]/_}
for m in $methods; do
  echo "== $(date +%F_%T) $model $m / tb50"
  python3 -m agentkit.runner --bench tb --model "$model" --method "$m" \
      --tasks tasks/tb_hard50.json --out "results/$tag/tb50/$m" --workers "${WORKERS:-2}"
done
