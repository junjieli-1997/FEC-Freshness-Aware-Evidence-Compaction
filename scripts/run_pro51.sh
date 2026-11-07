#!/usr/bin/env bash
# Run several methods on SWE-bench Pro HARD-51.  Usage: scripts/run_pro51.sh [model] [method ...]
#   scripts/run_pro51.sh claude-haiku-5-5 baseline cliff fec
set -euo pipefail
cd "$(dirname "$0")/.."
model=${1:-claude-haiku-5-5}; shift || true
methods=${*:-baseline cliff fec}
tag=${model//[-.]/_}
for m in $methods; do
  echo "== $(date +%F_%T) $model $m / pro51"
  python3 -m agentkit.runner --bench pro --model "$model" --method "$m" \
      --tasks tasks/pro_hard51.json --out "results/$tag/pro51/$m" --workers "${WORKERS:-2}"
done
