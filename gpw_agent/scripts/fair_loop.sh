#!/bin/bash
# Overnight loop of fair local matches vs Jiritsukun-Jr (4 threads each side), alternating sides.
#   bash scripts/fair_loop.sh <logroot> <n_pairs> [eval_file]
set -u
ROOTLOG=${1:?logroot}; NPAIRS=${2:-10}; EVAL=${3:-data/eval_local_v4.txt}
cd "$(dirname "$0")/.."
mkdir -p "$ROOTLOG"
for (( i=0; i<NPAIRS; i++ )); do
  for side in 0 1; do
    D="$ROOTLOG/p${i}_s${side}"
    GPW_BIN=./build_dev/Release/gpw_agent.exe GPW_EXTRA="--eval $EVAL" JIR_THREADS=4 \
      bash scripts/match_local_jiritsu.sh "$D" 4 config.json $side > "$ROOTLOG/p${i}_s${side}.txt" 2>&1
    R=$(grep "game_over" "$D/gpw.log")
    echo "$(date +%m-%d\ %H:%M) pair $i side $side: $R" | tee -a "$ROOTLOG/summary.txt"
  done
done
echo "loop finished $(date)" >> "$ROOTLOG/summary.txt"
