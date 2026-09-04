#!/bin/bash
# Parallel self-play data generation on a lab server (Linux build).
#
#   scripts/gen_selfplay.sh <out_dir> <n_procs> <games_per_proc> [threads_per_proc] [budget_sec] [explore] [extra args...]
#
# Example (bear, 128 logical cores): 16 procs x 6 threads, 40 games each, 0.4 s/shot:
#   nohup bash scripts/gen_selfplay.sh data/gen1 16 40 6 0.4 0.15 > data/gen1_launch.log 2>&1 &
set -u
OUT=${1:?out_dir}; NPROC=${2:?n_procs}; GAMES=${3:?games_per_proc}
THREADS=${4:-6}; BUDGET=${5:-0.4}; EXPLORE=${6:-0.15}
shift 6 2>/dev/null || shift $#
EXTRA="$*"
BIN=${GPW_BIN:-./build/gpw_agent}
HOST=$(hostname -s)
mkdir -p "$OUT"
for i in $(seq 0 $((NPROC - 1))); do
  SEED=$(( (RANDOM * 32768 + RANDOM) % 1000000 ))
  "$BIN" --selfplay --games "$GAMES" --ends 10 --threads "$THREADS" \
      --budget-a "$BUDGET" --budget-b "$BUDGET" --explore "$EXPLORE" --seed "$SEED" --quiet \
      --out "$OUT/sp_${HOST}_${i}.jsonl" $EXTRA > "$OUT/log_${HOST}_${i}.txt" 2>&1 &
done
wait
echo "all $NPROC processes finished"
