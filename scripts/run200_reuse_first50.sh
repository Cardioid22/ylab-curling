#!/bin/bash
################################################################################
# run200 キャンペーン: 先頭 50 局面の既存結果 (run50v2, 修正木) を再利用する
#
# test_positions200 の先頭 50 行は test_positions50 と完全に同一 (同じ順序) なので、
# global index 0..49 の state_seed も同一 → run50v2 の結果をそのまま流用できる
# (物理ノイズは未シードなので厳密に同一ではないが統計的に等価)。
#
# やること:
#   run50v2/<ARM>/seed_S/reinvest_results.csv -> run200/<ARM>/seed_S/reinvest_results_idx0.csv
#   run50v2/<ARM>/seed_S/cluster_table.csv    -> run200/<ARM>/seed_S/cluster_table_idx0.csv
#   審判: scorescreen/run50/referee/score_move_qtable.csv -> run200/referee/score_move_qtable_idx0.csv
# 残り 150 局面は run_reinvest.sh --start-index 50 --max-positions 150 で回す
# (出力は *_idx50.csv)。aggregate_reinvest.py / position_features.py は *_idx*.csv を全部読む。
#
# Usage: ./scripts/run200_reuse_first50.sh [--src reinvest_experiment/run50v2] [--dst reinvest_experiment/run200] [--arms "A1,A2,A5,A7,A8,A9"]
################################################################################
set -euo pipefail

SRC="reinvest_experiment/run50v2"
DST="reinvest_experiment/run200"
REFEREE_SRC="reinvest_experiment/scorescreen/run50/referee/score_move_qtable.csv"
ARMS="A1,A2,A5,A7,A8,A9"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)  SRC="$2"; shift 2 ;;
        --dst)  DST="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        --referee-src) REFEREE_SRC="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

n=0
IFS=',' read -ra ARM_LIST <<< "$ARMS"
for ARM in "${ARM_LIST[@]}"; do
    for sd in "$SRC/$ARM"/seed_*; do
        [ -d "$sd" ] || continue
        seed="$(basename "$sd")"
        out="$DST/$ARM/$seed"
        mkdir -p "$out"
        if [ -f "$sd/reinvest_results.csv" ]; then
            cp "$sd/reinvest_results.csv" "$out/reinvest_results_idx0.csv"; n=$((n + 1))
        fi
        if [ -f "$sd/cluster_table.csv" ]; then
            cp "$sd/cluster_table.csv" "$out/cluster_table_idx0.csv"
        fi
    done
done
mkdir -p "$DST/referee"
if [ -f "$REFEREE_SRC" ]; then
    cp "$REFEREE_SRC" "$DST/referee/score_move_qtable_idx0.csv"
    echo "referee: $REFEREE_SRC -> $DST/referee/score_move_qtable_idx0.csv"
else
    echo "WARNING: referee source not found: $REFEREE_SRC" >&2
fi
echo "copied $n reinvest_results.csv (+cluster_table) from $SRC to $DST as *_idx0.csv"
echo "next: run the remaining 150 positions with --start-index 50 --max-positions 150 (see scripts/README_run200.md)"
