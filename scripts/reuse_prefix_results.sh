#!/bin/bash
################################################################################
# 既存キャンペーンの結果を、局面セットを「先頭に保ったまま拡張した」新キャンペーンへ流用する
#
# 例: test_positions500 の先頭 200 行 = test_positions200 (同一順序) なので、
#     gpw_experiment/ (run200) の全結果・審判をそのまま gpw_experiment500/ にコピーし、
#     新規 300 局面だけを --start-index 200 --max-positions 300 で回す (出力は *_idx200.csv)。
# aggregate_reinvest.py / position_features.py は <ARM>/seed_*/reinvest_results*.csv,
# cluster_table*.csv, referee/score_move_qtable*.csv を全部読むので、ファイル名の衝突が無ければよい。
# (run200 由来: *_idx0.csv, *_idx50.csv, 無印 (A9P5/A9R05 の 200 局面); run500 由来: *_idx200.csv)
#
# Usage: ./scripts/reuse_prefix_results.sh --src gpw_experiment --dst gpw_experiment500 [--arms "A1,A2,A5,A9,A9P5"]
#   (--arms 省略時は src 直下の A* ディレクトリ全部。run.log はコピーしない)
################################################################################
set -euo pipefail
SRC=""; DST=""; ARMS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --src)  SRC="$2"; shift 2 ;;
        --dst)  DST="$2"; shift 2 ;;
        --arms) ARMS="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
[ -n "$SRC" ] && [ -n "$DST" ] || { echo "need --src and --dst" >&2; exit 1; }
if [ -z "$ARMS" ]; then
    ARMS="$(ls -d "$SRC"/A* 2>/dev/null | xargs -n1 basename | paste -sd, -)"
fi
n=0
IFS=',' read -ra ARM_LIST <<< "$ARMS"
for ARM in "${ARM_LIST[@]}"; do
    for sd in "$SRC/$ARM"/seed_*; do
        [ -d "$sd" ] || continue
        out="$DST/$ARM/$(basename "$sd")"; mkdir -p "$out"
        for f in "$sd"/reinvest_results*.csv "$sd"/cluster_table*.csv; do
            [ -f "$f" ] || continue
            cp "$f" "$out/"; n=$((n + 1))
        done
    done
done
mkdir -p "$DST/referee"
for f in "$SRC"/referee/score_move_qtable*.csv; do
    [ -f "$f" ] || continue
    cp "$f" "$DST/referee/"; n=$((n + 1))
done
echo "copied $n files from $SRC to $DST (arms: $ARMS)"
echo "next: run new positions with --start-index <N_existing> --max-positions <N_new> into $DST"
