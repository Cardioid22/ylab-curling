#!/bin/bash
################################################################################
# 審判 (Q_ref) ランナー — 全アーム共通の物差しを1回だけ生成する
#
# 各局面の全候補手を「その手を着手 → エンド終了まで K 回ロールアウト」して平均スコアで
# 採点した Q テーブルを出す (experiments/score_move_experiment)。
# 高 K (既定 1000) で1回走らせ、全アームの選択手を (game_id,end,shot_num,candidate_idx)
# で join してリグレットを測る。実装は --score-move (初手リサンプルがデフォルト)。
#
# 重要: アームと同じ POSITIONS_DIR / N_STATES / SEED 規約を使うこと
#       (候補 index = generatePool 順を一致させるため。SEED は審判の内部 RNG にのみ影響し、
#        候補生成は決定的なので join キーは一致する)。
#
# Usage:
#   ./scripts/run_referee.sh [OPTIONS]
#
# Options:
#   --k K                 候補1手あたりのロールアウト回数 (default: 1000)
#   --positions-dir PATH  (default: test_positions_isobudget10)
#   --n-states N          (default: 10)
#   --threads T           (default: 16)
#   --seed S              審判 RNG ベースシード (default: 42)
#   --output-dir PATH     (default: experiments/reinvest_referee)
#   --binary PATH         (default: 自動検出)
#   --frozen-first-shot   初手ノイズを固定 (旧挙動)。既定は毎回リサンプル (実行不確実性込み)
################################################################################

set -euo pipefail

K=1000
POSITIONS_DIR="test_positions_isobudget10"
N_STATES=10
THREADS=16
SEED=42
OUTPUT_DIR="experiments/reinvest_referee"
BINARY=""
FROZEN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --k)                 K="$2"; shift 2 ;;
        --positions-dir)     POSITIONS_DIR="$2"; shift 2 ;;
        --n-states)          N_STATES="$2"; shift 2 ;;
        --threads)           THREADS="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --output-dir)        OUTPUT_DIR="$2"; shift 2 ;;
        --binary)            BINARY="$2"; shift 2 ;;
        --frozen-first-shot) FROZEN="--frozen-first-shot"; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [ -z "$BINARY" ]; then
    for cand in \
        "./build/Release/ylab_client" \
        "./build/ylab_client" \
        "./build/Release/ylab_client.exe" \
        "./build/Debug/ylab_client.exe" \
        "./ylab_client" ; do
        if [ -x "$cand" ]; then BINARY="$cand"; break; fi
    done
fi
if [ -z "$BINARY" ] || [ ! -x "$BINARY" ]; then
    echo "Error: ylab_client binary not found. Use --binary PATH or build first." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Running referee: K=$K positions=$POSITIONS_DIR n_states=$N_STATES seed=$SEED -> $OUTPUT_DIR"
"$BINARY" \
    --score-move \
    --score-rollouts "$K" \
    --states "$N_STATES" \
    --threads "$THREADS" \
    --seed "$SEED" \
    --load-positions "$POSITIONS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    $FROZEN \
    2>&1 | tee "$OUTPUT_DIR/referee.log"

echo ""
echo "Referee done. Q table: $OUTPUT_DIR/score_move_qtable*.csv"
