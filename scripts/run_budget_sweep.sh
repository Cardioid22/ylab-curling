#!/bin/bash
################################################################################
# 予算スイープ・ドライバ (efficiency曲線用)
#
# 1台のマシンが、自分の担当局面で、複数の playout 予算 P を「順番に」回す。
# 各 P は run_reinvest.sh を1回呼び、コアを使い切るので予算間は逐次実行。
# 審判(q*)は予算非依存なので再利用する → ここではアームのみ。
#
# 出力: <parent-base>/P<P>_<tag>/   (例 experiments/sweep/P500_bear/)
# 後段: 各 P で4サーバー分を結合 → 既存 run50 審判で aggregate → regret vs P 曲線。
#
# Usage (各マシンで1回, nohup 推奨):
#   nohup ./scripts/run_budget_sweep.sh \
#       --positions-dir test_positions50_bear --n-states 17 --tag bear \
#       --threads-per-seed 6 --max-parallel 20 \
#       --playouts-list "50 100 500 1000" \
#       > sweep_bear.log 2>&1 &
################################################################################
set -euo pipefail
trap '' HUP

ARMS="A1,A2,A5,A7"
NUM_SEEDS=5
ROLLOUTS=10                 # R は全予算で固定 (P=200 の既存結果と揃える)
POSITIONS_DIR=""
N_STATES=""
TAG=""
THREADS_PER_SEED=""
MAX_PARALLEL="20"
PLAYOUTS_LIST="50 100 500 1000"
PARENT_BASE="experiments/sweep"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)             ARMS="$2"; shift 2 ;;
        --num-seeds)        NUM_SEEDS="$2"; shift 2 ;;
        --rollouts)         ROLLOUTS="$2"; shift 2 ;;
        --positions-dir)    POSITIONS_DIR="$2"; shift 2 ;;
        --n-states)         N_STATES="$2"; shift 2 ;;
        --tag)              TAG="$2"; shift 2 ;;
        --threads-per-seed) THREADS_PER_SEED="$2"; shift 2 ;;
        --max-parallel)     MAX_PARALLEL="$2"; shift 2 ;;
        --playouts-list)    PLAYOUTS_LIST="$2"; shift 2 ;;
        --parent-base)      PARENT_BASE="$2"; shift 2 ;;
        -h|--help)          grep '^#' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[ -z "$POSITIONS_DIR" ] && { echo "Error: --positions-dir required" >&2; exit 1; }
[ -z "$N_STATES" ]      && { echo "Error: --n-states required" >&2; exit 1; }
[ -z "$TAG" ]           && { echo "Error: --tag required" >&2; exit 1; }
[ -z "$THREADS_PER_SEED" ] && { echo "Error: --threads-per-seed required" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== budget sweep: tag=$TAG positions=$POSITIONS_DIR n_states=$N_STATES ==="
echo "    budgets P = $PLAYOUTS_LIST   (R=$ROLLOUTS, arms=$ARMS, seeds=$NUM_SEEDS)"
echo "    threads-per-seed=$THREADS_PER_SEED max-parallel=$MAX_PARALLEL"
echo ""

for P in $PLAYOUTS_LIST; do
    out="$PARENT_BASE/P${P}_${TAG}"
    echo "######## [$TAG] P=$P -> $out ########  $(date 2>/dev/null || true)"
    "$SCRIPT_DIR/run_reinvest.sh" \
        --arms "$ARMS" --num-seeds "$NUM_SEEDS" \
        --positions-dir "$POSITIONS_DIR" --n-states "$N_STATES" \
        --threads-per-seed "$THREADS_PER_SEED" --max-parallel "$MAX_PARALLEL" \
        --playouts "$P" --rollouts "$ROLLOUTS" \
        --parent-dir "$out"
    echo "######## [$TAG] P=$P done ########"
    echo ""
done

echo "=== [$TAG] ALL budgets done: $PLAYOUTS_LIST ==="
