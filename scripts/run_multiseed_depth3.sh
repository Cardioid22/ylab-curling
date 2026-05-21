#!/bin/bash
################################################################################
# Multi-seed Depth-3 MCTS Experiment Runner
#
# 同じ N 局面に対して K 個の独立 seed で depth-3 MCTS 実験を回し、
# ε-PAC 用の Q 推定分散と top-pick 安定性を測れるようにする。
#
# 前提:
#   POSITIONS_DIR に「ちょうど N 局面分の batch_*.csv」が入っていること。
#   そうしないと sampleTestPositions が seed 依存のシャッフルで違う局面を選んでしまう。
#   ベースの run_parallel_depth3_experiment.sh は変更しない（薄いラップ）。
#
# Usage:
#   ./scripts/run_multiseed_depth3.sh [OPTIONS]
#
# Options:
#   --num-seeds K              seed 数 (default: 10)
#   --base-seed S              先頭 seed; S, S+1, ..., S+K-1 を使う (default: 42)
#   --num-procs N              プロセス数 (default: 8)
#   --threads-per-proc N       スレッド/プロセス (default: 16)
#   --n-states N               全テスト盤面数 (default: 8)
#   --proposed-playouts N      Proposed プレイアウト数 (default: 1000)
#   --allgrid-playouts N       AllGrid プレイアウト数 (default: 10000)
#   --proposed-rollouts N      Proposed ロールアウト数/visit (default: 20)
#   --allgrid-rollouts N       AllGrid ロールアウト数/visit (default: 10)
#   --retention RATE           保持率 (default: 0.20)
#   --positions-dir PATH       8 局面のディレクトリ (default: test_positions_multiseed8)
#   --binary PATH              ylab_client パス (default: 自動検出)
#   --parent-dir PATH          出力親ディレクトリ
#                              (default: depth3_experiment/multiseed_YYYYMMDD_HHMMSS)
#
# 例: K=10, N=8, AG=10000 で 1 週間以内
#   ./scripts/run_multiseed_depth3.sh \
#       --num-seeds 10 --n-states 8 --allgrid-playouts 10000 \
#       --positions-dir test_positions_multiseed8
################################################################################

set -euo pipefail
trap '' HUP

# ---- デフォルト ----
NUM_SEEDS=10
BASE_SEED=42
NUM_PROCESSES=8
THREADS_PER_PROC=16
N_STATES=8
PROPOSED_PLAYOUTS=1000
ALLGRID_PLAYOUTS=10000
PROPOSED_ROLLOUTS=20
ALLGRID_ROLLOUTS=10
RETENTION=0.20
POSITIONS_DIR="test_positions_multiseed8"
BINARY=""
PARENT_DIR=""

# ---- 引数 ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-seeds)         NUM_SEEDS="$2"; shift 2 ;;
        --base-seed)         BASE_SEED="$2"; shift 2 ;;
        --num-procs)         NUM_PROCESSES="$2"; shift 2 ;;
        --threads-per-proc)  THREADS_PER_PROC="$2"; shift 2 ;;
        --n-states)          N_STATES="$2"; shift 2 ;;
        --proposed-playouts) PROPOSED_PLAYOUTS="$2"; shift 2 ;;
        --allgrid-playouts)  ALLGRID_PLAYOUTS="$2"; shift 2 ;;
        --proposed-rollouts) PROPOSED_ROLLOUTS="$2"; shift 2 ;;
        --allgrid-rollouts)  ALLGRID_ROLLOUTS="$2"; shift 2 ;;
        --retention)         RETENTION="$2"; shift 2 ;;
        --positions-dir)     POSITIONS_DIR="$2"; shift 2 ;;
        --binary)            BINARY="$2"; shift 2 ;;
        --parent-dir)        PARENT_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- POSITIONS_DIR 健全性チェック ----
if [ ! -d "$POSITIONS_DIR" ]; then
    echo "Error: POSITIONS_DIR not found: $POSITIONS_DIR" >&2
    echo "  prepare with:" >&2
    echo "    python scripts/pick_multiseed_positions.py --src-dir <SRC> --out-dir $POSITIONS_DIR --n $N_STATES" >&2
    exit 1
fi

# POSITIONS_DIR 内の総行数を数える (header を引いて n_states 一致確認)
total_rows=0
for f in "$POSITIONS_DIR"/batch_*.csv; do
    [ -f "$f" ] || continue
    rows=$(( $(wc -l < "$f") - 1 ))
    total_rows=$(( total_rows + rows ))
done
if [ "$total_rows" -ne "$N_STATES" ]; then
    echo "Error: POSITIONS_DIR has $total_rows rows but n_states=$N_STATES." >&2
    echo "  multi-seed が成立するには行数 == n_states が必要。" >&2
    exit 1
fi

# ---- 親ディレクトリ ----
if [ -z "$PARENT_DIR" ]; then
    TS=$(date +%Y%m%d_%H%M%S)
    PARENT_DIR="depth3_experiment/multiseed_${TS}"
fi
mkdir -p "$PARENT_DIR"

# ---- 設定メモ ----
cat > "$PARENT_DIR/multiseed_config.txt" <<EOF
Multi-seed Depth-3 MCTS run
============================
Timestamp:               $(date)
Num seeds (K):           $NUM_SEEDS
Base seed:               $BASE_SEED
Num processes:           $NUM_PROCESSES
Threads per process:     $THREADS_PER_PROC
N states (per seed):     $N_STATES
Proposed playouts:       $PROPOSED_PLAYOUTS
AllGrid playouts:        $ALLGRID_PLAYOUTS
Proposed rollouts/visit: $PROPOSED_ROLLOUTS
AllGrid rollouts/visit:  $ALLGRID_ROLLOUTS
Retention rate:          $RETENTION
Positions dir:           $POSITIONS_DIR
Total rows in positions: $total_rows
Parent dir:              $PARENT_DIR
EOF

echo "========================================"
cat "$PARENT_DIR/multiseed_config.txt"
echo "========================================"
echo ""

# ---- K seed をシーケンシャルに回す ----
INNER="$SCRIPT_DIR/run_parallel_depth3_experiment.sh"
if [ ! -x "$INNER" ]; then
    echo "Error: inner script not executable: $INNER" >&2
    exit 1
fi

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((BASE_SEED + i))
    SEED_DIR="$PARENT_DIR/seed_${SEED}"
    mkdir -p "$SEED_DIR"

    echo "==============================="
    echo "  [seed $((i+1))/$NUM_SEEDS] seed=$SEED -> $SEED_DIR"
    echo "==============================="
    SEED_START=$(date +%s)

    # 内側スクリプトに渡す。--parent-dir 相当が無いので環境変数で乗っ取る方式を取らず、
    # 内側を一旦動かしてから生成物を SEED_DIR に move する。
    INNER_ARGS=(
        --num-procs "$NUM_PROCESSES"
        --threads-per-proc "$THREADS_PER_PROC"
        --n-states "$N_STATES"
        --proposed-playouts "$PROPOSED_PLAYOUTS"
        --allgrid-playouts "$ALLGRID_PLAYOUTS"
        --proposed-rollouts "$PROPOSED_ROLLOUTS"
        --allgrid-rollouts "$ALLGRID_ROLLOUTS"
        --retention "$RETENTION"
        --seed "$SEED"
        --positions-dir "$POSITIONS_DIR"
    )
    if [ -n "$BINARY" ]; then
        INNER_ARGS+=( --binary "$BINARY" )
    fi

    # 内側スクリプトは depth3_run_<TS>_parallel/ を experiments/ 配下に作るので
    # 走らせた直後にそれを move する
    BEFORE_TS=$(ls -1 experiments/ 2>/dev/null | grep '^depth3_run_.*_parallel$' || true)

    bash "$INNER" "${INNER_ARGS[@]}" \
        > "$SEED_DIR/run.log" 2>&1

    # 新規生成ディレクトリを特定して seed_dir に move
    AFTER_TS=$(ls -1 experiments/ 2>/dev/null | grep '^depth3_run_.*_parallel$' || true)
    NEW=$(comm -13 <(echo "$BEFORE_TS" | sort) <(echo "$AFTER_TS" | sort) | head -1 || true)
    if [ -n "${NEW:-}" ] && [ -d "experiments/$NEW" ]; then
        mv "experiments/$NEW"/* "$SEED_DIR/" 2>/dev/null || true
        rmdir "experiments/$NEW" 2>/dev/null || true
    else
        # fallback: experiments/depth3_results/ も探す (run_config に依存)
        if [ -d "experiments/depth3_results" ] && [ -n "$(ls -A experiments/depth3_results 2>/dev/null)" ]; then
            mv experiments/depth3_results/* "$SEED_DIR/" 2>/dev/null || true
        fi
    fi

    SEED_END=$(date +%s)
    echo "  seed=$SEED took $((SEED_END - SEED_START)) sec"
done

echo ""
echo "========================================"
echo "All $NUM_SEEDS seeds done. Parent: $PARENT_DIR"
echo ""
echo "Aggregate metrics with:"
echo "  python scripts/aggregate_multiseed_depth3.py --parent-dir $PARENT_DIR"
echo "========================================"
