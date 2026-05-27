#!/bin/bash
################################################################################
# Multi-seed Depth-3 MCTS Experiment Runner (concurrent)
#
# 同じ N 局面に対して K 個の独立 seed で depth-3 MCTS 実験を回し、
# ε-PAC 用の Q 推定分散と top-pick 安定性を測れるようにする。
#
# 重要な設計:
#   - 1 局面の MCTS はこのコードでは並列化されない (スレッドは state キューから
#     丸ごと 1 局面を引く)。よって N 局面なら最大 N スレッドしか効かない。
#   - 各 seed を「1 プロセス × N スレッド」で起動し、K seed を *同時並列* で走らせる。
#     → 使用コア = K × N。最遅局面の単発時間が wall-clock を決める (seed 逐次の K 倍を回避)。
#   - C++ には触らない。binary を直接呼ぶ (inner script は使わない)。
#
# 前提:
#   POSITIONS_DIR に「ちょうど N 局面分の batch_*.csv」が入っていること。
#   そうしないと sampleTestPositions が seed 依存シャッフルで違う局面を選んでしまう。
#
# Usage:
#   ./scripts/run_multiseed_depth3.sh [OPTIONS]
#
# Options:
#   --num-seeds K              seed 数 (default: 10)
#   --base-seed S              先頭 seed; S..S+K-1 を使う (default: 42)
#   --n-states N               テスト局面数 (= POSITIONS_DIR の行数) (default: 8)
#   --threads-per-seed T       1 seed プロセスのスレッド数 (default: N_STATES)
#   --max-parallel-seeds M     同時に走らせる seed 数の上限 (default: K = 全部同時)
#   --proposed-playouts N      (default: 1000)
#   --allgrid-playouts N       (default: 10000)
#   --proposed-rollouts N      (default: 20)
#   --allgrid-rollouts N       (default: 10)
#   --retention RATE           (default: 0.20)
#   --positions-dir PATH       N 局面のディレクトリ (default: test_positions_multiseed8)
#   --binary PATH              ylab_client パス (default: 自動検出)
#   --parent-dir PATH          出力親 (default: depth3_experiment/multiseed_YYYYMMDD_HHMMSS)
#
# 例: K=10, N=8 を全 seed 同時 (80 コア使用、wall-clock ≈ 最遅局面 ~7 日)
#   ./scripts/run_multiseed_depth3.sh \
#       --num-seeds 10 --n-states 8 \
#       --allgrid-playouts 10000 \
#       --positions-dir test_positions_multiseed8
################################################################################

set -euo pipefail
trap '' HUP

# ---- デフォルト ----
NUM_SEEDS=10
BASE_SEED=42
N_STATES=8
THREADS_PER_SEED=""        # 空なら N_STATES に合わせる
MAX_PARALLEL_SEEDS=""      # 空なら NUM_SEEDS (全部同時)
PROPOSED_PLAYOUTS=1000
ALLGRID_PLAYOUTS=10000
PROPOSED_ROLLOUTS=20
ALLGRID_ROLLOUTS=10
RETENTION=0.20
POSITIONS_DIR="test_positions_multiseed8"
BINARY=""
PARENT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-seeds)           NUM_SEEDS="$2"; shift 2 ;;
        --base-seed)           BASE_SEED="$2"; shift 2 ;;
        --n-states)            N_STATES="$2"; shift 2 ;;
        --threads-per-seed)    THREADS_PER_SEED="$2"; shift 2 ;;
        --max-parallel-seeds)  MAX_PARALLEL_SEEDS="$2"; shift 2 ;;
        --proposed-playouts)   PROPOSED_PLAYOUTS="$2"; shift 2 ;;
        --allgrid-playouts)    ALLGRID_PLAYOUTS="$2"; shift 2 ;;
        --proposed-rollouts)   PROPOSED_ROLLOUTS="$2"; shift 2 ;;
        --allgrid-rollouts)    ALLGRID_ROLLOUTS="$2"; shift 2 ;;
        --retention)           RETENTION="$2"; shift 2 ;;
        --positions-dir)       POSITIONS_DIR="$2"; shift 2 ;;
        --binary)              BINARY="$2"; shift 2 ;;
        --parent-dir)          PARENT_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

[ -z "$THREADS_PER_SEED" ]   && THREADS_PER_SEED="$N_STATES"
[ -z "$MAX_PARALLEL_SEEDS" ] && MAX_PARALLEL_SEEDS="$NUM_SEEDS"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- バイナリ自動検出 ----
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

# ---- POSITIONS_DIR 健全性チェック (行数 == n_states) ----
if [ ! -d "$POSITIONS_DIR" ]; then
    echo "Error: POSITIONS_DIR not found: $POSITIONS_DIR" >&2
    echo "  prepare with: python scripts/pick_multiseed_positions.py --src-dir <SRC> --out-dir $POSITIONS_DIR --n $N_STATES" >&2
    exit 1
fi
total_rows=0
for f in "$POSITIONS_DIR"/batch_*.csv; do
    [ -f "$f" ] || continue
    total_rows=$(( total_rows + $(wc -l < "$f") - 1 ))
done
if [ "$total_rows" -ne "$N_STATES" ]; then
    echo "Error: POSITIONS_DIR has $total_rows rows but n_states=$N_STATES." >&2
    echo "  multi-seed が成立するには 行数 == n_states が必要 (シャッフル回避)。" >&2
    exit 1
fi

# ---- コア予算チェック ----
TOTAL_CORES=$(( MAX_PARALLEL_SEEDS * THREADS_PER_SEED ))
NPROC=$(nproc 2>/dev/null || echo 0)
echo "Concurrent core budget: ${MAX_PARALLEL_SEEDS} seeds x ${THREADS_PER_SEED} threads = ${TOTAL_CORES} cores (machine has ${NPROC})"
if [ "$NPROC" -gt 0 ] && [ "$TOTAL_CORES" -gt "$NPROC" ]; then
    echo "  WARNING: oversubscription. Consider lowering --max-parallel-seeds or --threads-per-seed." >&2
fi

# ---- 親ディレクトリ ----
if [ -z "$PARENT_DIR" ]; then
    PARENT_DIR="depth3_experiment/multiseed_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$PARENT_DIR"

cat > "$PARENT_DIR/multiseed_config.txt" <<EOF
Multi-seed Depth-3 MCTS run (concurrent)
========================================
Timestamp:               $(date)
Num seeds (K):           $NUM_SEEDS
Base seed:               $BASE_SEED  (uses $BASE_SEED..$((BASE_SEED + NUM_SEEDS - 1)))
N states (per seed):     $N_STATES
Threads per seed:        $THREADS_PER_SEED
Max parallel seeds:      $MAX_PARALLEL_SEEDS
Proposed playouts:       $PROPOSED_PLAYOUTS
AllGrid playouts:        $ALLGRID_PLAYOUTS
Proposed rollouts/visit: $PROPOSED_ROLLOUTS
AllGrid rollouts/visit:  $ALLGRID_ROLLOUTS
Retention rate:          $RETENTION
Positions dir:           $POSITIONS_DIR
Binary:                  $BINARY
Parent dir:              $PARENT_DIR
EOF

echo "========================================"
cat "$PARENT_DIR/multiseed_config.txt"
echo "========================================"
echo ""

# ---- seed プロセスを並列起動 (max-parallel-seeds で throttle) ----
declare -a PIDS=()
declare -a PID_SEED=()

launch_seed() {
    local seed="$1"
    local seed_dir="$PARENT_DIR/seed_${seed}"
    mkdir -p "$seed_dir"
    nohup "$BINARY" \
        --depth3-mcts \
        --states "$N_STATES" \
        --proposed-playouts "$PROPOSED_PLAYOUTS" \
        --allgrid-playouts "$ALLGRID_PLAYOUTS" \
        --proposed-rollouts "$PROPOSED_ROLLOUTS" \
        --allgrid-rollouts "$ALLGRID_ROLLOUTS" \
        --retention "$RETENTION" \
        --threads "$THREADS_PER_SEED" \
        --seed "$seed" \
        --load-positions "$POSITIONS_DIR" \
        --output-dir "$seed_dir" \
        < /dev/null > "$seed_dir/run.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    PID_SEED+=("$seed")
    echo "  launched seed=$seed pid=$pid -> $seed_dir"
}

running_jobs() { jobs -rp | wc -l; }

for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((BASE_SEED + i))
    # throttle: 走行中ジョブが上限に達していたら 1 つ終わるまで待つ
    while [ "$(running_jobs)" -ge "$MAX_PARALLEL_SEEDS" ]; do
        wait -n 2>/dev/null || true
    done
    launch_seed "$SEED"
done

echo ""
echo "All $NUM_SEEDS seeds launched. Waiting for completion..."
echo "  monitor: tail -f $PARENT_DIR/seed_*/run.log"
echo ""

# ---- 全完了待ち ----
FAILED=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    seed="${PID_SEED[$idx]}"
    if wait "$pid"; then
        echo "  seed=$seed (pid=$pid) done"
    else
        echo "  seed=$seed (pid=$pid) FAILED" >&2
        FAILED=$((FAILED + 1))
    fi
done

# ---- 各 seed の出力を aggregator が読める名前に揃える ----
# binary は (スライス無しなら) depth3_results.csv を出すので COMBINED にコピー
for i in $(seq 0 $((NUM_SEEDS - 1))); do
    SEED=$((BASE_SEED + i))
    SD="$PARENT_DIR/seed_${SEED}"
    if [ -f "$SD/depth3_results.csv" ]; then
        cp "$SD/depth3_results.csv" "$SD/depth3_results_COMBINED.csv"
    fi
done

echo ""
echo "========================================"
if [ "$FAILED" -eq 0 ]; then
    echo "All $NUM_SEEDS seeds completed. Parent: $PARENT_DIR"
else
    echo "$FAILED / $NUM_SEEDS seeds FAILED. Check $PARENT_DIR/seed_*/run.log" >&2
fi
echo ""
echo "Aggregate with:"
echo "  python3 scripts/aggregate_multiseed_depth3.py --parent-dir $PARENT_DIR --epsilon 0.5"
echo "========================================"
