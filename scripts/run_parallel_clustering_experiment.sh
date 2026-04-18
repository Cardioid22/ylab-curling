#!/bin/bash
# 10,000局面クラスタリング効果実験 並列実行スクリプト
# Usage: ./scripts/run_parallel_clustering_experiment.sh [num_processes]
#
# 例: 48プロセス並列（48コア想定）
#   ./scripts/run_parallel_clustering_experiment.sh 48
#
# 例: 10プロセス並列（軽めに試す場合）
#   ./scripts/run_parallel_clustering_experiment.sh 10

set -euo pipefail

# ---- 設定 ----
NUM_PROCESSES="${1:-48}"
TOTAL_POSITIONS=10000
POSITIONS_DIR="clustered_ayumu/test_positions_20260417_055725"
ROLLOUT=1
RETENTION=20

# プロジェクトルートを基準に実行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- バイナリ自動検出 (Linux/Windows 両対応) ----
BINARY_CANDIDATES=(
    "./build/Release/ylab_client.exe"   # Windows (MSVC Release)
    "./build/Debug/ylab_client.exe"     # Windows (MSVC Debug)
    "./build/ylab_client"               # Linux (Makefile)
    "./build/Release/ylab_client"       # Linux (multi-config)
)
BINARY=""
for cand in "${BINARY_CANDIDATES[@]}"; do
    if [ -x "$cand" ]; then
        BINARY="$cand"
        break
    fi
done

if [ -z "$BINARY" ]; then
    echo "Error: ylab_client binary not found. Checked:"
    for cand in "${BINARY_CANDIDATES[@]}"; do
        echo "    $cand"
    done
    echo "Run: cmake --build build --config Release"
    exit 1
fi
echo "Using binary: $BINARY"

if [ ! -d "$POSITIONS_DIR" ]; then
    echo "Error: positions directory not found: $POSITIONS_DIR"
    exit 1
fi

if [ ! -f "data/policy_param.dat" ]; then
    echo "Error: data/policy_param.dat not found"
    exit 1
fi

# 1プロセスあたりの局面数（切り上げ）
POSITIONS_PER_PROC=$(( (TOTAL_POSITIONS + NUM_PROCESSES - 1) / NUM_PROCESSES ))

# ログ保存ディレクトリ
LOG_DIR="logs/parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Parallel Clustering Experiment"
echo "========================================"
echo "  Total positions:    $TOTAL_POSITIONS"
echo "  Num processes:      $NUM_PROCESSES"
echo "  Positions/process:  $POSITIONS_PER_PROC"
echo "  Positions dir:      $POSITIONS_DIR"
echo "  Rollout count:      $ROLLOUT"
echo "  Retention:          $RETENTION%"
echo "  Deterministic:      ON"
echo "  Log dir:            $LOG_DIR"
echo "========================================"

# ---- プロセス起動 ----
PIDS=()
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    START=$((i * POSITIONS_PER_PROC))
    # 最後のプロセスは残り全部、それ以外は POSITIONS_PER_PROC 個
    if [ $((START + POSITIONS_PER_PROC)) -gt $TOTAL_POSITIONS ]; then
        MAX=$((TOTAL_POSITIONS - START))
    else
        MAX=$POSITIONS_PER_PROC
    fi

    # start_index が範囲外ならスキップ
    if [ $START -ge $TOTAL_POSITIONS ]; then
        continue
    fi

    LOG_FILE="$LOG_DIR/proc_${i}_idx${START}.log"
    printf "  [%2d] start=%5d max=%4d -> %s\n" "$i" "$START" "$MAX" "$LOG_FILE"

    "$BINARY" \
        --clustering-experiment \
        --rollout "$ROLLOUT" \
        --retention "$RETENTION" \
        --load-positions "$POSITIONS_DIR" \
        --max-positions "$MAX" \
        --start-index "$START" \
        --deterministic \
        > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} processes. Waiting for completion..."
echo "Tail any log with: tail -f $LOG_DIR/proc_0_idx0.log"
echo ""

# ---- 完了待ち ----
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  Process PID=$pid failed"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ $FAILED -eq 0 ]; then
    echo "All ${#PIDS[@]} processes completed successfully."
else
    echo "$FAILED / ${#PIDS[@]} processes failed. Check logs in $LOG_DIR/"
fi

# ---- 結果集計 ----
echo ""
echo "Combining CSV outputs..."
COMBINED="experiment_results/clustering_effectiveness_ret${RETENTION}_B${ROLLOUT}_combined_$(date +%Y%m%d_%H%M%S).csv"
FILES=(experiment_results/clustering_effectiveness_ret${RETENTION}_B${ROLLOUT}_idx*.csv)

if [ ${#FILES[@]} -gt 0 ] && [ -f "${FILES[0]}" ]; then
    # ヘッダを最初のファイルから取る
    head -1 "${FILES[0]}" > "$COMBINED"
    for f in "${FILES[@]}"; do
        tail -n +2 "$f" >> "$COMBINED"
    done
    TOTAL_ROWS=$(( $(wc -l < "$COMBINED") - 1 ))
    echo "  Combined: $COMBINED ($TOTAL_ROWS rows)"
else
    echo "  No output CSVs found (no processes completed or all failed)."
fi

echo ""
echo "Done."
