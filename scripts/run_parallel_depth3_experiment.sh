#!/bin/bash
################################################################################
# Parallel Depth-3 MCTS Experiment Runner
#
# 深さ3 MCTS 実験 (Proposed vs AllGrid) を複数プロセスで並列実行する。
# 100盤面を NUM_PROCESSES プロセスに分割、各プロセスは内部でさらに
# THREADS_PER_PROC スレッドで処理する。
# nohup でSSH切断後も継続。
#
# Usage:
#   ./scripts/run_parallel_depth3_experiment.sh [OPTIONS]
#
# Options:
#   --num-procs N             プロセス数 (default: 8)
#   --threads-per-proc N      1プロセスあたりのスレッド数 (default: 16)
#   --n-states N              全テスト盤面数 (default: 100)
#   --proposed-playouts N     Proposed プレイアウト数 (default: 1000)
#   --allgrid-playouts N      AllGrid プレイアウト数 (default: 5000)
#   --proposed-rollouts N     Proposed ロールアウト数/visit (default: 20)
#   --allgrid-rollouts N      AllGrid ロールアウト数/visit (default: 10)
#   --retention RATE          保持率 (0.0〜1.0, default: 0.20)
#   --seed S                  乱数シード (default: 42)
#   --positions-dir PATH      テスト盤面ディレクトリ
#                             (default: ~/ylab-curling/test_positions_20260417_055725)
#   --binary PATH             ylab_client バイナリパス (default: 自動検出)
#   --skip-combine            最後の CSV 結合をスキップ
#
# Examples:
#   # 推奨: bear で 8プロセス × 16スレッド = 128論理コアフル使用
#   ./scripts/run_parallel_depth3_experiment.sh
#
#   # 軽量テスト
#   ./scripts/run_parallel_depth3_experiment.sh \
#       --num-procs 2 --threads-per-proc 4 --n-states 10 \
#       --proposed-playouts 100 --allgrid-playouts 500
#
# SSH切断対策（推奨）:
#   tmux new -s exp
#   ./scripts/run_parallel_depth3_experiment.sh
#   (Ctrl-B, D で detach → 後で tmux attach -t exp)
################################################################################

set -euo pipefail

# SSH切断時のSIGHUP無視
trap '' HUP

# ---- デフォルト設定 ----
NUM_PROCESSES=8
THREADS_PER_PROC=16
N_STATES=100
PROPOSED_PLAYOUTS=1000
ALLGRID_PLAYOUTS=5000
PROPOSED_ROLLOUTS=20
ALLGRID_ROLLOUTS=10
RETENTION=0.20
SEED=42
POSITIONS_DIR="$HOME/ylab-curling/test_positions_20260417_055725"
BINARY=""
SKIP_COMBINE=0

# ---- 引数パース ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-procs)          NUM_PROCESSES="$2"; shift 2 ;;
        --threads-per-proc)   THREADS_PER_PROC="$2"; shift 2 ;;
        --n-states)           N_STATES="$2"; shift 2 ;;
        --proposed-playouts)  PROPOSED_PLAYOUTS="$2"; shift 2 ;;
        --allgrid-playouts)   ALLGRID_PLAYOUTS="$2"; shift 2 ;;
        --proposed-rollouts)  PROPOSED_ROLLOUTS="$2"; shift 2 ;;
        --allgrid-rollouts)   ALLGRID_ROLLOUTS="$2"; shift 2 ;;
        --retention)          RETENTION="$2"; shift 2 ;;
        --seed)               SEED="$2"; shift 2 ;;
        --positions-dir)      POSITIONS_DIR="$2"; shift 2 ;;
        --binary)             BINARY="$2"; shift 2 ;;
        --skip-combine)       SKIP_COMBINE=1; shift ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num-procs N] [--threads-per-proc N] ..."
            exit 1 ;;
    esac
done

# プロジェクトルートへ移動
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
        "./ylab_client" \
        "./ylab_client.exe"
    do
        if [ -x "$cand" ]; then BINARY="$cand"; break; fi
    done
fi

# ---- 事前チェック ----
if [ -z "$BINARY" ] || [ ! -x "$BINARY" ]; then
    echo "Error: ylab_client binary not found."
    echo "Specify with: --binary /path/to/ylab_client"
    echo "Or build: cmake --build build --config Release"
    exit 1
fi

if [ ! -d "$POSITIONS_DIR" ]; then
    echo "Error: positions directory not found: $POSITIONS_DIR"
    echo "Specify with: --positions-dir /path/to/test_positions_YYYYMMDD_HHMMSS"
    exit 1
fi

# 1プロセスあたりの盤面数（切り上げ）
POSITIONS_PER_PROC=$(( (N_STATES + NUM_PROCESSES - 1) / NUM_PROCESSES ))

# この実行専用のディレクトリ（既存実験データと衝突しないようタイムスタンプで一意化）
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="experiments/depth3_run_${TIMESTAMP}_parallel"
LOG_DIR="$RUN_DIR/logs"
OUTPUT_DIR="$RUN_DIR"
mkdir -p "$LOG_DIR"

# ---- 設定表示 ----
echo "========================================"
echo "Parallel Depth-3 MCTS Experiment"
echo "========================================"
echo "  Binary:                  $BINARY"
echo "  Positions dir:           $POSITIONS_DIR"
echo "  N states (total):        $N_STATES"
echo "  Num processes:           $NUM_PROCESSES"
echo "  Threads per process:     $THREADS_PER_PROC"
echo "  Total logical cores:     $(( NUM_PROCESSES * THREADS_PER_PROC ))"
echo "  Positions/process:       $POSITIONS_PER_PROC"
echo "  Proposed playouts:       $PROPOSED_PLAYOUTS"
echo "  AllGrid playouts:        $ALLGRID_PLAYOUTS"
echo "  Proposed rollouts/visit: $PROPOSED_ROLLOUTS"
echo "  AllGrid rollouts/visit:  $ALLGRID_ROLLOUTS"
echo "  Retention rate:          $RETENTION"
echo "  Seed:                    $SEED"
echo "  Run dir:                 $RUN_DIR"
echo "========================================"
echo ""

# 実行条件をメモ
cat > "$RUN_DIR/run_config.txt" <<EOF
Timestamp:               $TIMESTAMP
Num processes:           $NUM_PROCESSES
Threads per process:     $THREADS_PER_PROC
N states (total):        $N_STATES
Positions/process:       $POSITIONS_PER_PROC
Proposed playouts:       $PROPOSED_PLAYOUTS
AllGrid playouts:        $ALLGRID_PLAYOUTS
Proposed rollouts/visit: $PROPOSED_ROLLOUTS
AllGrid rollouts/visit:  $ALLGRID_ROLLOUTS
Retention rate:          $RETENTION
Seed:                    $SEED
Positions dir:           $POSITIONS_DIR
Binary:                  $BINARY
EOF

# ---- プロセス起動 ----
PIDS=()
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    START=$((i * POSITIONS_PER_PROC))
    if [ $START -ge $N_STATES ]; then
        continue
    fi
    if [ $((START + POSITIONS_PER_PROC)) -gt $N_STATES ]; then
        MAX=$((N_STATES - START))
    else
        MAX=$POSITIONS_PER_PROC
    fi

    LOG_FILE="$LOG_DIR/proc_${i}_idx${START}.log"
    printf "  [%2d] start=%4d max=%3d -> %s\n" "$i" "$START" "$MAX" "$LOG_FILE"

    # nohup + stdin切断で SSH断後も生存
    nohup "$BINARY" \
        --depth3-mcts \
        --states "$N_STATES" \
        --proposed-playouts "$PROPOSED_PLAYOUTS" \
        --allgrid-playouts "$ALLGRID_PLAYOUTS" \
        --proposed-rollouts "$PROPOSED_ROLLOUTS" \
        --allgrid-rollouts "$ALLGRID_ROLLOUTS" \
        --retention "$RETENTION" \
        --threads "$THREADS_PER_PROC" \
        --seed "$SEED" \
        --start-index "$START" \
        --max-positions "$MAX" \
        --load-positions "$POSITIONS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        < /dev/null > "$LOG_FILE" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "Launched ${#PIDS[@]} processes. PIDs: ${PIDS[*]}"
echo "Kill all with: pkill -f ylab_client"
echo "Monitor with : tail -f $LOG_DIR/proc_0_idx0.log"
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
if [ $SKIP_COMBINE -eq 0 ]; then
    echo ""
    echo "Combining CSV outputs..."

    OUT_CSV="$RUN_DIR/depth3_results_COMBINED.csv"
    set +u
    PARTS=("$OUTPUT_DIR"/depth3_results_idx*.csv)
    set -u

    if [ -f "${PARTS[0]:-}" ]; then
        head -1 "${PARTS[0]}" > "$OUT_CSV"
        for f in "${PARTS[@]}"; do
            [ -f "$f" ] && tail -n +2 "$f" >> "$OUT_CSV"
        done
        ROWS=$(( $(wc -l < "$OUT_CSV") - 1 ))
        echo "  Combined CSV: $OUT_CSV ($ROWS rows)"
    else
        echo "  No depth3_results_idx*.csv found."
    fi

    # 全体サマリ集計
    OUT_SUMMARY="$RUN_DIR/depth3_summary_COMBINED.txt"
    if [ -f "$OUT_CSV" ]; then
        python3 - <<PYEOF > "$OUT_SUMMARY" 2>/dev/null || true
import csv
path = "$OUT_CSV"
n = exact = cluster = same_type = 0
ssd = sptime = satime = 0.0
with open(path) as f:
    for row in csv.DictReader(f):
        try:
            if int(row['proposed_idx']) < 0 or int(row['allgrid_idx']) < 0:
                continue
        except (ValueError, KeyError):
            continue
        n += 1
        exact   += int(row['exact_match'])
        cluster += int(row['same_cluster'])
        same_type += int(row['same_type'])
        ssd     += float(row['score_diff'])
        sptime  += float(row['proposed_time_sec'])
        satime  += float(row['allgrid_time_sec'])
print("Depth-3 MCTS Combined Summary")
print("========================================")
print(f"valid_cases         = {n}")
if n > 0:
    print(f"exact_match_pct     = {100.0*exact/n:.2f}")
    print(f"same_cluster_pct    = {100.0*cluster/n:.2f}")
    print(f"same_type_pct       = {100.0*same_type/n:.2f}")
    print(f"avg_score_diff      = {ssd/n:.4f}")
    print(f"avg_proposed_time_s = {sptime/n:.2f}")
    print(f"avg_allgrid_time_s  = {satime/n:.2f}")
PYEOF
        if [ -s "$OUT_SUMMARY" ]; then
            echo "  Combined summary: $OUT_SUMMARY"
            cat "$OUT_SUMMARY"
        fi
    fi
fi

echo ""
echo "All outputs in: $RUN_DIR"
echo "To transfer to local: scp -r user@server:$PROJECT_ROOT/$RUN_DIR ./"
echo ""
echo "Done."
