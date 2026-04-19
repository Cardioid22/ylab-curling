#!/bin/bash
################################################################################
# Parallel Clustering Effectiveness Experiment Runner
#
# 10,000局面クラスタリング効果実験を並列実行する。
# nohup でSSH切断後も継続。
#
# Usage:
#   ./scripts/run_parallel_clustering_experiment.sh [OPTIONS]
#
# Options:
#   --parallel P          並列プロセス数 (default: 48)
#   --total N             全局面数 (default: 10000)
#   --positions-dir PATH  テスト局面ディレクトリ
#                         (default: clustered_ayumu/test_positions_20260417_055725)
#   --binary PATH         ylab_client バイナリパス (default: 自動検出)
#   --policy-param PATH   policy_param.dat パス (default: data/policy_param.dat)
#   --rollout R           ロールアウト回数 (default: 1)
#   --retention PCT       保持率 % (default: 20)
#   --skip-combine        最後の CSV 結合をスキップ
#
# Examples:
#   # 標準実行 (48並列)
#   ./scripts/run_parallel_clustering_experiment.sh
#
#   # 24並列、2000局面だけ
#   ./scripts/run_parallel_clustering_experiment.sh --parallel 24 --total 2000
#
#   # パス上書き
#   ./scripts/run_parallel_clustering_experiment.sh \
#       --positions-dir /path/to/positions \
#       --binary /path/to/ylab_client
#
# SSH切断対策（推奨）:
#   tmux new -s exp
#   ./scripts/run_parallel_clustering_experiment.sh 48
#   (Ctrl-B, D で detach → 後で tmux attach -t exp)
################################################################################

set -euo pipefail

# SSH切断時のSIGHUP無視
trap '' HUP

# ---- デフォルト設定 ----
NUM_PROCESSES=48
TOTAL_POSITIONS=10000
POSITIONS_DIR="clustered_ayumu/test_positions_20260417_055725"
BINARY=""
POLICY_PARAM="data/policy_param.dat"
ROLLOUT=1
RETENTION=20
SKIP_COMBINE=0

# ---- 引数パース ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel) NUM_PROCESSES="$2"; shift 2 ;;
        --total)    TOTAL_POSITIONS="$2"; shift 2 ;;
        --positions-dir) POSITIONS_DIR="$2"; shift 2 ;;
        --binary)   BINARY="$2"; shift 2 ;;
        --policy-param) POLICY_PARAM="$2"; shift 2 ;;
        --rollout)  ROLLOUT="$2"; shift 2 ;;
        --retention) RETENTION="$2"; shift 2 ;;
        --skip-combine) SKIP_COMBINE=1; shift ;;
        [0-9]*)     NUM_PROCESSES="$1"; shift ;;   # 後方互換: 第1引数でプロセス数指定
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--parallel N] [--total N] [--positions-dir PATH] ..."
            exit 1 ;;
    esac
done

# プロジェクトルートへ移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- バイナリ自動検出 (--binary 未指定時) ----
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

if [ ! -f "$POLICY_PARAM" ]; then
    echo "Error: policy param file not found: $POLICY_PARAM"
    echo "Specify with: --policy-param /path/to/policy_param.dat"
    exit 1
fi

# 1プロセスあたりの局面数（切り上げ）
POSITIONS_PER_PROC=$(( (TOTAL_POSITIONS + NUM_PROCESSES - 1) / NUM_PROCESSES ))

# この実行専用のディレクトリ (全出力・全ログ・結合CSVを1箇所に集約)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="experiment_results/run_${TIMESTAMP}_parallel"
LOG_DIR="$RUN_DIR/logs"
OUTPUT_DIR="$RUN_DIR"
mkdir -p "$LOG_DIR"

# ---- 設定表示 ----
echo "========================================"
echo "Parallel Clustering Experiment"
echo "========================================"
echo "  Binary:             $BINARY"
echo "  Positions dir:      $POSITIONS_DIR"
echo "  Policy param:       $POLICY_PARAM"
echo "  Total positions:    $TOTAL_POSITIONS"
echo "  Num processes:      $NUM_PROCESSES"
echo "  Positions/process:  $POSITIONS_PER_PROC"
echo "  Rollout count:      $ROLLOUT"
echo "  Retention:          $RETENTION%"
echo "  Deterministic:      ON"
echo "  Run dir:            $RUN_DIR"
echo "  Log dir:            $LOG_DIR"
echo "  Output CSVs -> $OUTPUT_DIR"
echo "========================================"
echo ""

# 実行条件をメモ
cat > "$RUN_DIR/run_config.txt" <<EOF
Timestamp:         $TIMESTAMP
Num processes:     $NUM_PROCESSES
Total positions:   $TOTAL_POSITIONS
Positions/proc:    $POSITIONS_PER_PROC
Positions dir:     $POSITIONS_DIR
Rollout count:     $ROLLOUT
Retention:         $RETENTION%
Deterministic:     ON
Binary:            $BINARY
Policy param:      $POLICY_PARAM
EOF

# ---- プロセス起動 ----
PIDS=()
for i in $(seq 0 $((NUM_PROCESSES - 1))); do
    START=$((i * POSITIONS_PER_PROC))
    if [ $START -ge $TOTAL_POSITIONS ]; then
        continue
    fi
    if [ $((START + POSITIONS_PER_PROC)) -gt $TOTAL_POSITIONS ]; then
        MAX=$((TOTAL_POSITIONS - START))
    else
        MAX=$POSITIONS_PER_PROC
    fi

    LOG_FILE="$LOG_DIR/proc_${i}_idx${START}.log"
    printf "  [%2d] start=%5d max=%4d -> %s\n" "$i" "$START" "$MAX" "$LOG_FILE"

    # nohup + stdin切断で SSH断後も生存
    nohup "$BINARY" \
        --clustering-experiment \
        --rollout "$ROLLOUT" \
        --retention "$RETENTION" \
        --load-positions "$POSITIONS_DIR" \
        --max-positions "$MAX" \
        --start-index "$START" \
        --output-dir "$OUTPUT_DIR" \
        --deterministic \
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

    # clustering_effectiveness (no-idx: 先頭プロセス + _idx*: それ以降) を全部統合
    combine_csvs() {
        local prefix="$1"
        local out="$RUN_DIR/${prefix}_ret${RETENTION}_B${ROLLOUT}_COMBINED.csv"
        set +u
        local no_idx=("$OUTPUT_DIR"/${prefix}_ret${RETENTION}_B${ROLLOUT}_2[0-9]*.csv)
        local with_idx=("$OUTPUT_DIR"/${prefix}_ret${RETENTION}_B${ROLLOUT}_idx*.csv)
        set -u

        local first_file=""
        for f in "${no_idx[@]}" "${with_idx[@]}"; do
            if [ -f "$f" ]; then first_file="$f"; break; fi
        done

        if [ -z "$first_file" ]; then
            echo "  [$prefix] No output CSVs found."
            return
        fi

        head -1 "$first_file" > "$out"
        for f in "${no_idx[@]}"; do
            [ -f "$f" ] && tail -n +2 "$f" >> "$out"
        done
        for f in "${with_idx[@]}"; do
            [ -f "$f" ] && tail -n +2 "$f" >> "$out"
        done
        local rows=$(( $(wc -l < "$out") - 1 ))
        echo "  [$prefix] $out ($rows rows)"
    }

    combine_csvs "clustering_effectiveness"
    combine_csvs "cluster_details"
fi

echo ""
echo "All outputs in: $RUN_DIR"
echo "To transfer to local:   scp -r user@server:$PROJECT_ROOT/$RUN_DIR ./"
echo ""
echo "Done."
