#!/bin/bash
################################################################################
# 計算再投資実験ランナー (GPW2026) — アーム単位の分散実行
#
# 1 アーム = (method, depth, playouts P, rollouts_per_visit R) の独立構成。
# すべてのアームを同一の総物理シミュ予算 B で走らせ、共通審判で選んだ手を採点する。
# (理論背景: experiments/REINVESTMENT_EXPERIMENT_GUIDE.md)
#
# 分散方針: マシンごとに担当アームを --arms で指定して起動する。
#   例) 強いマシン(bear)で重い A3(深さ5) を、他マシンで A1/A2/A4/A5 を担当。
#   各アームは --num-seeds 個の seed を別プロセスで並列実行する。
#
# 等予算 B の校正:
#   予算は実シミュ回数で揃える (深さ・ロールアウト数では揃わない。ガイド §2)。
#   下の P_*/R_* を smoke 実測 (actual_total_sims) を見ながら調整し、
#   全アームの平均 actual_total_sims を B±数% に合わせること。
#   校正の確認は scripts/aggregate_reinvest.py が自動で行う。
#
# Usage:
#   ./scripts/run_reinvest.sh --arms "A1,A2,A4,A5" [OPTIONS]   # 普通のマシン
#   ./scripts/run_reinvest.sh --arms "A3"          [OPTIONS]   # bear (深さ5担当)
#
# Options:
#   --arms LIST              実行するアーム (カンマ区切り; A1..A6) [必須]
#   --base-seed S            先頭 seed; S..S+K-1 を使う (default: 42)
#   --num-seeds K            seed 数 (default: 5)
#   --positions-dir PATH     N 局面のディレクトリ (default: test_positions_isobudget10)
#   --n-states N             テスト局面数 (= positions-dir の行数) (default: 10)
#   --threads-per-seed T     1 プロセスのスレッド数 (default: N_STATES)
#   --max-parallel M         同時に走らせる (arm,seed) プロセス上限 (default: K)
#   --binary PATH            ylab_client パス (default: 自動検出)
#   --parent-dir PATH        出力親 (default: experiments/reinvest/run_YYYYMMDD_HHMMSS は date 不可のため固定名)
################################################################################

set -euo pipefail
trap '' HUP

# ============================================================================
# 予算 B 校正パラメータ (smoke 実測を見て調整する。ここが実験の肝)
#   P_BASE/R_BASE : A1/A2/A5 の基準配分
#   P_DEEP        : A3(深さ5)/A6 — 深さに再投資 (P を減らして B を維持)
#   P_RINV/R_RINV : A4 — ロールアウトに再投資 (R を増やし P を減らして B を維持)
#   RETENTION     : Proposed/RandomK の保持率 (K = ceil(N*RETENTION))
# 既定値は「低予算帯」の暫定値。本番前に必ず校正すること。
# ============================================================================
P_BASE=400
R_BASE=20
P_DEEP=120
R_RINV=80
P_RINV=120
RETENTION=0.20

# アーム定義: "METHOD DEPTH PLAYOUTS ROLLOUTS RETENTION" を返す
arm_spec() {
    case "$1" in
        A1) echo "AllGrid  3 $P_BASE $R_BASE $RETENTION" ;;  # 基準 (削減なし)
        A2) echo "Proposed 3 $P_BASE $R_BASE $RETENTION" ;;  # クラスタリング効果単離 (A1と同配分)
        A3) echo "Proposed 5 $P_DEEP $R_BASE $RETENTION" ;;  # 深さ再投資
        A4) echo "Proposed 3 $P_RINV $R_RINV $RETENTION" ;;  # ロールアウト再投資
        A5) echo "RandomK  3 $P_BASE $R_BASE $RETENTION" ;;  # クラスタリング vs 単なる削減
        A6) echo "AllGrid  5 $P_DEEP $R_BASE $RETENTION" ;;  # (任意) 深さ5が予算内で破綻する実証
        *)  return 1 ;;
    esac
}

# ---- デフォルト ----
ARMS=""
BASE_SEED=42
NUM_SEEDS=5
POSITIONS_DIR="test_positions_isobudget10"
N_STATES=10
THREADS_PER_SEED=""
MAX_PARALLEL=""
BINARY=""
PARENT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --arms)             ARMS="$2"; shift 2 ;;
        --base-seed)        BASE_SEED="$2"; shift 2 ;;
        --num-seeds)        NUM_SEEDS="$2"; shift 2 ;;
        --positions-dir)    POSITIONS_DIR="$2"; shift 2 ;;
        --n-states)         N_STATES="$2"; shift 2 ;;
        --threads-per-seed) THREADS_PER_SEED="$2"; shift 2 ;;
        --max-parallel)     MAX_PARALLEL="$2"; shift 2 ;;
        --binary)           BINARY="$2"; shift 2 ;;
        --parent-dir)       PARENT_DIR="$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

if [ -z "$ARMS" ]; then
    echo "Error: --arms is required (e.g. --arms \"A1,A2,A4,A5\")" >&2
    exit 1
fi
[ -z "$THREADS_PER_SEED" ] && THREADS_PER_SEED="$N_STATES"
[ -z "$MAX_PARALLEL" ]     && MAX_PARALLEL="$NUM_SEEDS"

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

# ---- POSITIONS_DIR 健全性チェック (行数 == n_states; シャッフル回避) ----
if [ ! -d "$POSITIONS_DIR" ]; then
    echo "Error: POSITIONS_DIR not found: $POSITIONS_DIR" >&2
    exit 1
fi
total_rows=0
for f in "$POSITIONS_DIR"/batch_*.csv; do
    [ -f "$f" ] || continue
    total_rows=$(( total_rows + $(wc -l < "$f") - 1 ))
done
if [ "$total_rows" -ne "$N_STATES" ]; then
    echo "Error: POSITIONS_DIR has $total_rows rows but n_states=$N_STATES (need 行数==n_states)." >&2
    exit 1
fi

# ---- 親ディレクトリ ----
if [ -z "$PARENT_DIR" ]; then
    PARENT_DIR="experiments/reinvest/run_latest"
fi
mkdir -p "$PARENT_DIR"

# ---- アームの妥当性チェック ----
IFS=',' read -ra ARM_LIST <<< "$ARMS"
for ARM in "${ARM_LIST[@]}"; do
    if ! arm_spec "$ARM" >/dev/null 2>&1; then
        echo "Error: unknown arm '$ARM' (valid: A1..A6)" >&2
        exit 1
    fi
done

# ---- コア予算 ----
TOTAL_CORES=$(( MAX_PARALLEL * THREADS_PER_SEED ))
NPROC=$(nproc 2>/dev/null || echo 0)

cat > "$PARENT_DIR/reinvest_config.txt" <<EOF
Reinvestment run
========================================
Arms:               $ARMS
Base seed:          $BASE_SEED  (uses $BASE_SEED..$((BASE_SEED + NUM_SEEDS - 1)))
Num seeds:          $NUM_SEEDS
N states:           $N_STATES
Threads per seed:   $THREADS_PER_SEED
Max parallel:       $MAX_PARALLEL  (= $TOTAL_CORES cores; machine has $NPROC)
Positions dir:      $POSITIONS_DIR
Binary:             $BINARY
Parent dir:         $PARENT_DIR
Calibration:        P_BASE=$P_BASE R_BASE=$R_BASE P_DEEP=$P_DEEP P_RINV=$P_RINV R_RINV=$R_RINV RETENTION=$RETENTION
Arm specs:
$(for ARM in "${ARM_LIST[@]}"; do printf '  %s: %s\n' "$ARM" "$(arm_spec "$ARM")"; done)
EOF

echo "========================================"
cat "$PARENT_DIR/reinvest_config.txt"
echo "========================================"
if [ "$NPROC" -gt 0 ] && [ "$TOTAL_CORES" -gt "$NPROC" ]; then
    echo "  WARNING: oversubscription ($TOTAL_CORES > $NPROC). Lower --max-parallel or --threads-per-seed." >&2
fi
echo ""

running_jobs() { jobs -rp | wc -l; }

declare -a PIDS=()
declare -a PID_TAG=()

launch_job() {
    local arm="$1" seed="$2"
    read -r method depth playouts rollouts retention <<< "$(arm_spec "$arm")"
    local out_dir="$PARENT_DIR/$arm/seed_${seed}"
    mkdir -p "$out_dir"
    nohup "$BINARY" \
        --reinvest-arm \
        --method "$method" \
        --depth "$depth" \
        --playouts "$playouts" \
        --rollouts-per-visit "$rollouts" \
        --retention "$retention" \
        --arm-label "$arm" \
        --states "$N_STATES" \
        --threads "$THREADS_PER_SEED" \
        --seed "$seed" \
        --load-positions "$POSITIONS_DIR" \
        --output-dir "$out_dir" \
        < /dev/null > "$out_dir/run.log" 2>&1 &
    local pid=$!
    PIDS+=("$pid")
    PID_TAG+=("$arm/seed_$seed")
    echo "  launched $arm seed=$seed pid=$pid ($method d$depth P=$playouts R=$rollouts) -> $out_dir"
}

for ARM in "${ARM_LIST[@]}"; do
    for i in $(seq 0 $((NUM_SEEDS - 1))); do
        SEED=$((BASE_SEED + i))
        while [ "$(running_jobs)" -ge "$MAX_PARALLEL" ]; do
            wait -n 2>/dev/null || true
        done
        launch_job "$ARM" "$SEED"
    done
done

echo ""
echo "All jobs launched. monitor: tail -f $PARENT_DIR/*/seed_*/run.log"
echo ""

FAILED=0
for idx in "${!PIDS[@]}"; do
    pid="${PIDS[$idx]}"
    tag="${PID_TAG[$idx]}"
    if wait "$pid"; then
        echo "  $tag (pid=$pid) done"
    else
        echo "  $tag (pid=$pid) FAILED" >&2
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
if [ "$FAILED" -eq 0 ]; then
    echo "All jobs completed. Parent: $PARENT_DIR"
else
    echo "$FAILED jobs FAILED. Check $PARENT_DIR/*/seed_*/run.log" >&2
fi
echo ""
echo "Next: run the referee once (scripts/run_referee.sh), then aggregate:"
echo "  python3 scripts/aggregate_reinvest.py --reinvest-dir $PARENT_DIR --referee-dir experiments/reinvest_referee"
echo "========================================"
