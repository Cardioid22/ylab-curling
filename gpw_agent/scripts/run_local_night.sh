#!/bin/bash
# Detached local driver: finish the v4-vs-v2 comparison, seed models/best from the
# winner, then run the unattended generation loop.
#   bash scripts/run_local_night.sh <n_gens>
set -u
NGEN=${1:-3}
cd "$(dirname "$0")/.."
BIN=./build_dev/Release/gpw_agent.exe
mkdir -p models data
LOG=models/night_log.txt
echo "=== night run start $(date) ===" >> $LOG

# 1. v4 vs v2, 40 games x 2 ends
$BIN --selfplay --games 40 --ends 2 --budget-a 0.5 --budget-b 0.5 --threads 8 \
    --eval-a data/eval_local_v4.txt --eval-b data/eval_local_v2.txt --seed 121 --quiet \
    > data/log_ab_v4_vs_v2_40.txt 2>&1
R=$(grep "^RESULT" data/log_ab_v4_vs_v2_40.txt)
echo "v4 vs v2: $R" | tee -a $LOG
A=$(echo "$R" | sed 's/.*A=\([0-9]*\).*/\1/'); B=$(echo "$R" | sed 's/.*B=\([0-9]*\).*/\1/')
if [ "${A:-0}" -gt "${B:-0}" ]; then
  cp data/model_local_v4.txt models/best.txt
  sed 's#^model .*#model models/best.txt#' data/eval_local_v4.txt > models/best_eval.txt
  echo "best = v4" | tee -a $LOG
else
  cp data/model_local_v2.txt models/best.txt
  sed 's#^model .*#model models/best.txt#' data/eval_local_v2.txt > models/best_eval.txt
  echo "best = v2" | tee -a $LOG
fi

# 2. generation loop
bash scripts/iterate_local.sh 2 $NGEN 60 40 >> $LOG 2>&1
echo "=== night run end $(date) ===" >> $LOG
