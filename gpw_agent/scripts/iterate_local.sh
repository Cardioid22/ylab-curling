#!/bin/bash
# Unattended expert-iteration loop on the local Windows machine (Git Bash).
#   bash scripts/iterate_local.sh <first_gen> <n_gens> [games_per_gen] [ab_games]
# State: models/best.txt + models/best_eval.txt (start from data/eval_local_v2.txt if absent),
#        data/gen<id>.jsonl, models/history.txt
set -u
FIRST=${1:?first_gen}; NGEN=${2:?n_gens}; GAMES=${3:-60}; AB_GAMES=${4:-40}
THREADS=${THREADS:-8}; BUDGET=${BUDGET:-0.5}; EXPLORE=${EXPLORE:-0.15}
BIN=${GPW_BIN:-./build_dev/Release/gpw_agent.exe}
PY=${PY:-python}
cd "$(dirname "$0")/.."
mkdir -p models data
if [ ! -f models/best_eval.txt ]; then
  cp data/model_local_v2.txt models/best.txt
  sed 's#^model .*#model models/best.txt#' data/eval_local_v2.txt > models/best_eval.txt
  echo "seeded best from v2" | tee -a models/history.txt
fi
for (( g=FIRST; g<FIRST+NGEN; g++ )); do
  echo "[gen $g] self-play $GAMES games with best  ($(date))" | tee -a models/history.txt
  $BIN --selfplay --games $GAMES --ends 10 --threads $THREADS --budget-a $BUDGET --budget-b $BUDGET \
      --explore $EXPLORE --seed $(( 1000 + g * 17 )) --quiet \
      --eval-a models/best_eval.txt --eval-b models/best_eval.txt \
      --out data/gen$g.jsonl > data/gen${g}_log.txt 2>&1
  echo "[gen $g] train on all data  ($(date))" | tee -a models/history.txt
  $PY python/train_value.py data/sp_local_gen0_all_h.jsonl data/sp_local_gen1_a.jsonl "data/gen*.jsonl" \
      --out models/gen$g.txt --eval-out models/gen${g}_eval.txt --epochs ${EPOCHS:-30} --threads 6 \
      > models/gen${g}_train.txt 2>&1
  grep -E "loaded|hand-crafted|exported" models/gen${g}_train.txt | tee -a models/history.txt
  echo "[gen $g] A/B gen$g vs best, $AB_GAMES games x 2 ends  ($(date))" | tee -a models/history.txt
  $BIN --selfplay --games $AB_GAMES --ends 2 --threads $THREADS --budget-a $BUDGET --budget-b $BUDGET \
      --eval-a models/gen${g}_eval.txt --eval-b models/best_eval.txt --seed $(( 5000 + g )) --quiet \
      > models/gen${g}_ab.txt 2>&1
  R=$(grep "^RESULT" models/gen${g}_ab.txt)
  A=$(echo "$R" | sed 's/.*A=\([0-9]*\).*/\1/'); B=$(echo "$R" | sed 's/.*B=\([0-9]*\).*/\1/')
  echo "[gen $g] $R" | tee -a models/history.txt
  if [ "${A:-0}" -gt "${B:-0}" ]; then
    cp models/gen$g.txt models/best.txt
    sed 's#^model .*#model models/best.txt#' models/gen${g}_eval.txt > models/best_eval.txt
    echo "[gen $g] promoted" | tee -a models/history.txt
  else
    echo "[gen $g] kept previous best" | tee -a models/history.txt
  fi
done
