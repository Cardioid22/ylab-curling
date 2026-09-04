#!/bin/bash
# One expert-iteration generation on a lab server:
#   1. self-play data with the current best model (or hand-crafted eval for gen 0)
#   2. train a new model on all data so far
#   3. A/B the new model against the current best; promote it if it wins
#
#   bash scripts/iterate.sh <gen_id> [n_procs] [games_per_proc] [threads_per_proc] [budget] [explore]
#
# State: models/best.txt + models/best_eval.txt (absent for gen 0), data/gen<id>/*.jsonl
# Requires: ./build/gpw_agent (Linux build) and a python with torch+numpy at $PY (default ~/venv-gpw/bin/python).
set -eu
GEN=${1:?gen_id}; NPROC=${2:-16}; GAMES=${3:-40}; THREADS=${4:-6}; BUDGET=${5:-0.6}; EXPLORE=${6:-0.15}
AB_GAMES=${AB_GAMES:-100}; AB_ENDS=${AB_ENDS:-4}; AB_BUDGET=${AB_BUDGET:-0.6}; AB_THREADS=${AB_THREADS:-12}; AB_PAR=${AB_PAR:-6}
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venv-gpw/bin/python}
mkdir -p models data/gen$GEN
BIN=./build/gpw_agent

MODEL_ARGS=""
if [ -f models/best.txt ]; then
  MODEL_ARGS="--model-a models/best.txt --model-b models/best.txt --eval-a models/best_eval.txt --eval-b models/best_eval.txt"
fi

echo "[gen $GEN] 1/3 self-play: $NPROC x $GAMES games, threads=$THREADS budget=$BUDGET explore=$EXPLORE  ($(date))"
for i in $(seq 0 $((NPROC - 1))); do
  SEED=$(( GEN * 100000 + i * 7 + 1 ))
  $BIN --selfplay --games $GAMES --ends 10 --threads $THREADS --budget-a $BUDGET --budget-b $BUDGET \
      --explore $EXPLORE --seed $SEED --quiet $MODEL_ARGS \
      --out data/gen$GEN/sp_$i.jsonl > data/gen$GEN/log_$i.txt 2>&1 &
done
wait
$PY python/inspect_data.py "data/gen$GEN/*.jsonl" | tee data/gen$GEN/inspect.txt

echo "[gen $GEN] 2/3 train  ($(date))"
$PY python/train_value.py "data/gen*/*.jsonl" --out models/gen$GEN.txt --eval-out models/gen${GEN}_eval.txt \
    --epochs ${EPOCHS:-25} --threads ${TRAIN_THREADS:-32} | tee models/gen${GEN}_train.txt

echo "[gen $GEN] 3/3 A/B: gen$GEN vs best, $AB_GAMES games x $AB_ENDS ends  ($(date))"
if [ -f models/best.txt ]; then
  B_ARGS="--model-b models/best.txt --eval-b models/best_eval.txt"
else
  B_ARGS=""
fi
PER=$(( AB_GAMES / AB_PAR ))
for j in $(seq 0 $((AB_PAR - 1))); do
  $BIN --selfplay --games $PER --ends $AB_ENDS --threads $AB_THREADS --budget-a $AB_BUDGET --budget-b $AB_BUDGET \
      --model-a models/gen$GEN.txt --eval-a models/gen${GEN}_eval.txt $B_ARGS --seed $(( 900000 + GEN * 1000 + j )) --quiet \
      > models/gen${GEN}_ab_$j.txt 2>&1 &
done
wait
A=$(grep -h "^RESULT" models/gen${GEN}_ab_*.txt | sed 's/.*A=\([0-9]*\).*/\1/' | paste -sd+ | bc)
B=$(grep -h "^RESULT" models/gen${GEN}_ab_*.txt | sed 's/.*B=\([0-9]*\).*/\1/' | paste -sd+ | bc)
echo "[gen $GEN] A/B result: new $A - $B best" | tee -a models/history.txt
if [ "$A" -gt "$B" ]; then
  cp models/gen$GEN.txt models/best.txt
  cp models/gen${GEN}_eval.txt models/best_eval.txt
  sed -i "s#^model .*#model models/best.txt#" models/best_eval.txt
  echo "[gen $GEN] promoted gen$GEN to best" | tee -a models/history.txt
else
  echo "[gen $GEN] kept previous best" | tee -a models/history.txt
fi
