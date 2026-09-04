#!/bin/bash
# Two fair local matches (4 threads each side): gpw_agent as team0, then as team1.
#   bash scripts/fair_matches.sh <logroot> [eval_file]
set -u
ROOTLOG=${1:?logroot}; EVAL=${2:-data/eval_local_v4.txt}
cd "$(dirname "$0")/.."
for side in 0 1; do
  GPW_BIN=./build_dev/Release/gpw_agent.exe GPW_EXTRA="--eval $EVAL" JIR_THREADS=4 \
    bash scripts/match_local_jiritsu.sh "$ROOTLOG/side$side" 4 config.json $side 2>&1 | tail -2
done
