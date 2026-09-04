#!/bin/bash
cd /c/Users/atomu/GitHub/ylab-curling/gpw_agent || exit 1
mkdir -p data/fair_night2
export GPW_BIN=./build/Release/gpw_agent.exe
exec bash scripts/fair_loop.sh data/fair_night2 20 data/eval_gen1.txt > data/fair_night2/stdout.txt 2>&1
