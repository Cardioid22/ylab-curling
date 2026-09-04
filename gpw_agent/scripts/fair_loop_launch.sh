#!/bin/bash
cd /c/Users/atomu/GitHub/ylab-curling/gpw_agent || exit 1
mkdir -p data/fair_night
exec bash scripts/fair_loop.sh data/fair_night 20 data/eval_local_v4.txt > data/fair_night/stdout.txt 2>&1
