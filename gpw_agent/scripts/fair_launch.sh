#!/bin/bash
cd /c/Users/atomu/GitHub/ylab-curling/gpw_agent || exit 1
mkdir -p data/fair1
exec bash scripts/fair_matches.sh data/fair1 data/eval_local_v4.txt > data/fair1/stdout.txt 2>&1
