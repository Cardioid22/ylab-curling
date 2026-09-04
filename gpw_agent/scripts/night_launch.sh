#!/bin/bash
cd /c/Users/atomu/GitHub/ylab-curling/gpw_agent || exit 1
mkdir -p models
exec bash scripts/run_local_night.sh 3 > models/night_stdout.txt 2>&1
