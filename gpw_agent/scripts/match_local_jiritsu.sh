#!/bin/bash
# Local smoke-test match: gpw_agent (team0, port 10000) vs Jiritsukun-Jr (team1, port 10001).
# Usage: match_local_jiritsu.sh <logdir> [gpw_threads] [server_config]
set -u
LOG=${1:-/tmp/gpw_match}; THREADS=${2:-6}; CFG=${3:-config.json}
ROOT=/c/Users/atomu/GitHub/ylab-curling
JIR=/c/Users/atomu/GitHub/Jiritsukun-Jr_GAT2025/jiritsu
mkdir -p "$LOG"
cd "$ROOT/digitalcurling3_server/bin" && ./digitalcurling3_server.exe "$CFG" > "$LOG/server.log" 2>&1 &
SRV=$!
sleep 3
cd "$JIR" && ./jiritsu_server.exe 7000 > "$LOG/jiritsu_server.log" 2>&1 &
JS=$!
sleep 2
cd "$JIR" && python play_local.py --sim-port 7000 --server-port 10001 > "$LOG/jiritsu.log" 2>&1 &
JP=$!
sleep 2
cd "$ROOT/gpw_agent" && ${GPW_BIN:-./build/Release/gpw_agent.exe} localhost 10000 --threads "$THREADS" --name gpw_agent --log "$LOG/gpw_shots.log" ${GPW_EXTRA:-} > "$LOG/gpw.log" 2>&1
echo "gpw_agent exited: $?"
sleep 5
kill $JP $JS $SRV 2>/dev/null
cmd //c "taskkill /F /IM jiritsu_server.exe /T" >/dev/null 2>&1
cmd //c "taskkill /F /IM digitalcurling3_server.exe /T" >/dev/null 2>&1
echo done
