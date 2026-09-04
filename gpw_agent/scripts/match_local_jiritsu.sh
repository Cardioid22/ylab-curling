#!/bin/bash
# Local match: gpw_agent vs Jiritsukun-Jr on one PC.
#   match_local_jiritsu.sh <logdir> [gpw_threads] [server_config] [gpw_side: 0|1]
# Env: GPW_BIN (default ./build/Release/gpw_agent.exe), GPW_EXTRA (extra agent args),
#      JIR_THREADS (temporarily sets Jiritsukun's thread_num, restored afterwards)
set -u
LOG=${1:-/tmp/gpw_match}; THREADS=${2:-6}; CFG=${3:-config.json}; SIDE=${4:-0}
ROOT=/c/Users/atomu/GitHub/ylab-curling
JIR=/c/Users/atomu/GitHub/Jiritsukun-Jr_GAT2025/jiritsu
if [ "$SIDE" = "0" ]; then GPW_PORT=10000; JIR_PORT=10001; else GPW_PORT=10001; JIR_PORT=10000; fi
mkdir -p "$LOG"
LOG=$(cd "$LOG" && pwd)   # absolute: the redirects below happen after cd
if [ -n "${JIR_THREADS:-}" ]; then
  cp "$JIR/config.json" "$JIR/config.json.bak"
  sed -i "s/\"thread_num\": *[0-9]*/\"thread_num\": $JIR_THREADS/" "$JIR/config.json"
fi
cd "$ROOT/digitalcurling3_server/bin" && ./digitalcurling3_server.exe "$CFG" > "$LOG/server.log" 2>&1 &
SRV=$!
sleep 3
cd "$JIR" && ./jiritsu_server.exe 7000 > "$LOG/jiritsu_server.log" 2>&1 &
JS=$!
sleep 2
cd "$JIR" && python play_local.py --sim-port 7000 --server-port $JIR_PORT > "$LOG/jiritsu.log" 2>&1 &
JP=$!
sleep 2
cd "$ROOT/gpw_agent" && ${GPW_BIN:-./build/Release/gpw_agent.exe} localhost $GPW_PORT --threads "$THREADS" --name gpw_agent --log "$LOG/gpw_shots.log" ${GPW_EXTRA:-} > "$LOG/gpw.log" 2>&1
echo "gpw_agent exited: $?"
sleep 5
kill $JP $JS $SRV 2>/dev/null
cmd //c "taskkill /F /IM jiritsu_server.exe /T" >/dev/null 2>&1
cmd //c "taskkill /F /IM digitalcurling3_server.exe /T" >/dev/null 2>&1
if [ -n "${JIR_THREADS:-}" ] && [ -f "$JIR/config.json.bak" ]; then mv "$JIR/config.json.bak" "$JIR/config.json"; fi
grep "game_over" "$LOG/gpw.log"
echo done
