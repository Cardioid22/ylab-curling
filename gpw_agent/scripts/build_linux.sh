#!/bin/bash
# Build gpw_agent on Linux (lab servers: run inside the ylab-project docker image on lion).
#   cd ~/ylab-curling/gpw_agent && bash scripts/build_linux.sh
# From the host on lion:
#   docker run --rm -v "$HOME/ylab-curling:/app" ylab-project bash -c 'cd /app/gpw_agent && bash scripts/build_linux.sh'
set -eu
cd "$(dirname "$0")/.."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j "${JOBS:-16}"
ls -la gpw_agent
