#!/bin/bash
# DocRes server load test
# Usage: ./loadtest.sh [threads] [connections] [duration]

THREADS=${1:-4}
CONNS=${2:-50}
DURATION=${3:-60s}
SERVER="https://docres.s2.aezakmigroup.ru"
IMAGES="../DocRes/input"
SCRIPT="loadtest.lua"

echo "=== DocRes Load Test ==="
echo "Threads: $THREADS, Connections: $CONNS, Duration: $DURATION"
echo ""

for ENDPOINT in /enhance/quality /full /deblur; do
    echo "--- $ENDPOINT ---"
    wrk -t$THREADS -c$CONNS -d$DURATION \
        -s $SCRIPT \
        --timeout 120s \
        "$SERVER$ENDPOINT" \
        -- "$IMAGES"
    echo ""
done
