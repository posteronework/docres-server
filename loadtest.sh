#!/bin/bash
# DocRes load test: /full vs /full-dewarp (50/50, resolution=2048, upscale=false)
# Two runs: sustainable (-c10) then saturation (-c50)

SERVER="https://xtifagasquc4so-8000.proxy.runpod.net"
IMAGES="../DocRes/input"
SCRIPT="loadtest.lua"
DURATION=${1:-60s}

echo "=== DocRes Load Test (/full + /full-dewarp @ 2048, upscale=false) ==="
echo "Server: $SERVER"
echo "Duration per run: $DURATION"
echo ""

echo "########## RUN 1: SUSTAINABLE (-t4 -c10) ##########"
wrk -t4 -c10 -d$DURATION \
    -s $SCRIPT \
    --timeout 120s \
    "$SERVER" \
    -- "$IMAGES"
echo ""

echo "########## RUN 2: SATURATION (-t8 -c50) ##########"
wrk -t8 -c50 -d$DURATION \
    -s $SCRIPT \
    --timeout 120s \
    "$SERVER" \
    -- "$IMAGES"
echo ""
