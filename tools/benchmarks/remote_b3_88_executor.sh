#!/bin/bash
set -euo pipefail

DATE="${1:-$(date +%F)}"
RUN_ROOT="artifacts_remote/$DATE/b3_88"
REMOTE_BASE="/root/gretacore"
cd "$REMOTE_BASE"

# 32k mission
CTX=32768
TIMEOUT=3600 # 1 hour

REL_PATH="runs/ctx_32768/final"
TARGET_OUT="$RUN_ROOT/$REL_PATH"
mkdir -p "$TARGET_OUT"

echo "[B3.88] Attempting 32k prefill (Timeout 1h)..."
python3 -c "print('a' * 32767)" > /tmp/prompt.txt

export GRETA_VERBOSE_INFO=1
export GRETA_MAX_SEQ_LEN=65536
# Apply best settings known (if any)
# export GRETA_USE_FAST_PATH=1 

local START_TIME=$(date +%s.%N)
set +e
timeout --foreground "$TIMEOUT" ./tools/inference/build/greta_infer \
    --model ./models/greta-v1.gguf \
    --prompt-file /tmp/prompt.txt \
    --max-tokens 1 \
    --greedy > "$TARGET_OUT/run.log" 2>&1
EXIT_STATUS=$?
set -e
local END_TIME=$(date +%s.%N)
local WALL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

STATUS_STR="OK"
if [ $EXIT_STATUS -eq 124 ]; then
    STATUS_STR="FAIL_TIMEOUT"
    echo "[B3.88] 32k TIMED OUT"
elif [ $EXIT_STATUS -ne 0 ]; then
    STATUS_STR="FAIL_CRASH"
    echo "[B3.88] 32k CRASHED"
fi

TIMINGS=$(grep "\[PERF_TIMING\]" "$TARGET_OUT/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")

cat > "$TARGET_OUT/perf.json" << EOF
{
  "ticket": "b3_88",
  "context_len": $CTX,
  "wall_time_sec": $WALL_TIME,
  "exit_status": "$STATUS_STR",
  "timings": $TIMINGS
}
EOF

echo "DONE_REMOTE_B3_88"
