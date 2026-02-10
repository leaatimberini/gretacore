#!/bin/bash
set -euo pipefail

DATE="${1:-$(date +%F)}"
RUN_ROOT="artifacts_remote/$DATE/b3_87"
REMOTE_BASE="/root/gretacore"
cd "$REMOTE_BASE"

# Fixed params for decomposition
CTX=8192
GEN=128
DTYPE="bf16"

run_config() {
    local batch="$1"
    local deterministic="$2"
    
    local REL_PATH="runs/batch_${batch}/det_${deterministic}"
    local TARGET_OUT="$RUN_ROOT/$REL_PATH"
    mkdir -p "$TARGET_OUT"

    echo "[B3.87] Batch=$batch, Determinism=$deterministic..."
    
    # Environment
    export GRETA_VERBOSE_INFO=1
    if [ "$deterministic" == "on" ]; then
        export HIP_LAUNCH_BLOCKING=1
        export AMD_SERIALIZE_KERNEL=3
    else
        unset HIP_LAUNCH_BLOCKING
        unset AMD_SERIALIZE_KERNEL
    fi

    local START_TIME=$(date +%s.%N)
    ./tools/inference/build/greta_infer \
        --model ./models/greta-v1.gguf \
        --prompt "Hello" \
        --max-tokens $GEN \
        --batch-size $batch \
        --greedy > "$TARGET_OUT/run.log" 2>&1
    local END_TIME=$(date +%s.%N)
    local WALL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    local TPS=$(grep "Tokens/second:" "$TARGET_OUT/run.log" | awk '{print $2}' || echo "0")
    local TIMINGS=$(grep "\[PERF_TIMING\]" "$TARGET_OUT/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")

    cat > "$TARGET_OUT/perf.json" << EOF
{
  "ticket": "b3_87",
  "batch": $batch,
  "deterministic": "$deterministic",
  "tokens_per_sec": $TPS,
  "wall_time_sec": $WALL_TIME,
  "timings": $TIMINGS,
  "env": {
    "HIP_LAUNCH_BLOCKING": "${HIP_LAUNCH_BLOCKING:-0}",
    "AMD_SERIALIZE_KERNEL": "${AMD_SERIALIZE_KERNEL:-0}"
  }
}
EOF
}

# Matrix
run_config 1 "off"
run_config 1 "on"
run_config 8 "off"
run_config 8 "on"

echo "DONE_REMOTE_B3_87"
