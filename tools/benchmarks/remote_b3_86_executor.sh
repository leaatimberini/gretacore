#!/bin/bash
set -euo pipefail

DATE="${1:-$(date +%F)}"
RUN_ROOT="artifacts_remote/$DATE/b3_86"
REMOTE_BASE="/root/gretacore"
cd "$REMOTE_BASE"

run_config() {
    local ctx="$1"
    local impl="$2"
    local variant_name="${3:-default}"
    
    local REL_PATH="runs/ctx_${ctx}/impl_${impl}/${variant_name}"
    local TARGET_OUT="$RUN_ROOT/$REL_PATH"
    mkdir -p "$TARGET_OUT"

    echo "[B3.86] CTX=$ctx, IMPL=$impl ($variant_name)..."
    python3 -c "print('a' * ($ctx - 1))" > /tmp/prompt.txt

    # Setup environment for the specific impl if needed
    # (Currently we only have one, but we can fake/test flags)
    export GRETA_VERBOSE_INFO=1
    export GRETA_MAX_SEQ_LEN=65536
    
    local VRAM_LOG="$TARGET_OUT/vram_samples.csv"
    echo "timestamp,vram_mb" > "$VRAM_LOG"
    (
        while true; do
            VRAM=$(rocm-smi --showmeminfo vram | grep "VRAM Total Used" | head -n 1 | awk '{print $NF}' | sed 's/B//' || echo "0")
            echo "$(date +%s),$VRAM" >> "$VRAM_LOG"
            sleep 1
        done
    ) &
    VRAM_PID=$!

    local START_TIME=$(date +%s.%N)
    ./tools/inference/build/greta_infer \
        --model ./models/greta-v1.gguf \
        --prompt-file /tmp/prompt.txt \
        --max-tokens 1 \
        --greedy > "$TARGET_OUT/run.log" 2>&1
    local END_TIME=$(date +%s.%N)
    local WALL_TIME=$(echo "$END_TIME - $START_TIME" | bc)
    kill $VRAM_PID || true

    local TIMINGS=$(grep "\[PERF_TIMING\]" "$TARGET_OUT/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")
    local PEAK_VRAM=$(sort -t, -k2 -rn "$VRAM_LOG" | head -n 2 | tail -n 1 | cut -d, -f2 || echo "0")

    cat > "$TARGET_OUT/perf.json" << EOF
{
  "ticket": "b3_86",
  "context_len": $ctx,
  "attn_impl_request": "$impl",
  "wall_time_sec": $WALL_TIME,
  "peak_vram_mb": $PEAK_VRAM,
  "timings": $TIMINGS
}
EOF
}

# Matrix
run_config 8192 "auto"
run_config 16384 "auto"

echo "DONE_REMOTE_B3_86"
