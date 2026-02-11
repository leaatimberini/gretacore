#!/bin/bash
set -euo pipefail

DATE="${1:-$(date +%F)}"
RUN_ROOT="artifacts_remote/$DATE/b3_85"
REMOTE_BASE="/root/gretacore"
cd "$REMOTE_BASE"

# Constants
KV_ALIGNED=1
SEED=0
DTYPE="bf16"
GEN_LEN=1

run_config() {
    local ctx="$1"
    local timeout="$2"
    
    local REL_PATH="runs/ctx_${ctx}/kv_1/seed_0"
    local TARGET_OUT="$RUN_ROOT/$REL_PATH"
    mkdir -p "$TARGET_OUT"

    echo "[B3.85] Running CTX=$ctx, Timeout=${timeout}s..."
    
    # Generate a prompt. In ASCII fallback mode (char-wise), we want exactly ctx-1 chars.
    local PROMPT_FILE="/tmp/prompt_${ctx}.txt"
    python3 -c "print('a' * ($ctx - 1))" > "$PROMPT_FILE"

    export GRETA_KV_ALIGNED=1
    export GRETA_SEED=0
    export GRETA_VERBOSE_INFO=1
    export GRETA_MAX_SEQ_LEN=40000
    
    # Start VRAM sampling
    local VRAM_LOG="$TARGET_OUT/vram_samples.csv"
    echo "timestamp,vram_mb" > "$VRAM_LOG"
    (
        while true; do
            VRAM=$(rocm-smi --showmeminfo vram | grep "VRAM Total Used" | head -n 1 | awk '{print $NF}' | sed 's/B//' || echo "0")
            # Convert to MB if needed, but rocm-smi often gives bytes or human readable. 
            # Our analyzer expects MB. 
            echo "$(date +%s),$VRAM" >> "$VRAM_LOG"
            sleep 1
        done
    ) &
    VRAM_PID=$!

    START_TIME=$(date +%s.%N)
    set +e
    timeout --foreground "$timeout" ./tools/inference/build/greta_infer \
        --model ./models/greta-v1.gguf \
        --prompt-file "$PROMPT_FILE" \
        --max-tokens $GEN_LEN \
        --batch-size 1 \
        --greedy \
        --dump-logits "$TARGET_OUT" \
        --dump-logits-span 0 > "$TARGET_OUT/run.log" 2>&1
    local EXIT_STATUS=$?
    set -e
    END_TIME=$(date +%s.%N)
    WALL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    kill $VRAM_PID || true

    # Extract TPS and timings
    local TPS=$(grep "Tokens/second:" "$TARGET_OUT/run.log" | awk '{print $2}' || echo "0")
    local PEAK_VRAM=$(sort -t, -k2 -rn "$VRAM_LOG" | head -n 2 | tail -n 1 | cut -d, -f2 || echo "0")
    
    # Extract [PERF_TIMING] JSON
    local TIMINGS=$(grep "\[PERF_TIMING\]" "$TARGET_OUT/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")

    local STATUS_STR="OK"
    if [ $EXIT_STATUS -eq 124 ]; then
        STATUS_STR="FAIL_TIMEOUT"
        echo "[B3.85] CTX=$ctx TIMED OUT"
    elif [ $EXIT_STATUS -ne 0 ]; then
        STATUS_STR="FAIL_CRASH"
        echo "[B3.85] CTX=$ctx CRASHED"
    fi

    cat > "$TARGET_OUT/perf.json" << EOF
{
  "ticket": "b3_85",
  "phase": "prefill_rca",
  "context_len": $ctx,
  "gen_len": $GEN_LEN,
  "dump_span": 0,
  "dtype": "$DTYPE",
  "kv_aligned": $KV_ALIGNED,
  "seed": $SEED,
  "batch": 1,
  "wall_time_sec": $WALL_TIME,
  "tokens_per_sec": $TPS,
  "peak_vram_mb": $PEAK_VRAM,
  "exit_status": "$STATUS_STR",
  "timings": $TIMINGS
}
EOF
}

# Matrix: ctx increase with increasing timeout
run_config 4096 600
run_config 8192 900
run_config 16384 1500
run_config 24576 2100
run_config 32768 2700

echo "DONE_REMOTE_B3_85"
