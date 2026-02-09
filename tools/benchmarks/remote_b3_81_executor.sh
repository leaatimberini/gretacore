#!/bin/bash
set -euo pipefail

# Inputs
BATCH_SIZES="${1:-1,2,4,8}"
GEN_LEN="${2:-64}"
DUMP_SPAN="${3:-8}"
RUN_ROOT="$4"

REMOTE_BASE="/root/gretacore"
cd "$REMOTE_BASE"

# Constants
CTX_LEN=8192
KV_ALIGNED=1
SEED=0
DTYPE="bf16"
TIMEOUT_SEC=600
SAMPLING_PERIOD=1

# Helper to run a single configuration
run_config() {
    local batch="$1"
    local REL_PATH="runs/batch_${batch}"
    local TARGET_OUT="$RUN_ROOT/$REL_PATH"
    mkdir -p "$TARGET_OUT"

    echo "--- [B3.81] batch=$batch ctx=$CTX_LEN kv=$KV_ALIGNED seed=$SEED ---"

    local PROMPT_FILE="$REMOTE_BASE/tools/benchmarks/prompts/synthetic_${CTX_LEN}.txt"
    if [ ! -f "$PROMPT_FILE" ]; then
        mkdir -p "$(dirname "$PROMPT_FILE")"
        python3 -c "print('hello ' * $CTX_LEN)" > "$PROMPT_FILE"
    fi

    local VRAM_FILE="$TARGET_OUT/vram_samples.csv"
    local START_SAMPLE_TS=$(date +%s)
    (
        echo "ts_epoch,used_vram_mb" > "$VRAM_FILE"
        while true; do
            if val=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c 'import sys, json; d=json.load(sys.stdin); print(list(d.values())[0]["VRAM Total Used Memory (B)"])' 2>/dev/null); then
                mb=$(echo "$val / 1048576" | bc)
                echo "$(date +%s),$mb" >> "$VRAM_FILE"
            fi
            sleep $SAMPLING_PERIOD
        done
    ) &
    local MON_PID=$!

    local EXIT_STATUS="OK"
    for MODE in "prefill" "decode"; do
        local MODE_OUT="$TARGET_OUT/$MODE"
        mkdir -p "$MODE_OUT"

        export HIP_LAUNCH_BLOCKING=1
        export AMD_SERIALIZE_KERNEL=3
        export HSA_ENABLE_SDMA=0
        export GRETA_DETERMINISTIC=1
        export GRETA_SEED=$SEED

        local START_TS=$(date +%s.%N)
        
        # We assume batch size support is checked by the binary
        timeout --foreground "${TIMEOUT_SEC}s" ./tools/inference/build/greta_infer \
            --model ./models/greta-v1.gguf \
            --prompt "$PROMPT_FILE" \
            --seed $SEED \
            --kv-aligned $KV_ALIGNED \
            --mode "$MODE" \
            --dump-logits "$MODE_OUT" \
            --dump-logits-span $DUMP_SPAN \
            --dtype "$DTYPE" \
            --max-tokens $GEN_LEN \
            --batch-size $batch \
            --greedy \
            2>&1 | tee "$MODE_OUT/run.log" || {
                local ERR=$?
                if [ $ERR -eq 124 ]; then EXIT_STATUS="FAIL_TIMEOUT"; else EXIT_STATUS="FAIL_ERROR"; fi
                if grep -qi 'out of memory\|OOM' "$MODE_OUT/run.log"; then EXIT_STATUS="FAIL_OOM"; fi
                if grep -qi 'unsupported batch' "$MODE_OUT/run.log"; then EXIT_STATUS="SKIPPED_UNSUPPORTED_BATCH"; fi
            }

        local END_TS=$(date +%s.%N)
        local WALL_TIME=$(echo "$END_TS - $START_TS" | bc)

        # Extract tokens/s if available from log, else calc it
        local TPS=0
        if [ "$MODE" == "decode" ] && [ "$EXIT_STATUS" == "OK" ]; then
             TPS=$(grep "Tokens/second:" "$MODE_OUT/run.log" | awk '{print $2}' || echo 0)
        fi

        cat > "$MODE_OUT/perf.json" << EOF
{
  "ticket": "b3_81",
  "phase": "$MODE",
  "context_len": $CTX_LEN,
  "gen_len": $GEN_LEN,
  "dump_span": $DUMP_SPAN,
  "dtype": "$DTYPE",
  "kv_aligned": $KV_ALIGNED,
  "seed": $SEED,
  "batch": $batch,
  "wall_time_sec": $WALL_TIME,
  "tokens_per_sec": $TPS,
  "exit_status": "$EXIT_STATUS"
}
EOF
        if [[ "$EXIT_STATUS" == FAIL* ]]; then break; fi
        if [ "$EXIT_STATUS" == "SKIPPED_UNSUPPORTED_BATCH" ]; then break; fi
    done

    kill $MON_PID || true
    
    # Finalize VRAM
    if [ -s "$VRAM_FILE" ]; then
        local PEAK_DATA=$(awk -F, 'BEGIN {max=0; ts=0} NR>1 {if ($2 > max) {max=$2; ts=$1}} END {print max "," ts}' "$VRAM_FILE")
        local PEAK=$(echo $PEAK_DATA | cut -d, -f1)
        local PEAK_TS=$(echo $PEAK_DATA | cut -d, -f2)
        local COUNT=$(awk 'END {print NR-1}' "$VRAM_FILE")
        local OFFSET=$((PEAK_TS - START_SAMPLE_TS))
        local DEV=$(rocm-smi --showproductname --json | python3 -c 'import sys, json; d=json.load(sys.stdin); print(list(d.values())[0]["Card series"])' 2>/dev/null || echo "AMD MI300X")
        cat > "$TARGET_OUT/vram.json" << EOF
{
  "peak_vram_mb": $PEAK,
  "samples_count": $COUNT,
  "sampling_period_sec": $SAMPLING_PERIOD,
  "peak_timestamp_offset_sec": $OFFSET,
  "device_info": "$DEV",
  "status": "$EXIT_STATUS",
  "note": "1s sampling; micro-spikes might not be captured"
}
EOF
    fi
}

# Execution
IFS=',' read -ra BATCH_ARR <<< "$BATCH_SIZES"

for B in "${BATCH_ARR[@]}"; do
    run_config "$B"
done

echo "DONE_REMOTE_B3_81"
