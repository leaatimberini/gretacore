#!/bin/bash
set -euo pipefail

# Inputs
DATE="${1:-$(date +%F)}"
RUN_ROOT="artifacts_remote/$DATE/b3_82_84"
REMOTE_BASE="/root/gretacore"
cd "$REMOTE_BASE"

# Constants
KV_ALIGNED=1
SEED=0
DTYPE="bf16"
TIMEOUT_SEC=2400
SAMPLING_PERIOD=1

run_config() {
    local ticket="$1"
    local ctx="$2"
    local batch="$3"
    local gen="$4"
    local span="$5"
    
    local REL_PATH="runs/${ticket}/ctx_${ctx}/batch_${batch}"
    local TARGET_OUT="$RUN_ROOT/$REL_PATH"
    mkdir -p "$TARGET_OUT"

    echo "--- [$ticket] ctx=$ctx batch=$batch gen=$gen span=$span ---"

    local PROMPT_FILE="$REMOTE_BASE/tools/benchmarks/prompts/synthetic_${ctx}.txt"
    if [ ! -f "$PROMPT_FILE" ]; then
        mkdir -p "$(dirname "$PROMPT_FILE")"
        python3 -c "print('h' * $ctx)" > "$PROMPT_FILE"
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

        # Performance settings
        # export HIP_LAUNCH_BLOCKING=1
        # export AMD_SERIALIZE_KERNEL=3
        # export HSA_ENABLE_SDMA=0
        export GRETA_DETERMINISTIC=1
        export GRETA_SEED=$SEED
        export GRETA_MAX_SEQ_LEN=$((ctx + gen + 128))

        local START_TS=$(date +%s.%N)
        
        timeout --foreground "${TIMEOUT_SEC}s" ./tools/inference/build/greta_infer \
            --model ./models/greta-v1.gguf \
            --prompt-file "$PROMPT_FILE" \
            --demo-tokenizer \
            --seed $SEED \
            --kv-aligned $KV_ALIGNED \
            --mode "$MODE" \
            --dump-logits "$MODE_OUT" \
            --dump-logits-span "$span" \
            --dtype "$DTYPE" \
            --max-tokens "$gen" \
            --batch-size "$batch" \
            --greedy \
            2>&1 | tee "$MODE_OUT/run.log" || {
                local ERR=$?
                if [ $ERR -eq 124 ]; then EXIT_STATUS="FAIL_TIMEOUT"; else EXIT_STATUS="FAIL_ERROR"; fi
                if grep -qi 'out of memory\|OOM' "$MODE_OUT/run.log"; then EXIT_STATUS="FAIL_OOM"; fi
            }

        local END_TS=$(date +%s.%N)
        local WALL_TIME=$(echo "$END_TS - $START_TS" | bc)
        local TPS=$(grep "Tokens/second:" "$MODE_OUT/run.log" | awk '{print $2}' || echo 0)

        cat > "$MODE_OUT/perf.json" << EOF
{
  "ticket": "$ticket",
  "phase": "$MODE",
  "context_len": $ctx,
  "gen_len": $gen,
  "dump_span": $span,
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
    done

    kill $MON_PID || true
    
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

# B3.82: Steady-state batch scaling (8k ctx, 1024 gen)
for B in 1 2 4 8; do
    run_config "b3_82" 8192 "$B" 1024 0
done

# B3.83: Extreme long decode (32k ctx, 512 gen)
run_config "b3_83" 32768 1 512 0

# B3.84: 16k + batch 8 + decode 256
run_config "b3_84" 16384 8 256 0

echo "DONE_REMOTE_B3_82_84"
