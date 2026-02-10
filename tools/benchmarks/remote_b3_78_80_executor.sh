#!/bin/bash
set -euo pipefail

# Inputs
TICKETS="${1:-b3_78,b3_79,b3_80}"
GEN_LEN="${2:-64}"
DUMP_SPAN="${3:-16}"
RUN_ROOT="$4"

REMOTE_BASE="/root/gretacore"
cd "$REMOTE_BASE"

# Constants
PREFILL_TIMEOUT_SEC=600
DECODE_TIMEOUT_SEC=600
SAMPLING_PERIOD=1

# Helper to run a single configuration
run_config() {
    local ticket="$1"
    local ctx="$2"
    local kv="$3"
    local batch="$4"
    local seed="$5"
    local repeat_idx="$6"
    local dtype="bf16"
    local gen="$GEN_LEN"
    local span="$DUMP_SPAN"

    # Custom gen/span for B3.80 soak
    if [ "$ticket" == "b3_80" ]; then
        gen=32
        span=8
    fi

    local REL_PATH="runs/${ticket}/kv_${kv}/batch_${batch}/ctx_${ctx}/seed_${seed}/repeat_${repeat_idx}"
    local TARGET_OUT="$RUN_ROOT/$REL_PATH"
    mkdir -p "$TARGET_OUT"

    echo "--- [$ticket] ctx=$ctx kv=$kv batch=$batch seed=$seed repeat=$repeat_idx ---"

    local PROMPT_FILE="$REMOTE_BASE/tools/benchmarks/prompts/synthetic_${ctx}.txt"
    if [ ! -f "$PROMPT_FILE" ]; then
        mkdir -p "$(dirname "$PROMPT_FILE")"
        python3 -c "print('hello ' * $ctx)" > "$PROMPT_FILE"
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

        local TIMEOUT=$PREFILL_TIMEOUT_SEC
        if [ "$MODE" == "decode" ]; then TIMEOUT=$DECODE_TIMEOUT_SEC; fi

        export HIP_LAUNCH_BLOCKING=1
        export AMD_SERIALIZE_KERNEL=3
        export HSA_ENABLE_SDMA=0
        export GRETA_DETERMINISTIC=1
        export GRETA_SEED=$seed

        local START_TS=$(date +%s.%N)
        
        # We assume batch size support is checked by the binary
        timeout --foreground "${TIMEOUT}s" ./tools/inference/build/greta_infer \
            --model ./models/greta-v1.gguf \
            --prompt "$PROMPT_FILE" \
            --seed $seed \
            --kv-aligned $kv \
            --mode "$MODE" \
            --dump-logits "$MODE_OUT" \
            --dump-logits-span $span \
            --dtype "$dtype" \
            --max-tokens $gen \
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

        cat > "$MODE_OUT/perf.json" << EOF
{
  "ticket": "$ticket",
  "phase": "$MODE",
  "context_len": $ctx,
  "gen_len": $gen,
  "dump_span": $span,
  "dtype": "$dtype",
  "kv_aligned": $kv,
  "seed": $seed,
  "batch": $batch,
  "repeat_idx": $repeat_idx,
  "wall_time_sec": $WALL_TIME,
  "exit_status": "$EXIT_STATUS"
}
EOF
        if [[ "$EXIT_STATUS" == FAIL* ]]; then break; fi
        if [ "$EXIT_STATUS" == "SKIPPED_UNSUPPORTED_BATCH" ]; then break; fi
    done

    kill $MON_PID || true
    
    # Finalize VRAM
    if [ -s "$VRAM_FILE" ]; then
        local PEAK_DATA=$(awk -F, 'NR>1 {if ($2 > max) {max=$2; ts=$1}} END {print max "," ts}' "$VRAM_FILE")
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
  "status": "$EXIT_STATUS"
}
EOF
    fi
}

# Execution
IFS=',' read -ra TICKET_ARR <<< "$TICKETS"

for T in "${TICKET_ARR[@]}"; do
    case "$T" in
        b3_78)
            # 32k KV Control
            for KV in 0 1; do
                run_config "b3_78" 32768 $KV 1 0 0
            done
            ;;
        b3_79)
            # Batch Probe
            for B in 1 2; do
                for CTX in 8192 16384; do
                    run_config "b3_79" $CTX 1 $B 0 0
                done
            done
            ;;
        b3_80)
            # Micro-soak
            for R in {0..4}; do
                run_config "b3_80" 16384 1 1 0 $R
            done
            ;;
    esac
done

echo "DONE_REMOTE_ALL"
