#!/bin/bash
set -euo pipefail

# Inputs
CONTEXT_LEN=32768
KV_ALIGNED=1
GEN_LEN=64
DUMP_SPAN=16
DTYPE="bf16"
SEED=0
BATCH=1
RUN_ROOT=$1
PREFILL_TIMEOUT_SEC=$2
DECODE_TIMEOUT_SEC=$3

REMOTE_BASE="/root/gretacore"
cd $REMOTE_BASE

# Audit-ready layout: artifacts_remote/<DATE>/b3_77/runs/kv_<K>/seed_<S>/ctx_32768/
# Note: RUN_ROOT is already artifacts_remote/<DATE>/b3_77
REL_PATH="runs/kv_${KV_ALIGNED}/seed_${SEED}/ctx_${CONTEXT_LEN}"
REMOTE_OUT="$RUN_ROOT/$REL_PATH"
mkdir -p "$REMOTE_OUT"

echo "--- B3.77 MI300X Audit-Ready Probe (ctx=${CONTEXT_LEN}, kv=${KV_ALIGNED}) ---"

PROMPT_FILE="$REMOTE_BASE/tools/benchmarks/prompts/b3_77_synthetic_${CONTEXT_LEN}.txt"
mkdir -p "$(dirname "$PROMPT_FILE")"
python3 -c "print('hello ' * $CONTEXT_LEN)" > "$PROMPT_FILE"

VRAM_SAMPLE_FILE="$REMOTE_OUT/vram_samples.csv"
SAMPLING_PERIOD=1
START_SAMPLING_TS=$(date +%s)

# Background sampler with timestamps
(
    echo "ts_epoch,used_vram_mb" > "$VRAM_SAMPLE_FILE"
    while true; do
        if val=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c 'import sys, json; d=json.load(sys.stdin); print(list(d.values())[0]["VRAM Total Used Memory (B)"])' 2>/dev/null); then
            mb=$(echo "$val / 1048576" | bc)
            echo "$(date +%s),$mb" >> "$VRAM_SAMPLE_FILE"
        fi
        sleep $SAMPLING_PERIOD
    done
) &
MONITOR_PID=$!

MODES=("prefill" "decode")
EXIT_STATUS="OK"

for MODE in "${MODES[@]}"; do
    MODE_OUT="$REMOTE_OUT/$MODE"
    mkdir -p "$MODE_OUT"
    
    TIMEOUT=$PREFILL_TIMEOUT_SEC
    if [ "$MODE" == "decode" ]; then TIMEOUT=$DECODE_TIMEOUT_SEC; fi
    
    export HIP_LAUNCH_BLOCKING=1
    export AMD_SERIALIZE_KERNEL=3
    export HSA_ENABLE_SDMA=0
    export GRETA_DETERMINISTIC=1
    export GRETA_SEED=$SEED
    
    echo "  Starting $MODE (timeout=${TIMEOUT}s)..."
    START_TS=$(date +%s.%N)
    
    # Use 'timeout' command to ensure it applies to the process
    timeout --foreground "${TIMEOUT}s" ./tools/inference/build/greta_infer \
        --model ./models/greta-v1.gguf \
        --prompt "$PROMPT_FILE" \
        --seed $SEED \
        --kv-aligned $KV_ALIGNED \
        --mode "$MODE" \
        --dump-logits "$MODE_OUT" \
        --dump-logits-span $DUMP_SPAN \
        --dtype "$DTYPE" \
        --max-tokens $GEN_LEN \
        --greedy \
        2>&1 | tee "$MODE_OUT/run.log" || {
            EXIT_CODE=$?
            if [ $EXIT_CODE -eq 124 ]; then
                echo "FAIL_TIMEOUT" > "$REMOTE_OUT/verdict.txt"
                EXIT_STATUS="FAIL_TIMEOUT"
            elif grep -qi 'out of memory\|OOM' "$MODE_OUT/run.log"; then
                echo "FAIL_OOM" > "$REMOTE_OUT/verdict.txt"
                EXIT_STATUS="FAIL_OOM"
            else
                echo "FAIL_ERROR (Code: $EXIT_CODE)" > "$REMOTE_OUT/verdict.txt"
                EXIT_STATUS="FAIL_ERROR"
            fi
        }
    
    END_TS=$(date +%s.%N)
    WALL_TIME=$(echo "$END_TS - $START_TS" | bc)
    
    # Save perf.json for this phase
    cat > "$MODE_OUT/perf.json" << EOF
{
  "wall_time_sec": $WALL_TIME,
  "phase": "$MODE",
  "context_len": $CONTEXT_LEN,
  "gen_len": $GEN_LEN,
  "dump_span": $DUMP_SPAN,
  "dtype": "$DTYPE",
  "kv_aligned": $KV_ALIGNED,
  "seed": $SEED,
  "batch": $BATCH,
  "timeout_policy_sec": $TIMEOUT
}
EOF
    
    if [ "$EXIT_STATUS" != "OK" ]; then break; fi
done

kill $MONITOR_PID || true

# Enhanced VRAM metadata
if [ -s "$VRAM_SAMPLE_FILE" ]; then
    # Skip header for awk
    PEAK_DATA=$(awk -F, 'NR>1 {if ($2 > max) {max=$2; ts=$1}} END {print max "," ts}' "$VRAM_SAMPLE_FILE")
    PEAK=$(echo $PEAK_DATA | cut -d, -f1)
    PEAK_TS=$(echo $PEAK_DATA | cut -d, -f2)
    COUNT=$(awk 'END {print NR-1}' "$VRAM_SAMPLE_FILE")
    OFFSET=$((PEAK_TS - START_SAMPLING_TS))
    DEVICE_INFO=$(rocm-smi --showproductname --json | python3 -c 'import sys, json; d=json.load(sys.stdin); print(list(d.values())[0]["Card series"])' 2>/dev/null || echo "AMD MI300X")
else
    PEAK=0; COUNT=0; OFFSET=0; DEVICE_INFO="UNKNOWN"
fi

cat > "$REMOTE_OUT/vram.json" << EOF
{
  "peak_vram_mb": $PEAK,
  "samples_count": $COUNT,
  "sampling_period_sec": $SAMPLING_PERIOD,
  "peak_timestamp_offset_sec": $OFFSET,
  "device_info": "$DEVICE_INFO",
  "status": "$EXIT_STATUS",
  "note": "1s sampling; micro-spikes might not be captured"
}
EOF

if [ "$EXIT_STATUS" == "OK" ]; then
    echo "PASS_STABILITY" > "$REMOTE_OUT/verdict.txt"
fi

echo "DONE_REMOTE_$EXIT_STATUS"
