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
TIMEOUT_SEC=$2

REMOTE_BASE="/root/gretacore"
cd $REMOTE_BASE

REL_PATH="runs/context_${CONTEXT_LEN}/gen_${GEN_LEN}/span_${DUMP_SPAN}/dtype_${DTYPE}/kv_${KV_ALIGNED}/seed_${SEED}/batch_${BATCH}"
REMOTE_OUT="$RUN_ROOT/$REL_PATH"
mkdir -p "$REMOTE_OUT"

echo "--- context=${CONTEXT_LEN} kv=${KV_ALIGNED} ---"

PROMPT_FILE="$REMOTE_BASE/tools/benchmarks/prompts/b3_77_synthetic_${CONTEXT_LEN}.txt"
mkdir -p "$(dirname "$PROMPT_FILE")"
python3 -c "print('hello ' * $CONTEXT_LEN)" > "$PROMPT_FILE"

VRAM_SAMPLE_FILE="$REMOTE_OUT/vram_samples.csv"
(
    while true; do
        if val=$(rocm-smi --showmeminfo vram --json 2>/dev/null | python3 -c 'import sys, json; d=json.load(sys.stdin); print(list(d.values())[0]["VRAM Total Used Memory (B)"])' 2>/dev/null); then
            mb=$(echo "$val / 1048576" | bc)
            echo "$(date +%s),$mb" >> "$VRAM_SAMPLE_FILE"
        fi
        sleep 1
    done
) &
MONITOR_PID=$!

MODES=("prefill" "decode")
EXIT_STATUS="OK"

for MODE in "${MODES[@]}"; do
    MODE_OUT="$REMOTE_OUT/$MODE"
    mkdir -p "$MODE_OUT"
    
    export HIP_LAUNCH_BLOCKING=1
    export AMD_SERIALIZE_KERNEL=3
    export HSA_ENABLE_SDMA=0
    export GRETA_DETERMINISTIC=1
    export GRETA_SEED=$SEED
    
    # Timeout wrapper
    START_TIME=$(date +%s)
    
    ./tools/inference/build/greta_infer \
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
            if grep -qi 'out of memory\|OOM' "$MODE_OUT/run.log"; then
                echo "FAIL_OOM" > "$REMOTE_OUT/verdict.txt"
                EXIT_STATUS="FAIL_OOM"
            else
                echo "FAIL_ERROR (Code: $EXIT_CODE)" > "$REMOTE_OUT/verdict.txt"
                EXIT_STATUS="FAIL_ERROR"
            fi
        }
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    if [ "$ELAPSED" -gt "$TIMEOUT_SEC" ]; then
        echo "FAIL_TIMEOUT" > "$REMOTE_OUT/verdict.txt"
        EXIT_STATUS="FAIL_TIMEOUT"
        break
    fi
    
    if [ "$EXIT_STATUS" != "OK" ]; then break; fi
done

kill $MONITOR_PID || true

# Peak Calculation
if [ -s "$VRAM_SAMPLE_FILE" ]; then
    PEAK=$(awk -F, 'BEGIN {max=0} {if ($2 > max) max=$2} END {print max}' "$VRAM_SAMPLE_FILE")
else
    PEAK=0
fi
echo "{\"peak_vram_mb\": $PEAK, \"device\": \"MI300X\", \"status\": \"$EXIT_STATUS\"}" > "$REMOTE_OUT/vram.json"

if [ "$EXIT_STATUS" == "OK" ]; then
    echo "PASS_STABILITY" > "$REMOTE_OUT/verdict.txt"
fi

echo "DONE_REMOTE_$EXIT_STATUS"
