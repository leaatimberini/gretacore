#!/bin/bash
set -euo pipefail

# Inputs
CONTEXTS=$1
KV_ALIGNED=$2
GEN_LEN=$3
DUMP_SPAN=$4
DTYPE=$5
SEED=$6
BATCH=$7
RUN_ROOT=$8

REMOTE_BASE="/root/gretacore"
cd $REMOTE_BASE

IFS=',' read -ra CONTEXT_ARR <<< "$CONTEXTS"
IFS=',' read -ra KV_ARR <<< "$KV_ALIGNED"

OOM_STOPPED=0

for KV in "${KV_ARR[@]}"; do
    for CONTEXT in "${CONTEXT_ARR[@]}"; do
        REL_PATH="runs/context_${CONTEXT}/gen_${GEN_LEN}/span_${DUMP_SPAN}/dtype_${DTYPE}/kv_${KV}/seed_${SEED}/batch_${BATCH}"
        REMOTE_OUT="$RUN_ROOT/$REL_PATH"
        
        if [ "$OOM_STOPPED" -eq 1 ]; then
            mkdir -p "$REMOTE_OUT"
            echo '{"status": "SKIPPED_DUE_TO_OOM"}' > "$REMOTE_OUT/skip.json"
            continue
        fi

        echo "--- context=${CONTEXT} kv=${KV} ---"
        mkdir -p "$REMOTE_OUT"

        PROMPT_FILE="$REMOTE_BASE/tools/benchmarks/prompts/b3_76_synthetic_${CONTEXT}.txt"
        mkdir -p "$(dirname "$PROMPT_FILE")"
        python3 -c "print('hello ' * $CONTEXT)" > "$PROMPT_FILE"

        VRAM_SAMPLE_FILE="$REMOTE_OUT/vram_samples.csv"
        # Background sampler
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
        for MODE in "${MODES[@]}"; do
            MODE_OUT="$REMOTE_OUT/$MODE"
            mkdir -p "$MODE_OUT"
            
            export HIP_LAUNCH_BLOCKING=1
            export AMD_SERIALIZE_KERNEL=3
            export HSA_ENABLE_SDMA=0
            export GRETA_DETERMINISTIC=1
            export GRETA_SEED=$SEED
            
            ./tools/inference/build/greta_infer \
                --model ./models/greta-v1.gguf \
                --prompt "$PROMPT_FILE" \
                --seed $SEED \
                --kv-aligned $KV \
                --mode "$MODE" \
                --dump-logits "$MODE_OUT" \
                --dump-logits-span $DUMP_SPAN \
                --dtype "$DTYPE" \
                --max-tokens $GEN_LEN \
                --greedy \
                2>&1 | tee "$MODE_OUT/run.log" || {
                    if grep -qi 'out of memory\|OOM' "$MODE_OUT/run.log"; then
                        echo "OOM detected in $MODE"
                        OOM_STOPPED=1
                    fi
                }
            if [ "$OOM_STOPPED" -eq 1 ]; then break; fi
        done

        kill $MONITOR_PID || true
        
        # Peak Calculation
        if [ -s "$VRAM_SAMPLE_FILE" ]; then
            PEAK=$(awk -F, 'BEGIN {max=0} {if ($2 > max) max=$2} END {print max}' "$VRAM_SAMPLE_FILE")
        else
            PEAK=0
        fi
        echo "{\"peak_vram_mb\": $PEAK, \"device\": \"MI300X\", \"status\": \"OK\"}" > "$REMOTE_OUT/vram.json"
    done
done
echo "DONE_REMOTE"
