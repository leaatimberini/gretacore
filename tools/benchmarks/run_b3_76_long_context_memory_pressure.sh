#!/bin/bash
# =============================================================================
# B3.76 Long-Context & Memory Pressure Validation (MI300X)
# =============================================================================
set -euo pipefail

HOST="${1:-}"
DATE="${2:-$(date +%F)}"

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST> [DATE] [flags]"
    echo "Example: $0 129.212.184.200 2026-02-09"
    exit 1
fi

shift 2 || true

# Defaults
CONTEXTS="4096,8192,16384"
GEN_LEN=128
DUMP_SPAN=32
DTYPE="bf16"
KV_ALIGNED="1"
SEED=0
BATCH=1
OUT_ROOT="artifacts_remote"

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --contexts) CONTEXTS="$2"; shift 2 ;;
        --gen-len) GEN_LEN="$2"; shift 2 ;;
        --dump-span) DUMP_SPAN="$2"; shift 2 ;;
        --kv_aligned) KV_ALIGNED="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --out-root) OUT_ROOT="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

# Setup
IFS=',' read -ra CONTEXT_ARR <<< "$CONTEXTS"
IFS=',' read -ra KV_ARR <<< "$KV_ALIGNED"

REMOTE_BASE="/root/gretacore"
RUN_ROOT="$OUT_ROOT/$DATE/b3_76"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"
mkdir -p "$LOCAL_RUNS_DIR"

SSH_OPTS="-o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPath=/tmp/ssh-greta-b376-%r@%h:%p -o ControlPersist=600"

echo "=== B3.76 Long-Context & Memory Pressure Runner ==="
echo "Host: $HOST"
echo "Date: $DATE"
echo "Contexts: $CONTEXTS"

# Sync and Build
echo "[1/3] Syncing and building on $HOST..."
ssh $SSH_OPTS "root@$HOST" "
    set -e
    mkdir -p $REMOTE_BASE
    cd $REMOTE_BASE
    git fetch origin
    git reset --hard origin/main
    cd tools/inference/build
    make -j\$(nproc)
"

# Config.json for analyzer
GIT_COMMIT=$(ssh $SSH_OPTS "root@$HOST" "cd $REMOTE_BASE && git rev-parse --short HEAD")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$LOCAL_RUNS_DIR/config.json" << EOF
{
  "ticket": "B3.76",
  "project": "Long-Context Memory Pressure",
  "date": "$DATE",
  "host": "$HOST",
  "git_commit": "$GIT_COMMIT",
  "timestamp": "$TIMESTAMP",
  "matrix": {
    "contexts": [$(echo "${CONTEXT_ARR[*]}" | sed 's/ /, /g')],
    "kv_aligned": [$(echo "${KV_ARR[*]}" | sed 's/ /, /g')],
    "gen_len": $GEN_LEN,
    "dtype": "$DTYPE",
    "seed": $SEED,
    "batch": $BATCH
  }
}
EOF

# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------
echo "[2/3] Starting Matrix Execution..."

OOM_STOPPED=0

for KV in "${KV_ARR[@]}"; do
    for CONTEXT in "${CONTEXT_ARR[@]}"; do
        
        REL_PATH="runs/context_${CONTEXT}/gen_${GEN_LEN}/span_${DUMP_SPAN}/dtype_${DTYPE}/kv_${KV}/seed_${SEED}/batch_${BATCH}"
        REMOTE_OUT="$REMOTE_BASE/$RUN_ROOT/$REL_PATH"
        LOCAL_OUT="$LOCAL_RUNS_DIR/context_${CONTEXT}/gen_${GEN_LEN}/span_${DUMP_SPAN}/dtype_${DTYPE}/kv_${KV}/seed_${SEED}/batch_${BATCH}"

        if [ "$OOM_STOPPED" -eq 1 ]; then
            echo "  [SKIP] context=${CONTEXT} kv=${KV} (Prior OOM)"
            mkdir -p "$LOCAL_OUT"
            echo '{"status": "SKIPPED_DUE_TO_OOM"}' > "$LOCAL_OUT/skip.json"
            continue
        fi

        echo "  Processing context=${CONTEXT} kv=${KV}..."

        # 1. Generate Prompt on host
        PROMPT_FILE="$REMOTE_BASE/tools/benchmarks/prompts/b3_76_synthetic_${CONTEXT}.txt"
        ssh $SSH_OPTS "root@$HOST" "
            python3 -c \"print('hello ' * $CONTEXT)\" > $PROMPT_FILE
        "

        # 2. Start VRAM Monitoring (Background)
        VRAM_SAMPLE_FILE="$REMOTE_OUT/vram_samples.csv"
        ssh $SSH_OPTS "root@$HOST" "
            mkdir -p $REMOTE_OUT
            if command -v rocm-smi >/dev/null 2>&1; then
                # Background sampler: timestamp, used_vram_mb
                (while true; do
                    val=\$(rocm-smi --showmeminfo vram --json | python3 -c 'import sys, json; d=json.load(sys.stdin); print(d[\"card0\"][\"VRAM Total Memory (B)\"][\"Used\"])')
                    mb=\$(echo \"\$val / 1048576\" | bc)
                    echo \"\$(date +%s),\$mb\" >> $VRAM_SAMPLE_FILE
                    sleep 1
                done) &
                echo \$! > $REMOTE_OUT/vram_monitor.pid
            else
                echo \"VRAM Monitoring UNAVAILABLE\"
            fi
        "

        # 3. Run Inference
        START_TS=$(date +%s.%N)
        
        # Determine modes: we usually do both prefill and decode if it's following the guardrail pattern
        # But B3.76 might be a single end-to-end run to test pressure.
        # "compare first 32 generated tokens" implies we want prefill then decode.
        
        MODES=("prefill" "decode")
        EXIT_CODE=0

        for MODE in "${MODES[@]}"; do
            MODE_REMOTE_OUT="$REMOTE_OUT/$MODE"
            MODE_LOCAL_OUT="$LOCAL_OUT/$MODE"
            mkdir -p "$MODE_LOCAL_OUT"

            # Execute
            ssh $SSH_OPTS "root@$HOST" "
                mkdir -p $MODE_REMOTE_OUT
                export HIP_LAUNCH_BLOCKING=1
                export AMD_SERIALIZE_KERNEL=3
                export HSA_ENABLE_SDMA=0
                export GRETA_DETERMINISTIC=1
                export GRETA_SEED=$SEED
                
                cd $REMOTE_BASE
                ./tools/inference/build/greta_infer \
                    --model ./models/greta-v1.gguf \
                    --prompt $PROMPT_FILE \
                    --seed $SEED \
                    --kv-aligned $KV \
                    --mode $MODE \
                    --dump-logits $MODE_REMOTE_OUT \
                    --dump-logits-span $DUMP_SPAN \
                    --dtype $DTYPE \
                    --max-tokens $GEN_LEN \
                    --greedy \
                    2>&1 | tee $MODE_REMOTE_OUT/run.log
            " || EXIT_CODE=$?

            if [ $EXIT_CODE -ne 0 ]; then
                # Check for OOM in log
                if ssh $SSH_OPTS "root@$HOST" "grep -qi 'out of memory\|OOM' $MODE_REMOTE_OUT/run.log"; then
                    echo "    [FAIL] $MODE: OOM detected!"
                    OOM_STOPPED=1
                    break
                else
                    echo "    [FAIL] $MODE: Exit code $EXIT_CODE"
                fi
            fi
        done

        # 4. Stop VRAM Monitoring
        ssh $SSH_OPTS "root@$HOST" "
            if [ -f $REMOTE_OUT/vram_monitor.pid ]; then
                pid=\$(cat $REMOTE_OUT/vram_monitor.pid)
                kill \$pid || true
                rm $REMOTE_OUT/vram_monitor.pid
            fi
            
            # Post-run Peak check
            if command -v rocm-smi >/dev/null 2>&1; then
                VRAM_STATUS=\"OK\"
                # Process samples to get peak
                if [ -s $VRAM_SAMPLE_FILE ]; then
                    PEAK=\$(awk -F, 'BEGIN {max=0} {if (\$2 > max) max=\$2} END {print max}' $VRAM_SAMPLE_FILE)
                else
                    PEAK=0
                fi
                echo \"{\\\"peak_vram_mb\\\": \$PEAK, \\\"device\\\": \\\"MI300X\\\", \\\"status\\\": \\\"\$VRAM_STATUS\\\"}\" > $REMOTE_OUT/vram.json
            else
                echo \"{\\\"peak_vram_mb\\\": 0, \\\"status\\\": \\\"UNAVAILABLE\\\"}\" > $REMOTE_OUT/vram.json
            fi
        "

        # 5. Download Results & Perf Gen
        if [ "$OOM_STOPPED" -eq 0 ]; then
            # Get PEAK from remote vram.json
            PEAK=$(ssh $SSH_OPTS "root@$HOST" "cat $REMOTE_OUT/vram.json | python3 -c 'import sys, json; print(json.load(sys.stdin).get(\"peak_vram_mb\", 0))'")
            
            for MODE in "${MODES[@]}"; do
                MODE_REMOTE_OUT="$REMOTE_OUT/$MODE"
                MODE_LOCAL_OUT="$LOCAL_OUT/$MODE"
                
                scp -q $SSH_OPTS "root@$HOST:$MODE_REMOTE_OUT/metadata.json" "$MODE_LOCAL_OUT/" || true
                scp -q $SSH_OPTS "root@$HOST:$MODE_REMOTE_OUT/logits.jsonl.gz" "$MODE_LOCAL_OUT/" || true
                
                # Fetch wall time from metadata or log? Let's use metadata if possible.
                # If metadata doesn't have it, we can use the START/END TS we took.
                # But metadata should have it.
                
                cat > "$MODE_LOCAL_OUT/perf.json" << PERF_EOF
{
  "context_len": $CONTEXT,
  "gen_len": $GEN_LEN,
  "dtype": "$DTYPE",
  "kv_aligned": $KV,
  "seed": $SEED,
  "batch": $BATCH,
  "mode": "$MODE",
  "peak_vram_mb": $PEAK
}
PERF_EOF
            done
            scp -q $SSH_OPTS "root@$HOST:$REMOTE_OUT/vram.json" "$LOCAL_OUT/" || true
            scp -q $SSH_OPTS "root@$HOST:$REMOTE_OUT/vram_samples.csv" "$LOCAL_OUT/" || true
        fi

        if [ "$OOM_STOPPED" -eq 1 ]; then
            break
        fi
    done
done

echo "[3/3] B3.76 Runner Complete."
