#!/bin/bash
set -euo pipefail

# =============================================================================
# B3.70-71-72 Unified Sweep Runner
# =============================================================================
# Usage: ./run_b3_70_71_72_sweep.sh <NODE_IP> [YYYY-MM-DD] [--spans "32,128,512"]
#        [--dtypes "bf16,fp16"] [--kv_aligned "0,1"] [--seeds "0,1,2"]
#
# B3.70: Drift characterization (kv_aligned=0) - metrics only, no gate
# B3.71: Span escalation (32/128/512) + cost profiling
# B3.72: Cross-dtype sweep (bf16/fp16/fp8 if supported)
# =============================================================================

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
NODE_IP="${1:-129.212.184.200}"

if [[ -z "${2:-}" || "${2:-}" == --* ]]; then
    DATE=$(date +%Y-%m-%d)
    shift 1 2>/dev/null || true
else
    DATE="$2"
    shift 2 2>/dev/null || true
fi

# Defaults
SPANS="32,128,512"
DTYPES="bf16,fp16"
KV_ALIGNED="0,1"
SEEDS="0,1,2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spans)
            SPANS="$2"
            shift 2
            ;;
        --dtypes)
            DTYPES="$2"
            shift 2
            ;;
        --kv_aligned)
            KV_ALIGNED="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REMOTE_BASE="/root/gretacore"
LOCK_FILE="/tmp/greta_b3_70_71_72.lock"
RUN_DIR="b3_70_71_72"

MODEL="./models/greta-v1.gguf"
PROMPT="tools/benchmarks/prompts/p0_short.txt"

# Parse arrays
IFS=',' read -ra SPANS_ARRAY <<< "$SPANS"
IFS=',' read -ra DTYPES_ARRAY <<< "$DTYPES"
IFS=',' read -ra KV_ALIGNED_ARRAY <<< "$KV_ALIGNED"
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"

echo "=== B3.70-71-72 Unified Sweep ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "Spans: ${SPANS_ARRAY[*]}"
echo "Dtypes: ${DTYPES_ARRAY[*]}"
echo "KV aligned: ${KV_ALIGNED_ARRAY[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"

# -----------------------------------------------------------------------------
# Lock
# -----------------------------------------------------------------------------
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "ERROR: Lock file $LOCK_FILE is held by another process"
    exit 2
fi

# -----------------------------------------------------------------------------
# Sync and build
# -----------------------------------------------------------------------------
echo "[1/7] Sync remote repo..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "cd $REMOTE_BASE && git fetch origin && git reset --hard origin/main"

echo "[2/7] Build greta_infer..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "cd $REMOTE_BASE/tools/inference/build && make -j\$(nproc)"

echo "[3/7] Setup directories..."
LOCAL_RUNS_DIR="artifacts_remote/$DATE/$RUN_DIR/runs"
mkdir -p "$LOCAL_RUNS_DIR"
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "mkdir -p $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/runs"

# -----------------------------------------------------------------------------
# Emit config.json
# -----------------------------------------------------------------------------
echo "[4/7] Emit config.json..."
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse --short HEAD)

cat > "$LOCAL_RUNS_DIR/config.json" << EOF
{
  "spans": [$(echo "${SPANS_ARRAY[*]}" | sed 's/ /, /g')],
  "dtypes": ["$(echo "${DTYPES_ARRAY[*]}" | sed 's/ /", "/g')"],
  "kv_aligned": [$(echo "${KV_ALIGNED_ARRAY[*]}" | sed 's/ /, /g')],
  "seeds": [$(echo "${SEEDS_ARRAY[*]}" | sed 's/ /, /g')],
  "description": "B3.70-71-72 sweep: drift + span + dtype",
  "timestamp": "$TIMESTAMP",
  "git_commit": "$GIT_COMMIT"
}
EOF
scp -o StrictHostKeyChecking=no "$LOCAL_RUNS_DIR/config.json" "root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/runs/"

# -----------------------------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------------------------
echo "[5/7] Run benchmarks..."

TOTAL_RUNS=$((${#SPANS_ARRAY[@]} * ${#DTYPES_ARRAY[@]} * ${#KV_ALIGNED_ARRAY[@]} * ${#SEEDS_ARRAY[@]} * 2))
RUN_COUNT=0
FAILED_RUNS=()
SKIPPED_RUNS=()

for SPAN in "${SPANS_ARRAY[@]}"; do
    for DTYPE in "${DTYPES_ARRAY[@]}"; do
        for KV_VAL in "${KV_ALIGNED_ARRAY[@]}"; do
            for SEED in "${SEEDS_ARRAY[@]}"; do
                echo "  === span=$SPAN, dtype=$DTYPE, kv_aligned=$KV_VAL, seed=$SEED ==="

                for MODE in prefill decode; do
                    RUN_COUNT=$((RUN_COUNT + 1))
                    echo "    [$RUN_COUNT/$TOTAL_RUNS] mode=$MODE..."

                    OUTDIR="$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/runs/span_$SPAN/dtype_$DTYPE/kv_aligned_$KV_VAL/seed_$SEED/$MODE"
                    LOCAL_OUTDIR="$LOCAL_RUNS_DIR/span_$SPAN/dtype_$DTYPE/kv_aligned_$KV_VAL/seed_$SEED/$MODE"
                    mkdir -p "$LOCAL_OUTDIR"

                    # Execute with timing
                    START_TS=$(date +%s.%N)
                    
                    RUN_RESULT=$(ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
                        cd $REMOTE_BASE
                        export HIP_LAUNCH_BLOCKING=1
                        export AMD_SERIALIZE_KERNEL=3
                        export HSA_ENABLE_SDMA=0

                        mkdir -p $OUTDIR

                        # Check dtype support (fp8 may not be supported)
                        if [[ '$DTYPE' == 'fp8' ]]; then
                            # Try to detect fp8 support - for now assume not supported
                            echo 'DTYPE_NOT_SUPPORTED'
                            exit 0
                        fi

                        # Run greta_infer
                        ./tools/inference/build/greta_infer \
                            --model $MODEL \
                            --prompt $PROMPT \
                            --seed $SEED \
                            --kv-aligned $KV_VAL \
                            --mode $MODE \
                            --dump-logits $OUTDIR \
                            --dump-logits-span $SPAN \
                            --greedy \
                            2>&1 | tee $OUTDIR/run.log

                        # Verify outputs
                        if [ -f $OUTDIR/metadata.json ] && [ -f $OUTDIR/logits.jsonl.gz ]; then
                            echo 'FILES_OK'
                        else
                            echo 'FILES_MISSING'
                        fi
                    " 2>&1) || true

                    END_TS=$(date +%s.%N)
                    WALL_TIME=$(echo "$END_TS - $START_TS" | bc)

                    # Check result
                    if echo "$RUN_RESULT" | grep -q "DTYPE_NOT_SUPPORTED"; then
                        SKIPPED_RUNS+=("span=$SPAN dtype=$DTYPE kv=$KV_VAL seed=$SEED mode=$MODE")
                        echo "    SKIPPED: dtype $DTYPE not supported"
                        
                        # Write skip marker
                        echo "{\"status\": \"SKIPPED_UNSUPPORTED_DTYPE\", \"dtype\": \"$DTYPE\"}" > "$LOCAL_OUTDIR/skip.json"
                        continue
                    fi

                    if echo "$RUN_RESULT" | grep -q "FILES_MISSING"; then
                        FAILED_RUNS+=("span=$SPAN dtype=$DTYPE kv=$KV_VAL seed=$SEED mode=$MODE")
                        echo "    ERROR: Files missing!"
                        continue
                    fi

                    # Copy files
                    scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/metadata.json" "$LOCAL_OUTDIR/" 2>/dev/null || true
                    scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/logits.jsonl.gz" "$LOCAL_OUTDIR/" 2>/dev/null || true
                    scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/run.log" "$LOCAL_OUTDIR/" 2>/dev/null || true

                    # Generate perf.json
                    LOGITS_BYTES=0
                    if [ -f "$LOCAL_OUTDIR/logits.jsonl.gz" ]; then
                        LOGITS_BYTES=$(stat -c%s "$LOCAL_OUTDIR/logits.jsonl.gz" 2>/dev/null || echo 0)
                    fi

                    cat > "$LOCAL_OUTDIR/perf.json" << PERF_EOF
{
  "span": $SPAN,
  "dtype": "$DTYPE",
  "kv_aligned": $KV_VAL,
  "seed": $SEED,
  "mode": "$MODE",
  "wall_time_sec": $WALL_TIME,
  "logits_gz_bytes": $LOGITS_BYTES
}
PERF_EOF

                    echo "    Done (${WALL_TIME}s, ${LOGITS_BYTES} bytes)"
                done
            done
        done
    done
done

# Summary
echo ""
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED_RUNS[@]} runs failed:"
    for run in "${FAILED_RUNS[@]}"; do
        echo "  - $run"
    done
fi
if [ ${#SKIPPED_RUNS[@]} -gt 0 ]; then
    echo "INFO: ${#SKIPPED_RUNS[@]} runs skipped (unsupported dtype):"
    for run in "${SKIPPED_RUNS[@]}"; do
        echo "  - $run"
    done
fi

# -----------------------------------------------------------------------------
# Package artifacts
# -----------------------------------------------------------------------------
echo "[6/7] Package artifacts..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    cd $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR
    tar -czvf gretacore_b3_70_71_72_artifacts.tgz runs/
    ls -la
"

# -----------------------------------------------------------------------------
# Run analyzer
# -----------------------------------------------------------------------------
echo "[7/7] Run analyzer..."
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir "$LOCAL_RUNS_DIR" \
    --output "artifacts_remote/$DATE/$RUN_DIR/B3_70_71_72_SWEEP_REPORT.md" \
    --mode b3_70_71_72

echo ""
echo "=== B3.70-71-72 Execution Summary ==="
echo "Date: $DATE"
echo "Spans: ${SPANS_ARRAY[*]}"
echo "Dtypes: ${DTYPES_ARRAY[*]}"
echo "KV aligned: ${KV_ALIGNED_ARRAY[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "Total runs: $TOTAL_RUNS"
echo "Failed: ${#FAILED_RUNS[@]}"
echo "Skipped: ${#SKIPPED_RUNS[@]}"
echo "Output: artifacts_remote/$DATE/$RUN_DIR/"
echo ""
echo "Done. Copy artifacts with:"
echo "  scp root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/gretacore_b3_70_71_72_artifacts.tgz ."

flock -u 200
