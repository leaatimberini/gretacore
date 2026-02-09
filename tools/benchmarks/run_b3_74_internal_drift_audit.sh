#!/bin/bash
set -euo pipefail

# =============================================================================
# B3.74 Internal Drift Impact Audit
# =============================================================================
# Purpose: Quantify internal drift (attention/hidden states) on MI300X
#          using Direct Stage Tracing (GRETA_TRACE_STAGE) with B3.73-style matrix runner.
#
# Usage: ./run_b3_74_internal_drift_audit.sh <NODE_IP> [YYYY-MM-DD]
#        [--span N] [--seeds "0,1,2"] [--kv_aligned "0,1"]
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
SPAN="32"
SEEDS="0,1,2"
KV_ALIGNED="0,1"
PROMPTS="p0_short,p6_len_16,p6_len_32"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --span)
            SPAN="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --kv_aligned)
            KV_ALIGNED="$2"
            shift 2
            ;;
        --prompts)
            PROMPTS="$2"
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
LOCK_FILE="/tmp/greta_b3_74.lock"
RUN_DIR="b3_74"

MODEL="./models/greta-v1.gguf"
DTYPE="bf16"

# Parse arrays
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"
IFS=',' read -ra KV_ALIGNED_ARRAY <<< "$KV_ALIGNED"
IFS=',' read -ra PROMPTS_ARRAY <<< "$PROMPTS"

echo "=== B3.74 Internal Drift Impact Audit ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "Span: $SPAN"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "KV aligned: ${KV_ALIGNED_ARRAY[*]}"
echo "Prompts: ${PROMPTS_ARRAY[*]}"
echo "Internal Dump: Enabled (via GRETA_TRACE_STAGE)"
echo ""

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
  "ticket": "B3.74",
  "base_config": "B3.66",
  "spans": [$SPAN],
  "kv_aligned": [$(echo "${KV_ALIGNED_ARRAY[*]}" | sed 's/ /, /g')],
  "seeds": [$(echo "${SEEDS_ARRAY[*]}" | sed 's/ /, /g')],
  "modes": ["prefill", "decode"],
  "prompt_cases": ["$(echo "${PROMPTS_ARRAY[*]}" | sed 's/ /", "/g')"],
  "dtype": "$DTYPE",
  "dump_format": "B3.66 Traces (Internal) + B3.69 Logits",
  "description": "Internal drift audit (attention/hidden states) on MI300X",
  "internal_audit": true,
  "timestamp": "$TIMESTAMP",
  "git_commit": "$GIT_COMMIT"
}
EOF
scp -o StrictHostKeyChecking=no "$LOCAL_RUNS_DIR/config.json" "root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/runs/"

# -----------------------------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------------------------
echo "[5/7] Run benchmarks..."

TOTAL_RUNS=$((${#PROMPTS_ARRAY[@]} * ${#KV_ALIGNED_ARRAY[@]} * ${#SEEDS_ARRAY[@]} * 2))
RUN_COUNT=0
FAILED_RUNS=()

for PROMPT_CASE in "${PROMPTS_ARRAY[@]}"; do
    for KV_VAL in "${KV_ALIGNED_ARRAY[@]}"; do
        for SEED in "${SEEDS_ARRAY[@]}"; do
            echo "  === prompt=$PROMPT_CASE, kv_aligned=$KV_VAL, seed=$SEED ==="

            # Map KV_VAL to B3.66 MODE for tracing
            # 0 -> as_designed (fingerprints)
            # 1 -> kv_aligned (scores)
            if [ "$KV_VAL" == "0" ]; then
                B366_MODE="as_designed"
            else
                B366_MODE="kv_aligned"
            fi

            for MODE in prefill decode; do
                RUN_COUNT=$((RUN_COUNT + 1))
                echo "    [$RUN_COUNT/$TOTAL_RUNS] mode=$MODE (trace_mode=$B366_MODE)..."

                OUTDIR="$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/runs/kv_aligned_$KV_VAL/seed_$SEED/$PROMPT_CASE/$MODE"
                LOCAL_OUTDIR="$LOCAL_RUNS_DIR/kv_aligned_$KV_VAL/seed_$SEED/$PROMPT_CASE/$MODE"
                mkdir -p "$LOCAL_OUTDIR"

                # Execute with timing
                START_TS=$(date +%s.%N)
                
                RUN_RESULT=$(ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
                    cd $REMOTE_BASE
                    # B3.66 deterministic flags
                    export HIP_LAUNCH_BLOCKING=1
                    export AMD_SERIALIZE_KERNEL=3
                    export HSA_ENABLE_SDMA=0
                    
                    # B3.74 Tracing Flags (Direct Stage Trace)
                    export GRETA_SEED=$SEED
                    export GRETA_TRACE_STAGE=1
                    export GRETA_TRACE_STAGE_LAYERS='0,1,2,4,8,16,24,31'
                    export GRETA_TRACE_STAGE_POINTS='attn_out,mlp_out'
                    export GRETA_TRACE_STAGE_PHASES='prefill,decode'
                    export GRETA_TRACE_STAGE_OUT=$OUTDIR/internal.jsonl

                    mkdir -p $OUTDIR

                    # Run greta_infer
                    # Note: We dump logits to reconfirm B3.73 findings
                    # The internal traces are written to GRETA_TRACE_STAGE_OUT
                    ./tools/inference/build/greta_infer \\
                        --model $MODEL \\
                        --prompt tools/benchmarks/prompts/${PROMPT_CASE}.txt \\
                        --seed $SEED \\
                        --kv-aligned $KV_VAL \\
                        --mode $MODE \\
                        --dump-logits $OUTDIR \\
                        --dump-logits-span $SPAN \\
                        --max-tokens 1 \\
                        --greedy \\
                        2>&1 | tee $OUTDIR/run.log

                    # Compress internal trace if it exists
                    if [ -f $OUTDIR/internal.jsonl ]; then
                        echo "Found trace: $OUTDIR/internal.jsonl"
                        gzip $OUTDIR/internal.jsonl
                    else
                        echo "WARNING: No internal trace file generated at $OUTDIR/internal.jsonl"
                    fi

                    # Verify outputs
                    if [ -f $OUTDIR/internal.jsonl.gz ] && [ -f $OUTDIR/logits.jsonl.gz ]; then
                        echo 'FILES_OK'
                    else
                        echo 'FILES_MISSING'
                        ls -la $OUTDIR
                    fi
                " 2>&1) || true

                END_TS=$(date +%s.%N)
                WALL_TIME=$(echo "$END_TS - $START_TS" | bc)

                # Check result
                if echo "$RUN_RESULT" | grep -q "FILES_MISSING"; then
                    FAILED_RUNS+=("prompt=$PROMPT_CASE kv=$KV_VAL seed=$SEED mode=$MODE")
                    echo "    ERROR: Files missing! Log:"
                    echo "$RUN_RESULT" | grep "WARNING" || true
                    # Continue anyway to gather partial data
                fi

                # Copy files
                scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/metadata.json" "$LOCAL_OUTDIR/" 2>/dev/null || true
                scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/logits.jsonl.gz" "$LOCAL_OUTDIR/" 2>/dev/null || true
                scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/internal.jsonl.gz" "$LOCAL_OUTDIR/" 2>/dev/null || true
                # scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/run.log" "$LOCAL_OUTDIR/" 2>/dev/null || true

                # Generate perf.json
                LOGITS_BYTES=0
                INTERNAL_BYTES=0
                if [ -f "$LOCAL_OUTDIR/logits.jsonl.gz" ]; then
                    LOGITS_BYTES=$(stat -c%s "$LOCAL_OUTDIR/logits.jsonl.gz" 2>/dev/null || echo 0)
                fi
                if [ -f "$LOCAL_OUTDIR/internal.jsonl.gz" ]; then
                    INTERNAL_BYTES=$(stat -c%s "$LOCAL_OUTDIR/internal.jsonl.gz" 2>/dev/null || echo 0)
                fi

                cat > "$LOCAL_OUTDIR/perf.json" << PERF_EOF
{
  "prompt_case": "$PROMPT_CASE",
  "kv_aligned": $KV_VAL,
  "seed": $SEED,
  "mode": "$MODE",
  "span": $SPAN,
  "wall_time_sec": $WALL_TIME,
  "logits_gz_bytes": $LOGITS_BYTES,
  "internal_gz_bytes": $INTERNAL_BYTES
}
PERF_EOF

                echo "    Done (${WALL_TIME}s, internal: ${INTERNAL_BYTES} bytes)"
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

# -----------------------------------------------------------------------------
# Package artifacts
# -----------------------------------------------------------------------------
echo "[6/7] Package artifacts..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    cd $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR
    tar -czvf gretacore_b3_74_artifacts.tgz runs/
    ls -la
"

# -----------------------------------------------------------------------------
# Run analyzer (with b3_74 mode)
# -----------------------------------------------------------------------------
echo "[7/7] Run analyzer..."
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir "$LOCAL_RUNS_DIR" \
    --output "artifacts_remote/$DATE/$RUN_DIR/B3_74_INTERNAL_DRIFT_AUDIT.md" \
    --mode b3_74

echo ""
echo "=== B3.74 Execution Summary ==="
echo "Date: $DATE"
echo "Prompts: ${PROMPTS_ARRAY[*]}"
echo "Span: $SPAN"
echo "Total runs: $TOTAL_RUNS"
echo "Failed: ${#FAILED_RUNS[@]}"
echo "Output: artifacts_remote/$DATE/$RUN_DIR/"
echo ""
echo "Done. Copy artifacts with:"
echo "  scp root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/gretacore_b3_74_artifacts.tgz ."

flock -u 200
