#!/bin/bash
set -euo pipefail

# =============================================================================
# B3.73 Reconcile B3.66 vs B3.69
# =============================================================================
# Purpose: Run B3.66 configuration but with B3.69 logits dump format to
#          reconcile the apparent contradiction between:
#          - B3.66: reported drift (EXPECTED) in hidden states/attention
#          - B3.69: showed diff=0.0 in logits
#
# Usage: ./run_b3_73_reconcile_b3_66_vs_b3_69.sh <NODE_IP> [YYYY-MM-DD]
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

# Defaults (inherited from B3.66)
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
# Configuration (inherited from B3.66)
# -----------------------------------------------------------------------------
REMOTE_BASE="/root/gretacore"
LOCK_FILE="/tmp/greta_b3_73.lock"
RUN_DIR="b3_73"

MODEL="./models/greta-v1.gguf"
DTYPE="bf16"  # B3.66 default

# Parse arrays
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"
IFS=',' read -ra KV_ALIGNED_ARRAY <<< "$KV_ALIGNED"
IFS=',' read -ra PROMPTS_ARRAY <<< "$PROMPTS"

echo "=== B3.73 Reconcile B3.66 vs B3.69 ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "Span: $SPAN"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "KV aligned: ${KV_ALIGNED_ARRAY[*]}"
echo "Prompts: ${PROMPTS_ARRAY[*]}"
echo ""
echo "Base config: B3.66 (model, prompts, deterministic flags)"
echo "Dump format: B3.69 (logits.jsonl.gz)"

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
  "ticket": "B3.73",
  "base_config": "B3.66",
  "spans": [$SPAN],
  "kv_aligned": [$(echo "${KV_ALIGNED_ARRAY[*]}" | sed 's/ /, /g')],
  "seeds": [$(echo "${SEEDS_ARRAY[*]}" | sed 's/ /, /g')],
  "modes": ["prefill", "decode"],
  "prompt_cases": ["$(echo "${PROMPTS_ARRAY[*]}" | sed 's/ /", "/g')"],
  "dtype": "$DTYPE",
  "dump_format": "B3.69 logits.jsonl.gz",
  "description": "Reconcile B3.66 hidden-state drift vs B3.69 logits equivalence",
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

            for MODE in prefill decode; do
                RUN_COUNT=$((RUN_COUNT + 1))
                echo "    [$RUN_COUNT/$TOTAL_RUNS] mode=$MODE..."

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

                    mkdir -p $OUTDIR

                    # Run greta_infer with B3.66 config but B3.69 logits dump
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
                if echo "$RUN_RESULT" | grep -q "FILES_MISSING"; then
                    FAILED_RUNS+=("prompt=$PROMPT_CASE kv=$KV_VAL seed=$SEED mode=$MODE")
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
  "prompt_case": "$PROMPT_CASE",
  "kv_aligned": $KV_VAL,
  "seed": $SEED,
  "mode": "$MODE",
  "span": $SPAN,
  "wall_time_sec": $WALL_TIME,
  "logits_gz_bytes": $LOGITS_BYTES
}
PERF_EOF

                echo "    Done (${WALL_TIME}s, ${LOGITS_BYTES} bytes)"
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
    tar -czvf gretacore_b3_73_artifacts.tgz runs/
    ls -la
"

# -----------------------------------------------------------------------------
# Run analyzer
# -----------------------------------------------------------------------------
echo "[7/7] Run analyzer..."
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir "$LOCAL_RUNS_DIR" \
    --output "artifacts_remote/$DATE/$RUN_DIR/B3_73_RECONCILE_REPORT.md" \
    --mode b3_73

echo ""
echo "=== B3.73 Execution Summary ==="
echo "Date: $DATE"
echo "Prompts: ${PROMPTS_ARRAY[*]}"
echo "Span: $SPAN"
echo "KV aligned: ${KV_ALIGNED_ARRAY[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "Total runs: $TOTAL_RUNS"
echo "Failed: ${#FAILED_RUNS[@]}"
echo "Output: artifacts_remote/$DATE/$RUN_DIR/"
echo ""
echo "Done. Copy artifacts with:"
echo "  scp root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/gretacore_b3_73_artifacts.tgz ."

flock -u 200
