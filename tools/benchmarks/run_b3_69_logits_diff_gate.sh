#!/bin/bash
set -euo pipefail

# =============================================================================
# B3.69 Logits-Diff Equivalence Gate Runner
# =============================================================================
# Usage: ./run_b3_69_logits_diff_gate.sh <NODE_IP> [YYYY-MM-DD] [--span N] [--kv_aligned 0|1] [--seeds "0,1,2"]
# Default date = today
# Default span = 32 (tokens to dump)
# Default kv_aligned = all (0,1)
# Default seeds = "0,1,2"
#
# This runner executes B3.69 logits-diff equivalence gate using B3.68's
# --dump-logits and --dump-logits-span features for real numeric comparison.
#
# Key changes from B3.67:
# - Uses --dump-logits-span N instead of hardcoded gen_len=1
# - Dumps real logits for N tokens (not just metadata)
# - Enables numeric comparison: max_abs_diff, p99_abs_diff, top1_agreement
# =============================================================================

# -----------------------------------------------------------------------------
# Argument parsing (robust: detect flags vs positional date)
# -----------------------------------------------------------------------------
NODE_IP="${1:-129.212.184.200}"

# Check if second argument is a flag (starts with --) or a date
if [[ -z "${2:-}" || "${2:-}" == --* ]]; then
    DATE=$(date +%Y-%m-%d)
    shift 1 2>/dev/null || true
else
    DATE="$2"
    shift 2 2>/dev/null || true
fi

KV_ALIGNED=""  # Empty means all values
SEEDS="0,1,2"
SPAN="32"  # B3.69: Default span for logits dump

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv_aligned)
            KV_ALIGNED="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --span)
            SPAN="$2"
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
LOCK_FILE="/tmp/greta_b3_69.lock"
RUN_DIR="b3_69"

# Benchmark parameters - N-step equivalence via logits-diff
DTYPE="bf16"
MODEL="./models/greta-v1.gguf"
PROMPT="tools/benchmarks/prompts/p0_short.txt"

# Arrays
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"

# Parse KV_ALIGNED into array
if [[ -z "$KV_ALIGNED" ]]; then
    KV_ALIGNED_VALUES=(0 1)
else
    IFS=',' read -ra KV_ALIGNED_VALUES <<< "$KV_ALIGNED"
fi

echo "=== B3.69 Logits-Diff Equivalence Gate ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "KV aligned values: ${KV_ALIGNED_VALUES[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "Span: $SPAN tokens (logits dump)"

# -----------------------------------------------------------------------------
# Lock exclusivo
# -----------------------------------------------------------------------------
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "ERROR: Lock file $LOCK_FILE is held by another process"
    exit 2
fi

# -----------------------------------------------------------------------------
# Sync and remote setup
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
# Emit config.json at root of runs (for completeness guardrail)
# -----------------------------------------------------------------------------
echo "[4/7] Emit config.json..."
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse --short HEAD)

cat > "$LOCAL_RUNS_DIR/config.json" << EOF
{
  "kv_aligned": [$(IFS=,; echo "${KV_ALIGNED_VALUES[*]}" | sed 's/,/, /g')],
  "seeds": [$(IFS=,; echo "${SEEDS_ARRAY[*]}" | sed 's/,/, /g')],
  "dtype": "$DTYPE",
  "span": $SPAN,
  "mode": ["prefill", "decode"],
  "description": "B3.69 logits-diff equivalence gate matrix (span=$SPAN)",
  "timestamp": "$TIMESTAMP",
  "git_commit": "$GIT_COMMIT"
}
EOF

echo "  Created config.json with matrix: kv_aligned=[${KV_ALIGNED_VALUES[*]}] × seeds=[${SEEDS_ARRAY[*]}] × span=$SPAN"

# Also copy to remote
scp -o StrictHostKeyChecking=no "$LOCAL_RUNS_DIR/config.json" "root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/runs/"

# -----------------------------------------------------------------------------
# Run benchmarks for each configuration
# -----------------------------------------------------------------------------
echo "[5/7] Run benchmarks..."

TOTAL_RUNS=$((${#KV_ALIGNED_VALUES[@]} * ${#SEEDS_ARRAY[@]} * 2))  # 2 modes
RUN_COUNT=0
FAILED_RUNS=()

for KV_VAL in "${KV_ALIGNED_VALUES[@]}"; do
    for SEED in "${SEEDS_ARRAY[@]}"; do
        echo "  === kv_aligned=$KV_VAL, seed=$SEED ==="

        for MODE in prefill decode; do
            RUN_COUNT=$((RUN_COUNT + 1))
            echo "    [$RUN_COUNT/$TOTAL_RUNS] Running mode=$MODE (span=$SPAN)..."

            # Output directory for this configuration
            OUTDIR="$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/runs/kv_aligned_$KV_VAL/seed_$SEED/$MODE"
            LOCAL_OUTDIR="$LOCAL_RUNS_DIR/kv_aligned_$KV_VAL/seed_$SEED/$MODE"
            mkdir -p "$LOCAL_OUTDIR"

            # Execute on remote with deterministic env vars
            ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
                cd $REMOTE_BASE
                export HIP_LAUNCH_BLOCKING=1
                export AMD_SERIALIZE_KERNEL=3
                export HSA_ENABLE_SDMA=0

                # Create output directory
                mkdir -p $OUTDIR

                # Run greta_infer with --dump-logits and --dump-logits-span (B3.69)
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

                # Verify output files exist
                if [ ! -f $OUTDIR/metadata.json ]; then
                    echo 'ERROR: metadata.json not created!'
                    exit 1
                fi
                if [ ! -f $OUTDIR/logits.jsonl.gz ]; then
                    echo 'ERROR: logits.jsonl.gz not created!'
                    exit 1
                fi

                echo 'FILES_OK'
            "

            # Check if run succeeded
            if [ $? -ne 0 ]; then
                FAILED_RUNS+=("kv=$KV_VAL seed=$SEED mode=$MODE")
                echo "    ERROR: Run failed!"
                continue
            fi

            # Copy files locally
            scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/metadata.json" "$LOCAL_OUTDIR/" 2>/dev/null || true
            scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/logits.jsonl.gz" "$LOCAL_OUTDIR/" 2>/dev/null || true
            scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUTDIR/run.log" "$LOCAL_OUTDIR/" 2>/dev/null || true

            echo "    Done."
        done
    done
done

# Check for failed runs
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo ""
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
    tar -czvf gretacore_b3_69_artifacts.tgz runs/
    ls -la
"

# -----------------------------------------------------------------------------
# Run analyzer (B3.69 mode with logits-diff)
# -----------------------------------------------------------------------------
echo "[7/7] Run analyzer (local)..."
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir "$LOCAL_RUNS_DIR" \
    --output "artifacts_remote/$DATE/$RUN_DIR/B3_69_LOGITS_DIFF_GATE.md" \
    --mode b3_69

echo ""
echo "=== B3.69 Execution Summary ==="
echo "Date: $DATE"
echo "KV aligned values: ${KV_ALIGNED_VALUES[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "Span: $SPAN tokens"
echo "Total runs: $TOTAL_RUNS"
echo "Failed runs: ${#FAILED_RUNS[@]}"
echo "Output dir: artifacts_remote/$DATE/$RUN_DIR/"
echo "Report: artifacts_remote/$DATE/$RUN_DIR/B3_69_LOGITS_DIFF_GATE.md"
echo ""
echo "Done. Copy artifacts with:"
echo "  scp root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/gretacore_b3_69_artifacts.tgz ."

flock -u 200
