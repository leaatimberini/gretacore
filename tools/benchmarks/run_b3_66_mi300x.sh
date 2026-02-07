#!/bin/bash
set -euo pipefail

# Usage: ./run_b3_66_mi300x.sh <NODE_IP> [YYYY-MM-DD] [--mode as_designed|kv_aligned | <MODE>]
# Default date = today
# Default mode = as_designed
# <MODE> can be positional (4th arg) or via --mode flag (flag has priority)

# Extract NODE_IP and DATE from first two positional args
NODE_IP="${1:-129.212.184.200}"
DATE="${2:-$(date +%Y-%m-%d)}"

# Shift args to process only mode-related arguments
shift $(( $# > 2 ? 2 : 0 )) 2>/dev/null || true

# Parse mode arguments (support for both --mode flag and positional mode)
MODE_POSITIONAL=""
MODE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE_FLAG="$2"
            shift 2
            ;;
        as_designed|kv_aligned)
            MODE_POSITIONAL="$1"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# --mode flag has priority, then positional, then default
MODE="${MODE_FLAG:-${MODE_POSITIONAL:-as_designed}}"

# Validate mode
if [[ "$MODE" != "as_designed" && "$MODE" != "kv_aligned" ]]; then
    echo "ERROR: Invalid mode '$MODE'. Must be 'as_designed' or 'kv_aligned'"
    exit 1
fi

REMOTE_BASE="/root/gretacore"
LOCK_FILE="/tmp/greta_b3_66_v2.lock"
RUN_DIR="b3_66_v2"

echo "=== B3.66 v2 Prefill vs Decode Drift Probe ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "Mode: $MODE"

# Lock exclusivo
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "ERROR: Lock file $LOCK_FILE is held by another process"
    exit 2
fi

echo "[1/6] Sync remote repo..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "cd $REMOTE_BASE && git fetch origin && git reset --hard origin/main"

echo "[2/6] Build greta_infer..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "cd $REMOTE_BASE/tools/inference/build && make -j\$(nproc)"

echo "[3/6] Setup directories..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "mkdir -p $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/{run,traces}"

echo "[4/6] Kill stale greta_infer and set env..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    pkill -f greta_infer || true
    export HIP_LAUNCH_BLOCKING=1
    export AMD_SERIALIZE_KERNEL=3
    export HSA_ENABLE_SDMA=0
    export GRETA_SEED=1
    export GRETA_TRACE_B3_66=1
    export GRETA_TRACE_LAYERS='0,1,2,4,8'
    export GRETA_B3_66_MODE=$MODE
"

echo "[5/6] Run prompts..."
for prompt in p0_short p6_len_16 p6_len_32; do
    echo "  Running $prompt with mode=$MODE..."
    ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
        cd $REMOTE_BASE
        export HIP_LAUNCH_BLOCKING=1
        export AMD_SERIALIZE_KERNEL=3
        export HSA_ENABLE_SDMA=0
        export GRETA_SEED=1
        export GRETA_TRACE_B3_66=1
        export GRETA_TRACE_LAYERS='0,1,2,4,8'
        export GRETA_B3_66_MODE=$MODE
        ./tools/inference/build/greta_infer \
            --model ./models/greta-v1.gguf \
            --prompt tools/benchmarks/prompts/${prompt}.txt \
            --max-tokens 1 \
            --seed 1 \
            --mode $MODE \
            2>&1 | tee $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/run/${prompt}_${MODE}.log
    "
done

echo "[6/6] Package artifacts..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    cd $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR
    tar -czvf gretacore_b3_66_v2_${MODE}_artifacts.tgz run/ traces/
    ls -la
"

echo "[7/7] Run analyzer (local)..."
python3 tools/benchmarks/analyze_b3_66_prefill_decode_drift.py \
    --traces-dir artifacts_remote/$DATE/$RUN_DIR/traces \
    --mode $MODE \
    --output artifacts_remote/$DATE/$RUN_DIR/B3_66_V2_ANALYSIS.md

echo ""
echo "=== B3.66 v2 Execution Summary ==="
echo "Mode: $MODE"
echo "Date: $DATE"
echo "Prompts run: p0_short, p6_len_16, p6_len_32"
echo "Output dir: artifacts_remote/$DATE/$RUN_DIR/"
echo "Report: artifacts_remote/$DATE/$RUN_DIR/B3_66_V2_ANALYSIS.md"
echo ""
echo "Done. Copy artifacts with:"
echo "  scp root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/gretacore_b3_66_v2_${MODE}_artifacts.tgz ."

flock -u 200
