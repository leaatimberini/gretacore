#!/bin/bash
set -euo pipefail

# Usage: ./run_b3_66_mi300x.sh <NODE_IP> [YYYY-MM-DD]
# Default date = today

NODE_IP="${1:-129.212.184.200}"
DATE="${2:-$(date +%Y-%m-%d)}"
REMOTE_BASE="/root/gretacore"
LOCK_FILE="/tmp/greta_b3_66.lock"

echo "=== B3.66 Prefill vs Decode Drift Probe ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"

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
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "mkdir -p $REMOTE_BASE/artifacts_remote/$DATE/b3_66/{run,traces}"

echo "[4/6] Kill stale greta_infer and set env..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    pkill -f greta_infer || true
    export HIP_LAUNCH_BLOCKING=1
    export AMD_SERIALIZE_KERNEL=3
    export HSA_ENABLE_SDMA=0
    export GRETA_SEED=1
    export GRETA_TRACE_B3_66=1
    export GRETA_TRACE_LAYERS='0,1,2,4,8'
"

echo "[5/6] Run prompts..."
for prompt in p0_short p6_len_16 p6_len_32; do
    echo "  Running $prompt..."
    ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
        cd $REMOTE_BASE
        export HIP_LAUNCH_BLOCKING=1
        export AMD_SERIALIZE_KERNEL=3
        export HSA_ENABLE_SDMA=0
        export GRETA_SEED=1
        export GRETA_TRACE_B3_66=1
        export GRETA_TRACE_LAYERS='0,1,2,4,8'
        ./tools/inference/build/greta_infer \
            --model ./models/greta-v1.gguf \
            --prompt tools/benchmarks/prompts/${prompt}.txt \
            --max-tokens 1 \
            --seed 1 \
            2>&1 | tee $REMOTE_BASE/artifacts_remote/$DATE/b3_66/run/${prompt}.log
    "
done

echo "[6/6] Package artifacts..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    cd $REMOTE_BASE/artifacts_remote/$DATE/b3_66
    tar -czvf gretacore_b3_66_artifacts.tgz run/ traces/
    ls -la
"

echo "Done. Copy artifacts with:"
echo "  scp root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/b3_66/gretacore_b3_66_artifacts.tgz ."

flock -u 200
