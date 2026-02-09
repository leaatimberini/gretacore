#!/bin/bash
# =============================================================================
# B3.76 Long-Context & Memory Pressure Validation (MI300X) - SINGLE SHOT
# =============================================================================
set -euo pipefail

HOST="${1:-}"
DATE="${2:-$(date +%F)}"

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST> [DATE]"
    exit 1
fi

# Defaults
CONTEXTS="4096,8192,16384"
KV_ALIGNED="1"
GEN_LEN=128
DUMP_SPAN=32
DTYPE="bf16"
SEED=0
BATCH=1
OUT_ROOT="artifacts_remote"

REMOTE_BASE="/root/gretacore"
RUN_ROOT="$OUT_ROOT/$DATE/b3_76"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"
mkdir -p "$LOCAL_RUNS_DIR"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=120"

echo "=== B3.76 Single-Shot Runner ==="
echo "[1/4] Preparing remote environment..."
scp $SSH_OPTS tools/benchmarks/remote_b3_76_executor.sh root@$HOST:/tmp/remote_executor.sh
ssh $SSH_OPTS root@$HOST "chmod +x /tmp/remote_executor.sh"

echo "[2/4] Syncing and building on $HOST..."
ssh $SSH_OPTS root@$HOST "
    cd $REMOTE_BASE
    git fetch origin
    git reset --hard origin/main
    cd tools/inference/build
    make -j\$(nproc)
"

echo "[3/4] Executing benchmark matrix (SINGLE CONNECTION)..."
ssh $SSH_OPTS root@$HOST "/tmp/remote_executor.sh \"$CONTEXTS\" \"$KV_ALIGNED\" $GEN_LEN $DUMP_SPAN $DTYPE $SEED $BATCH \"$RUN_ROOT\""

echo "[4/4] Downloading artifacts..."
scp -r $SSH_OPTS root@$HOST:"$REMOTE_BASE/$RUN_ROOT/*" "$RUN_ROOT/" || true

# Post-process: Generate config.json and perf.json for analyzer compatibility
GIT_COMMIT=$(ssh $SSH_OPTS root@$HOST "cd $REMOTE_BASE && git rev-parse --short HEAD")
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
    "contexts": [4096, 8192, 16384],
    "kv_aligned": [1],
    "gen_len": $GEN_LEN,
    "dtype": "$DTYPE",
    "seed": $SEED,
    "batch": $BATCH
  }
}
EOF

# Ensure each mode has its perf.json
for CONTEXT in 4096 8192 16384; do
    CONTEXT_DIR="$LOCAL_RUNS_DIR/context_${CONTEXT}/gen_${GEN_LEN}/span_${DUMP_SPAN}/dtype_${DTYPE}/kv_1/seed_${SEED}/batch_${BATCH}"
    if [ -d "$CONTEXT_DIR" ]; then
        PEAK=$(cat "$CONTEXT_DIR/vram.json" 2>/dev/null | python3 -c 'import sys, json; print(json.load(sys.stdin).get("peak_vram_mb", 0))' || echo 0)
        for MODE in prefill decode; do
            MODE_DIR="$CONTEXT_DIR/$MODE"
            if [ -d "$MODE_DIR" ]; then
                cat > "$MODE_DIR/perf.json" << PERF_EOF
{
  "context_len": $CONTEXT,
  "gen_len": $GEN_LEN,
  "dtype": "$DTYPE",
  "kv_aligned": 1,
  "seed": $SEED,
  "batch": $BATCH,
  "mode": "$MODE",
  "peak_vram_mb": $PEAK
}
PERF_EOF
            fi
        done
    fi
done

echo "B3.76 Single-Shot Runner Complete."
