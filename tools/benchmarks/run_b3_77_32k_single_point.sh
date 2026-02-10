#!/bin/bash
# =============================================================================
# B3.77 Single-Point 32k Long-Context Attempt (MI300X) - AUDIT READY
# =============================================================================
set -euo pipefail

HOST="${1:-}"
DATE="${2:-$(date +%F)}"

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST> [DATE]"
    exit 1
fi

CONTEXT_LEN=32768
PREFILL_TIMEOUT_SEC=600
DECODE_TIMEOUT_SEC=600
OUT_ROOT="artifacts_remote"

REMOTE_BASE="/root/gretacore"
RUN_ROOT="$OUT_ROOT/$DATE/b3_77"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"
mkdir -p "$LOCAL_RUNS_DIR"

SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=120"

echo "=== B3.77 32k Single-Point Runner (Audit-Ready) ==="
echo "[1/4] Preparing remote environment..."
scp $SSH_OPTS tools/benchmarks/remote_b3_77_executor.sh root@$HOST:/tmp/remote_b3_77_executor.sh
ssh $SSH_OPTS root@$HOST "chmod +x /tmp/remote_b3_77_executor.sh"

echo "[2/4] Syncing and building on $HOST..."
ssh $SSH_OPTS root@$HOST "
    cd $REMOTE_BASE
    git fetch origin
    git reset --hard origin/main
    cd tools/inference/build
    make -j\$(nproc)
"

echo "[3/4] Executing benchmark 32k (SINGLE CONNECTION)..."
ssh $SSH_OPTS root@$HOST "/tmp/remote_b3_77_executor.sh \"$RUN_ROOT\" $PREFILL_TIMEOUT_SEC $DECODE_TIMEOUT_SEC"

echo "[4/4] Downloading artifacts..."
# The executor saves to artifacts_remote/<DATE>/b3_77/runs/...
scp -r $SSH_OPTS root@$HOST:"$REMOTE_BASE/$RUN_ROOT/*" "$RUN_ROOT/" || true

# Post-process: Generate overall config.json for analyzer
GIT_COMMIT=$(ssh $SSH_OPTS root@$HOST "cd $REMOTE_BASE && git rev-parse --short HEAD")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$LOCAL_RUNS_DIR/config.json" << EOF
{
  "ticket": "B3.77",
  "project": "32k Long-Context Attempt",
  "date": "$DATE",
  "host": "$HOST",
  "git_commit": "$GIT_COMMIT",
  "timestamp": "$TIMESTAMP",
  "timeout_policy": {
    "prefill_timeout_sec": $PREFILL_TIMEOUT_SEC,
    "decode_timeout_sec": $DECODE_TIMEOUT_SEC
  },
  "matrix": {
    "contexts": [$CONTEXT_LEN],
    "kv_aligned": [1],
    "dtype": "bf16",
    "seed": 0
  }
}
EOF

echo "B3.77 Runner Complete."
echo "Analyze with: python tools/benchmarks/analyze_b3_67_equivalence_guardrail.py --traces-dir $LOCAL_RUNS_DIR --output $RUN_ROOT/report.md --mode b3_77"
