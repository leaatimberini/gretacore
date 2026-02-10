#!/bin/bash
# =============================================================================
# B3.78-80 Unified Suite: Long-Context Decode & Batch Probe (MI300X)
# =============================================================================
set -euo pipefail

HOST=""
DATE=$(date +%F)
TICKETS="b3_78,b3_79,b3_80"
GEN_LEN=64
DUMP_SPAN=16

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node) HOST="$2"; shift 2 ;;
        --date) DATE="$2"; shift 2 ;;
        --tickets) TICKETS="$2"; shift 2 ;;
        --gen-len) GEN_LEN="$2"; shift 2 ;;
        --dump-span) DUMP_SPAN="$2"; shift 2 ;;
        *) HOST="$1"; shift ;;
    esac
done

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST_OR_--node IP> [--date YYYY-MM-DD] [--tickets b3_78,b3_79,b3_80]"
    exit 1
fi

OUT_ROOT="artifacts_remote"
RUN_ROOT="$OUT_ROOT/$DATE/b3_78_80"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"
mkdir -p "$LOCAL_RUNS_DIR"

REMOTE_BASE="/root/gretacore"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=120"

echo "=== B3.78-80 Unified Suite Runner (MI300X) ==="
echo "[1/4] Preparing remote environment..."
scp $SSH_OPTS tools/benchmarks/remote_b3_78_80_executor.sh root@$HOST:/tmp/remote_b3_78_80_executor.sh
ssh $SSH_OPTS root@$HOST "chmod +x /tmp/remote_b3_78_80_executor.sh"

echo "[2/4] Syncing and building on $HOST..."
ssh $SSH_OPTS root@$HOST "
    cd $REMOTE_BASE
    git fetch origin
    git reset --hard origin/main
    cd tools/inference/build
    make -j\$(nproc)
"

echo "[3/4] Executing Suite: $TICKETS..."
ssh $SSH_OPTS root@$HOST "/tmp/remote_b3_78_80_executor.sh \"$TICKETS\" $GEN_LEN $DUMP_SPAN \"$RUN_ROOT\""

echo "[4/4] Downloading artifacts..."
scp -r $SSH_OPTS root@$HOST:"$REMOTE_BASE/$RUN_ROOT/*" "$RUN_ROOT/" || true

# Generate config.json for analyzer
GIT_COMMIT=$(ssh $SSH_OPTS root@$HOST "cd $REMOTE_BASE && git rev-parse --short HEAD")
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)

cat > "$LOCAL_RUNS_DIR/config.json" << EOF
{
  "suite": "B3.78-80",
  "tickets": "$TICKETS",
  "date": "$DATE",
  "host": "$HOST",
  "git_commit": "$GIT_COMMIT",
  "timestamp": "$TIMESTAMP",
  "gen_len": $GEN_LEN,
  "dump_span": $DUMP_SPAN
}
EOF

echo "B3.78-80 Runner Complete."
echo "Analyze with: python tools/benchmarks/analyze_b3_67_equivalence_guardrail.py --traces-dir $LOCAL_RUNS_DIR --output $RUN_ROOT/report.md --mode b3_78_80"
