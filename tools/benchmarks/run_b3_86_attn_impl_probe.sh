#!/bin/bash
# =============================================================================
# B3.86 Prefill Kernel Switch Probe (MI300X)
# =============================================================================
set -euo pipefail

HOST=""
DATE=$(date +%F)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node) HOST="$2"; shift 2 ;;
        --date) DATE="$2"; shift 2 ;;
        *) if [ -z "$HOST" ]; then HOST="$1"; else echo "Unknown arg: $1"; exit 1; fi; shift ;;
    esac
done

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST_OR_--node IP> [--date YYYY-MM-DD]"
    exit 1
fi

OUT_ROOT="artifacts_remote"
RUN_ROOT="$OUT_ROOT/$DATE/b3_86"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"
mkdir -p "$LOCAL_RUNS_DIR"

REMOTE_BASE="/root/gretacore"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=120"

echo "=== B3.86 Attention Implementation Probe Runner (MI300X) ==="
echo "[1/4] Preparing remote environment..."
scp $SSH_OPTS tools/benchmarks/remote_b3_86_executor.sh root@$HOST:/tmp/remote_b3_86_executor.sh
ssh $SSH_OPTS root@$HOST "chmod +x /tmp/remote_b3_86_executor.sh"

echo "[2/4] Syncing and building on $HOST..."
ssh $SSH_OPTS root@$HOST "
    cd $REMOTE_BASE
    git fetch origin
    git reset --hard origin/main
    mkdir -p tools/inference/build
    cd tools/inference/build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j\$(nproc)
"

echo "[3/4] Executing Suite B3.86 (via nohup)..."
ssh $SSH_OPTS root@$HOST "nohup /tmp/remote_b3_86_executor.sh \"$DATE\" > /tmp/b3_86.log 2>&1 &"
echo "Remote suite started in background. Waiting for completion..."

until ssh $SSH_OPTS root@$HOST "grep -q 'DONE_REMOTE_B3_86' /tmp/b3_86.log" 2>/dev/null; do
    echo -n "."
    sleep 30
done
echo " Suite finished."

echo "[4/4] Downloading artifacts..."
scp -r $SSH_OPTS root@$HOST:"$REMOTE_BASE/$RUN_ROOT/*" "$RUN_ROOT/" 2>/dev/null

GIT_COMMIT=$(ssh $SSH_OPTS root@$HOST "cd $REMOTE_BASE && git rev-parse --short HEAD")

cat > "$LOCAL_RUNS_DIR/config.json" << EOF
{
  "suite": "B3.86",
  "project": "Attention Kernel Probe",
  "date": "$DATE",
  "host": "$HOST",
  "git_commit": "$GIT_COMMIT",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

echo "B3.86 Runner Complete."
echo "Analyze with: python tools/benchmarks/analyze_b3_67_equivalence_guardrail.py --traces-dir $LOCAL_RUNS_DIR --output $RUN_ROOT/report.md --mode b3_86"
