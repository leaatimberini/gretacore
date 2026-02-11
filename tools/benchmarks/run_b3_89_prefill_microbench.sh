#!/bin/bash
# =============================================================================
# B3.89 Prefill Kernel Microbench (MI300X)
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
RUN_ROOT="$OUT_ROOT/$DATE/b3_89"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"
mkdir -p "$LOCAL_RUNS_DIR"

REMOTE_BASE="/root/gretacore"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=120"

echo "=== B3.89 Prefill Kernel Microbench Runner (MI300X) ==="

# We use the same executor as B3.85 but focused on one run or a specific set
# If we don't have a dedicated remote script, we create one on the fly or reuse b3_85
# For B3.89, let's create a dedicated simplified one.

cat > /tmp/remote_b3_89_executor.sh << 'EOF'
#!/bin/bash
set -euo pipefail
DATE=$1
RUN_ROOT="artifacts_remote/$DATE/b3_89"
mkdir -p "$RUN_ROOT/runs"
cd /root/gretacore

run_config() {
    local ctx=$1
    local out="$RUN_ROOT/runs/ctx_$ctx"
    mkdir -p "$out"
    python3 -c "print('a' * ($ctx - 1))" > /tmp/prompt.txt
    export GRETA_MAX_SEQ_LEN=40000
    export GRETA_VERBOSE_INFO=1
    START=$(date +%s.%N)
    ./tools/inference/build/greta_infer \
        --model ./models/greta-v1.gguf \
        --prompt-file /tmp/prompt.txt \
        --max-tokens 1 \
        --greedy > "$out/run.log" 2>&1
    END=$(date +%s.%N)
    WALL=$(echo "$END - $START" | bc)
    TIMINGS=$(grep "\[PERF_TIMING\]" "$out/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")
    cat > "$out/perf.json" << EOT
{
  "ticket": "b3_89",
  "context_len": $ctx,
  "wall_time_sec": $WALL,
  "timings": $TIMINGS
}
EOT
}
run_config 8192
run_config 16384
run_config 32768
echo "DONE_REMOTE_B3_89"
EOF

scp $SSH_OPTS /tmp/remote_b3_89_executor.sh root@$HOST:/tmp/remote_b3_89_executor.sh
ssh $SSH_OPTS root@$HOST "chmod +x /tmp/remote_b3_89_executor.sh"

echo "[1/4] Starting remote microbench..."
ssh $SSH_OPTS root@$HOST "nohup /tmp/remote_b3_89_executor.sh \"$DATE\" > /tmp/b3_89.log 2>&1 &"

echo "Remote microbench started in background. Monitor with /tmp/b3_89.log"
echo "Analyze with: python tools/benchmarks/analyze_b3_67_equivalence_guardrail.py --mode b3_89 --traces-dir $LOCAL_RUNS_DIR --output $RUN_ROOT/report.md"
