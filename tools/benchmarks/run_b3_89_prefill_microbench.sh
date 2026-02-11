#!/bin/bash
# =============================================================================
# B3.89 Prefill Kernel Microbench (MI300X)
# =============================================================================
set -euo pipefail

HOST=""
DATE=$(date +%F)
CONTEXTS="4096,8192,16384"
GEN_LEN=1
DTYPE="bf16"
KV_ALIGNED="1"
DETERMINISM="on"
TAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --node) HOST="$2"; shift 2 ;;
        --date) DATE="$2"; shift 2 ;;
        --contexts) CONTEXTS="$2"; shift 2 ;;
        --gen-len) GEN_LEN="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        --kv_aligned) KV_ALIGNED="$2"; shift 2 ;;
        --determinism) DETERMINISM="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        *) if [ -z "$HOST" ]; then HOST="$1"; else echo "Unknown arg: $1"; exit 1; fi; shift ;;
    esac
done

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST> [--date YYYY-MM-DD] [--contexts 4k,8k,...] [--gen-len 1] [--tag tag]"
    exit 1
fi

OUT_ROOT="artifacts_remote"
FINAL_TAG=${TAG:-"runs"}
RUN_ROOT="$OUT_ROOT/$DATE/b3_89${TAG:+_$TAG}"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"
mkdir -p "$LOCAL_RUNS_DIR"

REMOTE_BASE="/root/gretacore"
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=120"

echo "=== B3.89 Prefill Kernel Microbench Runner (MI300X) ==="

cat > /tmp/remote_b3_89_executor.sh << EOF
#!/bin/bash
set -euo pipefail
DATE=\$1
RUN_ROOT="$RUN_ROOT"
mkdir -p "\$RUN_ROOT/runs"
cd /root/gretacore

run_config() {
    local ctx=\$1
    local out="\$RUN_ROOT/runs/ctx_\$ctx"
    mkdir -p "\$out"
    
    # Setup prompt
    python3 -c "print('a' * (\$ctx - 1))" > /tmp/prompt.txt
    
    # Setup environment
    export GRETA_MAX_SEQ_LEN=40000
    export GRETA_VERBOSE_INFO=1
    if [ "$DETERMINISM" == "on" ]; then
        export HIP_LAUNCH_BLOCKING=1
        export AMD_SERIALIZE_KERNEL=3
        export HSA_ENABLE_SDMA=0
    fi

    START=\$(date +%s.%N)
    ./tools/inference/build/greta_infer \\
        --model ./models/greta-v1.gguf \\
        --prompt-file /tmp/prompt.txt \\
        --max-tokens $GEN_LEN \\
        --greedy > "\$out/run.log" 2>&1
    local EXIT_STATUS=\$?
    END=\$(date +%s.%N)
    WALL=\$(echo "\$END - \$START" | bc)
    
    TIMINGS=\$(grep "\[PERF_TIMING\]" "\$out/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")
    
    STATUS_STR="OK"
    if [ \$EXIT_STATUS -ne 0 ]; then
        STATUS_STR="FAIL"
    fi

    cat > "\$out/perf.json" << EOT
{
  "ticket": "b3_89",
  "context_len": \$ctx,
  "wall_time_sec": \$WALL,
  "exit_status": "\$STATUS_STR",
  "timings": \$TIMINGS,
  "tag": "$TAG",
  "determinism": "$DETERMINISM"
}
EOT
}

IFS=',' read -ra ADDR <<< "$CONTEXTS"
for ctx in "\${ADDR[@]}"; do
    run_config "\$ctx"
done

echo "DONE_REMOTE_B3_89"
EOF

scp $SSH_OPTS /tmp/remote_b3_89_executor.sh root@$HOST:/tmp/remote_b3_89_executor.sh
ssh $SSH_OPTS root@$HOST "chmod +x /tmp/remote_b3_89_executor.sh"

echo "[1/4] Starting remote microbench..."
ssh $SSH_OPTS root@$HOST "nohup /tmp/remote_b3_89_executor.sh \"$DATE\" > /tmp/b3_89.log 2>&1 &"

echo "Remote microbench started in background. Monitor with /tmp/b3_89.log"
echo "Analyze with: python tools/benchmarks/analyze_b3_67_equivalence_guardrail.py --mode b3_89 --traces-dir $LOCAL_RUNS_DIR --output $RUN_ROOT/report.md"
