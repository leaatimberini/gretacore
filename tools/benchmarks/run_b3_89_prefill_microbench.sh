#!/bin/bash
set -euo pipefail

HOST=""
DATE=$(date +%F)
CONTEXTS="4096,8192,16384"
GEN_LEN=1
DTYPE="bf16"
KV_ALIGNED="1"
DETERMINISM="on"
TAG=""
BUILD_DIR="tools/inference/build"

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
        --build-dir) BUILD_DIR="$2"; shift 2 ;;
        --single-shot) SINGLE_SHOT="true"; shift ;;
        --variants) VARIANTS="$2"; shift 2 ;;
        --repeat) REPEATS="$2"; shift 2 ;;
        *) if [ -z "$HOST" ]; then HOST="$1"; else echo "Unknown arg: $1"; exit 1; fi; shift ;;
    esac
done

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST> [--date YYYY-MM-DD] [--contexts 4k,8k,...] [--tag tag]"
    exit 1
fi

OUT_ROOT="artifacts_remote"
RUN_ROOT="$OUT_ROOT/$DATE/b3_89${TAG:+_$TAG}"
mkdir -p "$RUN_ROOT/runs"

cat > /tmp/remote_vars.sh << EOF
export RUN_ROOT="$RUN_ROOT"
export CONTEXTS="$CONTEXTS"
export GEN_LEN="$GEN_LEN"
export DETERMINISM="$DETERMINISM"
export BUILD_DIR="$BUILD_DIR"
export TAG="$TAG"
EOF

cat > /tmp/remote_b3_89_executor.sh << 'EOF'
#!/bin/bash
set -euo pipefail
source /tmp/remote_vars.sh
cd /root/gretacore

run_config() {
    local ctx=$1
    local rep=$2
    local out="$RUN_ROOT/runs/ctx_${ctx}_${rep}"
    mkdir -p "$out"
    python3 -c "print('a' * ($ctx - 1))" > /tmp/prompt.txt
    export GRETA_MAX_SEQ_LEN=40000
    export GRETA_VERBOSE_INFO=1
    if [ "$DETERMINISM" == "on" ]; then
        export HIP_LAUNCH_BLOCKING=1
        export AMD_SERIALIZE_KERNEL=3
        export HSA_ENABLE_SDMA=0
    fi
    echo "[B3.89] Running CTX=$ctx ..."
    START=$(date +%s.%N)
    ./$BUILD_DIR/greta_infer \
        --model ./models/greta-v1.gguf \
        --prompt-file /tmp/prompt.txt \
        --max-tokens $GEN_LEN \
        --greedy > "$out/run.log" 2>&1
    local EXIT_STATUS=$?
    END=$(date +%s.%N)
    WALL=$(echo "$END - $START" | bc)
    TIMINGS=$(grep "\[PERF_TIMING\]" "$out/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")
    STATUS_STR="OK"
    if [ $EXIT_STATUS -ne 0 ]; then STATUS_STR="FAIL"; fi
    cat > "$out/perf.json" << EOT
{
  "ticket": "b3_89",
  "context_len": $ctx,
  "wall_time_sec": $WALL,
  "exit_status": "$STATUS_STR",
  "timings": $TIMINGS,
  "tag": "$TAG",
  "determinism": "$DETERMINISM"
}
EOT
}

IFS=',' read -ra ADDR <<< "$CONTEXTS"
for ctx in "${ADDR[@]}"; do
    if [ "$ctx" == "4096" ]; then
        # Run 4k twice (Warmup + Measurement)
        run_config "$ctx" "0"
        run_config "$ctx" "1"
    else
        run_config "$ctx" "0"
    fi
done
echo "DONE_REMOTE_B3_89"
EOF

SSH_OPTS="-o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o ConnectTimeout=30"

run_remote() {
    local cmd=$1
    local retries=5
    local count=0
    while [ $count -lt $retries ]; do
        if eval "$cmd"; then
            return 0
        fi
        count=$((count + 1))
        echo "Command FAILED: $cmd. Retry $count/$retries..."
        sleep 10
    done
    return 1
}

if [ "${SINGLE_SHOT:-false}" != "true" ]; then
    run_remote "cat /tmp/remote_vars.sh | ssh $SSH_OPTS root@$HOST \"cat > /tmp/remote_vars.sh\""
    run_remote "cat /tmp/remote_b3_89_executor.sh | ssh $SSH_OPTS root@$HOST \"cat > /tmp/remote_b3_89_executor.sh\""
    run_remote "ssh $SSH_OPTS root@$HOST \"chmod +x /tmp/remote_b3_89_executor.sh\""
    run_remote "ssh $SSH_OPTS root@$HOST \"nohup /tmp/remote_b3_89_executor.sh > /tmp/b3_89.log 2>&1 &\""
else
    echo "Uploading executor script..."
    run_remote "cat tools/benchmarks/remote_b3_89_executor.sh | ssh $SSH_OPTS root@$HOST \"cat > /tmp/remote_b3_89_executor.sh\""
    run_remote "ssh $SSH_OPTS root@$HOST \"chmod +x /tmp/remote_b3_89_executor.sh\""
    
    # Run the executor remotely with retries
    echo "Running single-shot execution..."
    run_remote "ssh $SSH_OPTS root@$HOST \"bash /tmp/remote_b3_89_executor.sh '$DATE' '$VARIANTS' '$CONTEXTS' '$REPEATS'\""
    
    # Fetch artifacts
    echo "Fetching artifacts..."
    # We still need SCP for recursive directory fetch, but maybe we can tar it
    run_remote "ssh $SSH_OPTS root@$HOST \"cd /root/gretacore && tar -czf /tmp/b3_89_artifacts.tar.gz artifacts_remote/$DATE/b3_89\""
    run_remote "ssh $SSH_OPTS root@$HOST \"cat /tmp/b3_89_artifacts.tar.gz\" | cat > /tmp/b3_89_artifacts.tar.gz"
    mkdir -p artifacts_remote/$DATE/
    tar -xzf /tmp/b3_89_artifacts.tar.gz -C artifacts_remote/$DATE/ || true
    
    echo "Artifacts fetched to artifacts_remote/$DATE/"
    exit 0
fi
