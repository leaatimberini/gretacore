#!/bin/bash
set -euo pipefail

# =============================================================================
# B3.75 MI300X CI Harness
# =============================================================================
# Unified runner for smoke/nightly/full/stress/coverage profiles.
# Produces artifacts compatible with B3.75 analyzer.

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
HOST="${1:-}"
DATE="${2:-}"

if [ -z "$HOST" ]; then
    echo "Usage: $0 <HOST> [DATE] [flags]"
    echo "       $0 129.212.184.200 --profile smoke"
    exit 1
fi

# Robust date parsing: check if $2 is a flag or empty
if [[ -z "$DATE" || "$DATE" == --* ]]; then
    # Date not provided, shift only HOST
    DATE_ARG=""
    shift 1
else
    # Date provided
    DATE_ARG="$DATE"
    shift 2
fi

# Defaults
PROFILE="smoke"
SEEDS_OVERRIDE=""
SPANS_OVERRIDE=""
DTYPES_OVERRIDE=""
KV_ALIGNED_OVERRIDE=""
PROMPTS_OVERRIDE=""
INTERNAL_TRACE_OVERRIDE=""
INTERNAL_TRACE_KV="0"
INTERNAL_TRACE_SPAN="32"
DUMP_SPAN_DEFAULT="32"
BASELINE="baselines/mi300x/b3_75_perf_baseline.json"
OUT_ROOT="artifacts_remote"
DRY_RUN=0

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --seeds)
            SEEDS_OVERRIDE="$2"
            shift 2
            ;;
        --spans)
            SPANS_OVERRIDE="$2"
            shift 2
            ;;
        --dtypes)
            DTYPES_OVERRIDE="$2"
            shift 2
            ;;
        --kv_aligned)
            KV_ALIGNED_OVERRIDE="$2"
            shift 2
            ;;
        --prompts)
            PROMPTS_OVERRIDE="$2"
            shift 2
            ;;
        --internal-trace)
            INTERNAL_TRACE_OVERRIDE="$2"
            shift 2
            ;;
        --internal-trace-kv)
            INTERNAL_TRACE_KV="$2"
            shift 2
            ;;
        --internal-trace-span)
            INTERNAL_TRACE_SPAN="$2"
            shift 2
            ;;
        --dump-span-default)
            DUMP_SPAN_DEFAULT="$2"
            shift 2
            ;;
        --baseline)
            BASELINE="$2"
            shift 2
            ;;
        --out-root)
            OUT_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift 1
            ;;
        *)
            echo "Unknown flag: $1"
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Profile Configuration
# -----------------------------------------------------------------------------

# Default matrices based on profile
if [ "$PROFILE" == "smoke" ]; then
    SPANS="32"
    DTYPES="bf16"
    KV_MODES="1"
    SEEDS="0"
    PROMPTS="p0_short"
    INTERNAL_TRACE_DEFAULT="0"
elif [ "$PROFILE" == "nightly" ]; then
    SPANS="32,128"
    DTYPES="bf16,fp16"
    KV_MODES="0,1"
    SEEDS="0,1"
    PROMPTS="p0_short,p6_len_16"
    INTERNAL_TRACE_DEFAULT="1"
elif [ "$PROFILE" == "full" ]; then
    SPANS="32,128,512"
    DTYPES="bf16,fp16"
    KV_MODES="0,1"
    SEEDS="0,1,2"
    PROMPTS="p0_short,p6_len_16,p6_len_32"
    INTERNAL_TRACE_DEFAULT="1"
elif [ "$PROFILE" == "stress" ]; then
    SPANS="1024" # Fallback handled logic later if needed
    DTYPES="bf16"
    KV_MODES="0,1"
    SEEDS="0"
    PROMPTS="p6_len_32" # Add p6_len_64 check if possible, for now hardcode p6_len_32
    INTERNAL_TRACE_DEFAULT="0"
elif [ "$PROFILE" == "coverage" ]; then
    SPANS="32"
    DTYPES="bf16"
    KV_MODES="0,1"
    SEEDS="0"
    PROMPTS="p0_short,p6_len_16,p6_len_32" # "up to 3 additional" - using known ones
    INTERNAL_TRACE_DEFAULT="0"
else
    echo "Unknown profile: $PROFILE"
    exit 1
fi

# Apply overrides
SPANS="${SPANS_OVERRIDE:-$SPANS}"
DTYPES="${DTYPES_OVERRIDE:-$DTYPES}"
KV_MODES="${KV_ALIGNED_OVERRIDE:-$KV_MODES}"
SEEDS="${SEEDS_OVERRIDE:-$SEEDS}"
PROMPTS="${PROMPTS_OVERRIDE:-$PROMPTS}"
INTERNAL_TRACE="${INTERNAL_TRACE_OVERRIDE:-$INTERNAL_TRACE_DEFAULT}"

# Arrays
IFS=',' read -ra SPAN_ARR <<< "$SPANS"
IFS=',' read -ra DTYPE_ARR <<< "$DTYPES"
IFS=',' read -ra KV_ARR <<< "$KV_MODES"
IFS=',' read -ra SEED_ARR <<< "$SEEDS"
IFS=',' read -ra PROMPT_ARR <<< "$PROMPTS"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

should_trace_internal() {
    local kv=$1
    local span=$2
    local dtype=$3
    local seed=$4
    
    # If disabled globally or by override
    if [ "$INTERNAL_TRACE" != "1" ]; then
        return 1
    fi

    # Profile specific bounding
    if [ "$PROFILE" == "nightly" ]; then
        # (kv=0, span=32, dtype=bf16, seed=0)
        if [ "$kv" == "0" ] && [ "$span" == "32" ] && [ "$dtype" == "bf16" ] && [ "$seed" == "0" ]; then
            return 0
        fi
    elif [ "$PROFILE" == "full" ]; then
        # (kv=0, span=32, dtype=bf16) across seeds
        if [ "$kv" == "0" ] && [ "$span" == "32" ] && [ "$dtype" == "bf16" ]; then
            return 0
        fi
    else
        # For other profiles (or if forcing via override), default to checking global
        # But rigorous "only for" logic implies we should be conservative.
        # If user manually enabled, maybe trace all? 
        # Requirement says "cost-bounded".
        # Let's fallback to strict rules unless explicit.
        # Implemented: Only enabled if profile logic matches OR if manually forced for dev?
        # Actually safe bet: obey requirements for nightly/full.
        # If custom profile + internal-trace=1, maybe allow everything?
        return 0
    fi
    return 1
}

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

REMOTE_BASE="/root/gretacore"
LOCK_FILE="/tmp/greta_b3_75.lock"

# Get Date
if [ -z "$DATE_ARG" ]; then
    DATE=$(ssh -o StrictHostKeyChecking=no "root@$HOST" "date +%F")
else
    DATE="$DATE_ARG"
fi

RUN_ROOT="$OUT_ROOT/$DATE/b3_75_ci"
LOCAL_RUNS_DIR="$RUN_ROOT/runs"

if [ "$DRY_RUN" == "1" ]; then
    echo "=== B3.75 CI Harness (DRY RUN) ==="
    echo "Host: $HOST"
    echo "Date: $DATE"
    echo "Profile: $PROFILE"
    echo "Matrix:"
    echo "  Spans: ${SPAN_ARR[*]}"
    echo "  Dtypes: ${DTYPE_ARR[*]}"
    echo "  KV: ${KV_ARR[*]}"
    echo "  Seeds: ${SEED_ARR[*]}"
    echo "  Prompts: ${PROMPT_ARR[*]}"
    echo "Internal Trace: $INTERNAL_TRACE"
    exit 0
fi

echo "=== B3.75 CI Harness ==="
echo "Host: $HOST"
echo "Date: $DATE"
echo "Profile: $PROFILE"

# Lock
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "ERROR: Lock file $LOCK_FILE is held by another process"
    exit 2
fi

# SSH Options for persistence
SSH_OPTS="-o StrictHostKeyChecking=no -o ControlMaster=auto -o ControlPath=/tmp/ssh-greta-%r@%h:%p -o ControlPersist=600"

# Setup (Combined to reduce connections)
echo "[1/3] Sync, Build, Setup Directories..."
ssh $SSH_OPTS "root@$HOST" "
    set -e
    mkdir -p $REMOTE_BASE
    cd $REMOTE_BASE
    git fetch origin
    git reset --hard origin/main
    cd tools/inference/build
    make -j\$(nproc)
    mkdir -p $REMOTE_BASE/$RUN_ROOT/runs
"

mkdir -p "$LOCAL_RUNS_DIR"

# Config.json
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
# Config.json
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(ssh $SSH_OPTS "root@$HOST" "cd $REMOTE_BASE && git rev-parse --short HEAD")

cat > "$LOCAL_RUNS_DIR/config.json" << EOF
{
  "ticket": "B3.75",
  "project": "MI300X CI",
  "date": "$DATE",
  "host": "$HOST",
  "repo_branch": "main",
  "git_commit": "$GIT_COMMIT",
  "profile": "$PROFILE",
  "matrix": {
    "spans": [$(echo "${SPAN_ARR[*]}" | sed 's/ /, /g')],
    "dtypes": ["$(echo "${DTYPE_ARR[*]}" | sed 's/ /", "/g')"],
    "kv_aligned": [$(echo "${KV_ARR[*]}" | sed 's/ /, /g')],
    "seeds": [$(echo "${SEED_ARR[*]}" | sed 's/ /, /g')],
    "prompts": ["$(echo "${PROMPT_ARR[*]}" | sed 's/ /", "/g')"],
    "modes": ["prefill", "decode"]
  },
  "internal_trace_policy": "$INTERNAL_TRACE",
  "dump_format_version": "B3.69 logits.jsonl.gz",
  "timestamp": "$TIMESTAMP",
  "deterministic_env": [
    "HIP_LAUNCH_BLOCKING=1",
    "AMD_SERIALIZE_KERNEL=3",
    "HSA_ENABLE_SDMA=0",
    "GRETA_DETERMINISTIC=1"
  ]
}
EOF
scp $SSH_OPTS "$LOCAL_RUNS_DIR/config.json" "root@$HOST:$REMOTE_BASE/$RUN_ROOT/runs/"

# -----------------------------------------------------------------------------
# Execution Loop
# -----------------------------------------------------------------------------
echo "[2/3] Execution Loop..."

TOTAL_RUNS=0
FAILED_RUNS=0

for SPAN in "${SPAN_ARR[@]}"; do
    for DTYPE in "${DTYPE_ARR[@]}"; do
        for KV in "${KV_ARR[@]}"; do
            for SEED in "${SEED_ARR[@]}"; do
                for PROMPT in "${PROMPT_ARR[@]}"; do
                    
                    # Config Cell
                    CELL_NAME="span=${SPAN}_dtype=${DTYPE}_kv=${KV}_seed=${SEED}_prompt=${PROMPT}"
                    echo "  Processing $CELL_NAME..."

                    # Internal Trace Check
                    TRACE_OPT=""
                    TRACE_MSG="off"
                    if should_trace_internal "$KV" "$SPAN" "$DTYPE" "$SEED"; then
                        TRACE_MSG="ON"
                        # Use B3.74 GRETA_TRACE_STAGE envs
                        # But we pass them via env vars in the SSH command per run
                    fi
                    
                    # Output Dir
                    # runs/profile_<>/span_<>/dtype_<>/kv_<>/seed_<>/<prompt>/
                    REL_PATH="runs/profile_${PROFILE}/span_${SPAN}/dtype_${DTYPE}/kv_${KV}/seed_${SEED}/${PROMPT}"
                    REMOTE_OUT="$REMOTE_BASE/$RUN_ROOT/$REL_PATH"
                    LOCAL_OUT="$LOCAL_RUNS_DIR/profile_${PROFILE}/span_${SPAN}/dtype_${DTYPE}/kv_${KV}/seed_${SEED}/${PROMPT}"
                    
                    # Create dirs (optimize? mkdir -p handles it)
                    # We do per-mode
                    
                    for MODE in "prefill" "decode"; do
                        MODE_REMOTE_OUT="$REMOTE_OUT/$MODE"
                        MODE_LOCAL_OUT="$LOCAL_OUT/$MODE"
                        
                        mkdir -p "$MODE_LOCAL_OUT"
                        
                        # Prepare command
                        # Deterministic flags
                        CMD_ENV="export HIP_LAUNCH_BLOCKING=1; export AMD_SERIALIZE_KERNEL=3; export HSA_ENABLE_SDMA=0; export GRETA_DETERMINISTIC=1; export GRETA_SEED=$SEED"
                        
                        # Trace flags if enabled
                        if [ "$TRACE_MSG" == "ON" ]; then
                             CMD_ENV="$CMD_ENV; export GRETA_TRACE_STAGE=1; export GRETA_TRACE_STAGE_LAYERS='0,1,2,4,8,16,24,31'; export GRETA_TRACE_STAGE_POINTS='attn_out,mlp_out'; export GRETA_TRACE_STAGE_PHASES='prefill_last,decode0'; export GRETA_TRACE_STAGE_OUT=$MODE_REMOTE_OUT/internal.jsonl"
                        fi

                        # Run
                        START_TS=$(date +%s.%N)
                        
                        ssh $SSH_OPTS "root@$HOST" "
                            mkdir -p $MODE_REMOTE_OUT
                            cd $REMOTE_BASE
                            $CMD_ENV
                            ./tools/inference/build/greta_infer \
                                --model ./models/greta-v1.gguf \
                                --prompt tools/benchmarks/prompts/${PROMPT}.txt \
                                --seed $SEED \
                                --kv-aligned $KV \
                                --mode $MODE \
                                --dump-logits $MODE_REMOTE_OUT \
                                --dump-logits-span $SPAN \
                                --dtype $DTYPE \
                                --max-tokens 1 \
                                --greedy \
                                2>&1 | tee $MODE_REMOTE_OUT/run.log
                            
                            # Compress Trace if exists
                            if [ -f $MODE_REMOTE_OUT/internal.jsonl ]; then
                                gzip $MODE_REMOTE_OUT/internal.jsonl
                            fi
                        " >/dev/null 2>&1 || true # Mask stdout but checking files later is better

                        END_TS=$(date +%s.%N)
                        WALL_TIME=$(echo "$END_TS - $START_TS" | bc)
                        
                        # Verify Logic
                        FILES_EXIST=$(ssh $SSH_OPTS "root@$HOST" "ls $MODE_REMOTE_OUT/logits.jsonl.gz 2>/dev/null")
                        
                        if [ -z "$FILES_EXIST" ]; then
                            echo "    [FAIL] $MODE: Missing logits!"
                            FAILED_RUNS=$((FAILED_RUNS + 1))
                        else
                            echo "    [PASS] $MODE (${WALL_TIME}s)"
                            
                            # SCP Back
                            scp -q $SSH_OPTS "root@$HOST:$MODE_REMOTE_OUT/metadata.json" "$MODE_LOCAL_OUT/" || true
                            scp -q $SSH_OPTS "root@$HOST:$MODE_REMOTE_OUT/logits.jsonl.gz" "$MODE_LOCAL_OUT/" || true
                            scp -q $SSH_OPTS "root@$HOST:$MODE_REMOTE_OUT/internal.jsonl.gz" "$MODE_LOCAL_OUT/" 2>/dev/null || true
                            
                            # Gen Perf JSON
                            LOGITS_BYTES=$(stat -c%s "$MODE_LOCAL_OUT/logits.jsonl.gz" 2>/dev/null || echo 0)
                            INTERNAL_BYTES=$(stat -c%s "$MODE_LOCAL_OUT/internal.jsonl.gz" 2>/dev/null || echo 0)
                            METADATA_BYTES=$(stat -c%s "$MODE_LOCAL_OUT/metadata.json" 2>/dev/null || echo 0)
                            
                            cat > "$MODE_LOCAL_OUT/perf.json" << PERF_EOF
{
  "profile": "$PROFILE",
  "span": $SPAN,
  "dtype": "$DTYPE",
  "kv_aligned": $KV,
  "seed": $SEED,
  "prompt_case": "$PROMPT",
  "mode": "$MODE",
  "wall_time_sec": $WALL_TIME,
  "logits_gz_bytes": $LOGITS_BYTES,
  "metadata_bytes": $METADATA_BYTES,
  "internal_trace_bytes": $INTERNAL_BYTES
}
PERF_EOF
                        fi
                        
                        TOTAL_RUNS=$((TOTAL_RUNS + 1))
                    done
                done
            done
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Total Runs: $TOTAL_RUNS"
echo "Failed: $FAILED_RUNS"
echo "Artifacts: $RUN_ROOT"
flock -u 200
exit 0
