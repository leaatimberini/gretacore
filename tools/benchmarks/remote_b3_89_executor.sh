#!/bin/bash
# =============================================================================
# B3.89 Single-Shot Remote Executor
# Usage: ./remote_b3_89_executor.sh <DATE> <VARIANTS> <CONTEXTS> <REPEAT_MAP>
# Example: ./remote_b3_89_executor.sh 2026-02-11 "v3" "4096,8192" "4096:2,8192:1"
#
# Environment:
#   B3_89_MODE=perf|debug  (default: perf)
#     perf  — no serialization env vars; fast runs for real measurements.
#     debug — HIP_LAUNCH_BLOCKING=1, AMD_SERIALIZE_KERNEL=3, HSA_ENABLE_SDMA=0.
#
# Observability:
#   The script emits single-line JSON events to stdout/log for machine parsing:
#     SUITE_START, TEST_START, HEARTBEAT, TEST_END, SUITE_END
#   Human-readable lines are prefixed with "PROGRESS:" and never wrap.
#   Use tools/benchmarks/parse_b3_89_events.py to build a live summary table.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
epoch_s() { date +%s; }

# ---------------------------------------------------------------------------
# JSON event emitter — single line, no unescaped newlines
# ---------------------------------------------------------------------------
emit_event() {
    local event_type="$1"; shift
    # Build JSON fields from key=value pairs passed as arguments
    local json="{"
    json+="\"event\":\"${event_type}\""
    json+=",\"ts\":\"$(ts)\""
    json+=",\"mode\":\"${B3_89_MODE:-perf}\""
    while (( $# )); do
        local kv="$1"; shift
        local key="${kv%%=*}"
        local val="${kv#*=}"
        # Auto-detect numeric vs string
        if [[ "$val" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
            json+=",\"${key}\":${val}"
        else
            # Escape double quotes and backslashes in value
            val="${val//\\/\\\\}"
            val="${val//\"/\\\"}"
            json+=",\"${key}\":\"${val}\""
        fi
    done
    json+="}"
    echo "$json"
}

# ---------------------------------------------------------------------------
# GPU snapshot helper (rocm-smi with fallback)
# ---------------------------------------------------------------------------
gpu_snapshot() {
    local gpu_use="NA" vram="NA"
    if command -v rocm-smi &>/dev/null; then
        gpu_use=$(rocm-smi --showuse 2>/dev/null | grep -oP '\d+%' | head -1 || echo "NA")
        vram=$(rocm-smi --showmeminfo vram 2>/dev/null \
               | grep -i 'used' | head -1 | awk '{print $NF}' || echo "NA")
    fi
    echo "${gpu_use}|${vram}"
}

# ---------------------------------------------------------------------------
# Human-readable progress bar (single line, prefixed PROGRESS:)
# ---------------------------------------------------------------------------
print_progress() {
    local cur=$1 total=$2 label="$3" eta_str="${4:-}"
    local width=30 pct=0
    if (( total > 0 )); then pct=$(( cur * 100 / total )); fi
    local filled=$(( width * cur / (total > 0 ? total : 1) ))
    local empty=$(( width - filled ))
    local bar
    bar="$(printf '#%.0s' $(seq 1 $filled 2>/dev/null))$(printf '.%.0s' $(seq 1 $empty 2>/dev/null))"
    if [ -n "$eta_str" ]; then
        printf 'PROGRESS: [%s] %3d%% (%d/%d) %s | ETA %s\n' "$bar" "$pct" "$cur" "$total" "$label" "$eta_str"
    else
        printf 'PROGRESS: [%s] %3d%% (%d/%d) %s\n' "$bar" "$pct" "$cur" "$total" "$label"
    fi
}

# ---------------------------------------------------------------------------
# Timeout budget per context length
# ---------------------------------------------------------------------------
ctx_timeout_s() {
    case "$1" in
        4096)  echo 10800  ;;  # 3 h
        8192)  echo 21600  ;;  # 6 h
        16384) echo 36000  ;;  # 10 h
        24576) echo 50400  ;;  # 14 h
        32768) echo 64800  ;;  # 18 h
        *)     echo 21600  ;;  # default 6 h
    esac
}

# ---------------------------------------------------------------------------
# ETA tracking
#   _ETA_WALL_SAMPLES[variant:ctx] = "wall1 wall2 wall3"  (space-separated)
# ---------------------------------------------------------------------------
declare -A _ETA_WALL_SAMPLES 2>/dev/null || true

record_wall_sample() {
    local key="$1:$2"  # variant:ctx
    local wall="$3"
    _ETA_WALL_SAMPLES["$key"]="${_ETA_WALL_SAMPLES[$key]:-} $wall"
}

# Compute median of space-separated numbers
_median() {
    local nums
    IFS=' ' read -ra nums <<< "$1"
    local sorted
    sorted=($(printf '%s\n' "${nums[@]}" | sort -n))
    local n=${#sorted[@]}
    if (( n == 0 )); then echo 0; return; fi
    echo "${sorted[$(( n / 2 ))]}"
}

# Estimate remaining seconds for a variant:ctx given completed count & total reps
eta_for_ctx() {
    local variant="$1" ctx="$2" done_count="$3" total_reps="$4"
    local key="${variant}:${ctx}"
    local samples="${_ETA_WALL_SAMPLES[$key]:-}"
    if [ -z "$samples" ]; then echo "-1"; return; fi
    local med
    med=$(_median "$samples")
    local remaining=$(( total_reps - done_count ))
    echo $(( ${med%.*} * remaining ))
}

# ---------------------------------------------------------------------------
# Diagnostics helper — called on non-zero exit or missing PERF_TIMING
# ---------------------------------------------------------------------------
collect_diag() {
    local out_dir=$1 exit_code=$2 run_log=$3
    local diag="$out_dir/diag.txt"
    {
        echo "=== DIAGNOSTICS (exit_code=${exit_code}) at $(ts) ==="
        echo ""
        echo "--- dmesg (OOM / killed) ---"
        dmesg -T 2>/dev/null | tail -n 200 | grep -Ei "out of memory|oom|killed process|cgroup" || echo "(none found)"
        echo ""
        echo "--- free -h ---"
        free -h 2>/dev/null || echo "(free not available)"
        echo ""
        echo "--- rocm-smi snapshot ---"
        if command -v rocm-smi &>/dev/null; then
            rocm-smi --showuse --showmeminfo vram --showpower --showclocks --showtemp 2>/dev/null \
                | sed -n '1,200p' || echo "(rocm-smi failed)"
        else
            echo "(rocm-smi not available)"
        fi
        echo ""
        echo "--- run.log last 80 lines ---"
        if [ -f "$run_log" ]; then
            tail -n 80 "$run_log"
        else
            echo "(run.log not found)"
        fi
    } > "$diag" 2>&1
    echo "$(ts) DIAG written to $diag"
}

# ---------------------------------------------------------------------------
# run_with_heartbeat — runs CMD in background, emits HEARTBEAT JSON every 60s
#   Usage: run_with_heartbeat <timeout_s> <log_file> <cmd...>
#   The command's stdout/stderr go to <log_file>.
#   Heartbeats and progress go to the caller's stdout (the main log).
# ---------------------------------------------------------------------------
run_with_heartbeat() {
    local timeout_s=$1; shift
    local log_file=$1; shift
    # Run command in background, redirect only its output to log_file
    "$@" > "$log_file" 2>&1 &
    local bg_pid=$!
    local start_epoch
    start_epoch=$(epoch_s)
    local last_hb=0

    while kill -0 "$bg_pid" 2>/dev/null; do
        sleep 5
        local now_epoch
        now_epoch=$(epoch_s)
        local elapsed=$(( now_epoch - start_epoch ))

        # Emit heartbeat every ~60s
        if (( elapsed - last_hb >= 60 )); then
            last_hb=$elapsed
            local snap
            snap=$(gpu_snapshot)
            local gpu_use="${snap%%|*}"
            local vram="${snap##*|}"

            local est_pct=0
            if (( timeout_s > 0 )); then
                est_pct=$(( elapsed * 100 / timeout_s ))
                if (( est_pct > 99 )); then est_pct=99; fi
            fi

            # Per-ctx ETA
            local ctx_eta=-1
            local _reps_total
            _reps_total=$(get_reps "${_RWP_CTX:-0}")
            ctx_eta=$(eta_for_ctx "${_RWP_VARIANT:-?}" "${_RWP_CTX:-0}" "${_RWP_DONE_THIS_CTX:-0}" "$_reps_total")

            # Suite ETA = remaining tests * median of this ctx (rough)
            local suite_eta=-1
            if (( ctx_eta >= 0 )); then
                local remaining_suite=$(( TOTAL_TESTS - TEST_INDEX ))
                suite_eta=$(( ctx_eta + (remaining_suite - (_reps_total - ${_RWP_DONE_THIS_CTX:-0})) * ${ctx_eta:-0} / (_reps_total - ${_RWP_DONE_THIS_CTX:-0} > 0 ? _reps_total - ${_RWP_DONE_THIS_CTX:-0} : 1) ))
            fi

            emit_event "HEARTBEAT" \
                "variant=${_RWP_VARIANT:-?}" \
                "ctx=${_RWP_CTX:-0}" \
                "run_idx=${_RWP_RUN:-0}" \
                "test_index=${TEST_INDEX}" \
                "total_tests=${TOTAL_TESTS}" \
                "pid=${bg_pid}" \
                "elapsed_s=${elapsed}" \
                "timeout_s=${timeout_s}" \
                "est_pct=${est_pct}" \
                "gpu_use=${gpu_use}" \
                "vram_used=${vram}" \
                "eta_remaining_s=${ctx_eta}" \
                "suite_eta_s=${suite_eta}"

            # Human-readable line
            local eta_label=""
            if (( ctx_eta >= 0 )); then
                eta_label="ctx:$(( ctx_eta / 60 ))m suite:$(( suite_eta / 60 ))m"
            fi
            print_progress "$TEST_INDEX" "$TOTAL_TESTS" \
                "variant=${_RWP_VARIANT:-?} ctx=${_RWP_CTX:-?} run=${_RWP_RUN:-?} elapsed=${elapsed}s ${est_pct}%" \
                "$eta_label"
        fi
    done
    wait "$bg_pid" 2>/dev/null
    return $?
}

# ===========================================================================
# Global suite progress counters
# ===========================================================================
TOTAL_TESTS=0
TEST_INDEX=0
SUITE_START_EPOCH=0

DATE=$1
VARIANTS=$2  # Comma separated: v3,v4,baseline
CONTEXTS=$3  # Comma separated: 4096,8192,...
REPEAT_MAP=$4 # Format: "ctx:rep,ctx:rep" e.g. "4096:2,8192:1"

# ---------------------------------------------------------------------------
# MODE switch: perf (default) vs debug
# ---------------------------------------------------------------------------
B3_89_MODE="${B3_89_MODE:-perf}"
if [[ "$B3_89_MODE" != "perf" && "$B3_89_MODE" != "debug" ]]; then
    echo "ERROR: B3_89_MODE must be 'perf' or 'debug' (got '$B3_89_MODE')"
    exit 1
fi

# Setup environment
export RUN_ROOT="artifacts_remote/$DATE/b3_89"
mkdir -p "$RUN_ROOT"
cd /root/gretacore

# Always set verbose info
export GRETA_VERBOSE_INFO=1

if [ "$B3_89_MODE" = "debug" ]; then
    export HIP_LAUNCH_BLOCKING=1
    export AMD_SERIALIZE_KERNEL=3
    export HSA_ENABLE_SDMA=0
else
    unset HIP_LAUNCH_BLOCKING 2>/dev/null || true
    unset AMD_SERIALIZE_KERNEL 2>/dev/null || true
    unset HSA_ENABLE_SDMA 2>/dev/null || true
fi

# VRAM sampling interval: 1s for debug, 10s for perf
VRAM_SAMPLE_INTERVAL=10
if [ "$B3_89_MODE" = "debug" ]; then
    VRAM_SAMPLE_INTERVAL=1
fi

# ---------------------------------------------------------------------------
# Startup Banner
# ---------------------------------------------------------------------------
echo "================================================================="
echo " B3.89 Remote Executor — $(ts)"
echo "================================================================="
echo " MODE            : $B3_89_MODE"
echo " DATE            : $DATE"
echo " VARIANTS        : $VARIANTS"
echo " CONTEXTS        : $CONTEXTS"
echo " REPEAT_MAP      : $REPEAT_MAP"
echo " RUN_ROOT        : $RUN_ROOT"
echo " GRETA_VERBOSE   : ${GRETA_VERBOSE_INFO:-0}"
echo " HIP_LAUNCH_BLK  : ${HIP_LAUNCH_BLOCKING:-<unset>}"
echo " AMD_SERIALIZE   : ${AMD_SERIALIZE_KERNEL:-<unset>}"
echo " HSA_ENABLE_SDMA : ${HSA_ENABLE_SDMA:-<unset>}"
echo " VRAM_SAMPLE_INT : ${VRAM_SAMPLE_INTERVAL}s"
echo "-----------------------------------------------------------------"
echo " Host RAM        : $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo 'NA')"
if command -v rocm-smi &>/dev/null; then
    echo " GPU VRAM        : $(rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'total' | head -1 | awk '{print $NF}' || echo 'NA')"
else
    echo " GPU VRAM        : NA (rocm-smi not found)"
fi
echo "================================================================="

# Helper to get repetition count
get_reps() {
    local ctx=$1
    local reps=1
    IFS=',' read -ra PAIRS <<< "$REPEAT_MAP"
    for pair in "${PAIRS[@]}"; do
        if [[ "$pair" == "$ctx:"* ]]; then
            reps=${pair#*:}
        fi
    done
    echo $reps
}

# Build and Run for each variant
# Force 32k context in model config
echo "Patching model_config.hpp for 32k context..."
CONFIG_H="src/inference/include/gcore/inference/model_config.hpp"
if [ ! -f "$CONFIG_H" ]; then
    echo "ERROR: $CONFIG_H not found! Cannot patch context length."
    exit 1
fi

# Detect original value
ORIG_LINE=$(grep "max_seq_len =" "$CONFIG_H" | head -1)
# Extract number from '    cfg.max_seq_len = 4096;'
ORIG_VAL=$(echo "$ORIG_LINE" | sed -r 's/.*max_seq_len = ([0-9]+);.*/\1/')

# Backup and Trap setup
CONFIG_BACKUP=$(mktemp)
cp "$CONFIG_H" "$CONFIG_BACKUP"
ORIG_SHA=$(sha256sum "$CONFIG_H" | awk '{print $1}')

restore_config() {
    if [ -f "$CONFIG_BACKUP" ]; then
        mv "$CONFIG_BACKUP" "$CONFIG_H"
        RESTORED_SHA=$(sha256sum "$CONFIG_H" | awk '{print $1}')
        
        emit_event "CONFIG_RESTORE" \
            "path=$CONFIG_H" \
            "sha256_restored=$RESTORED_SHA"
            
        echo "Restored original model_config.hpp."
    fi
}
trap restore_config EXIT

# Apply Patch
sed -i 's/max_seq_len = [0-9]\+/max_seq_len = 32768/g' "$CONFIG_H"
NEW_SHA=$(sha256sum "$CONFIG_H" | awk '{print $1}')

if grep -q "max_seq_len = 32768" "$CONFIG_H"; then
    echo "SUCCESS: model_config.hpp patched to 32768."
    emit_event "CONFIG_PATCH" \
        "path=$CONFIG_H" \
        "from=${ORIG_VAL:-unknown}" \
        "to=32768" \
        "sha256_before=$ORIG_SHA" \
        "sha256_after=$NEW_SHA"
else
    echo "ERROR: Failed to patch model_config.hpp!"
    exit 1
fi

IFS=',' read -ra VAR_LIST <<< "$VARIANTS"

# ---------------------------------------------------------------------------
# Compute TOTAL_TESTS upfront for all variants × contexts × reps
# ---------------------------------------------------------------------------
IFS=',' read -ra _ALL_CTX <<< "$CONTEXTS"
for _v in "${VAR_LIST[@]}"; do
    for _tc in "${_ALL_CTX[@]}"; do
        TOTAL_TESTS=$(( TOTAL_TESTS + $(get_reps "$_tc") ))
    done
done
SUITE_START_EPOCH=$(epoch_s)
emit_event "SUITE_START" \
    "total_tests=${TOTAL_TESTS}" \
    "variants=${VARIANTS}" \
    "contexts=${CONTEXTS}" \
    "repeat_map=${REPEAT_MAP}"
print_progress 0 "$TOTAL_TESTS" "suite starting"

for VARIANT in "${VAR_LIST[@]}"; do
    echo "=== Processing Variant: $VARIANT ==="

    BUILD_DIR="tools/inference/build_$VARIANT"
    mkdir -p "$BUILD_DIR"

    # Configure CMake flags
    CMAKE_FLAGS=""
    if [ "$VARIANT" == "v3" ]; then
        CMAKE_FLAGS="-DGRETA_PREFILL_Q_LDS=1"
    elif [ "$VARIANT" == "v4" ]; then
        CMAKE_FLAGS="-DGRETA_PREFILL_Q_LDS_V4=1"
    elif [ "$VARIANT" == "baseline" ]; then
        CMAKE_FLAGS="" # No special flags
    fi

    # Build — always rebuild to ensure fresh binaries
    echo "Building $VARIANT in $BUILD_DIR..."
    pushd "$BUILD_DIR" > /dev/null
    rm -f CMakeCache.txt build_passed.flag
    cmake .. $CMAKE_FLAGS > build.log 2>&1
    if make -j$(nproc) >> build.log 2>&1; then
        touch build_passed.flag
        if [ ! -f "greta_infer" ]; then
            echo "ERROR: Binary greta_infer not found after successful make!"
            popd > /dev/null
            exit 1
        fi
    else
        echo "BUILD FAILED for $VARIANT. Check build.log"
        cat build.log
        popd > /dev/null
        continue
    fi
    popd > /dev/null

    # Resource Dump & Gate (Only for opt variants)
    if [ "$VARIANT" != "baseline" ]; then
        if [ -f "$RUN_ROOT/${VARIANT}_gate_passed.flag" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "Gate for $VARIANT already passed, skipping..."
        else
            echo "Running No-Spill Gate for $VARIANT..."
            chmod +x tools/benchmarks/b3_89_no_spill_gate.sh
            if tools/benchmarks/b3_89_no_spill_gate.sh "$CMAKE_FLAGS" > "$RUN_ROOT/${VARIANT}_gate.log" 2>&1; then
                touch "$RUN_ROOT/${VARIANT}_gate_passed.flag"
            else
                echo "GATE FAILED for $VARIANT. Skipping execution."
                cat "$RUN_ROOT/${VARIANT}_gate.log"
                continue
            fi

            # Save resource dump
            tools/benchmarks/b3_89_dump_kernel_resources.sh "$CMAKE_FLAGS" > "$RUN_ROOT/${VARIANT}_resources.txt"
        fi
    fi

    # Execute Contexts
    IFS=',' read -ra CTX_LIST <<< "$CONTEXTS"

    for CTX in "${CTX_LIST[@]}"; do
        REPS=$(get_reps "$CTX")
        echo "Running $VARIANT at Context $CTX for $REPS repetitions..."

        # Generate Prompt
        python3 -c "print('a' * ($CTX - 1))" > /tmp/prompt.txt

        # Set GRETA_MAX_SEQ_LEN to CTX+2 to account for tokenization overhead
        export GRETA_MAX_SEQ_LEN=$((CTX + 2))
        echo "GRETA_MAX_SEQ_LEN=$GRETA_MAX_SEQ_LEN"

        # Track how many runs of this ctx completed (for ETA)
        _DONE_THIS_CTX=0

        for i in $(seq 0 $((REPS-1))); do
            OUT_DIR="$RUN_ROOT/${VARIANT}/ctx_${CTX}_run${i}"
            if [ -f "$OUT_DIR/perf.json" ] && grep -q '"exit_status": "OK"' "$OUT_DIR/perf.json" && [ "${FORCE:-0}" != "1" ]; then
                echo "  Run $i already exists, skipping..."
                TEST_INDEX=$(( TEST_INDEX + 1 ))
                emit_event "TEST_SKIP" \
                    "variant=${VARIANT}" \
                    "ctx=${CTX}" \
                    "run_idx=${i}" \
                    "test_index=${TEST_INDEX}" \
                    "total_tests=${TOTAL_TESTS}" \
                    "reason=perf.json exists (OK)"
                
                print_progress "$TEST_INDEX" "$TOTAL_TESTS" \
                    "SKIP variant=${VARIANT} ctx=${CTX} run=${i} reason=perf.json exists (OK)"

                _DONE_THIS_CTX=$(( _DONE_THIS_CTX + 1 ))
                continue
            fi
            mkdir -p "$OUT_DIR"

            echo "  Run $i (including VRAM sampling)..."

            # Start VRAM sampling in background (interval depends on mode)
            rocm-smi --showmeminfo vram --json > "$OUT_DIR/vram_before.json" 2>&1 || true
            (
                while true; do
                    rocm-smi --showmeminfo vram --json >> "$OUT_DIR/vram_samples.jsonl" 2>&1
                    echo "" >> "$OUT_DIR/vram_samples.jsonl"
                    sleep "$VRAM_SAMPLE_INTERVAL"
                done
            ) &
            SMI_PID=$!

            CTX_TIMEOUT_S="$(ctx_timeout_s "$CTX")"

            # --- TEST_START event ---
            emit_event "TEST_START" \
                "variant=${VARIANT}" \
                "ctx=${CTX}" \
                "run_idx=${i}" \
                "test_index=${TEST_INDEX}" \
                "total_tests=${TOTAL_TESTS}" \
                "timeout_s=${CTX_TIMEOUT_S}"
            print_progress "$TEST_INDEX" "$TOTAL_TESTS" \
                "START variant=${VARIANT} ctx=${CTX} run=${i} budget=${CTX_TIMEOUT_S}s"

            # Export context for heartbeat helper
            export _RWP_VARIANT="$VARIANT" _RWP_CTX="$CTX" _RWP_RUN="$i"
            export _RWP_DONE_THIS_CTX="$_DONE_THIS_CTX"
            set +e
            START=$(date +%s.%N)
            run_with_heartbeat "$CTX_TIMEOUT_S" "$OUT_DIR/run.log" \
                timeout -k 30s "${CTX_TIMEOUT_S}s" \
                ./$BUILD_DIR/greta_infer \
                    --model ./models/greta-v1.gguf \
                    --prompt-file /tmp/prompt.txt \
                    --max-tokens 1 \
                    --greedy
            EXIT_STATUS=$?
            set -e
            END=$(date +%s.%N)

            kill $SMI_PID 2>/dev/null || true
            wait $SMI_PID 2>/dev/null || true
            rocm-smi --showmeminfo vram --json > "$OUT_DIR/vram_after.json" 2>&1 || true

            WALL=$(echo "$END - $START" | bc)

            STATUS_STR="OK"
            if [ $EXIT_STATUS -ne 0 ]; then
                STATUS_STR="FAIL"
                echo "CRITICAL: Run $i crashed (Exit $EXIT_STATUS)"
                tail -n 30 "$OUT_DIR/run.log"
                collect_diag "$OUT_DIR" "$EXIT_STATUS" "$OUT_DIR/run.log"
            fi

            # Check for prompt tokens in log
            PROMPT_TOKENS=$(grep "Prompt tokens:" "$OUT_DIR/run.log" | sed 's/.*Prompt tokens: //' | sed 's/ .*//' || echo "NOT_FOUND")
            echo "  Prompt tokens: $PROMPT_TOKENS"

            TIMINGS=$(grep "\[PERF_TIMING\]" "$OUT_DIR/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")

            # Extract individual timing fields for TEST_END event
            _prefill_s=""
            _attn_impl=""
            _model_load_s=""
            _decode_s=""
            if [ -n "$TIMINGS" ] && [ "$TIMINGS" != "{}" ]; then
                _prefill_s=$(echo "$TIMINGS" | grep -oP '"prefill_s":\K[0-9.e+-]+' || true)
                _attn_impl=$(echo "$TIMINGS" | grep -oP '"attn_impl":"\K[^"]+' || true)
                _model_load_s=$(echo "$TIMINGS" | grep -oP '"model_load_s":\K[0-9.e+-]+' || true)
                _decode_s=$(echo "$TIMINGS" | grep -oP '"decode_s":\K[0-9.e+-]+' || true)
            fi

            # Guardrail: Check for silent failure
            if [ -z "$TIMINGS" ] || [ "$TIMINGS" == "{}" ]; then
                 echo "CRITICAL: Missing performance timings in run log!"
                 STATUS_STR="FAIL"
                 if [ ! -f "$OUT_DIR/diag.txt" ]; then
                     collect_diag "$OUT_DIR" "$EXIT_STATUS" "$OUT_DIR/run.log"
                 fi
            elif echo "$TIMINGS" | grep -q '"prefill_s":0'; then
                 echo "CRITICAL: Silent failure detected (prefill_s: 0)!"
                 echo "--- RUN LOG TAIL ---"
                 tail -n 20 "$OUT_DIR/run.log"
                 STATUS_STR="FAIL"
            fi

            cat > "$OUT_DIR/perf.json" << EOT
{
  "ticket": "b3_89",
  "variant": "$VARIANT",
  "context_len": $CTX,
  "repetition": $i,
  "wall_time_sec": $WALL,
  "exit_status": "$STATUS_STR",
  "mode": "$B3_89_MODE",
  "timings": $TIMINGS
}
EOT

            # Record wall sample for ETA (only OK runs)
            if [ "$STATUS_STR" = "OK" ]; then
                record_wall_sample "$VARIANT" "$CTX" "${WALL%.*}"
            fi
            _DONE_THIS_CTX=$(( _DONE_THIS_CTX + 1 ))

            # --- TEST_END event ---
            TEST_INDEX=$(( TEST_INDEX + 1 ))
            _suite_elapsed=$(( $(epoch_s) - SUITE_START_EPOCH ))
            emit_event "TEST_END" \
                "variant=${VARIANT}" \
                "ctx=${CTX}" \
                "run_idx=${i}" \
                "test_index=${TEST_INDEX}" \
                "total_tests=${TOTAL_TESTS}" \
                "exit_code=${EXIT_STATUS}" \
                "exit_status=${STATUS_STR}" \
                "wall_s=${WALL}" \
                "prefill_s=${_prefill_s:-NA}" \
                "attn_impl=${_attn_impl:-NA}" \
                "model_load_s=${_model_load_s:-NA}" \
                "decode_s=${_decode_s:-NA}" \
                "suite_elapsed_s=${_suite_elapsed}"

            # Compute suite-level ETA for progress line
            _avg_per_test=0; _suite_eta_str=""
            if (( TEST_INDEX > 0 )); then
                _avg_per_test=$(( _suite_elapsed / TEST_INDEX ))
                _remaining_tests=$(( TOTAL_TESTS - TEST_INDEX ))
                _suite_eta_s=$(( _avg_per_test * _remaining_tests ))
                _suite_eta_str="$(( _suite_eta_s / 60 ))m"
            fi
            print_progress "$TEST_INDEX" "$TOTAL_TESTS" \
                "DONE variant=${VARIANT} ctx=${CTX} run=${i} exit=${EXIT_STATUS} wall=${WALL}s" \
                "$_suite_eta_str"
        done
    done
done

# ---------------------------------------------------------------------------
# SUITE_END event
# ---------------------------------------------------------------------------
_SUITE_WALL=$(( $(epoch_s) - SUITE_START_EPOCH ))
emit_event "SUITE_END" \
    "total_tests=${TOTAL_TESTS}" \
    "completed=${TEST_INDEX}" \
    "suite_wall_s=${_SUITE_WALL}"
print_progress "$TEST_INDEX" "$TOTAL_TESTS" "SUITE COMPLETE in $(( _SUITE_WALL / 60 ))m${_SUITE_WALL}s"

# Generate Summary JSON
echo "Generating summary.json..."
python3 - << 'EOF' > "$RUN_ROOT/summary.json"
import os, json, math

run_root = os.environ.get('RUN_ROOT', '.')
results = {}

# Baseline constants from B3.89 specs
baseline_ref = {4096: 22.768, 8192: 114.265, 16384: 469.749}

if not os.path.exists(run_root):
    print(json.dumps({"error": "run_root not found"}))
    exit(0)

variants = [d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d)) and d != "runs"]

for var in variants:
    results[var] = {}
    var_dir = os.path.join(run_root, var)
    for ctx_dir in os.listdir(var_dir):
        if not ctx_dir.startswith('ctx_'): continue
        try:
            parts = ctx_dir.split('_')
            ctx = int(parts[1])
        except: continue

        perf_path = os.path.join(var_dir, ctx_dir, 'perf.json')
        if os.path.exists(perf_path):
            with open(perf_path, 'r') as f:
                data = json.load(f)

            if ctx not in results[var]: results[var][ctx] = []
            results[var][ctx].append(data)

summary = {"variants": {}, "baseline": baseline_ref, "gate_status": {}, "resources": {}}

# Extract resources
for var in ["v3", "v4"]:
    res_path = os.path.join(run_root, f"{var}_resources.txt")
    if os.path.exists(res_path):
        summary["resources"][var] = {}
        with open(res_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "VGPRs:" in line: summary["resources"][var]["vgpr"] = line.split(":")[1].strip()
                if "SGPRs:" in line: summary["resources"][var]["sgpr"] = line.split(":")[1].strip()
                if "LDS:" in line: summary["resources"][var]["lds_bytes"] = line.split(":")[1].strip()
                if "Scratch:" in line: summary["resources"][var]["scratch_bytes"] = line.split(":")[1].replace("bytes", "").strip()

# Check gate status
for var in ["v3", "v4"]:
    gate_log = os.path.join(run_root, f"{var}_gate.log")
    if os.path.exists(gate_log):
        with open(gate_log, 'r') as f:
            content = f.read()
            if "GATE PASSED" in content:
                summary["gate_status"][var] = "PASS"
            else:
                summary["gate_status"][var] = "FAIL"

for var, contexts in results.items():
    summary["variants"][var] = {}
    for ctx, runs in contexts.items():
        # Sort and discard warmup
        runs.sort(key=lambda x: x.get('repetition', 0))
        if len(runs) > 1: runs = runs[1:]

        prefills = []
        peak_vrams = []
        for r in runs:
            t = r.get("timings", {})
            if "prefill_s" in t:
                prefills.append(t["prefill_s"])
            else:
                prefills.append(r.get("wall_time_sec", 0))

            # Simple peak VRAM heuristic: check vram_after.json
            rep_idx = r.get('repetition', 0)
            ctx_run_dir = os.path.join(run_root, var, f"ctx_{ctx}_run{rep_idx}")
            vram_after = os.path.join(ctx_run_dir, "vram_after.json")
            if os.path.exists(vram_after):
                try:
                    with open(vram_after, 'r') as vf:
                        vjson = json.load(vf)
                        for key in vjson:
                            if "VRAM" in vjson[key]:
                                peak_vrams.append(vjson[key]["VRAM Used"])
                except: pass

        if not prefills: continue

        data_pts = prefills[1:] if len(prefills) > 1 else prefills
        median_p = sorted(data_pts)[len(data_pts)//2]

        summary["variants"][var][ctx] = {
            "prefill_median_s": median_p,
            "speedup": baseline_ref.get(ctx, 0) / median_p if median_p > 0 else 0,
            "runs": len(prefills)
        }
        if peak_vrams:
            summary["variants"][var][ctx]["peak_vram"] = max(peak_vrams)

# Scaling estimate for v3
if "v3" in summary["variants"]:
    v3_data = summary["variants"]["v3"]
    sorted_ctx = sorted(v3_data.keys())
    if len(sorted_ctx) >= 2:
        c1, c2 = sorted_ctx[0], sorted_ctx[-1]
        t1, t2 = v3_data[c1]["prefill_median_s"], v3_data[c2]["prefill_median_s"]
        if t1 > 0 and t2 > 0:
            slope = (math.log(t2) - math.log(t1)) / (math.log(c2) - math.log(c1))
            summary["v3_scaling_exponent"] = round(slope, 3)

print(json.dumps(summary, indent=2))
EOF

echo "DONE_REMOTE_B3_89_EXECUTOR"
