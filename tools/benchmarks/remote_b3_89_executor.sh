#!/bin/bash
# =============================================================================
# B3.89 Single-Shot Remote Executor
# Usage: ./remote_b3_89_executor.sh <DATE> <VARIANTS> <CONTEXTS> <REPEAT_MAP>
# Example: ./remote_b3_89_executor.sh 2026-02-11 "v3" "4096,8192" "4096:2,8192:1"
# =============================================================================
set -euo pipefail

DATE=$1
VARIANTS=$2  # Comma separated: v3,v4,baseline
CONTEXTS=$3  # Comma separated: 4096,8192,...
REPEAT_MAP=$4 # Format: "ctx:rep,ctx:rep" e.g. "4096:2,8192:1"

# Setup environment
export RUN_ROOT="artifacts_remote/$DATE/b3_89"
mkdir -p "$RUN_ROOT"
cd /root/gretacore

# Common env vars for determinism
export GRETA_VERBOSE_INFO=1
export HIP_LAUNCH_BLOCKING=1
export AMD_SERIALIZE_KERNEL=3
export HSA_ENABLE_SDMA=0

# Helper to get repetition count
get_reps() {
    local ctx=$1
    local reps=1
    # Parse REPEAT_MAP
    IFS=',' read -ra PAIRS <<< "$REPEAT_MAP"
    for pair in "${PAIRS[@]}"; do
        if [[ "$pair" == "$ctx:"* ]]; then
            reps=${pair#*:}
        fi
    done
    echo $reps
}

# Build and Run for each variant
IFS=',' read -ra VAR_LIST <<< "$VARIANTS"
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
    
    # Build
    echo "Building $VARIANT in $BUILD_DIR..."
    pushd "$BUILD_DIR" > /dev/null
    rm -f CMakeCache.txt
    cmake .. $CMAKE_FLAGS > build.log 2>&1
    make -j$(nproc) >> build.log 2>&1
    if [ $? -ne 0 ]; then
        echo "BUILD FAILED for $VARIANT. Check build.log"
        cat build.log
        popd > /dev/null
        continue
    fi
    popd > /dev/null
    
    # Resource Dump & Gate (Only for opt variants)
    if [ "$VARIANT" != "baseline" ]; then
        echo "Running No-Spill Gate for $VARIANT..."
        chmod +x tools/benchmarks/b3_89_no_spill_gate.sh
        tools/benchmarks/b3_89_no_spill_gate.sh "$CMAKE_FLAGS" > "$RUN_ROOT/${VARIANT}_gate.log" 2>&1
        EXIT_CODE=$?
        cat "$RUN_ROOT/${VARIANT}_gate.log"
        if [ $EXIT_CODE -ne 0 ]; then
            echo "GATE FAILED for $VARIANT. Skipping execution."
            continue
        fi
        
        # Save resource dump
        tools/benchmarks/b3_89_dump_kernel_resources.sh "$CMAKE_FLAGS" > "$RUN_ROOT/${VARIANT}_resources.txt"
    fi
    
    # Execute Contexts
    IFS=',' read -ra CTX_LIST <<< "$CONTEXTS"
    for CTX in "${CTX_LIST[@]}"; do
        REPS=$(get_reps "$CTX")
        echo "Running $VARIANT at Context $CTX for $REPS repetitions..."
        
        # Generate Prompt
        python3 -c "print('a' * ($CTX - 1))" > /tmp/prompt.txt
        
        for i in $(seq 0 $((REPS-1))); do
            OUT_DIR="$RUN_ROOT/${VARIANT}/ctx_${CTX}_run${i}"
            mkdir -p "$OUT_DIR"
            
            echo "  Run $i..."
            START=$(date +%s.%N)
            ./$BUILD_DIR/greta_infer \
                --model ./models/greta-v1.gguf \
                --prompt-file /tmp/prompt.txt \
                --max-tokens 1 \
                --greedy > "$OUT_DIR/run.log" 2>&1
            EXIT_STATUS=$?
            END=$(date +%s.%N)
            WALL=$(echo "$END - $START" | bc)
            
            STATUS_STR="OK"
            if [ $EXIT_STATUS -ne 0 ]; then STATUS_STR="FAIL"; fi
            
            TIMINGS=$(grep "\[PERF_TIMING\]" "$OUT_DIR/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")
            
            cat > "$OUT_DIR/perf.json" << EOT
{
  "ticket": "b3_89",
  "variant": "$VARIANT",
  "context_len": $CTX,
  "repetition": $i,
  "wall_time_sec": $WALL,
  "exit_status": "$STATUS_STR",
  "timings": $TIMINGS
}
EOT
        done
    done
done

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

variants = [d for d in os.listdir(run_root) if os.path.isdir(os.path.join(run_root, d))]

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

summary = {"variants": {}, "baseline": baseline_ref, "gate_status": {}}

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
        # Get prefill times. The script outputs 'prefill_s' in perf.json timings
        prefills = []
        for r in runs:
            t = r.get("timings", {})
            if "prefill_s" in t:
                prefills.append(t["prefill_s"])
            else:
                # Fallback to wall_time if perf_timing missing
                prefills.append(r.get("wall_time_sec", 0))
        
        if not prefills: continue
        
        # Repetition policy: skip first if multiple
        data_pts = prefills[1:] if len(prefills) > 1 else prefills
        median_p = sorted(data_pts)[len(data_pts)//2]
        
        summary["variants"][var][ctx] = {
            "prefill_median_s": median_p,
            "speedup": baseline_ref.get(ctx, 0) / median_p if median_p > 0 else 0,
            "runs": len(prefills)
        }

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
