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
# Force 32k context in model config
echo "Patching model_config.hpp for 32k context..."
CONFIG_H="src/inference/include/gcore/inference/model_config.hpp"
if [ ! -f "$CONFIG_H" ]; then
    echo "ERROR: $CONFIG_H not found! Cannot patch context length."
    exit 1
fi

sed -i 's/max_seq_len = [0-9]\+/max_seq_len = 32768/g' "$CONFIG_H"

if grep -q "max_seq_len = 32768" "$CONFIG_H"; then
    echo "SUCCESS: model_config.hpp patched to 32768."
else
    echo "ERROR: Failed to patch model_config.hpp!"
    grep "max_seq_len =" "$CONFIG_H"
    exit 1
fi

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
    # Always rebuild to ensure fresh binaries
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
        # The tokenizer produces CTX+2 tokens for (CTX-1) chars with v3/v4 attention
        export GRETA_MAX_SEQ_LEN=$((CTX + 2))
        echo "GRETA_MAX_SEQ_LEN=$GRETA_MAX_SEQ_LEN"
        
        for i in $(seq 0 $((REPS-1))); do
            OUT_DIR="$RUN_ROOT/${VARIANT}/ctx_${CTX}_run${i}"
            if [ -f "$OUT_DIR/perf.json" ] && grep -q '"exit_status": "OK"' "$OUT_DIR/perf.json" && [ "${FORCE:-0}" != "1" ]; then
                echo "  Run $i already exists, skipping..."
                continue
            fi
            mkdir -p "$OUT_DIR"
            
            echo "  Run $i (including VRAM sampling)..."
            
            # Start VRAM sampling in background
            rocm-smi --showmeminfo vram --json > "$OUT_DIR/vram_before.json" 2>&1 || true
            (
                while true; do
                    rocm-smi --showmeminfo vram --json >> "$OUT_DIR/vram_samples.jsonl" 2>&1
                    echo "" >> "$OUT_DIR/vram_samples.jsonl"
                    sleep 1
                done
            ) &
            SMI_PID=$!
            
            # Allow failure to capture log
            set +e
            START=$(date +%s.%N)
            ./$BUILD_DIR/greta_infer \
                --model ./models/greta-v1.gguf \
                --prompt-file /tmp/prompt.txt \
                --max-tokens 1 \
                --greedy > "$OUT_DIR/run.log" 2>&1
            EXIT_STATUS=$?
            set -e
            END=$(date +%s.%N)
            
            kill $SMI_PID || true
            rocm-smi --showmeminfo vram --json > "$OUT_DIR/vram_after.json" 2>&1 || true
            
            WALL=$(echo "$(date +%s.%N) - $START" | bc)
            
            STATUS_STR="OK"
            if [ $EXIT_STATUS -ne 0 ]; then 
                STATUS_STR="FAIL"
                echo "CRITICAL: Run $i crashed (Exit $EXIT_STATUS)"
                tail -n 30 "$OUT_DIR/run.log"
            fi
            
            # Check for prompt tokens in log
            PROMPT_TOKENS=$(grep "Prompt tokens:" "$OUT_DIR/run.log" | sed 's/.*Prompt tokens: //' | sed 's/ .*//' || echo "NOT_FOUND")
            echo "  Prompt tokens: $PROMPT_TOKENS"
            
            TIMINGS=$(grep "\[PERF_TIMING\]" "$OUT_DIR/run.log" | sed 's/\[PERF_TIMING\] //' || echo "{}")
            
            # Guardrail: Check for silent failure (0 prefill or missing json)
            if [ -z "$TIMINGS" ] || [ "$TIMINGS" == "{}" ]; then
                 echo "CRITICAL: Missing performance timings in run log!"
                 STATUS_STR="FAIL"
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
                        # SMI json structure varies, usually card_0 or similar
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
