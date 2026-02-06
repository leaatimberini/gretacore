#!/bin/bash
# run_b3_61_mi300x.sh
# Residual Stream Bisect Execution (B3.61)
# Identifies first divergent tensor in residual stream causing inference collapse
# at extended context lengths (~826 and ~1652 positions)

set -e

# Canonical Config
DATE=$(date +%Y-%m-%d)
B3_ID="b3_61"
OUT_DIR="artifacts_remote/$DATE/$B3_ID"
mkdir -p "$OUT_DIR/run" "$OUT_DIR/traces" "$OUT_DIR/checkpoints"

# Prompts for validation
PROMPTS=(
    "tools/benchmarks/prompts/p0_short.txt"
    "tools/benchmarks/prompts/p6_len_16.txt"
    "tools/benchmarks/prompts/p6_len_32.txt"
)

# Target layers for comprehensive coverage
LAYERS=("0" "1" "2" "4" "8")

# Build Target
BINARY="./tools/inference/build/greta_infer"
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary $BINARY not found. Build it in tools/inference/build first."
    exit 1
fi

echo "=== B3.61 Residual Stream Bisect Execution ==="
echo "Date: $DATE"
echo "Output: $OUT_DIR"
echo "Binary: $BINARY"

# Pre-Execution Synchronization
echo "--- Pre-Execution Synchronization ---"

# Verify local repository is clean
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: Local repository has uncommitted changes"
    git status
    exit 1
fi

# Get current commit hash
LOCAL_COMMIT=$(git rev-parse HEAD)
echo "Local commit: $LOCAL_COMMIT"

# Fetch and merge latest from origin/main
git fetch origin main 2>/dev/null || git fetch origin main 2>/dev/null || echo "Warning: Could not fetch from origin"
git merge origin/main --ff-only 2>/dev/null || echo "Warning: Could not merge origin/main"

# Record final commit hash
FINAL_COMMIT=$(git rev-parse HEAD)
echo "Final local commit: $FINAL_COMMIT"

# B3.61 Environment Variables for Residual Stream Tracing
export GRETA_B3_61=1
export GRETA_TRACE_STAGE=1
export GRETA_TRACE_STAGE_LAYERS="${LAYERS[*]}"
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

# Residual Stream Trace Points
export GRETA_TRACE_B3_61_RESIDUAL_PRE_ATTN=1
export GRETA_TRACE_B3_61_ATTN_IN=1
export GRETA_TRACE_B3_61_ATTN_OUT=1
export GRETA_TRACE_B3_61_RESIDUAL_POST_ATTN=1
export GRETA_TRACE_B3_61_FFN_NORM_IN=1
export GRETA_TRACE_B3_61_MLP_OUT=1
export GRETA_TRACE_B3_61_RESIDUAL_POST_MLP=1
export GRETA_TRACE_B3_61_LOGITS=1

# Tensor Hash Validation (reuse B3.59 format)
export GRETA_TRACE_EMBED_OUT=1

# Critical positions for explicit validation
# Position 826 for p6_len_16, Position 1652 for p6_len_32
export GRETA_TRACE_B3_61_POS_826=1
export GRETA_TRACE_B3_61_POS_1652=1

# Trace Output Directory
export GRETA_TRACE_DIR="$OUT_DIR/traces"

# Execution Loop
for prompt in "${PROMPTS[@]}"; do
    prompt_name=$(basename "$prompt" .txt)
    echo "--- Running $prompt_name ---"
    
    export GRETA_TRACE_PROMPT_ID="$prompt_name"
    export GRETA_TRACE_STAGE_OUT="$OUT_DIR/traces/${prompt_name}_trace.jsonl"
    
    # Run inference with probes
    $BINARY --model models/greta-v1.gguf --prompt-file "$prompt" \
            --max-tokens 5 --greedy \
            > "$OUT_DIR/run/${prompt_name}.log" 2>&1 || {
        echo "ERROR: Inference failed for $prompt_name"
        continue
    }
done

echo "--- Generating Analysis ---"
python3 tools/benchmarks/analyze_b3_61_residual_stream_bisect.py \
    --input_dir "$OUT_DIR/traces" \
    --baseline_dir "artifacts_remote/2026-02-05/b3_59/traces" \
    --output "$OUT_DIR/b3_61_analysis.txt" \
    --layers "${LAYERS[*]}"

echo "--- Packaging Artifacts ---"
tar -czf "$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz" -C "$OUT_DIR" run traces checkpoints b3_61_analysis.txt

echo "=== Execution Complete ==="
echo "Artifacts: $OUT_DIR/gretacore_${B3_ID}_artifacts.tgz"
echo ""
echo "Synchronization Verification:"
echo "  Local commit: $FINAL_COMMIT"
echo "  Date: $DATE"
echo ""
echo "Next Steps 1. SCP:"
echo "  artifacts to local repository"
echo "  2. Verify artifact integrity"
echo "  3. Review FIRST_FAIL table in b3_61_analysis.txt"
echo "  4. Generate documentation"
