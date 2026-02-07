#!/bin/bash
# B3.65: Decode Determinism Audit Runner
# Verifies decode output is deterministic and bit-stable across runs
# Usage: ./tools/benchmarks/run_b3_65_determinism_mi300x.sh <NODE_IP>

set -euo pipefail

# Lock file for exclusive access
LOCK_FILE="/tmp/gretacore_b3_65.lock"

# Default values
NODE_IP="${1:-129.212.184.200}"
DATE="${2:-$(date +%Y-%m-%d)}"
B3_ID="b3_65"
OUT_DIR="artifacts_remote/$DATE/$B3_ID/run"
PROMPT='What is 2+2?'
MAX_TOKENS=10
NUM_RUNS=10
SEED=42

echo "=== B3.65 Decode Determinism Audit ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "Output: $OUT_DIR"
echo "Runs: $NUM_RUNS"
echo "Prompt: $PROMPT"
echo "Seed: $SEED"

# =============================================
# 1. LOCK EXCLUSIVO
# =============================================
echo "--- Acquiring exclusive lock ---"

touch "$LOCK_FILE"

if ! flock -n "$LOCK_FILE" -c 'echo "Lock acquired"'; then
    HOLDING_PID=$(fuser "$LOCK_FILE" 2>/dev/null || echo "unknown")
    echo "ERROR: Lock file busy: $LOCK_FILE"
    echo "Holding PID: $HOLDING_PID"
    exit 2
fi

echo "Lock acquired successfully"

# Cleanup lock on exit
trap 'rm -f "$LOCK_FILE"' EXIT

# =============================================
# 2. VERIFICAR PERMISOS DE EJECUCION
# =============================================
echo "--- Verifying greta_infer_fixed permissions ---"

ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore

    BINARY='tools/inference/greta_infer_fixed'
    if [ ! -x \"\$BINARY\" ]; then
        echo \"ERROR: Binary not found or not executable: \$BINARY\"
        exit 1
    fi
    echo \"Binary permissions OK: \$BINARY\"
"

# =============================================
# 3. CREAR DIRECTORIO DE OUTPUT
# =============================================
echo "--- Creating output directory ---"
mkdir -p "$OUT_DIR"

# =============================================
# 4. CONFIGURAR ENVIRONMENT Y EJECUTAR RUNS
# =============================================
echo "--- Starting $NUM_RUNS consecutive runs ---"

ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore

    export GRETA_D2H_DEBUG=1
    export HIP_LAUNCH_BLOCKING=1
    export AMD_SERIALIZE_KERNEL=3
    export HSA_ENABLE_SDMA=0

    for i in \$(seq -w 1 $NUM_RUNS); do
        RUN_OUTPUT=\"$OUT_DIR/run_\${i}.txt\"
        
        echo \"--- Run \$i / $NUM_RUNS ---\"
        
        ./tools/inference/greta_infer_fixed \
            --model models/greta-v1.gguf \
            --prompt \"$PROMPT\" \
            --max-tokens $MAX_TOKENS \
            --temp 0.0 \
            --seed $SEED \
            2>&1 | tee \"\$RUN_OUTPUT\"
        
        echo \"Run \$i completed\"
        sleep 1
    done
"

# =============================================
# 5. GENERAR SUMMARY.TSV
# =============================================
echo "--- Generating summary.tsv ---"

# Create summary.tsv with columns: run_idx, top1_token, tokens_sec, hash_logits
{
    echo -e "run_idx\ttop1_token\ttokens_sec\thash_logits"
    
    for i in $(seq -w 1 $NUM_RUNS); do
        RUN_FILE="$OUT_DIR/run_$i.txt"
        
        if [ -f "$RUN_FILE" ]; then
            # Extract relevant metrics from run output
            # Note: Adjust parsing based on actual greta_infer_fixed output format
            TOP1_TOKEN=$(grep -oP 'top1_token:\s*\K\d+' "$RUN_FILE" 2>/dev/null || echo "N/A")
            TOKENS_SEC=$(grep -oP 'tokens\/sec:\s*\K[\d.]+' "$RUN_FILE" 2>/dev/null || echo "N/A")
            HASH_LOGITS=$(grep -oP 'logits_hash64:\s*\K[0-9a-fA-F]+' "$RUN_FILE" 2>/dev/null || echo "N/A")
            
            echo -e "$i\t$TOP1_TOKEN\t$TOKENS_SEC\t$HASH_LOGITS"
        else
            echo -e "$i\tN/A\tN/A\tN/A"
        fi
    done
} > "$OUT_DIR/summary.tsv"

echo "Summary saved to: $OUT_DIR/summary.tsv"

echo ""
echo "=== B3.65 RUN COMPLETED ==="
echo "Output directory: $OUT_DIR"
echo "Next step: Run analysis"
echo "  python3 tools/benchmarks/analyze_b3_65_determinism.py --dir $OUT_DIR"
