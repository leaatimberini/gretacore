#!/bin/bash
set -euo pipefail

# =============================================================================
# B3.67 Equivalence Guardrail Runner
# =============================================================================
# Usage: ./run_b3_67_equivalence_guardrail.sh <NODE_IP> [YYYY-MM-DD] [--kv_aligned 0|1] [--seeds "0,1,2"]
# Default date = today
# Default kv_aligned = all (0,1)
# Default seeds = "0,1,2"
#
# Este runner ejecuta el benchmark B3.67 para comparar hidden states entre prefill y decode
# con diferentes configuraciones de kv_aligned.
#
# NOTA: Usa GRETA_TRACE_B3_66 para generar traces (hidden states del último layer)
# Los traces se comparan entre prefill (phase=prefill_last) y decode (phase=decode0)
# =============================================================================

# -----------------------------------------------------------------------------
# Parseo de argumentos
# -----------------------------------------------------------------------------
NODE_IP="${1:-129.212.184.200}"
DATE="${2:-$(date +%Y-%m-%d)}"

# Shift args to process only mode-related arguments
shift $(( $# > 2 ? 2 : 0 )) 2>/dev/null || true

KV_ALIGNED=""  # Empty means all values
SEEDS="0,1,2"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --kv_aligned)
            KV_ALIGNED="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Configuración
# -----------------------------------------------------------------------------
REMOTE_BASE="/root/gretacore"
LOCK_FILE="/tmp/greta_b3_67.lock"
RUN_DIR="b3_67"

# Parámetros del benchmark
DTYPE="bf16"
PROMPT_LEN="512"
GEN_LEN="128"
MODE_VALUES="prefill decode"

# Arrays
IFS=',' read -ra SEEDS_ARRAY <<< "$SEEDS"

# Parse KV_ALIGNED into array (comma-separated)
if [[ -z "$KV_ALIGNED" ]]; then
    KV_ALIGNED_VALUES=(0 1)
else
    IFS=',' read -ra KV_ALIGNED_VALUES <<< "$KV_ALIGNED"
fi

echo "=== B3.67 Equivalence Guardrail ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "KV aligned values: ${KV_ALIGNED_VALUES[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"

# -----------------------------------------------------------------------------
# Lock exclusivo
# -----------------------------------------------------------------------------
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "ERROR: Lock file $LOCK_FILE is held by another process"
    exit 2
fi

# -----------------------------------------------------------------------------
# Sync y setup remoto
# -----------------------------------------------------------------------------
echo "[1/6] Sync remote repo..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "cd $REMOTE_BASE && git fetch origin && git reset --hard origin/main"

echo "[2/6] Build greta_infer..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "cd $REMOTE_BASE/tools/inference/build && make -j\$(nproc)"

echo "[3/6] Setup directories..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "mkdir -p $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/{traces,logs}"

# -----------------------------------------------------------------------------
# Ejecutar benchmark para cada configuración
# -----------------------------------------------------------------------------
echo "[4/6] Run benchmarks..."

for KV_VAL in "${KV_ALIGNED_VALUES[@]}"; do
    for SEED in "${SEEDS_ARRAY[@]}"; do
        echo "  === kv_aligned=$KV_VAL, seed=$SEED ==="

        for MODE in $MODE_VALUES; do
            echo "    Running mode=$MODE..."

            # Directorio local para esta configuración
            LOCAL_TRACES_DIR="artifacts_remote/$DATE/$RUN_DIR/traces/kv_aligned_$KV_VAL/seed_$SEED/$MODE"
            mkdir -p "$LOCAL_TRACES_DIR"

            # Metadata
            TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
            GIT_COMMIT=$(git rev-parse HEAD)

            # Map mode to B3.66 mode for tracing
            if [[ "$MODE" == "prefill" ]]; then
                B3_66_MODE="as_designed"  # Use as_designed for prefill traces
            else
                B3_66_MODE="as_designed"  # Use as_designed for decode traces
            fi

            # Ejecutar en remoto con variables de entorno determinísticas
            ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
                cd $REMOTE_BASE
                export HIP_LAUNCH_BLOCKING=1
                export AMD_SERIALIZE_KERNEL=3
                export HSA_ENABLE_SDMA=0
                export GRETA_SEED=$SEED
                export GRETA_TRACE_B3_66=1
                export GRETA_TRACE_LAYERS='0,1,2,4,8,16,24,31,32'
                export GRETA_B3_66_MODE=$B3_66_MODE
                export GRETA_KV_ALIGNED=$KV_VAL

                # Crear directorio para traces
                mkdir -p \$HOME/artifacts_remote/$DATE/$RUN_DIR/traces/kv_aligned_$KV_VAL/seed_$SEED/$MODE

                # Ejecutar greta_infer y guardar traces
                ./tools/inference/build/greta_infer \
                    --model ./models/greta-v1.gguf \
                    --prompt tools/benchmarks/prompts/p0_short.txt \
                    --max-tokens 1 \
                    --seed $SEED \
                    --mode $MODE \
                    --dtype $DTYPE \
                    2>&1 | tee /tmp/greta_b3_67_${MODE}.log

                # Mover traces al directorio correcto
                if [ -d \$HOME/artifacts_remote/$DATE/$RUN_DIR/traces ]; then
                    find \$HOME/artifacts_remote/$DATE/$RUN_DIR/traces -name '*.jsonl' -exec mv {} \$HOME/artifacts_remote/$DATE/$RUN_DIR/traces/kv_aligned_$KV_VAL/seed_$SEED/$MODE/ \;
                fi

                # Guardar metadata
                cat > \$HOME/artifacts_remote/$DATE/$RUN_DIR/traces/kv_aligned_$KV_VAL/seed_$SEED/$MODE/config.json << EOF
{
  \"dtype\": \"$DTYPE\",
  \"prompt_len\": $PROMPT_LEN,
  \"gen_len\": $GEN_LEN,
  \"seed\": $SEED,
  \"kv_aligned\": $KV_VAL,
  \"mode\": \"$MODE\",
  \"timestamp\": \"$TIMESTAMP\",
  \"git_commit\": \"$GIT_COMMIT\"
}
EOF
            "

            # Copiar logs a local
            scp -o StrictHostKeyChecking=no "root@$NODE_IP:/tmp/greta_b3_67_${MODE}.log" "$LOCAL_TRACES_DIR/" 2>/dev/null || true

            # Copiar traces comprimidos
            ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
                cd \$HOME/artifacts_remote/$DATE/$RUN_DIR/traces/kv_aligned_$KV_VAL/seed_$SEED/$MODE
                for f in *.jsonl; do
                    if [ -f \"\$f\" ]; then
                        gzip -f \"\$f\"
                    fi
                done
            "

            scp -o StrictHostKeyChecking=no "root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/traces/kv_aligned_$KV_VAL/seed_$SEED/$MODE/*.jsonl.gz" "$LOCAL_TRACES_DIR/" 2>/dev/null || true
            scp -o StrictHostKeyChecking=no "root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/traces/kv_aligned_$KV_VAL/seed_$SEED/$MODE/config.json" "$LOCAL_TRACES_DIR/" 2>/dev/null || true
        done
    done
done

# -----------------------------------------------------------------------------
# Empaquetar artifacts
# -----------------------------------------------------------------------------
echo "[5/6] Package artifacts..."
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    cd $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR
    tar -czvf gretacore_b3_67_artifacts.tgz traces/ logs/
    ls -la
"

# -----------------------------------------------------------------------------
# Ejecutar analyzer
# -----------------------------------------------------------------------------
echo "[6/6] Run analyzer (local)..."
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir "artifacts_remote/$DATE/$RUN_DIR/traces" \
    --output "artifacts_remote/$DATE/$RUN_DIR/B3_67_EQUIVALENCE_GUARDRAIL.md"

echo ""
echo "=== B3.67 Execution Summary ==="
echo "Date: $DATE"
echo "KV aligned values: ${KV_ALIGNED_VALUES[*]}"
echo "Seeds: ${SEEDS_ARRAY[*]}"
echo "Output dir: artifacts_remote/$DATE/$RUN_DIR/"
echo "Report: artifacts_remote/$DATE/$RUN_DIR/B3_67_EQUIVALENCE_GUARDRAIL.md"
echo ""
echo "Done. Copy artifacts with:"
echo "  scp root@$NODE_IP:$REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/gretacore_b3_67_artifacts.tgz ."

flock -u 200
