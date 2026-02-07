#!/bin/bash
# B3.64: D2H Illegal Memory Access Debug Runner
# Deterministic runner for debugging hipMemcpy D2H failures
# Usage: ./tools/benchmarks/run_b3_64_mi300x.sh <NODE_IP>

set -euo pipefail

# Lock file for exclusive access
LOCK_FILE="/tmp/gretacore_b3_64.lock"

# Default values
NODE_IP="${1:-129.212.184.200}"
DATE="${2:-$(date +%Y-%m-%d)}"
B3_ID="b3_64"
OUT_DIR="artifacts_remote/$DATE/$B3_ID"

echo "=== B3.64 D2H Illegal Memory Access Debug ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "Output: $OUT_DIR"

# =============================================
# 1. LOCK EXCLUSIVO
# =============================================
echo "--- Acquiring exclusive lock ---"

# Crear el lock file si no existe
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
# 2. VERIFICAR PROCESOS GRETA_INFER COLGADOS
# =============================================
echo "--- Checking for hung greta_infer processes ---"
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore

    # Buscar procesos greta_infer
    HUNG_PROCS=\$(ps aux | grep -E 'greta_infer' | grep -v grep | grep -v \$\$ || true)

    if [ -n \"\$HUNG_PROCS\" ]; then
        echo \"Found hung greta_infer processes:\"
        echo \"\$HUNG_PROCS\"

        # Extraer PIDs y verificar que pertenezcan a este repo
        for pid in \$(echo \"\$HUNG_PROCS\" | awk '{print \$2}'); do
            if [ -f \"/proc/\$pid/cmdline\" ]; then
                cmdline=\$(tr '\0' ' ' < /proc/\$pid/cmdline 2>/dev/null || true)
                if echo \"\$cmdline\" | grep -q '/root/gretacore'; then
                    echo \"Killing process \$pid (from this repo)\"
                    kill -9 \$pid 2>/dev/null || true
                else
                    echo \"Skipping process \$pid (from different repo)\"
                fi
            fi
        done

        # Wait a moment for cleanup
        sleep 1
    else
        echo \"No hung greta_infer processes found\"
    fi
"

# =============================================
# 3. SYNC REMOTO (STATELESS)
# =============================================
echo "--- Syncing remote (stateless) ---"
REMOTE_HEAD=$(ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore
    git fetch origin
    git checkout main
    git reset --hard origin/main
    git clean -fdx
    echo REMOTE_HEAD=\$(git rev-parse --short HEAD)
")
echo "Remote HEAD: $REMOTE_HEAD"

# =============================================
# 4. BUILD REMOTO
# =============================================
echo "--- Building remote ---"
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore
    if [ -d tools/inference/build ]; then
        cd tools/inference/build
        make -j\$(nproc)
    elif [ -f tools/inference/Makefile ] || [ -f Makefile ]; then
        make -j\$(nproc)
    fi
"

# =============================================
# 5. VERIFICAR PERMISOS DEL BINARY
# =============================================
echo "--- Verifying binary permissions ---"
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore

    BINARY='tools/inference/build/greta_infer'
    if [ -f \"\$BINARY\" ]; then
        if [ ! -x \"\$BINARY\" ]; then
            echo \"Adding +x permission to \$BINARY\"
            chmod +x \"\$BINARY\"
        else
            echo \"Binary already executable\"
        fi
    else
        echo \"WARNING: Binary not found at \$BINARY\"
    fi
"

# =============================================
# 6. EXPORTAR VARIABLES HIP DEBUG
# =============================================
echo "--- Setting HIP debug environment variables ---"
HIP_VARS="AMD_SERIALIZE_KERNEL=3
HIP_LAUNCH_BLOCKING=1
HSA_ENABLE_SDMA=0"

for var in $HIP_VARS; do
    VAR_NAME=$(echo "$var" | cut -d= -f1)
    VAR_VAL=$(echo "$var" | cut -d= -f2)
    echo "export $VAR_NAME=$VAR_VAL"
done

# =============================================
# 7. VERIFICACIÓN DE MODELO
# =============================================
echo "--- Validating model ---"
MODEL_PATH="/root/gretacore/models/greta-v1.gguf"
MODEL_CHECK=$(ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore

    if [ ! -f '$MODEL_PATH' ]; then
        echo \"ERROR: Model not found at $MODEL_PATH\"
        exit 1
    fi

    # Check file size to validate it's a Llama-2-7B compatible model
    # Expected: ~13-15GB for 7B model with f16
    SIZE=\$(stat -c%s '$MODEL_PATH' 2>/dev/null || stat -f%z '$MODEL_PATH' 2>/dev/null)
    SIZE_GB=\$((SIZE / 1073741824))

    echo \"Model size: \${SIZE_GB}GB\"

    # For now, just validate existence and approximate size
    if [ \$SIZE_GB -lt 10 ]; then
        echo \"ERROR: Model file too small (\${SIZE_GB}GB), expected ~13-15GB for Llama-2-7B\"
        exit 1
    fi

    echo \"Model validated: $MODEL_PATH (\${SIZE_GB}GB)\"
")
echo "$MODEL_CHECK"

# =============================================
# 8. EJECUTAR B3.64 CON CONFIGURACIÓN DETERMINÍSTICA
# =============================================
echo "--- Running B3.64 with deterministic config ---"

ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
    set -euo pipefail
    cd /root/gretacore

    # Setup dirs
    mkdir -p '$OUT_DIR/run' '$OUT_DIR/traces' '$OUT_DIR/checkpoints'

    # Export HIP debug vars
    export AMD_SERIALIZE_KERNEL=3
    export HIP_LAUNCH_BLOCKING=1
    export HSA_ENABLE_SDMA=0

    # Deterministic config
    export GRETA_B3_64=1
    export GRETA_TRACE_B3_64=1
    export GRETA_TRACE_B3_64_DIR='$OUT_DIR/traces'

    # Binary path
    BINARY='./tools/inference/build/greta_infer'
    if [ ! -f \"\$BINARY\" ]; then
        BINARY='./tools/inference/greta_infer'
    fi

    PROMPT='tools/benchmarks/prompts/p0_short.txt'
    PROMPT_NAME='p0_short'

    echo \"--- Running \$PROMPT_NAME ---\"
    echo \"Prompt: \$PROMPT\"
    echo \"Max tokens: 5\"
    echo \"Output: $OUT_DIR/run/\${PROMPT_NAME}.log\"

    export GRETA_TRACE_PROMPT_ID=\"\$PROMPT_NAME\"
    export GRETA_TRACE_STAGE_OUT='$OUT_DIR/traces/\${PROMPT_NAME}_trace.jsonl'

    # Run with deterministic settings
    \$BINARY \\
        --model models/greta-v1.gguf \\
        --prompt-file \"\$PROMPT\" \\
        --max-tokens 5 \\
        --grety \\
        > '$OUT_DIR/run/\${PROMPT_NAME}.log' 2>&1 || {
        echo \"ERROR: Inference failed for \$PROMPT_NAME\"
        cat '$OUT_DIR/run/\${PROMPT_NAME}.log'
        exit 1
    }

    echo \"--- Run complete ---\"

    # Package artifacts
    tar -czf '$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz' -C '$OUT_DIR' run traces checkpoints

    echo \"=== B3.64 Complete ===\"
    echo \"Artifacts: $OUT_DIR/gretacore_${B3_ID}_artifacts.tgz\"
"

# =============================================
# 9. COPIAR ARTEFACTOS A LOCAL
# =============================================
echo "=== Copying artifacts to local ==="
mkdir -p "$OUT_DIR"
scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUT_DIR/*.tgz" "$OUT_DIR/" 2>/dev/null || echo "WARNING: SCP failed"

# =============================================
# 10. EXTRAER SI EXISTE
# =============================================
if [ -f "$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz" ]; then
    cd "$OUT_DIR"
    tar -xzf "gretacore_${B3_ID}_artifacts.tgz"
    echo "=== Artifacts extracted ==="
    ls -la run/ 2>/dev/null || true
else
    echo "WARNING: No artifacts tgz found"
fi

echo "=== B3.64 Runner Done ==="
echo "Log file: $OUT_DIR/run/p0_short.log"
