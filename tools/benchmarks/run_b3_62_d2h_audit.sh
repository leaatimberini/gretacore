#!/bin/bash
# B3.62: HIP D2H Transfer Audit - Script de Ejecución
# =====================================================
# Este script ejecuta la auditoría de transferencias D2H con instrumentación

set -euo pipefail

REMOTE_HOST="root@129.212.184.200"
REMOTE_DIR="/root/gretacore"
DATE=$(date +%Y-%m-%d)
B3_ID="b3_62"
OUT_DIR="artifacts_remote/$DATE/$B3_ID"

echo "=== B3.62 HIP D2H Transfer Audit ==="
echo "Date: $DATE"
echo "Output: $OUT_DIR"

# Crear directorio de salida
ssh $REMOTE_HOST "mkdir -p $REMOTE_DIR/$OUT_DIR/run $REMOTE_DIR/$OUT_DIR/traces"

# Sincronizar código al remoto
echo "--- Sincronizando código al remoto ---"
rsync -avz --exclude='.git' --exclude='build*' --exclude='*.tgz' \
    /media/leandro/D08A27808A2762683/gretacore/gretacore/ \
    $REMOTE_HOST:$REMOTE_DIR/ \
    --delete

# Compilar con GRETA_TRACE_MEMCPY
echo "--- Compilando con GRETA_TRACE_MEMCPY ---"
ssh $REMOTE_HOST "
cd $REMOTE_DIR
rm -rf build_greta
mkdir -p build_greta
cd build_greta
cmake .. -DCMAKE_BUILD_TYPE=Debug -DGRETA_TRACE_MEMCPY=ON
make -j\$(nproc)
"

# Configurar variables de entorno
ssh $REMOTE_HOST "
cd $REMOTE_DIR
export GRETA_TRACE_MEMCPY=1
export GRETA_B3_62=1

# Prompts de prueba
PROMPTS=(
    'tools/benchmarks/prompts/p0_short.txt'
)

BINARY='./tools/inference/build/greta_infer'

for prompt in \"\${PROMPTS[@]}\"; do
    prompt_name=\$(basename \"\$prompt\" .txt)
    echo \"--- Running \$prompt_name ---\"
    
    \$BINARY --model models/greta-v1.gguf --prompt-file \"\$prompt\" \
            --max-tokens 5 --greedy \
            > \"$OUT_DIR/run/\${prompt_name}.log\" 2>&1 || {
        echo \"ERROR: Inference failed for \$prompt_name\"
    }
done

echo '--- Packaging Artifacts ---'
tar -czf \"$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz\" -C \"$OUT_DIR\" run traces

echo '=== Execution Complete ==='
echo \"Artifacts: $OUT_DIR/gretacore_${B3_ID}_artifacts.tgz\"
"

echo "=== B3.62 Execution Started ==="
echo "Waiting for completion..."
