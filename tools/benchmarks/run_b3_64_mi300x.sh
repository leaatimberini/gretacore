#!/bin/bash
# B3.64: Numerical Drift Audit Runner
# Audit de drift numérico prefill/decode
# Usage: ./tools/benchmarks/run_b3_64_mi300x.sh <NODE_IP> <YYYY-MM-DD>

set -euo pipefail

# Default values
NODE_IP="${1:-129.212.184.200}"
DATE="${2:-$(date +%Y-%m-%d)}"
B3_ID="b3_64"
OUT_DIR="artifacts_remote/$DATE/$B3_ID"

echo "=== B3.64 Numerical Drift Audit ==="
echo "Node: $NODE_IP"
echo "Date: $DATE"
echo "Output: $OUT_DIR"

# 1. SYNC REMOTO (STATELESS)
echo "--- Syncing remote (stateless) ---"
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
  set -euo pipefail
  cd /root/gretacore
  git fetch origin
  git checkout main
  git reset --hard origin/main
  git clean -fdx
  echo REMOTE_HEAD=\$(git rev-parse --short HEAD)
"

# 2. BUILD REMOTO
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

# 3. EJECUTAR B3.64
echo "--- Running B3.64 ---"
ssh -o StrictHostKeyChecking=no "root@$NODE_IP" "
  set -euo pipefail
  cd /root/gretacore

  # Fix permissions
  chmod +x tools/inference/greta_infer 2>/dev/null || true
  chmod +x tools/inference/build/greta_infer 2>/dev/null || true

  # Setup dirs
  mkdir -p \"$OUT_DIR/run\" \"$OUT_DIR/traces\" \"$OUT_DIR/checkpoints\"

  # Prompts
  PROMPTS=(
    'tools/benchmarks/prompts/p0_short.txt'
    'tools/benchmarks/prompts/p6_len_16.txt'
    'tools/benchmarks/prompts/p6_len_32.txt'
  )

  # Flags B3.64
  export GRETA_B3_64=1
  export GRETA_TRACE_B3_64=1
  export GRETA_TRACE_B3_64_DIR=\"$OUT_DIR/traces\"
  export GRETA_TRACE_STAGE=1
  export GRETA_TRACE_STAGE_DEBUG_INPUT=1

  # Binary path
  BINARY='./tools/inference/build/greta_infer'
  if [ ! -f \"\$BINARY\" ]; then
    BINARY='./tools/inference/greta_infer'
  fi

  for prompt in \"\${PROMPTS[@]}\"; do
    prompt_name=\$(basename \"\$prompt\" .txt)
    echo \"--- Running \$prompt_name ---\"

    export GRETA_TRACE_PROMPT_ID=\"\$prompt_name\"
    export GRETA_TRACE_STAGE_OUT=\"$OUT_DIR/traces/\${prompt_name}_trace.jsonl\"

    \$BINARY \
      --model models/greta-v1.gguf \
      --prompt-file \"\$prompt\" \
      --max-tokens 5 \
      --greedy \
      > \"$OUT_DIR/run/\${prompt_name}.log\" 2>&1 || {
      echo \"ERROR: Inference failed for \$prompt_name\"
      continue
    }
  done

  # Package
  tar -czf \"$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz\" -C \"$OUT_DIR\" run traces checkpoints

  echo \"=== B3.64 Complete ===\"
  echo \"Artifacts: $OUT_DIR/gretacore_${B3_ID}_artifacts.tgz\"
"

# 4. COPIAR A LOCAL
echo "=== Copying artifacts to local ==="
mkdir -p "$OUT_DIR"
scp -o StrictHostKeyChecking=no "root@$NODE_IP:$OUT_DIR/*.tgz" "$OUT_DIR/" 2>/dev/null || echo "WARNING: SCP failed, artifacts may not be copied"

# 5. EXTRAER SI EXISTE
if [ -f "$OUT_DIR/gretacore_${B3_ID}_artifacts.tgz" ]; then
  cd "$OUT_DIR"
  tar -xzf "gretacore_${B3_ID}_artifacts.tgz"
  echo "=== Artifacts extracted ==="
  ls -la traces/ run/ 2>/dev/null || true
else
  echo "WARNING: No artifacts tgz found at $OUT_DIR/gretacore_${B3_ID}_artifacts.tgz"
fi

echo "=== B3.64 Runner Done ==="
