#!/bin/bash
# B3.68 Smoke Test - Validates greta_infer logits dump functionality
# Usage: ./test_b3_68_smoke.sh [greta_infer_path] [model_path]
set -euo pipefail

GRETA_INFER="${1:-./tools/inference/build/greta_infer}"
MODEL_PATH="${2:-./models/greta-v1.gguf}"
TMPDIR=$(mktemp -d)

cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

echo "=== B3.68 Smoke Test ==="
echo "greta_infer: $GRETA_INFER"
echo "model: $MODEL_PATH"
echo "tmpdir: $TMPDIR"

# Run greta_infer with dump-logits
echo ""
echo "[1/3] Running greta_infer with --dump-logits..."
"$GRETA_INFER" \
    --model "$MODEL_PATH" \
    --prompt "Hello" \
    --max-tokens 1 \
    --kv-aligned 1 \
    --mode decode \
    --seed 42 \
    --dump-logits "$TMPDIR" \
    --greedy

# Validate files exist
echo ""
echo "[2/3] Validating output files..."
if [ ! -f "$TMPDIR/metadata.json" ]; then
    echo "FAIL: metadata.json not found"
    exit 1
fi

if [ ! -f "$TMPDIR/logits.jsonl.gz" ]; then
    echo "FAIL: logits.jsonl.gz not found"
    exit 1
fi

echo "  metadata.json: $(stat -c%s "$TMPDIR/metadata.json") bytes"
echo "  logits.jsonl.gz: $(stat -c%s "$TMPDIR/logits.jsonl.gz") bytes"

# Validate metadata.json is valid JSON
echo ""
echo "[3/3] Validating JSON format..."
python3 -c "import json, sys; json.load(open('$TMPDIR/metadata.json'))" || {
    echo "FAIL: metadata.json is not valid JSON"
    exit 1
}

# Validate logits.jsonl.gz is valid gzip
zcat "$TMPDIR/logits.jsonl.gz" >/dev/null || {
    echo "FAIL: logits.jsonl.gz is not valid gzip"
    exit 1
}

echo ""
echo "=== B3.68 Smoke Test PASSED ==="
echo "metadata.json contents:"
cat "$TMPDIR/metadata.json"
