#!/bin/bash
# =============================================================================
# B3.89 No-Spill Gate
# Usage: ./b3_89_no_spill_gate.sh [-DGRETA_PREFILL_Q_LDS=1]
# Output: Exits 0 if scratch==0, else 1
# =============================================================================
set -euo pipefail

VARIANT_FLAG=${1:-""}
DUMP_SCRIPT="tools/benchmarks/b3_89_dump_kernel_resources.sh"

echo "=== B3.89 Scratch Spill Gate ==="
echo "Building and analyzing variant: $VARIANT_FLAG"

# Run the dumb script and capture output
OUTPUT=$(bash $DUMP_SCRIPT "$VARIANT_FLAG" 2>&1 || true)
echo "$OUTPUT"

# Extract scratch size for the kernel
# Looking for: "Scratch: N bytes" under "flash_attention_prefill_kernel"
# Warning: The dump script prints multiline output. We need to parse carefully.

SCRATCH_VAL=$(echo "$OUTPUT" | grep -A 10 "Kernel:.*flash_attention_prefill_kernel" | grep "Scratch:" | grep -o "[0-9]\+")

if [ -z "$SCRATCH_VAL" ]; then
    echo "GATE ERROR: Could not parse scratch usage from dump output."
    exit 1
fi

echo "Detected Scratch Usage: $SCRATCH_VAL bytes"

if [ "$SCRATCH_VAL" -eq "0" ]; then
    echo "GATE PASSED: Zero scratch spilling."
    exit 0
else
    echo "GATE FAILED: Scratch spilling detected ($SCRATCH_VAL bytes)."
    exit 1
fi
