# B3.89 — Prefill Kernel Optimization Plan

**Date**: 2026-02-10  
**Status**: IN_PROGRESS  

## Overview
Based on the RCA conducted in B3.85 (escalado $O(N^2)$) and B3.88 (factibilidad en 32k), the primary bottleneck for long-context inference on AMD Instinct MI300X has been identified as the lack of staging (tiling) in the `flash_attention_prefill_kernel`.

## Key Findings
- **32k prefill time**: 2542.79 s (Baseline).
- **Bottleneck**: Excessive VRAM read/write operations per attention block.
- **Root Cause**: Implementation does not utilize LDS (Local Data Share) for block reuse.

## Proposed Plan
We are initiating an optimization milestone to rewrite the prefill kernel using LDS-tiling techniques similar to FlashAttention-2.

### Actionable Items
1. **Implementation Plan**: Documented in `plans/B3_89_prefill_kernel_optimization_plan.md`.
2. **Harness**: A new microbench script `tools/benchmarks/run_b3_89_prefill_microbench.sh` is ready for iterative testing.
3. **Analyzer**: The standard analyzer has been extended with `--mode b3_89` to track improvement against the target of < 600s for 32k context.

## Next Steps
- Implement the tiled kernel logic in `src/rt/backend/hip/kernels/attention_kernels.hip`.
- Validate with small context equivalence checks.
- Benchmark 32k performance on MI300X node.
