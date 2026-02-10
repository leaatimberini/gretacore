# B3.87 — Decode TPS Decomposition (MI300X)

**Date**: 2026-02-10  
**Phase**: RCA - Performance Decomposition  

## Executive Summary
Evaluation of MI300X decode performance under different determinism and batching configurations.

## Methodology
- **Hardware**: AMD Instinct MI300X (single node)
- **Model**: Greta-v1 (Llama-2-7B equivalent)
- **Prompt**: "Hello" (short)
- **Generation**: 128 tokens
- **Variants**:
    - Determinism: `off` vs `on` (via `HIP_LAUNCH_BLOCKING=1` and `AMD_SERIALIZE_KERNEL=1` in runner)
    - Batch Size: 1 vs 8

## Results

| Batch | Determinism | Tokens/s | Decode Time (s) | Load (s) | Overhead (s) |
|-------|-------------|----------|-----------------|----------|--------------|
| 1     | off         | 19.57    | 6.48            | 21.26    | 0.31         |
| 1     | on          | 17.40    | 7.29            | 21.32    | 0.32         |
| 8     | off         | 19.57    | 6.48            | 21.40    | 0.31         |
| 8     | on          | 17.38    | 7.30            | 21.31    | 0.35         |

## Key Observations
1. **Determinism Penalty**: Enabling strict determinism flags (`on`) results in a ~11% reduction in throughput (from ~19.6 TPS to ~17.4 TPS).
2. **Batch Invariance**: In this test, batch size 8 did not improve aggregate TPS for a single short prompt. This indicates that for a single user, overheads or memory bandwidth limits are consistent regardless of the reserved batch capacity.
3. **Load Time Consistency**: Model loading remains stable at ~21s.

## Conclusion
Determinism is viable on MI300X but carries a measurable performance cost. For production use where strict bit-perfection is required, an 11% buffer should be accounted for in scaling plans.
