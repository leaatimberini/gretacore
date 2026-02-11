# B3.86 — Attention Implementation Probe (MI300X)

**Date**: 2026-02-10  
**Phase**: RCA - Performance Characterization  

## Executive Summary
Identification of the active attention kernel used during prefill and decode phases on the MI300X architecture.

## Methodology
- **Mode**: Automated probe via `greta_infer` output.
- **Verification**: The engine prints the active kernel name in the `[PERF_TIMING]` JSON payload.

## Measured Results
| Phase | Context | Detected Implementation |
|---|---|---|
| Prefill | 8,192 | `flash_v2_naive` |
| Prefill | 16,384 | `flash_v2_naive` |
| Decode | Any | `flash_v2_naive` |

## Findings
1. **Single Backend**: The engine currently utilizes a unified naive FlashAttention-v2 implementation for all context lengths.
2. **Lack of Specialization**: No specialized kernels (e.g., PagedAttention or TiledAttention) are being triggered, explaining the consistent $O(N^2)$ prefill scaling seen in B3.85.
3. **Switching Logic**: The `attn_impl` field is correctly propagated through the inference engine to the CLI output, enabling automated performance tracking.

## Verdict
**PASS_PROBE**: Implementation identified. The "naive" nature of the current backend is confirmed as the primary bottleneck to address in B3.89.
