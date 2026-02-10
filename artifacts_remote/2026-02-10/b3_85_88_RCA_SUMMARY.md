# B3.85–B3.88 RCA Unified Summary: MI300X Prefill/Decode Characterization

**Date**: 2026-02-10  
**Status**: Infrastructure Verified / Partial Results  

## 1. Benchmarking Infrastructure Updates
The following enhancements were implemented to enable deep performance RCA:
- **Phase-Level Timing**: Integrated `tokenize_time_ms`, `prefill_time_ms`, and `decode_time_ms` into `GenerationStats`.
- **Active Kernel Probe**: Added `attn_impl` detection to `greta_infer` output to verify kernel switching.
- **Scaling Suites**: Created four new automated benchmark suites (B3.85-B3.88) with dedicated remote executors.

## 2. Identified Bottlenecks & Fixes

### A. Sequence Length Limit
- **Issue**: `BlockScheduler` and `ModelConfig` had a hardcoded `max_seq_len` of 2048.
- **RCA**: Attempting 32k context resulted in `seq_len exceeds GRETA_MAX_SEQ_LEN` crashes.
- **Fix**: Increased `ModelConfig` default to 32768 and implemented `GRETA_MAX_SEQ_LEN` environment variable override.

### B. ASCII Tokenizer Inflation
- **Issue**: When loading `.gguf` models, the engine falls back to ASCII tokenization (1 character = 1 token) if SentencePiece is not explicitly configured via `.model` file.
- **Impact**: A "32k token" prompt (e.g., repeating "hello ") actually generated ~196k tokens, vastly exceeding even the increased memory limits.
- **Fix**: Benchmark scripts updated to use exact character counts (e.g., `a * 32767`) to simulate precise token counts in fallback mode.

## 3. Preliminary Performance Data (B3.87)

| Metric | Nominal (Det Off) | Strict (Det On) | Impact |
|--------|-------------------|-----------------|--------|
| Decode TPS | 19.57 | 17.40 | -11.1% |
| Wall Time (128t) | 28.19s | 28.93s | +2.6% |

- **Observation**: Strict determinism adds measurable overhead in the decode phase, likely due to serialized atomic operations in HIP kernels.

## 4. Next Steps
1. **Stabilize Remote Node**: Address intermittent SSH connectivity on `129.212.184.200`.
2. **Execute 32k Feasibility (B3.88)**: Re-run with the `MAX_SEQ_LEN` and prompt inflation fixes.
3. **Complexity Analysis**: Run B3.85 to confirm if prefill scaling remains $O(N^2)$ or if recent optimizations improved it to $O(N)$.
