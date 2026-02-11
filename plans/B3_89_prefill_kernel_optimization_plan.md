# B3.89: Prefill Kernel Optimization Plan (MI300X)

**Status**: PROPOSED  
**Target Milestone**: 4x speedup for 32k context prefill  
**Root Cause**: `flash_attention_prefill_kernel` is bandwidth-bound due to repeated global memory loads and lack of LDS tiling.

## 1. Objective
Transform the current naive prefill kernel into a tiled implementation that maximizes MI300X throughput by:
1. Reusing Key/Value blocks in LDS (Shared Memory).
2. Tiling the Query dimension.
3. Using vectorized loads (e.g., `float4`).
4. Implementing a Split-K approach for large sequence lengths if necessary.

## 2. Technical Strategy

### A. Tiling and LDS Staging
- **Blocks**: Divide the sequence length into blocks of $B_r$ (rows/Queries) and $B_c$ (cols/Keys).
- **Shared Memory**: Load a block of Keys ($B_c \times D$) and Values ($B_c \times D$) into LDS once and reuse them for all Queries in the block $B_r$.
- **Dimensions**: Target $B_r = 64, B_c = 64$ for $D=128$.

### B. Computational Flow (Online Softmax)
Continue using the online softmax algorithm (Milakov & Gimelshein) to avoid storing the full $N \times N$ attention matrix, but perform the updates in-register for the tiled blocks.

### C. Resource Occupancy
Adjust block sizes to ensure high occupancy on MI300X (up to 320 CUs). Each CU should have multiple wavefronts in flight to hide memory latency.

## 3. Success Metrics
| Metric | Baseline (B3.88) | Target (B3.89) |
|---|---|---|
| 32k Prefill Time | 2542.79 s | < 600 s |
| Scaling | $O(N^2)$ (heavy constant) | $O(N^2)$ (optimized constant) |

## 4. Implementation Steps
1. **Microbench Harness**: Create `run_b3_89_prefill_microbench.sh` to isolate prefill time.
2. **Kernel Prototype**: Rewrite `flash_attention_prefill_kernel` in `attention_kernels.hip`.
3. **Verification**: Match logits with the naive implementation using B3.69 guardrails.
4. **Benchmarking**: Sweep contexts 4k-32k on MI300X.

## 5. Risks
- Register pressure from large LDS tiles.
- Complex boundary conditions for non-multiple of block sizes.
