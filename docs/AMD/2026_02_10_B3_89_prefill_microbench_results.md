# B3.89 — Prefill Kernel Optimization Results (MI300X)

**Date**: 2026-02-10  
**Phase**: Optimization Milestone B3.89  

## 1. Baseline Performance (Baseline)
Recorded with `flash_v2_naive` implementation.

| Context | Prefill Time (s) | Tokenize (s) | Load (s) | Scaling (to 4k) |
|---|---|---|---|---|
| 4,096 | 22.768 | 0.000 | 21.865 | 1.0x |
| 8,192 | 114.265 | 0.000 | 21.624 | 5.0x |
| 16,384 | 469.749 | 0.000 | 21.380 | 20.6x |

## 2. Optimized Performance (LDS Tiling v1)
Recorded with `GRETA_PREFILL_LDS_TILING=1`.

| Context | Prefill Time (s) | Tokenize (s) | Load (s) | Speedup |
|---|---|---|---|---|
| 4,096 | 206.737 | 0.000 | 22.185 | **0.11x (Regression)** |
| 8,192 | TBD | TBD | TBD | TBD |
| 16,384 | TBD | TBD | TBD | TBD |

## 3. Optimized Performance (Segmented v2)
Recorded with `GRETA_PREFILL_SEGMENTED=1` (SEG=8, Block=32).

| Context | Prefill Time (s) | Speedup | Notes |
|---|---|---|---|
| 4,096 | > 180s (TIMEOUT) | **< 0.12x (Regression)** | Failed strict no-spill gate. |

## 4. Optimized Performance (Q-LDS v3)
Recorded with `GRETA_PREFILL_Q_LDS=1`.

| Context | Prefill Time (s) | Speedup | Notes |
|---|---|---|---|
| 4,096 | ~18.7s | **1.21x** | **SCRATCH=0**. Gate Passed. |
| 8,192 | TBD | | |
| 16,384 | TBD | | |

## 5. Analysis and Diagnosis (V3)
- **Scratch**: **0 bytes**.
- **VGPR**: 19 (very lean).
- **Strategy**: Q, K, V all staged in LDS. Inner loop overhead is mainly LDS access + compute.
- **Scaling**: Re-loading K/V for each output segment (8 passes for SEG=16) is the cost.
  - But eliminating scratch spill provided huge win (timeout -> 18s).

## 6. V4 Exploration (SEG=32)
- **Goal**: Reduce K-load redundancy. V3 reloads K 8 times (128/16). V4 reloads 4 times (128/32).
- **LDS**: `sV[32][32]` (4KB). Total LDS consistent with V3 (~36KB).
- **Registers**: Uses 32 accumulators (`o0..o31`). Expected 0-scratch spill based on V3 occupancy.
- **Status**: Code implemented. Awaiting MI300X execution.

## 7. Conclusion
V3 (Q-LDS) has successfully broken the scratch-spill bottleneck on MI300X, providing the first valid optimized path for long-context prefill. V4 is expected to further improve bandwidth efficiency.

**Decision**: V3 is promoted to experimental status for long-context prefill.

## 8. Connectivity Blocker (2026-02-11)
- **Status**: **RUN_BLOCKED**
- **Reason**: Persistent SSH timeouts to MI300X node `129.212.184.200`.
- **Action**: All infrastructure (single-shot executor, summary analyzer, vram sampling) and code (V3/V4) are merged and verified via local build tests. 
- **Command Pending**:
  ```bash
  bash tools/benchmarks/run_b3_89_prefill_microbench.sh 129.212.184.200 --date 2026-02-11 \
    --variants "v3,v4" --single-shot \
    --contexts "4096,8192,16384" \
    --repeat "4096:2,8192:1,16384:1"
  ```
- **Evidence Ready**: V3 (4k) results show **1.21x speedup** and **0 scratch bytes**. 8k/16k and V4 scaling verification requires node restoration.

## 9. Run Robustness Fix (2026-02-11)
- **Issue**: Previous runs reported 0s prefill time due to silent context length mismatch (`model_config.hpp` fixed at 2048).
- **Fix**:
  1. **Force Patch**: Remote executor now sed-patches `model_config.hpp` to 32768 before build.
  2. **Verification**: Script aborts if patch fails.
  3. **Guardrails**: Runs with 0s prefill are marked as FAIL.
- **Status**: Infrastructure patched. Ready for definitive 16k run.
