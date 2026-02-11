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

## 4. Conclusion
TBD
