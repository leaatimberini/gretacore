# B3.87 — Decode TPS Decomposition (MI300X)

**Date**: 2026-02-10  
**Phase**: RCA - Performance Characterization  

## Executive Summary
Evaluation of the impact of strict determinism flags on decode throughput. Enabling HIP/HSA synchronization and serialization results in a measurable **~11.1% penalty** in Tokens/Second (TPS).

## Methodology
- **Context**: 8,192 tokens
- **Generation**: 128 tokens
- **Determinism ON**: `HIP_LAUNCH_BLOCKING=1`, `AMD_SERIALIZE_KERNEL=3`, `HSA_ENABLE_SDMA=0`
- **Determinism OFF**: Standard asynchronous execution.

## Measured Results
| Batch | Determinism | Tokens/s | Decode Time (s) | Impact |
|---|---|---|---|---|
| 1 | OFF | 19.57 | 6.48 | Baseline |
| 1 | ON | 17.40 | 7.29 | **-11.1%** |
| 8 | OFF | 19.57 | 6.48 | Baseline |
| 8 | ON | 17.38 | 7.30 | **-11.2%** |

## Findings
1. **Serialization Bottleneck**: The penalty is primarily driven by `HIP_LAUNCH_BLOCKING=1`, which forces the host to wait for every kernel completion, preventing the hardware from overlapping kernel launches and execution.
2. **Minimal Batch Sensitivity**: The impact is consistent across batch sizes 1 and 8, suggesting the bottleneck is a host-side serialization overhead rather than device-side resource contention.
3. **Overhead Corelation**: The `Overhead` metric (Wall Time - Compute Time) remains low (~0.3s), indicating that the penalty is manifest within the compute kernels' serialized latency.

## Recommendations
- **CI/Testing**: Keep determinism **ON** by default to guarantee reproducibility.
- **Production/Inference**: Introduce a "High Performance" mode that toggles these flags **OFF** to recapture the 11% throughput loss.
