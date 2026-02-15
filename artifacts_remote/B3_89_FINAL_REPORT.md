# B3.89 - Prefill Kernel Optimization V3/V4 Final Report

This document tracking the evolution and final results of the B3.89 prefill micro-benchmarks on AMD MI300X.

---

## 2026-02-14 perf-mode rerun (MI300X)

### Configuration
- **Mode:** `perf` (Debug flags unset)
- **Determinism:** Enabled
- **Kernel Implementation:**
  - `baseline`: `flash_v2_naive`
  - `v3`: `v3_q_lds`
  - `v4`: `v4_ql_v_lds`
- **Node:** MI300X Phase 5 (8x GPUs)

### Consolidated Results (Prefill Median Latency)

| Context | baseline (s) | v3 (s) | v3 Speedup | v4 (s) | v4 Speedup | v4 vs v3 |
|---------|--------------|--------|------------|--------|------------|----------|
| 4096    | 177.805      | 981.37 | 0.18118    | N/A    | N/A        | N/A      |
| 8192    | 403.256      | 3540.68| 0.11389    | 2026.69| 0.19897    | 1.74703  |
| 16384   | 1181.64      | 13366.9| 0.08840    | 7508.25| 0.15738    | 1.78030  |

### Conclusion
The February 14th rerun confirms that **v4 represents a significant technical leap over v3** (optimizing the Q-in-LDS path to achieve ~1.78x speedup at 16k context). However, for the current prefill sizes, the **baseline (MQA) remains significantly faster** than both experimental variants.

---

## Previous Results (2026-02-12)

[See artifacts_remote/2026-02-12/b3_89/summary.json for raw data]
- Core results showed prefill issues due to `GRETA_MAX_SEQ_LEN` limits, which were addressed in the 2026-02-14 rerun.
