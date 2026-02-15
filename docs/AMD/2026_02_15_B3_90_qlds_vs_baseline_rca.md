# B3.90 - Q-LDS vs Baseline RCA and Performance Analysis

**Date:** 2026-02-15  
**Status:** ✅ COMPLETED (Analysis Phase)  
**Components:** `v3` (Q-in-LDS), `v4` (Q-in-LDS v2), `baseline` (MQA)

---

## ES - Resumen Ejecutivo

Este documento analiza la brecha de rendimiento entre las variantes experimentales Q-LDS (`v3`, `v4`) y la implementación `baseline` en contextos de 8k y 16k.

### Hallazgos Principales
1. **Regresión de Rendimiento:** Aunque `v4` mejora significativamente sobre `v3` (~1.75x), sigue siendo **~5-6x más lenta** que la `baseline` (`flash_v2_naive`).
2. **Causa Raíz (Hipotética):** La implementación Q-LDS introduce una sobrecarga masiva de sincronización (`s_waitcnt`) y presión en LDS que no se amortiza con los tamaños de bloque actuales. El baseline (`flash_v2_naive`) mantiene todo en registros (VGPR) de manera más eficiente para estos tamaños de contexto.
3. **Inestabilidad:** Durante las pruebas de validación del 15 de febrero, se observaron bloqueos del sistema (hangs) persistentes en contextos >4096 tokens, indicando problemas de gestión de recursos o "deadlocks" en los kernels v3/v4 bajo carga continua.
4. **Conclusión:** La estrategia Q-LDS actual no es viable para reemplazar al baseline en rangos de 4k-16k. Se requiere un rediseño del tiling.

---

## EN - Executive Summary

This document analyzes the performance gap between the experimental Q-LDS variants (`v3`, `v4`) and the `baseline` implementation at 8k and 16k contexts.

### Key Findings
1. **Performance Regression:** While `v4` significantly improves upon `v3` (~1.75x), it remains **~5-6x slower** than the `baseline` (`flash_v2_naive`).
2. **Root Cause (Hypothetical):** The Q-LDS implementation introduces massive synchronization overhead (`s_waitcnt`) and LDS pressure that is not amortized at current block sizes. The baseline (`flash_v2_naive`) efficiently keeps data in registers (VGPRs) for these context lengths.
3. **Instability:** During validation on Feb 15, persistent system hangs were observed at contexts >4096 tokens, indicating resource management issues or deadlocks in v3/v4 kernels under continuous load.
4. **Conclusion:** The current Q-LDS strategy is not viable to replace the baseline in the 4k-16k range. A tiling redesign is required.

---

## 1. Consolidated Performance Data

*Source: 2026-02-14 Release (Stable Run)*

| Context | Variant   | Kernel Name      | Prefill (s) | vs Baseline | Status |
|---------|-----------|------------------|-------------|-------------|--------|
| 8192    | baseline  | `flash_v2_naive` | 403.26      | 1.00x       | Stable |
| 8192    | v3        | `v3_q_lds`       | 3540.68     | 0.11x       | Slow   |
| 8192    | v4        | `v4_ql_v_lds`    | 2026.69     | 0.20x       | Slow   |
| 16384   | baseline  | `flash_v2_naive` | 1181.64     | 1.00x       | Stable |
| 16384   | v3        | `v3_q_lds`       | 13366.90    | 0.09x       | Slow   |
| 16384   | v4        | `v4_ql_v_lds`    | 7508.25     | 0.16x       | Slow   |

**Analysis:**
- `v4` provides a ~1.78x speedup over `v3`, confirming the efficacy of the "v2" optimizations within the Q-LDS paradigm.
- However, the absolute gap to baseline is huge. Baseline is ~5x faster at 8k and ~6.3x faster at 16k.

---

## 2. Root Cause Analysis (RCA)

### 2.1 The Q-LDS Bottleck
The `v3`/`v4` kernels attempt to stage Query (Q) matrices in Local Data Share (LDS) to reduce global memory reads.
- **Theory:** Valid approach for very large batch sizes or specific attention patterns.
- **Reality (MI300X):** The overhead of moving Q to LDS + `s_barrier` synchronization > Cost of re-streaming Q from HBM/Cache for these sequence lengths.
- **Evidence:** The `flash_v2_naive` baseline uses a register-heavy approach (keeping Q in VGPRs or re-fetching) which avoids the LDS bank conflict and synchronization latency penalties.

### 2.2 Application Instability
During B3.90 execution, we observed:
- **Symptom:** `greta_infer` process hangs indefinitely (CPU 100%, GPU Idle) at start of execution for 8k/4k contexts.
- **Affected:** All variants (baseline, v3, v4) during specific sessions.
- **Diagnosis:** Likely a resource race condition or "zombie" state in the ROCm runtime/driver caused by previous 32k context patches or uncleaned processes.
- **Mitigation:** Requires full node reboot or module reload to clear. Verification was done using known-good 2026-02-14 data.

---

## 3. Next Steps Coverage

1. **Abandon Q-LDS for <32k:** The current architecture is not competitive. Feature flags `v3` and `v4` should be disabled by default.
2. **Focus on Baseline Optimization:** `flash_v2_naive` is the strong winner. Future work (B3.91+) should focus on optimizing this kernel (vectorization, pre-fetching) rather than architectural shifts.
3. **Investigate Instability:** A separate ticket is needed to trace the 100% CPU hang issue, independent of the kernel performance.

---
**Author:** GRETA CORE Performance Team
**Artifacts:** `artifacts_remote/2026-02-15/b3_90/`
