# B3.89 Prefill Microbench Closeout Report (Perf Mode)

**Date:** 2026-02-14  
**Benchmark:** B3.89 - Prefill Kernel Optimization V3/V4 (MI300X Rerun)  
**Status:** ✅ COMPLETED  
**Node:** MI300X (`greta-mi300x-phase5-int4`)

---

## ES - Resumen Ejecutivo

Este informe documenta la re-ejecución del benchmark B3.89 en modo `perf` el 14 de febrero de 2026. Se evaluaron las variantes `baseline` (MQA), `v3` (Q-in-LDS) y `v4` (Q-in-LDS v2) en contextos de 4k, 8k y 16k.

### Observaciones de Rendimiento
1. **Mejora v4 vs v3:** La variante `v4` demuestra una mejora significativa sobre `v3`, con un aumento de velocidad de **~1.75x a 1.78x** en contextos de 8k y 16k.
2. **Comparación con Baseline:** A pesar de las optimizaciones en `v3` y `v4`, la implementación `baseline` sigue siendo la más rápida en estos tamaños de contexto (speedups relativos < 1.0).
3. **Disponibilidad:** La variante `v4` no se evaluó para el contexto de 4096 (N/A).
4. **Configuración de Ejecución:** Se utilizó `B3_89_MODE=perf`, lo cual desactiva las flags de depuración (`HIP_LAUNCH_BLOCKING`, `AMD_SERIALIZE_KERNEL`, `HSA_ENABLE_SDMA`) para maximizar el throughput real y la exactitud temporal.

### Resultados Consolidados (Prefill Median Latency)

| Context | baseline (s) | v3 (s) | v3 Speedup | v4 (s) | v4 Speedup | v4 vs v3 |
|---------|--------------|--------|------------|--------|------------|----------|
| 4096    | 177.805      | 981.37 | 0.18118    | N/A    | N/A        | N/A      |
| 8192    | 403.256      | 3540.68| 0.11389    | 2026.69| 0.19897    | 1.74703  |
| 16384   | 1181.64      | 13366.9| 0.08840    | 7508.25| 0.15738    | 1.78030  |

---

## EN - Executive Summary

This report documents the rerun of the B3.89 benchmark in `perf` mode on February 14, 2026. The `baseline` (MQA), `v3` (Q-in-LDS), and `v4` (Q-in-LDS v2) variants were evaluated across 4k, 8k, and 16k contexts.

### Performance Observations
1. **v4 vs v3 Improvement:** The `v4` variant demonstrates a significant improvement over `v3`, with a speedup of **~1.75x to 1.78x** at 8k and 16k contexts.
2. **Comparison with Baseline:** Despite the optimizations in `v3` and `v4`, the `baseline` implementation remains faster at these context sizes (relative speedups < 1.0).
3. **Availability:** The `v4` variant was not evaluated for the 4096 context (N/A).
4. **Run Configuration:** `B3_89_MODE=perf` was enforced, unsetting debug flags (`HIP_LAUNCH_BLOCKING`, `AMD_SERIALIZE_KERNEL`, `HSA_ENABLE_SDMA`) to ensure real-world throughput and timing accuracy.

### Consolidated Results (Prefill Median Latency)

| Context | baseline (s) | v3 (s) | v3 Speedup | v4 (s) | v4 Speedup | v4 vs v3 |
|---------|--------------|--------|------------|--------|------------|----------|
| 4096    | 177.805      | 981.37 | 0.18118    | N/A    | N/A        | N/A      |
| 8192    | 403.256      | 3540.68| 0.11389    | 2026.69| 0.19897    | 1.74703  |
| 16384   | 1181.64      | 13366.9| 0.08840    | 7508.25| 0.15738    | 1.78030  |

---

## Source of Truth

All metrics derived from the raw artifacts generated on the MI300X node:
- **Root Path:** `artifacts_remote/2026-02-14/b3_89/`
- **Methodology:** Median of successful runs (`exit_status: OK`).

---
**Last Updated:** 2026-02-15  
**Author:** GRETA CORE CI/CD System (B3.89 Post-Processor)
