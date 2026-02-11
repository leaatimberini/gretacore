# MI300X Prefill/Decode RCA Summary (B3.85–B3.88)

**Date**: 2026-02-10  
**Context**: Hito de escalado a 32k y caracterización de kernels.

## 1. Context Length Milestone (B3.88)
- **Logro**: Procesamiento exitoso de **32,768 tokens** en una sola GPU MI300X.
- **Tiempo**: 2542.79 s.
- **Veredicto**: **PASS_32K_FEASIBLE**. El motor es capaz de direccionar y computar 32k tokens, superando el límite previo de 2k.

## 2. Prefill Complexity (B3.85)
Se observó un escalado de tiempo de prefill con tendencia **$O(N^2)$**:
- 4k -> 8k: 5.2x
- 8k -> 16k: 4.2x
- 16k -> 32k: 5.5x (Degradación de eficiencia).

**RCA Técnico**: El kernel `flash_attention_prefill_kernel` es **bandwidth-bound**. Carga repetidamente los mismos bloques de Key/Value desde la memoria global (VRAM) sin aprovechar la memoria compartida (LDS).

## 3. Decode Performance (B3.87)
El uso de flags de determinismo estricto (`HIP_LAUNCH_BLOCKING=1`) penaliza el rendimiento de decode en un **~11.1%**.
- Sin determinismo: 19.57 TPS.
- Con determinismo: 17.40 TPS.

## 4. Attention Implementation (B3.86)
- **Implementación Detectada**: `flash_v2_naive`.
- **Estado**: Funcional pero no optimizada para el ancho de banda masivo de las MI300X.

## 5. Next Milestone: B3.89
Se ha iniciado el plan de optimización enfocado en:
- Implementación de **tiling** de bloques en LDS.
- Reducción de lecturas globales por un factor de 64x.
- Objetivo: Reducir el prefill de 32k de 42 minutos a **menos de 10 minutos**.
