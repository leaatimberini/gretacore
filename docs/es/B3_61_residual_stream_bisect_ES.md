# B3.61 Análisis de Bisección del Residual Stream

**Fecha:** 2026-02-06  
**Fase:** GRETA Fase 3  
**Objetivo:** Identificar el primer tensor divergente en el residual stream que causa colapso de inferencia en contextos extendidos

---

## 1. Resumen Ejecutivo

B3.61 continúa la investigación de los modos de fallo de inferencia de contexto largo identificados en fases anteriores. El objetivo principal es **identificar determinísticamente el primer tensor que falla** (`FIRST_FAIL`): el primer punto en el residual stream donde el comportamiento diverge entre ejecuciones base exitosas y ejecuciones fallidas de contexto extendido.

### Puntos Críticos de Investigación

| Posición | Rango de Contexto | Hipótesis |
|----------|-------------------|-----------|
| 826 | 16K | Primer punto de fallo observado |
| 1652 | 32K | Segundo punto de fallo observado |
| Último token | EOS | Posible divergencia fin de secuencia |

### Capas Analizadas (Mínimo Requerido)

L0, L1, L2, L4, L8 con expansión automática según sea necesario.

---

## 2. Metodología

### 2.1 Puntos de Trazado del Residual Stream

El residual stream se instrumenta en las siguientes etapas en orden estricto:

| Etapa | Descripción | Validación |
|-------|-------------|------------|
| `embed_out` | Salida del embedding (verificación B3.59) | SHA256 + nz_count |
| `residual_pre_attn` | Estado residual antes de attention | SHA256 + nz_count |
| `attn_in` | Entradas del mecanismo attention (Q/K/V) | SHA256 + nz_count |
| `q_pre_rope` | Query antes de aplicación RoPE | SHA256 + nz_count |
| `k_pre_rope` | Key antes de aplicación RoPE | SHA256 + nz_count |
| `q_post_rope` | Query después de aplicación RoPE | SHA256 + nz_count |
| `k_post_rope` | Key después de aplicación RoPE | SHA256 + nz_count |
| `attn_out` | Salida de attention tras softmax | SHA256 + nz_count |
| `residual_post_attn` | Residual tras adición de attention | SHA256 + nz_count |
| `ffn_norm_in` | Entrada FFN tras layer norm | SHA256 + nz_count |
| `mlp_out` | Salida de la red feed-forward | SHA256 + nz_count |
| `residual_post_mlp` | Residual final tras MLP | SHA256 + nz_count |
| `logits` | Logits de salida final | SHA256 + top-5 tokens |

### 2.2 Clasificación de Causa Raíz

Cada fallo se clasifica en exactamente una categoría:

| Categoría | Descripción | Etapas |
|-----------|-------------|--------|
| `ROUTING/SELECTION` | Divergencia de buffer antes de attention | `embed_out`, `residual_pre_attn` |
| `ATTN_KERNEL_INPUTS` | Divergencia en proyecciones Q/K/V | `attn_in`, `q_pre_rope`, `k_pre_rope` |
| `ATTENTION_MECHANISM` | Divergencia dentro del cómputo attention | `q_post_rope`, `k_post_rope`, `attn_out` |
| `RESIDUAL_ADD` | Divergencia durante adición residual | `residual_post_attn`, `residual_post_mlp` |
| `FFN_NORM_PATH` | Divergencia en normalización FFN | `ffn_norm_in` |
| `MLP_OUTPUT` | Divergencia en cómputo MLP | `mlp_out` |
| `UNKNOWN` | No clasificable con instrumentación actual | - |

---

## 3. Prompts de Validación

| Prompt | Longitud Contexto | Propósito |
|--------|-------------------|-----------|
| `p0_short` | ~5 tokens | Baseline contexto corto |
| `p6_len_16` | ~827 tokens | Longitud media (equivalente 16K) |
| `p6_len_32` | ~1653 tokens | Longitud extendida (equivalente 32K) |

---

## 4. Entregables Generados

| Entregable | Estado |
|------------|--------|
| `tools/benchmarks/run_b3_61_mi300x.sh` | ✅ Implementado |
| `tools/benchmarks/analyze_b3_61_residual_stream_bisect.py` | ✅ Implementado |
| `docs/AMD/2026_02_06_B3_61_residual_stream_bisect.md` | ✅ Implementado |
| `docs/es/B3_61_residual_stream_bisect_ES.md` | ✅ Implementado |
| `docs/PROGRESS.md` | ⏳ Pendiente de ejecución |
| `artifacts_remote/YYYY-MM-DD/b3_61/` | ⏳ Pendiente de ejecución |

---

## 5. Resultados Esperados (Post-Ejecución)

### Tabla FIRST_FAIL (Formato Esperado)

| Prompt | Posición | Capa | Tensor | Coincidencia Hash | MAE | FIRST_FAIL (S/N) |
|--------|----------|------|--------|-------------------|-----|------------------|
| p6_len_16 | 826 | 0 | residual_pre_attn | NO | 0.142 | **SÍ** |
| p6_len_16 | 826 | 1 | residual_pre_attn | NO | 0.138 | NO |
| p6_len_16 | 826 | 2 | residual_pre_attn | NO | 0.141 | NO |
| p6_len_32 | 1652 | 0 | residual_pre_attn | NO | 0.201 | **SÍ** |

### Resumen de Causa Raíz

| Categoría | Ocurrencias | Porcentaje |
|-----------|-------------|------------|
| ROUTING/SELECTION | 0 | 0% |
| ATTN_KERNEL_INPUTS | 0 | 0% |
| ATTENTION_MECHANISM | 0 | 0% |
| RESIDUAL_ADD | 0 | 0% |
| FFN_NORM_PATH | 0 | 0% |
| MLP_OUTPUT | 0 | 0% |

---

## 6. Recomendaciones para B3.62

### 6.1 Basado en Identificación de FIRST_FAIL

Si `FIRST_FAIL` es **RESIDUAL_PRE_ATTN en Capa 0, Posición 826**:
> B3.62 debe investigar la codificación de posición del embedding en contextos extendidos y probar modificaciones del rotary position embedding.

Si `FIRST_FAIL` es **ATTN_IN/QKV en Capa 0**:
> B3.62 debe auditar los kernels de proyección query/key para errores específicos de posición y validar la aplicación de pesos Q/K/V en posiciones límite.

Si `FIRST_FAIL` es **ATTN_OUT/RoPE**:
> B3.62 debe investigar la mecánica de aplicación RoPE en contextos extendidos y verificar la estabilidad numérica del cómputo de scores de attention.

Si `FIRST_FAIL` es **RESIDUAL_POST_ATTN**:
> B3.62 debe depurar el kernel de adición residual para problemas específicos de posición y verificar consistencia de dtype entre capas.

### 6.2 Hipótesis Verificables

1. **Hipótesis de Codificación de Posición**: La divergencia se origina en la aplicación RoPE en frecuencias específicas
2. **Hipótesis de KV Cache**: La coherencia del cache se rompe en umbrales de posición
3. **Hipótesis de Precisión Numérica**: Errores de acumulación FP16 se acumulan más allá de cierta longitud
4. **Hipótesis de Layout de Memoria**: Patrones de acceso strided causan corrupción en offsets específicos

---

## 7. Próximos Pasos Inmediatos

1. **Sincronización con GitHub y Remote MI300X**
2. **Ejecutar pipeline en MI300X**: `./tools/benchmarks/run_b3_61_mi300x.sh`
3. **Transferir artifacts a local**: `scp` del tgz
4. **Ejecutar análisis local**: `python analyze_b3_61_residual_stream_bisect.py`
5. **Generar reporte final**: Completar `b3_61_analysis.txt`
6. **Actualizar PROGRESS.md**

---

**Versión del Documento:** 1.0  
**Última Actualización:** 2026-02-06  
**Autor:** Equipo de Ingeniería GRETA CORE
