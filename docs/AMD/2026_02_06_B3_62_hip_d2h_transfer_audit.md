# B3.62: HIP D2H Transfer Audit

**Fecha:** 2026-02-06  
**Commit:** 303b634  
**Modelo:** greta-v1.gguf (Llama-2-7B compatible)  
**Autor:** L.E.T / Leandro Emanuel Timberini

---

## Resumen Ejecutivo

Este documento presenta los hallazgos de la auditoría forense del error `hipMemcpy D2H failed: an illegal memory access was encountered` que ocurre durante la inferencia con el modelo compatible greta-v1.gguf.

El error ocurre específicamente después de que los pesos del modelo se cargan exitosamente, indicando que el problema está en las transferencias Device→Host (D2H) durante la fase de inferencia, no en la carga del modelo.

---

## Error Observado

```
Generation error: hipMemcpy D2H failed: an illegal memory access was encountered
```

### Contexto del Error
- Ocurre después de: `Weights loaded and config updated (vocab size: 128256)`
- Modelo: `Llama-2-7B (6.73815B params)`
- Binary: `greta_infer`

---

## Transferencias D2H Críticas Identificadas

### 1. post_wo_trace (Línea 1008-1009)
```cpp
hipMemcpyAsync(host.data(), ptr, sample_n * sizeof(float),
               hipMemcpyDeviceToHost, stream);
hipStreamSynchronize(stream);
```
**Contexto:** Copia de logits para sampling

### 2. RMSNorm Trace (Líneas 1069-1077)
```cpp
hipMemcpyAsync(in_host.data(), in_ptr, sample_n * sizeof(float), hipMemcpyDeviceToHost, stream);
hipMemcpyAsync(out_host.data(), out_ptr, sample_n * sizeof(float), hipMemcpyDeviceToHost, stream);
hipMemcpyAsync(w_host.data(), weight.data(), sample_n * sizeof(float), hipMemcpyDeviceToHost, stream);
hipStreamSynchronize(stream);
```
**Contexto:** Trazabilidad de RMSNorm input/output/weights

### 3. Attention Trace (Líneas 2146-2147, 2288-2301)
```cpp
hipMemcpy(host.data(), attn_out_buf.data(), q_bytes, hipMemcpyDeviceToHost);
hipMemcpyAsync(q_host.data(), q_head, Dh * sizeof(float), hipMemcpyDeviceToHost, hip_stream);
hipMemcpyAsync(k_host.data(), k_head_base, seq_len_used * Dh * sizeof(float), hipMemcpyDeviceToHost, hip_stream);
hipMemcpyAsync(v_host.data(), v_head_base, seq_len_used * Dh * sizeof(float), hipMemcpyDeviceToHost, hip_stream);
hipMemcpyAsync(attn_host.data(), attn_head, Dh * sizeof(float), hipMemcpyDeviceToHost, hip_stream);
```
**Contexto:** Trazabilidad de attention

### 4. MLP Trace (Líneas 2811-2814)
```cpp
hipError_t err_a = hipMemcpy(attn_out_host.data(), attn_vec, D * sizeof(float), hipMemcpyDeviceToHost);
hipError_t err_b = hipMemcpy(wo_out_host.data(), wo_vec, D * sizeof(float), hipMemcpyDeviceToHost);
```
**Contexto:** Trazabilidad de salida MLP

### 5. KV Cache Dump (Líneas 3170-3173)
```cpp
hipMemcpy(k_cache_host.data(), k_cache_layer, kv_layer_stride_bytes, hipMemcpyDeviceToHost);
hipMemcpy(v_cache_host.data(), v_cache_layer, kv_layer_stride_bytes, hipMemcpyDeviceToHost);
```
**Contexto:** Dump de KV cache para debugging

---

## Hipótesis Testeadas

### 1. Byte Size Mismatch
**Ubicación:** `block_scheduler.cpp:1008, 1069, 2146, 2811, 3124, 3170`

**Evidencia:**
- Múltiples sitios donde `sample_n * sizeof(float)` se calcula
- Cálculos como `static_cast<size_t>(seq_len_used) * Dh * sizeof(float)`

**Resultado:** REQUIERE VERIFICACIÓN EN TIEMPO DE EJECUCIÓN

### 2. Use-after-free of GPU Buffers
**Ubicación:** `block_scheduler.cpp:1187`, `greta_runtime_hip.cpp:37-41`

```cpp
bool GretaMemoryHip::copy_to_host(void *dst, size_t size) const {
  if (!ptr_ || size > size_)
    return false;
  return hipMemcpy(dst, ptr_, size, hipMemcpyDeviceToHost) == hipSuccess;
}
```

**Evidencia:**
- `ptr_` podría ser `nullptr` después de carga de modelo
- `size` podría exceder `size_` allocation

**Resultado:** REQUIERE VERIFICACIÓN - el check existente podría no ser suficiente

### 3. Stream Not Synchronized Before Memcpy
**Ubicación:** `block_scheduler.cpp:1008-1010, 1069-1077, 2274-2302`

**Evidencia:**
- `hipMemcpyAsync` sin sincronización explícita en algunos casos
- Patrón problemático: múltiples async copies seguidos + 1 sync

**Resultado:** ALTA PROBABILIDAD - sync podría no ser suficiente para todas las operaciones

### 4. Misaligned or Invalid Device Pointers
**Ubicación:** `block_scheduler.cpp:2146, 2274, 2290, 3124`

**Evidencia:**
- Offset calculations: `x + token_index * D`
- Cálculos: `static_cast<size_t>(seq_len_used) * Dh * sizeof(float)`

**Preocupaciones:**
- `token_index` podría estar fuera de rango
- `seq_len_used` podría exceder `max_seq_len`

**Resultado:** ALTA PROBABILIDAD - necesita bounds checking

### 5. Incorrect Reuse of Staging Buffers
**Ubicación:** `block_scheduler.cpp:1006-1011`

```cpp
std::vector<float> host(sample_n, 0.0f);
hipMemcpyAsync(host.data(), ptr, sample_n * sizeof(float), hipMemcpyDeviceToHost, stream);
hipStreamSynchronize(stream);
```

**Evidencia:** Vector recreado en cada llamada - no hay reuse

**Resultado:** DESCARTADO - El patrón es correcto

### 6. Overlapping Async Copies
**Ubicación:** `block_scheduler.cpp:1069-1077, 2274-2302`

**Evidencia:**
- 2-4 `hipMemcpyAsync` paralelas al mismo stream
- Sin dependencias explícitas entre ellas

**Resultado:** MEDIA PROBABILIDAD - podría causar race conditions

---

## Hipótesis Principal (Root Cause)

### Sincronización de Stream Insuficiente

La hipótesis principal es que la sincronización de stream no es suficiente para garantizar que todas las operaciones GPU hayan completado antes de las transferencias D2H.

**Evidencia:**
1. Múltiples `hipMemcpyAsync` seguidos sin sync intermediario
2. El error ocurre después de carga de modelo (primeras D2H copies)
3. Patrones de código que asumen completitud sin verificación

### Hipótesis Secundaria

### Offset Calculation Overflow

**Evidencia:**
1. Cálculos complejos de offset con casts de tamaño
2. Si `seq_len_used` excede el rango válido, el puntero sería inválido
3. Esto causaría "illegal memory access" en `hipMemcpy`

---

## Instrumentación Implementada

### Header de Debug: `tools/inference/src/greta_trace_memcpy.hpp`

```cpp
#ifdef GRETA_TRACE_MEMCPY
#define GRETA_TRACE_D2H_BEFORE(tensor_name, src, bytes, stream) do { \
    fprintf(stderr, "[D2H TRACE] %s:before\n", tensor_name); \
    fprintf(stderr, "  src_ptr=%p\n", (void*)(src)); \
    fprintf(stderr, "  bytes=%zu\n", (size_t)(bytes)); \
    fprintf(stderr, "  stream=%p\n", (void*)(stream)); \
} while(0)

#define GRETA_TRACE_D2H_AFTER(tensor_name, dst, bytes, stream) do { \
    fprintf(stderr, "[D2H TRACE] %s:after\n", tensor_name); \
    fprintf(stderr, "  dst_ptr=%p\n", (void*)(dst)); \
    fprintf(stderr, "  bytes=%zu\n", (size_t)(bytes)); \
} while(0)

#define GRETA_TRACE_D2H_ERROR(tensor_name, src, dst, bytes, err) do { \
    fprintf(stderr, "[D2H FATAL] %s:FAILED\n", tensor_name); \
    fprintf(stderr, "  src_ptr=%p\n", (void*)(src)); \
    fprintf(stderr, "  dst_ptr=%p\n", (void*)(dst)); \
    fprintf(stderr, "  bytes=%zu\n", (size_t)(bytes)); \
    fprintf(stderr, "  hipError=%s\n", hipGetErrorString(err)); \
    abort(); \
} while(0)
#endif
```

---

## Archivos Modificados/Creados

| Archivo | Tipo | Descripción |
|---------|------|-------------|
| `tools/inference/src/greta_trace_memcpy.hpp` | CREADO | Header de instrumentación D2H |
| `tools/benchmarks/run_b3_62_d2h_audit.sh` | CREADO | Script de ejecución B3.62 |
| `artifacts_remote/2026-02-06/b3_62/b3_62_analysis.txt` | CREADO | Análisis preliminar |

---

## Próximos Pasos

1. **Compilar con instrumentación:**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Debug -DGRETA_TRACE_MEMCPY=ON
   ```

2. **Ejecutar con tracing detallado:**
   ```bash
   export GRETA_TRACE_MEMCPY=1
   ./greta_infer --model models/greta-v1.gguf --prompt-file p0_short.txt --max-tokens 5
   ```

3. **Capturar y analizar:**
   - `src_ptr`, `dst_ptr`, `bytes` para cada D2H copy
   - `hipGetLastError` antes y después de cada copy
   - Errores de sincronización de stream

4. **Comparar con B3.61** para identificar diferencias

---

## Conclusiones Preliminares

| Hipótesis | Probabilidad | Requiere |
|-----------|--------------|----------|
| Stream sync insuficiente | ALTA | Instrumentación |
| Offset overflow | ALTA | Bounds check |
| Byte size mismatch | MEDIA | Instrumentación |
| Use-after-free | MEDIA | Verificación ptr |
| Overlapping async | BAJA | Revisión código |
| Staging buffer reuse | DESCARTADO | - |

---

## Firma

**Análisis realizado:** L.E.T / Leandro Emanuel Timberini  
**Fecha:** 2026-02-06  
**Estado:** PENDIENTE EJECUCIÓN CON INSTRUMENTACIÓN  
**Root Cause:** POR CONFIRMAR (requiere ejecución con GRETA_TRACE_MEMCPY=ON)
