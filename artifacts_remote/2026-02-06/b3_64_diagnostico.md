# B3.64 - Diagnóstico de Falla (2026-02-06)

## Resumen
El archivo `greta-v1.gguf` **NO se borra automáticamente**. El problema es un bug en el binary C++.

## Problemas Identificados

### 1. El archivo NO se borra solo
- ✅ Verificado: El archivo `greta-v1.gguf` (4.92GB) está presente en `/root/gretacore/models/`
- ✅ Espacio en disco: 573GB disponibles (18% usado)
- ✅ No hay cron jobs de limpieza automática

### 2. Error Real: `hipMemcpy D2H failed`
```
Generation error: hipMemcpy D2H failed: an illegal memory access was encountered
```

**Causa:** Bug en el código C++ del binary [`greta_infer`](tools/inference/greta_infer) durante la copia de resultados GPU→CPU.

**Stacktrace del error:**
```
- Model loads successfully
- Weights load successfully (32 layers)
- Generator initializes
- FAILURE: During token generation output copy
```

### 3. Problemas Adicionales
- ❌ El binary `greta_infer` perdió permisos de ejecución (`Permission denied`)
- ❌ Múltiples procesos ejecutando B3.64 simultáneamente (race conditions)

## Archivos Rescatados
Location: `artifacts_remote/_rescued_from_remote/b3_64/`

## Solución Requerida
Se necesita hacer **debug del código C++** en:
- `src/backend/hip/memory.cpp` - funciones de hipMemcpy
- `src/backend/hip/device_buffer.cpp` - gestión de buffers GPU

El error indica que se intenta acceder a memoria GPU que ya fue liberada o nunca fue asignada correctamente.
