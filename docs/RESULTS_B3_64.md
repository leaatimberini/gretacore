# B3.64 Numerical Drift Audit - Results

**Fecha**: 2026-02-06  
**Commit**: `28a5e49`

## Resumen Ejecutivo

**ESTADO**: **FAILED** ❌

El benchmark B3.64 **no pudo completarse** debido a un error de HIP D2H que impidió la generación de tokens. Este es el mismo error que fue documentado en B3.63 y supuestamente "arreglado" con `d2h_safe.hpp`.

## Root Cause

**El fix B3.63 (`d2h_safe.hpp`) es INCOMPLETO.**

### Evidencia

El código tiene múltiples llamadas `hipMemcpy` DIRECTAS en [`block_scheduler.cpp`](src/inference/src/block_scheduler.cpp) que **NO** están protegidas por el wrapper `greta_d2h_safe`:

| Línea | Tipo | Estado |
|-------|------|--------|
| 2221 | `hipMemcpy` D2H attn_out | **DIRECTO** |
| 2256-2259 | `hipMemcpy` D2H mfma/valu | **DIRECTO** |
| 2364-2376 | `hipMemcpy` D2H attention | **DIRECTO** |
| 2886-2889 | `hipMemcpy` D2H attn_out/wo | **DIRECTO** |
| 3199-3210 | `hipMemcpy` D2H q/x_out | **DIRECTO** |
| 3245-3260 | `hipMemcpy` D2H k/v cache | **DIRECTO** |
| 3559-3564 | `hipMemcpy` D2H qk/softmax/stats | **DIRECTO** |
| 3738-3741 | `hipMemcpy` D2H v_row/col | **DIRECTO** |
| 3935-3940 | `hipMemcpy` D2H k/v cur sample | **DIRECTO** |

### Log de Error

```
Generation error: hipMemcpy D2H failed: an illegal memory access was encountered
```

## Métricas de Drift

**NO DISPONIBLES** - El benchmark no pudo generar resultados debido al error.

## Artifacts

| Artifact | Estado |
|----------|--------|
| `artifacts_remote/2026-02-06/b3_64/run/p0_short.log` | Generado (con error) |
| `artifacts_remote/2026-02-06/b3_64/run/p6_len_16.log` | Falló al cargar modelo |
| `artifacts_remote/2026-02-06/b3_64/run/p6_len_32.log` | Falló al cargar modelo |

## Anomalías Detectadas

1. **HIP D2H Illegal Memory Access**: El error ocurre durante la fase de generación después de que los pesos se cargan exitosamente
2. **Fix Incompleto**: `d2h_safe.hpp` existe pero no está siendo usado en todos los paths D2H

## Recomendaciones

### Inmediato (para permitir B3.64)

1. **Auditar todas las llamadas `hipMemcpy` en `block_scheduler.cpp`**:
   ```bash
   grep -rn "hipMemcpy" src/inference/src/block_scheduler.cpp | grep -v "safe_hipMemcpy"
   ```

2. **Reemplazar todas las llamadas D2H directas** con `greta_d2h_safe::safe_hipMemcpyAsync`

3. **Verificar que no haya otras llamadas directas** en otros archivos `.cpp`

### Mediano Plazo

1. **Agregar linter/check en CI** que detecte llamadas `hipMemcpy` directas
2. **Crear wrapper de `#define`** que redirinja `hipMemcpy` a la versión safe
3. **Agregar tests de integración** que ejecuten inference completo

## Archivos Modificados

- `docs/PROGRESS.md` - Actualizado estado B3.64
- `docs/RESULTS_B3_64.md` - Este documento

## Notas

- El benchmark B3.64 fue diseñado para auditar drift numérico entre prefill/decode
- La infraestructura (scripts, analyzer) está lista pero no puede ejecutarse hasta que se arregle HIP D2H
- El fix B3.63 (`d2h_safe.hpp`) necesita ser aplicado a TODAS las llamadas D2H

**Signed**: L.E.T / Leandro Emanuel Timberini
