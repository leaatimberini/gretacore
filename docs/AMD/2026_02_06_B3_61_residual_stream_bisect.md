# B3.61: Residual Stream Bisect

**Fecha**: 2026-02-06  
**Commit**: `e09989c`  
**Modelo**: greta-v1.gguf (Llama-2-7B compatible)  
**Binary**: `tools/inference/greta_infer` (con fix B3.63)

## Resumen

B3.61 ejecuta residual stream bisect para verificar integridad de trazas. Esta ejecución confirma que el fix B3.63 (wrappers seguros para `hipMemcpyAsync`) resuelve el problema de illegal memory access en transferencias D2H.

## Configuración

- **Prompts**: p0_short, p6_len_16, p6_len_32
- **Layers**: 0, 1, 2, 4, 8
- **Max tokens**: 5

## Resultados de Ejecución

| Prompt | Prompt Tokens | Gen Tokens | Status | Tokens/sec |
|--------|---------------|------------|--------|------------|
| p0_short | 4 | 5 | OK ✓ | 16.15 |
| p6_len_16 | 343 | 5 | OK ✓ | 4.07 |
| p6_len_32 | 344 | 5 | OK ✓ | 4.06 |

## Dependencias

- **B3.63**: HIP D2H Fix - Wrappers seguros para 13 llamadas `hipMemcpyAsync`
  - Archivo: [`src/inference/include/gcore/inference/d2h_safe.hpp`](src/inference/include/gcore/inference/d2h_safe.hpp)
  - Root cause: Race condition async D2H

## Artefacts

- **Traces**: [`artifacts_remote/2026-02-06/b3_61/traces/`](../../artifacts_remote/2026-02-06/b3_61/traces/)
- **Análisis**: [`artifacts_remote/2026-02-06/b3_61/b3_61_analysis.txt`](../../artifacts_remote/2026-02-06/b3_61/b3_61_analysis.txt)
- **Logs**: [`artifacts_remote/2026-02-06/b3_61/run/`](../../artifacts_remote/2026-02-06/b3_61/run/)

## Conclusión

- **FIRST_FAIL**: Ninguno - todos los prompts completados exitosamente
- **ROOT_CAUSE**: El fix B3.63 resuelve el problema de illegal memory access en transferencias D2H
- **RESULT**: **OK** - Residual stream bisect funcionando correctamente

---

**Signed**: L.E.T / Leandro Emanuel Timberini
