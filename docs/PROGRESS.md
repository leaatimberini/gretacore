# GRETA Core Progress Index

## Sync Status (2026-02-07)
- **Local HEAD**: `e3143aa` ✅ B3.64 RoPE diagnostics committed
- **GitHub HEAD**: `ff39be3` ⚠️ (pending push)
- **Remote MI300X**: `28a5e49` ✅ (sync-ed, stateless verified)
- **AMD Reports**: 47 documents in `docs/AMD/`
- **Artifacts**: B3.64 forensics in `artifacts_remote/2026-02-07/b3_64/`

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| B3.64 | 2026-02-07 | `e3143aa` | RoPE Kernel Launch Diagnostics | `ROPE_KERNEL_UPSTREAM` | **DIAGNOSTICS_ADDED** 🔧 | [b3_64](artifacts_remote/2026-02-07/b3_64/) | [b3_64_analysis](artifacts_remote/2026-02-07/b3_64/b3_64_analysis.txt) |
| B3.63 | 2026-02-06 | `e09989c` | HIP D2H Root Cause Fix | `ASYNC_D2H_RACE` | **INCOMPLETE** ⚠️ | N/A | [d2h_safe.hpp](src/inference/include/gcore/inference/d2h_safe.hpp) |
| B3.62 | 2026-02-06 | `303b634` | HIP D2H Transfer Audit | `BUG_NOT_REPRODUCED` | **INSTRUMENTATION_ADDED** | [B3.62](artifacts_remote/2026-02-06/b3_62/) | [AMD_B3_62](docs/AMD/2026_02_06_B3_62_hip_d2h_transfer_audit.md) |
| B3.61 | 2026-02-06 | `e09989c` | Residual Stream Bisect | N/A | **OK** | Full traces: 3 prompts, layers 0,1,2,4,8 | [b3_61](artifacts_remote/2026-02-06/b3_61/) | [AMD_B3_61](docs/AMD/2026_02_06_B3_61_residual_stream_bisect.md) |
| B3.59 | 2026-02-05 | `d558073` | Embedding/DebugInput audit | CLEAN | Confirmed OK | [B3.59](artifacts_remote/2026-02-05/b3_59/) | [AMD_B3_59](docs/AMD/2026_02_05_B3_59_embedding_debug_input_audit.md) |
| B3.58 | 2026-02-05 | `d558073` | RMSNorm wiring audit | `UPSTREAM_X_MISMATCH` | X0/Ceros en decode0 | [B3.58](artifacts_remote/2026-02-04/b3_58/) | N/A |
| B3.57.1 | 2026-02-04 | `d558073` | RMSNorm divergence | `NORMOUT_SELECTION` | Confirmed | N/A | N/A |

---

## Complete AMD Report Index (40 documents)

See [docs/AMD/INDEX.md](docs/AMD/INDEX.md) for full index with categories and links.

---

## Technical Details (B3.64 - D2H → RoPE Kernel Fault Evolution)

| Etapa | Fecha | Error |ROOT_CAUSE | Estado | Siguiente |
|-------|-------|-------|-----------|--------|-----------|
| Forensics | 2026-02-07 | `hipMemcpy D2H failed` | `BUFFER_INVALID` | 🔍 FORENSICS_COMPLETED | B3.64 - Root Cause |
| Evolved | 2026-02-07 | `RoPE Q launch failed` | `ROPE_KERNEL_FAULT (upstream)` | 🔄 EVOLVED | B3.65 - Diagnosticar kernel RoPE |

**Evolución del diagnóstico:**
- Error original: "hipMemcpy D2H failed: an illegal memory access was encountered"
- Error actual:   "RoPE Q launch failed: an illegal memory access was encountered"
- El wrapper D2H [D2H_SAFE_WRAPPER] NO se ejecutó (no hay logs de D2H)

**Análisis:**
- El crash ocurre ANTES de llegar al código D2H
- El kernel RoPE falla durante el lanzamiento (hipModuleLaunchKernel o similar)
- Esto indica un problema en:
  1) Puntero/payload incorrecto pasado al kernel RoPE
  2) Buffer no inicializado/corrupto usado por RoPE
  3) Sincronización de stream incorrecta antes del kernel

**Ubicación esperada del bug:**
- src/kernels/rope.cpp o similar
- hipModuleLaunchKernel para RoPE Q
- Buffers de query/position pasados al kernel

**Artifacts Reference**: `artifacts_remote/2026-02-07/b3_64/`
**Status**: ROOT_CAUSE IDENTIFIED (upstream) - Esperando B3.65

---

## Closed Tickets

### B3.64: D2H Illegal Memory Access → RoPE Kernel Fault
- **ETAPA**: ROOT_CAUSE IDENTIFIED (upstream)
- **ERROR**: RoPE Q launch failed: an illegal memory access was encountered
- **D2H WRAPPER**: Instrumentado y enganchado, pero NO ejecutado
- **PRÓXIMO**: B3.65 - Diagnosticar kernel RoPE launch
