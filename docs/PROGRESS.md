# GRETA Core Progress Index

## Sync Status (2026-02-07)
- **Local HEAD**: `d28ea0e` ✅ B3.64 closed (hardening verified)
- **GitHub HEAD**: `d28ea0e` ✅ (pushed)
- **Remote MI300X**: `d28ea0e` ✅ (sync-ed, stateless verified)
- **AMD Reports**: 48 documents in `docs/AMD/`
- **Artifacts**: B3.64 stability sweep in `artifacts_remote/2026-02-07/b3_64/stability/`
- **B3.65**: BLOCKED (model-binary mismatch)

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| **B3.65** | 2026-02-07 | `d28ea0e` | Decode Determinism Audit | `MODEL_BINARY_MISMATCH` | **BLOCKED** | N/A | [B3.65_Analysis](artifacts_remote/2026-02-07/b3_65_analysis.txt) |
| **B3.64** | 2026-02-07 | `d28ea0e` | RoPE Kernel Launch Diagnostics | `UNSAFE_ASYNC_D2H_AND_KERNEL_LAUNCH_ORDERING` | **CLOSED** ✅ | [stability](artifacts_remote/2026-02-07/b3_64/stability/) | [b3_64_audit](docs/AMD/2026_02_06_B3_64_numerical_drift_audit.md) |
| B3.63 | 2026-02-06 | `e09989c` | HIP D2H Root Cause Fix | `ASYNC_D2H_RACE` | **INCOMPLETE** ⚠️ | N/A | [d2h_safe.hpp](src/inference/include/gcore/inference/d2h_safe.hpp) |
| B3.62 | 2026-02-06 | `303b634` | HIP D2H Transfer Audit | `BUG_NOT_REPRODUCED` | **INSTRUMENTATION_ADDED** | [B3.62](artifacts_remote/2026-02-06/b3_62/) | [AMD_B3_62](docs/AMD/2026_02_06_B3_62_hip_d2h_transfer_audit.md) |
| B3.61 | 2026-02-06 | `e09989c` | Residual Stream Bisect | N/A | **OK** | Full traces: 3 prompts, layers 0,1,2,4,8 | [b3_61](artifacts_remote/2026-02-06/b3_61/) | [AMD_B3_61](docs/AMD/2026_02_06_B3_61_residual_stream_bisect.md) |
| B3.59 | 2026-02-05 | `d558073` | Embedding/DebugInput audit | CLEAN | Confirmed OK | [B3.59](artifacts_remote/2026-02-05/b3_59/) | [AMD_B3_59](docs/AMD/2026_02_05_B3_59_embedding_debug_input_audit.md) |
| B3.58 | 2026-02-05 | `d558073` | RMSNorm wiring audit | `UPSTREAM_X_MISMATCH` | X0/Ceros en decode0 | [B3.58](artifacts_remote/2026-02-04/b3_58/) | N/A |
| B3.57.1 | 2026-02-04 | `d558073` | RMSNorm divergence | `NORMOUT_SELECTION` | Confirmed | N/A | N/A |

---

## Complete AMD Report Index (49 documents)

See [docs/AMD/INDEX.md](docs/AMD/INDEX.md) for full index with categories and links.

---

## Technical Details (B3.64 - D2H → RoPE Kernel - RESOLVED)

| Etapa | Fecha | Error | ROOT_CAUSE | Estado | Siguiente |
|-------|-------|-------|-----------|--------|-----------|
| Forensics | 2026-02-07 | `hipMemcpy D2H failed` | `BUFFER_INVALID` | 🔍 FORENSICS_COMPLETED | B3.64 - Root Cause |
| Evolved | 2026-02-07 | `RoPE Q launch failed` | `ROPE_KERNEL_FAULT (upstream)` | 🔄 EVOLVED | B3.65 - Diagnosticar kernel RoPE |
| **RESOLVED** | 2026-02-07 | `No error` | `BUFFER_TYPE_MISMATCH (d_pos FP16→FP32)` | ✅ **FIXED** | B3.65 - Siguiente feature |

**Evolución del diagnóstico:**
- Error original: "hipMemcpy D2H failed: an illegal memory access was encountered"
- Error evolucionó: "RoPE Q launch failed: an illegal memory access was encountered"
- **ROOT CAUSE IDENTIFIED**: Buffer `d_pos` allocated with `FP16` (2 bytes) but stores `uint32_t` (4 bytes)
- **FIX APPLIED**: Changed `FP16` → `FP32` at `block_scheduler.cpp:1645`
- Estado actual: **RESUELTO** - Fix verificado con sweep 20/20

**Resultado del benchmark B3.64.3:**
- STATUS: **OK**
- Tiempo total: 796.649 ms
- Tokens/second: 6.28
- Todos los parámetros RoPE son válidos
- D2H transfers funcionando correctamente

**Artifacts Reference**: `artifacts_remote/2026-02-07/b3_64/`
**Status**: **RESOLVED** - Bug transitorio corregido

---

## Closed Tickets

### B3.64: D2H Illegal Memory Access → RoPE Kernel Fault (FIXED)
- **ETAPA**: ROOT_CAUSE FOUND AND FIXED
- **ERROR**: "RoPE Q launch failed: an illegal memory access was encountered"
- **ROOT CAUSE**: `d_pos` buffer type mismatch (FP16 allocated, uint32_t stored)
- **FIX**: `FP16` → `FP32` at `block_scheduler.cpp:1645`
- **VERIFICATION**: 20/20 stability sweep PASSED
- **PRÓXIMO**: B3.65 - Numerical Drift Post-Hardening
