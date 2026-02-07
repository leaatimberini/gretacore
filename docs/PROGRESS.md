# GRETA Core Progress Index

## Sync Status (2026-02-07)
- **Repo HEAD (main)**: `56b755a`
- **B3.66 implementation commit**: `56b755a`

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| **B3.66** | 2026-02-07 | `56b755a` | Prefill vs Decode Drift Probe | N/A | **IMPLEMENTED_PENDING_RUN** | N/A | [2026_02_07_B3_66](docs/AMD/2026_02_07_B3_66_prefill_decode_drift_probe.md) |
| **B3.65** | 2026-02-07 | `d28ea0e` | Decode Determinism Audit | N/A | **PASS_DETERMINISTIC** | N/A | [B3.65_Analysis](artifacts_remote/2026-02-07/B3_65_FINAL_REPORT.md) |
| **B3.64** | 2026-02-07 | `d28ea0e` | RoPE Kernel Launch Diagnostics | `BUFFER_TYPE_MISMATCH (d_pos FP16→FP32)` | **CLOSED** | [stability](artifacts_remote/2026-02-07/b3_64/stability/) | [b3_64_audit](docs/AMD/2026_02_06_B3_64_numerical_drift_audit.md) |
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

## Technical Details (B3.64 - D2H → RoPE Kernel - CLOSED)

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
**Status**: **CLOSED**

---

## 2026-02-07 - B3.66: Prefill vs Decode Drift Probe

| Field | Value |
|-------|-------|
| **Date** | 2026-02-07 |
| **Commit** | `56b755a` |
| **Status** | IMPLEMENTED_PENDING_RUN |
| **Objective** | Identify first tensor/stage divergence between prefill_last and decode0 |
| **AMD Report** | `docs/AMD/2026_02_07_B3_66_prefill_decode_drift_probe.md` |
| **Artifacts** | `artifacts_remote/YYYY-MM-DD/b3_66/{run,traces}/` |

### Scripts
- `tools/benchmarks/run_b3_66_mi300x.sh` - Remote runner with build + trace collection
- `tools/benchmarks/analyze_b3_66_prefill_decode_drift.py` - Strict pairing + first-fail verdict

### Next Steps
1. Run `./tools/benchmarks/run_b3_66_mi300x.sh 129.212.184.200 2026-02-07`
2. Copy artifacts back
3. Run analyzer
4. Update ROOT_CAUSE in AMD report

Signed: L.E.T / Leandro Emanuel Timberini

---

## Closed Tickets

### B3.64: D2H Illegal Memory Access → RoPE Kernel Fault (FIXED)
- **ETAPA**: ROOT_CAUSE FOUND AND FIXED
- **ERROR**: "RoPE Q launch failed: an illegal memory access was encountered"
- **ROOT CAUSE**: `d_pos` buffer type mismatch (FP16 allocated, uint32_t stored)
- **FIX**: `FP16` → `FP32` at `block_scheduler.cpp:1645`
- **VERIFICATION**: 20/20 stability sweep PASSED
- **PRÓXIMO**: B3.66 - Prefill vs Decode Drift Probe
