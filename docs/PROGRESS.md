# GRETA Core Progress Index

## Sync Status (2026-02-06)
- **Local HEAD**: `e09989c` ✅ (B3.61 residual stream bisect OK + B3.63 HIP D2H fix)
- **GitHub HEAD**: `ff39be3` ⚠️ (pending push)
- **Remote MI300X**: `e09989c` ✅ (sync-ed, stateless verified)
- **AMD Reports**: 44 documents in `docs/AMD/`
- **Artifacts**: Full traces in `artifacts_remote/2026-02-06/b3_61/`

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| B3.61 | 2026-02-06 | `e09989c` | Residual Stream Bisect | N/A | **OK** | Full traces: 3 prompts, layers 0,1,2,4,8 | [b3_61](artifacts_remote/2026-02-06/b3_61/) | [AMD_B3_61](docs/AMD/2026_02_06_B3_61_residual_stream_bisect.md) |
| B3.62 | 2026-02-06 | `303b634` | HIP D2H Transfer Audit | `BUG_NOT_REPRODUCED` | **INSTRUMENTATION_ADDED** | [B3.62](artifacts_remote/2026-02-06/b3_62/) | [AMD_B3_62](docs/AMD/2026_02_06_B3_62_hip_d2h_transfer_audit.md) |
| B3.63 | 2026-02-06 | `e09989c` | HIP D2H Root Cause Fix | `ASYNC_D2H_RACE` | **FIXED** | N/A | [d2h_safe.hpp](src/inference/include/gcore/inference/d2h_safe.hpp) |
| B3.59 | 2026-02-05 | `d558073` | Embedding/DebugInput audit | CLEAN | Confirmed OK | [B3.59](artifacts_remote/2026-02-05/b3_59/) | [AMD_B3_59](docs/AMD/2026_02_05_B3_59_embedding_debug_input_audit.md) |
| B3.58 | 2026-02-05 | `d558073` | RMSNorm wiring audit | `UPSTREAM_X_MISMATCH` | X0/Ceros en decode0 | [B3.58](artifacts_remote/2026-02-04/b3_58/) | N/A |
| B3.57.1 | 2026-02-04 | `d558073` | RMSNorm divergence | `NORMOUT_SELECTION` | Confirmed | N/A | N/A |

---

## Complete AMD Report Index (40 documents)

See [docs/AMD/INDEX.md](docs/AMD/INDEX.md) for full index with categories and links.

---

## Technical Details (B3.59)
- **Objective**: Identify why `embedding_out/x_in` was reported zeroed. Resolved ambiguity using standardized `StageTrace` metadata (`token_id`, `route`).
- **Flags used**: `GRETA_TRACE_STAGE=1`, `GRETA_TRACE_STAGE_DEBUG_INPUT=1`.
- **Result**: **NO ZEROING found**. Perfect hash match between prefill/decode.
