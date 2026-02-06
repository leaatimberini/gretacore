# GRETA Core Progress Index

## Sync Status (2026-02-06)
- **Local HEAD**: `dc4b829` ✅
- **GitHub HEAD**: `dc4b829` ✅
- **Remote MI300X**: `dc4b829` ✅ (sync-ed, stateless verified)
- **AMD Reports**: 43 documents in `docs/AMD/`
- **Artifacts Remote**: Rescued to `artifacts_remote/2026-02-06/b3_61/`

## Phase Index

| Phase | Date | HEAD Hash | Objective | Root Cause | Result | Artifacts | AMD Report |
|-------|------|-----------|-----------|------------|--------|-----------|------------|
| B3.61 | 2026-02-06 | `dc4b829` | Residual Stream Bisect | `RMS_NORM_LAUNCH_FAILURE` | FAILED (binary memory error) | [B3.61](artifacts_remote/2026-02-06/b3_61/) | [AMD_B3_61](docs/AMD/2026_02_06_B3_61_residual_stream_bisect.md) |
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
