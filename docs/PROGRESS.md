# GRETA CORE - Progress Report

## Resumen Ejecutivo

**B3.63**: CLOSED - D2H Async Race Audit (NOT_REPRODUCED)
**B3.64**: CLOSED - D2H/RoPE crash fix (d_pos buffer type mismatch)
**B3.65**: CLOSED - Decode determinism audit (PASS_DETERMINISTIC 10/10)
**B3.66**: COMPLETED - Prefill vs decode drift probe (expected attention mismatch)
**B3.66 v2**: COMPLETED - kv_aligned mode probe

---

## Executive Summary

**B3.63**: CLOSED - D2H Async Race Audit (NOT_REPRODUCED)
**B3.64**: CLOSED - D2H/RoPE crash fix (d_pos buffer type mismatch)
**B3.65**: CLOSED - Decode determinism audit (PASS_DETERMINISTIC 10/10)
**B3.66**: COMPLETED - Prefill vs decode drift probe (expected attention mismatch)
**B3.66 v2**: COMPLETED - kv_aligned mode probe

---

## Sync Status (2026-02-07)

**Repo branch**: `main` (tracked via origin/main)

**Implementation Commits (por ticket)**:
- B3.63 D2H audit: `de3b888`
- B3.66 base mode: `56b755a`
- B3.66 v2 mode: `77fd6bd`
- Docs index: `1f662f1`

**Nota / Note**:
- ES: Repo HEAD es el snapshot actual del repositorio; los commits de implementación por ticket se mantienen separados para trazabilidad.
- EN: Repo HEAD is the current repository snapshot; per-ticket implementation commits are kept separate for traceability.

---

## Phase Index

| Ticket | Date | Objective | Status | Result | Root Cause | Artifacts | AMD Report |
|--------|------|-----------|--------|--------|------------|-----------|------------|
| B3.63 | 2026-02-07 | D2H Async Race Audit | CLOSED | NOT_REPRODUCED | N/A (instrumentation-only audit) | artifacts_remote/2026-02-07/B3_63_FINAL_REPORT.md | docs/AMD/2026_02_07_B3_63_d2h_async_race_audit.md |
| B3.64 | 2026-02-07 | D2H/RoPE crash hardening | CLOSED | PASS 20/20 | BUFFER_TYPE_MISMATCH (d_pos FP16→FP32) | artifacts_remote/2026-02-07/b3_64/stability/ | docs/AMD/2026_02_06_B3_64_numerical_drift_audit.md |
| B3.65 | 2026-02-07 | Decode Determinism Audit | CLOSED | PASS_DETERMINISTIC | N/A (no-code-change audit) | artifacts_remote/2026-02-07/B3_65_FINAL_REPORT.md | docs/AMD/2026_02_07_B3_65_decode_determinism_audit.md |
| B3.66 | 2026-02-07 | Prefill vs Decode Drift Probe | COMPLETED | FAIL (expected) | ATTENTION_COMPUTATION_MISMATCH | artifacts_remote/2026-02-07/b3_66/ | docs/AMD/2026_02_07_B3_66_prefill_decode_drift_probe.md |
| B3.66 v2 | 2026-02-07 | kv_aligned Mode | COMPLETED | EXPECTED (kv_aligned evidence added) | STRUCTURAL_DRIFT (EXPECTED; prefill vs decode semantics) | artifacts_remote/2026-02-07/b3_66_v2/ | docs/AMD/2026_02_07_B3_66_v2_kv_aligned_probe.md |

---

## Complete AMD Report Index (50 documents)

[See docs/AMD/INDEX.md for full list]

---

Signed: L.E.T / Leandro Emanuel Timberini
