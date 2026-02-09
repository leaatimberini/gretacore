# GRETA CORE - Progress Report

## Resumen Ejecutivo

**B3.63**: CLOSED - D2H Async Race Audit (NOT_REPRODUCED)
**B3.64**: CLOSED - D2H/RoPE crash fix (d_pos buffer type mismatch)
**B3.65**: CLOSED - Decode determinism audit (PASS_DETERMINISTIC 10/10)
**B3.66**: COMPLETED - Prefill vs decode drift probe (expected attention mismatch)
**B3.66 v2**: COMPLETED - kv_aligned mode probe
**B3.67**: COMPLETED - Equivalence guardrail (prefill vs decode) (PASS_GUARDRAIL; MI300X full matrix; metadata-only equiv)
**B3.69**: COMPLETED - Logits-diff equivalence gate (PASS_GUARDRAIL; max_abs_diff=0.0, top1_agreement=1.0)
**B3.70**: COMPLETED - Drift characterization (PASS; diff=0.0, top1=1.0)
**B3.71**: COMPLETED - Span escalation (PASS; 32→40s, 128→56s, 512→117s)
**B3.72**: COMPLETED - Cross-dtype sweep bf16/fp16 (PASS; max_diff=0.0)
**B3.73**: COMPLETED - Reconcile B3.66 vs B3.69 (RECONCILED_NO_LOGIT_DRIFT; MI300X full matrix; kv=0 diff=0.0)
**B3.74**: COMPLETED - Internal Drift Impact Audit (PASS_INTERNAL_AUDIT; kv0 drift confirmed benign; logits diff=0.0)
**B3.75**: COMPLETED - MI300X CI Suite (PASS_BENCHMARK; Nightly 64/64, Stress 4/4, Coverage 64/64; 100% Equivalence)
**B3.76**: COMPLETED - Long-Context Memory Pressure (PASS; up to 16k context; peak VRAM 49GB/MI300X)

---

## Executive Summary

**B3.63**: CLOSED - D2H Async Race Audit (NOT_REPRODUCED)
**B3.64**: CLOSED - D2H/RoPE crash fix (d_pos buffer type mismatch)
**B3.65**: CLOSED - Decode determinism audit (PASS_DETERMINISTIC 10/10)
**B3.66**: COMPLETED - Prefill vs decode drift probe (expected attention mismatch)
**B3.66 v2**: COMPLETED - kv_aligned mode probe
**B3.67**: COMPLETED - Equivalence guardrail (prefill vs decode) (PASS_GUARDRAIL; MI300X full matrix; metadata-only equiv)
**B3.69**: COMPLETED - Logits-diff equivalence gate (PASS_GUARDRAIL; max_abs_diff=0.0, top1_agreement=1.0)
**B3.70**: COMPLETED - Drift characterization (PASS; diff=0.0, top1=1.0)
**B3.71**: COMPLETED - Span escalation (PASS; 32→40s, 128→56s, 512→117s)
**B3.72**: COMPLETED - Cross-dtype sweep bf16/fp16 (PASS; max_diff=0.0)
**B3.73**: COMPLETED - Reconcile B3.66 vs B3.69 (RECONCILED_NO_LOGIT_DRIFT; MI300X full matrix; kv=0 diff=0.0)
**B3.74**: COMPLETED - Internal Drift Impact Audit (PASS_INTERNAL_AUDIT; kv0 drift confirmed benign; logits diff=0.0)
**B3.75**: COMPLETED - MI300X CI Suite (PASS_BENCHMARK; Nightly 64/64, Stress 4/4, Coverage 64/64; 100% Equivalence)
**B3.76**: COMPLETED - Long-Context Memory Pressure (PASS; up to 16k context; peak VRAM 49GB/MI300X)

---

## Sync Status (2026-02-09)

**Repo branch**: `main` (tracked via origin/main)

**Implementation Commits (por ticket)**:
- B3.63 D2H audit: `de3b888`
- B3.66 base mode: `56b755a`
- B3.66 v2 mode: `77fd6bd`
- B3.67 guardrail closeout (MI300X): `68b32be` (runner fix: `83770d8`; runner dump-logits: `d47c8f3`)
- B3.68 greta_infer kv_aligned + logits dump: `4a57383`
- B3.69 logits-diff gate: `e7418b2` (zlib linkage fix: `ee65f79`; docs: `e1f36ba`)
- B3.74 internal audit: `f31ab1c`
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
| B3.67 | 2026-02-08 | Equivalence Guardrail | COMPLETED | PASS_GUARDRAIL (MI300X full matrix; PASS_EQUIV_METADATA) | N/A (automation/guardrail) | artifacts_remote/2026-02-08/b3_67/ | docs/AMD/2026_02_07_B3_67_equivalence_guardrail.md |
| B3.69 | 2026-02-09 | Logits-Diff Equivalence Gate | COMPLETED | PASS_GUARDRAIL (diff=0.0, top1=1.0) | N/A (real numeric comparison) | artifacts_remote/2026-02-09/b3_69/ | docs/AMD/2026_02_08_B3_69_logits_diff_equivalence_gate.md |
| B3.70 | 2026-02-09 | Drift Characterization (kv=0) | COMPLETED | PASS (diff=0.0, top1=1.0) | N/A (no gate, metrics only) | artifacts_remote/2026-02-09/b3_70_71_72/ | docs/AMD/2026_02_09_B3_70_71_72_sweep.md |
| B3.71 | 2026-02-09 | Span Escalation + Perf | COMPLETED | PASS (32→40s, 128→56s, 512→117s) | N/A (profiling) | artifacts_remote/2026-02-09/b3_70_71_72/ | docs/AMD/2026_02_09_B3_70_71_72_sweep.md |
| B3.72 | 2026-02-09 | Cross-Dtype Sweep | COMPLETED | PASS (bf16=fp16; max_diff=0.0) | N/A (dtype gate kv=1) | artifacts_remote/2026-02-09/b3_70_71_72/ | docs/AMD/2026_02_09_B3_70_71_72_sweep.md |
| B3.73 | 2026-02-09 | Reconcile B3.66 vs B3.69 | COMPLETED | RECONCILED_NO_LOGIT_DRIFT (MI300X full matrix; kv=0 diff=0.0) | INTERNAL_DRIFT_NO_LOGIT_IMPACT | artifacts_remote/2026-02-09/b3_73/ | docs/AMD/2026_02_09_B3_73_reconcile_b3_66_vs_b3_69.md |
| B3.74 | 2026-02-09 | Internal Drift Impact Audit | COMPLETED | PASS_INTERNAL_AUDIT (kv0 drift; logits diff=0.0) | INTERNAL_DRIFT (BENIGN) | artifacts_remote/2026-02-09/b3_74/ | docs/AMD/2026_02_09_B3_74_internal_drift_impact_audit.md |
| B3.75 | 2026-02-09 | MI300X CI Suite | COMPLETED | PASS_BENCHMARK | N/A (CI Harness) | artifacts_remote/2026-02-09/b3_75_ci/ | docs/AMD/2026_02_09_B3_75_to_B3_80_mi300x_ci_suite.md |
| B3.76 | 2026-02-09 | Long-Context Memory Pressure | COMPLETED | PASS (up to 16k; peak VRAM 49GB) | N/A (pressure validation) | artifacts_remote/2026-02-09/b3_76/ | docs/AMD/2026_02_09_B3_76_long_context_memory_pressure.md | status:DONE

---

## Complete AMD Report Index (50 documents)

[See docs/AMD/INDEX.md for full list]

---

Signed: L.E.T / Leandro Emanuel Timberini
