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
**B3.77**: COMPLETED - 32k Long-Context Attempt (PASS; bit-perfect at 32k context; peak VRAM 17.1GB/MI300X; sampling 1s)
**B3.78**: COMPLETED - 32k KV-Aligned Control (PASS_EQUIV; diff=0.0 at 32k)
**B3.79**: COMPLETED - Batch Size Probe (8k/16k) (PASS; batch=2 stable)
**B3.80**: COMPLETED - Micro-Soak Repetition (16k) (PASS; 5/5 determinism)
**B3.81**: COMPLETED - Multi-Batch Throughput Scaling (8k) (PASS; batch=8 stable; bit-perfect)
**B3.82**: COMPLETED - Steady-State Decode Scaling (8k) (PASS; batch=8 stable; 100% tokens)
**B3.83**: COMPLETED - Long-Context Decode (32k) (TIMEOUT_PREFILL; limit=40min)
**B3.84**: COMPLETED - High-Pressure Batch Decode (16k) (PASS; batch=8; 110GB VRAM)

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
**B3.77**: COMPLETED - 32k Long-Context Attempt (PASS; bit-perfect at 32k context; peak VRAM 17.1GB/MI300X; sampling 1s)
**B3.78**: COMPLETED - 32k KV-Aligned Control (PASS_EQUIV; diff=0.0 at 32k)
**B3.79**: COMPLETED - Batch Size Probe (8k/16k) (PASS; batch=2 stable)
**B3.80**: COMPLETED - Micro-Soak Repetition (16k) (PASS; 5/5 determinism)
**B3.81**: COMPLETED - Multi-Batch Throughput Scaling (8k) (PASS; batch=8 stable)
**B3.82**: COMPLETED - Steady-State Decode Scaling (8k) (PASS; 100% tokens)
**B3.83**: COMPLETED - Long-Context Decode (32k) (TIMEOUT)
**B3.84**: COMPLETED - High-Pressure Batch Decode (16k) (PASS; 110GB VRAM)
**B3.89**: COMPLETED - Prefill Kernel Optimization V3/V4 (PASS; MI300X perf rerun; v4 speedup up to 1.78x vs v3; closeout: 2026-02-14)

---

## Sync Status (2026-02-13)

**Repo branch**: `main` (tracked via origin/main)

**Implementation Commits (por ticket)**:
- B3.63 D2H audit: `de3b888`
- B3.66 base mode: `56b755a`
- B3.66 v2 mode: `77fd6bd`
- B3.67 guardrail closeout (MI300X): `68b32be` (runner fix: `83770d8`; runner dump-logits: `d47c8f3`)
- B3.68 greta_infer kv_aligned + logits dump: `4a57383`
- B3.69 logits-diff gate: `e7418b2` (zlib linkage fix: `ee65f79`; docs: `e1f36ba`)
- [x] B3.89 V3 (Q-LDS): Commit: 766e239 (branch: main)
    - [x] B3.89: Prefill Kernel Optimization V3/V4 - **COMPLETED**
        - [x] GGUF context_length patch (2048 → 32768)
        - [x] Executor GRETA_MAX_SEQ_LEN fix ($((CTX+2)))
        - [x] Core tests: 3/3 PASS
        - [x] V3: Zero scratch spill achieved (MI300X)
        - [x] V3: 4k prefill speedup 1.21x
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
| B3.76 | 2026-02-09 | Long-Context Memory Pressure | COMPLETED | PASS (up to 16k; peak VRAM 49GB) | N/A (pressure validation) | artifacts_remote/2026-02-09/b3_76/ | docs/AMD/2026_02_09_B3_76_long_context_memory_pressure.md |
| B3.77 | 2026-02-09 | 32k Long-Context Attempt | COMPLETED | PASS (bf16=fp16; diff=0.0) | N/A (context escalation) | artifacts_remote/2026-02-09/b3_77/ | docs/AMD/2026_02_09_B3_77_to_B3_80_decode_batch_suite.md |
| B3.78 | 2026-02-09 | 32k KV-Aligned Control | COMPLETED | PASS_EQUIV (diff=0.0) | N/A (alignment validation) | artifacts_remote/2026-02-09/b3_78_80/ | docs/AMD/2026_02_09_B3_77_to_B3_80_decode_batch_suite.md |
| B3.79 | 2026-02-10 | Batch Size Probe (8k/16k) | COMPLETED | PASS (batch=2 stable) | N/A (batch escalation) | artifacts_remote/2026-02-10/b3_78_80/ | docs/AMD/2026_02_09_B3_77_to_B3_80_decode_batch_suite.md |
| B3.80 | 2026-02-10 | Micro-Soak Repetition (16k) | COMPLETED | PASS (5/5 determinism) | N/A (determinism soak) | artifacts_remote/2026-02-10/b3_78_80/ | docs/AMD/2026_02_09_B3_77_to_B3_80_decode_batch_suite.md |
| B3.81 | 2026-02-10 | Multi-Batch Throughput | COMPLETED | PASS (batch=8 stable) | N/A (throughput audit) | artifacts_remote/2026-02-10/b3_81/ | docs/AMD/2026_02_09_B3_81_multibatch_throughput.md |
| B3.82 | 2026-02-10 | Steady-State Decode Scaling | COMPLETED | PASS (100% tokens; 2.0 TPS) | N/A (steady-state audit) | artifacts_remote/2026-02-10/b3_82_84/ | docs/AMD/2026_02_09_B3_82_to_B3_84_decode_steady_state.md |
| B3.83 | 2026-02-10 | Long-Context Decode (32k) | COMPLETED | TIMEOUT_PREFILL | PREFILL_O(N^2)_BOTTLE_NECK | artifacts_remote/2026-02-10/b3_82_84/ | docs/AMD/2026_02_09_B3_82_to_B3_84_decode_steady_state.md |
| B3.84 | 2026-02-10 | High-Pressure Batch Decode | COMPLETED | PASS (110GB VRAM; 100% tokens) | N/A (high-pressure stability) | artifacts_remote/2026-02-10/b3_82_84/ | docs/AMD/2026_02_09_B3_82_to_B3_84_decode_steady_state.md |
| B3.85 | 2026-02-10 | Prefill Complexity RCA | COMPLETED | PASS_RCA_O_N2 | O(N^2) confirmed | artifacts_remote/2026-02-10/b3_85/ | docs/AMD/2026_02_10_B3_85_prefill_complexity_rca.md |
| B3.86 | 2026-02-10 | Attn Impl Probe | COMPLETED | PASS_PROBE | flash_v2_naive detected | artifacts_remote/2026-02-10/b3_86/ | docs/AMD/2026_02_10_B3_86_attention_impl_probe.md |
| B3.87 | 2026-02-10 | Decode TPS Decomposition | COMPLETED | PASS_RCA | -11.1% TPS delta | artifacts_remote/2026-02-10/b3_87/ | docs/AMD/2026_02_10_B3_87_decode_tps_decomposition.md |
| B3.88 | 2026-02-10 | 32k Feasibility | COMPLETED | PASS_32K_FEASIBLE | 32k prefill achieved | artifacts_remote/2026-02-10/b3_88/ | docs/AMD/2026_02_10_B3_88_32k_feasibility.md |
| B3.89 | 2026-02-14 | Prefill Kernel Optimization V3/V4 | COMPLETED | PASS | v4 improvement ~1.78x vs v3 in perf mode | artifacts_remote/B3_89_FINAL_REPORT.md | docs/AMD/2026_02_14_B3_89_prefill_microbench_closeout.md |

---

## Complete AMD Report Index (50 documents)

[See docs/AMD/INDEX.md for full list]

---

Signed: L.E.T / Leandro Emanuel Timberini
