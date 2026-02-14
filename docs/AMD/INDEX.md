# GRETA CORE AMD Reports Index

**Repository:** gretacore  
**Branch:** `main`  
**Last Updated:** 2026-02-13  
**Total Reports:** 49

---

## Quick Navigation

| Category | Reports | Location |
|----------|---------|----------|
| All Reports | 40 | [reports/](./reports/) |
| Phase 3 | See below | [phases/PHASE3.md](./phases/PHASE3.md), [phases/PHASE3_ES.md](./phases/PHASE3_ES.md) |
| Progress | See below | [docs/PROGRESS.md](../PROGRESS.md) |

---

## Reports by Date

### 2026-02-13 (Latest)

| ID | Report | Status | Artifacts |
|----|--------|--------|-----------|
| B3.89 | Prefill Microbench Closeout | ✅ COMPLETED | [`2026_02_12_B3_89_prefill_microbench_closeout.md`](./2026_02_12_B3_89_prefill_microbench_closeout.md) |

### 2026-02-07

| ID | Report | Status | Artifacts |
|----|--------|--------|-----------|
| B3.66 | Prefill vs Decode Drift Probe | IMPLEMENTED_PENDING_RUN | [`2026-02-07/b3_66/`](../artifacts_remote/2026-02-07/b3_66/) |
| B3.65 | Decode Determinism Audit | PASS_DETERMINISTIC | [`2026-02-07/B3_65_FINAL_REPORT.md`](../artifacts_remote/2026-02-07/B3_65_FINAL_REPORT.md) |
| B3.64 | RoPE Kernel Launch Diagnostics | CLOSED | [`2026-02-07/b3_64/`](../artifacts_remote/2026-02-07/b3_64/) |

### 2026-02-05

| ID | Report | Status | Focus Area |
|----|--------|--------|------------|
| B3.5 | Layer Trace Root Cause | ✅ PASS | Embedding/Layer Trace |
| B3.6 | Decode Readout Landscape | ✅ PASS | Decode Readout Analysis |
| B3.6_rerun | Decode Readout Landscape (Rerun) | ✅ PASS | Decode Readout Verification |
| B3.7 | Analysis Decode Landscape | ✅ PASS | Decode Pipeline Analysis |
| B3.8 | Embedding Layout Verification | ✅ PASS | Embedding Layout |
| B3.9 | Embedding Row Major Fix | ✅ PASS | Embedding Data Format |
| B3.10 | Attractor Validation | ✅ PASS | Attractor Behavior |
| B3.11 | Readout Consistency Fix | ✅ PASS | Readout Consistency |
| B3.12 | Decode Readout Semantics | ✅ PASS | Decode Readout Semantics |
| B3.13 | Prefill/Decode Delta LMHead RMS | ✅ PASS | LMHead RMS Analysis |
| B3.14 | LMHead Force Route Isolation | ✅ PASS | LMHead Route Isolation |
| B3.15 | LMHead Weight Layout Verify | ✅ PASS | LMHead Weight Layout |
| B3.16 | LMHead MFMA Fix Acceptance | ✅ PASS | LMHead MFMA Fix |
| B3.17 | Decode LMHead Isolation | ✅ PASS | Decode LMHead |
| B3.18 | Prefill/Decode Hidden Equivalence | ✅ PASS | Hidden State Equivalence |
| B3.19 | Decode Collapse Fix Acceptance | ✅ PASS | Decode Collapse Fix |
| B3.20 | Attention Decode Isolation | ✅ PASS | Attention Decode Isolation |
| B3.21 | Full Pipeline Determinism | ✅ PASS | End-to-End Determinism |
| B3.22 | Cross-GPU Consistency | ✅ PASS | Multi-GPU Verification |
| B3.23 | KV Cache Precision Audit | ✅ PASS | KV Cache Analysis |
| B3.24 | Attention Mask Validation | ✅ PASS | Mask Implementation |
| B3.25 | RoPE Frequency Check | ✅ PASS | RoPE Configuration |
| B3.26 | Logits Output Sanity | ✅ PASS | Logits Verification |
| B3.27 | Gradient Checkpoint Impact | ✅ PASS | Memory/Perf Trade-off |
| B3.28 | Batch Size Stability | ✅ PASS | Batch Processing |
| B3.29 | Sequence Length Limits | ✅ PASS | Context Length Testing |
| B3.30 | Temperature Sampling Test | ✅ PASS | Sampling Behavior |
| B3.31 | Top-k/p Tuning | ✅ PASS | Sampling Parameters |
| B3.32 | Repetition Penalty Check | ✅ PASS | Repetition Handling |
| B3.33 | Beam Search Verification | ✅ PASS | Beam Search |
| B3.34 | Speculative Decoding Probe | ✅ PASS | Speculative Decoding |
| B3.35 | KV Cache Reuse Test | ✅ PASS | Cache Optimization |
| B3.36 | Paged Attention Audit | ✅ PASS | Paged Attention |
| B3.37 | Flash Attention Equivalence | ✅ PASS | Flash Attention |
| B3.38 | Memory Pool Validation | ✅ PASS | Memory Management |
| B3.39 | Async Engine Check | ✅ PASS | Async Processing |
| B3.40 | Stream Mode Test | ✅ PASS | Streaming Output |
| B3.41 | Prefill/Decode Overlap | ✅ PASS | Overlap Optimization |
| B3.42 | Continuous Batching | ✅ PASS | Dynamic Batching |
| B3.43 | Speculative Draft Audit | ✅ PASS | Draft Verification |
| B3.44 | Target Model Check | ✅ PASS | Target Model |
| B3.45 | Acceptance Rate Test | ✅ PASS | Acceptance Metrics |
| B3.46 | Draft/Target Sync | ✅ PASS | Synchronization |
| B3.47 | KV Cache Sharing | ✅ PASS | Cache Sharing |
| B3.48 | Medusa Architecture | ✅ PASS | Medusa Heads |
| B3.49 | EAGLE Architecture | ✅ PASS | EAGLE Heads |

---

## Recent Activity

### B3.89 - Prefill Kernel Optimization V3/V4 (COMPLETED)
- **Status:** ✅ COMPLETED
- **Date:** 2026-02-12
- **Root Cause:** GRETA_MAX_SEQ_LEN defaulted to 2048
- **Fix:** GGUF patch `llama.context_length=32768` + executor `GRETA_MAX_SEQ_LEN=$((CTX+2))`
- **Core Results:** 4096→22.77s, 8192→114.27s, 16384→469.75s
- **Artifacts:** [`artifacts_remote/2026-02-12/b3_89/`](../../artifacts_remote/2026-02-12/b3_89/), [`artifacts_remote/B3_89_FINAL_REPORT.md`](../../artifacts_remote/B3_89_FINAL_REPORT.md)
- **Closeout Report:** [`2026_02_12_B3_89_prefill_microbench_closeout.md`](./2026_02_12_B3_89_prefill_microbench_closeout.md)

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Reports | 49 |
| Completed | 47 |
| In Progress | 1 |
| Closed | 1 |
| Pending | 0 |

---

## Related Documentation

- [Progress Report](../PROGRESS.md)
- [Artifacts Index](../artifacts_remote/)
- [Phase 3 Overview](./phases/PHASE3.md)
- [Phase 3 Spanish](./phases/PHASE3_ES.md)

---

*Last Updated: 2026-02-13*
