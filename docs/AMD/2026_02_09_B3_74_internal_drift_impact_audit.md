# B3.74 Internal Drift Impact Audit

**Date:** 2026-02-09
**Author:** Leandro Emanuel Timberini
**Status:** IN PROGRESS
**Base:** B3.66 (Internal Tracing) + B3.73 (Logits Comparison)

## Objective
Audit the impact of internal hidden state drift (observed in B3.66) on final logits equivalence between prefill and decode phases on MI300X.
B3.73 confirmed `INTERNAL_DRIFT_NO_LOGIT_IMPACT` by comparing logits only. This audit re-enables B3.66 tracing to simultaneously capture internal states and verify if the drift persists while logits remain identical.

## Methodology
- **Runner:** `tools/benchmarks/run_b3_74_internal_drift_audit.sh`
- **Tracing:** `GRETA_TRACE_B3_66=1` (captures attention scores/fingerprints)
- **Analyzer:** `tools/benchmarks/analyze_b3_67_equivalence_guardrail.py --mode b3_74`
- **Metrics:**
  - Internal Drift (Proxy: `abs_sum` diff between prefill/decode tensors)
  - Logits Drift (`max_abs_diff`, `p99_abs_diff`)

## Expected Outcome
- **kv_aligned=1:** Both internal states and logits match (PASS).
- **kv_aligned=0:** Internal states drift (B3.66 behavior), but logits match (B3.73 behavior). This proves the drift is benign w.r.t final output.

## Results
(Pending Execution)
