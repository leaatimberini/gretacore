# B3.67 Equivalence Guardrail Report

**Date:** 2026-02-07
**Layer Analyzed:** 32
**Mode:** Hidden State Comparison (Prefill vs Decode)

## Completeness Guardrail

- **Present pairs:** 0
- **Missing pairs:** 0
- **Status:** COMPLETE

## Comparison Results

| kv_aligned | seed | p99_abs_diff | max_abs_diff | top1_agreement | verdict |
|------------|------|--------------|--------------|----------------|---------|
| 0 | 0 | N/A | N/A | N/A | INCONCLUSIVE |
| 0 | 1 | N/A | N/A | N/A | INCONCLUSIVE |
| 0 | 2 | N/A | N/A | N/A | INCONCLUSIVE |
| 1 | 0 | N/A | N/A | N/A | INCONCLUSIVE |
| 1 | 1 | N/A | N/A | N/A | INCONCLUSIVE |
| 1 | 2 | N/A | N/A | N/A | INCONCLUSIVE |

## Summary

- **PASS_EQUIV:** 0
- **FAIL_EQUIV:** 0
- **EXPECTED_DRIFT:** 0
- **INCONCLUSIVE:** 6

**Global Verdict:** INCONCLUSIVE - No conclusive results

## Thresholds Reference

| Metric | Threshold | Condition |
|--------|-----------|-----------|
| p99_abs_diff | <= 1e-3 | PASS_EQUIV if <= threshold |
| max_abs_diff | <= 5e-3 | PASS_EQUIV if <= threshold |
| top1_agreement | >= 0.999 | PASS_EQUIV if >= threshold |
| kv_aligned | 0 or 1 | 1 = expect equivalence, 0 = expect drift |
