# B3.70-71-72 Sweep Report

**Date:** 2026-02-09
**Mode:** Span Escalation + Dtype Sweep + Drift Characterization

## Global Verdict

**PASS**

- PASS_EQUIV: 18
- FAIL_EQUIV: 0
- EXPECTED_DRIFT: 18
- INCOMPLETE: 0
- SKIPPED: 0

## Counting Semantics: Runs vs Pairs

- A **run** = one execution (prefill OR decode) for a single config
- A **pair** = (prefill, decode) for the same (span, dtype, kv_aligned, seed)
- **Total runs:** 72
- **Total pairs:** 36 (runs / 2)
- Verdict counts (PASS_EQUIV, EXPECTED_DRIFT, etc.) are **per pair**, not per run

## Observation: kv_aligned=0 Produced Identical Logits (diff=0.0)

In this sweep, **kv_aligned=0** also produced identical logits (max_abs_diff=0.0, top1=1.0)
for all spans (32/128/512) and dtypes (bf16/fp16).

**Interpretation:**

- The effective prefill/decode routes are numerically equivalent for this model/config
- The kv_aligned flag does not alter observable logits in this scenario (maintained by contract)

**Note (contrast with B3.66):**

B3.66 reported drift (EXPECTED) under a different metric/route. This sweep does not invalidate
that finding, but suggests drift does not manifest in logits under this specific configuration.

## Comparison Results

| span | dtype | kv_aligned | seed | max_abs_diff | p99_abs_diff | top1_agreement | verdict |
|------|-------|------------|------|--------------|--------------|----------------|---------|
| 128 | bf16 | 0 | 0 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 128 | bf16 | 0 | 1 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 128 | bf16 | 0 | 2 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 128 | bf16 | 1 | 0 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 128 | bf16 | 1 | 1 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 128 | bf16 | 1 | 2 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 128 | fp16 | 0 | 0 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 128 | fp16 | 0 | 1 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 128 | fp16 | 0 | 2 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 128 | fp16 | 1 | 0 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 128 | fp16 | 1 | 1 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 128 | fp16 | 1 | 2 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 32 | bf16 | 0 | 0 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 32 | bf16 | 0 | 1 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 32 | bf16 | 0 | 2 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 32 | bf16 | 1 | 0 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 32 | bf16 | 1 | 1 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 32 | bf16 | 1 | 2 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 32 | fp16 | 0 | 0 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 32 | fp16 | 0 | 1 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 32 | fp16 | 0 | 2 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 32 | fp16 | 1 | 0 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 32 | fp16 | 1 | 1 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 32 | fp16 | 1 | 2 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 512 | bf16 | 0 | 0 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 512 | bf16 | 0 | 1 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 512 | bf16 | 0 | 2 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 512 | bf16 | 1 | 0 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 512 | bf16 | 1 | 1 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 512 | bf16 | 1 | 2 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 512 | fp16 | 0 | 0 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 512 | fp16 | 0 | 1 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 512 | fp16 | 0 | 2 | 0.000000 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 512 | fp16 | 1 | 0 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 512 | fp16 | 1 | 1 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |
| 512 | fp16 | 1 | 2 | 0.000000 | 0.000000 | 1.0000 | PASS_EQUIV |

## Drift Characterization (kv_aligned=0)

| span | dtype | max_abs_diff (max/mean) | top1_agreement (min/mean) | samples |
|------|-------|-------------------------|---------------------------|---------|
| 128 | bf16 | 0.000000 / 0.000000 | 1.0000 / 1.0000 | 3 |
| 128 | fp16 | 0.000000 / 0.000000 | 1.0000 / 1.0000 | 3 |
| 32 | bf16 | 0.000000 / 0.000000 | 1.0000 / 1.0000 | 3 |
| 32 | fp16 | 0.000000 / 0.000000 | 1.0000 / 1.0000 | 3 |
| 512 | bf16 | 0.000000 / 0.000000 | 1.0000 / 1.0000 | 3 |
| 512 | fp16 | 0.000000 / 0.000000 | 1.0000 / 1.0000 | 3 |

## Performance Profiling

| span | dtype | mode | wall_time (mean/p95) | logits_bytes (mean) | samples |
|------|-------|------|----------------------|---------------------|---------|
| 128 | bf16 | decode | 55.67s / 56.09s | 19,726,091 | 6 |
| 128 | bf16 | prefill | 55.61s / 56.03s | 19,726,091 | 6 |
| 128 | fp16 | decode | 56.00s / 56.69s | 19,726,091 | 6 |
| 128 | fp16 | prefill | 55.56s / 56.05s | 19,726,091 | 6 |
| 32 | bf16 | decode | 40.81s / 41.45s | 4,919,006 | 6 |
| 32 | bf16 | prefill | 40.87s / 41.97s | 4,919,006 | 6 |
| 32 | fp16 | decode | 41.23s / 41.77s | 4,919,006 | 6 |
| 32 | fp16 | prefill | 41.15s / 41.82s | 4,919,006 | 6 |
| 512 | bf16 | decode | 117.36s / 117.72s | 78,937,041 | 6 |
| 512 | bf16 | prefill | 117.71s / 118.54s | 78,937,041 | 6 |
| 512 | fp16 | decode | 117.50s / 119.09s | 78,937,041 | 6 |
| 512 | fp16 | prefill | 117.16s / 117.41s | 78,937,041 | 6 |
