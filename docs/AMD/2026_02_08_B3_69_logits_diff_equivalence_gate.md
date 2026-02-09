# B3.69 Logits-Diff Equivalence Gate

**Date**: 2026-02-08  
**Ticket**: B3.69  
**Status**: IN_PROGRESS  

## Objective

Enable real numeric logits-diff comparison between prefill and decode modes over a configurable span of N tokens (N > 1), providing a true equivalence gate with quantitative metrics.

## Changes from B3.67

| Aspect | B3.67 | B3.69 |
|--------|-------|-------|
| Span | 1 token | N tokens (default 32) |
| Comparison | Metadata-only | Real logits diff |
| Metrics | None (structural) | max_abs_diff, p99_abs_diff, top1_agreement |
| Verdict (kv=1) | PASS_EQUIV_METADATA | PASS_EQUIV / FAIL_EQUIV |

## Implementation

### greta_infer Changes

- **New flag**: `--dump-logits-span N` (default 1)
- **Behavior**: Overrides `max_tokens` to N when `--dump-logits` is set
- **Output**: Real `logits.jsonl.gz` with full vocab logits per token
- **AlignmentStep**: Extended with `full_logits` vector

### Runner Script

`run_b3_69_logits_diff_gate.sh`:
- Accepts `--span N` argument (default 32)
- Passes `--dump-logits-span N` to greta_infer
- Emits `config.json` with `span` field

### Analyzer

`analyze_b3_67_equivalence_guardrail.py --mode b3_69`:
- Reads real logits from `logits.jsonl.gz`
- Computes metrics: max_abs_diff, p99_abs_diff, top1_agreement
- Applies thresholds for kv_aligned=1:
  - p99_abs_diff ≤ 1e-3
  - max_abs_diff ≤ 5e-3
  - top1_agreement ≥ 0.999
- INCONCLUSIVE if logits missing (no metadata-only fallback)

## Thresholds

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| p99_abs_diff | ≤ 1e-3 | 99th percentile within bf16 precision |
| max_abs_diff | ≤ 5e-3 | Worst-case within 5x bf16 ulp |
| top1_agreement | ≥ 0.999 | Same top token in 99.9% of positions |

## MI300X Results

*Pending execution*

## Artifact Layout

```
artifacts_remote/<DATE>/b3_69/
  runs/
    config.json
    kv_aligned_0/seed_0/prefill/
    kv_aligned_0/seed_0/decode/
    ...
  B3_69_LOGITS_DIFF_GATE.md
```

## Verification Plan

```bash
./tools/benchmarks/run_b3_69_logits_diff_gate.sh 129.212.184.200 2026-02-08 --span 32 --seeds "0,1,2"
```

**Success Criteria**:
- All kv_aligned=1 configs: PASS_EQUIV
- All kv_aligned=0 configs: EXPECTED_DRIFT
- global_verdict: PASS_GUARDRAIL

---

## Changelog

- 2026-02-08: Initial implementation
