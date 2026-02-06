# B3.64: Numerical Drift Audit

**Date**: 2026-02-06  
**Status**: READY_TO_RUN  
**Type**: Numerical Analysis  

## Objective

Determine whether remaining prefill/decode divergence (if any) is **numerical only** (accumulation order/precision) and localize the first point where divergence appears, with **strict token logical pairing**.

## Methodology

### A. Strict Pairing
- Compare `prefill_last` vs `decode0` re-processing of the **same logical token**
- Use StageTrace stable metadata: `token_id`, `prompt_id`, `phase`, `pos_id`, `logical_tok_idx`, `step`
- If no exact pairing → `ROOT_CAUSE=TRACE_OFFSET` and abort with diagnosis

### B. Points to Trace (Layer 0, optional L1/L2)
1. `embedding_out` (control hash)
2. `rmsnorm(attn)` out (`norm_out`)
3. `q_pre_rope`
4. `q_post_rope`
5. `attn_out`
6. `residual_post_attn`
7. `ffn_norm_in`
8. `logits` (topk + hash + stats)

### C. Metrics
- `hash64` (same as B3.59/60)
- `nz_count`
- `abs_sum`
- `MAE` (prefill vs decode) per tensor
- **Logits**: top1 id, top1 logit, top5 ids+logits, KL approx (if cheap), L∞ and L2 diff
- NaN/Inf: report

### D. Verdict/Root Cause
| Code | Description |
|------|-------------|
| **PASS** | MAE < 1e-6 on all points + top1 match |
| **NORM_NUMERICS** | First fail in norm_out with embedding_out OK |
| **ROPE_NUMERICS** | q_pre OK but q_post diverges |
| **ATTN_NUMERICS** | q_post OK but attn_out diverges |
| **RESIDUAL_NUMERICS** | attn_out OK but residual_post_attn diverges |
| **FFN_NORM_NUMERICS** | residual OK but ffn_norm_in diverges |
| **LOGITS_NUMERICS** | All OK but logits diverge |
| **TRACE_OFFSET** | Pairing not exact |

## Configuration

### Prompts / Pos Target
- `p0_short`
- `p6_len_16` (pos target 826)
- `p6_len_32` (pos target 1652)

Max tokens: 5, greedy

### Environment Flags
```bash
GRETA_B3_64=1
GRETA_TRACE_B3_64=1
GRETA_TRACE_B3_64_DIR=<dir>
GRETA_TRACE_STAGE=1
GRETA_TRACE_STAGE_DEBUG_INPUT=1
```

## Files

| File | Description |
|------|-------------|
| `tools/benchmarks/run_b3_64_mi300x.sh` | Remote execution runner |
| `tools/benchmarks/analyze_b3_64_numerical_drift.py` | Analysis script |
| `artifacts_remote/<date>/b3_64/` | Artifacts directory |

## Usage

```bash
# Run on MI300X
./tools/benchmarks/run_b3_64_mi300x.sh 129.212.184.200 2026-02-06

# Analyze results
python3 tools/benchmarks/analyze_b3_64_numerical_drift.py \
  --dir artifacts_remote/2026-02-06/b3_64 \
  --out artifacts_remote/2026-02-06/b3_64/b3_64_analysis.txt
```

## Dependencies

- B3.61: Residual Stream Bisect (traces baseline)
- B3.63: HIP D2H Fix (safe wrappers for `hipMemcpyAsync`)

## Signed: L.E.T / Leandro Emanuel Timberini
