# B3.66 Prefill vs Decode Drift Probe

**Date**: 2026-02-07
**Commit**: `56b755a`
**Status**: IMPLEMENTED_PENDING_RUN

## Objective

Determine whether any divergence between prefill and decode paths is purely numerical and identify the FIRST tensor/stage where it appears.

## Methodology

### Strict Pairing
Tensors are paired using exact matching of:
- `prompt_id`
- `token_id`
- `logical_tok_idx`
- `pos_id`
- `layer`
- `tensor` name

### Tensors Monitored
1. `embed_out` - Embedding output
2. `residual_pre_attn` - Residual before attention
3. `attn_norm_out` - Attention norm output
4. `q_pre_rope` - Q before RoPE
5. `q_post_rope` - Q after RoPE
6. `attn_out` - Attention output
7. `residual_post_attn` - Residual after attention
8. `ffn_norm_in` - FFN norm input
9. `mlp_out` - MLP output
10. `residual_post_mlp` - Residual after MLP
11. `logits_top1` - Top-1 logits

### Layers Sampled
`GRETA_TRACE_LAYERS=0,1,2,4,8`

### Prompts
- `p0_short` - Minimal single token
- `p6_len_16` - 16 token prompt
- `p6_len_32` - 32 token prompt

### Environment
- `HIP_LAUNCH_BLOCKING=1`
- `AMD_SERIALIZE_KERNEL=3`
- `HSA_ENABLE_SDMA=0`
- `GRETA_SEED=1`
- `GRETA_TRACE_B3_66=1`

## Execution

### Runner
```bash
./tools/benchmarks/run_b3_66_mi300x.sh 129.212.184.200 2026-02-07
```

### Analyzer
```bash
python3 tools/benchmarks/analyze_b3_66_prefill_decode_drift.py \
    --dir artifacts_remote/2026-02-07/b3_66 \
    --out artifacts_remote/2026-02-07/b3_66/b3_66_analysis.txt
```

## Verdict Logic

Failure is determined in this order:
1. `TRACE_OFFSET` - No pair found
2. `EMBED_DRIFT` - embed_out mismatch
3. `RESIDUAL_PRE_ATTN_DRIFT`
4. `ATTN_NORM_DRIFT`
5. `Q_PRE_ROPE_DRIFT`
6. `ROPE_DRIFT` - q_pre ok, q_post mismatch
7. `ATTN_OUT_DRIFT`
8. `RESIDUAL_POST_ATTN_DRIFT`
9. `FFN_NORM_DRIFT`
10. `MLP_OUT_DRIFT`
11. `RESIDUAL_POST_MLP_DRIFT`
12. `LOGITS_DRIFT` - All prior ok, logits differ

## Results Section (Template)

| PROMPT | TOKEN_ID | POS_ID | LAYER | FIRST_FAIL | ROOT_CAUSE | DETAILS |
|--------|----------|--------|-------|------------|------------|---------|
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Conclusion

**ROOT_CAUSE**: TBD
**NEXT_STEP**: TBD

Signed: L.E.T / Leandro Emanuel Timberini
