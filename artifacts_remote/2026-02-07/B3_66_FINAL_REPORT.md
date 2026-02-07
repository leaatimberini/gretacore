# B3.66 Final Report - Prefill vs Decode Drift Analysis

**Date**: 2026-02-07
**Host**: MI300X (129.212.184.200)
**Run Mode**: Single SSH Connection

---

## Verdict: **FAIL** + ROOT_CAUSE: **ATTENTION_COMPUTATION_MISMATCH**

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Pairs** | 48 |
| **Pass** | 6 (12.5%) |
| **Fail** | 42 (87.5%) |
| **Missing Pairs** | 0 |

## Failure Breakdown

| Root Cause | Count | Percentage |
|------------|-------|------------|
| ATTN_OUT_DRIFT | 15 | 31.25% |
| MLP_OUT_DRIFT | 15 | 31.25% |
| X_IN_DRIFT | 12 | 25.00% |

## First Failure

**Location**: `attn_out` tensor, Layer 0, Prompt `p0_short`

| Metric | Prefill (step=0) | Decode (step=1) |
|--------|-----------------|-----------------|
| **Hash** | 4144761570523161186 | 10153995460507501903 |
| **Match** | ❌ NO | ❌ NO |

**Root Cause**: ATTENTION_COMPUTATION_MISMATCH - The attention output computation differs between prefill and decode phases at the very first layer, causing drift to propagate through all subsequent layers.

---

## Sample Trace Evidence

### Prefill Last - attn_out Layer 0 (p0_short)
```json
{
  "event": "stage_trace",
  "prompt_id": "p0_short",
  "phase": "prefill_last",
  "point": "attn_out",
  "layer": 0,
  "step": 0,
  "pos_id": 37,
  "seq_len": 38,
  "hash": 4144761570523161186,
  "min": -0.00517679,
  "max": 0.00719859,
  "mean": -0.0000471849
}
```

### Decode0 - attn_out Layer 0 (p0_short)
```json
{
  "event": "stage_trace",
  "prompt_id": "p0_short",
  "phase": "decode0",
  "point": "attn_out",
  "layer": 0,
  "step": 1,
  "pos_id": 37,
  "seq_len": 1,
  "hash": 10153995460507501903,
  "min": -0.00519423,
  "max": 0.00725079,
  "mean": -0.0000474128
}
```

---

## Drift Propagation Pattern

```
Layer 0:
  ├── embed_out: ✓ MATCH (hash identical)
  ├── x_in: ✓ MATCH (hash identical)  
  ├── attn_out: ✗ DRIFT (hash mismatch)
  └── mlp_out: ✗ DRIFT (propagated from attn_out)

Layers 1, 2, 4, 8:
  ├── x_in: ✗ DRIFT (propagated from layer-1 mlp_out)
  ├── attn_out: ✗ DRIFT (computation mismatch)
  └── mlp_out: ✗ DRIFT (propagated from attn_out)
```

---

## Root Cause Analysis

The drift originates in the **attention computation** at Layer 0 between prefill and decode phases:

1. **Prefill** uses full sequence attention with causal masking over 38 tokens
2. **Decode** uses single-token attention with causal masking over 1 token
3. The attention patterns (attention scores, mask application) differ
4. This causes different `attn_out` values even for the same token position (pos_id=37)
5. The drift propagates through residual connections to `mlp_out`
6. Subsequent layers receive different `x_in` values, compounding the drift

---

## Recommendations

1. **Investigate attention masking logic** - The causal mask application may differ between prefill and decode
2. **Verify RoPE position encoding** - Check if `pos_id=37` is being applied correctly in both phases
3. **Check attention score normalization** - Softmax normalization may differ due to different sequence lengths
4. **Compare Q/K/V projections** - Verify that weight matrices produce identical outputs for same input

---

**Signed: L.E.T / Leandro Emanuel Timberini**
