# B3.66 v2 FINAL REPORT

**Date**: 2026-02-07  
**Status**: PENDING_EXECUTION  
**Mode**: kv_aligned

---

## 1. Execution Summary

| Mode | Status | First Failure | Pass/Fail Count |
|------|--------|---------------|-----------------|
| as_designed | PENDING | TBD | TBD |
| kv_aligned | PENDING | TBD | TBD |

---

## 2. KV Alignment Analysis (kv_aligned mode)

### Projection Hashes

| Tensor | Layer | Prompt | Hash | Notes |
|--------|-------|--------|------|-------|
| q_proj | 0 | p0_short | TBD | |
| q_proj | 0 | p6_len_16 | TBD | |
| q_proj | 0 | p6_len_32 | TBD | |
| k_proj | 0 | p0_short | TBD | |
| k_proj | 0 | p6_len_16 | TBD | |
| k_proj | 0 | p6_len_32 | TBD | |
| v_proj | 0 | p0_short | TBD | |
| v_proj | 0 | p6_len_16 | TBD | |
| v_proj | 0 | p6_len_32 | TBD | |

### Attention Statistics

| Metric | Pre-Softmax | Post-Softmax |
|--------|-------------|--------------|
| Min | TBD | TBD |
| Max | TBD | TBD |
| Mean | TBD | TBD |

### KV Alignment Indicators

| Indicator | Value | Interpretation |
|-----------|-------|----------------|
| K Hash Consistency | TBD | True = consistent, False = varies |
| V Hash Consistency | TBD | True = consistent, False = varies |
| K-V Aligned | TBD | True = K and V match, False = mismatch |

---

## 3. Drift Structural vs Real Mismatch

### Hypothesis Testing

**H₀ (Null Hypothesis)**: Drift es estructural, no hay bug en atención KV  
**H₁ (Alternative)**: Drift es real, hay bug en atención KV

### Evidence

| Evidence | Supports H₀ | Supports H₁ |
|----------|-------------|-------------|
| K hashes vary across prompts | ✓ | |
| V hashes vary across prompts | ✓ | |
| K-V hashes match for same layer | | |
| Attention outputs diverge | | |

### Conclusion

**PENDING** - Requires execution of both modes.

---

## 4. Recommendations

1. **If structural drift confirmed**: Document as expected behavior, no fix needed
2. **If real mismatch detected**: Investigate KV projection and RoPE implementation
3. **If KV misaligned**: Check attention softmax and score normalization

---

**Generated**: 2026-02-07  
**Script**: `tools/benchmarks/analyze_b3_66_prefill_decode_drift.py`
