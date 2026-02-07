# B3.66 v2 FINAL REPORT

**Date**: 2026-02-07  
**Status**: READY_FOR_EXECUTION  
**Mode**: kv_aligned (dual mode: as_designed + kv_aligned)

---

## 1. Execution Summary

| Mode | Status | First Failure | Pass/Fail Count |
|------|--------|---------------|-----------------|
| as_designed | PENDING | TBD | TBD |
| kv_aligned | PENDING | TBD | TBD |

### Prompts Executed
- `p0_short`: Short prompt (1-2 tokens)
- `p6_len_16`: Prompt length 16 tokens
- `p6_len_32`: Prompt length 32 tokens

### Output Paths
- **Run logs**: `artifacts_remote/2026-02-07/b3_66_v2/run/`
- **Traces**: `artifacts_remote/2026-02-07/b3_66_v2/traces/`
- **Analysis**: `artifacts_remote/2026-02-07/b3_66_v2/B3_66_V2_ANALYSIS.md`
- **TSV**: `artifacts_remote/2026-02-07/b3_66_v2/B3_66_V2_ANALYSIS.tsv`

---

## 2. KV Alignment Analysis (kv_aligned mode)

### Projection Hashes (Layer 0, p0_short)

| Tensor | Layer | Prompt | Hash | Notes |
|--------|-------|--------|------|-------|
| q_proj | 0 | p0_short | TBD | |
| k_proj | 0 | p0_short | TBD | |
| v_proj | 0 | p0_short | TBD | |

### Attention Statistics

| Metric | Pre-Softmax | Post-Softmax |
|--------|-------------|--------------|
| Min | TBD | TBD |
| Max | TBD | TBD |
| Mean | TBD | TBD |

### KV Alignment Indicators

| Indicator | Value | Interpretation |
|-----------|-------|-----------------|
| K Hash Consistency | TBD | True = consistent, False = varies |
| V Hash Consistency | TBD | True = consistent, False = varies |
| K-V Aligned | TBD | True = K and V match, False = mismatch |

---

## 3. Drift Structural vs Real Mismatch

### Hypothesis Testing

| Hypothesis | Description |
|------------|-------------|
| H₀ (Null) | Drift es estructural — diferencias por seq_len/cache semantics — no bug |
| H₁ (Alt) | Drift es real — bug en implementación de atención KV |

### Evidence Matrix

| Evidence | Supports H₀ | Supports H₁ |
|----------|-------------|-------------|
| Q hashes vary across prompts (expected: different seq_len) | ✓ | |
| K/V hashes consistent for same layer (cache reuse) | ✓ | |
| Attention scores diverge beyond seq_len expectations | | ✓ |
| Q/K/V hashes inconsistent where expected to match | | ✓ |

### Verdict Criteria

| Verdict | Condition |
|---------|-----------|
| **STRUCTURAL_DRIFT** | Drift explained by seq_len/cache semantics |
| **MISMATCH** | Evidence of unexpected inconsistency (bug) |
| **N/A** | Trace incomplete (KV alignment not available) |

---

## 4. First Failure by Mode

### as_designed Mode
- **First Failure**: TBD (pending execution)
- **Root Cause**: TBD
- **Tensor**: TBD

### kv_aligned Mode
- **First Failure**: TBD (pending execution)
- **Root Cause**: TBD
- **Evidence**: QKV/softmax statistics

---

## 5. Evidence Snippets (JSON)

> **Note**: Evidence snippets are limited to 25 words maximum per citation.

### Sample Structure (to be populated after execution)

```json
{
  "prompt": "p0_short",
  "layer": 0,
  "tensor": "q_proj",
  "hash": "0x...",
  "phase": "prefill_last"
}
```

---

## 6. Conclusion

**PENDING** — Requires execution of both modes.

### If STRUCTURAL_DRIFT confirmed:
- Document as expected behavior
- No fix needed
- Variance due to seq_len/cache semantics is normal

### If MISMATCH detected:
- Investigate KV projection implementation
- Check RoPE application
- Audit attention softmax and score normalization

---

## 7. Recommendations

1. **Execute both modes** for complete analysis:
   ```bash
   # Mode 1: as_designed (baseline)
   ./run_b3_66_mi300x.sh 129.212.184.200 2026-02-07 as_designed
   
   # Mode 2: kv_aligned (deep probe)
   ./run_b3_66_mi300x.sh 129.212.184.200 2026-02-07 kv_aligned
   ```

2. **Compare results** across modes to validate conclusions

3. **Document findings** in appropriate AMD report

---

## 8. Artifacts Reference

| File | Path |
|------|------|
| Runner script | `tools/benchmarks/run_b3_66_mi300x.sh` |
| Analyzer script | `tools/benchmarks/analyze_b3_66_prefill_decode_drift.py` |
| AMD probe doc | `docs/AMD/2026_02_07_B3_66_v2_kv_aligned_probe.md` |
| Raw traces | `artifacts_remote/2026-02-07/b3_66_v2/traces/` |
| Run logs | `artifacts_remote/2026-02-07/b3_66_v2/run/` |

---

**Generated**: 2026-02-07  
**Signed**: L.E.T / Leandro Emanuel Timberini
