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

---

## B3.64.1 D2H Illegal Memory Access Forensics (EN)

### Crash Summary

**Error**: `hipMemcpy D2H failed: an illegal memory access was encountered`

The B3.64 numerical drift audit encountered a critical memory fault during Device-to-Host (D2H) transfer. The error indicates that the device pointer passed to `hipMemcpy` is invalid, causing an illegal memory access.

### Instrumentation Implemented (D2H Safe Wrappers)

Per commit `1173ae5`, D2H safe wrappers were added with debug instrumentation:
- Logging of device pointers before D2H operations
- Validation of pointer validity
- Async/sync handling for HIP operations

### Hypothesis Test Results

| Hypothesis | Status | Description |
|------------|--------|-------------|
| H1: Use-after-free | **INCONCLUSIVE** | Wrappers not invoked - no free evidence |
| H2: Incorrect offset | **INCONCLUSIVE** | Wrappers not invoked - no tracing data |
| H3: Race condition | **FAIL** | Crash persists with sync - not a race |

### Probable Root Cause

**BUFFER_INVALID**: The device pointer (`src_device`) passed to `hipMemcpy` does not point to valid device memory.

Likely scenarios:
1. **Corrupted pointer**: Device pointer was overwritten or not initialized correctly
2. **Buffer freed**: Buffer was freed (hipFree) before D2H transfer
3. **Invalid address**: Device address was not obtained correctly from hipMalloc

### Recommendations

1. **Additional instrumentation**: Add pointer logging before hipMemcpy
2. **Pointer validation**: Verify device pointer validity before D2H
3. **Buffer lifecycle review**: Ensure buffers are not prematurely freed
4. **HIP memory inspector**: Use AMD tools to inspect memory state

### Artifact Reference

Full forensics analysis: `artifacts_remote/2026-02-07/b3_64/b3_64_analysis.txt`

---

## B3.64.2 RoPE Kernel Launch Diagnostics (EN)

### Evolution of the Issue

| Stage | Date | Error | Root Cause |
|-------|------|-------|-------------|
| Original | 2026-02-07 | `hipMemcpy D2H failed` | Initially suspected D2H transfer |
| Evolved | 2026-02-07 | `RoPE Q launch failed` | Error occurs BEFORE D2H - upstream kernel |
| Current | 2026-02-07 | `RoPE Q launch failed: illegal memory access` | RoPE kernel launch fault |

**Critical Finding**: The D2H safe wrapper `[D2H_SAFE_WRAPPER]` was NOT invoked. This proves the crash occurs in the kernel launch path, NOT in D2H operations.

### Root Cause Analysis

**Location**: `src/inference/src/block_scheduler.cpp:2083-2100`

**Kernel Launch Sites Identified**:
1. `launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos)` - Fused KV path (line 2084)
2. `launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos)` - Decode path Q (line 2089)
3. `launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, d_pos)` - Decode path K (line 2092)
4. `launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, pos)` - Prefill path Q (line 2096)
5. `launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, pos)` - Prefill path K (line 2099)

### Instrumentation Added (commit e3143aa)

Added `greta_rope_diag::validate_rope_params()` helper with:

1. **Pointer Validation**:
   - `x_ptr` != nullptr
   - `pos_ptr` != nullptr (for decode path with `d_pos`)

2. **Dimension Validation**:
   - `seq_len` > 0
   - `num_heads` > 0
   - `head_dim` > 0
   - `head_dim` is even (RoPE requires even)

3. **Configuration Validation**:
   - `rope_base` > 0.0

4. **Structured Logging**:
   ```
   [ROPE_DIAG] <kernel_name> x_ptr=<hex> seq_len=<n> num_heads=<n> head_dim=<n> rope_base=<f> pos_ptr=<hex> valid=<true|false>
   ```

### Hypotheses to Test

| Hypothesis | Description | Test |
|------------|-------------|------|
| H1 | Null pointer in `q` or `d_pos` | Check `[ROPE_DIAG]` output for `x_ptr=0x0` |
| H2 | Invalid `seq_len` (0 or unexpected) | Check `[ROPE_DIAG]` output for `seq_len=0` |
| H3 | Invalid `rope_base` (0 or corrupted) | Check `[ROPE_DIAG]` output for `rope_base<=0` |
| H4 | `head_dim` is odd (must be even) | Check `[ROPE_DIAG]` output for odd `head_dim` |
| H5 | Module/kernel loading failure | Check for HIP API errors before launch |

### Expected Diagnostic Output

```
[ROPE_DIAG] RoPE Q decode x_ptr=0x7f... seq_len=1 num_heads=32 head_dim=128 rope_base=10000.0 pos_ptr=0x7f... valid=true
```

If crash occurs, the `[ROPE_DIAG]` line will show which parameter is invalid.

### Next Steps

1. **Run with diagnostics**: Execute `run_b3_64_mi300x.sh` to capture `[ROPE_DIAG]` output
2. **Identify invalid parameter**: Determine which validation fails
3. **Trace to source**: Find where invalid parameter originates
4. **Fix or escalate**: Apply fix or create B3.65 for deeper kernel investigation

### Files Modified

| File | Change |
|------|--------|
| `src/inference/src/block_scheduler.cpp` | Added `greta_rope_diag` namespace and 4 validation calls |

---

## Signed: L.E.T / Leandro Emanuel Timberini
