# B3.64 RoPE Kernel Launch Failure - Diagnostic Plan

**Date:** 2026-02-07
**Objective:** Diagnose "RoPE Q launch failed: an illegal memory access was encountered"

---

## Context

**Evolution of the issue:**
- Original symptom: `hipMemcpy D2H failed: an illegal memory access was encountered`
- Current error: `RoPE Q launch failed: an illegal memory access was encountered`
- D2H wrapper `[D2H_SAFE_WRAPPER] engaged` did NOT appear - crash occurs BEFORE D2H

**Root cause hypothesis:** Bug is UPSTREAM in RoPE kernel launch, not in D2H operations.

---

## Files Involved

### Kernel Declaration
- [`src/rt/backend/hip/include/gcore/rt/hip/kernels/attention_kernels.hpp`](src/rt/backend/hip/include/gcore/rt/hip/kernels/attention_kernels.hpp:18)
  - `void launch_rope(hipStream_t stream, float *x, uint32_t seq_len, uint32_t num_heads, uint32_t head_dim, float base, uint32_t pos_offset);`

### Kernel Invocation
- [`src/inference/src/block_scheduler.cpp`](src/inference/src/block_scheduler.cpp:2008)
  - Line 2014: `launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos)`
  - Line 2019: `launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, d_pos)`
  - Line 2022: `launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, d_pos)`
  - Line 2026: `launch_rope(hip_stream, q, S, Hq, Dh, config_.rope_base, pos)`
  - Line 2029: `launch_rope(hip_stream, k, S, Hkv, Dh, config_.rope_base, pos)`

---

## Diagnostic Hypothesis

### H1: Null pointer in kernel arguments
- `q` or `k` buffers are null or invalid device pointers
- `d_pos` pointer is invalid (used in decode path)

### H2: Invalid dimension parameters
- `S` (seq_len) = 0 or unexpectedly zero
- `Hq` or `Hkv` (num_heads) = 0
- `Dh` (head_dim) is odd (must be even for RoPE)

### H3: rope_base configuration
- `config_.rope_base` = 0 or invalid value
- Missing or corrupted RoPE frequency base from model

### H4: Module/kernel loading failure
- hipModuleLoad or hipModuleGetFunction fails
- Kernel binary not properly loaded

### H5: Grid/block configuration mismatch
- Grid dimensions exceed device limits
- Block size incompatible with MI300X architecture

---

## Diagnostic Steps

### Step 1: Run with verbose HIP debugging
```bash
cd /root/gretacore
export HIP_VISIBLE_DEVICES=0
export HIP_DUMP_API_LOG=1
export HIP_LAUNCH_BLOCKING=1
export AMD_SERIALIZE_KERNEL=3
export HSA_ENABLE_SDMA=0
./tools/inference/greta_infer --model models/greta-v1.gguf --prompt "Hello" --max-tokens 5 2>&1 | tee rope_debug.log
```

### Step 2: Capture kernel launch parameters
Add instrumentation BEFORE `launch_rope` calls in `block_scheduler.cpp`:

```cpp
fprintf(stderr, "[ROPE_DIAG] q=%p k=%p S=%u Hq=%u Hkv=%u Dh=%u rope_base=%.1f d_pos=%p\n",
        q, k, S, Hq, Hkv, Dh, config_.rope_base, d_pos);
```

### Step 3: Verify model configuration
Check that `llama.rope.freq_base` is correctly loaded:
- Verify `config_.rope_base` matches expected value (typically 10000.0 or 100000.0)
- Check model GGUF metadata

### Step 4: Compare with B3.63 working commit
```bash
git log --oneline | grep -i b3.63
git diff <b3.63_commit>..HEAD -- src/inference/src/block_scheduler.cpp
```

---

## Expected Outputs

### Case A - Null pointer
```
[ROPE_DIAG] q=0x0 k=0x7f... S=1 Hq=32 Hkv=32 Dh=128 rope_base=10000.0 d_pos=0x7f...
```

### Case B - Invalid dimensions
```
[ROPE_DIAG] q=0x7f... k=0x7f... S=0 Hq=32 Hkv=32 Dh=128 rope_base=10000.0 d_pos=0x7f...
```

### Case C - Module load failure
```
error: hipModuleLoad failed: ... (or similar)
```

---

## Next Actions

1. **Execute diagnostic run** on MI300X with verbose logging
2. **Add kernel parameter logging** to block_scheduler.cpp
3. **Verify model metadata** for rope_base configuration
4. **Compare with B3.63** to identify parameter changes

---

**Signed:** L.E.T / Leandro Emanuel Timberini
