# B3.61: Model Compatibility Guard Rail

**Date**: 2026-02-06
**Commit**: `df42049`
**Status**: CLOSED (Defensive Correctness Milestone)

## Executive Summary

B3.61 was initiated to execute residual stream bisect on MI300X. During investigation, a **model architecture mismatch** was discovered between:
- **GRETA binary**: Compiled for Llama-2-7B (dim=4096, layers=32, hidden=11008)
- **TinyLlama model**: dim=2048, layers=22, hidden=5632

A **guard rail** was implemented to detect incompatible models and abort safely, preventing illegal GPU memory access.

---

## Final Status

B3.61 does not represent a pipeline failure.

The observed crash (illegal memory access in RMSNorm) was traced to a **model architecture mismatch** (TinyLlama vs GRETA kernels compiled for Llama-2-7B).

A guard rail was implemented to:
- Detect incompatible models before kernel launch
- Abort safely with a clear diagnostic
- Prevent undefined GPU memory access

This closes B3.61 as a **defensive correctness milestone**, not a functional regression.

---

## Root Cause Analysis

**ROOT CAUSE**: `MODEL_DIMENSION_MISMATCH`

| Dimension | GRETA Binary (Llama-2-7B) | TinyLlama 1.1B |
|-----------|---------------------------|----------------|
| dim       | 4096 | 2048 |
| num_layers | 32 | 22 |
| hidden_dim | 11008 | 5632 |
| num_heads | 32 | 16 |

---

## Guard Rail Implementation

Added model compatibility validation in `tools/inference/src/greta_infer.cpp` that:
1. Validates model dimensions (dim, num_layers, hidden_dim, num_heads) before kernel launch
2. Aborts with clear error message if model is incompatible
3. Uses `realpath()` to resolve symlinks and detect actual model file

### Guard Rail Output (Test with TinyLlama)

```
[GUARD_RAIL] Validating model compatibility...
[GUARD_RAIL_ERROR] dim mismatch!
  Expected: 4096
  Got:      2048
[GUARD_RAIL_ERROR] num_layers mismatch!
  Expected: 32
  Got:      22
[GUARD_RAIL_ERROR] hidden_dim mismatch!
  Expected: 11008
  Got:      5632
[GUARD_RAIL] Model path (realpath): /root/models/tinyllama/tinyllama.gguf

[GUARD_RAIL_FATAL] Model incompatible with GRETA kernels!
GRETA binary was compiled with hardcoded tensor dimensions
for Llama-2-7B (dim=4096, heads=32, layers=32).
Running a different architecture will cause illegal memory
access in kernels (RMSNorm, attention, etc.).

Solutions:
  1. Use greta-v1.gguf (Llama-2-7B compatible)
  2. Recompile GRETA with dynamic shape support
  3. Use a model matching GRETA's expected dimensions
```

---

## Next Steps

1. Execute residual stream bisect using greta-v1.gguf (Llama-2-7B compatible)
2. Or recompile GRETA with dynamic shape support for TinyLlama

---

Signed:
L.E.T / Leandro Emanuel Timberini
