# B3.64: HIP D2H Regression

**Date**: 2026-02-06  
**Status**: HIP_D2H_REGRESSION  
**Commit**: `28a5e49`

## Error
```
Generation error: hipMemcpy D2H failed: an illegal memory access was encountered
```

## Analysis

### Execution Results
| Prompt | Status | Error |
|--------|--------|-------|
| p0_short | **FAILED** | hipMemcpy D2H illegal memory access |
| p6_len_16 | FAILED | Model file not found |
| p6_len_32 | FAILED | Model file not found |

### Key Observations
- **p0_short.log**: Error occurred during generation phase after successful weight loading
- B3.63 applied 13 safe wrappers for `hipMemcpyAsync` in `d2h_safe.hpp`
- Error persists in `block_scheduler.cpp` - regression confirmed
- Additional wrappers or different fix strategy required

## Root Cause
Unknown - requires investigation

The fix applied in B3.63 (d2h_safe.hpp safe wrappers) was insufficient. The error occurs during the generation phase, suggesting there may be additional hipMemcpy D2H calls that were not covered by the existing wrappers.

## Artifacts
- [`artifacts_remote/2026-02-06/b3_64/run/p0_short.log`](artifacts_remote/2026-02-06/b3_64/run/p0_short.log) - Full error trace
- [`artifacts_remote/2026-02-06/b3_64/run/p6_len_16.log`](artifacts_remote/2026-02-06/b3_64/run/p6_len_16.log) - Model not found
- [`artifacts_remote/2026-02-06/b3_64/run/p6_len_32.log`](artifacts_remote/2026-02-06/b3_64/run/p6_len_32.log) - Model not found

## Next Steps
1. Investigate `block_scheduler.cpp` for additional D2H transfers not covered by safe wrappers
2. Review all hipMemcpy/hipMemcpyAsync calls in inference pipeline
3. Consider adding synchronization points before D2H transfers
4. Run local test to confirm model loading (p6_len_* errors are remote-specific)

**Signed**: L.E.T / Leandro Emanuel Timberini
