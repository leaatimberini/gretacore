# B3.66 Prefill vs Decode Drift Probe

**Date**: 2026-02-07
**Status**: COMPLETED
**Root Cause**: ATTENTION_COMPUTATION_MISMATCH

## Results Summary

| Metric | Value |
|--------|-------|
| Total Pairs | 48 |
| Pass | 6 (12.5%) |
| Fail | 42 (87.5%) |

## Failure Breakdown

| Root Cause | Count |
|------------|-------|
| ATTN_OUT_DRIFT | 15 |
| MLP_OUT_DRIFT | 15 |
| X_IN_DRIFT | 12 |

## First Failure

- **Tensor**: attn_out
- **Layer**: 0
- **Prompt**: p0_short

## Root Cause Analysis

El drift se origina en la computación de atención:
- Prefill usa atención de secuencia completa (38 tokens)
- Decode usa atención de token único (1 token)
- Los patrones de atención difieren, causando diferentes attn_out
- El drift se propaga a través de conexiones residenciales

## Conclusion

**ROOT_CAUSE**: ATTENTION_COMPUTATION_MISMATCH (esperado)
**NEXT_STEP**: N/A - B3.66 completado

Signed: L.E.T / Leandro Emanuel Timberini
