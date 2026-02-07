# B3.66 v2 — KV-Aligned Probe: Structural vs Real Drift

**Date**: 2026-02-07  
**Status**: READY_FOR_EXECUTION  
**Branch**: `feature/b3_66_v2_kv_aligned`  
**Scripts**: 
- `tools/benchmarks/run_b3_66_mi300x.sh`
- `tools/benchmarks/analyze_b3_66_prefill_decode_drift.py`

---

## Resumen Ejecutivo (ES)

### Objetivo

Diferenciar entre **drift estructural** (esperado por diferencias en seq_len/cache semantics) y **mismatch real** (bug en implementación de atención KV) en la comparación prefill_last vs decode0.

### Enfoque v2 vs v1

| Aspecto | v1 (as_designed) | v2 (kv_aligned) |
|---------|------------------|-----------------|
| Modo default | Drift probe estándar | Añade evidencia QKV/softmax |
| Evidencia | Hashes de tensores, first failure | Q/K/V hashes + stats pre/post-softmax |
| Veredicto | PASS/FAIL por mismatch | STRUCTURAL_DRIFT vs MISMATCH |
| Complexidad | Baja | Media |

### Modos de Operación

#### `--mode as_designed` (default)
- Comportamiento original: pairing strict + first-fail
- Reporta drift sin distinción de causa
- Útil para comparación histórica

#### `--mode kv_aligned`
- Registra Q/K/V projections hashes
- Stats pre-softmax y post-softmax (min/max/mean)
- Determina si drift se explica por semánticas de KV cache
- Produce veredicto: **STRUCTURAL_DRIFT** (esperado) o **MISMATCH** (bug real)

---

## Executive Summary (EN)

### Objective

Distinguish between **structural drift** (expected due to seq_len/cache semantics differences) and **real mismatch** (KV attention implementation bug) in prefill_last vs decode0 comparison.

### v2 vs v1 Approach

| Aspect | v1 (as_designed) | v2 (kv_aligned) |
|--------|------------------|-----------------|
| Default mode | Standard drift probe | Adds QKV/softmax evidence |
| Evidence | Tensor hashes, first failure | Q/K/V hashes + pre/post-softmax stats |
| Verdict | PASS/FAIL by mismatch | STRUCTURAL_DRIFT vs MISMATCH |
| Complexity | Low | Medium |

### Operation Modes

#### `--mode as_designed` (default)
- Original behavior: strict pairing + first-fail
- Reports drift without cause distinction
- Useful for historical comparison

#### `--mode kv_aligned`
- Records Q/K/V projection hashes
- Pre-softmax and post-softmax stats (min/max/mean)
- Determines if drift is explained by KV cache semantics
- Produces verdict: **STRUCTURAL_DRIFT** (expected) or **MISMATCH** (real bug)

---

## Implementation Details

### Runner: `run_b3_66_mi300x.sh`

```bash
# Usage
./run_b3_66_mi300x.sh <NODE_IP> [YYYY-MM-DD] [--mode as_designed|kv_aligned]

# Example with kv_aligned
./run_b3_66_mi300x.sh 129.212.184.200 2026-02-07 kv_aligned
```

**Features**:
- Parsing `--mode` flag (default: `as_designed`)
- Propagates mode to python analyzer
- Outputs to `artifacts_remote/2026-02-07/b3_66_v2/`
- Includes mode, prompts, and output paths in log

### Analyzer: `analyze_b3_66_prefill_decode_drift.py`

**CLI Options**:
```
--traces-dir/-i  : Directory containing trace JSONL files (required)
--output/-o      : Output markdown file (default: B3_66_V2_ANALYSIS.md)
--mode/-m        : as_designed | kv_aligned (default: as_designed)
```

**kv_aligned Mode Outputs**:
1. **Projection Hashes**: Q/K/V tensors per layer/prompt
2. **Attention Stats**: Pre-softmax and post-softmax min/max/mean
3. **KV Alignment Indicators**: K/V hash consistency, K-V alignment
4. **Verdict**: `STRUCTURAL_DRIFT` | `MISMATCH` | `N/A` (trace incomplete)

**Fallback for Incomplete Traces**:
- If trace lacks Q/K/V/softmax points: report as `N/A`
- Note: "KV alignment not available in current trace; kv_aligned reports QKV/softmax as N/A"

---

## Artifacts Structure

```
artifacts_remote/2026-02-07/b3_66_v2/
├── run/
│   ├── p0_short_as_designed.log
│   ├── p0_short_kv_aligned.log
│   ├── p6_len_16_as_designed.log
│   ├── p6_len_16_kv_aligned.log
│   ├── p6_len_32_as_designed.log
│   └── p6_len_32_kv_aligned.log
├── traces/
│   ├── p0_short_trace.jsonl
│   ├── p0_short_decode_trace.jsonl
│   └── ...
├── B3_66_V2_ANALYSIS.md
├── B3_66_V2_ANALYSIS.tsv
└── B3_66_V2_FINAL_REPORT.md
```

---

## Hypothesis Testing

| Hypothesis | Description |
|------------|-------------|
| H₀ (Null) | Drift es estructural (seq_len/cache semantics) — no bug |
| H₁ (Alt) | Drift es real — bug en atención KV |

### Evidence Required

**For H₀ (STRUCTURAL_DRIFT)**:
- Q hashes vary across prompts (expected: different seq_len)
- K/V hashes consistent for same layer (cache reuse)
- Attention score diff explained by seq_len

**For H₁ (MISMATCH)**:
- Q/K/V hashes inconsistent where expected to match
- Attention scores diverge beyond seq_len expectations
- Clear pattern of implementation bug

---

## Next Steps

1. **Execute as_designed mode** for baseline comparison
2. **Execute kv_aligned mode** for deep evidence
3. **Compare verdicts** across modes
4. **Document conclusion** in `B3_66_V2_FINAL_REPORT.md`

---

**Signed**: L.E.T / Leandro Emanuel Timberini
**Generated**: 2026-02-07
