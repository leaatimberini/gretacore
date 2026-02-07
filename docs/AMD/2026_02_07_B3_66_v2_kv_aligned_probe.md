# B3.66 v2 kv_aligned Probe Report / Informe de Sonda kv_aligned

**Date / Fecha**: 2026-02-07  
**Author / Autor**: L.E.T (Leandro Emanuel Timberini)  
**Mode / Modo**: kv_aligned  
**Status / Estado**: PREPARED_FOR_EXECUTION

---

## 1. Executive Summary / Resumen Ejecutivo

### ES
**B3.66 v2** introduce un nuevo modo de operación `kv_aligned` diseñado para investigar si el drift observado entre prefill y decode es:

1. **Drift estructural** (diferencias inherentes al pipeline) - esperado
2. **Mismatch real** (bug/caption en atención KV) - requiere fix

### EN
**B3.66 v2** introduces a new `kv_aligned` operation mode designed to investigate whether the observed drift between prefill and decode is:

1. **Structural drift** (inherent pipeline differences) - expected
2. **Real mismatch** (KV attention bug/caption) - requires fix

---

## 2. Cambios vs v1 / Changes vs v1

| Aspecto / Aspect | B3.66 v1 | B3.66 v2 |
|-----------------|----------|----------|
| **Modo / Mode** | Implícito (solo as_designed) | Explícito con flag CLI |
| **Flag CLI** | N/A | `--mode as_designed\|kv_aligned` |
| **Env Variable** | `GRETA_TRACE_B3_66=1` | `GRETA_B3_66_MODE=kv_aligned` |
| **Analyzer** | Solo hash matching | + Q/K/V hashes + attention stats |
| **KV Probes** | No | Sí (hash consistency, alignment) |
| **Attention Scores** | No | Pre-softmax + post-softmax stats |

---

## 3. Architecture / Arquitectura

### 3.1 Runner Modifications / Modificaciones del Runner

```bash
./run_b3_66_mi300x.sh <NODE_IP> [YYYY-MM-DD] [--mode as_designed|kv_aligned]
```

**Key changes / Cambios clave:**

```bash
# Mode validation
if [[ "$MODE" != "as_designed" && "$MODE" != "kv_aligned" ]]; then
    echo "ERROR: Invalid mode '$MODE'. Must be 'as_designed' or 'kv_aligned'"
    exit 1
fi

# Env var propagation
export GRETA_B3_66_MODE=$MODE

# Grep_infer invocation
./tools/inference/build/grep_infer \
    --model ./models/greta-v1.gguf \
    --prompt tools/benchmarks/prompts/${prompt}.txt \
    --max-tokens 1 \
    --seed 1 \
    --mode $MODE \
    2>&1 | tee $REMOTE_BASE/artifacts_remote/$DATE/$RUN_DIR/run/${prompt}_${MODE}.log
```

### 3.2 Analyzer Enhancements / Mejoras del Analyzer

```python
# Mode: kv_aligned
- Registers Q/K/V projection hashes per layer
- Attention score statistics (pre-softmax, post-softmax)
- KV alignment indicators (consistency, K-V match)
- Status: PASS | KV_MISMATCH
```

---

## 4. Results by Mode / Resultados por Modo

### 4.1 Mode: as_designed (B3.66 v1 equivalent)

**Expected outcome / Resultado esperado:**
- FAIL (expected) - ATTENTION_COMPUTATION_MISMATCH
- Drift detected in attention computation stage
- First failure: Q/K/V projection mismatch

### 4.2 Mode: kv_aligned (NEW)

**Expected outcome / Resultado esperado:**
- Probe to distinguish:
  - **Structural drift**: K and V have different hashes (expected due to different computation paths)
  - **Real mismatch**: K and V hashes should match but don't (bug)

**KV Alignment Indicators / Indicadores de Alineación KV:**

| Indicator | Expected if Structural | Expected if Bug |
|-----------|------------------------|-----------------|
| K hash consistency | May vary across prompts | Should be consistent |
| V hash consistency | May vary across prompts | Should be consistent |
| K-V aligned | **FALSE** (different computations) | **TRUE** (same semantics) |

---

## 5. Interpretation Guide / Guía de Interpretación

### 5.1 Drift Estructural vs Mismatch Real

| Scenario | Evidence | Interpretation |
|----------|-----------|---------------|
| **Structural Drift** | K and V hashes differ, but attention outputs are semantically correct | Expected behavior - different computation paths |
| **Real Mismatch** | K and V hashes differ AND attention outputs diverge | Bug in KV projection or attention |
| **KV Misalignment** | K and V hashes match but attention scores wrong | Caption in RoPE or softmax |

### 5.2 Hypothesis Testing / Prueba de Hipótesis

**H₀ (Null Hypothesis):** Drift es estructural, no hay bug en atención KV  
**H₁ (Alternative):** Drift es real, hay bug en atención KV

**Evidence collected:**
1. Q/K/V projection hashes per layer
2. Attention score distributions (pre/post softmax)
3. KV alignment consistency across prompts

---

## 6. Execution Instructions / Instrucciones de Ejecución

### Run v1 (as_designed):
```bash
./tools/benchmarks/run_b3_66_mi300x.sh 129.212.184.200 2026-02-07 as_designed
```

### Run v2 (kv_aligned):
```bash
./tools/benchmarks/run_b3_66_mi300x.sh 129.212.184.200 2026-02-07 kv_aligned
```

### Analyze results:
```bash
python3 tools/benchmarks/analyze_b3_66_prefill_decode_drift.py \
    --traces-dir artifacts_remote/2026-02-07/b3_66_v2/traces \
    --mode kv_aligned \
    --output artifacts_remote/2026-02-07/b3_66_v2/B3_66_V2_FINAL_REPORT.md
```

---

## 7. Artifact Structure / Estructura de Artefactos

```
artifacts_remote/
└── 2026-02-07/
    └── b3_66_v2/
        ├── run/
        │   ├── p0_short_as_designed.log
        │   ├── p0_short_kv_aligned.log
        │   ├── p6_len_16_as_designed.log
        │   ├── p6_len_16_kv_aligned.log
        │   ├── p6_len_32_as_designed.log
        │   └── p6_len_32_kv_aligned.log
        ├── traces/
        │   ├── p0_short_*_trace.jsonl
        │   ├── p6_len_16_*_trace.jsonl
        │   └── p6_len_32_*_trace.jsonl
        ├── B3_66_V2_FINAL_REPORT.md
        └── B3_66_V2_ANALYSIS.tsv
```

---

## 8. Next Steps / Próximos Pasos

1. **Execute both modes** on MI300X
2. **Compare KV alignment** between as_designed and kv_aligned
3. **If KV aligned but attention fails**: Bug in RoPE/softmax
4. **If KV not aligned but semantics OK**: Structural drift (expected)
5. **Document findings** in follow-up AMD report

---

**Signed / Firmado**: L.E.T / Leandro Emanuel Timberini  
**Date / Fecha**: 2026-02-07
