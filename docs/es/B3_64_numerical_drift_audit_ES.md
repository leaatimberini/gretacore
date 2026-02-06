# B3.64: Auditoría de Drift Numérico

**Fecha**: 2026-02-06  
**Estado**: LISTO_PARA_EJECUTAR  
**Tipo**: Análisis Numérico  

## Objetivo

Determinar si la divergencia prefill/decode restante (si existe) es **solo numérica** (orden de acumulación/precisión) y localizar el primer punto donde aparece, con **pairing estricto por token lógico**.

## Metodología

### A. Pairing Estricto
- Comparar `prefill_last` vs `decode0` re-procesando el **mismo token lógico**
- Usar metadata estable de StageTrace: `token_id`, `prompt_id`, `phase`, `pos_id`, `logical_tok_idx`, `step`
- Si no hay pairing exacto → `ROOT_CAUSE=TRACE_OFFSET` y abortar con diagnóstico

### B. Puntos a Trazar (Capa 0, opcional L1/L2)
1. `embedding_out` (hash de control)
2. `rmsnorm(attn)` out (`norm_out`)
3. `q_pre_rope`
4. `q_post_rope`
5. `attn_out`
6. `residual_post_attn`
7. `ffn_norm_in`
8. `logits` (topk + hash + stats)

### C. Métricas
- `hash64` (mismo que B3.59/60)
- `nz_count`
- `abs_sum`
- `MAE` (prefill vs decode) por tensor
- **Logits**: top1 id, top1 logit, top5 ids+logits, KL aprox (si es barato), L∞ y L2 diff
- NaN/Inf: reportar

### D. Veredicto/Raíz Causa
| Código | Descripción |
|--------|-------------|
| **PASS** | MAE < 1e-6 en todos los puntos + top1 match |
| **NORM_NUMERICS** | Primer fail en norm_out con embedding_out OK |
| **ROPE_NUMERICS** | q_pre OK pero q_post diverge |
| **ATTN_NUMERICS** | q_post OK pero attn_out diverge |
| **RESIDUAL_NUMERICS** | attn_out OK pero residual_post_attn diverge |
| **FFN_NORM_NUMERICS** | residual OK pero ffn_norm_in diverge |
| **LOGITS_NUMERICS** | Todo OK pero logits divergen |
| **TRACE_OFFSET** | Pairing no exacto |

## Configuración

### Prompts / Pos Objetivo
- `p0_short`
- `p6_len_16` (pos objetivo 826)
- `p6_len_32` (pos objetivo 1652)

Max tokens: 5, greedy

### Flags de Entorno
```bash
GRETA_B3_64=1
GRETA_TRACE_B3_64=1
GRETA_TRACE_B3_64_DIR=<dir>
GRETA_TRACE_STAGE=1
GRETA_TRACE_STAGE_DEBUG_INPUT=1
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `tools/benchmarks/run_b3_64_mi300x.sh` | Runner de ejecución remota |
| `tools/benchmarks/analyze_b3_64_numerical_drift.py` | Script de análisis |
| `artifacts_remote/<date>/b3_64/` | Directorio de artefactos |

## Uso

```bash
# Ejecutar en MI300X
./tools/benchmarks/run_b3_64_mi300x.sh 129.212.184.200 2026-02-06

# Analizar resultados
python3 tools/benchmarks/analyze_b3_64_numerical_drift.py \
  --dir artifacts_remote/2026-02-06/b3_64 \
  --out artifacts_remote/2026-02-06/b3_64/b3_64_analysis.txt
```

## Dependencias

- B3.61: Residual Stream Bisect (baseline de trazas)
- B3.63: HIP D2H Fix (wrappers seguros para `hipMemcpyAsync`)

## Firmado: L.E.T / Leandro Emanuel Timberini
