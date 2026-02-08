# B3.67 Equivalence Guardrail (Prefill vs Decode)

## Metadata

| Campo | Valor |
|-------|-------|
| **Ticket** | B3.67 |
| **Date** | 2026-02-07 |
| **Author** | L.E.T |
| **Status** | COMPLETED (LOCAL) \| PENDING_MI300X |
| **Labels** | Repo branch: main; pending greta_infer kv_aligned |
| **Parent** | B3.66, B3.66 v2 |

## Resumen Ejecutivo

Este benchmark implementa un **guardrail de equivalencia** para comparar logits entre modos prefill y decode, con diferentes configuraciones de `kv_aligned`:

- **kv_aligned=1**: Comprobación de equivalencia estricta dentro de umbrales
- **kv_aligned=0**: Drift esperado (semántica prefill vs decode diferente)

B3.66 identificó drift estructural debido a semánticas diferentes entre prefill y decode. B3.67 formaliza este comportamiento con veredictos automatizados.

## Objetivos

1. **Verificar equivalencia numérica** cuando KV está alineado (kv_aligned=1)
2. **Documentar drift esperado** cuando KV no está alineado (kv_aligned=0)
3. **Fail-fast** en caso de drift inesperado en kv_aligned=1
4. **First-fail tracking** para diagnóstico rápido

## Antecedentes

### B3.66 / B3.66 v2

- **Resultado**: EXPECTED por STRUCTURAL_DRIFT (semántica prefill vs decode)
- **Evidencia**: kv_aligned mode agregó claridad sobre alineación de KV
- **Hallazgo**: El drift en atención es esperado cuando prefill usa múltiples tokens vs decode single token

### B3.67 como Guardrail

B3.67 convierte el conocimiento de B3.66 en un guardrail automatizado:
- Si kv_aligned=1 Y drift > umbrales → FAIL (regresión)
- Si kv_aligned=0 → EXPECTED_DRIFT (comportamiento documentado)

## Metodología

### Matriz de Ejecución

| Parámetro | Valores |
|-----------|---------|
| `kv_aligned` | 0, 1 |
| `mode` | prefill, decode |
| `dtype` | bf16 (default) |
| `prompt_len` | 512 (default) |
| `gen_len` | 128 (default) |
| `seeds` | 0, 1, 2 |

### Arquitectura del Benchmark

```mermaid
flowchart TD
    A[Runner Shell] --> B[Loop: kv_aligned 0,1]
    B --> C[Loop: seeds 0,1,2]
    C --> D[Run Prefill Mode]
    C --> E[Run Decode Mode]
    D --> F[Save Logits + Metadata]
    E --> F
    F --> G{Analyzer Python}
    G --> H{Compare Logits}
    H -->|kv_aligned=1| I{Within Thresholds?}
    H -->|kv_aligned=0| J[EXPECTED_DRIFT]
    I -->|Yes| K[Record PASS_EQUIV]
    I -->|No| L[Record FAIL_EQUIV + First Fail]
    J --> M[Record EXPECTED_DRIFT]
    K --> N[Summary + Report]
    L --> N
    M --> N
```

## Métricas de Logits

### Métricas Computadas (por token)

| Métrica | Descripción |
|---------|-------------|
| `max_abs_diff` | Diferencia absoluta máxima entre logits |
| `p99_abs_diff` | Percentil 99 de diferencias absolutas |
| `top1_agreement` | Porcentaje de tokens donde argmax coincide |
| `cos_sim_mean` | Similaridad coseno promedio (opcional) |

### Umbrales de Equivalencia

| Métrica | Threshold | Condición |
|---------|-----------|-----------|
| `p99_abs_diff` | ≤ 1e-3 | PASS_EQUIV |
| `max_abs_diff` | ≤ 5e-3 | PASS_EQUIV |
| `top1_agreement` | ≥ 0.999 | PASS_EQUIV |

### Veredictos

| kv_aligned | Condición | Veredicto | Acción |
|------------|-----------|-----------|--------|
| 1 | Cumple todos los thresholds | `PASS_EQUIV` | Continuar |
| 1 | No cumple algún threshold | `FAIL_EQUIV` | **FAIL del benchmark** |
| 0 | Cualquier drift | `EXPECTED_DRIFT` | Registrar métricas, continuar |

### Veredicto Global

```
SIEMPRE FAIL si:
  - Cualquier run con kv_aligned=1 tiene verdict FAIL_EQUIV

PASS si:
  - TODOS los runs con kv_aligned=1 tienen verdict PASS_EQUIV

EXPECTED_DRIFT_ONLY si:
  - NO hay runs con kv_aligned=1 (solo drift esperado)
```

## Completeness Guardrail

### Configuración con config.json

El benchmark soporta un archivo `config.json` en el directorio raíz de artifacts para declarar la matriz esperada:

```json
{
  "kv_aligned": [0, 1],
  "seeds": [0, 1, 2],
  "dtype": "bf16",
  "prompt_len": 512,
  "gen_len": 128
}
```

### Veredictos de Completitud

| Condición | Veredicto Global | Descripción |
|-----------|------------------|-------------|
| Matriz completa + PASS | `PASS_GUARDRAIL` | Todos los pares ejecutados y pasaron umbrales |
| Matriz completa + FAIL | `FAIL_GUARDRAIL` | Al menos un par con kv_aligned=1 falló |
| Matriz incompleta | `INCOMPLETE` | Faltan pares declarados en config.json |
| Sin config.json | `PASS_GUARDRAIL_LOCAL` | Modo best-effort (smoke test local) |

### Nota sobre PASS_GUARDRAIL_LOCAL

- **Qué significa**: Smoke test local exitoso, pero sin verificación de completitud de matriz
- **Requiere**: Ejecución completa en MI300X con config.json para veredicto final
- **No es un PASS definitivo** hasta que se ejecute la matriz completa en hardware de producción

## Artifact Structure

```
artifacts_remote/YYYY-MM-DD/b3_67/
├── config.json                    # Opcional: matriz esperada
├── runs/
│   ├── kv_aligned_0/
│   │   ├── seed_0/
│   │   │   ├── prefill/
│   │   │   │   ├── logits.jsonl.gz
│   │   │   │   └── metadata.json
│   │   │   └── decode/
│   │   │       ├── logits.jsonl.gz
│   │   │       └── metadata.json
│   │   ├── seed_1/
│   │   └── seed_2/
│   └── kv_aligned_1/
│       ├── seed_0/
│       ├── seed_1/
│       └── seed_2/
├── metrics/
│   ├── kv_aligned_0/
│   │   ├── seed_0_metrics.json
│   │   ├── seed_1_metrics.json
│   │   └── seed_2_metrics.json
│   └── kv_aligned_1/
│       ├── seed_0_metrics.json
│       ├── seed_1_metrics.json
│       └── seed_2_metrics.json
├── summary.json
└── B3_67_EQUIVALENCE_GUARDRAIL.md
```

### Formato de Logits

```json
{"token_idx": 0, "token_id": 101, "logits": [0.1, -0.2, 0.3, ...]}
{"token_idx": 1, "token_id": 102, "logits": [0.05, 0.15, -0.1, ...]}
```

### Formato de Metadata

```json
{
  "dtype": "bf16",
  "prompt_len": 512,
  "gen_len": 128,
  "seed": 0,
  "kv_aligned": 1,
  "mode": "prefill",
  "timestamp": "2026-02-07T20:00:00Z",
  "git_commit": "abc123def"
}
```

## Implementación

### Archivos del Benchmark

| Archivo | Descripción |
|---------|-------------|
| `tools/benchmarks/run_b3_67_equivalence_guardrail.sh` | Runner shell |
| `tools/benchmarks/analyze_b3_67_equivalence_guardrail.py` | Analyzer Python |

### Uso del Runner

```bash
# Ejecución completa (todos los valores de kv_aligned)
./tools/benchmarks/run_b3_67_equivalence_guardrail.sh <NODE_IP>

# Solo kv_aligned=1
./tools/benchmarks/run_b3_67_equivalence_guardrail.sh <NODE_IP> --kv_aligned 1

# Con seeds específicos
./tools/benchmarks/run_b3_67_equivalence_guardrail.sh <NODE_IP> --seeds "0,1"
```

### Uso del Analyzer

```bash
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir artifacts_remote/YYYY-MM-DD/b3_67/runs \
    --output artifacts_remote/YYYY-MM-DD/b3_67/B3_67_EQUIVALENCE_GUARDRAIL.md
```

## Flags de Determinismo

Para minimizar flakiness, el runner configura:

```bash
export HIP_LAUNCH_BLOCKING=1        # Síncrono
export AMD_SERIALIZE_KERNEL=3       # Serializar kernels
export HSA_ENABLE_SDMA=0            # Disable SDMA
export GRETA_SEED=${SEED}           # Seed fixed
```

## Resultados Esperados

### kv_aligned=1

| Métrica | Valor Esperado |
|---------|---------------|
| `max_abs_diff` | ≤ 5e-3 |
| `p99_abs_diff` | ≤ 1e-3 |
| `top1_agreement` | ≥ 0.999 |
| **Veredicto** | `PASS_EQUIV` |

### kv_aligned=0

| Métrica | Valor Esperado |
|---------|---------------|
| Drift | Permitido |
| **Veredicto** | `EXPECTED_DRIFT` |

## Interpretación de Resultados

### PASS_GUARDRAIL

- Todos los runs con kv_aligned=1 pasaron equivalencia
- Runs con kv_aligned=0 muestran drift esperado
- **Acción**: Continuar desarrollo

### FAIL_GUARDRAIL

- Al menos un run con kv_aligned=1 falló equivalencia
- **Acción**: Investigar regresión

### EXPECTED_DRIFT_ONLY

- Solo se ejecutaron runs con kv_aligned=0
- Drift documentado y esperado
- **Acción**: Ejecutar con kv_aligned=1 para verificación completa

## Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Variabilidad en seeds | Media | Métricas varían | Seeds fijos + promediar |
| Precision BF16 vs FP32 | Baja | Errores numéricos | Normalizar a float32 |
| Non-determinismo GPU | Baja | Resultados diferentes | Flags de determinismo |
| Paring incorrecto | Media | Comparar tokens wrong | Verificar token_id |
| Archivo corrupto | Baja | Falla de análisis | Validar schema JSON |

## Current Status

### Local Smoke Test
- **Completeness**: ✓ PASS (12/12 pairs)
- **Equivalence kv_aligned=0**: EXPECTED_DRIFT (correct)
- **Equivalence kv_aligned=1**: FAIL_EQUIV (synthetic data noise > threshold)

### Token Span Alignment (Critical for MI300X)
For equivalence comparison, both prefill and decode must dump the **same** token range:
- `token_span: {"start": prompt_len, "count": 1}` (first generated token only)
- Span mismatch → `SPAN_MISMATCH` error in pairing_errors → FAIL_GUARDRAIL

> [!IMPORTANT]
> **1-Step Equivalence Guardrail**
> 
> This guardrail compares only the **first generated token** (position `prompt_len`).
> This is intentional to ensure robust, unambiguous comparison between prefill and decode.
> Full-span comparison (gen_len > 1) may be added in a future ticket (B3.69).

---

## MI300X Runbook

### Runner Script (v2 - uses B3.68 --dump-logits)

The B3.67 runner now uses `greta_infer --dump-logits` (B3.68 feature) instead of the legacy `GRETA_TRACE_B3_66` environment variable.

```bash
# Full matrix run (12 runs: kv ∈ {0,1} × seeds ∈ {0,1,2} × modes ∈ {prefill,decode})
./tools/benchmarks/run_b3_67_equivalence_guardrail.sh <MI300X_IP>

# Custom run with specific parameters
./tools/benchmarks/run_b3_67_equivalence_guardrail.sh <MI300X_IP> 2026-02-07 --kv_aligned 1 --seeds "0"
```

### Output Structure

```
artifacts_remote/<DATE>/b3_67/
├── config.json              # Matrix definition (enables completeness guardrail)
├── runs/
│   ├── kv_aligned_0/
│   │   ├── seed_0/
│   │   │   ├── prefill/
│   │   │   │   ├── metadata.json
│   │   │   │   └── logits.jsonl.gz
│   │   │   └── decode/
│   │   │       ├── metadata.json
│   │   │       └── logits.jsonl.gz
│   │   └── seed_1/...
│   └── kv_aligned_1/...
└── B3_67_EQUIVALENCE_GUARDRAIL.md
```

### Manual Run (Single Configuration)

```bash
ssh root@<MI300X_IP> "
  cd /root/gretacore
  mkdir -p /tmp/logits_test

  ./tools/inference/build/greta_infer \\
    --model ./models/greta-v1.gguf \\
    --prompt tools/benchmarks/prompts/p0_short.txt \\
    --max-tokens 1 \\
    --seed 42 \\
    --kv-aligned 1 \\
    --mode prefill \\
    --dump-logits /tmp/logits_test \\
    --greedy

  cat /tmp/logits_test/metadata.json
"
```

### Expected Verdicts

| kv_aligned | Expected Verdict | Rationale |
|------------|-----------------|-----------|
| 0 | EXPECTED_DRIFT | Without KV alignment, prefill vs decode produces different hidden states |
| 1 | PASS_EQUIV | With KV alignment, hidden states should be numerically equivalent |

### Key Changes from v1

| v1 (B3.66 format) | v2 (B3.68 format) |
|-------------------|-------------------|
| `GRETA_TRACE_B3_66=1` | `--dump-logits <DIR>` |
| Hidden states only | Full logits + metadata |
| No token_span | `token_span: {"start": prompt_len, "count": 1}` |
| Per-run config.json | Root config.json + per-run metadata.json |

---

## Changelog

| Fecha | Autor | Cambio |
|-------|-------|--------|
| 2026-02-07 | L.E.T | Creación inicial |
| 2026-02-07 | L.E.T | Add completeness guardrail, summary.json, pairing validation |
| 2026-02-08 | L.E.T | v2: Runner uses B3.68 --dump-logits, emits root config.json |

## Referencias

- B3.66: Prefill vs Decode Drift Probe
- B3.68: greta_infer kv_aligned + logits dump
- docs/PROGRESS.md: Estado del benchmark
