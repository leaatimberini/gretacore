# B3.68 greta_infer KV Aligned + Logits Dump

## Metadata

| Campo | Valor |
|-------|-------|
| **Ticket** | B3.68 |
| **Date** | 2026-02-07 |
| **Author** | L.E.T |
| **Status** | IMPLEMENTED |
| **Labels** | Repo branch: main |
| **Dependency** | Required by B3.67 for MI300X execution |

## Resumen Ejecutivo

Este ticket implementa las flags necesarias en `greta_infer` para soportar el guardrail de equivalencia B3.67 en ejecución MI300X:

- `--kv-aligned {0|1}`: Control de alineación KV
- `--mode prefill|decode`: Modo de ejecución
- `--dump-logits <dir>`: Dump de logits para análisis
- `--seed <n>`: Seed para reproducibilidad

## Flags Implementados

### --kv-aligned {0|1}
Fuerza el modo de alineación de KV cache:
- `0`: Sin alineación (comportamiento default)
- `1`: Con alineación (esperado: equivalencia prefill == decode)
- También lee de `GRETA_KV_ALIGNED` env var

### --mode prefill|decode
Modo de ejecución para tracing:
- `prefill`: Solo fase de prefill
- `decode`: Solo fase de decode

### --dump-logits <dir>
Directorio de salida para dump de logits:
- Crea `metadata.json` con configuración
- Crea `logits.jsonl.gz` con logits dumpeados

### --seed <n>
Seed para reproducibilidad:
- También lee de `GRETA_SEED` env var

## Contrato de Salida

### metadata.json
```json
{
  "dtype": "bf16",
  "prompt_len": 512,
  "gen_len": 128,
  "seed": 42,
  "kv_aligned": 1,
  "mode": "decode",
  "token_span": {"start": 512, "count": 1},
  "timestamp": "2026-02-07T20:00:00Z",
  "repo_branch": "main"
}
```

> [!IMPORTANT]
> **Token Span Alignment for B3.67**
> 
> Both `prefill` and `decode` modes MUST dump the **same** token_span for B3.67 comparison:
> - `token_span: {"start": prompt_len, "count": 1}` (first generated token)
> 
> This ensures the analyzer can compare identical token_idx/token_id/logits between modes.

### logits.jsonl.gz
Formato JSONL comprimido con gzip:
```json
{"token_idx": 0, "token_id": 101, "logits": [0.1, -0.2, 0.3, ...]}
{"token_idx": 1, "token_id": 102, "logits": [0.05, 0.15, -0.1, ...]}
```

## Uso

### Línea de comando
```bash
./tools/inference/build/greta_infer \
    --model ./models/greta-v1.gguf \
    --prompt "Hello, world" \
    --max-tokens 10 \
    --kv-aligned 1 \
    --mode decode \
    --seed 42 \
    --dump-logits /tmp/logits_output
```

### Con variables de entorno
```bash
export GRETA_SEED=42
export GRETA_KV_ALIGNED=1
./tools/inference/build/greta_infer \
    --model ./models/greta-v1.gguf \
    --dump-logits /tmp/logits_output
```

## Cómo lo Usa B3.67

El runner de B3.67 (`run_b3_67_equivalence_guardrail.sh`) usa estas flags para:
1. Ejecutar greta_infer con `--kv-aligned=1 --mode=prefill --dump-logits`
2. Ejecutar greta_infer con `--kv-aligned=1 --mode=decode --dump-logits`
3. Comparar logits entre prefill y decode
4. Verificar equivalencia con el analyzer

## Smoke Test

```bash
./tools/benchmarks/test_b3_68_smoke.sh
```

Valida:
- `metadata.json` existe y es JSON válido
- `logits.jsonl.gz` existe y es gzip válido

## Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `tools/inference/src/greta_infer.cpp` | Agregar flags y dump logic |
| `tools/benchmarks/test_b3_68_smoke.sh` | Smoke test script |

## Changelog

| Fecha | Autor | Cambio |
|-------|-------|--------|
| 2026-02-07 | L.E.T | Implementación inicial |

## Referencias

- B3.67: Equivalence Guardrail
- docs/PROGRESS.md: Estado del benchmark
