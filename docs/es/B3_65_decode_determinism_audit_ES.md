# B3.65: Auditoría de Determinismo del Decode

**Fecha**: 2026-02-07
**Estado**: READY_TO_RUN
**Objetivo**: Verificar que la salida del decode sea determinista y bit-estable

## Metodología

- Mismo prompt, misma seed, mismo binario
- 10 ejecuciones consecutivas
- Comparar hash de logits, token top-1, MAE

## Variables de Entorno

| Variable | Valor | Propósito |
|----------|-------|-----------|
| `GRETA_D2H_DEBUG` | 1 | Habilitar output de debug de logits |
| `HIP_LAUNCH_BLOCKING` | 1 | Lanzamiento síncrono de kernels |
| `AMD_SERIALIZE_KERNEL` | 3 | Serialización de kernels para determinismo |
| `HSA_ENABLE_SDMA` | 0 | Deshabilitar SDMA para timing consistente |

## Métricas

- `logits_hash64`: Hash SHA256 de la salida completa de logits
- `top1_token`: ID del token más probable
- `tokens/sec`: Rendimiento de generación
- `MAE`: Error Absoluto Medio entre ejecuciones (si hay datos numéricos)

## Códigos de Veredicto

| Código | Descripción |
|--------|-------------|
| `PASS_DETERMINISTIC` | Bit-idéntico en todas las ejecuciones |
| `NUMERICAL_JITTER` | MAE < 1e-7, aceptable para FP32 |
| `NON_DETERMINISTIC` | Debe explicar la fuente de no-determinismo |

## Uso

```bash
# Ejecutar en MI300X
./tools/benchmarks/run_b3_65_determinism_mi300x.sh 129.212.184.200

# Analizar resultados
python3 tools/benchmarks/analyze_b3_65_determinism.py --dir artifacts_remote/2026-02-07/b3_65
```

## Estructura de Salida Esperada

```
artifacts_remote/2026-02-07/b3_65/
├── run/
│   ├── run_01.txt
│   ├── run_02.txt
│   ├── ...
│   ├── run_10.txt
│   └── summary.tsv
└── b3_65_analysis.txt
```

## Evidencia

- `artifacts_remote/2026-02-07/b3_65/`
- Binario: `tools/inference/greta_infer_fixed`
- Modelo: `models/greta-v1.gguf` (Llama-2-7B)

## Restricciones

- ❌ Sin cambios en lógica de atención
- ❌ Sin refactorización de kernels
- ❌ Sin regresiones de performance
- ❌ Sin reabrir fases anteriores

---

**Firmado: L.E.T / Leandro Emanuel Timberini**
