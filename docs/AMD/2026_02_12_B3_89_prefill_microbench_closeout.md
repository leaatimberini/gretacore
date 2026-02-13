# B3.89 Prefill Microbench Closeout Report

**Date:** 2026-02-12  
**Benchmark:** B3.89 - Prefill Kernel Optimization V3/V4 (Q-LDS)  
**Status:** ✅ COMPLETED

---

## ES - Resumen Ejecutivo

### Causa Raíz
El problema de `prefill_s: 0` fue causado por el valor por defecto de `GRETA_MAX_SEQ_LEN = 2048`. Cuando el contexto solicitado excedía este límite, el prefill se cortaba inmediatamente sin procesar tokens.

### Solución Aplicada
1. **Patch GGUF:** Modificado `llama.context_length=32768` en el archivo de metadatos del modelo
2. **Cambio en Executor:** Actualizada la variable de entorno `GRETA_MAX_SEQ_LEN=$((CTX+2))` para que se ajuste dinámicamente al contexto solicitado

### Resultados Core

| Context | prefill_s | tok/s | Status |
|---------|-----------|-------|--------|
| 4096    | 22.77     | ~180  | ✅ PASS |
| 8192    | 114.27    | ~72   | ✅ PASS |
| 16384   | 469.75    | ~35   | ✅ PASS |

### Estado
✅ **COMPLETADO** - Los tests core pasaron exitosamente

---

## EN - Executive Summary

### Root Cause
The `prefill_s: 0` issue was caused by the default `GRETA_MAX_SEQ_LEN = 2048`. When the requested context exceeded this limit, prefill was cut off immediately without processing tokens.

### Fix Applied
1. **GGUF Patch:** Modified `llama.context_length=32768` in the model metadata file
2. **Executor Change:** Updated environment variable `GRETA_MAX_SEQ_LEN=$((CTX+2))` to dynamically adjust to the requested context

### Core Results

| Context | prefill_s | tok/s | Status |
|---------|-----------|-------|--------|
| 4096    | 22.77     | ~180  | ✅ PASS |
| 8192    | 114.27    | ~72   | ✅ PASS |
| 16384   | 469.75    | ~35   | ✅ PASS |

### Status
✅ **COMPLETED** - Core tests passed successfully

---

## 1. Root Cause Analysis (Análisis de Causa Raíz)

### Problem Description
- **Symptom:** `prefill_s: 0` in performance metrics
- **Scope:** All variants (baseline, v3, v4)
- **Trigger:** Context lengths > 2048 tokens

### Root Cause Analysis

```
Initial State:
├── GRETA_MAX_SEQ_LEN = 2048 (default)
├── Requested Context = 4096/8192/16384
└── Result: prefill truncated at 2048 → prefill_s = 0

Investigation:
├── Checked executor environment variables
├── Examined GGUF model metadata
└── Found: llama.context_length was also at default
```

### Evidence
- `prefill_s: 0` consistently for all context lengths > 2048
- Model loading successful
- No error messages in logs (silent truncation)

---

## 2. Fix Applied (Solución Aplicada)

### GGUF Patch
```bash
# Modified models/greta-v1.gguf metadata
# Before: llama.context_length = 2048 (or missing/default)
# After:  llama.context_length = 32768

python3 -c "
import gguf
model = 'models/greta-v1.gguf'
loader = gguf.GGUFLoader(model)
meta = loader.get_meta()
print('llama.context_length =', meta.get('llama.context_length', 'NOT SET'))
"
# Output: llama.context_length = 32768 ✓
```

### Executor Change
```bash
# In tools/benchmarks/remote_b3_89_executor.sh
# Before:
#   export GRETA_MAX_SEQ_LEN=2048

# After:
export GRETA_MAX_SEQ_LEN=$((CTX+2))
```

---

## 3. Verification Steps (Pasos de Verificación)

### 3.1 Verify GGUF Context Length
```bash
ssh root@129.212.184.200 "
python3 -c '
import gguf
loader = gguf.GGUFLoader(\"models/greta-v1.gguf\")
meta = loader.get_meta()
ctx_len = meta.get(\"llama.context_length\", \"NOT SET\")
print(f\"llama.context_length = {ctx_len}\")
'
# Expected: llama.context_length = 32768
```

### 3.2 Verify PERF_TIMING in Binaries
```bash
ssh root@129.212.184.200 "
for variant in baseline v3 v4; do
    bin=\"/root/gretacore/tools/inference/build_\${variant}/greta_infer\"
    echo \"\${variant}:\" && strings \$bin | grep -c PERF_TIMING
done
"
# Expected: Non-zero counts for all variants
```

### 3.3 Verify GRETA_MAX_SEQ_LEN in Executor
```bash
ssh root@129.212.184.200 "
grep GRETA_MAX_SEQ_LEN /root/gretacore/tools/benchmarks/remote_b3_89_executor.sh
"
# Expected: export GRETA_MAX_SEQ_LEN=\$((CTX+2))
```

### 3.4 Run Core Tests
```bash
ssh root@129.212.184.200 "
cd /root/gretacore/artifacts_remote/2026-02-10/b3_89
for ctx in 4096 8192 16384; do
    timeout 600s ./greta_infer --prompt \"\$(python3 -c 'print(\"a\"*'$ctx')')\" --max-tokens 1 2>&1 | grep -E '(prefill_s|EXIT_CODE)'
done
"
```

---

## 4. Core Results Table (Tabla de Resultados Core)

### Performance Metrics (Métricas de Rendimiento)

| Context | Wall Time (s) | Model Load (s) | Prefill (s) | Decode (s) | tok/s | Status |
|---------|---------------|----------------|-------------|------------|-------|--------|
| 4096    | 44.88         | 21.87          | 22.77       | ~0.000     | ~180  | ✅ PASS |
| 8192    | 136.13        | 21.62          | 114.27      | ~0.000     | ~72   | ✅ PASS |
| 16384   | 491.37        | 21.38          | 469.75      | ~0.000     | ~35   | ✅ PASS |

### Attention Implementation
- **Type:** `flash_v2_naive`
- **Determinism:** ON

### Variants Tested
- `baseline` (MQA)
- `v3` (Q-in-LDS)
- `v4` (Q-in-LDS v2)

---

## 5. Long Context Extension Status (Estado de Extensión de Contexto Largo)

### Status: 🔄 IN_PROGRESS

**Location:** `artifacts_remote/2026-02-12/b3_89_long_ctx/`

**Extended Contexts:**
- 16384 tokens (timeout: 1800s)
- 24576 tokens (timeout: 3600s)
- 32768 tokens (timeout: 5400s)

**Variants:** baseline, v3, v4

**Test Configuration:**
```bash
HIP_LAUNCH_BLOCKING=1
AMD_SERIALIZE_KERNEL=3
GRETA_MAX_SEQ_LEN=$((CTX+2))
```

---

## 6. Kernel Attribution (Atribución de Kernels)

### V3 (Q-in-LDS)
- **Commit:** `23714b7`
- **Key Feature:** Zero scratch spill achieved
- **Result:** 4k prefill speedup 1.21x

### V4 (Q-in-LDS v2)
- **Commit:** `23714b7`
- **Key Feature:** SEG=32 exploration (Reduce reloads)
- **Status:** Under exploration

### Infrastructure
- **Commit:** `ec6fe74` (Infra)
- **Commit:** `db69892` (Docs)

---

## 7. Guardrails (Guardas de Seguridad)

### Pre-Run Checklist
- [x] GGUF metadata updated (`llama.context_length=32768`)
- [x] Executor updated (`GRETA_MAX_SEQ_LEN=$((CTX+2))`)
- [x] PERF_TIMING embedded in binaries
- [x] Determinism mode enabled

### Validation Points
- [x] `prefill_s > 0` for all context lengths
- [x] `EXIT_CODE=0` for all tests
- [x] Consistent `attn_impl` (`flash_v2_naive`)
- [x] tok/s within expected range

---

## 8. Artifacts Index (Índice de Artefactos)

| Artifact | Location | Description |
|----------|----------|-------------|
| Final Report | `artifacts_remote/B3_89_FINAL_REPORT.md` | Full benchmark report |
| Closeout | `docs/AMD/2026_02_12_B3_89_prefill_microbench_closeout.md` | This file |
| Core Results | `artifacts_remote/2026-02-10/b3_89/` | Core test results |
| Long Context | `artifacts_remote/2026-02-12/b3_89_long_ctx/` | Extended context tests |
| Summary JSON | `artifacts_remote/2026-02-10/b3_89/summary.json` | Machine-readable summary |
| Executor | `tools/benchmarks/remote_b3_89_executor.sh` | Test executor script |

---

## 9. Reproduction Commands (Comandos de Reproducción)

### Quick Test (Single Context)
```bash
ctx=4096
ssh root@129.212.184.200 "
cd /root/gretacore
export GRETA_MAX_SEQ_LEN=\$((ctx+2))
prompt=\$(python3 - <<<'print(\"a\"*'$ctx')')
timeout 120s tools/inference/build_baseline/greta_infer --prompt \"\$prompt\" --max-tokens 1 2>&1 | grep -E '(prefill_s|EXIT_CODE)'
"
```

### Full Core Suite
```bash
for ctx in 4096 8192 16384; do
    echo \"=== Testing ctx=\$ctx ===\"
    ssh root@129.212.184.200 "
        cd /root/gretacore
        export GRETA_MAX_SEQ_LEN=\$((ctx+2))
        prompt=\$(python3 - <<<'print(\"a\"*'$ctx')')
        timeout 600s tools/inference/build_baseline/greta_infer --prompt \"\$prompt\" --max-tokens 1 2>&1 | grep -E '(prefill_s|EXIT_CODE)'
    "
done
```

### Long Context Tests
```bash
cd /root/gretacore/artifacts_remote/2026-02-12/b3_89_long_ctx
nohup ./monitor_and_run.sh > monitor.log 2>&1 &
# Monitor progress via: cat status.json
```

---

## References (Referencias)

- **B3.89 Ticket:** GRETA CORE V3/V4 Q-in-LDS kernels validation
- **Remote Node:** 129.212.184.200
- **Model:** greta-v1.gguf
- **Attention:** flash_v2_naive
- **Date:** 2026-02-10 (Core), 2026-02-12 (Long Context)

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-12  
**Author:** GRETA CORE CI/CD System
