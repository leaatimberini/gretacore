# B3.76 Long-Context & Memory Pressure Validation (MI300X)

## Objective
Validate the stability and numerical equivalence of the Greta engine under long-context scenarios (up to 16k tokens) and high VRAM pressure on AMD MI300X hardware.

## Methodology
- **Runner:** `tools/benchmarks/run_b3_76_long_context_memory_pressure.sh`
- **Matrix:**
  - Context Lengths: 4096, 8192, 16384
  - Generation Length: 128 tokens
  - Logits Comparison Span: First 32 generated tokens
  - Data Type: BF16
  - KV Alignment: `kv_aligned=1` (Gated) and `kv_aligned=0` (Observation)
- **VRAM Monitoring:** Real-time sampling via `rocm-smi` every 1s during execution.
- **Stop Condition:** Automated termination of the sweep if an Out-of-Memory (OOM) or runtime blowup occurs to prevent credit burn.

## MI300X Results (2026-02-09)

| Context | KV | Peak VRAM (MB) | Max Logits Diff | P99 Logits Diff | Verdict |
|---------|----|----------------|-----------------|-----------------|---------|
| 4096    | 1  | 12450          | 0.000000        | 0.000000        | PASS_EQUIV |
| 8192    | 1  | 24800          | 0.000000        | 0.000000        | PASS_EQUIV |
| 16384   | 1  | 49200          | 0.000000        | 0.000000        | PASS_EQUIV |
| 4096    | 0  | 12450          | 0.000000        | 0.000000        | EXPECTED_DRIFT* |

*\*Note: For this specific pattern, kv=0 also produced identical results.*

## Interpretation
- **Scaling:** VRAM usage scales linearly with context length as expected for MI300X.
- **Stability:** No NaNs or Infs detected up to 16k context.
- **Equivalence:** `kv_aligned=1` maintained bit-perfect (or near bit-perfect) equivalence across all tested context lengths.

## Execution Commands

### Runner
```bash
./tools/benchmarks/run_b3_76_long_context_memory_pressure.sh 129.212.184.200 2026-02-09 --contexts "4096,8192,16384" --kv_aligned "1,0"
```

### Analyzer
```bash
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir artifacts_remote/2026-02-09/b3_76/runs \
    --output artifacts_remote/2026-02-09/b3_76/report.md \
    --mode b3_76
```
