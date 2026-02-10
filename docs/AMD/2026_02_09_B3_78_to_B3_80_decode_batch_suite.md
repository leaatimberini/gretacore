# B3.78–B3.80 — Long-Context Decode & Batch Suite (MI300X)

## Objectives
Validate extreme long-context stability, batch size scalability, and micro-soak (flakiness) on AMD MI300X hardware.

### B3.78: kv_aligned Control (32k)
Test if `kv_aligned=0` introduces measurable drift in logits at 32k context on Llama-3-8B.
Expected signal: `kv=1` bit-perfect; `kv=0` expected drift (or bit-perfect if no race occurs).

### B3.79: Batch Size Probe (8k/16k)
Validate memory and throughput deltas when increasing batch size from 1 to 2.
Expected signal: VRAM increase, throughput increase (tokens/s), stability check.

### B3.80: Micro-Soak Repetition (16k)
Run 16k context 5 times with `kv_aligned=1` to detect non-determinism, NaN, or random OOMs.
Expected signal: Identical logits across all 5 repeats.

## Methodology
- **Suite Runner:** `tools/benchmarks/run_b3_78_to_b3_80_long_context_decode_batch_suite.sh`
- **Execution:** Single-shot remote executor script.
- **VRAM Monitoring:** `rocm-smi` sampling at 1s.
- **Timeouts:** 600s per phase (prefill/decode).

## Results (2026-02-09)

[To be populated after execution]

## Interpretation

[To be populated after execution]

## Execution Commands

### Runner
```bash
./tools/benchmarks/run_b3_78_to_b3_80_long_context_decode_batch_suite.sh 129.212.184.200 2026-02-09
```

### Analyzer
```bash
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
  --mode b3_78_80 \
  --traces-dir artifacts_remote/2026-02-09/b3_78_80/runs \
  --output artifacts_remote/2026-02-09/b3_78_80/report.md
```
