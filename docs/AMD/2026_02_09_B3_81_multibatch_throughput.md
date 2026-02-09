# B3.81 — Multi-Batch Throughput Scaling @ 8k (MI300X)

## Objective
Quantify throughput and VRAM scaling for decode under increasing batch sizes on AMD MI300X hardware. This benchmark validates the engine's efficiency and numerical stability in multi-batch scenarios.

## Methodology
- **Runner:** `tools/benchmarks/run_b3_81_multibatch_throughput.sh`
- **Matrix:**
  - Context Length: 8192
  - Generation Length: 64 tokens
  - Logits Comparison Span: First 8 generated tokens
  - Data Type: BF16
  - KV Alignment: `kv_aligned=1`
  - Batch Sizes: 1, 2, 4, 8
- **Stop Conditions:**
  - **OOM:** Immediate termination and record `FAIL_OOM`.
  - **Timeout:** 600s per phase.
  - **Unsupported Batch:** Record `SKIPPED_UNSUPPORTED_BATCH`.
- **VRAM Monitoring:** Real-time sampling via `rocm-smi` every 1s.

## Results (2026-02-09)

[To be populated after execution]

## Interpretation

[To be populated after execution]

## Execution Commands

### Runner
```bash
./tools/benchmarks/run_b3_81_multibatch_throughput.sh 129.212.184.200 2026-02-09 --batches "1,2,4,8"
```

### Analyzer
```bash
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
  --mode b3_81 \
  --traces-dir artifacts_remote/2026-02-09/b3_81/runs \
  --output artifacts_remote/2026-02-09/b3_81/report.md
```
