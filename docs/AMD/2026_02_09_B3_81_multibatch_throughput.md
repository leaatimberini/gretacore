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

### Throughput & VRAM Scaling
| Batch | Peak VRAM (MB) | Prefill (s) | Decode (s) | Tokens/s | Speedup | VRAM Delta | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 24250 | 34.88 | 34.53 | 7.29 | 1.00x | +0 MB | PASS_EQUIV |
| 2 | 17741 | 34.50 | 34.29 | 7.23 | 0.99x | -6509 MB | PASS_EQUIV |
| 4 | 19033 | 34.91 | 34.98 | 7.26 | 1.00x | -5217 MB | PASS_EQUIV |
| 8 | 21617 | 35.09 | 35.44 | 7.23 | 0.99x | -2633 MB | PASS_EQUIV |

## Interpretation
- **Numerical Stability:** Bit-perfect equivalence (`diff=0.0`) maintained across all tested batch sizes (1, 2, 4, 8).
- **Throughput Consistency:** Throughput remained nearly constant at ~7.2-7.3 tokens/s regardless of batch size. This indicates that the current engine implementation for MI300X may be processing batches with minimal parallel speedup or is limited by context-loading overhead in this single-point probe.
- **VRAM Scaling:** Observed a peak of ~24GB for batch=1, which unexpectedly dropped for higher batches. This may be due to 1s sampling quantization or transient memory behavior during the first run. For batches 2-8, VRAM scaled from ~17.7GB to ~21.6GB as expected.

## Execution Commands

### Runner
```bash
./tools/benchmarks/run_b3_81_multibatch_throughput.sh --node 129.212.184.200 --date 2026-02-09 --batches "1,2,4,8"
```

### Analyzer
```bash
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
  --mode b3_81 \
  --traces-dir artifacts_remote/2026-02-09/b3_81/runs \
  --output artifacts_remote/2026-02-09/b3_81/report.md
```
