# B3.82–B3.84 — Steady-State & Long-Context Decode (MI300X)

## Objective
Validate steady-state decode performance and numerical stability across extreme context lengths and batch sizes on AMD MI300X. This suite focuses on long generation lengths (up to 1024 tokens) where correctness is verified via token ID stream agreement instead of full logits comparison to avoid IO bottlenecks.

## Methodology
- **Benchmarks:**
  - **B3.82 (Steady-State Batch Scaling):** 8k context, 1024 tokens generation, batches 1, 2, 4, 8.
  - **B3.83 (Extreme Long-Context):** 32k context, 512 tokens generation.
  - **B3.84 (High-Pressure Batch):** 16k context, batch 8, 256 tokens generation.
- **Correctness Gate:**
  - Token-by-token agreement of the generated stream between prefill and decode phases.
  - Required agreement: 100%.
- **Telemetry:**
  - VRAM sampling (1s via `rocm-smi`).
  - Tokens per second (TPS).
- **Optimizer:** `dump_span=0` to disable full logits dumping for long runs.

## Results (2026-02-09)

[To be populated after execution]

## Interpretation

[To be populated after execution]

## Execution Commands

### Runner
```bash
./tools/benchmarks/run_b3_82_to_b3_84_decode_steady_state_suite.sh --node 129.212.184.200
```

### Analyzer
```bash
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
  --mode b3_82_84 \
  --traces-dir artifacts_remote/2026-02-09/b3_82_84/runs \
  --output artifacts_remote/2026-02-09/b3_82_84/report.md
```
