# B3.76 Long-Context & Memory Pressure Validation (MI300X)

## Objective
Validate the stability and numerical equivalence of the Greta engine under long-context scenarios (up to 16k tokens) and high VRAM pressure on AMD MI300X hardware.

## Methodology
- **Runner:** `tools/benchmarks/run_b3_76_long_context_memory_pressure.sh` (Single-Shot implementation to minimize SSH overhead)
- **Matrix:**
  - Context Lengths: 4096, 8192, 16384
  - Generation Length: 128 tokens
  - Logits Comparison Span: First 32 generated tokens
  - Data Type: BF16
  - KV Alignment: `kv_aligned=1` (Gated)
- **VRAM Monitoring:** Real-time sampling via `rocm-smi` every 1s during execution on remote host.
- **Stop Condition:** Integrated OOM detection in the remote executor script.

## MI300X Results (2026-02-09)

| Context | KV | Peak VRAM (MB) | Max Logits Diff | P99 Logits Diff | Top-1 Agreement | Verdict |
|---------|----|----------------|-----------------|-----------------|-----------------|---------|
| 4096    | 1  | 19586          | 0.000000        | 0.000000        | 1.0000          | PASS_EQUIV |
| 8192    | 1  | 17656          | 0.000000        | 0.000000        | 1.0000          | PASS_EQUIV |
| 16384   | 1  | 17095          | 0.000000        | 0.000000        | 1.0000          | PASS_EQUIV |

## Interpretation
- **Stability:** The engine remained stable and bit-perfect up to 16k context on MI300X.
- **VRAM:** Peak VRAM was monitored throughout the prefill and decode phases. Synthetic prompt data ("hello repeated") showed stable memory utilization.
- **Numeric Equivalence:** Bit-perfect agreement (diff=0.0) between prefill-heavy and decode-only paths validates the KV-cache alignment logic even at extended context lengths.

## Execution Commands

### Runner (Single-Shot)
```bash
./tools/benchmarks/run_b3_76_long_context_memory_pressure.sh 129.212.184.200 2026-02-09
```

### Analyzer
```bash
python3 tools/benchmarks/analyze_b3_67_equivalence_guardrail.py \
    --traces-dir artifacts_remote/2026-02-09/b3_76/runs \
    --output artifacts_remote/2026-02-09/b3_76/report.md \
    --mode b3_76
```
