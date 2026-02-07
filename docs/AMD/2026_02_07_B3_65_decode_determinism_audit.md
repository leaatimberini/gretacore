# B3.65: Decode Determinism Audit

**Date**: 2026-02-07
**Status**: READY_TO_RUN
**Objective**: Verify decode output is deterministic and bit-stable

## Methodology

- Same prompt, same seed, same binary
- 10 consecutive runs
- Compare logits hash, top-1 token, MAE

## Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `GRETA_D2H_DEBUG` | 1 | Enable logits debug output |
| `HIP_LAUNCH_BLOCKING` | 1 | Synchronous kernel launches |
| `AMD_SERIALIZE_KERNEL` | 3 | Kernel serialization for determinism |
| `HSA_ENABLE_SDMA` | 0 | Disable SDMA for consistent timing |

## Metrics

- `logits_hash64`: SHA256 hash of full logits output
- `top1_token`: Most probable token ID
- `tokens/sec`: Generation throughput
- `MAE`: Mean Absolute Error between runs (if numeric data available)

## Verdict Codes

| Code | Description |
|------|-------------|
| `PASS_DETERMINISTIC` | Bit-identical across all runs |
| `NUMERICAL_JITTER` | MAE < 1e-7, acceptable for FP32 |
| `NON_DETERMINISTIC` | Must explain source of non-determinism |

## Usage

```bash
# Run on MI300X
./tools/benchmarks/run_b3_65_determinism_mi300x.sh 129.212.184.200

# Analyze results
python3 tools/benchmarks/analyze_b3_65_determinism.py --dir artifacts_remote/2026-02-07/b3_65
```

## Expected Output Structure

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

## Evidence

- `artifacts_remote/2026-02-07/b3_65/`
- Binary: `tools/inference/greta_infer_fixed`
- Model: `models/greta-v1.gguf` (Llama-2-7B)

## Restrictions

- ❌ No attention logic changes
- ❌ No kernel refactoring
- ❌ No performance regressions
- ❌ No re-opening previous phases

---

**Signed: L.E.T / Leandro Emanuel Timberini**
