# B3.81 Multi-Batch Throughput Scaling Report

**Global Verdict:** PASS

## Throughput & VRAM Scaling

| Batch | Peak VRAM (MB) | Prefill (s) | Decode (s) | Tokens/s | Speedup | VRAM Delta | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | 24250 | 34.88 | 34.53 | 7.28705 | 1.00x | +0 MB | PASS_EQUIV |
| 2 | 17741 | 34.50 | 34.29 | 7.22846 | 0.99x | -6509 MB | PASS_EQUIV |
| 4 | 19033 | 34.91 | 34.98 | 7.26163 | 1.00x | -5217 MB | PASS_EQUIV |
| 8 | 21617 | 35.09 | 35.44 | 7.22782 | 0.99x | -2633 MB | PASS_EQUIV |
