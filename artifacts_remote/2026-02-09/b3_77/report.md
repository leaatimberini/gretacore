# B3.77 Long-Context Report

**Global Verdict:** PASS

## Context Matrix & Performance

| Context | Gen | Span | KV | Peak VRAM (MB) | Prefill (s) | Decode (s) | Max Diff | Top1 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 32768 | 64 | 16 | 1 | 17095 | 36.14 | 35.68 | 0.000000 | 1.0000 | PASS_EQUIV |

## Timeout Policy & Timing

- **Prefill Timeout:** 600s
- **Decode Timeout:** 600s

## VRAM Sampling Method

- **Device:** AMD MI300X
- **Sampling Period:** 1s
- **Samples Count:** 66
- **Peak Offset:** 34s
- **Note:** 1s sampling; micro-spikes might not be captured
