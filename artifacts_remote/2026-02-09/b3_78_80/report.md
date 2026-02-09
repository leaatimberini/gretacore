# B3.78-80 Suite Analysis Report

**Global Verdict:** PASS

## B3_78 Results

| Context | Batch | KV | Repeat | Peak VRAM | Prefill (s) | Decode (s) | Max Diff | Top1 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 32768 | 1 | 0 | 0 | 17095 | 35.41 | 35.11 | 0.000000 | 1.0000 | EXPECTED_DRIFT |
| 32768 | 1 | 1 | 0 | 17095 | 35.57 | 35.48 | 0.000000 | 1.0000 | PASS_EQUIV |

## B3_79 Results

| Context | Batch | KV | Repeat | Peak VRAM | Prefill (s) | Decode (s) | Max Diff | Top1 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 8192 | 1 | 1 | 0 | 17095 | 35.54 | 35.27 | 0.000000 | 1.0000 | PASS_EQUIV |
| 8192 | 2 | 1 | 0 | 18720 | 35.66 | 35.41 | 0.000000 | 1.0000 | PASS_EQUIV |
| 16384 | 1 | 1 | 0 | 20958 | 35.78 | 35.83 | 0.000000 | 1.0000 | PASS_EQUIV |
| 16384 | 2 | 1 | 0 | 17741 | 35.51 | 35.07 | 0.000000 | 1.0000 | PASS_EQUIV |

## B3_80 Results

| Context | Batch | KV | Repeat | Peak VRAM | Prefill (s) | Decode (s) | Max Diff | Top1 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| 16384 | 1 | 1 | 0 | 17095 | 34.33 | 33.60 | 0.000000 | 1.0000 | PASS_EQUIV |
| 16384 | 1 | 1 | 1 | 17095 | 33.74 | 33.94 | 0.000000 | 1.0000 | PASS_EQUIV |
| 16384 | 1 | 1 | 2 | 17095 | 33.93 | 34.44 | 0.000000 | 1.0000 | PASS_EQUIV |
| 16384 | 1 | 1 | 3 | 17095 | 34.13 | 33.84 | 0.000000 | 1.0000 | PASS_EQUIV |
| 16384 | 1 | 1 | 4 | 17095 | 34.40 | 34.19 | 0.000000 | 1.0000 | PASS_EQUIV |

