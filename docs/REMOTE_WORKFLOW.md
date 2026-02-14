# Remote Workflow (MI300X) — GRETA CORE

This repository is operated remote-first on the MI300X node for reliability and reproducibility.

## 1) SSH Access

Key location (default):
- ~/.ssh/id_ed25519

Quick connectivity check:
  ssh -T git@github.com

Optional: SSH config convenience (~/.ssh/config):
  Host greta-mi300x
    HostName 129.212.184.200
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 6

Then connect:
  ssh greta-mi300x

## 2) Safe Git Workflow

  cd /root/gretacore
  git status
  git pull --ff-only origin main

Stage only intended files (docs/source/scripts). Never add large run artifacts.

  git add <explicit paths>
  git diff --staged
  git commit -m "<message>"
  git push origin main

## 3) B3.89 — Core Microbench Runner

Canonical runner:

```bash
B3_89_MODE=perf ./tools/benchmarks/remote_b3_89_executor.sh \
    2026-02-14 "baseline,v3" "4096,8192" "4096:2,8192:1" \
    2>&1 | tee /tmp/b3_89_remote.log
```

Environment variable `B3_89_MODE` controls kernel serialization:
- `perf` (default) — no serialization, fast runs for real measurements.
- `debug` — `HIP_LAUNCH_BLOCKING=1`, `AMD_SERIALIZE_KERNEL=3`, `HSA_ENABLE_SDMA=0`.

## 4) Verification

PERF_TIMING present in binaries:
  for v in baseline v3 v4; do
    echo "== $v =="
    strings tools/inference/build_${v}/greta_infer | grep -c PERF_TIMING
  done

Executor runtime seq-len (must be CTX+2):
  grep -n "GRETA_MAX_SEQ_LEN" tools/benchmarks/remote_b3_89_executor.sh

GGUF context length (should be 32768):
  python3 -c 'import gguf; r=gguf.GGUFReader("models/greta-v1.gguf"); print(r.fields["llama.context_length"].parts[-1][0])'

## 5) Monitoring

The executor emits **single-line JSON events** to stdout (and the tee'd log file)
for every significant lifecycle point. These are machine-parseable and always
contain `"event"`, `"ts"`, `"mode"`, and context fields.

### Live tail

```bash
tail -f /tmp/b3_89_remote.log
```

### Filter specific event types

```bash
# All heartbeats:
grep '"event":"HEARTBEAT"' /tmp/b3_89_remote.log | tail

# All completed tests:
grep '"event":"TEST_END"' /tmp/b3_89_remote.log

# Human-readable progress lines only:
grep '^PROGRESS:' /tmp/b3_89_remote.log | tail
```

### Compact summary table

```bash
# One-shot (after run completes):
python3 tools/benchmarks/parse_b3_89_events.py /tmp/b3_89_remote.log

# Live (refreshes every 2s, Ctrl-C to stop):
python3 tools/benchmarks/parse_b3_89_events.py /tmp/b3_89_remote.log --follow
```

### JSON event glossary

| Event         | When emitted | Key fields |
|---------------|-------------|------------|
| `SUITE_START` | Once at start | `total_tests`, `variants`, `contexts`, `repeat_map` |
| `TEST_START`  | Before each `greta_infer` invocation | `variant`, `ctx`, `run_idx`, `test_index`, `timeout_s` |
| `HEARTBEAT`   | Every ~60 s while `greta_infer` is running | `elapsed_s`, `est_pct`, `gpu_use`, `vram_used`, `eta_remaining_s`, `suite_eta_s`, `pid` |
| `TEST_END`    | After each run completes | `exit_code`, `exit_status`, `wall_s`, `prefill_s`, `attn_impl`, `model_load_s`, `decode_s` |
| `SUITE_END`   | Once when entire suite finishes | `completed`, `suite_wall_s` |

### Human-readable progress

Lines prefixed with `PROGRESS:` are single-line, never-wrapping progress bars:

```
PROGRESS: [################..............] 53% (8/15) DONE variant=v3 ctx=8192 run=1 exit=0 wall=1024.1s | ETA 17m
```

### ETA estimation

After the first successful run of a given (variant, ctx), the executor estimates
remaining time based on the median wall time of completed runs for that ctx.
The `eta_remaining_s` field in HEARTBEAT events shows per-ctx ETA; `suite_eta_s`
shows an estimate for the full suite.

### Diagnostics (Exit 137 / OOM)

If a run exits non-zero or `PERF_TIMING` is missing, a `diag.txt` is written to
the run directory with:
- `dmesg` filtered for OOM/cgroup/killed
- `free -h`
- `rocm-smi` snapshot
- Last 80 lines of `run.log`
