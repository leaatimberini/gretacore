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

  bash tools/benchmarks/remote_b3_89_executor.sh \
  --date 2026-02-12 \
  --variants "baseline,v3,v4" \
  --contexts "4096,8192,16384" \
  --repeat "4096:2,8192:1,16384:1" \
  --single-shot

If monitoring on the remote node:
  tail -f /tmp/b3_89_remote.log


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

## 5) Progress / Heartbeat

The remote executor (`remote_b3_89_executor.sh`) emits structured log lines so you
can monitor long-running benchmark suites without instrumenting the binary.

### Watching live progress

```bash
tail -f /tmp/b3_89_remote.log
```

Or filter for specific events:

```bash
grep HEARTBEAT /tmp/b3_89_remote.log | tail
grep 'SUITE '    /tmp/b3_89_remote.log | tail
```

### Log line glossary

| Prefix        | Meaning |
|---------------|---------|
| `START`       | A single test run is about to begin. Shows variant, ctx, run index and timeout budget. |
| `HEARTBEAT`   | Emitted every ~60 s while a test is running. Shows elapsed time and a snapshot of GPU utilization and VRAM usage via `rocm-smi` (falls back to `NA` if unavailable). |
| `PROGRESS`    | Accompanies each heartbeat. Shows estimated % completion of the **current test** based on elapsed time vs. the timeout budget for that context length. |
| `SUITE_START` | Printed once at the beginning with the total number of test runs in the suite. |
| `SUITE`       | Printed after every completed test run. Shows a progress bar with global suite completion (tests done / total) and the exit code of the last run. |

### Caveats

- The per-test `PROGRESS` percentage is an **estimate** based on the timeout budget
  (e.g. 3 h for ctx 4096). The real completion % would require instrumenting the binary
  (`greta_infer`), which is deliberately out of scope for this change.
- If `rocm-smi` is not installed or fails, GPU metrics gracefully fall back to `NA`.
