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

  bash tools/benchmarks/run_b3_89_prefill_microbench.sh 129.212.184.200 \
    --date 2026-02-12 \
    --variants "baseline,v3,v4" --single-shot \
    --contexts "4096,8192,16384" \
    --repeat "4096:2,8192:1,16384:1"

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
  python3 -c 'import gguf; l=gguf.GGUFLoader("models/greta-v1.gguf"); print(l.get_meta().get("llama.context_length"))'
