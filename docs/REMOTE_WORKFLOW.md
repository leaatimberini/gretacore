# Remote Workflow for B3.89 Benchmark

## SSH Connection
To connect to the remote MI300X node:
```bash
ssh <hostname>
# Or if configured in ~/.ssh/config:
ssh greta-runpod
```

## Running B3.89 Benchmark
Navigate to the repository root:
```bash
cd /root/gretacore
```

Execute the runner script:
```bash
./scripts/b3_89_runner.sh
```
Ensure that `PERF_TIMING=1` is set in the environment or the script output confirms it to capture performance metrics.

## Verifying PERF_TIMING
Check the run output for:
```
PERF_TIMING: ON
```
Or inspect generated `perf.json` files in `artifacts_remote/`.

## Safe Commit & Push
The `artifacts_remote/` directory is git-ignored to prevent accidental commits of large artifacts.
To commit changes:
1. Stage only intended files (source, docs, scripts).
2. Use `git status` to verify no artifacts are being added.
3. Commit and push:
```bash
git add <files>
git commit -m "feat: description"
git push origin main
```
