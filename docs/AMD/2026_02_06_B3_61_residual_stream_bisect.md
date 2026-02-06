# B3.61 Residual Stream Bisect Analysis Report

**Date:** 2026-02-06  
**Phase:** GRETA Phase 3  
**Objective:** Identify the first divergent tensor in the residual stream causing inference collapse at extended context lengths

---

## 1. Executive Summary

B3.61 continues the investigation of long-context inference failure modes identified in prior phases. The primary objective is to deterministically identify the **FIRST_FAIL** tensor—the first point in the residual stream where behavior diverges between successful baseline runs and failing extended-context runs.

### Key Findings (Expected Post-Execution)
- The first failing tensor location will be classified into one of seven root cause categories
- Critical positions: **826** (16K context) and **1652** (32K context) require explicit validation
- Minimum layer coverage: L0, L1, L2, L4, L8 with expansion capability

---

## 2. Methodology

### 2.1 Residual Stream Trace Points

The residual stream is instrumented at the following stages in strict order:

| Stage | Description | Hash Validation |
|-------|-------------|-----------------|
| `embed_out` | Embedding output (B3.59 sanity check) | SHA256 + nz_count |
| `residual_pre_attn` | Residual state before attention | SHA256 + nz_count |
| `attn_in` | Attention mechanism inputs (Q/K/V) | SHA256 + nz_count |
| `q_pre_rope` | Query before RoPE application | SHA256 + nz_count |
| `k_pre_rope` | Key before RoPE application | SHA256 + nz_count |
| `q_post_rope` | Query after RoPE application | SHA256 + nz_count |
| `k_post_rope` | Key after RoPE application | SHA256 + nz_count |
| `attn_out` | Attention output after softmax | SHA256 + nz_count |
| `residual_post_attn` | Residual after attention addition | SHA256 + nz_count |
| `ffn_norm_in` | FFN input following layer norm | SHA256 + nz_count |
| `mlp_out` | Feed-forward network output | SHA256 + nz_count |
| `residual_post_mlp` | Final residual after MLP | SHA256 + nz_count |
| `logits` | Final output logits | SHA256 + top-5 tokens |

### 2.2 Trace Record Schema

Every trace record includes the following mandatory metadata:

```json
{
    "prompt_id": "p6_len_16",
    "token_id": 826,
    "pos_id": 826,
    "logical_tok_idx": 5,
    "phase": "prefill_last",
    "layer": 0,
    "tensor_name": "residual_pre_attn",
    "route": "decode_attention_main",
    "timestamp": "2026-02-06T15:14:13Z",
    "git_commit_hash": "aad424d...",
    "hash": "sha256:...",
    "nz_count": 4096,
    "top5_tokens": [15, 42, 100, 256, 1024]
}
```

### 2.3 First Failure Detection Algorithm

The detection algorithm operates on composite keys: `(prompt_id, token_id, pos_id, layer, tensor_name)`. For each matched pair:

1. Compare SHA256 hashes in TRACE_STAGES order
2. The first stage with non-matching hashes is designated `FIRST_FAIL`
3. Root cause is classified based on `FIRST_FAIL` stage location
4. MAE is computed as secondary metric when raw tensor data is available

---

## 3. Root Cause Classification

Each failure is bucketed into exactly one category:

| Category | Description | Stages |
|----------|-------------|--------|
| `ROUTING/SELECTION` | Buffer divergence before attention | `embed_out`, `residual_pre_attn` |
| `ATTN_KERNEL_INPUTS` | Divergence in Q/K/V projections | `attn_in`, `q_pre_rope`, `k_pre_rope` |
| `ATTENTION_MECHANISM` | Divergence within attention computation | `q_post_rope`, `k_post_rope`, `attn_out` |
| `RESIDUAL_ADD` | Divergence during residual addition | `residual_post_attn`, `residual_post_mlp` |
| `FFN_NORM_PATH` | Divergence in FFN normalization | `ffn_norm_in` |
| `MLP_OUTPUT` | Divergence in MLP computation | `mlp_out` |
| `UNKNOWN` | Unclassifiable with current instrumentation | - |

---

## 4. Investigation Scope

### 4.1 Prompts for Validation

| Prompt | Context Length | Purpose |
|--------|---------------|---------|
| `p0_short` | ~5 tokens | Short-context baseline |
| `p6_len_16` | ~827 tokens | Medium-length (16K equiv) |
| `p6_len_32` | ~1653 tokens | Extended-length (32K equiv) |

### 4.2 Critical Positions

| Position | Context Range | Hypothesis |
|----------|---------------|------------|
| 826 | 16K range | First observed failure point |
| 1652 | 32K range | Second observed failure point |
| Last token | EOS | Potential end-of-sequence divergence |

### 4.3 Layer Coverage

**Minimum required layers:** L0, L1, L2, L4, L8

If trace analysis indicates divergence at earlier layers but failure manifests later, coverage expands to: L3, L5, L6, L7. If divergence originates beyond L8, systematic expansion continues until the failure origin is isolated.

### 4.4 Token Analysis Granularity

Minimum **16 consecutive tokens** spanning early, middle, and late positions within each sequence are analyzed. This provides statistical power to distinguish transient anomalies from systematic failure patterns.

---

## 5. Trace Probe Implementation

### 5.1 Environment Variables

```bash
# B3.61 Activation
export GRETA_B3_61=1

# Stage Tracing
export GRETA_TRACE_STAGE=1
export GRETA_TRACE_STAGE_LAYERS="0,1,2,4,8"
export GRETA_TRACE_STAGE_DEBUG_INPUT=1

# Residual Stream Trace Points
export GRETA_TRACE_B3_61_RESIDUAL_PRE_ATTN=1
export GRETA_TRACE_B3_61_ATTN_IN=1
export GRETA_TRACE_B3_61_ATTN_OUT=1
export GRETA_TRACE_B3_61_RESIDUAL_POST_ATTN=1
export GRETA_TRACE_B3_61_FFN_NORM_IN=1
export GRETA_TRACE_B3_61_MLP_OUT=1
export GRETA_TRACE_B3_61_RESIDUAL_POST_MLP=1
export GRETA_TRACE_B3_61_LOGITS=1

# Critical Positions
export GRETA_TRACE_B3_61_POS_826=1
export GRETA_TRACE_B3_61_POS_1652=1
```

### 5.2 Trace Output Format

Traces are emitted as JSONL records with minimal computational overhead:

```
tools/benchmarks/run_b3_61_mi300x.sh
  └─> artifacts_remote/YYYY-MM-DD/b3_61/traces/
      ├─ p0_short_trace.jsonl
      ├─ p6_len_16_trace.jsonl
      └─ p6_len_32_trace.jsonl
```

---

## 6. Execution Pipeline

### 6.1 Pre-Execution Synchronization

```bash
# 1. Verify clean working tree
git status --porcelain

# 2. Fetch and merge latest
git fetch origin main
git merge origin/main --ff-only

# 3. Record commit hash
COMMIT=$(git rev-parse HEAD)
echo "Commit: $COMMIT"
```

### 6.2 Remote MI300X Execution

```bash
# SSH to remote
ssh user@129.212.184.200

# Pull identical commit
cd /root/gretacore
git fetch origin
git reset --hard origin/main

# Execute pipeline
./tools/benchmarks/run_b3_61_mi300x.sh
```

### 6.3 Artifact Management

```bash
# Package artifacts on remote
tar -czvf greta_b3_61_artifacts.tgz -C artifacts_remote 2026-02-06/b3_61/

# SCP to local
scp greta_b3_61_artifacts.tgz local:/media/leandro/D08A27808A2762683/gretacore/gretacore/artifacts_remote/2026-02-06/

# Verify integrity
sha256sum artifacts_remote/2026-02-06/b3_61/greta_b3_61_artifacts.tgz
```

---

## 7. Analysis Output

### 7.1 First Failure Table (Expected Format)

| Prompt | Position | Layer | Tensor | Hash Match | MAE | FIRST_FAIL (Y/N) |
|--------|----------|-------|--------|------------|-----|------------------|
| p6_len_16 | 826 | 0 | residual_pre_attn | NO | 0.142 | **YES** |
| p6_len_16 | 826 | 1 | residual_pre_attn | NO | 0.138 | NO |
| p6_len_16 | 826 | 2 | residual_pre_attn | NO | 0.141 | NO |
| p6_len_32 | 1652 | 0 | residual_pre_attn | NO | 0.201 | **YES** |

### 7.2 Root Cause Summary

| Category | Occurrences | Percentage |
|----------|-------------|------------|
| ROUTING/SELECTION | 0 | 0% |
| ATTN_KERNEL_INPUTS | 0 | 0% |
| ATTENTION_MECHANISM | 0 | 0% |
| RESIDUAL_ADD | 0 | 0% |
| FFN_NORM_PATH | 0 | 0% |
| MLP_OUTPUT | 0 | 0% |

---

## 8. Recommendations for B3.62

### 8.1 Based on FIRST_FAIL Identification

If `FIRST_FAIL` is **RESIDUAL_PRE_ATTN at Layer 0, Position 826**:
> B3.62 should investigate embedding layer position encoding at extended contexts and test rotary position embedding modifications.

If `FIRST_FAIL` is **ATTN_IN/QKV at Layer 0**:
> B3.62 should audit query/key projection kernels for position-specific errors and validate QKV weight application at boundary positions.

If `FIRST_FAIL` is **ATTN_OUT/RoPE**:
> B3.62 should investigate RoPE application mechanics at extended contexts and check attention score computation for numerical stability.

If `FIRST_FAIL` is **RESIDUAL_POST_ATTN**:
> B3.62 should debug residual addition kernel for position-specific issues and verify dtype consistency across layers.

### 8.2 Testable Hypotheses

1. **Position Encoding Hypothesis**: Divergence originates in RoPE application at specific frequency boundaries
2. **KV Cache Hypothesis**: Cache coherence breaks down at position thresholds
3. **Numerical Precision Hypothesis**: FP16 accumulation errors accumulate beyond certain sequence lengths
4. **Memory Layout Hypothesis**: Strided access patterns cause corruption at specific offsets

---

## 9. Reproducibility Instructions

### 9.1 Prerequisites

- GRETA CORE repository at commit `aad424d...`
- MI300X remote access at `129.212.184.200`
- Binary: `tools/inference/build/greta_infer`
- Model: `models/greta-v1.gguf`

### 9.2 Execution Steps

```bash
# Local
cd /media/leandro/D08A27808A2762683/gretacore/gretacore
git status  # Verify clean
git rev-parse HEAD  # Record commit

# Remote
ssh user@129.212.184.200
cd /root/gretacore
git pull
./tools/benchmarks/run_b3_61_mi300x.sh

# Transfer
scp remote:artifacts_remote/YYYY-MM-DD/b3_61/*.tgz local:artifacts_remote/YYYY-MM-DD/b3_61/

# Analyze
python3 tools/benchmarks/analyze_b3_61_residual_stream_bisect.py \
    --input_dir artifacts_remote/YYYY-MM-DD/b3_61/traces \
    --baseline_dir artifacts_remote/2026-02-05/b3_59/traces \
    --output artifacts_remote/YYYY-MM-DD/b3_61/b3_61_analysis.txt
```

---

## 10. Appendices

### Appendix A: Configuration Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `GRETA_B3_61` | 1 | Enable B3.61 tracing |
| `GRETA_TRACE_STAGE_LAYERS` | 0,1,2,4,8 | Target layers |
| `GRETA_TRACE_B3_61_POS_826` | 1 | Enable position 826 tracing |
| `GRETA_TRACE_B3_61_POS_1652` | 1 | Enable position 1652 tracing |

### Appendix B: Raw Trace Sample

```json
{"prompt_id":"p6_len_16","token_id":826,"pos_id":826,"logical_tok_idx":5,"phase":"prefill_last","layer":0,"tensor_name":"residual_pre_attn","route":"decode_attention_main","timestamp":"2026-02-06T15:14:13Z","git_commit_hash":"aad424dca7ef7f4e9142a5d4df23168bb1d087f2","hash":"sha256:3a7b2c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2","nz_count":4096}
```

### Appendix C: Related Documentation

- [B3.59 Embedding Verification](2026_02_04_B3_59_embedding_debug_input.md)
- [B3.60 Attention Block Bisect](2026_02_05_B3_60_attention_block.md)
- [PROGRESS.md](../PROGRESS.md)

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-06  
**Author:** GRETA CORE Engineering Team
