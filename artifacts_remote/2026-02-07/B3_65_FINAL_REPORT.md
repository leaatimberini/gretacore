================================================================================
              B3.65 DECODE DETERMINISM AUDIT - FINAL REPORT
================================================================================

Date: 2026-02-07
Author: L.E.T / Leandro Emanuel Timberini
Status: COMPLETED_WITH_ISSUES

================================================================================
                           WORKFLOW EXECUTION SUMMARY
================================================================================

[✓] HF Token Provided: [REDACTED]
[✓] Llama-2-7B Downloaded: meta-llama/Llama-2-7b (13GB PyTorch format)
[✓] GGUF Conversion Skipped: Used pre-quantized model from TheBloke
[✓] GGUF Model Downloaded: TheBloke/Llama-2-7B-GGUF (Q8_0, 6.7GB)
[✓] Model Compatibility Verified: GRETA guard rail PASSED
    - layers=32, dim=4096, heads=32, hidden=11008
[✗] B3.65 Determinism Sweep: FAILED (Segmentation Fault)

================================================================================
                            MODEL VERIFICATION
================================================================================

Model File: /root/gretacore/models/greta-v1-llama2-7b.gguf
Size: 6.7 GB
Format: GGUF Q8_0 (8-bit quantization)

GRETA Guard Rail Check:
  Expected: hidden_dim=11008
  Got:      hidden_dim=11008
  Status:   PASSED ✓

Model Configuration:
  - Architecture: llama
  - Layers: 32
  - Embedding Dim: 4096
  - Attention Heads: 32
  - FFN Hidden Dim: 11008
  - Vocab Size: 32000
  - Parameters: 6.74B

================================================================================
                         B3.65 SWEEP EXECUTION
================================================================================

Attempted: 2026-02-07 13:02 UTC
Runs Requested: 10
Runs Completed: 0
Result: SEGMENTATION FAULT

Error Details:
  - Model passed guard rail compatibility check
  - Segmentation fault occurred during weight loading (layer 0/32)
  - Error: "Segmentation fault (core dumped)"
  - The Q8_0 quantization format may be incompatible with binary

================================================================================
                            ROOT CAUSE ANALYSIS
================================================================================

The segmentation fault occurs because:
1. The GGUF model uses Q8_0 quantization (8-bit)
2. The GRETA binary (greta_infer_fixed) was compiled for f16 precision
3. The quantized weights have a different memory layout than f16 weights

Evidence:
  - Model loads successfully (guard rail passes)
  - Error occurs at: "[GRETA_SCHED] Loading layer 0/32..."
  - Core dump generated

================================================================================
                         POSSIBLE SOLUTIONS
================================================================================

Option A: Use f16 GGUF model (not quantized)
  - Download: TheBloke/Llama-2-7B-GGUF (file: llama-2-7b.f16.gguf)
  - Size: ~13GB (uncompressed)
  - Risk: LOW - should be compatible with f16 binary

Option B: Recompile GRETA with Q8_0 support
  - Requires: Modify src/kernels to handle quantized weights
  - Time: ~2-4 hours
  - Risk: MEDIUM - kernel modifications needed

Option C: Use INT8 quantization compatible model
  - Find model with INT8 quantization
  - Requires GRETA binary recompilation for INT8

================================================================================
                            ARTIFACTS
================================================================================

Location: artifacts_remote/2026-02-07/
  - b3_65/run/run_01.txt (error log)
  - workflow_llama2_download.log
  - greta-v1-llama2-7b.gguf (6.7GB) - NOT included in rescue tgz (too large)

Remote Rescue Archive:
  - /root/gretacore_remote_rescue_2026-02-07_b3_65.tgz (41KB)
  - Contains error logs and configuration files

================================================================================
                            NEXT STEPS
================================================================================

1. Download f16 GGUF model for compatibility testing:
   python3 -c "
   from huggingface_hub import HfApi
   api = HfApi(token='[REDACTED]')
   api.snapshot_download('TheBloke/Llama-2-7B-GGUF',
                        local_dir='/root/models/gguf/f16',
                        allow_patterns=['llama-2-7b.f16.gguf'])
   "

2. Copy f16 model to greta-v1.gguf and retry B3.65

3. Alternatively, recompile GRETA binary with quantization support

================================================================================
                            VERDICT
================================================================================

VERDICT: BLOCKED_BY_SEGMENTATION_FAULT

Reason: Model-binary quantization mismatch (Q8_0 vs f16 expected)

Resolution Required:
  - Use f16 GGUF model instead of Q8_0
  - OR recompile GRETA with Q8_0 support

Signed: L.E.T / Leandro Emanuel Timberini
Timestamp: 2026-02-07T13:04:00Z
