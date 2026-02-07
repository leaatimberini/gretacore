# GRETA CORE

**Status**: Phase 3 - B3.xx Audit Series (Up to B3.66)

---

## Authorship & Leadership

**GRETA CORE** is an independent engineering project conceived, founded, and led by:

- **Leandro Emanuel Timberini**
  - Founder & Principal Systems Architect
  - All architectural decisions originate from this authorship
  - Long-term vision and foundational principles defined by the founder

---

## Project Description

GRETA CORE is a long-term engineering initiative focused on building a **high-performance, minimal, CUDA-like compute stack for AMD hardware**, designed specifically for Large Language Models (LLMs).

The project exists to break the current CUDA lock-in by addressing the problem at its root: **software**.

---

## Phase 3 Progress (B3.xx Audit Series)

| Milestone | Status | Description |
|-----------|--------|-------------|
| B3.52 | ✅ PASS | KV cache addressing fix |
| B3.55-B3.58 | ✅ PASS | Root cause isolation (RoPE/Q-proj/RMSNorm) |
| B3.59 | ✅ PASS | Embedding + StageDebugInput audit |
| B3.64 | ✅ CLOSED | RoPE Kernel Launch Diagnostics (d_pos FP16→FP32 fix) |
| B3.65 | ✅ PASS_DETERMINISTIC | Decode Determinism Audit |
| B3.66 | 🔄 IMPLEMENTED_PENDING_RUN | Prefill vs Decode Drift Probe |

**Documentation**:
- [Progress Index](docs/PROGRESS.md)
- [AMD Reports Index](docs/AMD/INDEX.md)

---

## Motivation

The modern AI ecosystem is dominated by a single compute platform. This dominance has created:

- Artificial barriers to entry
- Inflated hardware costs
- Limited innovation

GRETA CORE approaches this problem from a **software-first perspective**, aiming to unlock the full potential of AMD hardware through a focused, performance-driven compute stack.

---

## Philosophy

All principles originate from the founder's vision:

- **Software over hardware** - Control the stack, not just the silicon
- **Full stack control** - From kernel to inference
- **Minimalism over bloat** - Every line must justify its existence
- **Performance over abstraction** - Zero-cost abstractions only
- **Long-term engineering discipline** - Decades, not quarters

---

## What GRETA CORE Is

- A custom compute runtime for AMD hardware (MI300X, MI200, RDNA)
- A kernel-first LLM execution stack
- A CUDA-like developer experience without replicating CUDA
- A long-term research and engineering initiative
- An install that bundles torch, triton, and jax (no extra installs required)

---

## What GRETA CORE Is Not

- Not a CUDA fork
- Not a thin wrapper around existing frameworks
- Not a general-purpose GPU compute platform
- Not a short-term optimization project

---

## Architecture Highlights

### Runtime Stack

```
src/rt/
├── allocator/      # Memory management
├── backend/       # HIP, Vulkan backends
├── dispatch/      # Graph dispatch
├── graph/         # Graph execution
├── stream/        # Stream management
└── telemetry/     # Performance monitoring
```

### Inference Engine

```
src/inference/
├── block_scheduler/    # Block-level scheduling
├── generator/          # Token generation
├── layer_trace/        # Layer-by-layer tracing
├── model_config/      # Model configuration
├── stage_trace/       # Stage-level tracing
├── tokenizer/         # Tokenization
├── trace/             # General tracing
└── weight_loader/     # Weight loading
```

---

## Supported Hardware

- **AMD MI300X** - Primary development target
- **AMD MI200 series** - Supported
- **AMD RDNA3+** - Compatible

---

## Documentation Structure

```
docs/
├── AMD/              # AMD-specific audit reports (B3.xx series)
│   └── phases/       # Phase documentation
├── en/              # English documentation
├── es/              # Spanish documentation
├── strategy/        # Strategic planning documents
├── PROGRESS.md      # Overall progress tracking
├── CHANGELOG.md     # Version history
└── WORKSPACE_RULES.md  # Development guidelines
```

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/leaatimberini/gretacore.git
cd gretacore

# View documentation
cat docs/PROGRESS.md
cat docs/AMD/INDEX.md

# Run benchmarks
cd tools/benchmarks
./run_bench.py
```

---

## Contributing

This is a **long-term engineering initiative** led by the founder. All contributions must align with the project's philosophy of minimalism, performance, and full stack control.

**Focus areas**:
- Source code improvements
- Technical documentation
- Reproducible benchmarks
- Verifiable audits

**Non-acceptable changes**:
- Feature bloat
- Undocumented modifications
- Changes unrelated to the inference engine

---

## License & Attribution

All code, documentation, and architectural decisions are the intellectual property of **Leandro Emanuel Timberini** as Founder & Principal Systems Architect.

---

## Contact

- GitHub: [@leaatimberini](https://github.com/leaatimberini)
- Repository: https://github.com/leaatimberini/gretacore

---

*Last updated: February 2026*
*Version: 0.1.0*
