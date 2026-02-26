# Architecture Decisions Record: redis-infer

**Date**: 2026-02-26
**Status**: Approved (Discovery stage)
**Context**: Decisions made in response to Risk Register findings (5 binary decisions required)

---

## Decision 1: Inference Engine Source

**Decision**: Link llama.cpp C API

**Rationale**: No pure-C Qwen3-4B text generation engine exists. llama.cpp provides
a stable C API (`llama.h`), supports GGUF format including Qwen3-4B, and is the
most battle-tested inference engine available. This abandons the "pure-C zero
dependency" philosophy but delivers working inference in ~1 month instead of 12+.

**Implications**:
- C++ dependency (llama.cpp is C++ internally, C API externally)
- Must link against libllama at build time
- Gains access to GGUF quantization, KV cache management, batching, sampling
- Can still use antirez-style pure-C engines for embeddings (gte-pure-C) and
  speech (voxtral.c) alongside llama.cpp for text generation

---

## Decision 2: Concurrency Model

**Decision**: Max thread count = number of CPUs

**Rationale**: Unbounded thread creation (current code) leads to OOM. A fixed thread
pool sized to CPU count provides natural backpressure. Each inference thread runs
a forward pass that is CPU-bound; more threads than cores causes contention, not
parallelism.

**Implications**:
- Need to detect CPU count at module load time (sysconf/_SC_NPROCESSORS_ONLN)
- Implement a work queue with bounded thread pool
- Requests beyond pool capacity wait in queue or return error
- Memory budget: N concurrent inferences x KV cache per inference must fit in
  available RAM (after Redis maxmemory reservation)

---

## Decision 3: DMA vs Copy

**Decision**: Investigate deeper

**Context**: The risk scan claims zero-copy DMA is architecturally impossible because
the DMA pointer is only valid while the GIL is held, and holding the GIL blocks
Redis. However, there may be nuances:
- Can we use READONLY DMA on immutable keys with specific eviction protection?
- What is the actual cost of memcpy for typical token contexts (4k-128k tokens)?
- Is there a Redis module API mechanism to pin keys against eviction?
- Does llama.cpp's C API accept external buffers or does it copy internally anyway?

**Action**: Research phase will produce testable hypotheses and benchmarks for this.

---

## Decision 4: Chunking Strategy

**Decision**: Split by paragraph

**Rationale**: Paragraph-level chunking provides a good balance between granularity
and coherence for code files:
- In Rust: paragraphs roughly correspond to function bodies, impl blocks, and
  doc comment sections (separated by blank lines)
- More granular than whole-file (which was identified as a critical RAG problem)
- Less fragile than AST-based function extraction (which requires a Rust parser)
- Simple to implement: split on double-newline boundaries
- Overlap between chunks (e.g., 1 paragraph overlap) preserves context

**Implications**:
- Tokenization script needs paragraph splitting before storing
- Each chunk gets its own Redis key (tokens:filepath:chunk_N)
- Each chunk gets its own embedding in Vector Sets
- Key naming must include full relative path + chunk index to avoid collisions

---

## Decision 5: GPU Availability

**Decision**: GPU on development machine, no GPU in production

**Rationale**: Development/benchmarking can use GPU for fast iteration. Production
deployment on the 256 GB box is CPU-only.

**Implications**:
- Performance targets must be based on CPU inference, not GPU
- The risk scan's finding stands: <800ms per generation is not achievable on CPU
  for 256-token outputs. Realistic CPU targets:
  - Qwen3-4B Q4 on 64-core CPU: ~5-15 tok/s generation
  - 256 tokens output: ~17-51 seconds per request
  - Concurrent requests: limited by memory (KV cache per session)
- Value proposition shifts to: operational simplicity + atomic RAG, not raw speed
- GPU dev machine useful for: benchmarking, comparing DMA vs copy overhead,
  validating correctness before CPU deployment
- Consider: could production box get a GPU later? If so, design should not
  preclude GPU acceleration path via llama.cpp's CUDA/Metal support

---

## Summary

| # | Decision | Choice |
|---|----------|--------|
| 1 | Inference engine | llama.cpp C API |
| 2 | Max threads | = CPU count |
| 3 | DMA vs copy | Investigate deeper (research phase) |
| 4 | Chunking | Paragraph-level splitting |
| 5 | GPU | Dev only; production is CPU-only |
