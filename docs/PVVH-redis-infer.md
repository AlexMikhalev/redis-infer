# Product Vision and Value Hypothesis: redis-infer

**Date**: 2026-02-26
**Author**: alex (with ZDP Discovery assist)
**Status**: Draft
**ZDP Stage**: Discovery

---

## Vision Statement

For developers who run local AI inference alongside Redis and want to eliminate the
network hop between data store and model, redis-infer is a pure-C Redis module that
executes inference in-process with zero-copy access to pre-tokenized data and Vector
Sets. Unlike running llama.cpp or vLLM as a separate sidecar, redis-infer removes
serialization overhead entirely by executing the forward pass inside the Redis
process with direct memory access to stored tokens and embeddings.

---

## Target Users

| Segment | Description | Current Behaviour | Pain Points |
|---------|-------------|-------------------|-------------|
| Local inference developers | Developers running open-weight models (3-8B) on their own hardware for code completion, RAG, transcription | Run llama.cpp/Ollama as separate server, fetch context from Redis over TCP, serialize/deserialize token arrays | Network roundtrip adds latency to every inference call; two processes to monitor, configure, and keep in sync; token data copied multiple times between Redis and inference engine |
| Redis power users doing AI/RAG | Teams already using Redis Vector Sets for semantic search who want to add generation without a separate service | Deploy Redis for embeddings + a second service (vLLM, TGI, llama.cpp server) for generation; glue them together with application code | Operational complexity of two services; context assembly requires multiple roundtrips (VSIM query, GET tokens, POST to inference API); no atomicity across retrieval and generation |
| Antirez-style minimalists | Developers who value zero-dependency, pure-C, single-binary deployments following antirez's philosophy | Build custom inference pipelines inspired by antirez's voxtral.c, iris.c, qwen-asr repos | No standardized way to embed these engines inside Redis; each project is standalone with its own I/O; reusable module pattern does not yet exist |

---

## Jobs to Be Done

| JTBD | Context | Desired Outcome | Current Workaround |
|------|---------|-----------------|-------------------|
| Run code completion over my codebase with sub-second latency | Developer working on a Rust project, wants completions informed by existing code patterns | Type a partial function, get a contextually relevant completion in <1s using similar functions from the repo as context | Run Ollama/llama.cpp separately, write custom glue to fetch embeddings from Redis, assemble prompt, call inference API, parse response |
| Retrieve relevant code chunks and generate in a single operation | Building a RAG pipeline where Vector Sets hold code embeddings and tokens are pre-stored | Issue one command (INFER.GENERATE or INFER.RAG) that does VSIM + token assembly + generation atomically | Application-level orchestration: VSIM call, multiple GET calls for token keys, serialize into prompt format, HTTP POST to inference server, parse streaming response |
| Eliminate repeated tokenization on the hot path | Processing the same codebase files for inference repeatedly (multi-turn editing, iterative generation) | Tokenize once offline, store forever, never re-tokenize at inference time | Re-tokenize on every call (or maintain a fragile external cache), wasting CPU cycles on BPE encoding that produces identical results each time |
| Deploy a single process for data + inference | Running on a single high-RAM box (64-256 GB) where operational simplicity matters | One redis-server process that handles storage, search, and inference; one thing to monitor, restart, and backup | Manage Redis + llama.cpp server + process supervisor + health checks + log aggregation for two separate services |

---

## Value Mechanisms

| Mechanism | Description | User Benefit | Community Benefit |
|-----------|-------------|-------------|-----------------|
| Zero-copy DMA | RedisModule_StringDMA gives the inference engine a direct pointer to token data in Redis memory; no memcpy, no serialization | Eliminates ~10-100us per context fetch that would otherwise be spent on TCP + deserialization for each token key | Demonstrates a pattern for in-process data access that other Redis AI modules can adopt |
| Pre-tokenized storage | Tokenize once offline (BPE is deterministic), store as packed uint32 binary, reuse forever | Removes tokenization from the hot path entirely; for a 10k-token context, saves ~5-50ms per call depending on tokenizer complexity | Establishes a standard Redis key format for pre-tokenized sequences that tooling can build on |
| Vector Sets integration | Native HNSW similarity search + attached JSON metadata + token key references, all in one data structure | Sub-ms semantic retrieval + zero-copy token fetch in a single operation; no cross-service coordination | Showcases Vector Sets (antirez's invention) for code RAG, providing real-world usage patterns |
| Background pthreads | RedisModule_BlockClient + worker threads for concurrent inference without blocking the main Redis event loop | Multiple users/sessions can generate concurrently; Redis stays responsive for other workloads | Provides a reference implementation of the module threading pattern for heavy compute |
| Antirez engine compatibility | Designed to embed antirez's pure-C inference engines (voxtral.c, iris.c, qwen-asr, future LLM engines) as drop-in libraries | Access to high-quality, minimal, zero-dependency inference without framework bloat | Creates an ecosystem where antirez's standalone engines gain a deployment target inside Redis |

---

## Success Metrics

| Metric | Definition | Target | Measurement Method | Timeframe |
|--------|-----------|--------|-------------------|-----------|
| Working inference | Module loads, accepts pre-tokenized input, produces coherent output from a real (non-stub) model | Qwen3-4B or equivalent generates compilable Rust code from pre-tokenized prompt | Manual test: store tokenized Rust function, call INFER.GENERATE, compile output | Phase 1 |
| Latency improvement vs sidecar | Time from INFER.GENERATE command to first token, compared to equivalent Redis GET + HTTP POST to llama.cpp server | Measurable improvement (target: >30% reduction in context assembly latency) | Benchmark script comparing redis-infer vs hiredis+llama.cpp-server for identical prompts | Phase 1 |
| Zero-copy verified | Confirm no memcpy occurs between Redis string storage and inference engine token buffer | DMA pointer used directly; verified via perf/strace or Redis module debug logging | Instrumented build with pointer address logging | Phase 1 |
| Community adoption signal | GitHub stars, forks, or issues from developers other than the author | >10 stars or >1 external issue/PR within 3 months of public release | GitHub metrics | Phase 2 |
| Concurrent generation | Multiple simultaneous INFER.GENERATE calls complete without blocking Redis or corrupting state | 4+ concurrent generations on a 64-core box without main thread stalls | Load test with redis-benchmark or custom concurrent client | Phase 2 |

---

## Value Hypothesis

"We believe that developers running local AI inference with Redis will adopt
redis-infer because it eliminates the network hop and serialization overhead between
their data store and inference engine, reducing context assembly latency and
operational complexity. We will know this is true when the module demonstrates
measurable latency improvement over the sidecar pattern in benchmarks and attracts
community interest (stars/forks/issues) after public release."

### Validation Plan

| Hypothesis Element | Validation Method | Evidence Required | Status |
|-------------------|-------------------|-------------------|--------|
| Developers want in-process inference in Redis | Search for existing demand: GitHub issues on Redis/llama.cpp requesting tighter integration; antirez's own neural-redis precedent | Existence of neural-redis (antirez, 2016) + RedisAI (Redis Labs) proves prior attempts; community discussion of latency in sidecar setups | Partially validated -- prior art exists but those projects used different engines/approaches |
| Zero-copy DMA provides measurable latency benefit | Benchmark: redis-infer DMA vs hiredis GET + deserialize for 10k-128k token contexts | >30% reduction in context assembly time for contexts >4k tokens | Untested |
| Pre-tokenization eliminates meaningful overhead | Profile: BPE tokenization time for typical Rust files (1k-10k tokens) vs zero (already stored) | Tokenization takes >5ms for contexts >4k tokens, making pre-storage worthwhile | Untested -- needs profiling |
| Qwen3-4B (or similar) is viable in-process | Load Qwen3-4B Q4 in a Redis module, generate tokens without crashing Redis or exhausting memory | Stable generation of 256+ tokens without OOM, segfault, or main thread stall | Untested -- core engineering risk |
| Community finds this useful | Publish on GitHub, post to Redis community forums, reference antirez's engines | >10 GitHub stars or >1 external contribution within 3 months | Untested |

---

## Positioning

| Dimension | redis-infer | llama.cpp server (sidecar) | RedisAI / redis-inference-optimization | Ollama |
|-----------|------------|---------------------------|---------------------------------------|--------|
| Architecture | In-process Redis module | Separate HTTP/gRPC server | In-process Redis module (deprecated/stale) | Separate daemon with REST API |
| Data access | Zero-copy DMA to Redis keys | Network roundtrip (TCP/Unix socket) | Module API (but framework-heavy) | Network roundtrip |
| Dependencies | Pure C, zero external deps | C++ (ggml), optional CUDA/Metal | PyTorch/TensorFlow/ONNX runtime | Go wrapper around llama.cpp |
| Model support | Antirez-style pure-C engines (voxtral, iris, future LLM) | Broad GGUF ecosystem | ONNX, TF, PyTorch models | GGUF models via llama.cpp |
| Maturity | Experimental / early | Production-grade, widely deployed | Stale (last meaningful update 2024) | Production-grade, widely deployed |
| Vector search integration | Native (same process, same memory space) | Requires separate Redis calls + app glue | Native (but separate from inference path) | None built-in |
| Philosophy | Antirez minimalism: pure C, no bloat | Performance-pragmatic C++ | Enterprise/framework-heavy | User-friendly wrapper |

---

## Non-Goals

What this project is explicitly NOT:

- **Not a general-purpose inference server**: redis-infer targets antirez-style pure-C
  engines only. It will not support PyTorch, ONNX, or TensorFlow models. Use
  RedisAI or a sidecar for those.

- **Not a replacement for llama.cpp**: llama.cpp has broad model support, GPU
  acceleration, speculative decoding, and a mature ecosystem. redis-infer is for
  the specific niche of zero-copy in-Redis inference with minimal engines.

- **Not a hosted/cloud service**: This is a loadable .so module for self-hosted Redis
  instances. No SaaS, no managed offering.

- **Not for models >8B parameters**: The in-process constraint means model weights
  share Redis memory. Practical limit is ~4-8B quantized models on typical hardware.

- **Not a tokenizer**: Tokenization happens offline via Python scripts. The module
  consumes pre-tokenized binary data. An optional in-module tokenizer may come later.

- **Not production-critical (yet)**: This is an experimental open-source project. A
  crash in the inference engine will crash Redis. Use replication/persistence if you
  run this alongside production data.

---

## Time Horizon

| Phase | Scope | Key Milestones |
|-------|-------|----------------|
| Phase 1 (POC) | Replace stubs with real inference engine (Qwen3-4B or smallest viable). Zero-copy DMA verified. Single-threaded generation works. Builds on both macOS and Linux. | Stub replacement; first real token generated inside Redis; latency benchmark vs sidecar |
| Phase 2 (Usable) | Thread pool for concurrent generation. RAG integration (VSIM + token concat + generate in one command). Pre-tokenization ingest scripts hardened (path-based keys, chunking). Basic error handling and graceful shutdown. | Concurrent generation; INFER.RAG command; public GitHub release |
| Phase 3 (Community) | Multi-engine support (voxtral for speech, iris for image alongside LLM). KV cache persistence for multi-turn. Documentation and examples. Community feedback integration. | Multi-modal; published examples; external contributors |

---

## Risks and Assumptions

| Assumption | Basis | Risk if Wrong |
|------------|-------|---------------|
| A pure-C Qwen3-4B (or similar) inference engine can be built or adapted | Antirez has built voxtral.c, qwen-asr, iris.c in pure C; llama.cpp core is C/C++; community ports exist | **Critical**: If no viable pure-C LLM engine materializes, the module has no core functionality. Mitigation: could link llama.cpp as a C library instead of pure-C. |
| Zero-copy DMA provides meaningful latency improvement over TCP | RedisModule_StringDMA eliminates memcpy; TCP adds ~50-200us per roundtrip on localhost | **Medium**: If the improvement is negligible compared to inference time (seconds), the value proposition weakens. Still provides operational simplicity. |
| In-process inference does not destabilize Redis | Redis modules run in the same address space; a segfault kills Redis | **High**: Memory corruption in C inference code affects all of Redis. Mitigation: extensive testing, memory sanitizers, process isolation via replication. |
| Qwen3-4B produces useful Rust code at 4-bit quantization | Grok conversation claims "excellent Rust coding quality" for Qwen3-4B | **Medium**: Quality may be insufficient for practical use. Need empirical evaluation. Can switch models if needed. |
| Pre-tokenization is worthwhile vs on-the-fly | BPE tokenization for typical contexts takes >5ms | **Low**: Even if tokenization is fast, pre-storage enables instant multi-turn and batch assembly. Marginal benefit varies by workload. |
| Community interest exists for this approach | Antirez's neural-redis (2016) and RedisAI prove prior demand; antirez's 2026 pure-C engines are popular | **Medium**: Niche may be too small. But experimental/open-source framing means low cost of failure. |

---

## Stakeholder Alignment

| Stakeholder | Interest | Approval Required |
|-------------|----------|-------------------|
| alex (project owner) | Personal exploration of in-Redis inference; potential Terraphim integration | Yes -- sole decision maker |
| Open source community | Novel Redis module combining Vector Sets + inference; reference implementation of antirez's engine pattern | No -- but feedback shapes direction |
| antirez (indirect) | Potential user/endorser if module demonstrates clean integration with his engines | No -- but alignment with his philosophy is a design constraint |

---

## References

- [Antirez's pure-C AI inference engines (flux2.c, voxtral.c)](https://www.ultrathink.ai/news/flux-2-klein-pure-c-inference-antirez)
- [Redis Creator builds speech recognition in pure C](https://www.abit.ee/en/artificial-intelligence/redis-voxtral-speech-recognition-c-mistral-antirez-machine-learning-ai-en)
- [antirez GitHub repositories](https://github.com/antirez?tab=repositories)
- [antirez/neural-redis -- prior art for in-Redis inference](https://github.com/antirez/neural-redis)
- [RedisAI / redis-inference-optimization](https://github.com/RedisAI/redis-inference-optimization)
- [Redis RAG at Scale (2026)](https://redis.io/blog/rag-at-scale/)
- [Redis Vector similarity and Vector Sets](https://redis.io/blog/vector-similarity/)
- [Qwen3-4B GGUF on HuggingFace](https://huggingface.co/Qwen/Qwen3-4B-GGUF)
- [llama.cpp -- LLM inference in C/C++](https://github.com/ggml-org/llama.cpp)
- [vLLM vs llama.cpp comparison](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [antirez blog: "Don't fall into the anti-AI hype"](https://antirez.com/news/158)

---

## Validation Checklist

- [x] Vision statement follows the "[For...who...] [product] is a [category] that [benefit]" structure
- [x] At least one target user segment defined with current behaviour and pain points
- [x] At least one JTBD stated with context and desired outcome
- [x] Value hypothesis stated in falsifiable form with measurable signal
- [x] Validation plan has concrete methods and evidence thresholds
- [x] Non-goals section populated (what the product will NOT do)
- [x] Success metrics have definitions, targets, and measurement methods
- [x] Stakeholder alignment section identifies who must approve
