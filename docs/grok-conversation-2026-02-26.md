---
created: 2026-02-26T16:54:01 (UTC +00:00)
tags: [grok, redis-infer, antirez, inference, redis-module]
source: https://grok.com/c/c64ecfae-a86f-4f9a-ad53-0a5df6b2b029?rid=1e334748-adbc-4458-825a-1d917eadaa55
author: alex
---

# Grok Conversation: redis-infer -- Pure-C Inference Inside Redis

## Context (User-Provided Background)

Antirez (Salvatore Sanfilippo, Redis creator) is extremely active in 2026 on pure C,
zero-dependency inference engines for AI models. He rapidly builds minimal,
high-performance inference pipelines (often in a weekend or hours) with heavy
assistance from LLMs like Claude for code generation, while steering the architecture.

Key recent projects (all on GitHub under antirez/):
- iris.c / flux2.c: Pure C inference for FLUX.2-klein-4B image generation
- qwen-asr (updated ~Feb 16, 2026): C inference for Qwen3-ASR 0.6B/1.7B speech-to-text
- voxtral.c: Pure C inference for Mistral Voxtral Realtime 4B speech-to-text
- Earlier: Pure C BERT/GTE-small-style embedding inference (~700 LOC)

Redis 8.x (Feb 2026) is the unified in-memory data platform with:
- Vector Sets (native data type, created by antirez): HNSW, quantization, VSIM/VRANGE
- Enhanced Redis Query Engine
- JSON as first-class citizen, LangCache, TimeSeries

## Conversation Flow

### Phase 1: Initial Plan -- In-Memory Inference Pipeline

**Goal**: Ultra-low-latency, scalable LLM/multimodal inference where tokenization is
done once offline. Pre-tokenized token ID sequences live in Redis for O(1) retrieval.

**Core Architecture**:
- Redis = central in-memory store (single instance or cluster)
- Holds: pre-tokenized sequences, embeddings (Vector Sets), vocab, KV cache blobs, session state
- Inference Engine = pure C binary (extend antirez's voxtral.c/flux2.c pattern)
- Uses hiredis or custom Redis module for zero-copy access

**Storage Schema**:
```redis
# Vocabulary
HSET vocab:forward "hello" 12345
HSET vocab:reverse 12345 "hello"

# Pre-Tokenized Data
SET chunk:doc123 '{"tokens": [12, 45, 678, ...], "metadata": {"source": "..."}}'
# Or binary: SET chunk:doc123 <binary-blob-of-uint32s>

# Vector Sets integration
VADD rag:chunks chunk:doc123 VECTOR <embedding_vector> JSON '{"tokens_key": "chunk:doc123"}'

# KV Cache / Session State
HSET session:uuid layer:0_pos:42 kv_past <binary-tensor-blob>

# Model Weights
SET model:llama3:layer0:weights <safetensors-converted-binary>
```

**Inference Flow**:
1. Offline: tokenize corpus, compute embeddings, insert into Redis
2. At inference: fetch pre-tokenized prompt, VSIM RAG, assemble tokens, forward pass, generate, detokenize

### Phase 2: User Asks "Extend -- fork redis and move antirez inference engine inside redis"

User interrupted and clarified: "Yes as module" (not a fork).

### Phase 3: Redis Module Approach

**Module Name**: redis-infer
**Namespace**: INFER.* (INFER.GENERATE, INFER.RAG, INFER.TRANSCRIBE)
**Supported engines**: voxtral/qwen-asr (speech), iris/flux2 (image), future LLM

**Why module beats fork**:
- Load/unload with MODULE LOAD/UNLOAD -- zero downtime
- Full access to native Vector Sets, JSON, Strings
- Memory tracked by Redis maxmemory
- Works on Redis Cluster, Sentinel, Enterprise

**Architecture**:
```
Client -> INFER.GENERATE
  -> Redis Module (in-process)
     -> Zero-copy DMA on pre-tokenized STRING / JSON / VectorSet tokens
     -> Vector Sets -> VSIM -> retrieve token keys (HNSW <1 ms)
     -> Antirez engine core (voxtral.c / iris.c adapted as library)
     -> KV cache stored as binary STRING per session
     -> Reply: tokens or detokenized text / image bytes
```

**Proposed Commands**:
```
INFER.LOAD flux weights:model:flux:weights
INFER.TOKSTORE tokens:doc123 12 45 678 ...
INFER.GENERATE llama prompt:tokens:user max_tokens 512 temp 0.7
INFER.RAG vecset:rag query_embed:vec topk 5 model:llama system:tokens:system
INFER.TRANSCRIBE voxtral audio:wav_key
INFER.IMAGE flux prompt:tokens:text steps 4
```

**Non-blocking pattern**:
```c
RedisModuleBlockedClient *bc = RedisModule_BlockClient(ctx, NULL, NULL, NULL, 0);
// spawn pthread
// in background: run_antirez_inference(...)
// RedisModule_UnblockClient(bc, reply);
```

### Phase 4: Complete Voxtral Module Skeleton (~320 lines)

Grok delivered a full redis-infer.c with:
- `INFER.VOXLOAD <model_dir>` -- loads Voxtral model
- `INFER.TRANSCRIBE <audio_key>` -- zero-copy WAV from Redis STRING -> text
- `INFER.STREAMSTART` -- streaming stub
- `GetAudioDMA()` helper for zero-copy audio access
- Makefile linking voxtral.c objects
- `RedisModule_OnUnload` for cleanup

### Phase 5: Best Model for Rust?

Grok's ranking (Feb 2026):
1. Claude Opus 4 / Claude Code -- leader for serious Rust work
2. GPT-5.2 / Codex-high -- very close second
3. Gemini 3 Pro -- excellent all-rounder
4. Local: Devstral-2-123B, Qwen3-Coder, GLM-4.7

### Phase 6: Best SMALLEST Model for Redis Module?

User refocused: "what's the best smallest model we can run inside redis via our new module?"

**Recommendation**: Qwen3-4B (Q4_K_M, ~2.5 GB)
- Best quality/size sweet spot for Rust coding
- 0.6B too weak, 4B is the minimum useful size
- High pure-C feasibility (Qwen family has clean ports)
- Alternatives: GLM-4-Air (excellent coding), Phi-4 Mini, Gemma 3 4B

### Phase 7: 256 GB RAM -- Cluster or Standalone?

User asked if Redis Cluster needed for sharding on 256 GB box.

**Answer**: Standalone is fine.
- Redis 8 has no per-instance RAM limit
- Set `maxmemory 220gb`, leave 40-60 GB headroom
- Module uses `RedisModule_BlockClient` + pthreads -> full multi-core
- Main thread stays responsive (only fast DMA + VSIM)
- Only need cluster when data > ~200 GB or need HA/horizontal scaling

### Phase 8: Detailed Plan v3.1 (Final)

Complete single-file blueprint with:
- Redis config: maxmemory 220gb, io-threads 16, activedefrag yes
- Model: Qwen3-4B Q4 (~2.5 GB)
- Tokenization: Byte-level BPE, uint32 binary packed, offline script
- Embeddings: antirez/gte-pure-C (384 dim) -> upgrade to Qwen3-Embedding-4B
- Storage schema: tokens as binary STRING, VADD with QUANT int8
- Full C module code (~80 lines skeleton)
- Performance target: 170 GB data + 40-80 concurrent generations

### Phase 9: Final Zip Artefact

Project structure:
```
redis-infer-project/
  README.md
  docs/REDIS-INFER-DETAILED-PLAN-v3.1.md
  src/redis-infer.c
  Makefile
  scripts/tokenize-and-store.py
  scripts/embed-and-store.py
```

## Key Technical Decisions Made in Conversation

1. **Module, not fork**: Follow antirez's own pattern (Vector Sets started as module)
2. **Qwen3-4B**: Smallest usable coding model for Rust
3. **Standalone Redis**: 256 GB box does not need cluster
4. **Pre-tokenization**: Offline once, store as binary uint32, zero-copy DMA
5. **gte-pure-C**: Start with antirez's own embedding engine (384 dim)
6. **Background pthreads**: BlockClient pattern for concurrent inference
7. **Vector Sets**: Native HNSW + int8 quantization for code RAG

## Unresolved / Stub Items

1. Core inference engine (qwen_load, qwen_generate, gte_embed) are stubs
2. No real Qwen3-4B C implementation exists yet
3. No tests of any kind
4. Key collision in scripts (filename-only, not path-based)
5. No chunking strategy for large files
6. No error handling in worker thread (NULL check on qwen_generate)
7. No thread pool (unbounded pthread_create + detach)
8. No graceful shutdown for detached threads
9. Performance claims (40-80 concurrent, <800ms) are unvalidated
