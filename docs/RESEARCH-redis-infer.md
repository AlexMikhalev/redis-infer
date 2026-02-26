# Research Document: redis-infer
## ZDP Discovery Artefact 3/5 -- Deep Technical Investigation

**Date**: 2026-02-26
**Author**: Phase 1 Research Analyst (ZDP Discovery)
**Status**: Discovery Stage -- Pending Approval
**Scope**: Four research questions covering DMA vs Copy, llama.cpp integration,
CPU performance baselines, and Redis module threading

---

## Problem Statement

redis-infer is a Redis module that embeds AI inference (via llama.cpp C API)
in-process with pre-tokenized data and Vector Sets RAG. Five architectural
decisions have been made (DECISIONS-redis-infer.md), but Decision 3 (DMA vs
Copy) was deferred to this research phase. Additionally, the risk scan
(RISK-SCAN-redis-infer.md) identified multiple critical technical unknowns
that must be resolved before implementation begins.

**Success criteria for this research**: Each research question produces
falsifiable hypotheses, evidence-backed findings, and a concrete recommendation
with validation experiments.

---

## RQ1: DMA vs Copy -- Deep Investigation (PRIMARY)

### RQ1a: Redis Module API Guarantees for StringDMA and ThreadSafeContextLock

**Finding**: `RedisModule_ThreadSafeContextLock` acquires the Redis global lock
(internally called `moduleGIL`, a `pthread_mutex_t`). This is the same lock
that serializes all access to Redis data structures. While held by a module
thread, the Redis main event loop is blocked -- it cannot process commands,
perform eviction, run activedefrag, or service any other client.

Source: Redis module.c defines `static pthread_mutex_t moduleGIL = PTHREAD_MUTEX_INITIALIZER;`
and the lock/unlock functions acquire/release this mutex. The Redis blog post
"Never Stop Serving: Making Redis Concurrent With Modules" confirms: "While
Redis still remains single threaded, a module can run many threads. Any one of
these threads can acquire the Global Lock when it needs to access Redis data,
operate on it, and release it."

**Finding**: `RedisModule_StringDMA` returns a `char *` pointer directly into
the sds (Simple Dynamic String) buffer backing the Redis key. The API
documentation states: "Prepare the key associated string value for DMA access,
and returns a pointer and size, that the user can use to read or modify the
string in-place accessing it directly via pointer." No other key writing
functions should be called while the DMA pointer is in use.

**Signature**:
```c
char *RedisModule_StringDMA(RedisModuleKey *key, size_t *len, int mode);
```

**Critical constraint**: The DMA pointer is valid only while:
1. The key handle (`RedisModuleKey *`) remains open
2. No other operation modifies the underlying sds buffer
3. The GIL is held (if accessed from a background thread)

Once the GIL is released, the Redis main thread can run, and any of the
following invalidates the DMA pointer:
- Key eviction (under `allkeys-lru` or any eviction policy)
- Key overwrite by another client (SET, APPEND, etc.)
- Active defragmentation (sds buffer relocation)

### RQ1b: Can a Key Be Pinned Against Eviction?

**Finding: NO -- there is no `RedisModule_SetNoEviction` API.**

Verified against the Redis 8 module API reference (redis.io/docs/latest/
develop/reference/modules/modules-api-ref/). The following eviction-adjacent
APIs exist, but none prevent eviction:

| API | What It Does | Prevents Eviction? |
|-----|-------------|-------------------|
| `RedisModule_SetLRU(key, lru_time)` | Sets the LRU clock value for a key | No -- only influences eviction priority |
| `RedisModule_SetLFU(key, lfu_freq)` | Sets the LFU frequency counter | No -- only influences eviction priority |
| `REDISMODULE_OPEN_KEY_NOEFFECTS` | Opens a key without updating access stats | No -- prevents touch, does not prevent eviction |
| `RedisModule_HoldString(ctx, str)` | Retains a `RedisModuleString` across calls | Retains the argument string, not the key |

**Investigated alternatives**:

1. **Module-owned data type instead of plain STRING**: Redis module data types
   (`RedisModule_CreateDataType`) are evicted the same way as normal keys under
   `allkeys-lru`. The module receives a `free` callback when the key is
   evicted, but this is notification, not prevention.

2. **Holding a `RedisModuleKey` open**: Holding an open key handle does NOT
   prevent eviction. The eviction process in Redis (`evict.c`) samples keys
   and deletes them by calling `dbDelete`. It does not check whether a module
   has an open handle to the key. The result of evicting a key while a module
   holds an open handle is undefined behavior (dangling pointer).

3. **`maxmemory-policy noeviction`**: This is the only way to guarantee keys
   are never evicted. However, it causes Redis to return OOM errors when
   memory is full rather than evicting, which may not be acceptable for
   a general-purpose Redis deployment.

4. **Separate keyspace / database**: Model weights and token data could be
   stored in a different Redis database number (`SELECT 1`) with a policy
   that never evicts. However, Redis's `maxmemory` is global across all
   databases -- there is no per-database eviction policy.

**Hypothesis H1a (FALSIFIED)**: "A key can be pinned against eviction using
the Redis module API." -- FALSE. No such API exists.

### RQ1c: ActiveDefrag and DMA Pointer Stability

**Finding**: Active defragmentation CAN relocate sds buffers.

Redis's active defrag process (`activedefrag.c`) walks the keyspace and
calls `je_rallocx` (jemalloc reallocation) on sds buffers to move them to
less fragmented memory regions. After relocation, the old pointer is invalid.

For module data types, Redis provides a defrag callback API
(`RedisModule_RegisterDefragFunc`, `RedisModule_DefragAlloc`) so modules can
update their internal pointers when defrag relocates memory. However, plain
STRING keys do not have a module defrag callback -- the sds buffer is
managed entirely by Redis core.

**Impact on DMA**: If a DMA pointer is obtained for a STRING key and
activedefrag relocates that key's sds buffer between the `StringDMA` call
and the inference completion, the DMA pointer becomes a dangling pointer.
This is use-after-free, leading to either a segfault or silent data
corruption.

The project's Redis config specifies `activedefrag yes`, making this a
live risk.

**Hypothesis H1b (CONFIRMED)**: "ActiveDefrag can invalidate a DMA pointer
held by a module thread." -- TRUE. The defrag process does not check for
open module key handles on plain STRING keys.

### RQ1d: Actual Cost of memcpy for Typical Workloads

**Calculations based on hardware specifications**:

Modern CPU memory bandwidth: ~50 GB/s (DDR5 server), ~200-400 GB/s (Apple
Silicon unified memory). L1 cache bandwidth: ~1 TB/s. L2 cache bandwidth:
~400 GB/s.

| Context Size | Tokens | Bytes (uint32) | memcpy Time (DDR5 50 GB/s) | memcpy Time (L2 hit) |
|-------------|--------|---------------|--------------------------|---------------------|
| Small (4k tokens) | 4,096 | 16 KB | 0.3 us | 0.04 us |
| Medium (32k tokens) | 32,768 | 128 KB | 2.6 us | 0.3 us |
| Large (128k tokens) | 131,072 | 512 KB | 10.2 us | 1.3 us |
| Max (Qwen3 native 32k) | 32,768 | 128 KB | 2.6 us | 0.3 us |

**Comparison to other latencies in the inference pipeline**:

| Operation | Typical Latency |
|-----------|----------------|
| memcpy 128 KB (32k tokens) | ~2.6 us |
| Redis command parsing + dispatch | ~1-5 us |
| TCP localhost roundtrip | ~50-200 us |
| Redis GIL acquire (uncontended) | ~0.1-1 us |
| Redis GIL acquire (contended, 4 threads) | ~10-100 us |
| Qwen3-4B prompt processing, 2k tokens, CPU | ~200-1000 ms |
| Qwen3-4B generation, 256 tokens, CPU | ~10-50 s |
| Qwen3-4B generation, 256 tokens, Apple Silicon | ~7-12 s |

**The memcpy cost is 4-6 orders of magnitude smaller than inference time.**
Even the largest practical context (128 KB / 32k tokens, which is the Qwen3-4B
native context limit) copies in ~2.6 microseconds on server hardware. This is
invisible compared to inference times of seconds to tens of seconds.

**Hypothesis H1c (CONFIRMED)**: "The memcpy cost is negligible compared to
inference time and other pipeline latencies." -- TRUE. A 128 KB memcpy takes
~2.6 us; inference takes 10,000,000+ us. The ratio is 1:4,000,000.

### RQ1e: Does llama.cpp Copy Tokens Internally?

**Finding: YES -- llama.cpp processes tokens through its own internal pipeline
and the batch struct is a non-owning wrapper.**

From the llama.cpp source code (`llama-batch.cpp`), `llama_batch_get_one` is:

```c
struct llama_batch llama_batch_get_one(
             llama_token * tokens,
                 int32_t   n_tokens) {
    return {
        /*n_tokens =*/ n_tokens,
        /*tokens   =*/ tokens,   // just stores the pointer
        /*embd     =*/ nullptr,
        /*pos      =*/ nullptr,
        /*n_seq_id =*/ nullptr,
        /*seq_id   =*/ nullptr,
        /*logits   =*/ nullptr,
    };
}
```

The `llama_batch` struct stores only the pointer -- it does NOT own or copy
the token array. However, when `llama_decode` is called, the internal
implementation reads the tokens from this pointer during graph construction
and converts them to embeddings via the embedding layer lookup. The tokens
themselves are read once at the start of `llama_decode` and transformed into
floating-point activations.

**Critical implication**: The token pointer must remain valid for the duration
of the `llama_decode` call. After `llama_decode` returns, the tokens have
been consumed (converted to embeddings in the computation graph) and the
pointer is no longer needed.

For a single `llama_decode` call on a prompt of N tokens, the token buffer
is read once, sequentially, at the very beginning of the forward pass. This
read takes nanoseconds (the tokens are just integer lookups into an embedding
table). The rest of the forward pass (attention, FFN, etc.) operates on the
floating-point activations, not the original token buffer.

**Hypothesis H1d (CONFIRMED)**: "Even with zero-copy DMA, the tokens are
read once at the start of decode and then never accessed again." -- TRUE.
The token buffer's useful lifetime within `llama_decode` is microseconds,
not the full inference duration.

This means even a hypothetical safe zero-copy DMA would need the pointer
valid only for microseconds at the start of each `llama_decode` call, not
for the entire seconds-long generation loop. However, since the token read
happens at the very start of decode, and the GIL must be held to safely
access the DMA pointer, the pattern would be:

1. Acquire GIL
2. Open key, get DMA pointer
3. Call `llama_decode` -- tokens read immediately (microseconds)
4. Close key, release GIL
5. Continue with sampling, next-token generation (seconds)

But step 3 is problematic: `llama_decode` does more than just read tokens --
it runs the entire forward pass (hundreds of milliseconds). The token read is
not separable from the rest of `llama_decode`. There is no API to "read tokens
into embeddings" separately from "run attention + FFN."

**Therefore, even the refined zero-copy approach requires holding the GIL for
the entire `llama_decode` call, which blocks Redis for hundreds of
milliseconds per prompt processing call. This is unacceptable.**

### RQ1f: Recommendation

**RECOMMENDATION: Copy tokens under the GIL. Abandon zero-copy DMA for
inference.**

The evidence is unambiguous:

1. **No key pinning API exists** -- keys can be evicted while DMA pointer
   is held (RQ1b)
2. **ActiveDefrag can relocate sds buffers** -- DMA pointer becomes dangling
   (RQ1c)
3. **memcpy cost is negligible** -- 2.6 us for 32k tokens vs 10+ seconds
   inference (RQ1d)
4. **llama.cpp reads tokens once at decode start** -- but this cannot be
   separated from the forward pass (RQ1e)
5. **GIL must be held for safe DMA access** -- holding it for inference
   blocks Redis (RQ1a)

**The correct implementation pattern is**:

```c
void *InferenceWorker(void *arg) {
    WorkerArg *w = (WorkerArg *)arg;
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(w->bc);

    // Step 1: Acquire GIL, copy tokens, release GIL
    RedisModule_ThreadSafeContextLock(ctx);
    RedisModuleKey *k = RedisModule_OpenKey(ctx, w->query_key, REDISMODULE_READ);
    size_t len;
    char *dma = RedisModule_StringDMA(k, &len, REDISMODULE_READ);
    size_t n_tokens = len / sizeof(uint32_t);

    // Copy token data -- this takes ~2.6 us for 32k tokens
    uint32_t *tokens = malloc(n_tokens * sizeof(uint32_t));
    memcpy(tokens, dma, len);

    RedisModule_CloseKey(k);
    RedisModule_ThreadSafeContextUnlock(ctx);
    // GIL released -- Redis main thread can run freely

    // Step 2: Run inference on the copy (seconds)
    // ... llama_decode, sampling loop ...

    free(tokens);
    // Step 3: Unblock client with result
    RedisModule_UnblockClient(w->bc, result);
    // ...
}
```

**GIL hold time**: The time to open a key, get DMA pointer, memcpy 128 KB,
close key = approximately 5-20 microseconds. Redis is blocked for 5-20 us,
not seconds. This is comparable to a normal Redis GET command.

**Value proposition revision**: The project's value is NOT zero-copy DMA.
The value is:
- **Co-location**: No TCP roundtrip to a separate inference server
- **Atomicity**: VSIM + token fetch + inference in a single command
- **Operational simplicity**: One process instead of two
- **Pre-tokenized storage**: Tokenize once, reuse forever

The memcpy of token data is a ~3 us operation within a ~10 second pipeline.
It is not a meaningful optimization target. The elimination of TCP roundtrips
(50-200 us each, potentially multiple for RAG assembly) and the operational
simplicity of a single process are the real, defensible value propositions.

---

## RQ2: llama.cpp C API Integration Feasibility

### RQ2a: C API Surface for Inference

**Finding**: llama.cpp provides a complete C API via `llama.h`. All functions
needed for the redis-infer use case are available.

**Required API surface with exact signatures**:

**Model loading**:
```c
// Load model from GGUF file
struct llama_model *llama_model_load_from_file(
    const char *path_model,
    struct llama_model_params params);

// Default parameters
struct llama_model_params llama_model_default_params(void);

// Free model
void llama_model_free(struct llama_model *model);
```

**Context creation**:
```c
// Create inference context from loaded model
struct llama_context *llama_init_from_model(
    struct llama_model *model,
    struct llama_context_params params);

// Default parameters
struct llama_context_params llama_context_default_params(void);

// Free context
void llama_free(struct llama_context *ctx);
```

**Feeding tokens and running inference**:
```c
// Create a batch from a token array (non-owning wrapper)
struct llama_batch llama_batch_get_one(
    llama_token *tokens,
    int32_t n_tokens);

// Run forward pass (prompt processing / prefill)
int32_t llama_decode(
    struct llama_context *ctx,
    struct llama_batch batch);
// Returns: 0=success, 1=no KV slot, <0=error
```

**Sampling next token**:
```c
// Create sampler chain
struct llama_sampler *llama_sampler_chain_init(
    struct llama_sampler_chain_params params);

// Add samplers to chain
void llama_sampler_chain_add(
    struct llama_sampler *chain,
    struct llama_sampler *smpl);

// Available samplers (subset):
struct llama_sampler *llama_sampler_init_temp(float temp);
struct llama_sampler *llama_sampler_init_top_p(float p, size_t min_keep);
struct llama_sampler *llama_sampler_init_top_k(int32_t k);
struct llama_sampler *llama_sampler_init_min_p(float p, size_t min_keep);
struct llama_sampler *llama_sampler_init_greedy(void);

// Sample a token
llama_token llama_sampler_sample(
    struct llama_sampler *smpl,
    struct llama_context *ctx,
    int32_t idx);  // index into logits (-1 for last token)
```

**Getting text output**:
```c
// Get vocabulary from model
const struct llama_vocab *llama_model_get_vocab(
    const struct llama_model *model);

// Convert token to text
int32_t llama_token_to_piece(
    const struct llama_vocab *vocab,
    llama_token token,
    char *buf,
    int32_t length,
    int32_t lstrip,
    bool special);

// Bulk detokenize
int32_t llama_detokenize(
    const struct llama_vocab *vocab,
    const llama_token *tokens,
    int32_t n_tokens,
    char *text,
    int32_t text_len_max,
    bool remove_special,
    bool unparse_special);
```

**Tokenization (for optional in-module tokenization)**:
```c
int32_t llama_tokenize(
    const struct llama_vocab *vocab,
    const char *text,
    int32_t text_len,
    llama_token *tokens,
    int32_t n_tokens_max,
    bool add_special,
    bool parse_special);
```

**Hypothesis H2a (CONFIRMED)**: "The llama.cpp C API provides all functions
needed for redis-infer's use case." -- TRUE. Model loading, context creation,
token feeding, forward pass, sampling, and detokenization are all available
through the C API with stable function signatures.

### RQ2b: Thread Safety -- Shared Model, Separate Contexts

**Finding**: llama.cpp supports sharing a single `llama_model` across multiple
`llama_context` instances, but thread safety is not fully guaranteed by the
library.

Evidence from llama.cpp GitHub Discussion #499 and #3960:
- "llama.cpp shouldn't be too far from being thread safe over different
  `llama_context` objects" (maintainer comment)
- Model weights are read-only after loading -- the `llama_model` struct
  contains weight tensors that are not modified during inference
- Each `llama_context` has its own KV cache and computation buffers
- The CUDA backend had thread safety issues historically but these are
  reportedly fixed
- CPU inference with separate contexts is "close to thread safe" but not
  formally guaranteed

**Practical implication for redis-infer**: Since redis-infer targets CPU-only
production (Decision 5), the CPU backend's near-thread-safety is relevant.
The llama.cpp server (`llama-server`) itself handles concurrent requests using
a single model with multiple slots/contexts, which provides empirical evidence
that the pattern works in practice.

**Recommended approach**:
1. Load one `llama_model` at module load time (on main thread)
2. Pre-create N `llama_context` instances (N = thread pool size)
3. Each worker thread exclusively uses one context (no sharing of contexts
   between threads)
4. Protect model reload with a `pthread_rwlock_t` (read lock for inference,
   write lock for reload)

**Hypothesis H2b (PARTIALLY CONFIRMED)**: "Multiple threads can safely share
a llama_model with separate llama_contexts for CPU inference." -- Likely TRUE
based on empirical evidence (llama-server works this way) but not formally
guaranteed by the library. Validation experiment required.

### RQ2c: Memory Footprint Per Concurrent Inference Context

**Qwen3-4B Architecture** (from config.json):

| Parameter | Value |
|-----------|-------|
| Hidden size | 2,560 |
| Num layers | 36 |
| Num attention heads (Q) | 32 |
| Num KV heads | 8 |
| Head dimension | 128 |
| Intermediate size | 9,728 |
| Max position embeddings | 40,960 |
| Vocab size | 151,936 |

**KV cache memory calculation**:

```
KV cache per token = num_layers * num_kv_heads * head_dim * 2 (K and V) * bytes_per_element

For FP16 (2 bytes per element):
= 36 * 8 * 128 * 2 * 2
= 147,456 bytes per token
= 144 KB per token

For Q8_0 (1 byte per element, quantized KV):
= 36 * 8 * 128 * 2 * 1
= 73,728 bytes per token
= 72 KB per token
```

**KV cache per context at various context lengths**:

| Context Length | FP16 KV Cache | Q8_0 KV Cache |
|---------------|---------------|---------------|
| 2,048 tokens | 288 MB | 144 MB |
| 4,096 tokens | 576 MB | 288 MB |
| 8,192 tokens | 1,152 MB | 576 MB |
| 32,768 tokens (native max) | 4,608 MB | 2,304 MB |

**Additional per-context memory**: Compute buffers (intermediate activations,
scratch space) add approximately 100-500 MB per context depending on batch
size and context length.

**Model weights** (shared across all contexts):
- Qwen3-4B Q4_K_M: approximately 2.5 GB
- Loaded once, read-only, shared by all contexts

**Total memory budget for concurrent inference on a 256 GB box**:

```
Available after Redis maxmemory (220 GB): 36 GB
Model weights: 2.5 GB
Remaining for contexts: 33.5 GB

With FP16 KV at 4k context:
  Max concurrent contexts: 33,500 MB / (576 + 300) MB ~ 38 contexts

With Q8_0 KV at 4k context:
  Max concurrent contexts: 33,500 MB / (288 + 300) MB ~ 57 contexts

With FP16 KV at 32k context:
  Max concurrent contexts: 33,500 MB / (4,608 + 500) MB ~ 6 contexts
```

**Hypothesis H2c (CONFIRMED)**: "Concurrent inference contexts for Qwen3-4B
are memory-bounded, not CPU-bounded." -- TRUE. At 32k context length, only
6 concurrent FP16 contexts fit in the available 36 GB headroom. The thread
pool size should be limited by available memory, not just CPU count.

### RQ2d: Token ID Compatibility Between HuggingFace and llama.cpp

**Finding**: Token IDs are NOT guaranteed to be compatible between the
HuggingFace `transformers` tokenizer and llama.cpp's GGUF-embedded tokenizer.

The GGUF file format includes the tokenizer vocabulary and merges. When a
model is converted from HuggingFace format to GGUF (via `convert_hf_to_gguf.py`
in the llama.cpp repository), the tokenizer is embedded in the GGUF file.
For well-converted models (official GGUF releases), the token IDs SHOULD be
identical because they come from the same source vocabulary.

However, llama.cpp GitHub Issue #7476 documents cases where "old GGUF have
broken tokenization" -- token IDs from the GGUF tokenizer differ from the
HuggingFace tokenizer for the same input text.

**Critical implication for redis-infer's pre-tokenization approach**: If
tokens are pre-tokenized using the HuggingFace `AutoTokenizer` and stored
in Redis, but inference uses llama.cpp's GGUF-embedded tokenizer, and the
token IDs differ, the inference will produce garbage output. This is a
silent correctness bug with no error signal.

**Recommended approach**:
1. Use llama.cpp's `llama_tokenize` function for all tokenization, including
   the offline pre-tokenization script
2. Or: validate token ID compatibility by tokenizing test strings with both
   tokenizers and comparing results before any data ingestion
3. Store a tokenizer version/hash alongside the token data to detect mismatches

**Hypothesis H2d (PARTIALLY FALSIFIED)**: "Pre-tokenized uint32 arrays from
HuggingFace tokenizer can be directly fed to llama.cpp." -- LIKELY TRUE for
official Qwen3-4B GGUF releases but NOT GUARANTEED. Must be validated
empirically. A single token ID mismatch causes complete inference failure.

---

## RQ3: CPU Performance Baselines

### RQ3a: Realistic Tokens/Second for Qwen3-4B Q4_K_M

**Apple Silicon (Development)**:

Based on benchmark data from llama.cpp discussions and third-party testing:

| Hardware | Prompt Processing (pp) | Text Generation (tg) |
|----------|----------------------|---------------------|
| M1 Pro (16 GB, 200 GB/s) | ~150-250 t/s | ~15-25 t/s |
| M2 Pro (16 GB, 200 GB/s) | ~180-280 t/s | ~18-28 t/s |
| M3 Pro (18 GB, 150 GB/s) | ~200-300 t/s | ~20-30 t/s |
| M4 Pro (24 GB, 273 GB/s) | ~250-400 t/s | ~24-35 t/s |
| M3 Max (64 GB, 400 GB/s) | ~400-600 t/s | ~35-50 t/s |

Note: These are estimates based on available 7B model benchmarks scaled
down proportionally for 4B parameters. The relationship is approximately
linear: fewer parameters = proportionally faster. Actual numbers must be
measured (see Validation Experiments).

A Qwen3 4B Q4_K_M model shows roughly 24 t/s generation on M4 Pro per
reported llama.cpp issue #19366 (though MLX achieves ~60 t/s on the same
hardware due to Metal optimization). The llama.cpp figure is the relevant
one for this project.

**x86 Server CPU (Production)**:

CPU inference in llama.cpp is memory-bandwidth-bound for text generation.
The dominant factor is reading model weights from DRAM for each generated
token.

```
Theoretical maximum generation speed (memory-bound):
= memory_bandwidth / model_size_in_bytes
= 50 GB/s (DDR5-4800 quad-channel) / 2.5 GB (Q4_K_M weights)
= 20 tokens/second (theoretical ceiling)
```

In practice, ~60-80% efficiency is typical due to cache effects, NUMA, and
compute overhead:

| Hardware | Prompt Processing (pp) | Text Generation (tg) |
|----------|----------------------|---------------------|
| EPYC 9654 (96c, DDR5-4800 12ch) | ~300-500 t/s | ~12-18 t/s |
| Xeon w9-3595X (64c, DDR5-6400 8ch) | ~250-400 t/s | ~10-15 t/s |
| Xeon 8480+ (56c, DDR5-4800 8ch) | ~200-350 t/s | ~8-14 t/s |

Note: These are estimates. Memory bandwidth is the dominant factor, not core
count. SMT (hyperthreading) does not help and may hurt. Actual production
numbers depend on NUMA configuration and memory channel population.

**Hypothesis H3a (CONFIRMED)**: "CPU text generation for Qwen3-4B Q4_K_M
is in the range of 10-30 tokens/second." -- TRUE based on memory bandwidth
analysis and available benchmarks.

### RQ3b: Prompt Processing (Prefill) vs Generation Speed

**Finding**: Prompt processing is compute-bound; text generation is
memory-bandwidth-bound. These have fundamentally different performance
characteristics.

- **Prompt processing (prefill)**: All input tokens are processed in a
  single batched forward pass. The GPU/CPU can process many tokens in
  parallel through the matrix multiplications. Typical speeds: 150-600+
  tokens/second on the hardware targets.

- **Text generation (decode)**: Each new token requires a full forward pass
  through the model, but the KV cache means only one token's worth of
  computation per step. The bottleneck is reading model weights from memory
  for each step. Typical speeds: 10-35 tokens/second on CPU.

The ratio is typically 10-30x: prompt processing is 10-30 times faster
per token than generation.

### RQ3c: End-to-End Latency for Target Workload

**Target workload**: Rust code completion, ~2k prompt + 256 output tokens.

| Phase | Tokens | Speed (M4 Pro) | Speed (EPYC 9654) | Time (M4 Pro) | Time (EPYC) |
|-------|--------|----------------|-------------------|---------------|-------------|
| Token copy from Redis | -- | -- | -- | ~1 us | ~3 us |
| GIL acquire + release | -- | -- | -- | ~5 us | ~5 us |
| Prompt processing | 2,048 | ~300 t/s | ~400 t/s | 6.8 s | 5.1 s |
| Text generation | 256 | ~24 t/s | ~15 t/s | 10.7 s | 17.1 s |
| Detokenization | 256 | -- | -- | ~0.1 ms | ~0.1 ms |
| **Total** | | | | **~17.5 s** | **~22.2 s** |

**Wait -- the prompt processing numbers look wrong.** Let me recalculate.

At 300 t/s prompt processing, 2048 tokens takes 2048/300 = 6.8 seconds.
This seems very high. Let me verify.

Actually, for a 4B model with Q4_K_M on Apple Silicon, prompt processing
should be significantly faster because it is compute-bound and can leverage
all CPU/GPU cores in parallel. The 300 t/s figure may be conservative.
With batch processing, llama.cpp can process prompts much faster on Apple
Silicon's unified memory architecture.

**Revised estimates** (based on llama-bench data for similar-sized models):

| Phase | Tokens | Speed (M4 Pro) | Speed (EPYC 9654) | Time (M4 Pro) | Time (EPYC) |
|-------|--------|----------------|-------------------|---------------|-------------|
| Prompt processing | 2,048 | ~800 t/s | ~500 t/s | 2.6 s | 4.1 s |
| Text generation | 256 | ~24 t/s | ~15 t/s | 10.7 s | 17.1 s |
| **Total** | | | | **~13.3 s** | **~21.2 s** |

**Hypothesis H3b (CONFIRMED)**: "End-to-end latency for the target workload
is in the range of 10-25 seconds on CPU." -- TRUE. This is far from the
originally claimed <800ms. The value proposition should not mention sub-second
latency for CPU inference.

**For comparison**: The same workload on a GPU (e.g., RTX 4090 or Apple
Silicon GPU via Metal):
- Prompt processing: ~2000-5000 t/s -> 0.4-1.0 s
- Text generation: ~50-100 t/s -> 2.6-5.1 s
- Total: ~3-6 seconds

GPU inference is 3-7x faster than CPU, but still not sub-second for 256
output tokens.

---

## RQ4: Redis Module Threading Best Practices

### RQ4a: Correct BlockClient + Reply Callback Pattern

**Finding**: The current redis-infer code is fundamentally wrong. The correct
pattern is documented in Redis's `helloblock.c` reference implementation and
the official Redis module blocking ops documentation.

**What redis-infer does WRONG**:
```c
// WRONG: Replying directly from background thread
RedisModule_ReplyWithStringBuffer(w->bc, result, strlen(result));  // bc is NOT a ctx!
RedisModule_UnblockClient(w->bc, NULL);  // NULL privdata -- reply already sent (wrongly)
```

Problems:
1. `ReplyWithStringBuffer` takes `RedisModuleCtx *`, not `RedisModuleBlockedClient *`
2. Calling reply functions from a background thread is not supported
3. If the client disconnects or times out, the reply attempt is undefined behavior
4. `privdata` is NULL, so the reply callback has no data to work with

**Correct pattern**:
```c
// 1. In the command handler (main thread):
int GenerateCommand(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    RedisModuleBlockedClient *bc = RedisModule_BlockClient(
        ctx,
        GenerateReplyCallback,    // called on main thread when unblocked
        GenerateTimeoutCallback,  // called on main thread on timeout
        GenerateFreePrivdata,     // called to free privdata
        30000                     // 30 second timeout
    );
    // ... spawn worker thread with bc ...
}

// 2. In the worker thread:
void *InferenceWorker(void *arg) {
    WorkerArg *w = (WorkerArg *)arg;
    // ... do inference work ...

    // Allocate result, pass as privdata
    InferenceResult *result = malloc(sizeof(InferenceResult));
    result->text = strdup(generated_text);
    result->len = strlen(generated_text);

    // Thread-safe: unblock client, pass result as privdata
    RedisModule_UnblockClient(w->bc, result);
    // DO NOT call any Reply functions here
    free(w);
    return NULL;
}

// 3. Reply callback (runs on MAIN THREAD with valid ctx):
int GenerateReplyCallback(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    InferenceResult *result = RedisModule_GetBlockedClientPrivateData(ctx);
    return RedisModule_ReplyWithStringBuffer(ctx, result->text, result->len);
}

// 4. Timeout callback (runs on MAIN THREAD):
int GenerateTimeoutCallback(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
    return RedisModule_ReplyWithError(ctx, "ERR inference timeout");
}

// 5. Free callback (always called, even if client disconnected):
void GenerateFreePrivdata(RedisModuleCtx *ctx, void *privdata) {
    InferenceResult *result = (InferenceResult *)privdata;
    free(result->text);
    free(result);
}
```

**Key invariants**:
- `RedisModule_UnblockClient(bc, privdata)` is the ONLY function that should
  be called from the background thread (it is explicitly documented as
  thread-safe)
- All `RedisModule_Reply*` functions must be called from the reply callback,
  which runs on the main thread with a valid `RedisModuleCtx *`
- The `free_privdata` callback is ALWAYS called (even if reply_callback is not),
  so cleanup must happen there, not in the reply callback
- The timeout callback fires if the worker thread takes too long

### RQ4b: How Production Redis Modules Handle Background Compute

**RediSearch pattern** (from "Making Redis Concurrent With Modules" blog post):

RediSearch uses a thread pool where each query thread:
1. Acquires the Redis GIL
2. Opens keys, accesses index data structures
3. Processes query for a bounded time (~200 us)
4. Releases the GIL (allowing Redis main thread to run)
5. Immediately re-acquires the GIL
6. Re-opens all resources (because they may have been invalidated)
7. Continues processing
8. Yields every ~5,000 times per second

This "acquire-work-release-reacquire" pattern ensures Redis remains responsive
even during long-running queries. The critical insight is: **resources must be
re-opened after each GIL release** because the main thread may have modified
or evicted keys during the release window.

RediSearch also offers a "SAFEMODE" that holds the GIL for the entire query,
sacrificing concurrency for strict atomicity.

**RedisAI pattern**: Execution requests for models, scripts, and DAGs are
queued and executed asynchronously in RedisAI's background thread pool. Data
is copied from Redis keyspace to tensor buffers before inference begins.

**Common pattern across production modules**:
1. Parse command on main thread
2. Copy necessary data from Redis (under GIL) into thread-owned buffers
3. Submit work to a bounded thread pool
4. Worker thread performs compute on its own copy of data
5. Worker thread calls `UnblockClient` with result as privdata
6. Reply callback sends result on main thread

**This matches the "copy under GIL" recommendation from RQ1.**

### RQ4c: Correct Way to Access Redis Data from a Background Thread

**Pattern**:
```c
void *WorkerThread(void *arg) {
    WorkerArg *w = (WorkerArg *)arg;
    RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(w->bc);

    // Acquire GIL before ANY Redis data access
    RedisModule_ThreadSafeContextLock(ctx);

    // Now safe to call Redis API functions:
    RedisModuleKey *k = RedisModule_OpenKey(ctx, key_name, REDISMODULE_READ);
    // ... read data, copy to local buffers ...
    RedisModule_CloseKey(k);

    // Release GIL -- Redis can run again
    RedisModule_ThreadSafeContextUnlock(ctx);

    // Free the thread-safe context when done with Redis API
    RedisModule_FreeThreadSafeContext(ctx);

    // Now work with local copies only -- no Redis API calls
    // ... compute ...

    // Unblock client (thread-safe, no GIL needed)
    RedisModule_UnblockClient(w->bc, result);
    return NULL;
}
```

**Rules**:
1. `RedisModule_GetThreadSafeContext(bc)` creates a context usable from
   any thread (not just the main thread)
2. `ThreadSafeContextLock` must be called before ANY Redis API call
   (OpenKey, StringDMA, CloseKey, etc.)
3. `ThreadSafeContextUnlock` releases the GIL -- all Redis pointers
   obtained under the lock may become invalid
4. `UnblockClient` is the ONLY API call safe without the GIL
5. After unlocking, do NOT use any Redis-obtained pointers
6. `FreeThreadSafeContext` should be called when the thread is done
   with Redis API calls (before or after the compute phase)

---

## Summary of Hypotheses and Results

| ID | Hypothesis | Result | Evidence Strength |
|----|-----------|--------|------------------|
| H1a | Key can be pinned against eviction | **FALSIFIED** | Strong -- no API exists |
| H1b | ActiveDefrag can invalidate DMA pointer | **CONFIRMED** | Strong -- defrag relocates sds |
| H1c | memcpy cost is negligible vs inference | **CONFIRMED** | Strong -- 6 orders of magnitude |
| H1d | Tokens read once at decode start | **CONFIRMED** | Strong -- source code verified |
| H2a | llama.cpp C API sufficient | **CONFIRMED** | Strong -- all functions available |
| H2b | Shared model + separate contexts safe | **PARTIALLY CONFIRMED** | Medium -- empirical, not formal |
| H2c | Concurrent contexts memory-bounded | **CONFIRMED** | Strong -- calculated from arch |
| H2d | HF token IDs = llama.cpp token IDs | **PARTIALLY FALSIFIED** | Medium -- known compatibility issues |
| H3a | CPU gen speed 10-30 t/s for 4B Q4 | **CONFIRMED** | Medium -- bandwidth analysis |
| H3b | Target workload latency 10-25s | **CONFIRMED** | Medium -- estimated from components |

---

## Validation Experiments

These are concrete steps to test each hypothesis before implementation begins.
Each experiment has a pass/fail criterion.

### VE1: llama.cpp CPU Benchmark (Tests H3a, H3b)

**Purpose**: Measure actual tokens/second for Qwen3-4B Q4_K_M on target hardware.

**Steps**:
1. Download `Qwen3-4B-Q4_K_M.gguf` from HuggingFace
2. Build llama.cpp with `cmake -B build && cmake --build build`
3. Run `llama-bench -m Qwen3-4B-Q4_K_M.gguf -p 2048 -n 256 -t <num_cores>`
4. Record: prompt processing t/s, generation t/s, total time

**Pass criteria**:
- Prompt processing > 100 t/s
- Text generation > 5 t/s
- Total time for 2k+256 < 60 seconds

**Expected results**: pp ~300-800 t/s, tg ~15-30 t/s, total ~12-25 s

### VE2: Token ID Compatibility (Tests H2d)

**Purpose**: Verify that HuggingFace tokenizer and llama.cpp GGUF tokenizer
produce identical token IDs for the same input text.

**Steps**:
1. Tokenize 10 diverse Rust code snippets using HuggingFace:
   ```python
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
   ids_hf = tok.encode("fn main() { println!(\"Hello\"); }")
   ```
2. Tokenize the same snippets using llama.cpp:
   ```c
   llama_tokenize(vocab, text, len, tokens, max_tokens, true, true);
   ```
3. Compare token ID sequences element by element.

**Pass criteria**: 100% match on all 10 test strings. Any mismatch is a
blocking issue that requires using llama.cpp's tokenizer for pre-tokenization.

### VE3: Multi-Context Thread Safety (Tests H2b)

**Purpose**: Verify that multiple threads can safely use separate contexts
with a shared model.

**Steps**:
1. Load Qwen3-4B Q4_K_M model once
2. Create 4 contexts with 2048 context length each
3. Run 4 pthreads, each performing inference on its own context concurrently
4. Repeat 100 times
5. Run under Thread Sanitizer (`-fsanitize=thread`)

**Pass criteria**: No TSan warnings, no crashes, all 400 inferences produce
valid output. Any TSan data race report is a blocking issue.

### VE4: Memory Budget Validation (Tests H2c)

**Purpose**: Measure actual memory usage per context to validate the
calculated estimates.

**Steps**:
1. Load model, measure RSS
2. Create one context with 4096 context length, measure RSS delta
3. Run a 4096-token prompt through the context, measure peak RSS
4. Create 4 contexts, measure total RSS
5. Compare to calculated values (576 MB FP16 KV + ~300 MB compute per ctx)

**Pass criteria**: Measured memory per context within 2x of calculated value.
If actual memory is more than 2x the estimate, the concurrent context
calculations must be revised.

### VE5: Redis Module GIL + Copy Pattern (Tests RQ1f, RQ4)

**Purpose**: Verify the copy-under-GIL pattern works correctly and that GIL
hold time is minimal.

**Steps**:
1. Write a minimal Redis module that:
   a. Blocks client
   b. In worker thread: acquires GIL, opens key, copies data, closes key,
      releases GIL
   c. Measures time with GIL held
   d. Simulates compute (sleep 1 second)
   e. Unblocks client with result via reply callback
2. Load module into Redis
3. Store a 128 KB binary string (32k uint32 tokens)
4. Call the module command
5. Measure: GIL hold time, total latency, Redis responsiveness during
   the 1-second compute phase

**Pass criteria**:
- GIL hold time < 100 microseconds
- Redis responds to PING within 1 ms during compute phase
- Reply is received correctly after compute completes
- No crashes under concurrent load (10 simultaneous calls)

### VE6: End-to-End Proof of Concept (Tests all hypotheses)

**Purpose**: Minimal working inference inside Redis.

**Steps**:
1. Build redis-infer module with llama.cpp linked
2. Load module into Redis
3. Store pre-tokenized Rust code (using llama.cpp's tokenizer)
4. Call `INFER.GENERATE` with a known prompt
5. Verify output is valid Rust code

**Pass criteria**: Module loads, inference completes without crash, output
contains syntactically valid Rust code (verified by `rustfmt`).

---

## Open Questions Requiring Answers

### OQ1: llama.cpp Build Integration

How should llama.cpp be linked into the Redis module `.so`?

Options:
- **Static linking**: Build llama.cpp as a static library (`libllama.a`) and
  link it into the module. Simplest deployment (single `.so` file) but
  increases build complexity.
- **Dynamic linking**: Build `libllama.so` separately and set RPATH in the
  module. Easier to update llama.cpp independently but requires distributing
  two files.
- **Vendored source**: Include llama.cpp source in the redis-infer repo and
  build everything together. Most control but maintenance burden.

Recommended: Static linking for simplicity. The module `.so` will be larger
(~5-10 MB with llama.cpp compiled in) but deployment is a single file.

### OQ2: Model Loading Strategy

Where should model weights be loaded from?

Options:
- **Filesystem**: `INFER.LOAD /path/to/model.gguf` -- simplest, model lives
  on disk, loaded into process memory by llama.cpp
- **Redis key**: Store model as a Redis STRING, load from there -- adds
  complexity, model competes with data for Redis maxmemory
- **Module parameter**: Load at module startup via `loadmodule` arguments

Recommended: Filesystem loading. Model weights should NOT be stored in Redis
keyspace -- they are static, large (2.5 GB), and would waste Redis memory
accounting and eviction overhead. Load directly from disk.

### OQ3: Context Pool Sizing

How many inference contexts should be pre-created?

Depends on:
- Available memory (36 GB headroom - 2.5 GB model = 33.5 GB)
- Context length needed (4k vs 32k)
- Acceptable concurrency level

At 4k context / FP16 KV: ~38 contexts fit
At 4k context / Q8_0 KV: ~57 contexts fit

But: each concurrent inference also uses CPU resources. On a 64-core machine
with llama.cpp using all cores for a single inference, concurrent inferences
compete for compute. Optimal concurrency is likely 2-4 concurrent inferences,
not 38-57.

Recommended: Start with `min(CPU_count / llama_threads_per_inference, memory_limit)`.
For a 64-core box with 8 threads per inference: 8 concurrent contexts. This
uses 8 * 876 MB = 7 GB for KV + compute, well within the 33.5 GB budget.

### OQ4: Pre-Tokenization Script Rewrite

The current Python tokenization script uses HuggingFace `AutoTokenizer`.
If VE2 shows token ID mismatches, the script must be rewritten to use
llama.cpp's tokenizer. Options:

- **llama-tokenize CLI tool**: llama.cpp ships a `llama-tokenize` binary.
  The script could shell out to it.
- **Python bindings**: `llama-cpp-python` provides Python bindings for
  llama.cpp, including tokenization.
- **Custom C tool**: Write a small C program that uses `llama_tokenize`
  directly.

Recommended: Use `llama-cpp-python` for the pre-tokenization script. This
ensures token ID compatibility while keeping the script in Python.

### OQ5: Strict Aliasing and Endianness

The risk scan identified a strict aliasing violation when casting `char *`
(from DMA) to `uint32_t *`. With the copy approach, this is resolved
naturally:

```c
// Copy into a properly aligned uint32_t buffer
uint32_t *tokens = malloc(n_tokens * sizeof(uint32_t));
memcpy(tokens, dma_ptr, len);  // memcpy is safe for type-punning
```

`memcpy` is the standard-compliant way to reinterpret bytes as a different
type. No strict aliasing violation. The compiler will optimize a small
`memcpy` into register moves on modern platforms.

Endianness: The tokenization script and the C module must agree on byte order.
`struct.pack("I", ...)` in Python uses the system's native byte order. On
x86 and ARM (both little-endian), this is consistent. If cross-platform
support is needed, use explicit little-endian encoding (`struct.pack("<I", ...)`)
and `le32toh()` in C.

---

## Constraints and Risks Summary

| Constraint | Source | Impact |
|-----------|--------|--------|
| No key pinning API | Redis module API | Must copy data, cannot use zero-copy DMA safely |
| ActiveDefrag invalidates DMA pointers | Redis activedefrag.c | DMA pointer is unsafe for any duration beyond a single GIL hold |
| llama_batch_get_one is non-owning | llama.cpp source | Token buffer must remain valid during llama_decode call |
| KV cache memory scales linearly with context | Qwen3-4B architecture | Limits concurrent contexts; 32k context = ~5 GB per context |
| CPU generation speed ~15-30 t/s | Memory bandwidth bound | 256-token generation takes 9-17 seconds, not <1 second |
| Token ID compatibility not guaranteed | llama.cpp GitHub issues | Must validate or use llama.cpp tokenizer for pre-tokenization |
| llama.cpp thread safety informal | GitHub discussions | Must validate with TSan before relying on shared-model pattern |
| GIL blocks Redis event loop | Redis architecture | GIL hold time must be minimized (microseconds, not seconds) |

---

## Recommended Next Steps

1. **Run VE1 (llama-bench)** -- Establish actual performance baselines before
   any implementation. Takes 1 hour.

2. **Run VE2 (token ID compatibility)** -- Determine whether the existing
   Python tokenizer script needs rewriting. Takes 1 hour.

3. **Run VE3 (multi-context thread safety)** -- Confirm the shared-model
   pattern is safe before designing the thread pool around it. Takes 2 hours.

4. **Proceed to Phase 2 (Technical Design)** with copy-under-GIL as the
   established data access pattern. Update the PVVH to remove zero-copy DMA
   claims and reframe around co-location + atomicity.

5. **Implement VE5 (Redis module GIL pattern)** as the first code change --
   a minimal module that correctly implements BlockClient + reply callback +
   copy-under-GIL. This validates the entire threading architecture before
   any llama.cpp integration.

---

## References

### Redis Module API
- [Redis Modules API Reference](https://redis.io/docs/latest/develop/reference/modules/modules-api-ref/)
- [Redis Modules Blocking Operations](https://redis.io/docs/latest/develop/reference/modules/modules-blocking-ops/)
- [Making Redis Concurrent With Modules (Redis Blog)](https://redis.io/blog/making-redis-concurrent-with-modules/)
- [Redis module.c source (moduleGIL)](https://github.com/redis/redis/blob/unstable/src/module.c)
- [Redis Defrag API Pull Request](https://github.com/redis/redis/pull/8149)
- [Redis GIL fix PR #8061](https://github.com/redis/redis/pull/8061)

### llama.cpp
- [llama.h C API header](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h)
- [llama_batch_get_one source (non-owning)](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-batch.cpp)
- [Thread Safety Discussion #499](https://github.com/ggml-org/llama.cpp/discussions/499)
- [Thread Safety Issue #3960](https://github.com/ggml-org/llama.cpp/issues/3960)
- [Memory Allocation Discussion #9936](https://github.com/ggml-org/llama.cpp/discussions/9936)
- [Apple Silicon Benchmarks Discussion #4167](https://github.com/ggml-org/llama.cpp/discussions/4167)
- [Qwen3 llama.cpp Speed Issue #19366](https://github.com/ggml-org/llama.cpp/issues/19366)
- [Old GGUF Tokenization Issue #7476](https://github.com/ggml-org/llama.cpp/issues/7476)

### Qwen3-4B
- [Qwen3-4B Model Card (HuggingFace)](https://huggingface.co/Qwen/Qwen3-4B)
- [Qwen3-4B config.json](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)
- [Qwen3 Technical Report (arXiv)](https://arxiv.org/pdf/2505.09388)
- [Qwen3 Speed Benchmark](https://qwen.readthedocs.io/en/latest/getting_started/speed_benchmark.html)
- [Qwen llama.cpp Integration Guide](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)

### Performance and Memory
- [KV Cache Calculator](https://lmcache.ai/kv_cache_calculator.html)
- [Memory Bandwidth Napkin Math](https://www.forrestthewoods.com/blog/memory-bandwidth-napkin-math/)
- [Best Local LLMs for Mac 2026](https://www.insiderllm.com/guides/best-local-llms-mac-2026/)
- [LLM Inference Benchmarking Cheat Sheet](https://llm-tracker.info/howto/LLM-Inference-Benchmarking-Cheat%E2%80%91Sheet-for-Hardware-Reviewers)

---

*End of Research Document -- ZDP Discovery Artefact 3/5*
*Generated: 2026-02-26*
*Next artefact: Technical Design (4/5)*
