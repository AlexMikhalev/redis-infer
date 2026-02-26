# Risk Register: redis-infer
## ZDP Discovery Artefact 2/5 -- Via Negativa Analysis

**Date**: 2026-02-26
**Author**: Via Negativa Analyst (ZDP Discovery)
**Status**: Discovery Stage
**Scope**: Full project scan -- code, architecture, process, security, strategic

---

## Executive Summary

This analysis goes deeper than the preliminary risk list (R1-R10). After
line-by-line examination of every project file, verification of API
correctness against Redis module source, and confirmation of what
pure-C inference engines actually exist versus what the Grok conversation
assumed, the finding is stark:

**The project currently contains no working code, is built on at least
three false technical assumptions, and the primary value mechanism
(zero-copy DMA used safely in a worker thread) is architecturally
impossible as implemented.**

The Grok conversation was productive for ideation but generated multiple
plausible-sounding falsehoods that will consume months of engineering time
if not corrected at Discovery stage.

---

## Risk Register

### Category 1: Technical -- C Code and Redis Module API

---

#### T1: CRITICAL API MISUSE -- ReplyWithStringBuffer Called on Wrong Type

**Severity**: Critical
**Likelihood**: Certain (code is demonstrably wrong)
**Impact**: Segfault or silent wrong behavior on every INFER.GENERATE call

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, lines 36-37):

```c
RedisModule_ReplyWithStringBuffer(w->bc, result, strlen(result));
RedisModule_UnblockClient(w->bc, NULL);
```

`RedisModule_ReplyWithStringBuffer` takes a `RedisModuleCtx *ctx` as its
first argument. `w->bc` is a `RedisModuleBlockedClient *`, not a context.
This is a type confusion: both are pointers, so the C compiler will accept
it without warning (no `-Wstrict-prototypes` equivalent catches opaque
pointer mismatches unless `redismodule.h` uses distinct typedefs, which it
does). At runtime the function will cast `bc` to `ctx`, then dereference
`ctx->module` at an offset that does not correspond to the blocked client
struct layout. The result is a segfault or memory corruption.

The correct pattern from Redis's own `helloblock.c` reference
implementation is:
1. Pass a reply callback to `RedisModule_BlockClient`
2. Store the result in `privdata` passed to `RedisModule_UnblockClient`
3. Redis calls the reply callback on the main thread with a valid `ctx`

As implemented, the module cannot produce a single successful reply without
crashing or corrupting Redis.

**Mitigation**: Rewrite `InferenceWorker` to store `result` as `privdata`,
register a proper reply callback at `BlockClient` time, and reply from
within that callback. This is a complete rewrite of the threading reply
path.

---

#### T2: CRITICAL THREADING VIOLATION -- OpenKey Called Without GIL

**Severity**: Critical
**Likelihood**: Certain (code is demonstrably wrong)
**Impact**: Data structure corruption, heap corruption, Redis crash

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, lines 26-28):

```c
RedisModuleKey *k = RedisModule_OpenKey(ctx, w->query_key, REDISMODULE_READ);
RedisModule_StringDMA(k, (void**)&data, &len, REDISMODULE_READ);
uint32_t *tokens = (uint32_t*)data;
```

The Redis module API documentation and source code (`module.c`) are
explicit: to call non-Reply APIs from a thread-safe context, the GIL
(Global Interpreter Lock / server lock) must be held:

```c
RedisModule_ThreadSafeContextLock(ctx);
/* ... OpenKey, StringDMA, CloseKey here ... */
RedisModule_ThreadSafeContextUnlock(ctx);
```

The current code calls `OpenKey` and `StringDMA` without acquiring the GIL.
Redis is single-threaded for data access; worker threads accessing data
structures without the lock race against the main event loop. Under any
concurrent load this produces undefined behavior: heap corruption,
use-after-free from eviction, or read of partially-modified data.

**Additional compounding factor**: The DMA pointer obtained via
`StringDMA` is only valid while the key is held open AND the server is not
running. Once the GIL is released (if it were held), the Redis main thread
can run, and any operation that modifies the string key (eviction,
overwrite, defrag) invalidates the DMA pointer. Holding a DMA pointer
across a 800ms inference call while other threads hold the GIL is not
architecturally safe -- the GIL would need to be held for the entire
inference duration, which defeats single-threaded Redis entirely.

**The zero-copy DMA claim is false as described.** Either the GIL is held
(defeating concurrency) or the DMA pointer is unsafely used (causing
crashes). There is no third option within the current Redis module API.
The correct approach is to copy the token data under the GIL, release the
GIL, then run inference on the copy. This eliminates the zero-copy
advantage, which is the primary stated value proposition.

**Mitigation**: Accept that true zero-copy inference is impossible in a
background thread with the current Redis module API. Copy tokens under the
GIL, run inference on the copy. Revise the value proposition accordingly.

---

#### T3: CRITICAL -- Global State with No Synchronization

**Severity**: Critical
**Likelihood**: Certain
**Impact**: Race condition on model load, use-after-free, silent corruption

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, lines 11, 61-62):

```c
static qwen_ctx_t *g_qwen = NULL;

int LoadCommand(...) {
    if (g_qwen) { /* free old context */ }  // EMPTY COMMENT -- no actual free
    g_qwen = qwen_load(...);
    ...
}
```

`g_qwen` is read in `InferenceWorker` (line 31) from a detached pthread
while potentially being written by `LoadCommand` on the main thread.
No mutex, no atomic operation, no memory barrier protects this access.

Secondary issue: `if (g_qwen) { /* free old context */ }` is a comment
with no code. Calling `INFER.LOAD_QWEN3` twice leaks the first model
context (2.5 GB) and leaves dangling pointers if any worker thread holds
a reference to the old context.

**Mitigation**: Add a `pthread_rwlock_t` guarding `g_qwen`. The reload
operation must drain all in-flight workers before freeing the old context.
This requires tracking active worker count, which the current design has
no mechanism for.

---

#### T4: HIGH -- Unbounded Thread Creation, No Backpressure

**Severity**: High
**Likelihood**: Certain under any concurrent load
**Impact**: Memory exhaustion, OOM kill of Redis process

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, lines 53-55):

```c
pthread_t tid;
pthread_create(&tid, NULL, InferenceWorker, w);
pthread_detach(tid);
```

Every `INFER.GENERATE` call creates a new detached thread. There is no
thread pool, no semaphore, no queue depth limit. Under 40-80 concurrent
requests (the stated performance target), 40-80 threads are created
simultaneously, each attempting to load Qwen3-4B weights into the forward
pass. At 2.5 GB model size, the weight data is shared but the activation
memory per inference call is substantial (KV cache alone for 128k context
at FP16 is multiple GB per call). The process will OOM before approaching
the claimed concurrent performance target.

`pthread_detach` also means there is no mechanism to drain threads before
`RedisModule_OnUnload`. If `MODULE UNLOAD` is called while inference
threads are running, the `.so` file is unmapped from memory while threads
are executing code in it, producing a segfault.

**Mitigation**: Implement a fixed-size thread pool with a bounded work
queue. Reject new requests with a queue-full error rather than spawning
infinitely. Implement a drain-and-wait mechanism in `OnUnload`. This is
a substantial implementation task, not a minor fix.

---

#### T5: HIGH -- malloc Failure Not Checked

**Severity**: High
**Likelihood**: Low under normal conditions, High under memory pressure
**Impact**: NULL dereference, segfault in Redis process

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, line 48):

```c
WorkerArg *w = malloc(sizeof(WorkerArg));
w->bc = bc;  // immediate dereference without NULL check
```

Under memory pressure (which is expected when Redis is configured with
`maxmemory 220gb` and the box has 256 GB total, leaving only 36 GB
headroom for OS, inference activations, and all worker thread stacks),
`malloc` can return NULL. The subsequent `w->bc = bc` dereferences NULL,
segfaulting Redis. This is the most basic C safety check that is absent.

**Mitigation**: Check `malloc` return. Use `RedisModule_Alloc` instead,
which Redis tracks for memory accounting and which will respect the
configured `maxmemory` limit rather than going to the OS.

---

#### T6: HIGH -- atoi for Token Count Without Validation

**Severity**: High
**Likelihood**: High (user-supplied input)
**Impact**: Negative max_tokens passed to inference, integer overflow, undefined behavior

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, line 51):

```c
w->max_tokens = atoi(RedisModule_StringPtrLen(argv[2], NULL));
```

`atoi` returns 0 on parse failure and does not signal error. It also
returns a signed int. Passing `max_tokens = 0` or `max_tokens = -1` to
the inference engine (whatever eventually replaces the stub) produces
undefined behavior in the generation loop. A client can pass
`"INFER.GENERATE mykey -9999999"` and the module accepts it silently.

**Mitigation**: Use `RedisModule_StringToLongLong` with error checking,
validate range (positive, below a sane maximum like 8192), reject invalid
input with a proper error reply.

---

#### T7: HIGH -- StringDMA Pointer Aliasing Violation

**Severity**: High
**Likelihood**: Certain (type system violation)
**Impact**: Undefined behavior under strict aliasing optimization

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, lines 25-29):

```c
uint8_t *data; size_t len;
RedisModuleKey *k = RedisModule_OpenKey(ctx, w->query_key, REDISMODULE_READ);
RedisModule_StringDMA(k, (void**)&data, &len, REDISMODULE_READ);
uint32_t *tokens = (uint32_t*)data;
size_t n = len / 4;
```

The buffer obtained via DMA is `char *` internally (Redis stores strings
as `char *`). Reinterpreting it as `uint32_t *` via a cast violates the
C strict aliasing rule. With `-O3` (as in the Makefile), the compiler
is permitted to assume that `char *` and `uint32_t *` do not alias, and
may reorder or eliminate reads/writes accordingly. Additionally, if the
token data was stored on a platform with a different byte order than
expected, or if the storing Python script and the reading C code disagree
on endianness, token IDs will be silently corrupted.

The Makefile uses `-O3` without `-fno-strict-aliasing`, making this a
live bug.

**Mitigation**: Use `memcpy` into a properly aligned `uint32_t` buffer,
or use `-fno-strict-aliasing` consistently, or store tokens with explicit
byte order and use `le32toh`/`be32toh` for conversion.

---

#### T8: MEDIUM -- pthread_create Failure Not Handled

**Severity**: Medium
**Likelihood**: Low under normal conditions, guaranteed under resource exhaustion
**Impact**: Blocked client is never unblocked; client hangs forever

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`, lines 53-55):

```c
pthread_t tid;
pthread_create(&tid, NULL, InferenceWorker, w);
pthread_detach(tid);
```

`pthread_create` returns non-zero on failure (EAGAIN when thread limit is
hit, ENOMEM when stack cannot be allocated). The return value is ignored.
If thread creation fails, `bc` is a blocked client that will never be
unblocked. The client hangs until timeout. Under the unbounded thread
creation regime of T4, this failure mode is guaranteed to be reached.

The reference implementation `helloblock.c` from the Redis source handles
this correctly with `RedisModule_AbortBlock(bc)`.

**Mitigation**: Check `pthread_create` return value. On failure, call
`RedisModule_AbortBlock(bc)` and free `w`. Return an error to the client.

---

#### T9: MEDIUM -- No OnUnload Handler

**Severity**: Medium
**Likelihood**: Certain on any MODULE UNLOAD or Redis shutdown
**Impact**: Memory leak, potential use-after-free if threads are still running

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/src/redis-infer.c`):

There is no `RedisModule_OnUnload` function. When the module is unloaded,
the model context (`g_qwen`) is never freed, any in-flight detached
threads continue executing code in the now-unmapped `.so` library, and
the `RedisModuleString` held in `w->query_key` (line 50) is leaked for
any threads that have been created but not yet started.

**Mitigation**: Implement `RedisModule_OnUnload` that frees the model
context. Implement a thread drain mechanism (see T4). This requires the
bounded thread pool.

---

### Category 2: Architectural -- Viability of Core Approach

---

#### A1: CRITICAL -- No Pure-C Text Generation Engine Exists for Qwen3-4B

**Severity**: Critical
**Likelihood**: Certain (verified via live GitHub API check)
**Impact**: Core functionality cannot be built; entire project is blocked

**Evidence**: Verification of antirez's GitHub repositories confirms:
- `voxtral.c`: speech-to-text (Mistral Voxtral Realtime 4B)
- `qwen-asr`: speech-to-text (Qwen3-ASR 0.6B/1.7B)
- `iris.c`: image generation (FLUX.2-klein-4B)
- `gte-pure-C`: embedding only (GTE Small)

None of these are text generation (LLM autoregressive) engines. The Grok
conversation assumed "Antirez has built voxtral.c, qwen-asr, iris.c in
pure C" implies he could or would build a Qwen3-4B text generation engine.
This is not a valid inference. Speech-to-text and image diffusion have
completely different computational graphs than transformer-based
autoregressive LLM generation.

As of 2026-02-26, there is no pure-C Qwen3-4B text generation engine in
antirez's repositories or anywhere else (verified search of GitHub). The
closest candidates -- `llama.c`, `gemma3.c`, `microgpt-c` -- are
experimental, incomplete, and not compatible with the Qwen3-4B GGUF
format without substantial porting work.

The project plan says "Day 2: replace stubs with real Qwen3 C core." This
is not a two-day task. It is a multi-month research engineering task, and
it may not be achievable at production quality by a single developer.
`llama.cpp` represents years of multi-contributor effort to reach its
current state of GGUF compatibility.

**Mitigation options** (ranked by feasibility):
1. Link `llama.cpp` as a C library (it has a C API via `llama.h`). This
   abandons "pure-C" but delivers working inference.
2. Wait for antirez to publish a text generation engine and contribute.
3. Accept the 6-18 month timeline to build a production-quality pure-C
   transformer inference engine from scratch.

The project vision must be revised to reflect which of these is chosen.
Without resolution, the project has no viable path from stub to working.

---

#### A2: CRITICAL -- The Zero-Copy DMA Value Proposition is Architecturally Invalid

**Severity**: Critical
**Likelihood**: Certain (follows from Redis module API constraints)
**Impact**: Primary differentiating value mechanism does not exist

The project's primary stated value is: "RedisModule_StringDMA gives the
inference engine a direct pointer to token data in Redis memory; no memcpy,
no serialization."

This is architecturally impossible as described. The reasons are:

1. `StringDMA` returns a pointer into Redis's internal sds (Simple Dynamic
   Strings) buffer. This pointer is only valid while the key is open AND
   the Redis event loop is not running (i.e., the GIL is held).

2. A Qwen3-4B inference call takes hundreds of milliseconds to seconds.
   Holding the GIL for this duration is equivalent to blocking Redis for
   the entire inference duration -- destroying Redis's availability for all
   other clients.

3. If the GIL is released during inference (the correct approach), any
   Redis operation that modifies the string (eviction under LRU policy,
   SET by another client, active defrag which is enabled in the config)
   invalidates the DMA pointer. Using the pointer after this point is
   use-after-free.

4. The config specifies `maxmemory-policy allkeys-lru` and
   `activedefrag yes`. Both of these will move or free memory backing
   DMA pointers while inference runs. This combination is specifically
   dangerous.

**The practical outcome**: the token data MUST be copied before inference
begins. The copy happens under the GIL (fast, bounded), the GIL is
released, and inference runs on the copy. This copy is a `memcpy` of
N * 4 bytes for N tokens. For a 10k-token context, this is 40 KB --
a single memcpy that completes in microseconds. The claimed "10-100us
elimination" is already this fast WITH a memcpy; the zero-copy DMA
provides no measurable benefit for this data size, and is unsafe.

**Mitigation**: Reframe the value proposition. The real benefit is
co-location (no TCP roundtrip, atomicity of RAG + generate). Eliminate
the zero-copy claim as a primary selling point. Add honest copy-then-infer
implementation.

---

#### A3: HIGH -- The Embedding Model Does Not Exist

**Severity**: High
**Likelihood**: Certain (verified)
**Impact**: `embed-and-store.py` fails immediately on execution

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/scripts/embed-and-store.py`, line 14):

```python
embedder = SentenceTransformer("antirez/gte-small-pure-c-converter")
```

Verification against the HuggingFace API confirms: model
`antirez/gte-small-pure-c-converter` does not exist. The API returns
`{"error": "Invalid username or password."}` which is the HuggingFace
API's way of indicating the model is not found or is private.

`antirez/gte-pure-C` is a C inference engine repository, not a
SentenceTransformers-compatible model on HuggingFace. These are
categorically different things. `SentenceTransformer("antirez/gte-pure-C")`
would also fail -- it expects a model directory with `config.json`,
tokenizer files, and model weights in the SentenceTransformers format.

Anyone running `embed-and-store.py` will get an immediate import-time
crash downloading a model that does not exist.

**Mitigation**: Replace with an actual SentenceTransformers model
(e.g., `Alibaba-NLP/gte-small` or `thenlper/gte-small`). Update
documentation accordingly.

---

#### A4: HIGH -- Key Collision Destroys Multi-Repository and Multi-File Correctness

**Severity**: High
**Likelihood**: Certain for any real codebase
**Impact**: Silent data corruption -- wrong token data served for inference

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/scripts/tokenize-and-store.py`, lines 29-30):

```python
store(f.replace("/", "_").replace(".", "_"), fh.read())
```

The key is derived from the filename only (e.g., `lib_rs`, `main_rs`,
`mod_rs`). Any Rust project with more than one `main.rs` across different
subdirectories will silently overwrite previous entries. A project with
`src/main.rs` and `examples/main.rs` stores only one under key
`tokens:main_rs`.

`embed-and-store.py` has the identical bug: the same `key` derivation is
used for the Vector Set element, meaning the embedding stored for
`codebase:embeddings/main_rs` corresponds to one file's content but the
`tokens_key` may point to a different file's tokens. RAG retrieval will
return semantically correct embeddings but incorrect token data for
generation -- silent inference corruption with no error signal.

**Mitigation**: Use the full relative path from the repository root as the
key, normalized to be Redis-key-safe. Hash collisions are still possible
but rare. This is a known preliminary risk (R6) but the silent corruption
consequence is more severe than described.

---

#### A5: HIGH -- Performance Claims Are Mathematically Impossible

**Severity**: High
**Likelihood**: Certain
**Impact**: Benchmark results will falsify the primary performance claim

The plan states: "40-80 concurrent Rust generations (<800 ms each)".

Analysis of why this is impossible without GPU:

- Qwen3-4B Q4 requires approximately 2.5 GB for weights.
- Each inference call requires KV cache memory proportional to context
  length. At 4k context, FP16 KV cache for Qwen3-4B is approximately
  1-2 GB per concurrent session.
- 40 concurrent calls with 4k context each: 40-80 GB KV cache alone,
  plus 2.5 GB shared weights, plus activations per layer.
- This approaches or exceeds available system RAM on a 256 GB box
  configured with `maxmemory 220gb` (leaving 36 GB for OS and module).

- CPU throughput for Qwen3-4B Q4 on a typical server CPU (not GPU):
  llama.cpp on a high-end 32-core server CPU generates approximately
  5-15 tokens/second for 4B models. At 256 tokens output per call,
  that is 17-51 seconds per call, not 800ms, for a single thread.

- 40-80 concurrent threads multiplying CPU contention makes this worse,
  not better. Pthreads do not parallelize a single forward pass; each
  thread runs its own forward pass competing for CPU cache and memory
  bandwidth.

The 800ms target is achievable with GPU inference (e.g., a single
NVIDIA A100 can do Qwen3-4B in <100ms). Without GPU, the number is
off by one to two orders of magnitude.

The Grok conversation never questioned this performance target. It was
stated as fact in the plan without derivation.

**Mitigation**: Benchmark `llama.cpp` with the same model on the target
hardware before writing any module code. Revise the performance claim based
on empirical data. Consider that the value proposition may need to shift
from "concurrent at scale" to "single-user, lower operational complexity."

---

#### A6: MEDIUM -- VADD Syntax in embed-and-store.py Likely Incorrect

**Severity**: Medium
**Likelihood**: High
**Impact**: Embedding script fails to store any vectors; RAG is non-functional

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/scripts/embed-and-store.py`, lines 24-32):

```python
r.execute_command(
    "VADD", "codebase:embeddings", key,
    "VECTOR", len(vec), *vec,
    "JSON", json.dumps({...}),
    "QUANT", "int8"
)
```

The Redis Vector Sets `VADD` command syntax (from Redis 8 source and
documentation) does not take `len(vec)` as a separate positional argument
before the vector components in the same way as some other Redis commands.
The correct syntax and argument ordering needs to be verified against the
current Redis 8 `VADD` specification. Additionally, the element identifier
and vector set name argument ordering may be reversed relative to what
different Redis versions expect. This has not been tested against a running
Redis 8 instance.

Furthermore, `sentence-transformers` `encode()` returns a numpy array;
`tolist()` converts it to Python floats. The number of arguments passed
to `execute_command` via `*vec` could be 384 individual float arguments,
which may exceed client-side argument limits depending on the redis-py
version and connection configuration.

**Mitigation**: Test the VADD syntax against a running Redis 8 instance.
Consult current Redis 8 VADD documentation for exact argument ordering.
Consider using a batch VADD approach for efficiency.

---

#### A7: MEDIUM -- No Chunking Means Unbounded Key Size and Unusable RAG

**Severity**: Medium
**Likelihood**: Certain for any real codebase
**Impact**: Individual files tokenized to 50k+ tokens make RAG useless and DMA unsafe

The tokenization script reads entire files as single strings. A large Rust
source file (common in production codebases) can be 5,000-50,000 tokens.
Storing a 50k-token file as a single Redis string means:

1. The Vector Set embedding is for the entire file -- HNSW similarity
   search will find "the file most similar to the query" not "the
   function most similar to the query." Retrieval granularity is file-level.

2. At inference time, the entire 50k-token context is assembled, which
   exceeds Qwen3-4B's context window (32k tokens). The inference engine
   must truncate, but there is no truncation logic in the current code.

3. A 50k uint32 binary string is 200 KB. The `StringDMA` call operates on
   this as a monolithic blob with no offset support.

**Mitigation**: Implement function-level or chunk-level splitting (by
function, by N-token window with overlap). This is R7 from the preliminary
list but the consequence is more severe than a missing feature -- without
chunking, the RAG component produces degraded results for any real-world
input.

---

#### A8: LOW -- maxmemory-policy allkeys-lru Evicts Model Weights

**Severity**: Low (depends on workload)
**Likelihood**: Medium under heavy data ingestion
**Impact**: Silently evicts model weights from Redis; next INFER.GENERATE returns ERR

The config specifies `maxmemory-policy allkeys-lru`. If Redis fills its
220 GB allocation, LRU eviction will evict any key including
`model:qwen3-4b-q4`. This evicts model weights without the module's
knowledge. The C-side `g_qwen` pointer becomes a dangling reference to
memory that Redis has freed. The next `qwen_generate` call dereferences
freed memory.

The module needs to either: use `RedisModule_NoEviction` flags on model
weight keys, or use Redis module data types (not plain strings) for model
storage so that the module is notified on eviction, or store the model
outside of Redis's managed keyspace entirely.

**Mitigation**: Mark model weight keys with `OBJECT NOTOUCH` or use the
module API to mark keys as not evictable. Alternatively, load model
weights from disk into process memory (not Redis keyspace) and keep them
separate from the managed keyspace.

---

### Category 3: Process -- Development Risks

---

#### P1: CRITICAL -- "Day 2: Replace Stubs" Is a Fiction

**Severity**: Critical
**Likelihood**: Certain
**Impact**: Project stalls indefinitely at stub phase; months of wasted effort

The plan states "Day 2: replace stubs with real Qwen3 C core." As
established in A1, there is no pure-C Qwen3-4B text generation engine
that can be dropped in. Building one from scratch is a research project.
Porting `llama.cpp` to pure C (removing the C++ dependencies) is a
months-long effort that `llama.cpp` contributors have explicitly rejected
doing. Waiting for antirez to build one is not a plan.

The risk is that the project spends Day 1 getting stubs to compile and
load into Redis, declares partial success, then spends weeks trying to
make Day 2 happen before losing momentum.

**Mitigation**: Make a firm decision between the three options in A1
before writing another line of code. The decision determines whether this
is a 1-month project (link llama.cpp), a 12-month project (write pure-C
transformer), or a dependency-on-antirez project (indefinite wait).

---

#### P2: HIGH -- No Test Infrastructure of Any Kind

**Severity**: High
**Likelihood**: Certain impact
**Impact**: Every bug found in production (by Redis crashing); no regression signal

There are no tests. Not a single test file exists in the repository. For
a C Redis module with multiple confirmed critical bugs in 80 lines of code,
the absence of tests means:

- Bugs are only discovered by crashing Redis
- Refactoring is impossible without confidence
- The critical bugs identified in T1-T8 would have been caught by even
  minimal integration tests

The global state race (T3), the missing GIL acquisition (T2), and the
API misuse (T1) are all detectable by a test that calls `INFER.GENERATE`
twice concurrently under thread sanitizer (`-fsanitize=thread`).

**Mitigation**: Before implementing any real inference engine, write a
test harness using `redis-py` that loads the module and exercises all
commands. Run with Address Sanitizer (`-fsanitize=address`) and Thread
Sanitizer (`-fsanitize=thread`) from the start. These flags catch T1,
T2, T3, T5, and T7 automatically.

---

#### P3: HIGH -- Makefile Does Not Link Real Dependencies

**Severity**: High
**Likelihood**: Certain
**Impact**: Module compiles only because stubs resolve at link time; real engine integration will break the build

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/Makefile`):

```makefile
LDFLAGS = -shared -lm -pthread
```

The stub functions `qwen_load`, `qwen_generate`, and `gte_embed` are
declared but never defined. The `.so` compiles because shared libraries
allow undefined symbols to be resolved at load time. When a real inference
engine is linked in, the Makefile has no provisions for:
- Static library paths for the inference engine
- Include paths for inference engine headers
- RPATH configuration for finding shared libraries at runtime
- Platform-specific flags (macOS requires `-dynamiclib` not `-shared`)
- macOS requires `undefined dynamic_lookup` for Redis module symbols

On macOS (the author's platform per the environment info: `darwin`),
`-shared` is incorrect; the correct flag is `-dynamiclib`. The module will
fail to build on macOS as-is. This is immediately testable.

**Mitigation**: Fix macOS build immediately (`-dynamiclib` for Darwin).
Add platform detection in Makefile. Plan the inference engine linkage
strategy before beginning engine integration.

---

#### P4: MEDIUM -- Python Scripts Have No Dependency Pinning

**Severity**: Medium
**Likelihood**: High over time
**Impact**: Scripts break silently as transformers/sentence-transformers APIs change

Both Python scripts use `from transformers import AutoTokenizer` and
`from sentence_transformers import SentenceTransformer` with no version
pinning (no `requirements.txt`, no `pyproject.toml`). The HuggingFace
`transformers` library has broken backward compatibility multiple times.
`trust_remote_code=True` in `tokenize-and-store.py` executes arbitrary
Python code from the model repository, which is a security concern and
also means tokenizer behavior can change if the model author updates
their tokenizer code.

**Mitigation**: Create `requirements.txt` with pinned versions. Remove
`trust_remote_code=True` or document explicitly why it is needed.

---

#### P5: MEDIUM -- No Chunking Strategy Defined Before Data Is Ingested

**Severity**: Medium
**Likelihood**: Certain to matter when real data is processed
**Impact**: Re-ingestion required once chunking is implemented; wasted work

If the team runs tokenization and embedding on real data before chunking
is implemented, the resulting Redis keyspace will have whole-file entries.
When chunking is eventually added (it must be, per A7), all data must be
re-ingested. For a large codebase (100k+ files, 170 GB of token data),
this is a multi-hour or multi-day re-ingestion job.

**Mitigation**: Define the chunking strategy (function-boundary, N-token
window, or semantic) before running any data ingestion. This is a design
decision, not an implementation task, and costs one day of thinking now
versus weeks of re-ingestion later.

---

### Category 4: Security

---

#### S1: CRITICAL -- Inference Engine Code Runs in Redis Process with Root Access

**Severity**: Critical
**Likelihood**: Medium (depends on deployment context)
**Impact**: Any vulnerability in the inference engine or model weights is a full Redis compromise

A Redis module runs in the same process as Redis with the same OS
privileges. On many deployments, Redis runs as root or as a high-privilege
service user with access to all data in the keyspace. A memory corruption
bug in the inference engine (buffer overflow in token generation, heap
corruption in attention computation) is not contained -- it gives an
attacker arbitrary code execution in the Redis process with all its
privileges and access to all data.

The product vision explicitly acknowledges this: "A crash in the inference
engine will crash Redis." But the security implication goes further than
crashes. Crafted prompt inputs or malformed token sequences could trigger
controlled memory corruption, not just crashes. This is the class of
vulnerability that jailbreaks GPU inference services; in-process inference
has no sandbox boundary.

**Mitigation**: This is the fundamental security tradeoff of the in-process
architecture. Mitigations are limited: run Redis as a dedicated low-privilege
user with no network access to sensitive resources; consider a sidecar
with Unix socket if the security requirement is high; implement strict
input validation on token counts and sequence lengths; use memory sanitizers
during development to eliminate corruption bugs before deployment.

---

#### S2: HIGH -- trust_remote_code=True in Tokenizer

**Severity**: High
**Likelihood**: Low for official Qwen model, High for any fork or custom model
**Impact**: Arbitrary Python code execution during data ingestion

**Evidence** (`/Users/alex/projects/terraphim/redis-infer/scripts/tokenize-and-store.py`, line 14):

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
```

`trust_remote_code=True` downloads and executes Python code from the model
repository on HuggingFace. If the model repository is compromised (supply
chain attack), if a fork is used instead of the official repository, or if
the model ID is typo-squatted, arbitrary malicious Python executes during
the data ingestion pipeline. This is not a theoretical risk -- HuggingFace
has had supply chain incidents.

Qwen3-4B's tokenizer is based on the Qwen BPE tokenizer which does not
require `trust_remote_code`. The flag is unnecessary for the official model
and should be removed.

**Mitigation**: Remove `trust_remote_code=True`. If the tokenizer genuinely
requires it, pin the model revision using `revision="<specific-commit-hash>"`
to prevent runtime modification.

---

#### S3: HIGH -- Redis Exposed to Network with Inference Capability

**Severity**: High
**Likelihood**: High (Redis default config has no auth)
**Impact**: Any network client can trigger inference, exhausting compute resources

`INFER.GENERATE` is registered as `"write fast"` with key specification
`1, 1, 1` (key at argv[1]). There is no ACL check, no rate limiting, and
no authentication requirement beyond standard Redis AUTH (which is often
not configured for local deployments). Any client with network access to
Redis can issue unlimited `INFER.GENERATE` commands, each spawning a
thread and consuming CPU/RAM.

This is denial-of-service by design. The unbounded thread creation (T4)
means an attacker can exhaust threads and memory with a single for-loop
of `INFER.GENERATE` calls.

**Mitigation**: Require explicit Redis ACL configuration restricting
`INFER.*` commands to authorized users. Document this requirement clearly.
Implement rate limiting in the module (request queue depth). Fix T4 first.

---

#### S4: MEDIUM -- No Input Sanitization on Key Names

**Severity**: Medium
**Likelihood**: Medium
**Impact**: Key name injection if key names are constructed from user input

The `GenerateCommand` receives `argv[1]` as the key name to look up for
token data. This key is passed directly to `RedisModule_OpenKey`. While
Redis itself handles arbitrary binary key names safely, if future code
constructs composite key names from user input (e.g., prefixing with a
namespace), there is a path injection risk. The current code does not
prefix or validate the key name.

**Mitigation**: Define a strict key naming convention and validate that
incoming key names match the expected pattern before using them.

---

### Category 5: Strategic

---

#### ST1: HIGH -- antirez May Ship What This Project Needs (or Compete With It)

**Severity**: High
**Likelihood**: Medium
**Impact**: Months of work duplicated or obsoleted by antirez's own implementation

antirez is actively building pure-C inference engines at a rapid pace
(multiple repositories updated on 2026-02-26 alone: iris.c, linenoise,
voxtral.c, qwen-asr). The project's value proposition is explicitly built
around "antirez engine compatibility." If antirez builds a pure-C LLM
text generation engine (which his trajectory suggests is plausible), he
may simultaneously publish:
1. The engine this project needs (solving A1 instantly)
2. His own Redis module integrating it (competing with this project directly)

Neural-redis (2016) is his prior work doing exactly this pattern. He has
the Redis module API knowledge, the pure-C inference engine knowledge, and
the motivation. The question is timing.

**Risk to project**: If antirez ships a competing module, community
attention will follow his implementation, not a third-party module.
Antirez's implementations attract thousands of GitHub stars immediately.

**Mitigation**: Either move faster than antirez (days, not months) to
establish priority, or position as a contributor/integrator of his engines
rather than a competitor. Contributing to his repositories rather than
maintaining a separate module may be the highest-leverage approach.

---

#### ST2: HIGH -- llama.cpp Already Solves the Core Problem Adequately

**Severity**: High
**Likelihood**: Certain (it exists and works)
**Impact**: The project may be solving a problem that is already solved well enough

llama.cpp with its server mode runs on localhost, uses Unix sockets (not
just TCP), has GGUF support for Qwen3-4B, provides streaming, KV cache,
batching, and GPU support. The network hop from Redis to llama.cpp over
Unix socket is approximately 10-50 microseconds. For inference calls that
take 500ms-5000ms, this is <0.01% of total latency.

The value proposition requires that the 10-100us network hop saved by
zero-copy DMA is meaningful. For inference at 4B scale without GPU, the
inference itself dominates by 4-5 orders of magnitude. The architecture
eliminates a cost that is already negligible.

The honest assessment: the operational simplicity argument (one process
instead of two) is real and meaningful. The performance argument is not
meaningful without GPU inference, where the network hop matters more.

**Mitigation**: Reframe the project around operational simplicity and
atomicity of RAG + generate as the primary value, not performance. This
is a honest and defensible position. The current framing around zero-copy
DMA performance will be falsified immediately by benchmarks.

---

#### ST3: MEDIUM -- RedisAI / redis-inference-optimization is Stale but Could Be Revived

**Severity**: Medium
**Likelihood**: Low
**Impact**: Competitor has first-mover advantage and Redis Labs institutional support

The positioning table lists RedisAI/redis-inference-optimization as "stale
(last meaningful update 2024)." Redis Labs (now Redis, Inc.) has commercial
incentives to revive this if the AI inference in Redis market shows traction.
A well-funded commercial team could ship a polished version in weeks that
would outcompete an experimental open-source project.

**Mitigation**: Open source early, build community before commercial
alternatives arrive. The pure-C / antirez-philosophy angle is a defensible
niche that a commercial team building on PyTorch/ONNX would not occupy.

---

#### ST4: MEDIUM -- The Target User Is a Very Small Population

**Severity**: Medium
**Likelihood**: High
**Impact**: Community adoption signal (10 stars in 3 months) may not be achievable

The Venn diagram of: "uses Redis" AND "runs local inference" AND "cares about
the network hop between them" AND "is comfortable loading experimental C modules
into their production Redis instance" AND "uses Rust specifically" is extremely
small. The Grok conversation did not quantify this market size.

The success metric of ">10 stars or >1 external issue/PR within 3 months" is
achievable via Hacker News posts even for mediocre projects. It does not
validate actual adoption. The harder question is whether anyone uses it in
production.

**Mitigation**: This risk is acceptable for an experimental open-source
project with low cost of failure. The framing in the PVVH is honest about
experimental status. No mitigation needed beyond maintaining realistic
expectations.

---

#### ST5: LOW -- Qwen3-4B Rust Code Quality is Unvalidated

**Severity**: Low
**Likelihood**: Medium
**Impact**: Value proposition weakened if code quality is insufficient for practical use

The Grok conversation claims Qwen3-4B has "excellent Rust coding quality."
This is based on Grok's internal knowledge, not on empirical evaluation of
Qwen3-4B at Q4 quantization generating Rust code for the specific use case
(code completion from partial functions with context). 4-bit quantization
degrades generation quality relative to full precision, and coding quality
at 4B parameters is competitive but not leadership-tier compared to larger
models or specialized coding models.

**Mitigation**: Evaluate Qwen3-4B Q4 Rust code generation quality before
committing to it as the target model. HumanEval-Rust benchmarks exist. This
is a one-day evaluation task that should happen before Phase 1 begins.

---

## Summary Risk Matrix

| ID  | Risk                                             | Severity | Likelihood | Category     |
|-----|--------------------------------------------------|----------|------------|--------------|
| T1  | ReplyWithStringBuffer called on blocked client   | Critical | Certain    | Technical    |
| T2  | OpenKey without GIL in worker thread             | Critical | Certain    | Technical    |
| A1  | No pure-C text generation engine exists          | Critical | Certain    | Architectural|
| A2  | Zero-copy DMA value proposition is invalid       | Critical | Certain    | Architectural|
| P1  | "Day 2 stub replacement" is a fiction            | Critical | Certain    | Process      |
| S1  | Inference in Redis process, no sandbox           | Critical | Medium     | Security     |
| T3  | Global state race on g_qwen                      | Critical | Certain    | Technical    |
| T4  | Unbounded thread creation                        | High     | Certain    | Technical    |
| T5  | malloc failure not checked                       | High     | Low        | Technical    |
| T6  | atoi without validation on max_tokens            | High     | High       | Technical    |
| T7  | Strict aliasing violation at -O3                 | High     | Certain    | Technical    |
| A3  | Embedding model does not exist on HuggingFace    | High     | Certain    | Architectural|
| A4  | Key collision destroys RAG correctness           | High     | Certain    | Architectural|
| A5  | Performance claims off by 10-100x without GPU    | High     | Certain    | Architectural|
| P2  | No test infrastructure                           | High     | Certain    | Process      |
| P3  | Makefile broken on macOS                         | High     | Certain    | Process      |
| S2  | trust_remote_code=True in tokenizer              | High     | Low        | Security     |
| S3  | No rate limiting on inference commands           | High     | High       | Security     |
| ST1 | antirez ships competing implementation           | High     | Medium     | Strategic    |
| ST2 | llama.cpp already solves the problem adequately  | High     | Certain    | Strategic    |
| T8  | pthread_create failure not handled               | Medium   | Low        | Technical    |
| T9  | No OnUnload handler                              | Medium   | Certain    | Technical    |
| A6  | VADD syntax may be incorrect                     | Medium   | High       | Architectural|
| A7  | No chunking; RAG is file-grained                 | Medium   | Certain    | Architectural|
| A8  | allkeys-lru evicts model weights                 | Medium   | Medium     | Architectural|
| P4  | No Python dependency pinning                     | Medium   | High       | Process      |
| P5  | Chunking strategy undefined before ingestion     | Medium   | Certain    | Process      |
| S4  | No input sanitization on key names               | Medium   | Medium     | Security     |
| ST3 | RedisAI revival by commercial team               | Medium   | Low        | Strategic    |
| ST4 | Target user population too small                 | Medium   | High       | Strategic    |
| ST5 | Qwen3-4B Rust quality unvalidated at Q4          | Low      | Medium     | Strategic    |

---

## Decisions Required Before Any Further Development

The following are binary decisions, not implementation tasks. Code written
before these are resolved is likely to be thrown away:

**Decision 1: Inference engine source**
Options: (a) Link llama.cpp C API, (b) write pure-C transformer from scratch,
(c) wait for antirez. This decision determines project timeline (1 month vs
12+ months vs indefinite).

**Decision 2: Concurrency model**
The unbounded pthread create must be replaced with a bounded thread pool.
What is the maximum concurrent inference count? This determines memory
requirements and is the basis for any performance claim.

**Decision 3: DMA vs copy**
Accept that token data must be copied before inference. Revise value
proposition documentation to reflect co-location + atomicity as the value,
not zero-copy.

**Decision 4: Chunking strategy**
Define before any data ingestion. Cannot be changed cheaply after ingest.

**Decision 5: GPU requirement**
The performance claims require GPU. Does the target hardware have a GPU?
If not, the performance section of the plan must be rewritten.

---

## What the Grok Conversation Got Wrong

The Grok conversation was useful for architectural ideation but generated
several confident-sounding falsehoods that now appear as project assumptions:

1. "High pure-C feasibility (Qwen family has clean ports)" -- No such port
   exists. Speech-to-text ports (qwen-asr) do not imply text generation ports.

2. "antirez/gte-small-pure-c-converter" as a SentenceTransformers model --
   This does not exist on HuggingFace. The C inference engine and the
   SentenceTransformers model are different artifacts.

3. "40-80 concurrent generations <800ms each" on CPU -- Physically impossible
   without GPU acceleration for a 4B parameter model.

4. "Zero-copy DMA: RedisModule_StringDMA gives the inference engine a direct
   pointer" -- True in theory, impossible to use safely in a background thread
   without either holding the GIL (blocking Redis) or risking use-after-free.

5. "Day 2: replace stubs with real Qwen3 C core" -- No such core exists.

LLM-assisted design conversations generate plausible architecture. They do
not validate that the assumed dependencies exist, that the API contracts hold
under threading, or that the performance claims are physically achievable.
Via negativa analysis is the complement to LLM-assisted design: it asks
"what would have to be true for this to work?" and verifies each assumption
independently.

---

*End of Risk Register -- ZDP Discovery Artefact 2/5*
*Generated: 2026-02-26*
*Next artefact: Technical Design Decision Record (3/5)*
