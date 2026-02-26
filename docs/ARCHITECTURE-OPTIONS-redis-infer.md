# Architecture Options Analysis: redis-infer
## ZDP Discovery Artefact -- Comparative Design Investigation

**Date**: 2026-02-26
**Author**: Multi-Perspective Investigation (ZDP Discovery)
**Status**: Discovery Stage -- Pending Decision
**Scope**: Two design questions: (1) Rust vs C for Redis module, (2) Module-inside-Redis vs Cache-on-top architecture
**Depends on**: PVVH-redis-infer.md, RISK-SCAN-redis-infer.md, DECISIONS-redis-infer.md, RESEARCH-redis-infer.md

---

## S1 -- Action Context

Two design questions require resolution before implementation begins. Both affect the
fundamental engineering strategy for redis-infer and interact with the critical risks
identified in the risk scan (T1-T9, A1-A8, S1-S4).

**Question 1**: Should redis-infer be written in Rust (using `redismodule-rs`) instead
of C? The current C code has 9 confirmed bugs in 80 lines. The project owner works
in Rust (Terraphim ecosystem). A mature Rust crate exists for Redis modules.

**Question 2**: Should the architecture be inverted -- placing Redis as a caching/RAG
layer *on top of* an existing inference server (llama.cpp server or MLX), rather than
embedding inference *inside* Redis as a module?

These questions are not independent. The answer to Question 2 potentially eliminates
the need to answer Question 1 entirely (if the module approach is abandoned). They
are analyzed in sequence, with Question 1 first (it may inform the module variant of
Question 2).

---

## S2 -- Epistemic Status

### Question 1 (Rust vs C for Redis Module)

**Partially Known / Sufficient for Action**

- `redismodule-rs` is actively maintained, production-proven (RedisJSON, RediSearch),
  and provides safe abstractions for all Redis module API patterns relevant to
  redis-infer (blocking commands, thread-safe contexts, GIL management, memory
  allocator integration).
- Rust FFI to llama.cpp's C API is well-established via multiple crates (`llama-cpp-2`,
  `llama_cpp_rs`).
- The project owner has Rust expertise (Terraphim is a Rust project).
- **Gap**: No production example of a Rust Redis module linking against llama.cpp
  specifically. The integration is novel but composed of proven parts.

### Question 2 (Architecture: Module vs Cache Layer)

**Contested / Partially Known**

- The risk scan established that the module approach's primary claimed advantage
  (zero-copy DMA) is invalid, and CPU inference takes 10-25 seconds per request.
- The cache-on-top architecture eliminates most critical risks (T1-T9, S1) but
  changes the value proposition fundamentally.
- **Contested ground**: Whether operational simplicity (one process) outweighs
  safety (process isolation) depends on deployment context, which is a judgment
  call by the project owner, not a technical determination.
- **Gap**: No benchmarks exist comparing the two approaches on the target hardware.
  The latency difference between in-process and localhost HTTP is estimated (50-200
  us) but not measured.

---

## INVESTIGATION 1: Rust vs C for Redis Module

### Expert Lens Analysis

#### Security Expert Perspective

The current C code contains 9 bugs in 80 lines. Three are CRITICAL (T1: wrong reply
API, T2: missing GIL, T3: global state race). These are not subtle algorithmic errors
-- they are fundamental API misuse that Rust's type system would structurally prevent.

**Bugs eliminated by Rust's type system**:

| Risk ID | C Bug | How Rust Prevents It |
|---------|-------|---------------------|
| T1 | `ReplyWithStringBuffer(bc, ...)` -- passing `BlockedClient*` where `Context*` expected | `redismodule-rs` uses distinct types: `BlockedClient` and `Context` are not interchangeable. The `reply()` method is only available on `ThreadSafeContext`, which provides a valid context. Compilation fails if types are confused. |
| T2 | `OpenKey` called without GIL in worker thread | `ThreadSafeContext::lock()` returns a guard that provides `Context` access. Without calling `lock()`, there is no `Context` to call `open_key()` on. The borrow checker enforces this -- you cannot access Redis data without holding the lock guard. |
| T3 | `g_qwen` read/written without synchronization | `RedisGILGuard<T>` wraps shared state and requires the GIL lock to access the inner value. Alternatively, `Arc<RwLock<T>>` provides standard Rust synchronization. Both enforce synchronized access at compile time. |
| T5 | `malloc` return not checked | Rust's `Box::new()` and `Vec::new()` cannot return null. Allocation failure in Rust either calls the global allocator's error handler or panics (which can be caught). There is no null pointer dereference path. |
| T6 | `atoi` without validation | Rust's `str::parse::<i64>()` returns `Result<i64, ParseIntError>`. The compiler forces handling the error case. `atoi`-style silent-zero-on-failure is impossible. |
| T7 | Strict aliasing violation casting `char*` to `uint32_t*` | Rust's `bytemuck` crate or `std::mem::transmute` require explicit unsafe blocks. The standard approach -- `u8` slice to `u32` slice via `align_to()` -- returns `Result` and handles alignment. The copy pattern (`memcpy` equivalent) is `slice::copy_from_slice()` which is always safe. |
| T8 | `pthread_create` failure not handled | `std::thread::spawn` returns `Result`. Or with a thread pool crate like `rayon` or `threadpool`, thread creation failures are handled internally. |
| T9 | No OnUnload handler | The `redis_module!` macro supports `deinit` callbacks. Rust's `Drop` trait ensures cleanup. Forgetting to implement `Drop` for the model context is still possible, but the module framework provides the hook naturally. |

**Assessment**: 8 of 9 identified C bugs (T1-T3, T5-T9) are structurally prevented
or made significantly harder by Rust's type system. T4 (unbounded thread creation)
is a design problem, not a language problem -- it requires a thread pool regardless
of language. However, Rust's ecosystem provides production-grade thread pools
(`rayon`, `threadpool`, `tokio`) that make T4 trivial to solve.

#### Pragmatist Perspective

**redismodule-rs capabilities relevant to redis-infer**:

The crate provides safe Rust wrappers for every Redis module API pattern needed:

1. **Blocking commands with background threads**:
   ```rust
   fn generate(ctx: &Context, args: Vec<RedisString>) -> RedisResult {
       let blocked_client = ctx.block_client();
       thread::spawn(move || {
           let thread_ctx = ThreadSafeContext::with_blocked_client(blocked_client);
           let ctx = thread_ctx.lock(); // Acquires GIL
           // Access Redis keys safely here
           drop(ctx); // Release GIL
           // Run inference on copied data
           thread_ctx.reply(Ok(result.into()));
       });
       Ok(RedisValue::NoReply)
   }
   ```
   This is the exact copy-under-GIL pattern recommended in RESEARCH-redis-infer.md
   (RQ1f), expressed in safe Rust with compile-time correctness guarantees.

2. **Memory allocator integration**: `redismodule-rs` implements `GlobalAlloc` using
   `RedisModule_Alloc`/`RedisModule_Free`, so all Rust allocations go through Redis's
   memory tracking. This is automatic -- no manual `RedisModule_Alloc` calls needed.

3. **GIL-protected global state**: `RedisGILGuard<T>` provides safe access to module
   globals, directly solving T3 (global state race on `g_qwen`).

4. **Platform compatibility**: `redismodule-rs` handles macOS (`dylib`) vs Linux
   (`so`) build differences automatically via Cargo. This solves P3 (Makefile broken
   on macOS).

**FFI to llama.cpp**: Two proven approaches exist:

- **`llama-cpp-2` crate**: Uses `bindgen` to auto-generate Rust bindings from
  `llama.h`. Provides raw FFI access to all llama.cpp C API functions. The redis-infer
  module would link against `libllama` (static or dynamic) via this crate.

- **Custom `build.rs`**: Write a Cargo build script that compiles llama.cpp as a
  static library and generates bindings. More control, more maintenance.

**FFI overhead**: Rust FFI to C has zero overhead -- it is a direct function call with
no marshaling. `llama_decode()`, `llama_sampler_sample()`, etc. are called at native
speed. The FFI boundary adds no measurable latency compared to calling these functions
from C.

**Build system**: Cargo replaces the broken Makefile entirely. `cargo build` handles:
- Platform detection (macOS/Linux)
- Static linking of llama.cpp
- Correct shared library output (`.dylib` / `.so`)
- Dependency management
- Test infrastructure (`cargo test`)
- Address Sanitizer and Thread Sanitizer via `RUSTFLAGS`

#### Analyst Perspective

**Development velocity comparison**:

| Factor | C | Rust |
|--------|---|------|
| Lines of code to fix T1-T9 | ~200+ (rewrite threading, add thread pool, add validation) | 0 (prevented by type system) |
| Build system | Makefile needs platform fixes, llama.cpp integration | Cargo handles everything |
| Test infrastructure | Must build from scratch (P2) | `cargo test` + `redis-rs` for integration tests |
| Sanitizer integration | Manual CFLAGS/LDFLAGS | `RUSTFLAGS="-Zsanitizer=address"` |
| Memory safety bugs during development | Expected (C) | Prevented at compile time (Rust) |
| Developer expertise | Owner works in Rust (Terraphim) | Native fit |
| Ecosystem for thread pools | Manual pthread pool or find C library | `threadpool` / `rayon` / `crossbeam` crates |
| Ecosystem for llama.cpp binding | Direct C linking (native) | `llama-cpp-2` crate or custom `build.rs` |

**Risk transfer**: Moving to Rust transfers the risk from "runtime crashes discovered
in production" (C) to "compilation errors discovered during development" (Rust). For
a project that runs inside the Redis server process (where a crash kills Redis), this
risk transfer is exceptionally valuable.

#### Critic Perspective

**Risks of the Rust approach**:

1. **`redismodule-rs` API coverage gap**: While the crate covers the common Redis
   module APIs, it may not expose every low-level API function. If redis-infer needs
   a Redis module API that `redismodule-rs` does not wrap, the fallback is `unsafe`
   blocks calling the raw C API via `redis_module::raw`. This is possible but loses
   the safety guarantees for that specific call.

2. **Vector Sets command support**: `redismodule-rs` does not provide built-in
   wrappers for Vector Sets commands (`VADD`, `VSIM`). These would need to be called
   via `ctx.call("VADD", &[...])` -- the generic command execution path. This works
   but is stringly-typed and does not provide compile-time argument validation.
   **Assessment**: Acceptable. Vector Sets commands are called as Redis commands, not
   as module API functions. The generic `call()` method is the correct interface.

3. **Compilation time**: Rust + llama.cpp (C++) compilation is slower than pure C.
   Initial build may take 5-10 minutes. Incremental builds are fast.
   **Assessment**: Minor inconvenience, not a blocking issue.

4. **Binary size**: Rust binaries include the standard library. The module `.dylib`
   will be larger (10-20 MB with llama.cpp statically linked vs 5-10 MB for C).
   **Assessment**: Irrelevant for a server-side module.

5. **antirez philosophy alignment**: The project vision references "antirez-style
   minimalism: pure C, no bloat." Writing in Rust diverges from this philosophy.
   **Assessment**: The vision already abandoned "pure C" when Decision 1 chose
   llama.cpp (which is C++). Rust is closer to C in spirit than C++ is, and
   antirez himself has said "use the right tool for the job." The philosophical
   purity argument was already settled by Decision 1.

#### User Advocate Perspective

The end user (developer loading the module into Redis) does not see or care whether
the module is written in C or Rust. They interact via Redis commands
(`INFER.GENERATE`, `INFER.LOAD`). The module binary (`.so`/`.dylib`) is loaded
identically. The only user-visible difference: the Rust module is far less likely to
crash their Redis instance.

### Question 1 Recommendation

**RECOMMENDATION: Write redis-infer in Rust using `redismodule-rs`.**

**Rationale**:
- Eliminates 8 of 9 identified critical/high bugs at compile time
- Native fit for the project owner's expertise (Terraphim is Rust)
- Production-proven crate (RedisJSON, RediSearch depend on it)
- Zero FFI overhead to llama.cpp C API
- Cargo solves all build system problems (P3)
- Built-in test infrastructure solves P2
- Memory allocator integration is automatic
- `RedisGILGuard` directly implements the safe copy-under-GIL pattern

**What Rust does NOT solve** (still requires design work):
- Thread pool sizing (T4) -- must be implemented, but Rust ecosystem makes it trivial
- Model loading strategy (OQ2) -- design decision, language-independent
- Token ID compatibility (H2d) -- validation experiment, language-independent
- Performance characteristics -- identical to C (same llama.cpp backend)

---

## INVESTIGATION 2: Architecture Comparison

Three architectures are compared. For each, the analysis covers development
complexity, safety, performance, operational complexity, value proposition
preservation, code reuse, and risk elimination.

### Option A: Redis Module (Current Plan, Revised for Rust + llama.cpp)

**Description**: Inference runs inside the Redis process as a loadable module. The
module links against llama.cpp's C API. Pre-tokenized data is copied from Redis keys
under the GIL. Inference runs on background threads with a bounded thread pool.

**Architecture diagram**:
```
+-------------------------------------------+
|              Redis Process                 |
|                                            |
|  +-------+  +---------------------------+ |
|  | Redis |  | redis-infer module (Rust)  | |
|  | Core  |  |                            | |
|  |       |  | +-----+ +---------------+  | |
|  | Keys  |<-+-| GIL | | Thread Pool   |  | |
|  | VSIM  |  | | Copy| | llama_context | | |
|  |       |  | +-----+ | llama_context | | |
|  +-------+  |         | llama_context | | |
|             |         +---------------+  | |
|             | llama_model (shared)       | |
|             +---------------------------+ |
+-------------------------------------------+
```

**Development complexity**: MEDIUM-HIGH
- Requires writing a Redis module in Rust (~500-1000 lines)
- Requires linking llama.cpp statically into the module
- Requires implementing: thread pool, copy-under-GIL, reply callbacks,
  model loading, context pool, sampling configuration
- Build system complexity: Cargo with custom `build.rs` for llama.cpp
- Estimated time to working prototype: 2-4 weeks

**Safety (process isolation / crash impact)**:
- **No process isolation**. A bug in llama.cpp, a corrupted GGUF file, or an OOM
  in the inference engine crashes the entire Redis process.
- Rust prevents most memory safety bugs in the *module code*, but llama.cpp itself
  is C++ -- any bug in llama.cpp is unmitigated.
- **Crash blast radius**: All Redis data, all connected clients, all in-flight
  operations.
- **Risk S1 from risk scan**: RETAINED. This is the fundamental tradeoff of the
  in-process architecture.

**Performance characteristics**:
- Token data copy from Redis: ~3 us (negligible)
- No network roundtrip for data access (saved: ~50-200 us per key fetch)
- Inference speed: identical to standalone llama.cpp (same C++ code path)
- RAG assembly (VSIM + multiple key fetches): atomic, single-process, ~1-5 ms
- **Net latency advantage over sidecar**: ~100-500 us saved on data assembly
  out of a 10-25 second total pipeline. This is ~0.002-0.005% improvement.
- **The performance advantage is real but negligible relative to inference time.**

**Operational complexity**: LOW
- Single process: `redis-server --loadmodule redis-infer.dylib`
- One thing to monitor, one thing to restart
- Model loaded via `INFER.LOAD /path/to/model.gguf`
- No inter-process communication to configure

**Value proposition preservation**: HIGH
- Preserves: co-location, atomic RAG+generate, pre-tokenized storage, Vector Sets
  integration, operational simplicity
- Lost: zero-copy DMA (already invalidated by risk scan)
- Added: Rust safety guarantees for module code

**Code/knowledge reuse**:
- Existing C code: fully rewritten (but only 80 lines, all buggy)
- Existing Python scripts: reusable with modifications (token ID validation needed)
- Owner's Rust knowledge: directly applicable
- llama.cpp knowledge: directly applicable via C API

**Risk elimination from risk scan**:

| Risk | Status in Option A (Rust) |
|------|--------------------------|
| T1 (wrong reply API) | ELIMINATED -- type system prevents |
| T2 (missing GIL) | ELIMINATED -- lock guard pattern |
| T3 (global state race) | ELIMINATED -- RedisGILGuard |
| T4 (unbounded threads) | ELIMINATED -- thread pool crate |
| T5 (malloc null) | ELIMINATED -- Rust allocation |
| T6 (atoi validation) | ELIMINATED -- parse returns Result |
| T7 (strict aliasing) | ELIMINATED -- safe copy pattern |
| T8 (pthread_create) | ELIMINATED -- thread pool |
| T9 (no OnUnload) | MITIGATED -- Drop trait + deinit |
| A1 (no pure-C engine) | RESOLVED -- llama.cpp decision |
| A2 (zero-copy invalid) | RESOLVED -- copy-under-GIL |
| S1 (in-process risk) | **RETAINED** -- fundamental to architecture |
| S3 (no rate limiting) | MITIGATED -- thread pool provides backpressure |
| P2 (no tests) | ELIMINATED -- cargo test |
| P3 (macOS build) | ELIMINATED -- Cargo |

---

### Option B: Redis Cache Layer on llama.cpp Server

**Description**: llama.cpp runs as its own HTTP server (`llama-server`). Redis sits
in front as a caching and RAG assembly layer. The application (or a thin middleware
/ Redis module) queries Redis for context assembly and cache lookup, then forwards
inference requests to llama-server over HTTP/localhost.

**Architecture diagram**:
```
+-------------------+         +----------------------+
|   Redis Process   |         | llama-server Process |
|                   |  HTTP   |                      |
| Keys (tokens)    |         | llama_model          |
| Vector Sets (RAG)|  <----> | llama_context(s)     |
| LangCache        |  :8080  | Sampling             |
|                   |         | KV Cache             |
+-------------------+         +----------------------+
        ^                              ^
        |          +----------+        |
        +----------| App/Proxy|--------+
                   +----------+
```

Two sub-variants exist:

**B1: Application-level orchestration** -- The application code (Python, Rust, etc.)
calls Redis for VSIM + token fetch, assembles the prompt, checks LangCache for a
cached response, and if no cache hit, calls llama-server's HTTP API. No custom Redis
module needed.

**B2: Redis module as proxy** -- A Redis module receives the `INFER.GENERATE` command,
internally calls Redis for VSIM + token fetch, checks LangCache, and if no cache hit,
makes an outbound HTTP call to llama-server. This preserves the single-command
interface but introduces outbound HTTP from a Redis module.

**Development complexity**: LOW (B1) / MEDIUM (B2)
- **B1**: No module code at all. Application code uses standard Redis client + HTTP
  client. LangCache is a managed Redis service (or RedisVL for self-hosted). VSIM
  queries use standard Redis commands. The "module" is replaced by ~100-200 lines of
  application code.
  Estimated time to working prototype: 3-5 days.
- **B2**: Requires a Redis module that makes outbound HTTP calls. This is unusual for
  Redis modules (they typically do not make network calls). Adds complexity of HTTP
  client in module context, timeout handling, connection pooling. Estimated time:
  2-3 weeks.

**Safety (process isolation / crash impact)**:
- **Full process isolation**. llama-server runs in its own process. A crash in
  llama-server (OOM, corrupted model, segfault in C++ code) does not affect Redis.
  Redis continues serving all other workloads uninterrupted.
- **Crash blast radius**: Only the inference request that triggered the crash is
  lost. Redis data is unaffected.
- **This eliminates risk S1 entirely.** It also eliminates the practical impact of
  any llama.cpp bug -- the bug crashes llama-server, not Redis.

**Performance characteristics**:
- Data assembly: Redis VSIM + GET commands over localhost TCP (~50-200 us per command)
- For RAG with 5 context chunks: ~250-1000 us for data assembly
- LangCache lookup: ~5-20 ms (vector similarity search for semantic cache)
- Cache hit: response returned in ~5-20 ms (no inference needed)
- Cache miss: HTTP POST to llama-server (~50-100 us network), then 10-25 s inference
- **Net latency vs Option A**: ~0.5-2 ms additional for data assembly + network hop
  out of 10-25 seconds total. This is ~0.01% overhead.
- **LangCache provides a massive latency win on cache hits** -- 5-20 ms vs 10-25 s.
  This is a 500-5000x improvement for repeated/similar queries. Option A has no
  equivalent caching mechanism built in.

**Operational complexity**: MEDIUM
- Two processes to manage (Redis + llama-server)
- Both can be managed via `docker-compose`, `systemd`, or `tmux`
- llama-server has its own health check endpoint (`/health`)
- Need to coordinate model loading (llama-server loads model at startup)
- Need to handle llama-server restarts (health check + auto-restart)

**Value proposition preservation**: MEDIUM-HIGH
- **Preserved**: Pre-tokenized storage in Redis, Vector Sets for RAG, operational
  use of Redis as the data layer
- **Preserved (enhanced)**: Semantic caching via LangCache -- this is a new
  capability that Option A does not have
- **Lost**: Single-command atomicity of RAG + generate (now requires multiple steps)
- **Lost**: Single-process deployment (now two processes)
- **Changed**: The "value" shifts from "inference inside Redis" to "Redis as the
  intelligent caching and retrieval layer for inference"

**Code/knowledge reuse**:
- Existing C code: not needed (no module for B1, rewritten for B2)
- Existing Python scripts: reusable (tokenize-and-store, embed-and-store)
- llama-server: production-grade, no custom code needed
- Redis knowledge: directly applicable
- **New capability**: LangCache integration (learning curve: 1-2 days)

**Risk elimination from risk scan**:

| Risk | Status in Option B |
|------|-------------------|
| T1-T9 (all C bugs) | ELIMINATED -- no custom C/Rust module code (B1) or rewritten safely (B2) |
| A1 (no pure-C engine) | ELIMINATED -- using llama-server directly |
| A2 (zero-copy invalid) | ELIMINATED -- not attempted |
| S1 (in-process risk) | **ELIMINATED** -- full process isolation |
| S3 (no rate limiting) | ELIMINATED -- llama-server has built-in slot limits |
| P2 (no tests) | REDUCED -- less custom code to test |
| P3 (macOS build) | ELIMINATED -- no custom build needed for B1 |

---

### Option C: Redis Cache Layer on MLX (Apple Silicon)

**Description**: Identical architecture to Option B, but inference uses MLX on Apple
Silicon instead of llama.cpp. MLX provides an OpenAI-compatible server
(`mlx_lm.server`) that runs natively on Apple Silicon with Metal GPU acceleration.

**Architecture diagram**:
```
+-------------------+         +-------------------------+
|   Redis Process   |         | mlx_lm.server Process   |
|                   |  HTTP   |                         |
| Keys (tokens)    |         | MLX Framework           |
| Vector Sets (RAG)|  <----> | Metal GPU Acceleration  |
| LangCache        |  :8080  | Unified Memory Access   |
|                   |         |                         |
+-------------------+         +-------------------------+
        ^                              ^
        |          +----------+        |
        +----------| App/Proxy|--------+
                   +----------+
```

**Development complexity**: LOW
- `mlx_lm.server` provides an OpenAI-compatible HTTP API out of the box
- `pip install mlx-lm && mlx_lm.server --model Qwen/Qwen3-4B` -- running in minutes
- Application code identical to Option B1 (same HTTP API, different endpoint)
- Estimated time to working prototype: 1-2 days

**Safety (process isolation / crash impact)**:
- Identical to Option B. Full process isolation. MLX crash does not affect Redis.

**Performance characteristics**:
- MLX on Apple Silicon M4 Pro: ~60 tokens/second generation for Qwen3-4B
  (reported in llama.cpp issue #19366 comparative benchmarks)
- llama.cpp on same hardware: ~24 tokens/second
- **MLX is 2-2.5x faster than llama.cpp on Apple Silicon** for this model
- Unified memory architecture: model weights accessed at ~273 GB/s (M4 Pro)
  vs DDR5 ~50 GB/s on x86 servers
- For 256-token generation: MLX ~4.3 seconds vs llama.cpp ~10.7 seconds on M4 Pro
- **Development iteration speed is significantly faster with MLX on Mac**

**Operational complexity**: LOW-MEDIUM
- Two processes, but `mlx_lm.server` is a single pip install + command
- No compilation needed (unlike llama.cpp which must be built from source)
- Python ecosystem -- easy to extend and debug
- **Critical limitation**: MLX runs ONLY on Apple Silicon. No x86, no Linux server,
  no GPU cluster. Production deployment on non-Apple hardware requires switching to
  llama.cpp or another engine.

**Value proposition preservation**: MEDIUM
- Same as Option B for the Redis layer
- **Development advantage**: Faster inference on Mac for dev/test cycles
- **Production disadvantage**: Cannot deploy MLX to x86 production servers
- **Portability gap**: Code works on dev Mac but needs a different inference
  backend for production. This is a significant operational concern.

**Code/knowledge reuse**:
- Everything from Option B applies
- MLX uses HuggingFace model format directly -- no GGUF conversion needed
- Token ID compatibility: MLX uses the HuggingFace tokenizer, so pre-tokenized
  data from the existing Python scripts is directly compatible (no H2d concern)

**Risk elimination**: Same as Option B, plus:
- H2d (token ID mismatch): ELIMINATED for development (MLX uses HF tokenizer)
- Reintroduced for production if switching to llama.cpp

---

## Comparative Analysis Matrix

| Dimension | Option A: Module (Rust) | Option B: Cache on llama.cpp | Option C: Cache on MLX |
|-----------|------------------------|-----------------------------|-----------------------|
| **Time to prototype** | 2-4 weeks | 3-5 days (B1) / 2-3 weeks (B2) | 1-2 days |
| **Process isolation** | None | Full | Full |
| **Redis crash risk** | HIGH (S1 retained) | NONE | NONE |
| **Inference speed (M4 Pro)** | ~24 t/s (llama.cpp) | ~24 t/s (llama.cpp) | ~60 t/s (MLX Metal) |
| **Inference speed (x86 prod)** | ~15 t/s | ~15 t/s | N/A (Apple only) |
| **Semantic caching** | Not built in | LangCache (500-5000x on hits) | LangCache (500-5000x on hits) |
| **Single-command RAG+gen** | Yes (atomic) | No (multi-step) | No (multi-step) |
| **Single-process deploy** | Yes | No (2 processes) | No (2 processes) |
| **Bugs prevented by design** | 8 of 9 (Rust type system) | All (no custom module for B1) | All (no custom module) |
| **Production portability** | Linux + macOS | Linux + macOS + any | macOS only (dev) |
| **Streaming support** | Must implement | Built into llama-server | Built into mlx_lm.server |
| **GPU acceleration path** | Via llama.cpp CUDA/Metal | Via llama-server flags | Native Metal |
| **Concurrent request handling** | Custom thread pool | Built into llama-server (slots) | Built into mlx_lm.server |
| **OpenAI API compatibility** | No | Yes (llama-server) | Yes (mlx_lm.server) |
| **Build complexity** | Medium (Cargo + llama.cpp) | None (B1) / Low (B2) | None (pip install) |
| **Custom code volume** | ~500-1000 lines Rust | ~100-200 lines Python/Rust | ~100-200 lines Python/Rust |

---

## S3 -- Actionable Options

### Option A: Redis Module in Rust

**Preconditions**:
- Rust toolchain installed
- llama.cpp source available for static linking
- Redis 8 with module loading support
- GGUF model file on disk

**Expected effects**:
- Single-binary Redis module providing `INFER.LOAD`, `INFER.GENERATE`, `INFER.RAG`
- Atomic RAG+generate in a single Redis command
- All T1-T9 bugs prevented by Rust type system
- Model loaded into Redis process memory

**Primary risks**:
- S1 (in-process crash risk) is *retained* and *cannot be mitigated* within this
  architecture. Any llama.cpp bug crashes Redis.
- llama.cpp C++ build integration with Cargo may have platform-specific issues
- Thread pool sizing and KV cache memory management require careful engineering
- No semantic caching (must be built separately if needed)

**Supporting perspectives**: Security Expert (Rust safety), Pragmatist (builds on
proven crate), User Advocate (invisible to end user)

### Option B: Redis Cache on llama.cpp Server

**Preconditions**:
- llama-server built and running
- Redis 8 with Vector Sets
- LangCache available (managed or self-hosted via RedisVL)
- Application code for orchestration

**Expected effects**:
- Two-process deployment: Redis (data + cache) + llama-server (inference)
- Full process isolation -- inference crashes do not affect Redis
- Semantic caching via LangCache provides 500-5000x speedup on cache hits
- OpenAI-compatible API enables ecosystem tool integration
- Streaming responses built in

**Primary risks**:
- Two processes to manage (moderate operational complexity)
- No atomic RAG+generate (requires application-level orchestration)
- LangCache is currently in preview (may have limitations or breaking changes)
- Value proposition changes: redis-infer becomes "Redis as inference cache"
  rather than "inference inside Redis"

**Supporting perspectives**: Critic (process isolation is critical), Systems Thinker
(LangCache provides emergent value), Strategist (OpenAI API compatibility enables
broader adoption)

### Option C: Redis Cache on MLX (Development) + llama.cpp (Production)

**Preconditions**:
- Apple Silicon Mac for development
- `mlx-lm` installed
- llama-server for production deployment
- Application code for orchestration (same as B1)

**Expected effects**:
- Fastest development iteration: 2.5x faster inference during dev/test
- Identical Redis layer to Option B
- Production deployment uses llama-server (Option B architecture)
- Two inference backends to maintain

**Primary risks**:
- Platform divergence: dev (MLX) vs prod (llama.cpp) use different engines
- Behavioral differences between MLX and llama.cpp inference (temperature,
  sampling, token generation patterns) may cause dev/prod discrepancies
- Token ID compatibility: MLX uses HF tokenizer, llama.cpp uses GGUF tokenizer.
  Pre-tokenized data may not be compatible across both.
- Maintenance burden of supporting two inference backends

**Supporting perspectives**: Innovator (faster dev cycles), Critic (platform
divergence is risky), Pragmatist (realistic about dev workflow)

### Hybrid Option D: Option B + Thin Rust Module for Atomic RAG (Recommended)

This option was not in the original request but emerges from the analysis as the
approach that preserves the most value while eliminating the most risk.

**Description**: Use Option B as the primary architecture (Redis cache + llama-server).
Additionally, write a thin Rust Redis module that provides a single atomic `INFER.RAG`
command that:
1. Runs VSIM to find relevant chunks
2. Fetches pre-tokenized data for those chunks
3. Checks LangCache for a cached response
4. If cache miss: assembles the prompt and makes an HTTP call to llama-server
5. Stores the response in LangCache
6. Returns the result

This module does NOT embed llama.cpp. It is a thin orchestration layer (~200-300 lines
of Rust) that provides the single-command UX while delegating inference to an isolated
process.

**Preconditions**: Same as Option B + `redismodule-rs` for the thin module

**Expected effects**:
- Single-command RAG+generate (`INFER.RAG key max_tokens`)
- Full process isolation (inference in llama-server)
- Semantic caching via LangCache
- Module crash risk: MINIMAL (no heavy compute, no llama.cpp in process)
- Estimated time to prototype: 1-2 weeks

**Primary risks**:
- Module makes outbound HTTP calls (unusual for Redis modules, but not prohibited)
- Adds latency for the HTTP roundtrip (~100-200 us) -- negligible vs inference time
- More moving parts than pure Option B1

**Supporting perspectives**: Strategist (best of both worlds), Pragmatist (thin
module = small attack surface), User Advocate (single-command interface preserved)

---

## S4 -- Bounded Commitments

### Recommended Commitment: Phased Approach

**Phase 0 (Immediate, 1-2 days): Option C for validation**
- **Scope**: Install `mlx-lm`, run `mlx_lm.server`, confirm Qwen3-4B generates
  useful Rust code. Run VE1 and VE2 validation experiments.
- **Duration**: 2 days
- **Assumptions**: Apple Silicon Mac available; `mlx-lm` installs without issues
- **Revocation condition**: If Qwen3-4B code quality is insufficient (ST5), reassess
  model choice before proceeding
- **Purpose**: Establish ground truth for inference quality and speed before investing
  in architecture

**Phase 1 (1 week): Option B1 for working end-to-end prototype**
- **Scope**: Redis with Vector Sets + llama-server running locally. Application
  script that does VSIM query, fetches tokens, calls llama-server, returns result.
  No custom module.
- **Duration**: 1 week
- **Assumptions**: Redis 8 available with Vector Sets; llama-server builds on target
  platform
- **Revocation condition**: If llama-server HTTP latency overhead is unexpectedly
  high (>10ms per call), or if the multi-step orchestration proves unworkable,
  reconsider Option A
- **Purpose**: Validate the cache-on-top architecture with minimum code

**Phase 2 (2 weeks): Hybrid Option D -- thin Rust module**
- **Scope**: Write a thin Rust Redis module using `redismodule-rs` that provides
  `INFER.RAG` as a single atomic command, delegating to llama-server for inference.
  Include LangCache integration.
- **Duration**: 2 weeks
- **Assumptions**: Phase 1 validates the architecture; `redismodule-rs` supports
  the needed APIs; outbound HTTP from a Redis module is feasible
- **Revocation condition**: If `redismodule-rs` lacks critical APIs, or if outbound
  HTTP from module context proves unreliable, fall back to Option B1 (no module)
- **Purpose**: Provide single-command UX while keeping inference out of Redis process

**Phase 3 (Optional, 2-4 weeks): Full Rust module (Option A) if justified**
- **Scope**: Only if benchmarks from Phase 1-2 show that the HTTP overhead is
  significant (it should not be), or if the single-process deployment requirement
  is non-negotiable for production
- **Duration**: 2-4 weeks
- **Assumptions**: Phase 2 identifies a concrete, measurable limitation of the
  cache-on-top architecture
- **Revocation condition**: If Phase 2 works well enough, skip Phase 3 entirely
- **Purpose**: Fallback path that preserves the original vision if the cache-on-top
  architecture proves insufficient

### Why This Ordering

1. Phase 0 validates the model (cheapest possible experiment)
2. Phase 1 validates the architecture (minimum code, maximum learning)
3. Phase 2 adds UX polish (single command) while maintaining safety
4. Phase 3 is insurance (only if earlier phases reveal problems)

Each phase is independently valuable and can be stopped without wasting prior work.

---

## S5 -- Escalation Assessment

**Escalation is NOT required for Question 1 (Rust vs C).**
The evidence is sufficient to act. Rust is strictly superior for this project given
the owner's expertise, the crate maturity, and the bug elimination properties.
No uncertainty blocks this decision.

**Escalation is PARTIALLY required for Question 2 (Architecture).**

The choice between Option A (module) and Option B/D (cache layer) depends on a
value judgment that only the project owner can make:

**Decision required from project owner**:

> Is the "inference inside Redis" identity of the project essential, or is
> "Redis as the intelligent layer for inference" an acceptable reframing?

If the identity is essential (the project IS about running inference in Redis),
then Option A is the only viable path, accepting risk S1.

If the identity is flexible (the project is about making Redis + inference
better together), then Option D (hybrid) provides the best risk/value tradeoff.

**What would resolve this**: A statement from the project owner on whether
single-process deployment and in-process inference are non-negotiable requirements
or flexible preferences.

**Additional escalation trigger**: If Phase 0 (model quality validation) reveals
that Qwen3-4B at Q4 quantization produces inadequate Rust code, the entire project
scope needs reassessment regardless of architecture choice.

---

## S6 -- Control-Surface Summary

### What the evidence constrains:

- **Rust over C**: The evidence strongly favors Rust. 8/9 bugs eliminated, owner
  expertise, production crate, zero FFI overhead. This is actionable without
  further investigation.

- **Zero-copy DMA is dead**: Confirmed invalid across all architectures. The
  copy-under-GIL pattern is the correct approach for any Redis module variant.
  This constrains Option A's implementation but does not affect the architecture
  choice.

- **CPU inference is 10-25 seconds**: This constrains the performance narrative
  for all options equally. No architecture makes CPU inference faster. The
  difference between options is in the ~0.1-2ms data assembly overhead, which
  is invisible at this timescale.

- **LangCache provides transformative value on cache hits**: This favors Options
  B/C/D. Semantic caching turns a 10-25 second operation into a 5-20 ms
  operation for similar queries. Option A has no equivalent.

### What the evidence does NOT resolve:

- **Whether single-process deployment is a hard requirement**: This is a values
  decision, not a technical one. The analysis provides the tradeoffs; the project
  owner must decide.

- **Actual llama-server HTTP overhead on target hardware**: Estimated at 50-200
  us but not measured. Phase 1 validates this.

- **Whether outbound HTTP from a Redis module is reliable in production**: Option
  D depends on this. It is technically possible but uncommon. Phase 2 validates this.

- **Qwen3-4B Rust code quality at Q4 quantization**: ST5 from the risk scan.
  Unvalidated. Phase 0 validates this.

### What remains open and revisable:

- The architecture choice can be revisited after each phase. The phased approach
  is designed for this: each phase produces evidence that either confirms or
  challenges the current trajectory.

- The model choice (Qwen3-4B) is not locked in. If Phase 0 shows inadequate code
  quality, switching to a different model (Qwen3-8B, DeepSeek-Coder, CodeLlama)
  is straightforward with both llama.cpp and MLX.

- The Rust decision applies regardless of architecture. Even Option D's thin
  module benefits from Rust's safety guarantees.

### Governance summary:

This analysis stabilizes the decision space by:
1. Closing the Rust vs C question (Rust, with strong evidence)
2. Narrowing the architecture question to a values-based choice between A and D
3. Proposing a phased validation path that defers the final architecture commitment
   to after empirical evidence is gathered
4. Identifying the specific escalation trigger (project identity question) that
   only the project owner can resolve

---

## References

### redismodule-rs
- [redismodule-rs GitHub Repository](https://github.com/RedisLabsModules/redismodule-rs)
- [redis-module crate on crates.io](https://crates.io/crates/redis-module)
- [redis_module API docs on docs.rs](https://docs.rs/redis-module)
- [ThreadSafeContext usage discussion (Issue #119)](https://github.com/RedisLabsModules/redismodule-rs/issues/119)

### Rust llama.cpp Bindings
- [llama-cpp-2 crate on crates.io](https://crates.io/crates/llama-cpp-2)
- [llama_cpp_rs GitHub (edgenai)](https://github.com/edgenai/llama_cpp-rs)
- [rust-llama.cpp GitHub (mdrokz)](https://github.com/mdrokz/rust-llama.cpp)

### llama.cpp Server
- [llama-server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama-server HTTP API (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/5.2-server)
- [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp)

### MLX
- [MLX GitHub (ml-explore)](https://github.com/ml-explore/mlx)
- [MLX LM GitHub](https://github.com/ml-explore/mlx-lm)
- [mlx_lm.server documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/SERVER.md)
- [MLX OpenAI Server (cubist38)](https://github.com/cubist38/mlx-openai-server)
- [WWDC 2025: MLX for Apple Silicon](https://developer.apple.com/videos/play/wwdc2025/298/)

### Redis LangCache
- [Redis LangCache](https://redis.io/langcache/)
- [LLM Caching with Redis](https://redis.io/docs/latest/develop/ai/redisvl/user_guide/llmcache/)
- [Semantic Caching Guide](https://redis.io/blog/what-is-semantic-caching/)
- [Redis LangCache Documentation](https://redis.io/docs/latest/develop/ai/langcache/)

### Redis Module Development
- [Redis Modules Blocking Operations](https://redis.io/docs/latest/develop/reference/modules/modules-blocking-ops/)
- [Making Redis Concurrent With Modules](https://redis.io/blog/making-redis-concurrent-with-modules/)
- [Redis Modules API Reference](https://redis.io/docs/latest/develop/reference/modules/modules-api-ref/)

---

*End of Architecture Options Analysis*
*Generated: 2026-02-26*
*Depends on: PVVH, RISK-SCAN, DECISIONS, RESEARCH artefacts*
