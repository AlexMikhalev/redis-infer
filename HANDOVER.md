# redis-infer Handover Document

**Date**: 2026-02-27
**Repo**: https://github.com/AlexMikhalev/redis-infer
**Branch**: main (clean, all pushed, 17 commits)

---

## What redis-infer Is

A Rust Redis module that embeds llama.cpp for in-process AI inference. The core design principle is **pre-tokenized inference**: tokens are stored as packed little-endian uint32 arrays in Redis STRING keys, then fed directly to llama.cpp without runtime tokenization.

809 lines of Rust across two crates. No mocks. Everything tested against real Redis and real models.

---

## What Works

### Redis Commands

| Command | Purpose |
|---------|---------|
| `INFER.INFO` | Show module version, loaded model path, vocab size, worker count |
| `INFER.LOAD <path> [workers] [ctx_size] [gpu_layers]` | Load a GGUF model, create worker pool |
| `INFER.GENERATE <token_key> [max_tokens] [temp]` | Inference from pre-tokenized uint32 binary |
| `INFER.GENERATE_TEXT <text_key> [max_tokens] [temp]` | Inference from raw UTF-8 text (runtime tokenize) |

### Architecture

- **Copy-under-GIL pattern**: Worker acquires Redis GIL, reads key data to owned buffer, releases GIL, runs inference GIL-free. Redis stays responsive (PING <2ms during inference).
- **Worker pool**: Each worker owns a `LlamaContext` (not Send/Sync). `mpsc::sync_channel` with `Arc<Mutex<Receiver>>` for work stealing. Bounded -- returns error immediately if all workers busy.
- **Model sharing**: `LlamaModel` wrapped in `Arc` (Send+Sync safe), shared across all workers.
- **Hot-swap**: `INFER.LOAD` drains existing pool before loading new model. No restart needed.
- **GPU**: Metal via `system-ggml` feature flag (links system Homebrew llama.cpp build 8140).

### Build

```bash
# CPU-only
cargo build

# Metal GPU (requires: brew install llama.cpp)
cargo build --features system-ggml

# Run
redis-server --loadmodule target/debug/libredis_infer.dylib
redis-cli INFER.LOAD models/Qwen3-0.6B-Q8_0.gguf 1 4096
```

### Models Available

- `models/Qwen3-0.6B-Q8_0.gguf` (639 MB) -- primary dev model, works with Metal
- `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` (669 MB) -- works with system-ggml Metal

### Python Scripts

| Script | Purpose |
|--------|---------|
| `tokenize-and-store.py` | Offline tokenization of code repos to Redis keys |
| `prove-e2e.py` | End-to-end proof: tokenize -> store -> infer |
| `prove-pretokenized.py` | Switcheroo proof that model reads stored tokens |
| `ab-test.py` | A/B test: pre-tokenized vs runtime tokenization |
| `bench-mlx-vs-llama.py` | MLX vs llama.cpp Metal speed comparison |
| `test-inference.py` | Basic inference smoke test |
| `test-concurrent.py` | Concurrent inference test |
| `test-tinyllama.py` | TinyLlama-specific test |

---

## Key Findings

### Pre-tokenization A/B Test Results (Release Build)

| Prompt Size | Tokens | Pre-tokenized (A) | Runtime (B) | Diff | Std Dev A | Std Dev B |
|------------|--------|-------------------|-------------|------|-----------|-----------|
| SHORT | 27 | 232.9ms | 232.2ms | noise | 21.5 | 15.9 |
| MEDIUM | 209 | 277.1ms | 282.5ms | +5.5ms | 7.2 | 5.2 |
| LONG | 795 | 466.8ms | 464.6ms | noise | 8.6 | 8.9 |
| VERY LONG | 1826 | 839.2ms | 838.3ms | noise | 9.2 | 10.8 |

Release build shows dramatically tighter variance (stdev 5-22ms) vs debug (15-43ms). Median times similar because llama.cpp C++ inference dominates wall-clock time regardless of Rust optimization level.

Pre-tokenization value is in **CPU savings at scale**: at 100 req/s with 1500-token prompts, runtime tokenization burns 71.4% of a CPU core (7.14ms/req). Pre-tokenization eliminates this per-request cost entirely -- tokenization happens once during offline indexing.

### Release vs Debug Build

| Metric | Debug | Release |
|--------|-------|---------|
| Binary size | 5.1 MB | 2.1 MB |
| Offline tokenize (1826 tok) | 10.74ms | 7.14ms |
| Variance (VERY LONG stdev) | 12-43ms | 9-11ms |
| Median inference (VERY LONG) | 844-861ms | 838-839ms |

### MLX vs llama.cpp Metal (Qwen3-0.6B)

| Prompt | llama.cpp | MLX | Speedup |
|--------|-----------|-----|---------|
| short | 1347ms | 998ms | 1.35x MLX |
| medium | 1091ms | 1018ms | 1.07x MLX |
| long | 1055ms | 1044ms | 1.01x MLX |

MLX is modestly faster. mlx-rs Rust bindings are not ready for embedding (v0.0.1, generate module commented out).

### Critical Bug Found and Fixed

Python `llm.tokenize()` defaults to `special=False`, causing chat template tokens (`<|im_start|>`, `<|im_end|>`) to be tokenized as regular text (47 tokens instead of 26). Fix: always use `special=True` for chat-templated text.

---

## Crate Structure

```
redis-infer/
  Cargo.toml                    # workspace root
  crates/
    redis-infer/                # cdylib Redis module (206 + 268 = 474 lines)
      src/
        lib.rs                  # redis_module! macro, init/deinit
        state.rs                # OnceLock<RwLock<Option<Arc<...>>>> globals
        worker.rs               # WorkerPool, copy-under-GIL, TokenSource enum
        commands/
          load.rs               # INFER.LOAD
          generate.rs           # INFER.GENERATE (pre-tokenized)
          generate_text.rs      # INFER.GENERATE_TEXT (runtime tokenize)
          info.rs               # INFER.INFO
    infer-engine/               # inference abstraction (275 lines)
      src/
        lib.rs
        error.rs                # InferError enum (thiserror)
        tokens.rs               # bytes_to_tokens (packed LE uint32 -> LlamaToken)
        llama/
          model.rs              # LlamaBackend singleton, InferModel::load()
          generate.rs           # generation loop (batch, decode, sample)
```

### Key Dependencies

- `redis-module = "2.0.7"` (redismodule-rs)
- `llama-cpp-2 = "0.1.136"` (llama.cpp Rust bindings)
- `llama-cpp-sys-2 = "0.1.136"` (for system-ggml feature)
- `encoding_rs`, `thiserror`, `num_cpus`, `bytemuck`

---

## Open Issues

| # | Issue | Status | Notes |
|---|-------|--------|-------|
| 1 | INFER.RAG command | Open | Deferred by user. Would require Redis Vector Sets module. |
| 2 | TinyLlama Metal crash | Closed | Fixed via system-ggml (56f7a5f) |
| 3 | MLX backend | Closed | Assessed, not viable to embed. Benchmark script added (1c148ac) |
| 4 | A/B test | Closed | Complete with 4 prompt sizes (36991f2) |

---

## Possible Next Steps

1. **Production hardening**: Release build done (`cargo build --release --features system-ggml`, 2.1MB), proper error recovery if worker panics, configurable log levels
2. **INFER.RAG** (issue #1): Requires Redis Vector Sets module. Pattern: VSIM search -> read token keys -> concatenate -> generate
3. **Larger models**: Test with Qwen3-4B-Q4_K_M or Llama-3.2-3B for real workloads
4. **Batch inference**: Support multiple prompts in one command for throughput
5. **Streaming**: Return tokens incrementally instead of waiting for full generation
6. **CI/CD**: GitHub Actions with Redis + module loading for automated testing

---

## How to Resume

```bash
cd /Users/alex/projects/terraphim/redis-infer

# Build (debug)
cargo build --features system-ggml

# Build (release -- recommended for production)
cargo build --release --features system-ggml

# Start Redis (debug)
redis-server --loadmodule target/debug/libredis_infer.dylib

# Start Redis (release)
redis-server --loadmodule target/release/libredis_infer.dylib

# Load model
redis-cli INFER.LOAD models/Qwen3-0.6B-Q8_0.gguf 1 4096

# Test
redis-cli INFER.GENERATE_TEXT mykey 30 0.0  # (after storing text in mykey)

# Run proofs
python3 scripts/prove-pretokenized.py models/Qwen3-0.6B-Q8_0.gguf
python3 scripts/ab-test.py models/Qwen3-0.6B-Q8_0.gguf
```
