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

### Pre-tokenization A/B Test Results

| Prompt Size | Tokens | Pre-tokenized | Runtime | Winner |
|------------|--------|---------------|---------|--------|
| SHORT | 27 | 248.7ms | 234.5ms | noise |
| MEDIUM | 209 | 282.5ms | 280.1ms | noise |
| LONG | 795 | 449.9ms | 454.9ms | Pre-tok +5ms |
| VERY LONG | 1826 | 822.6ms | 831.3ms | Pre-tok +9ms |

Pre-tokenization wins at 800+ tokens. At scale (100 req/s, 1500-token prompts), runtime tokenization burns 93% of a CPU core. Pre-tokenization eliminates that entirely.

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

1. **Production hardening**: Release build (`cargo build --release`), proper error recovery if worker panics, configurable log levels
2. **INFER.RAG** (issue #1): Requires Redis Vector Sets module. Pattern: VSIM search -> read token keys -> concatenate -> generate
3. **Larger models**: Test with Qwen3-4B-Q4_K_M or Llama-3.2-3B for real workloads
4. **Batch inference**: Support multiple prompts in one command for throughput
5. **Streaming**: Return tokens incrementally instead of waiting for full generation
6. **CI/CD**: GitHub Actions with Redis + module loading for automated testing

---

## How to Resume

```bash
cd /Users/alex/projects/terraphim/redis-infer

# Build
cargo build --features system-ggml

# Start Redis
redis-server --loadmodule target/debug/libredis_infer.dylib

# Load model
redis-cli INFER.LOAD models/Qwen3-0.6B-Q8_0.gguf 1 4096

# Test
redis-cli INFER.GENERATE_TEXT mykey 30 0.0  # (after storing text in mykey)

# Run proofs
python3 scripts/prove-pretokenized.py models/Qwen3-0.6B-Q8_0.gguf
python3 scripts/ab-test.py models/Qwen3-0.6B-Q8_0.gguf
```
