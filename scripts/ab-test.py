#!/usr/bin/env python3
"""A/B test: pre-tokenized (Path A) vs runtime tokenization (Path B).

Use case: Question answering where answers/context are known ahead of time.
Pre-tokenize all prompts offline, then compare inference latency and
Redis responsiveness against runtime tokenization.

Tests SHORT (~50 tokens), MEDIUM (~250 tokens), LONG (~800 tokens),
and VERY LONG (~1500+ tokens) prompts to show where pre-tokenization
overhead becomes significant.

Also measures offline tokenization time separately to show the cost
that pre-tokenization amortizes.

Requirements:
    pip install llama-cpp-python redis

Usage:
    # Start Redis with redis-infer module loaded and model loaded
    # Use context_size >= 4096 for long prompts:
    #   redis-cli INFER.LOAD models/Qwen3-0.6B-Q8_0.gguf 1 4096
    python3 scripts/ab-test.py models/Qwen3-0.6B-Q8_0.gguf
"""

import struct
import sys
import time
import threading
import statistics
import redis
from llama_cpp import Llama


TEMPLATE_PRE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
TEMPLATE_POST = "<|im_end|>\n<|im_start|>assistant\n"

def wrap(question):
    return f"{TEMPLATE_PRE}{question}{TEMPLATE_POST}"

# --- SHORT prompts (~50 tokens) ---
SHORT_PROMPTS = [
    wrap("What is the capital of France?"),
    wrap("What is 7 times 8?"),
    wrap("Name the largest planet in our solar system."),
]

# --- MEDIUM prompts (~250 tokens) ---
MEDIUM_CONTEXT = """The Redis module system allows developers to extend Redis with new commands, data types, and capabilities by writing shared libraries that Redis loads at startup. Modules have access to the Redis API through a set of functions that allow reading and writing keys, blocking clients, creating background tasks, and more. The module API provides thread-safe contexts that allow background threads to interact with Redis data while the main thread continues serving other clients. This is critical for operations that take significant time, such as machine learning inference, because it prevents Redis from becoming unresponsive. The blocked client pattern works by having the command handler call block_client(), which tells Redis to keep the client connection open but free the main thread. A background thread then performs the heavy computation and calls reply() when done. The client receives the response as if the command had completed synchronously, but Redis remained responsive throughout."""

MEDIUM_PROMPTS = [
    wrap(f"Based on the following context, explain how Redis modules handle long-running operations:\n\n{MEDIUM_CONTEXT}\n\nAnswer concisely."),
    wrap(f"Based on the following context, what is the blocked client pattern?\n\n{MEDIUM_CONTEXT}\n\nAnswer in one sentence."),
    wrap(f"Based on the following context, why is thread safety important for Redis modules?\n\n{MEDIUM_CONTEXT}\n\nAnswer briefly."),
]

# --- LONG prompts (~800 tokens) ---
LONG_CONTEXT = """The redis-infer module is a Redis module written in Rust that embeds llama.cpp for in-process AI inference. It implements a worker pool pattern where each worker thread owns a pre-created LlamaContext, which is not Send or Sync and must live on the thread that created it.

The module provides two commands for inference:

INFER.GENERATE reads pre-tokenized uint32 data from a Redis STRING key, where tokens were prepared offline using llama-cpp-python with the same GGUF model. This ensures token ID compatibility between the offline tokenizer and the runtime model. The binary format is little-endian packed uint32 arrays, where each 4-byte group represents one token ID.

INFER.GENERATE_TEXT reads raw UTF-8 text from a Redis STRING key and tokenizes it at runtime on the worker thread. This happens after the Redis GIL is released, so Redis remains responsive during tokenization. The worker calls model.str_to_token() which uses llama.cpp's built-in tokenizer.

Both commands use the copy-under-GIL pattern:
1. The command handler calls ctx.block_client() to free the Redis main thread
2. The request is submitted to the worker pool via a bounded channel
3. A worker thread acquires the Redis GIL via thread_ctx.lock()
4. Under the GIL, it opens the Redis key, reads the data, and copies it to an owned buffer
5. The GIL is released by dropping the lock guard
6. Inference runs on the owned data, completely GIL-free
7. The result is sent back via thread_ctx.reply()

The worker pool uses mpsc::sync_channel with Arc<Mutex<Receiver>> for work stealing across threads. Each worker creates its own LlamaContext at startup and reuses it across requests by calling clear_kv_cache() between inferences. The pool is bounded: if all workers are busy, new requests get an immediate error rather than queuing unboundedly.

The model itself (LlamaModel) is wrapped in Arc and shared across all workers because it is Send+Sync safe. The LlamaBackend is initialized once globally using OnceLock. Model loading happens on the Redis main thread via INFER.LOAD and blocks Redis during the load (typically 2-10 seconds), which is acceptable since it is an admin operation run once.

The generation loop works as follows: input tokens are added to a LlamaBatch, then ctx.decode() performs the prefill pass. A LlamaSampler is configured based on temperature: greedy sampling for temperature=0, or a chain of top_k, top_p, temperature scaling, and distribution sampling otherwise. The decode-sample loop runs until max_tokens is reached or an end-of-generation token is produced. Output tokens are converted to text pieces using model.token_to_piece() with a UTF-8 decoder that handles multi-byte sequences spanning token boundaries.

Error handling follows the pattern of converting llama-cpp-2 errors into Redis error responses. If a worker thread panics or fails to create a context at startup, it logs the error and exits, reducing the effective pool size. The module supports hot-swapping models: INFER.LOAD drains the existing pool (dropping the sender closes the channel, workers see Err on recv and exit), drops the old model, then loads the new one and creates a fresh pool.

For GPU acceleration, the module supports Metal on macOS via the system-ggml feature flag, which links against the system Homebrew installation of llama.cpp rather than the bundled version. This resolves Metal pipeline initialization bugs in certain quantization formats. The gpu_layers parameter controls how many transformer layers are offloaded to the GPU, with the default being all layers."""

LONG_PROMPTS = [
    wrap(f"Based on the following technical documentation, explain the copy-under-GIL pattern used in redis-infer:\n\n{LONG_CONTEXT}\n\nProvide a step-by-step explanation."),
    wrap(f"Based on the following technical documentation, what is the difference between INFER.GENERATE and INFER.GENERATE_TEXT?\n\n{LONG_CONTEXT}\n\nAnswer concisely."),
    wrap(f"Based on the following technical documentation, how does the worker pool handle concurrent requests?\n\n{LONG_CONTEXT}\n\nAnswer in detail."),
]

# --- VERY LONG prompts (~1500+ tokens) ---
# Simulates full document-level Q&A context
VLONG_CONTEXT_PART2 = """

ARCHITECTURE AND DESIGN DECISIONS

The choice of Rust for the Redis module provides several advantages over the original C implementation. Memory safety guarantees from the borrow checker prevent use-after-free and double-free bugs that are common in C Redis modules. The type system enforces correct usage of thread-unsafe types like LlamaContext, which cannot accidentally be shared across threads. Error handling with Result types replaces error-prone C patterns of checking return codes.

The llama-cpp-2 crate provides safe Rust bindings to llama.cpp. The LlamaModel type is Send+Sync because the underlying C structure uses only immutable shared state after initialization. The LlamaContext type is deliberately marked as not Send and not Sync because it contains mutable state that is not thread-safe, including the KV cache, sampling state, and batch buffers.

The pre-tokenization architecture was chosen based on several observations. First, in a question-answering system, the context documents are known ahead of time and can be tokenized offline during indexing. This moves the tokenization cost from the hot path (query time) to the cold path (indexing time). Second, pre-tokenized data can be stored efficiently as packed binary arrays, enabling direct memory mapping in future optimizations. Third, token IDs from the offline tokenizer must exactly match the runtime model, which is guaranteed by using the same GGUF model file for both tokenization and inference.

The worker pool design was influenced by the constraint that LlamaContext is not Send. This means each context must be created on and used exclusively by its owning thread. The pool pre-creates contexts at startup rather than lazily, ensuring predictable memory usage and failing fast if the GPU cannot support the requested number of concurrent contexts. The bounded channel prevents unbounded memory growth under load, preferring immediate errors over growing latency.

BENCHMARKS AND PERFORMANCE CHARACTERISTICS

Inference performance depends primarily on three factors: prefill speed (processing input tokens), decode speed (generating output tokens), and tokenization overhead. Prefill is parallelized across the input sequence and benefits from GPU acceleration. Decode is inherently sequential as each token depends on the previous one. Tokenization is typically a small fraction of total latency for short inputs but can become significant for very long documents.

On Apple Silicon with Metal acceleration, the Qwen3-0.6B model achieves approximately 800-900 tokens per second for prefill and 150-180 tokens per second for decode. This means a 1000-token input takes roughly 1.1-1.25 seconds for prefill plus the decode time. For 30 output tokens, decode adds approximately 170-200 milliseconds.

The tokenization overhead for llama.cpp's built-in tokenizer is approximately 1-2 microseconds per token for BPE-based models. For a 1000-token input, this translates to 1-2 milliseconds of tokenization time, which is negligible compared to the 1+ second prefill time. However, for batch processing scenarios where thousands of documents are tokenized per second, the cumulative tokenization cost becomes meaningful. Pre-tokenizing offline eliminates this per-request cost entirely.

Redis responsiveness during inference is maintained by the copy-under-GIL pattern. The GIL is held only during the key read operation, which takes microseconds. The actual inference runs entirely outside the GIL. Measurements show Redis PING latency remains under 2 milliseconds even during active inference, compared to the hundreds of milliseconds or seconds that inference takes.

SECURITY AND OPERATIONAL CONSIDERATIONS

The module loads model files from the local filesystem using paths provided via the INFER.LOAD command. This command should be restricted to admin users via Redis ACLs because loading a model consumes significant memory (proportional to model size) and creates GPU contexts. Malicious model files could potentially exploit vulnerabilities in the GGUF parser, though llama.cpp includes validation of file headers and tensor dimensions.

The pre-tokenized data path trusts that the stored token IDs are valid. Invalid token IDs (outside the vocabulary range) will cause the model to produce garbage output but should not cause crashes because llama.cpp bounds-checks token indices internally. The runtime tokenization path validates UTF-8 encoding before passing text to the tokenizer.

Memory usage per worker context depends on the context size parameter. With the default 4096 context size and float16 KV cache, each context consumes approximately 128-256 MB of RAM or VRAM depending on the model architecture. The total memory footprint is: model weights (shared) + N workers x context memory. For a 4-bit quantized 0.6B model, this is roughly 400 MB base plus 150 MB per worker.

The module registers a deinit handler that drains the worker pool and releases the model on shutdown. This ensures GPU resources (Metal command buffers, VRAM allocations) are properly freed. Without this cleanup, residual GPU state can interfere with subsequent module loads or cause memory leaks in long-running Redis instances."""

VLONG_CONTEXT = LONG_CONTEXT + VLONG_CONTEXT_PART2

VLONG_PROMPTS = [
    wrap(f"Based on the following comprehensive documentation, explain why pre-tokenization was chosen as the architecture and what trade-offs it involves:\n\n{VLONG_CONTEXT}\n\nProvide a thorough analysis."),
    wrap(f"Based on the following comprehensive documentation, describe the memory usage characteristics and security considerations:\n\n{VLONG_CONTEXT}\n\nAnswer in detail."),
    wrap(f"Based on the following comprehensive documentation, explain the performance characteristics and how Redis remains responsive during inference:\n\n{VLONG_CONTEXT}\n\nInclude specific numbers."),
]

MAX_TOKENS = 30
TEMPERATURE = 0.0
RUNS_PER_PROMPT = 3


def ping_monitor(stop_event, latencies):
    """Continuously ping Redis to measure responsiveness."""
    r = redis.Redis()
    while not stop_event.is_set():
        start = time.time()
        r.ping()
        latencies.append((time.time() - start) * 1000)
        time.sleep(0.02)


def run_single(r, command, key, max_tokens, temperature):
    """Run a single inference request and return (elapsed_ms, result_text)."""
    start = time.time()
    result = r.execute_command(command, key, str(max_tokens), str(temperature))
    elapsed = (time.time() - start) * 1000
    if isinstance(result, bytes):
        result = result.decode("utf-8", errors="replace")
    return elapsed, result


def run_category(r, llm, category_name, prompts, base_key):
    """Run A/B test for a category of prompts."""
    print(f"\n--- {category_name} ---")
    token_counts = []
    tokenize_times = []

    for i, prompt in enumerate(prompts):
        # Measure offline tokenization time
        t0 = time.time()
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        tok_ms = (time.time() - t0) * 1000
        tokenize_times.append(tok_ms)
        token_counts.append(len(tokens))

        binary = struct.pack(f"<{len(tokens)}I", *[t & 0xFFFFFFFF for t in tokens])
        r.set(f"{base_key}:tokens:{i}", binary)
        r.set(f"{base_key}:text:{i}", prompt)
        print(f"  P{i}: {len(prompt):5d} chars -> {len(tokens):4d} tokens"
              f"  (offline tokenize: {tok_ms:.2f}ms,"
              f" binary: {len(binary)} bytes vs text: {len(prompt.encode())} bytes)")

    a_times = []
    b_times = []

    for i, prompt in enumerate(prompts):
        short_q = prompt.split("user\n")[1][:70].replace("\n", " ").strip()
        print(f"\n  P{i}: {short_q}...")

        for run in range(RUNS_PER_PROMPT):
            # Path A: pre-tokenized
            stop_a = threading.Event()
            pings_a = []
            ping_t = threading.Thread(target=ping_monitor, args=(stop_a, pings_a))
            ping_t.start()
            elapsed_a, result_a = run_single(
                r, "INFER.GENERATE", f"{base_key}:tokens:{i}", MAX_TOKENS, TEMPERATURE
            )
            stop_a.set()
            ping_t.join()
            a_times.append(elapsed_a)
            ping_a_max = max(pings_a) if pings_a else 0

            # Path B: runtime tokenization
            stop_b = threading.Event()
            pings_b = []
            ping_t = threading.Thread(target=ping_monitor, args=(stop_b, pings_b))
            ping_t.start()
            elapsed_b, result_b = run_single(
                r, "INFER.GENERATE_TEXT", f"{base_key}:text:{i}", MAX_TOKENS, TEMPERATURE
            )
            stop_b.set()
            ping_t.join()
            b_times.append(elapsed_b)
            ping_b_max = max(pings_b) if pings_b else 0

            if run == 0:
                snippet_a = result_a.strip().replace("\n", " ")[:55] if result_a else "(empty)"
                snippet_b = result_b.strip().replace("\n", " ")[:55] if result_b else "(empty)"
                print(f"    A: {elapsed_a:7.1f}ms  ping={ping_a_max:4.1f}ms  {snippet_a}")
                print(f"    B: {elapsed_b:7.1f}ms  ping={ping_b_max:4.1f}ms  {snippet_b}")
            else:
                print(f"    run {run+1}: A={elapsed_a:7.1f}ms  B={elapsed_b:7.1f}ms")

    # Cleanup
    for i in range(len(prompts)):
        r.delete(f"{base_key}:tokens:{i}", f"{base_key}:text:{i}")

    return a_times, b_times, token_counts, tokenize_times


def stats_line(times):
    avg = statistics.mean(times)
    med = statistics.median(times)
    mn = min(times)
    mx = max(times)
    std = statistics.stdev(times) if len(times) > 1 else 0
    return avg, med, mn, mx, std


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
    r = redis.Redis()

    info = r.execute_command("INFER.INFO").decode()
    print(f"Module: {info}\n")

    print(f"Loading tokenizer from {model_path} (vocab_only)...")
    llm = Llama(model_path=model_path, n_ctx=32, n_gpu_layers=0, vocab_only=True,
                verbose=False)

    categories = [
        ("SHORT (~50 tok)",     SHORT_PROMPTS,  "ab:short"),
        ("MEDIUM (~250 tok)",   MEDIUM_PROMPTS, "ab:medium"),
        ("LONG (~800 tok)",     LONG_PROMPTS,   "ab:long"),
        ("VERY LONG (~1500 tok)", VLONG_PROMPTS, "ab:vlong"),
    ]

    print(f"\n{'='*78}")
    print(f"A/B TEST: pre-tokenized (A) vs runtime tokenization (B)")
    print(f"  {RUNS_PER_PROMPT} runs per prompt, max_tokens={MAX_TOKENS}, temperature={TEMPERATURE}")
    print(f"{'='*78}")

    all_results = []

    for cat_name, prompts, base_key in categories:
        a_times, b_times, tok_counts, tok_times = run_category(
            r, llm, cat_name, prompts, base_key
        )
        avg_tokens = statistics.mean(tok_counts)
        avg_tok_time = statistics.mean(tok_times)
        all_results.append((cat_name, a_times, b_times, avg_tokens, avg_tok_time))

    # --- Summary table ---
    print(f"\n{'='*78}")
    print(f"SUMMARY")
    print(f"{'='*78}")
    print(f"{'Category':<22} {'Tokens':>6} {'A med':>8} {'B med':>8}"
          f" {'B-A':>8} {'%':>7} {'Tok.ms':>7}")
    print(f"{'-'*78}")

    for cat_name, a_times, b_times, avg_tokens, avg_tok_time in all_results:
        _, med_a, _, _, _ = stats_line(a_times)
        _, med_b, _, _, _ = stats_line(b_times)
        diff = med_b - med_a
        pct = (diff / med_a * 100) if med_a > 0 else 0
        print(f"  {cat_name:<20} {avg_tokens:>5.0f}  {med_a:7.1f}ms {med_b:7.1f}ms"
              f" {diff:+7.1f}ms {pct:+6.1f}% {avg_tok_time:6.2f}ms")

    print(f"\n  A = INFER.GENERATE (pre-tokenized binary, no tokenization at request time)")
    print(f"  B = INFER.GENERATE_TEXT (raw text, tokenized at runtime on worker thread)")
    print(f"  Tok.ms = average offline tokenization time per prompt (amortized by pre-tokenization)")
    print(f"  B-A > 0 means pre-tokenization is faster; B-A < 0 means runtime is faster")

    # Detailed stats
    print(f"\nDETAILED STATISTICS")
    print(f"{'-'*78}")
    for cat_name, a_times, b_times, avg_tokens, avg_tok_time in all_results:
        avg_a, med_a, min_a, max_a, std_a = stats_line(a_times)
        avg_b, med_b, min_b, max_b, std_b = stats_line(b_times)
        print(f"\n  {cat_name} (avg {avg_tokens:.0f} tokens, offline tokenize: {avg_tok_time:.2f}ms):")
        print(f"    Path A: avg={avg_a:7.1f}  med={med_a:7.1f}  min={min_a:7.1f}"
              f"  max={max_a:7.1f}  std={std_a:5.1f}")
        print(f"    Path B: avg={avg_b:7.1f}  med={med_b:7.1f}  min={min_b:7.1f}"
              f"  max={max_b:7.1f}  std={std_b:5.1f}")

    # Scaling analysis
    print(f"\nSCALING ANALYSIS")
    print(f"{'-'*78}")
    print(f"  At 100 requests/sec with runtime tokenization:")
    for cat_name, _, _, avg_tokens, avg_tok_time in all_results:
        cpu_ms = avg_tok_time * 100
        print(f"    {cat_name:<20}: {avg_tok_time:.2f}ms/req x 100 = {cpu_ms:.0f}ms CPU/sec"
              f" ({cpu_ms/10:.1f}% of one core)")
    print(f"\n  Pre-tokenization eliminates this per-request CPU cost entirely.")
    print(f"  The tokenization happens once during offline indexing.")

    print(f"\n{'='*78}")
    print("A/B TEST COMPLETE")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
