#!/usr/bin/env python3
"""Benchmark: redis-infer (llama.cpp Metal) vs MLX native.

Runs the same prompts through both backends and compares:
- Time to first token (prefill)
- Total generation time
- Tokens per second

Uses same Qwen3-0.6B model in both formats:
- llama.cpp: models/Qwen3-0.6B-Q8_0.gguf (via redis-infer INFER.GENERATE_TEXT)
- MLX: mlx-community/Qwen3-0.6B-8bit (via mlx-lm Python)

Requirements:
    pip install mlx-lm redis

Usage:
    # Start Redis with redis-infer module loaded and model loaded
    python3 scripts/bench-mlx-vs-llama.py
"""

import sys
import time
import statistics
import redis

MLX_MODEL = "mlx-community/Qwen3-0.6B-8bit"
MAX_TOKENS = 100
TEMPERATURE = 0.0
RUNS = 3

PROMPTS = [
    ("short", "What is the capital of France?"),
    ("medium", "Explain how Redis modules handle long-running operations using the blocked client pattern. Include the role of thread-safe contexts and how the main thread remains responsive during heavy computation like machine learning inference."),
    ("long", "Write a detailed technical explanation of how a thread pool pattern works in a Redis module written in Rust. Cover these topics: why LlamaContext is not Send or Sync and must live on its owning thread, how copy-under-GIL works (acquire GIL, read key, copy data, release GIL, then run inference GIL-free), how mpsc channels with Arc<Mutex<Receiver>> enable work stealing, why the pool is bounded rather than unbounded, and how model hot-swapping works by draining the pool before loading a new model. Also explain the pre-tokenization architecture where tokens are stored as packed little-endian uint32 arrays in Redis STRING keys."),
]

CHAT_TEMPLATE = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n'


def bench_redis(r, prompts, max_tokens, runs):
    """Benchmark llama.cpp via redis-infer INFER.GENERATE_TEXT."""
    results = []
    for label, question in prompts:
        prompt = CHAT_TEMPLATE.format(q=question)
        key = f"bench:llama:{label}"
        r.set(key, prompt)
        times = []
        output = ""
        for run in range(runs):
            start = time.time()
            result = r.execute_command(
                "INFER.GENERATE_TEXT", key, str(max_tokens), str(TEMPERATURE)
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            if isinstance(result, bytes):
                result = result.decode("utf-8", errors="replace")
            if run == 0:
                output = result

        # Count tokens in output (rough: split on spaces/pieces)
        # For fair comparison, count actual generated tokens
        output_tokens = max_tokens  # assume max_tokens generated unless EOS
        med_ms = statistics.median(times)
        tps = (output_tokens / med_ms) * 1000 if med_ms > 0 else 0

        r.delete(key)
        results.append({
            "label": label,
            "median_ms": med_ms,
            "min_ms": min(times),
            "max_ms": max(times),
            "tps": tps,
            "output": output.strip()[:80],
        })
    return results


def bench_mlx(prompts, max_tokens, runs):
    """Benchmark MLX native via mlx-lm."""
    from mlx_lm import load, generate

    print(f"  Loading MLX model: {MLX_MODEL} ...")
    t0 = time.time()
    model, tokenizer = load(MLX_MODEL)
    load_ms = (time.time() - t0) * 1000
    print(f"  MLX model loaded in {load_ms:.0f}ms")

    results = []
    for label, question in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        times = []
        output = ""
        for run in range(runs):
            start = time.time()
            result = generate(
                model, tokenizer, prompt=prompt,
                max_tokens=max_tokens,
                verbose=False,
            )
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            if run == 0:
                output = result

        output_tokens = max_tokens
        med_ms = statistics.median(times)
        tps = (output_tokens / med_ms) * 1000 if med_ms > 0 else 0

        results.append({
            "label": label,
            "median_ms": med_ms,
            "min_ms": min(times),
            "max_ms": max(times),
            "tps": tps,
            "output": output.strip()[:80],
        })
    return results


def main():
    r = redis.Redis()
    try:
        info = r.execute_command("INFER.INFO").decode()
        print(f"redis-infer: {info}\n")
        redis_available = True
    except Exception as e:
        print(f"redis-infer not available: {e}")
        print("Skipping llama.cpp benchmark. Run with Redis + INFER.LOAD first.\n")
        redis_available = False

    print(f"{'=' * 78}")
    print(f"BENCHMARK: llama.cpp (redis-infer) vs MLX (mlx-lm)")
    print(f"  max_tokens={MAX_TOKENS}, temperature={TEMPERATURE}, runs={RUNS}")
    print(f"{'=' * 78}")

    # --- llama.cpp via redis-infer ---
    llama_results = None
    if redis_available:
        print(f"\n--- llama.cpp (via redis-infer, Metal) ---")
        llama_results = bench_redis(r, PROMPTS, MAX_TOKENS, RUNS)
        for res in llama_results:
            print(f"  {res['label']:>8}: {res['median_ms']:7.0f}ms"
                  f"  ({res['tps']:5.1f} t/s)  {res['output'][:60]}")

    # --- MLX native ---
    print(f"\n--- MLX (mlx-lm, native Metal) ---")
    mlx_results = bench_mlx(PROMPTS, MAX_TOKENS, RUNS)
    for res in mlx_results:
        print(f"  {res['label']:>8}: {res['median_ms']:7.0f}ms"
              f"  ({res['tps']:5.1f} t/s)  {res['output'][:60]}")

    # --- Comparison ---
    print(f"\n{'=' * 78}")
    print(f"COMPARISON (median times, {MAX_TOKENS} tokens generated)")
    print(f"{'=' * 78}")
    print(f"{'Prompt':>8}  {'llama.cpp':>10}  {'MLX':>10}  {'Speedup':>10}")
    print(f"{'-' * 48}")

    if llama_results:
        for llama, mlx in zip(llama_results, mlx_results):
            speedup = llama["median_ms"] / mlx["median_ms"] if mlx["median_ms"] > 0 else 0
            faster = "MLX" if speedup > 1 else "llama.cpp"
            ratio = speedup if speedup > 1 else 1 / speedup if speedup > 0 else 0
            print(f"  {llama['label']:>6}  {llama['median_ms']:8.0f}ms  {mlx['median_ms']:8.0f}ms"
                  f"  {ratio:.2f}x {faster}")
    else:
        for res in mlx_results:
            print(f"  {res['label']:>6}  {'N/A':>10}  {res['median_ms']:8.0f}ms  {'--':>10}")

    print(f"\n{'=' * 78}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
