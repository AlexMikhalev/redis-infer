#!/usr/bin/env python3
"""A/B test: pre-tokenized (Path A) vs runtime tokenization (Path B).

Use case: Question answering where answers/context are known ahead of time.
Pre-tokenize all prompts offline, then compare inference latency and
Redis responsiveness against runtime tokenization.

Requirements:
    pip install llama-cpp-python redis

Usage:
    # Start Redis with redis-infer module loaded and model loaded
    python3 scripts/ab-test.py models/Qwen3-0.6B-Q8_0.gguf
"""

import struct
import sys
import time
import threading
import statistics
import redis
from llama_cpp import Llama


# Q&A prompts -- chat template for Qwen3
QA_PROMPTS = [
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 7 times 8?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nName the largest planet in our solar system.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat language is redis-infer written in?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat year did the Berlin Wall fall?<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is the boiling point of water in Celsius?<|im_end|>\n<|im_start|>assistant\n",
]

MAX_TOKENS = 30
TEMPERATURE = 0.0  # deterministic for fair comparison
RUNS_PER_PROMPT = 3  # repeat each prompt to reduce variance


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


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen3-0.6B-Q8_0.gguf"
    r = redis.Redis()

    info = r.execute_command("INFER.INFO").decode()
    print(f"Module: {info}\n")

    # --- Setup: tokenize all prompts offline ---
    print(f"Loading tokenizer from {model_path} (vocab_only)...")
    llm = Llama(model_path=model_path, n_ctx=32, n_gpu_layers=0, vocab_only=True,
                verbose=False)

    print(f"Preparing {len(QA_PROMPTS)} Q&A prompts...\n")
    for i, prompt in enumerate(QA_PROMPTS):
        # Path A: pre-tokenize and store as packed uint32
        tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True)
        binary = struct.pack(f"<{len(tokens)}I", *[t & 0xFFFFFFFF for t in tokens])
        key_a = f"ab:tokens:{i}"
        r.set(key_a, binary)

        # Path B: store raw text
        key_b = f"ab:text:{i}"
        r.set(key_b, prompt)

        print(f"  Q{i}: {len(prompt)} chars -> {len(tokens)} tokens")

    # --- Run A/B test ---
    print(f"\n{'='*70}")
    print(f"A/B TEST: {len(QA_PROMPTS)} prompts x {RUNS_PER_PROMPT} runs each")
    print(f"  Path A: INFER.GENERATE    (pre-tokenized uint32)")
    print(f"  Path B: INFER.GENERATE_TEXT (runtime tokenization)")
    print(f"  max_tokens={MAX_TOKENS}, temperature={TEMPERATURE}")
    print(f"{'='*70}\n")

    path_a_times = []
    path_b_times = []

    for i, prompt in enumerate(QA_PROMPTS):
        short_q = prompt.split("user\n")[1].split("<|im_end|>")[0].strip()
        print(f"Q{i}: {short_q}")

        for run in range(RUNS_PER_PROMPT):
            # Path A: pre-tokenized
            stop_a = threading.Event()
            pings_a = []
            ping_t = threading.Thread(target=ping_monitor, args=(stop_a, pings_a))
            ping_t.start()

            elapsed_a, result_a = run_single(
                r, "INFER.GENERATE", f"ab:tokens:{i}", MAX_TOKENS, TEMPERATURE
            )
            stop_a.set()
            ping_t.join()
            path_a_times.append(elapsed_a)
            ping_a_max = max(pings_a) if pings_a else 0

            # Path B: runtime tokenization
            stop_b = threading.Event()
            pings_b = []
            ping_t = threading.Thread(target=ping_monitor, args=(stop_b, pings_b))
            ping_t.start()

            elapsed_b, result_b = run_single(
                r, "INFER.GENERATE_TEXT", f"ab:text:{i}", MAX_TOKENS, TEMPERATURE
            )
            stop_b.set()
            ping_t.join()
            path_b_times.append(elapsed_b)
            ping_b_max = max(pings_b) if pings_b else 0

            if run == 0:
                # Show first result for each prompt
                answer_a = result_a.split("\n")[-1].strip()[:60] if result_a else "(empty)"
                answer_b = result_b.split("\n")[-1].strip()[:60] if result_b else "(empty)"
                print(f"  A: {elapsed_a:7.1f}ms  ping_max={ping_a_max:5.1f}ms  {answer_a}")
                print(f"  B: {elapsed_b:7.1f}ms  ping_max={ping_b_max:5.1f}ms  {answer_b}")
            else:
                print(f"  run {run+1}: A={elapsed_a:7.1f}ms  B={elapsed_b:7.1f}ms")

    # --- Results ---
    print(f"\n{'='*70}")
    print(f"RESULTS ({len(path_a_times)} measurements each)")
    print(f"{'='*70}")

    def stats(times, label):
        avg = statistics.mean(times)
        med = statistics.median(times)
        mn = min(times)
        mx = max(times)
        std = statistics.stdev(times) if len(times) > 1 else 0
        print(f"  {label}:")
        print(f"    avg={avg:7.1f}ms  median={med:7.1f}ms  min={mn:7.1f}ms  max={mx:7.1f}ms  stdev={std:5.1f}ms")
        return avg

    avg_a = stats(path_a_times, "Path A (pre-tokenized)")
    avg_b = stats(path_b_times, "Path B (runtime tokenize)")

    diff_ms = avg_b - avg_a
    diff_pct = (diff_ms / avg_a) * 100 if avg_a > 0 else 0

    print(f"\n  Tokenization overhead: {diff_ms:+.1f}ms ({diff_pct:+.1f}%)")
    if diff_ms > 0:
        print(f"  Pre-tokenization is {diff_ms:.1f}ms faster per request")
    else:
        print(f"  Runtime tokenization is {-diff_ms:.1f}ms faster per request")

    # Cleanup test keys
    for i in range(len(QA_PROMPTS)):
        r.delete(f"ab:tokens:{i}", f"ab:text:{i}")

    print(f"\n{'='*70}")
    print("A/B TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
