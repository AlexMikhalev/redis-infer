#!/usr/bin/env python3
"""Test concurrent inference requests and verify Redis stays responsive."""

import struct
import time
import threading
import redis


def store_prompt(r, key_name, tokens):
    """Store pre-tokenized prompt in Redis."""
    binary = struct.pack(f"<{len(tokens)}I", *tokens)
    r.set(key_name, binary)


def run_inference(thread_id, key_name, results):
    """Run INFER.GENERATE and record timing."""
    r = redis.Redis(host="127.0.0.1", port=6379)
    start = time.time()
    try:
        result = r.execute_command("INFER.GENERATE", key_name, "32", "0.7")
        elapsed = time.time() - start
        if isinstance(result, bytes):
            result = result.decode("utf-8", errors="replace")
        results[thread_id] = {"status": "ok", "elapsed": elapsed, "text": result[:80]}
    except Exception as e:
        elapsed = time.time() - start
        results[thread_id] = {"status": "error", "elapsed": elapsed, "error": str(e)}


def ping_during_inference(stop_event, latencies):
    """Continuously ping Redis to measure responsiveness."""
    r = redis.Redis(host="127.0.0.1", port=6379)
    while not stop_event.is_set():
        start = time.time()
        r.ping()
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        time.sleep(0.05)


def main():
    r = redis.Redis(host="127.0.0.1", port=6379)

    # Qwen3 chat template: "What is {N}+{N}?"
    base_tokens = [
        151643, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151644, 198,
        151643, 872, 198,
    ]
    end_tokens = [151644, 198, 151643, 77091, 198]

    # Create 4 different prompts
    prompts = {
        "tokens:test:concurrent_0": base_tokens + [3838, 374, 220, 18, 10, 18, 30] + end_tokens,  # 3+3
        "tokens:test:concurrent_1": base_tokens + [3838, 374, 220, 19, 10, 20, 30] + end_tokens,  # 4+5
        "tokens:test:concurrent_2": base_tokens + [3838, 374, 220, 22, 10, 23, 30] + end_tokens,  # 7+8
        "tokens:test:concurrent_3": base_tokens + [3838, 374, 220, 24, 10, 24, 30] + end_tokens,  # 9+9
    }

    for key, tokens in prompts.items():
        store_prompt(r, key, tokens)
    print(f"Stored {len(prompts)} test prompts")

    # Start ping monitor
    stop_event = threading.Event()
    latencies = []
    ping_thread = threading.Thread(target=ping_during_inference, args=(stop_event, latencies))
    ping_thread.start()

    # Launch 4 concurrent inference requests
    results = {}
    threads = []
    start_time = time.time()
    for i, key in enumerate(prompts):
        t = threading.Thread(target=run_inference, args=(i, key, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    total_time = time.time() - start_time

    stop_event.set()
    ping_thread.join()

    # Report results
    print(f"\n--- Concurrent Inference Results ({total_time:.1f}s total) ---")
    for tid in sorted(results):
        r_data = results[tid]
        if r_data["status"] == "ok":
            print(f"  Thread {tid}: {r_data['elapsed']:.1f}s - {r_data['text']}...")
        else:
            print(f"  Thread {tid}: ERROR after {r_data['elapsed']:.1f}s - {r_data['error']}")

    # PING latency stats
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        max_lat = max(latencies)
        print(f"\n--- PING Latency During Inference ---")
        print(f"  Samples: {len(latencies)}")
        print(f"  Avg: {avg_lat:.1f}ms")
        print(f"  Max: {max_lat:.1f}ms")
        print(f"  Target: <10ms")
        if max_lat < 10:
            print("  PASS: Redis stayed responsive during inference")
        else:
            print(f"  WARN: Max latency {max_lat:.1f}ms exceeded 10ms target")

    # Check all 4 succeeded
    ok_count = sum(1 for r_data in results.values() if r_data["status"] == "ok")
    print(f"\n{ok_count}/4 concurrent requests completed successfully")


if __name__ == "__main__":
    main()
