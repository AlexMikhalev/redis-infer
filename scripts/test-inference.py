#!/usr/bin/env python3
"""Quick test: store pre-tokenized data and run INFER.GENERATE.

This script uses the Qwen3 chat template tokens to create a simple
prompt that the model can complete. Uses known token IDs from Qwen3
vocabulary.

Requirements:
    pip install redis
"""

import struct
import sys
import redis

def main():
    r = redis.Redis(host="127.0.0.1", port=6379)

    # Verify module is loaded
    info = r.execute_command("INFER.INFO")
    print(f"Module info: {info.decode()}")

    if "model: none" in info.decode():
        print("ERROR: No model loaded. Run INFER.LOAD first.")
        sys.exit(1)

    # For Qwen3 model, we can use the BOS token (151643) and some common tokens
    # to form a simple prompt. These are known Qwen3 token IDs:
    # 151643 = <|im_start|>
    # 151644 = <|im_end|>
    # 8948   = "system"
    # 198    = "\n"
    # 2610   = "You"
    # 525    = " are"
    # 264    = " a"
    # 10950  = " helpful"
    # 17847  = " assistant"
    # 13     = "."

    # Build: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n
    # Using known Qwen3 token IDs
    tokens = [
        151643,  # <|im_start|>
        8948,    # system
        198,     # \n
        2610,    # You
        525,     # are
        264,     # a
        10950,   # helpful
        17847,   # assistant
        13,      # .
        151644,  # <|im_end|>
        198,     # \n
        151643,  # <|im_start|>
        872,     # user
        198,     # \n
        3838,    # What
        374,     # is
        220,     # " "
        17,      # 2
        10,      # +
        17,      # 2
        30,      # ?
        151644,  # <|im_end|>
        198,     # \n
        151643,  # <|im_start|>
        77091,   # assistant
        198,     # \n
    ]

    # Pack as little-endian uint32
    binary = struct.pack(f"<{len(tokens)}I", *tokens)

    key = "tokens:test:math_prompt"
    r.set(key, binary)
    print(f"Stored {len(tokens)} tokens in key '{key}' ({len(binary)} bytes)")

    # Run inference
    print("\nRunning INFER.GENERATE...")
    try:
        result = r.execute_command("INFER.GENERATE", key, "64", "0.7")
        if isinstance(result, bytes):
            result = result.decode("utf-8", errors="replace")
        print(f"Result: {result}")
    except redis.exceptions.ResponseError as e:
        print(f"ERROR: {e}")

    # Test PING during inference to verify non-blocking behavior
    print(f"\nPING test: {r.ping()}")


if __name__ == "__main__":
    main()
