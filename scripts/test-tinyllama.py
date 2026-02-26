#!/usr/bin/env python3
"""Quick test with TinyLlama-specific token IDs.

TinyLlama uses the Llama2 chat template with <s>, [INST], [/INST] tokens.
Token IDs verified from the TinyLlama tokenizer.
"""

import struct
import redis

def main():
    r = redis.Redis(host="127.0.0.1", port=6379)

    info = r.execute_command("INFER.INFO")
    print(f"Module info: {info.decode()}")

    # TinyLlama uses Llama2-style chat format:
    # <s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]
    # For simplicity, use a plain text prompt that works with any model
    # Just encode "Hello, world! The capital of France is" as ASCII bytes
    # and use character codes as approximate token IDs

    # Actually, for TinyLlama (Llama2 tokenizer), common token IDs:
    # 1 = <s> (BOS)
    # 2 = </s> (EOS)
    # Token IDs for common words (from Llama2 sentencepiece):
    # 15043 = Hello
    # 29892 = ,
    # 3186  = world
    # 29991 = !
    # 450   = The
    # 7483  = capital
    # 310   = of
    # 3444  = France
    # 338   = is
    tokens = [
        1,      # <s> BOS
        15043,  # Hello
        29892,  # ,
        3186,   # world
        29991,  # !
        450,    # The
        7483,   # capital
        310,    # of
        3444,   # France
        338,    # is
    ]

    binary = struct.pack(f"<{len(tokens)}I", *tokens)
    key = "tokens:test:tinyllama_prompt"
    r.set(key, binary)
    print(f"Stored {len(tokens)} tokens in key '{key}'")

    print("\nRunning INFER.GENERATE with TinyLlama...")
    try:
        result = r.execute_command("INFER.GENERATE", key, "32", "0.7")
        if isinstance(result, bytes):
            result = result.decode("utf-8", errors="replace")
        print(f"Result: {result}")
    except redis.exceptions.ResponseError as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
