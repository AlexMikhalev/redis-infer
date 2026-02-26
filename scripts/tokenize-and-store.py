#!/usr/bin/env python3
"""Offline tokenizer: walks a Rust repo and stores pre-tokenized binary in Redis."""

from transformers import AutoTokenizer
import struct
import redis
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python3 tokenize-and-store.py /path/to/rust/repo")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
r = redis.Redis(host="127.0.0.1", port=6379)


def store(key, text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    binary = struct.pack(f"{len(tokens)}I", *tokens)
    r.set(f"tokens:{key}", binary)
    print(f"Stored {len(tokens):,} tokens -> tokens:{key}")


for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        if f.endswith((".rs", ".toml", ".md")):
            path = os.path.join(root, f)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                store(f.replace("/", "_").replace(".", "_"), fh.read())
