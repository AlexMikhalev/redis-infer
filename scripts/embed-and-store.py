#!/usr/bin/env python3
"""Offline embedder: walks a Rust repo and stores embeddings via Vector Sets."""

from sentence_transformers import SentenceTransformer
import redis
import json
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python3 embed-and-store.py /path/to/rust/repo")
    sys.exit(1)

embedder = SentenceTransformer("antirez/gte-small-pure-c-converter")
r = redis.Redis(host="127.0.0.1", port=6379)

for root, _, files in os.walk(sys.argv[1]):
    for f in files:
        if f.endswith((".rs", ".toml", ".md")):
            key = f.replace("/", "_").replace(".", "_")
            path = os.path.join(root, f)
            with open(path, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
                vec = embedder.encode(text, normalize_embeddings=True).tolist()
                r.execute_command(
                    "VADD", "codebase:embeddings", key,
                    "VECTOR", len(vec), *vec,
                    "JSON", json.dumps({
                        "tokens_key": f"tokens:{key}",
                        "lang": "rust"
                    }),
                    "QUANT", "int8"
                )
                print(f"Embedded {key}")
