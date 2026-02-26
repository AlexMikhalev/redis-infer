#!/usr/bin/env python3
"""Offline tokenizer: walks a code repo, splits into paragraphs,
tokenizes with llama-cpp-python (same tokenizer as GGUF model),
and stores as packed little-endian uint32 binary in Redis.

Requirements:
    pip install llama-cpp-python redis

Usage:
    python3 tokenize-and-store.py /path/to/repo --model /path/to/model.gguf
"""

import argparse
import os
import struct
import sys

import redis
from llama_cpp import Llama


def split_paragraphs(text: str, min_chars: int = 200) -> list[str]:
    """Split text on double-newline boundaries.
    Merge very short paragraphs with the next one to avoid tiny chunks."""
    raw_paragraphs = text.split("\n\n")
    paragraphs = []
    buffer = ""
    for p in raw_paragraphs:
        p = p.strip()
        if not p:
            continue
        if buffer:
            buffer += "\n\n" + p
        else:
            buffer = p
        if len(buffer) >= min_chars:
            paragraphs.append(buffer)
            buffer = ""
    if buffer:
        if paragraphs:
            paragraphs[-1] += "\n\n" + buffer
        else:
            paragraphs.append(buffer)
    return paragraphs


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize code files and store in Redis as packed uint32"
    )
    parser.add_argument("repo_path", help="Path to code repository")
    parser.add_argument(
        "--model", required=True, help="Path to GGUF model file (for tokenizer)"
    )
    parser.add_argument("--redis-host", default="127.0.0.1")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument(
        "--extensions",
        default=".rs,.toml,.md,.py",
        help="Comma-separated file extensions to process",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum characters per paragraph chunk",
    )
    parser.add_argument(
        "--key-prefix", default="tokens", help="Redis key prefix"
    )
    args = parser.parse_args()

    # Use llama-cpp-python tokenizer -- produces same token IDs as the GGUF model
    print(f"Loading tokenizer from {args.model}...")
    llm = Llama(model_path=args.model, n_ctx=0, n_gpu_layers=0, vocab_only=True)
    r = redis.Redis(host=args.redis_host, port=args.redis_port)
    extensions = tuple(args.extensions.split(","))
    repo_path = os.path.abspath(args.repo_path)

    total_chunks = 0
    total_tokens = 0
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common non-code dirs
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "target" and d != "node_modules"]
        for f in files:
            if not f.endswith(extensions):
                continue
            filepath = os.path.join(root, f)
            rel_path = os.path.relpath(filepath, repo_path)
            # Normalize path separators for key naming
            key_path = rel_path.replace(os.sep, "/")

            with open(filepath, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()

            if not text.strip():
                continue

            paragraphs = split_paragraphs(text, min_chars=args.min_chars)
            for chunk_idx, paragraph in enumerate(paragraphs):
                tokens = llm.tokenize(paragraph.encode("utf-8"), add_bos=False)
                if not tokens:
                    continue
                # Pack as explicit little-endian uint32
                binary = struct.pack(
                    f"<{len(tokens)}I", *[t & 0xFFFFFFFF for t in tokens]
                )
                key = f"{args.key_prefix}:{key_path}:chunk_{chunk_idx}"
                r.set(key, binary)
                total_chunks += 1
                total_tokens += len(tokens)
                print(f"  {key} ({len(tokens)} tokens)")

    print(f"\nStored {total_chunks} chunks, {total_tokens:,} total tokens")


if __name__ == "__main__":
    main()
