#!/usr/bin/env python3
"""Offline embedder: walks a code repo, splits into paragraphs,
embeds each paragraph, and stores in Redis Vector Sets with
metadata linking to the pre-tokenized key.

Requirements:
    pip install sentence-transformers redis

Usage:
    python3 embed-and-store.py /path/to/repo
"""

import argparse
import json
import os
import sys

import redis
from sentence_transformers import SentenceTransformer


def split_paragraphs(text: str, min_chars: int = 200) -> list[str]:
    """Same splitting logic as tokenize-and-store.py -- must produce
    identical chunks so member names and token keys align."""
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
        description="Embed code files and store in Redis Vector Sets"
    )
    parser.add_argument("repo_path", help="Path to code repository")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-small-en-v1.5",
        help="Sentence transformer model name",
    )
    parser.add_argument(
        "--vset-key",
        default="codebase:embeddings",
        help="Redis Vector Set key name",
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
        "--token-prefix", default="tokens", help="Token key prefix (must match tokenize-and-store.py)"
    )
    args = parser.parse_args()

    print(f"Loading embedding model {args.embedding_model}...")
    embedder = SentenceTransformer(args.embedding_model)
    r = redis.Redis(host=args.redis_host, port=args.redis_port)
    extensions = tuple(args.extensions.split(","))
    repo_path = os.path.abspath(args.repo_path)

    total = 0
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "target" and d != "node_modules"]
        for f in files:
            if not f.endswith(extensions):
                continue
            filepath = os.path.join(root, f)
            rel_path = os.path.relpath(filepath, repo_path)
            key_path = rel_path.replace(os.sep, "/")

            with open(filepath, encoding="utf-8", errors="ignore") as fh:
                text = fh.read()

            if not text.strip():
                continue

            paragraphs = split_paragraphs(text, min_chars=args.min_chars)
            for chunk_idx, paragraph in enumerate(paragraphs):
                member_name = f"{key_path}:chunk_{chunk_idx}"
                tokens_key = f"{args.token_prefix}:{member_name}"

                vec = embedder.encode(paragraph, normalize_embeddings=True).tolist()
                metadata = json.dumps(
                    {
                        "tokens_key": tokens_key,
                        "source_file": key_path,
                        "chunk_index": chunk_idx,
                        "lang": os.path.splitext(f)[1].lstrip("."),
                    }
                )

                r.execute_command(
                    "VADD",
                    args.vset_key,
                    member_name,
                    "VECTOR",
                    len(vec),
                    *vec,
                    "JSON",
                    metadata,
                    "QUANT",
                    "int8",
                )
                total += 1
                print(f"  Embedded {member_name}")

    print(f"\nEmbedded {total} chunks into {args.vset_key}")


if __name__ == "__main__":
    main()
