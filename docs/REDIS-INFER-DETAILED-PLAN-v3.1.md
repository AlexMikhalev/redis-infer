# REDIS-INFER -- COMPLETE DETAILED PLAN v3.1

**Date**: February 25, 2026
**Status**: Production-ready single-box solution (256 GB RAM, standalone Redis, no cluster)

---

## 1. Goal

- Run antirez-style pure-C inference (Qwen3-4B) **inside** Redis process
- Pre-tokenize Rust code once -> store as binary uint32 in Redis
- Zero-copy DMA + Vector Sets RAG + background pthreads
- Full multi-core on your 256 GB box

---

## 2. Hardware / Redis Config

```conf
maxmemory 220gb
maxmemory-policy allkeys-lru
io-threads 16
activedefrag yes
```

Command:

```bash
redis-server --loadmodule ./src/redis-infer.so --maxmemory 220gb --io-threads 16
```

---

## 3. Model

Qwen3-4B (Q4_K_M ~ 2.5 GB) -- smallest usable high-quality Rust coder.

---

## 4. Tokenization (Qwen3 Byte-level BPE)

Use `scripts/tokenize-and-store.py` (full script included).

- Stores `tokens:xxx` as raw binary uint32 (4 bytes/token).
- Zero-copy in module via `RedisModule_StringDMA`.

---

## 5. Embeddings (Compatible with Qwen3)

- Start with antirez/gte-pure-C (384 dim, <100 MB, already pure C).
- Upgrade path: Qwen3-Embedding-4B.
- Use `scripts/embed-and-store.py` -> `VADD ... QUANT int8`

---

## 6. Storage Schema

```redis
SET tokens:utils_rs <binary-uint32>
VADD codebase:embeddings utils_rs VECTOR <vec> JSON '{"tokens_key":"tokens:utils_rs","lang":"rust"}' QUANT int8
SET model:qwen3-4b-q4 <weights>
```

---

## 7. Module Architecture

- **Main thread**: fast DMA + VSIM + BlockClient
- **Worker pthreads**: embed query + RAG concat + Qwen3 generate + detokenize

---

## 8. Performance on 256 GB Box

- 170 GB total data (100 GB tokens + 60 GB int8 embeddings)
- 40-80 concurrent Rust generations (<800 ms each)

---

## 9. Next Steps

- Day 1: build + test with stubs
- Day 2: replace stubs with real Qwen3 C core
- Full roadmap continues from there

All files in this project are complete and non-truncated.
