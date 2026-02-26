# redis-infer -- Pure-C Inference Inside Redis (Feb 25 2026)

**Single downloadable artefact** -- everything you need in one zip.

**What you get**:
- Full non-truncated detailed plan (docs/REDIS-INFER-DETAILED-PLAN-v3.1.md)
- Ready-to-compile Redis module (src/redis-infer.c)
- Makefile
- Offline tokenization script (Qwen3-4B)
- Offline embedding + Vector Sets script (gte-pure-C)
- Zero-copy, multi-threaded, 256 GB standalone ready

**How to use**:
1. `cd redis-infer`
2. `make`
3. `redis-server --loadmodule ./src/redis-infer.so --maxmemory 220gb --io-threads 16`
4. `python3 scripts/tokenize-and-store.py /path/to/your-rust-project`
5. Start coding with `INFER.GENERATE` + RAG

All files are properly formatted Markdown and C. No truncation.
