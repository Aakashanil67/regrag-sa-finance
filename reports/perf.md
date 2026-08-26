# Performance: response cache impact

15 factual golden-set questions, run once cold (cache empty, every call hits real retrieval + the live Anthropic API) and once warm (same questions, every call a cache hit) via `obslog.timed_answer()`.

| | p50 latency | p95 latency | mean cost/query | cache hits |
|---|---|---|---|---|
| cold | 4489 ms | 25591 ms | $0.00365 | 0/15 |
| warm | 1 ms | 2 ms | $0.00000 | 15/15 |

A cache hit skips retrieval and the LLM call entirely, so `RAGResult.retrieved_chunks` is empty on a hit — the API's "what was retrieved" debug view has nothing to show for a cached response, which is a real trade-off of exact-match caching, not a bug.

Cache is exact-match on normalised question text (see `src/cache.py`), not semantic — a rephrased question is a miss. Traded hit rate for the guarantee that a cached answer is only ever served for the literal question it was generated for, which matters for a tool whose whole premise is citation accuracy.
