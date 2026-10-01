# Performance: response cache impact

15 factual questions, run once with an empty cache and once with a warm cache via `obslog.timed_answer()`. The cold pass calls retrieval and the model. The table records the cache hits in each pass.

| | p50 latency | p95 latency | mean cost/query | cache hits |
|---|---|---|---|---|
| cold | 4767 ms | 22603 ms | $0.00396 | 0/15 |
| warm | 2 ms | 3 ms | $0.00000 | 15/15 |

A cache hit skips retrieval and generation. `RAGResult.retrieved_chunks` is empty, so the API's retrieval debug view has no chunks to show.

The cache matches normalised question text (`src/cache.py`). A rephrased question is a miss. Pipeline settings also form part of the cache key.
