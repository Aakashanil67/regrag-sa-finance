# Improvement log: chunk size x reranking sweep

Each row re-chunks and re-embeds the whole corpus at that chunk size into a throwaway collection (never the production one), then runs the same 20-question retrieval benchmark used in `reports/retrieval_bench.md`.

| chunk size | rerank | chunks | hit-rate@3 | hit-rate@5 | hit-rate@10 | MRR |
|---|---|---|---|---|---|---|
| 300 | False | 1861 | 65% | 85% | 85% | 0.578 |
| 300 | True | 1861 | 85% | 85% | 90% | 0.717 |
| 500 | False | 1081 | 80% | 85% | 85% | 0.654 |
| 500 | True | 1081 | 80% | 90% | 90% | 0.747 |
| 800 | False | 635 | 85% | 90% | 90% | 0.752 |
| 800 | True | 635 | 95% | 95% | 95% | 0.808 |

**Best on hit-rate@5 (tiebreak MRR): chunk size 800, rerank=True** — hit-rate@5=95%, MRR=0.808.
