# Retrieval benchmark

Split `dev`, answerable questions only, k=5. A hit means a retrieved chunk covers an evidence page.

| config | chunks truncated at embedding | strategy | rerank | any hit | all docs hit | multi-doc all hit | MRR | p50 ms |
|---|---|---|---|---|---|---|---|---|
| baseline | 1606/1857 | semantic | no | 29/34 | 23/34 | 2/8 | 0.614 | 262 |
| baseline | 1606/1857 | semantic | yes | 32/34 | 28/34 | 5/8 | 0.841 | 2398 |
| baseline | 1606/1857 | named_balanced | no | 29/34 | 23/34 | 2/8 | 0.599 | 261 |
| baseline | 1606/1857 | named_balanced | yes | 32/34 | 28/34 | 5/8 | 0.826 | 2397 |
| baseline | 1606/1857 | bm25 | no | 32/34 | 27/34 | 2/8 | 0.675 | 13 |
| baseline | 1606/1857 | bm25 | yes | 31/34 | 27/34 | 4/8 | 0.797 | 2145 |
| baseline | 1606/1857 | hybrid | no | 33/34 | 26/34 | 2/8 | 0.681 | 287 |
| baseline | 1606/1857 | hybrid | yes | 32/34 | 26/34 | 3/8 | 0.826 | 2472 |
| wordpiece | 0/6152 | semantic | no | 30/34 | 24/34 | 2/8 | 0.541 | 654 |
| wordpiece | 0/6152 | semantic | yes | 34/34 | 28/34 | 3/8 | 0.698 | 1602 |
| wordpiece | 0/6152 | named_balanced | no | 30/34 | 24/34 | 2/8 | 0.536 | 636 |
| wordpiece | 0/6152 | named_balanced | yes | 34/34 | 28/34 | 3/8 | 0.683 | 1625 |
| wordpiece | 0/6152 | bm25 | no | 28/34 | 25/34 | 2/8 | 0.598 | 40 |
| wordpiece | 0/6152 | bm25 | yes | 33/34 | 26/34 | 2/8 | 0.702 | 1029 |
| wordpiece | 0/6152 | hybrid | no | 31/34 | 26/34 | 2/8 | 0.599 | 692 |
| wordpiece | 0/6152 | hybrid | yes | 33/34 | 26/34 | 2/8 | 0.696 | 1711 |
| bge | 0/3238 | semantic | no | 28/34 | 22/34 | 2/8 | 0.631 | 406 |
| bge | 0/3238 | semantic | yes | 29/34 | 26/34 | 4/8 | 0.68 | 2470 |
| bge | 0/3238 | named_balanced | no | 28/34 | 22/34 | 2/8 | 0.617 | 402 |
| bge | 0/3238 | named_balanced | yes | 29/34 | 26/34 | 4/8 | 0.665 | 2515 |
| bge | 0/3238 | bm25 | no | 29/34 | 24/34 | 1/8 | 0.575 | 22 |
| bge | 0/3238 | bm25 | yes | 29/34 | 27/34 | 4/8 | 0.695 | 2160 |
| bge | 0/3238 | hybrid | no | 26/34 | 24/34 | 2/8 | 0.597 | 434 |
| bge | 0/3238 | hybrid | yes | 29/34 | 26/34 | 4/8 | 0.68 | 2509 |

The hybrid config uses the wordpiece store: with named_balanced and rerank it hit all evidence documents on 28/34 questions, against 26/34 for bge.
