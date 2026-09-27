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


Test split, run once after the pipeline was frozen:

Split `test`, answerable questions only, k=5. A hit means a retrieved chunk covers an evidence page.

| config | chunks truncated at embedding | strategy | rerank | any hit | all docs hit | multi-doc all hit | MRR | p50 ms |
|---|---|---|---|---|---|---|---|---|
| baseline | 1606/1857 | semantic | no | 32/48 | 26/48 | 4/12 | 0.423 | 237 |
| baseline | 1606/1857 | semantic | yes | 37/48 | 29/48 | 3/12 | 0.585 | 2372 |
| baseline | 1606/1857 | named_balanced | no | 32/48 | 26/48 | 4/12 | 0.423 | 237 |
| baseline | 1606/1857 | named_balanced | yes | 37/48 | 29/48 | 3/12 | 0.585 | 2301 |
| baseline | 1606/1857 | bm25 | no | 41/48 | 37/48 | 6/12 | 0.686 | 13 |
| baseline | 1606/1857 | bm25 | yes | 41/48 | 36/48 | 6/12 | 0.641 | 2087 |
| baseline | 1606/1857 | hybrid | no | 37/48 | 33/48 | 6/12 | 0.615 | 253 |
| baseline | 1606/1857 | hybrid | yes | 39/48 | 34/48 | 6/12 | 0.618 | 2324 |
| wordpiece | 0/6152 | semantic | no | 32/48 | 24/48 | 1/12 | 0.396 | 639 |
| wordpiece | 0/6152 | semantic | yes | 40/48 | 32/48 | 4/12 | 0.677 | 1573 |
| wordpiece | 0/6152 | named_balanced | no | 32/48 | 24/48 | 1/12 | 0.396 | 636 |
| wordpiece | 0/6152 | named_balanced | yes | 40/48 | 32/48 | 4/12 | 0.677 | 1585 |
| wordpiece | 0/6152 | bm25 | no | 36/48 | 30/48 | 4/12 | 0.596 | 35 |
| wordpiece | 0/6152 | bm25 | yes | 42/48 | 33/48 | 2/12 | 0.753 | 994 |
| wordpiece | 0/6152 | hybrid | no | 38/48 | 33/48 | 6/12 | 0.507 | 705 |
| wordpiece | 0/6152 | hybrid | yes | 43/48 | 36/48 | 4/12 | 0.765 | 1620 |
| bge | 0/3238 | semantic | no | 39/48 | 30/48 | 2/12 | 0.496 | 392 |
| bge | 0/3238 | semantic | yes | 40/48 | 34/48 | 4/12 | 0.653 | 2437 |
| bge | 0/3238 | named_balanced | no | 39/48 | 30/48 | 2/12 | 0.496 | 405 |
| bge | 0/3238 | named_balanced | yes | 40/48 | 34/48 | 4/12 | 0.653 | 2442 |
| bge | 0/3238 | bm25 | no | 38/48 | 33/48 | 6/12 | 0.627 | 20 |
| bge | 0/3238 | bm25 | yes | 41/48 | 33/48 | 4/12 | 0.652 | 2054 |
| bge | 0/3238 | hybrid | no | 41/48 | 35/48 | 5/12 | 0.65 | 436 |
| bge | 0/3238 | hybrid | yes | 39/48 | 34/48 | 5/12 | 0.649 | 2458 |
