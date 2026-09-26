# Dev-set configuration selection

The rule was fixed before any run. I compare task outcome (answerable questions answered plus unanswerable questions refused) on the 40 dev questions for `wordpiece`, `bge`, `hybrid` and `baseline`, and pick the first config in that order whose count is within one of the best. Truncation-free configs come first because truncating most chunks is a defect, and one item in 40 is noise.

| config | answered | refused unanswerable | task outcome | retrieval all docs hit | raw citation precision | repairs | cost |
|---|---|---|---|---|---|---|---|
| wordpiece | 32/34 | 6/6 | 38/40 | 28/34 | 64/64 | 0 | $0.0204 |
| bge | 30/34 | 6/6 | 36/40 | 26/34 | 83/83 | 1 | $0.0307 |
| hybrid | 30/34 | 6/6 | 36/40 | 26/34 | 66/66 | 0 | $0.0214 |
| baseline | 30/34 | 6/6 | 36/40 | 28/34 | 119/119 | 0 | $0.0368 |

Selected config: `wordpiece`

Fixing truncation alone moved task outcome from 36 to 38 of 40 and lifted "an evidence page in the top k" from 32/34 to 34/34, while the all-documents hit rate stayed at 28/34. Neither bge nor hybrid beat it on any outcome count. Every config refused all six unanswerable questions.

Closed-book GPT-5.6 Luna (no retrieved context, default config) answered 28/34 answerable questions but refused only 1 of 6 unanswerable ones, for 29/40 overall. It is fluent without the documents, and it almost never declines, which is what the grounding is there to prevent.
