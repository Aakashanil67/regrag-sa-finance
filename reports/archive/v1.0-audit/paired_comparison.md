# Is the improvement real? A paired re-analysis

The historical before/after table compared the mean of one RAGAS run against the mean of another. Those two runs did not score the same questions: the pre-improvement run scored 34 items and the post-improvement run scored 39, overlapping on 33. Reranking pulled 6 previously-refused questions (`g02`, `g31`, `g32`, `g36`, `g39`, `g43`) into a real answer, and pushed 1 the other way (`g29`).

That matters, because a mean over a changing item set moves for two unrelated reasons: the answers got better, or the mix of questions being averaged changed. Only the first is a quality claim. This report separates them.

## Paired comparison, on the 33 items both runs scored

| metric | before (all) | after (all) | before (paired) | after (paired) | paired Δ | improved | regressed | tied | sign test |
|---|---|---|---|---|---|---|---|---|---|
| faithfulness | 0.791 | 0.829 | 0.800 | 0.823 | +0.023 | 9 | 9 | 15 | p = 1.00 |
| answer relevancy | 0.851 | 0.810 | 0.859 | 0.818 | -0.041 | 10 | 11 | 12 | p = 1.00 |
| context precision | 0.673 | 0.790 | 0.694 | 0.802 | +0.109 | 14 | 5 | 14 | p = 0.06 |
| context recall | 0.912 | 0.968 | 0.939 | 0.970 | +0.030 | 1 | 0 | 32 | p = 1.00 |

### What survives

Context precision gains +0.109 on the paired set, with 14 items improving and 5 regressing. The sign test p of 0.06 does not meet the conventional 0.05 threshold. The direction is consistent with reranking removing irrelevant chunks, but that mechanism does not establish a reliable effect. Thirty-three items cannot confirm it.

The headline faithfulness gain of +0.038 shrinks to +0.023 once the item set is held fixed, and the per-item split is 9 improved against 9 regressed. This does not show a faithfulness gain (p = 1.00). Most of the apparent gain was composition: the six questions that entered the average happened to score above the old mean.

Context recall rises +0.056 across the full runs but only +0.030 paired, with 32 of 33 items completely unchanged and exactly 1 item moving. At a paired baseline of 0.939 there was almost nothing left to win.

Answer relevancy drifts slightly negative (-0.041 paired, 10 improved against 11 regressed, p = 1.00). Also indistinguishable from noise, and reported rather than dropped.

## Retrieval benchmark: 20 questions

Hit-rate@5 went from 85% to 95%. In absolute terms that is 17/20 to 19/20, two questions. Wilson score intervals:

| | hit-rate@5 | 95% CI |
|---|---|---|
| before | 17/20 (85%) | 64% – 95% |
| after | 19/20 (95%) | 76% – 99% |

The intervals overlap. The 20-question benchmark does not establish a reliable hit-rate gain. MRR (0.654 → 0.808) records rank changes that hit-rate misses: moving a correct chunk from rank 4 to rank 1 changes MRR even when both ranks meet the cutoff. No power calculation or paired interval for MRR is reported.

## Scope and cost

The 800-token configuration produced five net additional answers on the observed runs. The paired analysis covers 33 items. Those counts describe this dataset; they do not establish how the change will perform on new questions. The 20-question retrieval benchmark is also too small to settle the difference. Each query adds a cross-encoder pass over 20 candidates. An 800-token chunk can span more pages than a 500-token chunk, so its page-range citation can be less precise.

These are archived results. Use the current sealed-test reports when assessing the shipped pipeline.
